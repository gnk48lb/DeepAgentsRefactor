"""会话存储：会话 / 事件日志 / 挂起态 / 活动会话指针 / 用户 (所在层: L1)

标准库 sqlite3 实现，单连接 + RLock。事件日志只追加（除 delete_session 外不做 UPDATE / DELETE），
“待处理”由会话游标 processed_seq 表达。模型上下文的续跑不在这里：它由 LangGraph checkpoint 负责。

多用户（schema v2）：每个会话归属一个 user_id，所有会话相关接口都带必填的关键字参数 user_id，
只在该用户自己的会话里查找 / 解析 / 搜索；别人的会话对它“不存在”。事件级接口（emit_event / get_events /
get_pending_events / mark_processed / save_turn_state ...）仍然只认 session_id：调用方必须已经通过带 user_id 的
接口拿到了会话，才能再去碰它的事件。
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from app import config
from app.utils.identity import OWNER_USER_ID, validate_user_id

# 这两类事件是“入站”：它们会驱动一次 wake，处理完后由 mark_processed 推进游标
INBOUND_EVENT_TYPES = ("user_message", "approval")
EVENT_FILE_UPLOADED = "file_uploaded"   # 用户上传文件的记录：不是入站事件（不触发 wake、不进 pending），只给回溯与"未消费文件"判断用
EVENT_SCHEDULED_PUSH = "scheduled_push"  # 系统按定时任务主动推给用户的内容：同样不是入站事件，给回溯与“接着聊刚才那条提醒”用

_SCHEMA_VERSION = 2
# v1 的单一指针键；v2 起每个用户一个键：f"{_ACTIVE_KEY}:{user_id}"
_ACTIVE_KEY = "active_session_id"
_USER_LABEL_MAX_CHARS = 40

# v1 的表结构。全新库与旧库走同一条路径：先建 v1 的表（IF NOT EXISTS），再由 _migrate_v2 补齐。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    processed_seq INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    search_text TEXT NOT NULL DEFAULT '',
    event_key TEXT,
    UNIQUE(session_id, seq),
    UNIQUE(session_id, event_key)
);
CREATE INDEX IF NOT EXISTS idx_events_session_type ON events(session_id, type, seq);
CREATE TABLE IF NOT EXISTS turn_state (
    session_id TEXT PRIMARY KEY,
    user_text TEXT,
    outbox TEXT NOT NULL DEFAULT '[]',
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '',
    blocked INTEGER NOT NULL DEFAULT 0,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL
)
"""


@dataclass(frozen=True)
class SessionInfo:
    id: str
    title: str
    created_at: float
    updated_at: float
    processed_seq: int
    user_id: str
    turns: int = 0      # user_message 事件数（仅 list_sessions / get_session 填充）
    events: int = 0     # 事件总数

    @property
    def short_id(self) -> str:
        return self.id[:8]


@dataclass(frozen=True)
class UserInfo:
    user_id: str
    label: str
    blocked: bool
    first_seen: float
    last_seen: float


@dataclass(frozen=True)
class SessionEvent:
    id: int
    session_id: str
    seq: int
    ts: float
    type: str
    payload: dict
    event_key: Optional[str] = None


@dataclass(frozen=True)
class TurnState:
    user_text: Optional[str]
    outbox: list


def _string_leaves(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _string_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _string_leaves(v)


def _like_escape(token: str) -> str:
    return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _auto_title(text: str) -> str:
    one_line = " ".join(str(text).split())
    return one_line[: config.SESSION_TITLE_MAX_CHARS]


def _active_key(user_id: str) -> str:
    """某个用户的活动会话指针在 meta 里的键。user_id 先过 validate_user_id 再拼进键名。"""
    return f"{_ACTIVE_KEY}:{validate_user_id(user_id)}"


class SessionStore:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.RLock()
        self._closed = False
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        # isolation_level=None：自动提交模式，多语句事务由 _tx() 显式 BEGIN IMMEDIATE / COMMIT
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    # ── 基础设施 ──────────────────────────────────────────────

    @property
    def path(self) -> str:
        return self._path

    def _init_schema(self) -> None:
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > _SCHEMA_VERSION:
                raise RuntimeError(
                    f"sessions.db 的 schema 版本 {version} 高于当前代码支持的 {_SCHEMA_VERSION}，请升级代码后再打开"
                )
            self._conn.executescript(_SCHEMA)
            if version < _SCHEMA_VERSION:
                self._migrate_v2()

    def _migrate_v2(self) -> None:
        """v1 -> v2：会话归属用户、users 表、活动指针按用户分键。

        整个迁移在一个 BEGIN IMMEDIATE 事务里，失败整体回滚；幂等（先用 PRAGMA table_info 判断列是否已存在），
        两个进程同时打开同一个旧库也安全。不删除任何旧数据：旧会话全部归 owner。
        """
        with self._tx() as c:
            cols = {r["name"] for r in c.execute("PRAGMA table_info(sessions)")}
            if "user_id" not in cols:
                # 整个项目里唯一允许出现 'owner' 默认值的地方：旧数据全部归 owner。
                # 注意 SQLite 无法再去掉列默认值，所以所有 INSERT INTO sessions 都必须显式写 user_id。
                c.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'owner'")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC)")
            c.execute(_USERS_TABLE)
            old = self._get_meta_locked(_ACTIVE_KEY)
            if old is not None:
                new_key = f"{_ACTIVE_KEY}:{OWNER_USER_ID}"
                if self._get_meta_locked(new_key) is None:
                    self._set_meta_locked(c, new_key, old)
                c.execute("DELETE FROM meta WHERE key = ?", (_ACTIVE_KEY,))
            c.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.close()
            except Exception:
                pass

    @contextmanager
    def _tx(self):
        """持锁的写事务：BEGIN IMMEDIATE ... COMMIT，异常 ROLLBACK。"""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # ── 会话 ──────────────────────────────────────────────────

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> SessionInfo:
        keys = row.keys()
        return SessionInfo(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            processed_seq=row["processed_seq"],
            user_id=row["user_id"],
            turns=row["turns"] if "turns" in keys else 0,
            events=row["events"] if "events" in keys else 0,
        )

    _SESSION_SELECT = (
        "SELECT s.id, s.title, s.created_at, s.updated_at, s.processed_seq, s.user_id,"
        " (SELECT COUNT(*) FROM events e WHERE e.session_id = s.id AND e.type = 'user_message') AS turns,"
        " (SELECT COUNT(*) FROM events e WHERE e.session_id = s.id) AS events"
        " FROM sessions s"
    )

    def create_session(self, title: str = "", activate: bool = True, *, user_id: str) -> SessionInfo:
        validate_user_id(user_id)
        title = " ".join(str(title or "").split())[: config.SESSION_TITLE_MAX_CHARS]
        sid = uuid.uuid4().hex
        now = time.time()
        with self._tx() as c:
            c.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at, processed_seq, user_id) VALUES (?,?,?,?,0,?)",
                (sid, title, now, now, user_id),
            )
            self._emit_locked(c, sid, "session_created", {"title": title}, None, now)
            if activate:
                self._set_meta_locked(c, _active_key(user_id), sid)
        return self.get_session(sid)  # type: ignore[return-value]

    def get_session(self, session_id: str, *, user_id: Optional[str] = None) -> Optional[SessionInfo]:
        """传了 user_id 而会话不属于该用户时返回 None。不传 user_id 只用于“先查归属再决定怎么做”的内部场景（如 wake）。"""
        sql = self._SESSION_SELECT + " WHERE s.id = ?"
        args: list = [session_id]
        if user_id is not None:
            sql += " AND s.user_id = ?"
            args.append(validate_user_id(user_id))
        with self._lock:
            row = self._conn.execute(sql, args).fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(self, limit: int = 10, *, user_id: str) -> list[SessionInfo]:
        validate_user_id(user_id)
        with self._lock:
            rows = self._conn.execute(
                self._SESSION_SELECT + " WHERE s.user_id = ? ORDER BY s.updated_at DESC, s.rowid DESC LIMIT ?",
                (user_id, int(limit)),
            ).fetchall()
        return [self._session_from_row(r) for r in rows]

    def list_sessions_any(self, limit: int = 50, *, user_id: Optional[str] = None) -> list[SessionInfo]:
        """仅供运维脚本：跨用户列出会话（user_id 给了就只列该用户的）。聊天链路不要用。"""
        sql = self._SESSION_SELECT
        args: list = []
        if user_id is not None:
            sql += " WHERE s.user_id = ?"
            args.append(validate_user_id(user_id))
        sql += " ORDER BY s.updated_at DESC, s.rowid DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._session_from_row(r) for r in rows]

    def count_sessions(self, *, user_id: Optional[str] = None) -> int:
        """user_id=None 表示全部用户，仅运维用。"""
        with self._lock:
            if user_id is None:
                return self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            return self._conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE user_id = ?", (validate_user_id(user_id),)
            ).fetchone()[0]

    def _resolve(self, ref: str, user_id: Optional[str]) -> Optional[SessionInfo]:
        """完整 id 或唯一前缀；user_id 不为 None 时，匹配与多义判断都只算该用户自己的会话。"""
        ref = (ref or "").strip().lower()
        if not ref:
            return None
        scope = ""
        scope_args: tuple = ()
        if user_id is not None:
            scope = " AND user_id = ?"
            scope_args = (validate_user_id(user_id),)
        with self._lock:
            exact = self._conn.execute(f"SELECT id FROM sessions WHERE id = ?{scope}", (ref, *scope_args)).fetchone()
            if exact:
                return self.get_session(exact["id"], user_id=user_id)
            rows = self._conn.execute(
                f"SELECT id FROM sessions WHERE substr(id, 1, ?) = ?{scope} ORDER BY updated_at DESC, rowid DESC",
                (len(ref), ref, *scope_args),
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(f"前缀 {ref} 匹配到 {len(rows)} 个会话: " + ", ".join(r["id"][:12] for r in rows))
        return self.get_session(rows[0]["id"], user_id=user_id)

    def resolve_session(self, ref: str, *, user_id: str) -> Optional[SessionInfo]:
        """完整 id 或唯一前缀，只在该用户自己的会话里找。没有匹配返回 None，多义抛 ValueError。"""
        validate_user_id(user_id)
        return self._resolve(ref, user_id)

    def resolve_session_any(self, ref: str) -> Optional[SessionInfo]:
        """仅供运维脚本：跨用户解析会话 id / 前缀。聊天链路不要用。"""
        return self._resolve(ref, None)

    def get_active_session_id(self, *, user_id: str) -> str:
        """该用户当前活动会话 id。指针缺失或指向已删除 / 不属于该用户的会话时，回落到该用户最近更新的会话；
        该用户一个会话都没有则为其新建默认会话。"""
        key = _active_key(user_id)
        with self._lock:
            value = self._get_meta_locked(key)
            if value and self._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND user_id = ?", (value, user_id)
            ).fetchone():
                return value
            row = self._conn.execute(
                "SELECT id FROM sessions WHERE user_id = ? ORDER BY updated_at DESC, rowid DESC LIMIT 1", (user_id,)
            ).fetchone()
            if row:
                with self._tx() as c:
                    self._set_meta_locked(c, key, row["id"])
                return row["id"]
            return self.create_session("", activate=True, user_id=user_id).id

    def set_active_session(self, session_id: str, *, user_id: str) -> None:
        key = _active_key(user_id)
        with self._tx() as c:
            if not c.execute("SELECT 1 FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)).fetchone():
                raise ValueError(f"会话不存在: {session_id}")     # 别人的会话与不存在的会话，报错完全一样
            self._set_meta_locked(c, key, session_id)

    def rename_session(self, session_id: str, title: str, *, user_id: str) -> None:
        validate_user_id(user_id)
        title = " ".join(str(title or "").split())[: config.SESSION_TITLE_MAX_CHARS]
        with self._tx() as c:
            cur = c.execute("UPDATE sessions SET title = ? WHERE id = ? AND user_id = ?", (title, session_id, user_id))
            if cur.rowcount == 0:
                raise ValueError(f"会话不存在: {session_id}")

    def delete_session(self, session_id: str) -> bool:
        """硬删：事件 + 挂起态 + 会话行。仅运维脚本使用（不限用户）。活动会话被删时清掉它所属用户的指针。"""
        with self._tx() as c:
            row = c.execute("SELECT user_id FROM sessions WHERE id = ?", (session_id,)).fetchone()
            c.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM turn_state WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            if row is not None:
                # user_id 来自库里已有的行而不是外部输入，这里不再 validate，以免脏数据让清理卡死
                key = f"{_ACTIVE_KEY}:{row['user_id']}"
                if self._get_meta_locked(key) == session_id:
                    c.execute("DELETE FROM meta WHERE key = ?", (key,))
        return row is not None

    # ── 用户 ──────────────────────────────────────────────────

    @staticmethod
    def _user_from_row(row: sqlite3.Row) -> UserInfo:
        return UserInfo(
            user_id=row["user_id"],
            label=row["label"],
            blocked=bool(row["blocked"]),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
        )

    def touch_user(self, user_id: str) -> tuple[UserInfo, bool]:
        """upsert 并刷新 last_seen。返回 (用户, 是否首次出现)；owner 的行同样创建。"""
        validate_user_id(user_id)
        now = time.time()
        with self._tx() as c:
            existed = c.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is not None
            if existed:
                c.execute("UPDATE users SET last_seen = ? WHERE user_id = ?", (now, user_id))
            else:
                c.execute(
                    "INSERT INTO users(user_id, label, blocked, first_seen, last_seen) VALUES (?, '', 0, ?, ?)",
                    (user_id, now, now),
                )
            row = c.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return self._user_from_row(row), not existed

    def get_user(self, user_id: str) -> Optional[UserInfo]:
        validate_user_id(user_id)
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return self._user_from_row(row) if row else None

    def list_users(self) -> list[UserInfo]:
        """按首次出现的先后排序（序号稳定：/users 与 /label /block 用的是同一个顺序）。"""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM users ORDER BY first_seen ASC, rowid ASC").fetchall()
        return [self._user_from_row(r) for r in rows]

    def resolve_user(self, ref: str) -> Optional[UserInfo]:
        """按 user_id 的完整值或唯一前缀（不区分大小写）解析。没有匹配返回 None，多义抛 ValueError。"""
        ref = (ref or "").strip().lower()
        if not ref:
            return None
        with self._lock:
            exact = self._conn.execute("SELECT * FROM users WHERE lower(user_id) = ?", (ref,)).fetchall()
            if len(exact) == 1:
                return self._user_from_row(exact[0])
            rows = self._conn.execute(
                "SELECT * FROM users WHERE lower(substr(user_id, 1, ?)) = ? ORDER BY first_seen ASC, rowid ASC",
                (len(ref), ref),
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(f"前缀 {ref} 匹配到 {len(rows)} 个用户: " + ", ".join(r["user_id"][:8] for r in rows))
        return self._user_from_row(rows[0])

    def set_user_label(self, user_id: str, label: str) -> None:
        validate_user_id(user_id)
        label = " ".join(str(label or "").split())[:_USER_LABEL_MAX_CHARS]
        with self._tx() as c:
            cur = c.execute("UPDATE users SET label = ? WHERE user_id = ?", (label, user_id))
            if cur.rowcount == 0:
                raise ValueError(f"用户不存在: {user_id}")

    def set_user_blocked(self, user_id: str, blocked: bool) -> None:
        validate_user_id(user_id)
        if user_id == OWNER_USER_ID:
            raise ValueError("owner 不可被封禁")
        with self._tx() as c:
            cur = c.execute("UPDATE users SET blocked = ? WHERE user_id = ?", (1 if blocked else 0, user_id))
            if cur.rowcount == 0:
                raise ValueError(f"用户不存在: {user_id}")

    # ── meta ──────────────────────────────────────────────────

    def _get_meta_locked(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    @staticmethod
    def _set_meta_locked(c: sqlite3.Connection, key: str, value: str) -> None:
        c.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    # ── 事件 ──────────────────────────────────────────────────

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> SessionEvent:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return SessionEvent(
            id=row["id"],
            session_id=row["session_id"],
            seq=row["seq"],
            ts=row["ts"],
            type=row["type"],
            payload=payload,
            event_key=row["event_key"],
        )

    def _emit_locked(
        self,
        c: sqlite3.Connection,
        session_id: str,
        type_: str,
        payload: dict,
        key: Optional[str],
        now: float,
    ) -> Optional[int]:
        """在已开启的事务里追加事件。同一会话同一 key 已存在则返回 None。"""
        sess = c.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if sess is None:
            raise ValueError(f"会话不存在: {session_id}")
        if key is not None and c.execute(
            "SELECT 1 FROM events WHERE session_id = ? AND event_key = ?", (session_id, key)
        ).fetchone():
            return None
        seq = c.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE session_id = ?", (session_id,)).fetchone()[0]
        search_text = " ".join(s for s in _string_leaves(payload) if s)
        c.execute(
            "INSERT INTO events(session_id, seq, ts, type, payload, search_text, event_key) VALUES (?,?,?,?,?,?,?)",
            (session_id, seq, now, type_, json.dumps(payload, ensure_ascii=False, default=str), search_text, key),
        )
        c.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        if type_ == "user_message" and not sess["title"]:
            title = _auto_title(payload.get("text", ""))
            if title:
                c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
        return seq

    def emit_event(self, session_id: str, type: str, payload: Optional[dict] = None, *, key: Optional[str] = None) -> Optional[int]:
        """追加一条事件，返回 seq；带 key 且同一会话已有该 key 时不插入、返回 None（幂等）。"""
        payload = payload or {}
        if key is not None:
            with self._lock:  # 先做只读预检：重复镜像时不必抢写锁
                if self._conn.execute(
                    "SELECT 1 FROM events WHERE session_id = ? AND event_key = ?", (session_id, key)
                ).fetchone():
                    return None
        with self._tx() as c:
            return self._emit_locked(c, session_id, type, payload, key, time.time())

    @staticmethod
    def _type_clause(types: Optional[Iterable[str]], col: str = "type") -> tuple[str, list]:
        types = [t for t in (types or []) if t]
        if not types:
            return "", []
        return f" AND {col} IN ({','.join('?' for _ in types)})", types

    def get_events(self, session_id: str, *, start_seq: int = 1, limit: int = 50, types: Optional[Iterable[str]] = None) -> list[SessionEvent]:
        clause, args = self._type_clause(types)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM events WHERE session_id = ? AND seq >= ?{clause} ORDER BY seq ASC LIMIT ?",
                [session_id, int(start_seq), *args, int(limit)],
            ).fetchall()
        return [self._event_from_row(r) for r in rows]

    def get_last_events(self, session_id: str, limit: int = 10, types: Optional[Iterable[str]] = None) -> list[SessionEvent]:
        """最新 N 条，升序返回。"""
        clause, args = self._type_clause(types)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM events WHERE session_id = ?{clause} ORDER BY seq DESC LIMIT ?",
                [session_id, *args, int(limit)],
            ).fetchall()
        return [self._event_from_row(r) for r in reversed(rows)]

    def get_pending_events(self, session_id: str) -> list[SessionEvent]:
        marks = ",".join("?" for _ in INBOUND_EVENT_TYPES)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT e.* FROM events e JOIN sessions s ON s.id = e.session_id"
                f" WHERE e.session_id = ? AND e.type IN ({marks}) AND e.seq > s.processed_seq ORDER BY e.seq ASC",
                [session_id, *INBOUND_EVENT_TYPES],
            ).fetchall()
        return [self._event_from_row(r) for r in rows]

    def mark_processed(self, session_id: str, up_to_seq: Optional[int] = None) -> None:
        """推进处理游标（只前进不后退）；默认推进到当前最大 seq。"""
        with self._tx() as c:
            if up_to_seq is None:
                up_to_seq = c.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
            c.execute(
                "UPDATE sessions SET processed_seq = MAX(processed_seq, ?) WHERE id = ?", (int(up_to_seq), session_id)
            )

    def last_seq(self, session_id: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session_id = ?", (session_id,)
            ).fetchone()[0]

    def count_events(self, session_id: str, types: Optional[Iterable[str]] = None) -> int:
        clause, args = self._type_clause(types)
        with self._lock:
            return self._conn.execute(
                f"SELECT COUNT(*) FROM events WHERE session_id = ?{clause}", [session_id, *args]
            ).fetchone()[0]

    def sessions_with_pending(self) -> list[str]:
        marks = ",".join("?" for _ in INBOUND_EVENT_TYPES)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT DISTINCT s.id FROM sessions s JOIN events e ON e.session_id = s.id"
                f" WHERE e.type IN ({marks}) AND e.seq > s.processed_seq",
                list(INBOUND_EVENT_TYPES),
            ).fetchall()
        return [r["id"] for r in rows]

    def search_events(
        self,
        query: str,
        *,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        types: Optional[Iterable[str]] = None,
    ) -> list[SessionEvent]:
        """空白分词，全部命中（AND），大小写不敏感（SQLite LIKE 对 ASCII 不区分大小写），最新优先。
        只搜该用户自己的会话（JOIN sessions 过滤；JOIN 之后 id / type 会有歧义，所以事件表统一用别名 e）。"""
        validate_user_id(user_id)
        tokens = [t for t in (query or "").split() if t]
        if not tokens:
            return []
        where = ["s.user_id = ?"]
        args: list = [user_id]
        if session_id:
            where.append("e.session_id = ?")
            args.append(session_id)
        clause, type_args = self._type_clause(types, col="e.type")
        for tok in tokens:
            where.append("e.search_text LIKE ? ESCAPE '\\'")
            args.append(f"%{_like_escape(tok)}%")
        sql = (
            "SELECT e.* FROM events e JOIN sessions s ON s.id = e.session_id"
            f" WHERE {' AND '.join(where)}{clause} ORDER BY e.id DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, [*args, *type_args, int(limit)]).fetchall()
        return [self._event_from_row(r) for r in rows]

    # ── 挂起态（取代原来的 _turn_user_text / _pending_outbox 全局变量）──

    def save_turn_state(self, session_id: str, user_text: Optional[str], outbox: list) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO turn_state(session_id, user_text, outbox, updated_at) VALUES (?,?,?,?)"
                " ON CONFLICT(session_id) DO UPDATE SET user_text = excluded.user_text,"
                " outbox = excluded.outbox, updated_at = excluded.updated_at",
                (session_id, user_text, json.dumps(outbox or [], ensure_ascii=False), time.time()),
            )

    def load_turn_state(self, session_id: str) -> Optional[TurnState]:
        with self._lock:
            row = self._conn.execute(
                "SELECT user_text, outbox FROM turn_state WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            outbox = json.loads(row["outbox"])
        except (TypeError, ValueError):
            outbox = []
        return TurnState(user_text=row["user_text"], outbox=outbox if isinstance(outbox, list) else [])

    def clear_turn_state(self, session_id: str) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM turn_state WHERE session_id = ?", (session_id,))


# ── 模块级单例（惰性创建；测试用 set_store 注入，风格同 sandbox.set_docker_client）──

_store: Optional[SessionStore] = None
_store_lock = threading.Lock()


def get_store() -> SessionStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = SessionStore(config.SESSIONS_DB_PATH)
        return _store


def set_store(store: Optional[SessionStore]) -> None:
    global _store
    with _store_lock:
        _store = store


def close_store() -> None:
    global _store
    with _store_lock:
        store, _store = _store, None
    if store is not None:
        store.close()
