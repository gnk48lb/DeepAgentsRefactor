"""会话存储：会话 / 事件日志 / 挂起态 / 活动会话指针 (所在层: L1)

标准库 sqlite3 实现，单连接 + RLock。事件日志只追加（除 delete_session 外不做 UPDATE / DELETE），
“待处理”由会话游标 processed_seq 表达。模型上下文的续跑不在这里：它由 LangGraph checkpoint 负责。
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

# 这两类事件是“入站”：它们会驱动一次 wake，处理完后由 mark_processed 推进游标
INBOUND_EVENT_TYPES = ("user_message", "approval")

_SCHEMA_VERSION = 1
_ACTIVE_KEY = "active_session_id"

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


@dataclass(frozen=True)
class SessionInfo:
    id: str
    title: str
    created_at: float
    updated_at: float
    processed_seq: int
    turns: int = 0      # user_message 事件数（仅 list_sessions / get_session 填充）
    events: int = 0     # 事件总数

    @property
    def short_id(self) -> str:
        return self.id[:8]


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
                self._conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")

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
            turns=row["turns"] if "turns" in keys else 0,
            events=row["events"] if "events" in keys else 0,
        )

    _SESSION_SELECT = (
        "SELECT s.id, s.title, s.created_at, s.updated_at, s.processed_seq,"
        " (SELECT COUNT(*) FROM events e WHERE e.session_id = s.id AND e.type = 'user_message') AS turns,"
        " (SELECT COUNT(*) FROM events e WHERE e.session_id = s.id) AS events"
        " FROM sessions s"
    )

    def create_session(self, title: str = "", activate: bool = True) -> SessionInfo:
        title = " ".join(str(title or "").split())[: config.SESSION_TITLE_MAX_CHARS]
        sid = uuid.uuid4().hex
        now = time.time()
        with self._tx() as c:
            c.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at, processed_seq) VALUES (?,?,?,?,0)",
                (sid, title, now, now),
            )
            self._emit_locked(c, sid, "session_created", {"title": title}, None, now)
            if activate:
                self._set_meta_locked(c, _ACTIVE_KEY, sid)
        return self.get_session(sid)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        with self._lock:
            row = self._conn.execute(self._SESSION_SELECT + " WHERE s.id = ?", (session_id,)).fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(self, limit: int = 10) -> list[SessionInfo]:
        with self._lock:
            rows = self._conn.execute(
                self._SESSION_SELECT + " ORDER BY s.updated_at DESC, s.rowid DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [self._session_from_row(r) for r in rows]

    def count_sessions(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def resolve_session(self, ref: str) -> Optional[SessionInfo]:
        """完整 id 或唯一前缀。没有匹配返回 None，多义抛 ValueError。"""
        ref = (ref or "").strip().lower()
        if not ref:
            return None
        with self._lock:
            exact = self._conn.execute("SELECT id FROM sessions WHERE id = ?", (ref,)).fetchone()
            if exact:
                return self.get_session(exact["id"])
            rows = self._conn.execute(
                "SELECT id FROM sessions WHERE substr(id, 1, ?) = ? ORDER BY updated_at DESC, rowid DESC",
                (len(ref), ref),
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(f"前缀 {ref} 匹配到 {len(rows)} 个会话: " + ", ".join(r["id"][:12] for r in rows))
        return self.get_session(rows[0]["id"])

    def get_active_session_id(self) -> str:
        """当前活动会话 id。指针缺失或指向已删除的会话时，回落到最近更新的会话；一个会话都没有则新建默认会话。"""
        with self._lock:
            value = self._get_meta_locked(_ACTIVE_KEY)
            if value and self._conn.execute("SELECT 1 FROM sessions WHERE id = ?", (value,)).fetchone():
                return value
            row = self._conn.execute(
                "SELECT id FROM sessions ORDER BY updated_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
            if row:
                with self._tx() as c:
                    self._set_meta_locked(c, _ACTIVE_KEY, row["id"])
                return row["id"]
            return self.create_session("", activate=True).id

    def set_active_session(self, session_id: str) -> None:
        with self._tx() as c:
            if not c.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone():
                raise ValueError(f"会话不存在: {session_id}")
            self._set_meta_locked(c, _ACTIVE_KEY, session_id)

    def rename_session(self, session_id: str, title: str) -> None:
        title = " ".join(str(title or "").split())[: config.SESSION_TITLE_MAX_CHARS]
        with self._tx() as c:
            cur = c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            if cur.rowcount == 0:
                raise ValueError(f"会话不存在: {session_id}")

    def delete_session(self, session_id: str) -> bool:
        """硬删：事件 + 挂起态 + 会话行。仅运维脚本使用。活动会话被删时清掉指针。"""
        with self._tx() as c:
            exists = c.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
            c.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM turn_state WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            if self._get_meta_locked(_ACTIVE_KEY) == session_id:
                c.execute("DELETE FROM meta WHERE key = ?", (_ACTIVE_KEY,))
        return bool(exists)

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
    def _type_clause(types: Optional[Iterable[str]]) -> tuple[str, list]:
        types = [t for t in (types or []) if t]
        if not types:
            return "", []
        return f" AND type IN ({','.join('?' for _ in types)})", types

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
        session_id: Optional[str] = None,
        limit: int = 10,
        types: Optional[Iterable[str]] = None,
    ) -> list[SessionEvent]:
        """空白分词，全部命中（AND），大小写不敏感（SQLite LIKE 对 ASCII 不区分大小写），最新优先。"""
        tokens = [t for t in (query or "").split() if t]
        if not tokens:
            return []
        where = ["1=1"]
        args: list = []
        if session_id:
            where.append("session_id = ?")
            args.append(session_id)
        clause, type_args = self._type_clause(types)
        for tok in tokens:
            where.append("search_text LIKE ? ESCAPE '\\'")
            args.append(f"%{_like_escape(tok)}%")
        sql = f"SELECT * FROM events WHERE {' AND '.join(where)}{clause} ORDER BY id DESC LIMIT ?"
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
