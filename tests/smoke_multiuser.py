"""
tests/smoke_multiuser.py

多用户改造（阶段 1）离线冒烟测试：不需要 API Key / Docker / 网络 / Milvus。
  1 身份与请求上下文   resolve_platform_user / validate_user_id / cli_member_identity / require_* / config 解析
  2 SessionStore v2    v1 库迁移（数据归 owner、指针迁移、幂等、整体回滚）与按用户隔离、users 表
  3 会话回溯工具       只读自己的会话：猜完整 session_id 也读不到、缺 user_id 拒绝
需要各自已有夹具的几组放在对应的测试文件里（复用它们现成的受控 embedding / 真实 AsyncSqliteSaver 栈 / MCP 与密钥桩）：
记忆隔离在 smoke_memory.py（10~12）、双图编排与 member 提示词在 smoke_orchestrator.py（E~H）、
chat_service 并发与命令在 smoke_chat_service.py、SQL 加固在 smoke_units.py。
"""
import asyncio
import dataclasses
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 本机 .env 里可能是 DEPLOY_MODE=cloud：load_dotenv 不覆盖已有的环境变量，所以这一行能压过它（本测试针对 desktop 形态）。
os.environ["DEPLOY_MODE"] = "desktop"

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from app import config
from app.infra import session_store
from app.infra.session_store import SessionStore
from app.tools import session_tools
from app.utils.context import (
    RequestContext,
    get_request_context,
    request_context,
    require_owner,
    require_request_context,
    reset_request_context,
    set_request_context,
)
from app.utils.identity import (
    Identity,
    cli_member_identity,
    invalid_owner_ids,
    owner_identity,
    resolve_platform_user,
    validate_user_id,
)
from _support import member_context, owner_context

ROOT = Path(__file__).resolve().parents[1]

OPENID = "A1B2C3D4E5F60718"        # 16 位，贴近真实 user_openid 的形状
OPENID_2 = "ZZ99YY88XX77WW66"


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return
    except Exception as e:
        raise AssertionError(f"期望 {exc.__name__}，实际 {type(e).__name__}: {e}")
    raise AssertionError(f"期望抛 {exc.__name__}，但没有抛：{fn.__name__}{args!r}")


class _OwnerIds:
    """临时改写 config.OWNER_USER_IDS。"""

    def __init__(self, ids):
        self.ids = frozenset(ids)

    def __enter__(self):
        self.old = config.OWNER_USER_IDS
        config.OWNER_USER_IDS = self.ids

    def __exit__(self, *exc):
        config.OWNER_USER_IDS = self.old


# ══════════════════════════════════════════════════════════════════════════════
# 组 1：身份与请求上下文
# ══════════════════════════════════════════════════════════════════════════════

def test_1_resolve_platform_user():
    """1.1 格式校验与 owner 映射。"""
    with _OwnerIds({OPENID}):
        assert resolve_platform_user(OPENID) == Identity(user_id="owner", is_owner=True, raw_id=OPENID)
        assert resolve_platform_user(OPENID_2) == Identity(user_id=OPENID_2, is_owner=False, raw_id=OPENID_2)
        assert resolve_platform_user("a" * 8).user_id == "a" * 8
        assert resolve_platform_user("a" * 64).user_id == "a" * 64
    with _OwnerIds(set()):
        assert not resolve_platform_user(OPENID).is_owner, "OWNER_USER_IDS 为空时没有任何平台用户是 owner"

    bad = [
        "", "a" * 7, "a" * 65, "abc-defgh1", "abc defgh1", " " + "a" * 8, "a" * 8 + " ",
        "a" * 8 + "\n", "\n" + "a" * 8, "a" * 8 + "\r\n",           # re.match + $ 会放过末尾换行，必须用 fullmatch
        'abc"defgh', "abc'defgh1", "abc;defgh1", "owner", "cli-alt12345",
        "１２３４５６７８", "中文用户标识符号啊啊",                       # 全角数字 / 非 ASCII
        None, 12345678, b"abcdefgh",
    ]
    for raw in bad:
        _raises(ValueError, resolve_platform_user, raw)
    # owner 的 ID 带上空白变体也进不来
    with _OwnerIds({OPENID}):
        _raises(ValueError, resolve_platform_user, OPENID + "\n")
        _raises(ValueError, resolve_platform_user, " " + OPENID)


def test_2_no_impersonation():
    """1.2 平台来源的 id 不可能产生 owner，也不可能产生 cli- 前缀。"""
    with _OwnerIds(set()):
        for raw in ("owner", "ownerownerowner", "Owner0000", "cli0alt1xx", "cli_alt1xxx", "cli-alt1xxxx", "cli-owner"):
            try:
                ident = resolve_platform_user(raw)
            except ValueError:
                continue
            assert ident.user_id != "owner" and not ident.user_id.startswith("cli-") and not ident.is_owner, ident
        for raw in ("owner", "cli-alt1xxxx", "cli-owner"):
            _raises(ValueError, resolve_platform_user, raw)


def test_3_validate_user_id():
    """1.3 validate_user_id 只接受 owner / 平台 openid / cli-...，拒绝引号、空白、换行与冒充。"""
    for ok in ("owner", OPENID, "a" * 8, "a" * 64, "cli-alt1", "cli-A_b_9", "cli-" + "x" * 32, "ownerxyz1"):
        assert validate_user_id(ok) == ok
    bad = [
        "", "Owner", "OWNER", "owner ", " owner", "owner\n", "owner\x00",
        'abc"defgh', "abc'defgh1", "a b c d e f g h", "abcdefg", "a" * 65, "a" * 8 + "\n",
        "cli-", "cli-a b", "cli-" + "x" * 33, "CLI-alt1", "cli--alt1", "cli-alt1\n", "cli-alt-1",
        "x;DROP", "id == 1 or 1==1", "owner' or '1'=='1", "user_memory_v2_u_",
        None, 1, b"owner", ["owner"],
    ]
    for raw in bad:
        _raises(ValueError, validate_user_id, raw)


def test_4_cli_identities():
    """1.4 owner_identity / cli_member_identity。"""
    assert owner_identity() == Identity("owner", True, "cli")
    assert owner_identity("X").raw_id == "X"
    m = cli_member_identity("alt1")
    assert m.user_id == "cli-alt1" and not m.is_owner and validate_user_id(m.user_id) == "cli-alt1"
    assert cli_member_identity("a" * 32).user_id == "cli-" + "a" * 32
    for bad in ("", "a-b", "a b", "a" * 33, "中文", "a\n", None, 7):
        _raises(ValueError, cli_member_identity, bad)


def test_5_request_context():
    """1.5 require_request_context / require_owner：失败即关闭，嵌套与异常路径都能还原。"""
    assert get_request_context() is None
    _raises(RuntimeError, require_request_context)
    _raises(RuntimeError, require_owner)            # 没有上下文是 RuntimeError，不是 PermissionError（后者会被当成“是 member”）

    with owner_context("s1") as c:
        assert get_request_context() is c and c.user_id == "owner" and c.is_owner and c.session_id == "s1"
        assert require_request_context() is c and require_owner() is c
        with member_context("member001", "s2") as m:
            assert get_request_context() is m and require_request_context() is m
            _raises(PermissionError, require_owner)
        assert get_request_context() is c            # 内层退出后还原外层
    assert get_request_context() is None

    try:
        with owner_context():
            raise KeyError("boom")
    except KeyError:
        pass
    assert get_request_context() is None, "异常路径也必须还原上下文"

    ctx = RequestContext("owner", "s", True)
    try:
        ctx.is_owner = False
        raise AssertionError("RequestContext 应当不可变")
    except dataclasses.FrozenInstanceError:
        pass
    token = set_request_context(ctx)
    assert get_request_context() is ctx
    reset_request_context(token)
    assert get_request_context() is None


def test_6_context_isolated_between_tasks():
    """1.6 并发的 asyncio task 各看各的上下文，主 task 不受影响。"""
    seen = {}

    async def worker(name, ctx, delay):
        with request_context(ctx):
            await asyncio.sleep(delay)
            seen[name] = get_request_context().user_id
            await asyncio.sleep(0.01)
            seen[name + "_after"] = get_request_context().user_id

    async def main():
        a = RequestContext("owner", "sa", True)
        b = RequestContext("member001", "sb", False)
        await asyncio.gather(worker("a", a, 0.05), worker("b", b, 0.01))
        assert get_request_context() is None

    asyncio.run(main())
    assert seen == {"a": "owner", "b": "member001", "a_after": "owner", "b_after": "member001"}, seen


def _config_probe(env_extra: dict) -> list:
    code = (
        "from app import config\n"
        "print(','.join(sorted(config.OWNER_USER_IDS)))\n"
        "print(config.MAX_CONCURRENT_MEMBER_TURNS)\n"
        "print(config.MEMBER_WELCOME_TEXT)\n"
    )
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_extra)
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8"
    )
    assert res.returncode == 0, res.stderr
    return res.stdout.splitlines()          # 不能 strip：第一行（OWNER_USER_IDS）可能是空串


def test_7_config_parsing():
    """1.7 OWNER_USER_IDS 去空白去空项；MAX_CONCURRENT_MEMBER_TURNS 钳制到 >= 1；欢迎语可覆盖、空值回落默认。"""
    assert config._parse_id_set("") == frozenset() and config._parse_id_set(None) == frozenset()
    assert config._parse_id_set(" A1 , ,B2,, A1 ") == frozenset({"A1", "B2"})

    out = _config_probe({"OWNER_USER_IDS": " A1B2C3D4 , ,E5F6G7H8 ", "MAX_CONCURRENT_MEMBER_TURNS": "0", "MEMBER_WELCOME_TEXT": ""})
    assert out[0] == "A1B2C3D4,E5F6G7H8" and out[1] == "1" and out[2].strip(), out
    out = _config_probe({"OWNER_USER_IDS": "", "MAX_CONCURRENT_MEMBER_TURNS": "5", "MEMBER_WELCOME_TEXT": "你好呀"})
    assert out == ["", "5", "你好呀"], out


def test_8_invalid_owner_ids():
    """1.8 OWNER_USER_IDS 里格式不合法的项（永远不会命中）能被找出来，供启动时警告。"""
    with _OwnerIds({OPENID, "bad-id", "short"}):
        assert invalid_owner_ids() == ["bad-id", "short"]
    with _OwnerIds({OPENID}):
        assert invalid_owner_ids() == []
    with _OwnerIds(set()):
        assert invalid_owner_ids() == []


# ══════════════════════════════════════════════════════════════════════════════
# 组 2：SessionStore v2（迁移与隔离）
# ══════════════════════════════════════════════════════════════════════════════

# 改造前（schema v1）的建表脚本，原样复制：用它造旧库，再用新代码打开
_V1_SCHEMA = """
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

U = "owner"
A_USER = "memberaaa1"
B_USER = "memberbbb2"


class _Dir:
    """临时目录；退出时清掉全局 store 并删除目录（Windows 要求先关连接，所以各场景自己负责 close）。"""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="mu_")
        return self

    def __exit__(self, *exc):
        session_store.set_store(None)
        shutil.rmtree(self.path, ignore_errors=True)


def _make_v1_db(path: str, sessions=("s1", "s2"), active="s2", with_turn_state=True) -> None:
    """用改造前的 schema 造一个 v1 库：含会话、事件、活动指针与挂起态。"""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.executescript(_V1_SCHEMA)
    conn.execute("PRAGMA user_version=1")
    for i, sid in enumerate(sessions, 1):
        conn.execute(
            "INSERT INTO sessions(id, title, created_at, updated_at, processed_seq) VALUES (?,?,?,?,?)",
            (sid, f"旧会话{i}", 100.0 + i, 200.0 + i, 1),
        )
        conn.execute(
            "INSERT INTO events(session_id, seq, ts, type, payload, search_text) VALUES (?,?,?,?,?,?)",
            (sid, 1, 100.0 + i, "user_message", json.dumps({"text": f"旧消息{i} 北京"}, ensure_ascii=False), f"旧消息{i} 北京"),
        )
        conn.execute(
            "INSERT INTO events(session_id, seq, ts, type, payload, search_text) VALUES (?,?,?,?,?,?)",
            (sid, 2, 101.0 + i, "user_message", json.dumps({"text": "未处理"}, ensure_ascii=False), "未处理"),
        )
    if with_turn_state and sessions:
        conn.execute(
            "INSERT INTO turn_state(session_id, user_text, outbox, updated_at) VALUES (?,?,?,?)",
            (sessions[0], "挂起前的原话", "[]", 1.0),
        )
    conn.execute("INSERT INTO meta(key, value) VALUES ('active_session_id', ?)", (active,))
    conn.close()


def _raw(path: str, sql: str, args=()):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def test_9_migrate_v1_database():
    """2.1 v1 库用新代码打开：数据归 owner、指针迁移、索引与 users 表存在、再次打开幂等。"""
    with _Dir() as d:
        path = os.path.join(d.path, "sessions.db")
        _make_v1_db(path)
        assert _raw(path, "PRAGMA user_version")[0][0] == 1

        s = SessionStore(path)
        try:
            assert _raw(path, "PRAGMA user_version")[0][0] == 2
            assert {r[0] for r in _raw(path, "SELECT user_id FROM sessions")} == {"owner"}, "旧会话全部归 owner"
            assert [i.id for i in s.list_sessions(10, user_id=U)] == ["s2", "s1"], "旧会话在 owner 名下可见，顺序不变"
            assert s.count_sessions(user_id=U) == 2 and s.count_sessions() == 2
            # 活动指针：旧键改名为 active_session_id:owner，旧键丢弃
            meta = dict(_raw(path, "SELECT key, value FROM meta"))
            assert meta == {"active_session_id:owner": "s2"}, meta
            assert s.get_active_session_id(user_id=U) == "s2"
            # 索引与 users 表
            idx = {r[0] for r in _raw(path, "SELECT name FROM sqlite_master WHERE type='index'")}
            assert "idx_sessions_user_updated" in idx and "idx_events_session_type" in idx, idx
            assert _raw(path, "SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            assert s.list_users() == []
            # 事件、游标与挂起态原样保留
            assert [e.seq for e in s.get_events("s1", limit=10)] == [1, 2]
            assert s.get_session("s1").processed_seq == 1
            assert [e.seq for e in s.get_pending_events("s1")] == [2]
            assert s.load_turn_state("s1").user_text == "挂起前的原话"
            assert len(s.search_events("北京", user_id=U)) == 2
        finally:
            s.close()

        # 再次打开：幂等（不报错、不重复、数据不变）
        before = _raw(path, "SELECT id, user_id, title, processed_seq FROM sessions ORDER BY id")
        s2 = SessionStore(path)
        try:
            assert _raw(path, "SELECT id, user_id, title, processed_seq FROM sessions ORDER BY id") == before
            assert dict(_raw(path, "SELECT key, value FROM meta")) == {"active_session_id:owner": "s2"}
            assert s2.get_active_session_id(user_id=U) == "s2"
            assert _raw(path, "PRAGMA user_version")[0][0] == 2
        finally:
            s2.close()


def test_10_migrate_edge_cases():
    """2.2 迁移边角：新旧指针键同时存在时丢弃旧键；没有指针的旧库；全新库走同一条路径；版本过高拒绝打开。"""
    with _Dir() as d:
        # 新键已存在 -> 保留新键、丢弃旧键
        path = os.path.join(d.path, "both.db")
        _make_v1_db(path, active="s1")
        conn = sqlite3.connect(path, isolation_level=None)
        conn.execute("INSERT INTO meta(key, value) VALUES ('active_session_id:owner', 's2')")
        conn.close()
        SessionStore(path).close()
        assert dict(_raw(path, "SELECT key, value FROM meta")) == {"active_session_id:owner": "s2"}

        # 没有指针的旧库：迁移后回落到最近更新的会话
        path = os.path.join(d.path, "nopointer.db")
        _make_v1_db(path)
        conn = sqlite3.connect(path, isolation_level=None)
        conn.execute("DELETE FROM meta")
        conn.close()
        s = SessionStore(path)
        try:
            assert s.get_active_session_id(user_id=U) == "s2"
        finally:
            s.close()

        # 全新库与旧库走同一条路径：得到同样的 schema
        fresh = os.path.join(d.path, "fresh.db")
        SessionStore(fresh).close()
        old = os.path.join(d.path, "old.db")
        _make_v1_db(old, sessions=(), active="x", with_turn_state=False)
        SessionStore(old).close()

        def shape(p):
            return (
                [(r[1], r[2], r[3], r[4]) for r in _raw(p, "PRAGMA table_info(sessions)")],
                sorted(r[0] for r in _raw(p, "SELECT name FROM sqlite_master WHERE type IN ('table','index') AND name NOT LIKE 'sqlite_%'")),
                _raw(p, "PRAGMA user_version")[0][0],
            )
        assert shape(fresh) == shape(old), (shape(fresh), shape(old))

        # 版本号高于代码支持的版本：仍拒绝打开
        future = os.path.join(d.path, "future.db")
        _make_v1_db(future)
        conn = sqlite3.connect(future, isolation_level=None)
        conn.execute("PRAGMA user_version=3")
        conn.close()
        try:
            SessionStore(future)
            raise AssertionError("版本过高应拒绝打开")
        except RuntimeError as e:
            assert "高于当前代码支持" in str(e), e


def test_11_migration_is_atomic():
    """2.3 迁移中途失败：整体回滚（列没加、版本号仍是 1、旧指针键还在），修好后再打开能完成迁移。"""
    with _Dir() as d:
        path = os.path.join(d.path, "sessions.db")
        _make_v1_db(path)
        orig = SessionStore.__dict__["_set_meta_locked"]       # staticmethod 对象本身，恢复时才不会丢掉 staticmethod

        def boom(c, key, value):
            raise RuntimeError("模拟迁移中途失败")

        SessionStore._set_meta_locked = staticmethod(boom)
        try:
            try:
                SessionStore(path)
                raise AssertionError("迁移失败应向上抛")
            except RuntimeError as e:
                assert "模拟迁移中途失败" in str(e), e
        finally:
            SessionStore._set_meta_locked = orig
        assert _raw(path, "PRAGMA user_version")[0][0] == 1
        assert "user_id" not in [r[1] for r in _raw(path, "PRAGMA table_info(sessions)")]
        assert dict(_raw(path, "SELECT key, value FROM meta")) == {"active_session_id": "s2"}
        assert not _raw(path, "SELECT name FROM sqlite_master WHERE name IN ('users','idx_sessions_user_updated')")
        s = SessionStore(path)
        try:
            assert _raw(path, "PRAGMA user_version")[0][0] == 2 and s.get_active_session_id(user_id=U) == "s2"
        finally:
            s.close()


def test_12_user_isolation():
    """2.4 A 与 B 的会话、活动指针、前缀解析、搜索互不可见；跨用户 set_active_session 抛错。"""
    with _Dir() as d:
        s = SessionStore(os.path.join(d.path, "sessions.db"))
        try:
            a1 = s.create_session("A的会话一", user_id=A_USER)
            a2 = s.create_session("A的会话二", user_id=A_USER)
            b1 = s.create_session("B的会话", user_id=B_USER)
            o1 = s.create_session("owner的会话", user_id=U)
            for sess, text in ((a1, "A的机密 北京"), (b1, "B的机密 北京"), (o1, "owner的机密 北京")):
                s.emit_event(sess.id, "user_message", {"text": text})
            assert a1.user_id == A_USER and o1.user_id == U

            # 列表与计数
            assert {i.id for i in s.list_sessions(10, user_id=A_USER)} == {a1.id, a2.id}
            assert [i.id for i in s.list_sessions(10, user_id=B_USER)] == [b1.id]
            assert s.count_sessions(user_id=A_USER) == 2 and s.count_sessions(user_id=B_USER) == 1 and s.count_sessions() == 4

            # get_session：传了 user_id 而不属于该用户 -> None；不传（内部归属查询）才能看到
            assert s.get_session(b1.id, user_id=A_USER) is None
            assert s.get_session(b1.id, user_id=B_USER).id == b1.id
            assert s.get_session(b1.id).user_id == B_USER

            # 活动指针互相独立
            assert s.get_active_session_id(user_id=A_USER) == a2.id
            assert s.get_active_session_id(user_id=B_USER) == b1.id
            assert s.get_active_session_id(user_id=U) == o1.id
            s.set_active_session(a1.id, user_id=A_USER)
            assert s.get_active_session_id(user_id=A_USER) == a1.id and s.get_active_session_id(user_id=B_USER) == b1.id
            for who, target in ((A_USER, b1.id), (B_USER, a1.id), (A_USER, o1.id), (U, a1.id)):
                _raises(ValueError, s.set_active_session, target, user_id=who)
            assert s.get_active_session_id(user_id=A_USER) == a1.id, "跨用户设置失败后指针不变"
            _raises(ValueError, s.rename_session, b1.id, "改名", user_id=A_USER)
            assert s.get_session(b1.id).title == "B的会话"
            s.rename_session(b1.id, "B改名", user_id=B_USER)
            assert s.get_session(b1.id).title == "B改名"

            # 该用户还没有会话 -> 为其新建，且归属正确
            c_user = "memberccc3"
            new_sid = s.get_active_session_id(user_id=c_user)
            assert s.get_session(new_sid).user_id == c_user and s.count_sessions(user_id=c_user) == 1

            # 前缀 / 完整 id 解析只在自己的会话里：别人的会话解析不到
            assert s.resolve_session(a1.id, user_id=A_USER).id == a1.id
            assert s.resolve_session(b1.id, user_id=A_USER) is None
            assert s.resolve_session(b1.id[:8], user_id=A_USER) is None
            assert s.resolve_session(b1.id.upper(), user_id=A_USER) is None
            assert s.resolve_session(b1.id[:8], user_id=B_USER).id == b1.id
            assert s.resolve_session(a1.id, user_id=U) is None, "owner 也只在自己的会话里解析（运维走 resolve_session_any）"
            assert s.resolve_session_any(b1.id).id == b1.id and s.resolve_session_any(a1.id[:10]).id == a1.id

            # 多义判断只算自己的：给 A 与 B 各造一个同前缀 id，各自只看到自己的那个
            with s._tx() as c:
                c.execute("INSERT INTO sessions(id,title,created_at,updated_at,user_id) VALUES ('ffff0001','',1,1,?)", (A_USER,))
                c.execute("INSERT INTO sessions(id,title,created_at,updated_at,user_id) VALUES ('ffff0002','',1,1,?)", (B_USER,))
            assert s.resolve_session("ffff", user_id=A_USER).id == "ffff0001", "前缀只命中自己的一个，不算多义"
            assert s.resolve_session("ffff", user_id=B_USER).id == "ffff0002"
            _raises(ValueError, s.resolve_session_any, "ffff")

            # 搜索：只搜自己的会话
            assert [e.session_id for e in s.search_events("机密", user_id=A_USER)] == [a1.id]
            assert [e.session_id for e in s.search_events("机密", user_id=B_USER)] == [b1.id]
            assert [e.session_id for e in s.search_events("机密", user_id=U)] == [o1.id]
            assert s.search_events("B的机密", user_id=A_USER) == []
            assert s.search_events("机密", user_id=A_USER, session_id=b1.id) == [], "指定别人的 session_id 也搜不到"
            assert len(s.search_events("北京", user_id=A_USER, types=["user_message"])) == 1
            assert s.search_events("北京", user_id=A_USER, types=["assistant_message"]) == []

            # 非法 user_id 一律拒绝（不拼进键名 / SQL）
            for bad in ("", "x", "owner' OR '1'='1", "a" * 8 + "\n", None):
                _raises(ValueError, s.list_sessions, 10, user_id=bad)
                _raises(ValueError, s.get_active_session_id, user_id=bad)
                _raises(ValueError, s.create_session, "x", user_id=bad)
                _raises(ValueError, s.search_events, "x", user_id=bad)
            # user_id 是必填的关键字参数：不给就是 TypeError，没有默认值
            for fn, args in ((s.create_session, ("t",)), (s.list_sessions, (5,)), (s.get_active_session_id, ()),
                             (s.resolve_session, ("abc",)), (s.search_events, ("x",)), (s.rename_session, (a1.id, "x")),
                             (s.set_active_session, (a1.id,))):
                _raises(TypeError, fn, *args)

            # 删除会话清理的是它所属用户的活动指针，不碰别人的
            s.set_active_session(a2.id, user_id=A_USER)
            assert s.delete_session(a2.id) is True
            meta = dict(_raw(s.path, "SELECT key, value FROM meta"))
            assert f"active_session_id:{A_USER}" not in meta and meta.get(f"active_session_id:{B_USER}") == b1.id, meta
            assert s.get_active_session_id(user_id=A_USER) == a1.id, "A 的指针回落到 A 自己最近的会话"
        finally:
            s.close()


def test_13_users_table():
    """2.5 touch_user / get_user / list_users / resolve_user / set_user_label / set_user_blocked。"""
    with _Dir() as d:
        s = SessionStore(os.path.join(d.path, "sessions.db"))
        try:
            u, created = s.touch_user(U)
            assert created and u.user_id == U and not u.blocked and u.label == "", "owner 的行同样创建"
            u2, created2 = s.touch_user(U)
            assert not created2 and u2.first_seen == u.first_seen and u2.last_seen >= u.last_seen
            a, created = s.touch_user(A_USER)
            assert created and a.first_seen >= u.first_seen
            s.touch_user(B_USER)
            assert [x.user_id for x in s.list_users()] == [U, A_USER, B_USER], "按首次出现排序，序号稳定"
            s.touch_user(U)                                                # 刷新 last_seen 不改变顺序
            assert [x.user_id for x in s.list_users()] == [U, A_USER, B_USER]
            assert s.get_user(A_USER).user_id == A_USER and s.get_user("memberzzz9") is None
            for bad in ("", "x", "a b", None):
                _raises(ValueError, s.touch_user, bad)

            # 备注：折叠空白、限长
            s.set_user_label(A_USER, "  小明   QQ 12345  " + "x" * 100)
            lab = s.get_user(A_USER).label
            assert lab.startswith("小明 QQ 12345 x") and len(lab) <= 40, lab
            _raises(ValueError, s.set_user_label, "memberzzz9", "x")

            # 封禁：owner 不可被封禁；其余可封可解，touch_user 不会清掉封禁状态
            _raises(ValueError, s.set_user_blocked, U, True)
            assert not s.get_user(U).blocked
            s.set_user_blocked(A_USER, True)
            assert s.get_user(A_USER).blocked and not s.get_user(B_USER).blocked
            ua, _ = s.touch_user(A_USER)
            assert ua.blocked, "touch_user 只刷新 last_seen，不动封禁状态"
            s.set_user_blocked(A_USER, False)
            assert not s.get_user(A_USER).blocked
            _raises(ValueError, s.set_user_blocked, "memberzzz9", True)

            # resolve_user：完整值 / 唯一前缀（不区分大小写）；多义抛错；没有返回 None
            assert s.resolve_user(A_USER).user_id == A_USER
            assert s.resolve_user("MEMBERA").user_id == A_USER
            assert s.resolve_user("owner").user_id == U
            assert s.resolve_user("zzz") is None and s.resolve_user("") is None and s.resolve_user("   ") is None
            _raises(ValueError, s.resolve_user, "member")                  # memberaaa1 / memberbbb2 都以 member 开头
            s.touch_user("ownerxyz1")                                       # owner 是 ownerxyz1 的前缀：完整匹配优先，不算多义
            assert s.resolve_user("owner").user_id == U
        finally:
            s.close()


def test_14_static_session_insert_is_explicit():
    """2.6 sessions.user_id 在表里保留了 DEFAULT 'owner'（迁移要求）：app/ 里所有 INSERT INTO sessions 必须显式写 user_id；
    全项目不得出现 user_id / identity / default 的 'owner' 默认值。"""
    hits = []
    for f in (ROOT / "app").rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"INSERT\s+INTO\s+sessions\s*\(([^)]*)\)", text, re.IGNORECASE):
            hits.append((f.name, m.group(1)))
    assert hits, "没有找到 INSERT INTO sessions：这条静态检查失效了"
    for name, cols in hits:
        assert "user_id" in [c.strip() for c in cols.split(",")], f"{name} 里的 INSERT INTO sessions 没有显式写 user_id: {cols}"

    bad = re.compile(r"""(?:\buser_id|\bidentity|\bdefault)\s*(?::[^=\n]+)?=\s*['"]owner['"]""")
    # 自检：这条正则不能是空转的
    for sample in ('def f(user_id="owner"):', "user_id: str = 'owner'", 'identity="owner"', "default='owner'", 'default = "owner"'):
        assert bad.search(sample), sample
    for fine in ("OWNER_USER_ID = 'owner'", "user_id == 'owner'", "ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'owner'"):
        assert not bad.search(fine), fine
    for d in ("app", "plugins", "scripts"):
        for f in (ROOT / d).rglob("*.py"):
            for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                assert not bad.search(line), f"{f.relative_to(ROOT)}:{ln} 出现 owner 默认值: {line.strip()}"
    main_py = (ROOT / "main.py").read_text(encoding="utf-8")
    assert not bad.search(main_py), "main.py 里出现 owner 默认值"


# ══════════════════════════════════════════════════════════════════════════════
# 组 3：会话回溯工具只读自己的会话
# ══════════════════════════════════════════════════════════════════════════════

def _cfg(user_id=None, thread_id=None):
    configurable = {}
    if user_id is not None:
        configurable["user_id"] = user_id
    if thread_id is not None:
        configurable["thread_id"] = thread_id
    return {"configurable": configurable}


def test_15_session_tools_isolation():
    """3.1 B 无法读取或搜索 A 的会话（包括直接猜完整 session_id、拿别人的 thread_id）；缺 user_id 拒绝。"""
    with _Dir() as d:
        s = SessionStore(os.path.join(d.path, "sessions.db"))
        session_store.set_store(s)
        try:
            a = s.create_session("A的会话", user_id=A_USER)
            b = s.create_session("B的会话", user_id=B_USER)
            s.emit_event(a.id, "user_message", {"text": "A的秘密：银行卡尾号 8888"})
            s.emit_event(b.id, "user_message", {"text": "B的日常：红烧肉"})
            g = session_tools.get_session_events
            f = session_tools.search_session_history

            # 读：自己的会话可读；B 用 A 的完整 id / 前缀 / thread_id 都读不到
            assert "8888" in g.invoke({"session_id": a.id}, config=_cfg(A_USER))
            assert "8888" in g.invoke({}, config=_cfg(A_USER, a.id))
            assert "8888" in g.invoke({}, config=_cfg(A_USER)), "不带 session_id / thread_id：读该用户的活动会话"
            for out in (
                g.invoke({"session_id": a.id}, config=_cfg(B_USER)),
                g.invoke({"session_id": a.id[:8]}, config=_cfg(B_USER)),
                g.invoke({"session_id": a.id, "start_seq": 1, "limit": 5}, config=_cfg(B_USER)),
                g.invoke({}, config=_cfg(B_USER, a.id)),                                  # 别人的 thread_id：拒绝，不回落
                g.invoke({"session_id": a.id}, config=_cfg(B_USER, b.id)),
            ):
                assert "8888" not in out and "银行卡" not in out and "找不到" in out, out
            assert "红烧肉" in g.invoke({}, config=_cfg(B_USER)), "B 自己的活动会话不受影响"

            # 搜：B 搜不到 A 的内容；指定 A 的 session_id 也搜不到；概览只列自己的会话
            assert "8888" in f.invoke({"query": "银行卡"}, config=_cfg(A_USER))
            for out in (
                f.invoke({"query": "银行卡"}, config=_cfg(B_USER)),
                f.invoke({"query": "银行卡", "session_id": a.id}, config=_cfg(B_USER)),
                f.invoke({"query": "银行卡"}, config=_cfg(B_USER, a.id)),
            ):
                assert "8888" not in out, out
            ov = f.invoke({"query": ""}, config=_cfg(B_USER))
            assert "B的会话" in ov and "A的会话" not in ov and a.id[:8] not in ov, ov
            assert "A的会话" in f.invoke({"query": ""}, config=_cfg(A_USER))

            # 缺 user_id / user_id 非法：只给文字错误，不回落到活动会话或 owner
            for cfg in (None, {}, {"configurable": {}}, {"configurable": {"thread_id": a.id}},
                        _cfg("", a.id), _cfg("owner' OR 1=1", a.id), _cfg("x", a.id), {"configurable": {"user_id": 123}}):
                kw = {} if cfg is None else {"config": cfg}
                for out in (
                    g.invoke({"session_id": a.id}, **kw),
                    f.invoke({"query": "银行卡"}, **kw),
                    f.invoke({"query": ""}, **kw),
                ):
                    assert "无法确定当前用户" in out and "8888" not in out and "A的会话" not in out, (cfg, out)
        finally:
            s.close()


# ══════════════════════════════════════════════════════════════════════════════
# 组 7：请求上下文穿过 deepagents 的 task 子专家 / 线程池 / asyncio.to_thread，并发时互不串
# ══════════════════════════════════════════════════════════════════════════════

register_harness_profile(
    "reactivemodel",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset({"write_file", "edit_file", "delete", "execute", "ls", "glob", "grep"}),
    ),
)


class ReactiveModel(BaseChatModel):
    """无状态假模型：按输入消息决定回复（不靠调用次数），同一张图被多个用户并发使用时也不会串。"""
    role: str      # "main"（主管）| "sub"（声明式子专家）

    @property
    def _llm_type(self) -> str:
        return "reactive"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        has_tool_result = any(isinstance(m, ToolMessage) for m in messages)
        if self.role == "main":
            if has_tool_result:
                msg = AIMessage(content="主任务完成")
            else:   # 主管自己的探针 + 委派给子专家，并行发出
                msg = AIMessage(content="", tool_calls=[
                    {"name": "sup_probe", "args": {}, "id": "m_sup"},
                    {"name": "task", "args": {"subagent_type": "Probe", "description": "d"}, "id": "m_task"},
                ])
        else:
            if has_tool_result:
                msg = AIMessage(content="【内部专家汇报 - 来源：Probe】探针完成")
            else:   # 三种执行路径的探针并行发出：async 工具 / 同步工具（线程池）/ async 工具里再 to_thread
                msg = AIMessage(content="", tool_calls=[
                    {"name": "probe_async", "args": {}, "id": "s1"},
                    {"name": "probe_sync", "args": {}, "id": "s2"},
                    {"name": "probe_to_thread", "args": {}, "id": "s3"},
                ])
        return ChatResult(generations=[ChatGeneration(message=msg)])


_probe_log: list = []


def _snap(name: str, config_: RunnableConfig) -> None:
    ctx = get_request_context()
    _probe_log.append({
        "tool": name,
        "user": ctx.user_id if ctx else None,
        "session": ctx.session_id if ctx else None,
        "is_owner": ctx.is_owner if ctx else None,
        "cfg_user": ((config_ or {}).get("configurable") or {}).get("user_id"),
        "thread": threading.get_ident(),
    })


@tool
def sup_probe(config: RunnableConfig) -> str:
    """主管自己的同步探针。"""
    time.sleep(0.05)
    _snap("sup_probe", config)
    return "ok"


@tool
async def probe_async(config: RunnableConfig) -> str:
    """子专家里的 async 探针。"""
    await asyncio.sleep(0.1)
    _snap("probe_async", config)
    return "ok"


@tool
def probe_sync(config: RunnableConfig) -> str:
    """子专家里的同步探针：由 LangChain 放进线程池执行。"""
    time.sleep(0.1)
    _snap("probe_sync", config)
    return "ok"


@tool
async def probe_to_thread(config: RunnableConfig) -> str:
    """子专家里的 async 探针，内部再走 asyncio.to_thread。"""
    await asyncio.to_thread(lambda: (time.sleep(0.1), _snap("probe_to_thread", config)))
    return "ok"


def test_16_context_propagation_concurrent():
    """7.1 两个用户同时运行同一张图：经由 task 调用的声明式子专家里，每个探针都看到各自的 user_id 与 session_id；
    同步工具（线程池）与 asyncio.to_thread 两条路径都覆盖；没有上下文时探针看到 None。"""
    _probe_log.clear()
    agent = create_deep_agent(
        model=ReactiveModel(role="main"),
        tools=[sup_probe],
        subagents=[{
            "name": "Probe",
            "description": "探针专家",
            "system_prompt": "你是 Probe，用于测试。",
            "tools": [probe_async, probe_sync, probe_to_thread],
            "model": ReactiveModel(role="sub"),
        }],
        checkpointer=MemorySaver(),
    )
    main_thread = threading.get_ident()

    async def run_user(user_id, session_id, is_owner, with_context=True):
        cfg = {"configurable": {"thread_id": session_id, "user_id": user_id}, "recursion_limit": 60}
        t0 = time.monotonic()
        if with_context:
            with request_context(RequestContext(user_id=user_id, session_id=session_id, is_owner=is_owner)):
                await agent.ainvoke({"messages": [HumanMessage(content="go")]}, cfg)
        else:
            await agent.ainvoke({"messages": [HumanMessage(content="go")]}, cfg)
        return t0, time.monotonic()

    async def main():
        (a0, a1), (b0, b1) = await asyncio.gather(
            run_user("owner", "sess-owner", True),
            run_user("memberaaa1", "sess-member", False),
        )
        assert a0 < b1 and b0 < a1, "两个用户的运行必须真正重叠，否则这条测试证明不了并发隔离"
        assert get_request_context() is None, "主 task 的上下文不受影响"

    asyncio.run(main())

    by_session: dict = {}
    for rec in _probe_log:
        by_session.setdefault(rec["session"], []).append(rec)
    assert None not in by_session, f"有探针没看到请求上下文: {[r for r in _probe_log if r['session'] is None]}"
    assert set(by_session) == {"sess-owner", "sess-member"}, set(by_session)
    expected_tools = ["probe_async", "probe_sync", "probe_to_thread", "sup_probe"]
    for sess, uid, is_owner in (("sess-owner", "owner", True), ("sess-member", "memberaaa1", False)):
        recs = by_session[sess]
        assert sorted(r["tool"] for r in recs) == expected_tools, (sess, recs)
        assert {r["user"] for r in recs} == {uid} and {r["is_owner"] for r in recs} == {is_owner}, (sess, recs)
        assert {r["cfg_user"] for r in recs} == {uid}, "RunnableConfig.configurable.user_id 也要一路传到工具里"
        by_tool = {r["tool"]: r for r in recs}
        assert by_tool["probe_sync"]["thread"] != main_thread, "同步工具应当跑在线程池里（这样才算验证了线程池路径）"
        assert by_tool["probe_to_thread"]["thread"] != main_thread

    # 对照：不设置上下文，探针看到的就是 None（上下文确实来自 request_context，而不是别处泄漏）
    _probe_log.clear()

    async def control():
        await run_user("memberaaa1", "sess-control", False, with_context=False)

    asyncio.run(control())
    assert len(_probe_log) == 4 and all(r["user"] is None and r["session"] is None for r in _probe_log), _probe_log
    assert {r["cfg_user"] for r in _probe_log} == {"memberaaa1"}


# ══════════════════════════════════════════════════════════════════════════════
# 组 8：纵深防御——宿主机相关工具在 member / 无请求上下文时一律拒绝（member 的图里本来就没有它们，这是第二道保险）
# ══════════════════════════════════════════════════════════════════════════════

def test_17_run_host_command_requires_owner():
    """8.1 run_host_command：无上下文 / member 被拒且命令根本不执行；owner 放行。同步工具经线程池路径也一样。"""
    from app.agents.experts import file_agent
    from app.infra import host_shell

    calls = []
    orig_run, orig_fmt = host_shell.run_command, host_shell.format_result
    host_shell.run_command = lambda command, cwd=None, timeout=None: (calls.append(command), "RESULT")[1]
    host_shell.format_result = lambda res: f"ran:{res}"
    try:
        with tempfile.TemporaryDirectory() as td:
            tool_ = {t.name: t for t in file_agent.make_workspace_tools(td)}["run_host_command"]

            out = tool_.invoke({"command": "echo hi"})                                   # 没有请求上下文
            assert "仅管理员可用" in out and out.startswith("Error:") and calls == [], (out, calls)
            with member_context():
                out = tool_.invoke({"command": "echo hi"})
            assert "仅管理员可用" in out and calls == [], (out, calls)

            async def via_thread_pool(ctx_factory):
                with ctx_factory():
                    return await tool_.ainvoke({"command": "echo hi"})        # 同步工具由 LangChain 放进线程池执行

            assert "仅管理员可用" in asyncio.run(via_thread_pool(member_context)) and calls == []
            with owner_context():
                assert tool_.invoke({"command": "echo hi"}) == "ran:RESULT"
            assert asyncio.run(via_thread_pool(owner_context)) == "ran:RESULT"
            assert calls == ["echo hi", "echo hi"], calls
    finally:
        host_shell.run_command, host_shell.format_result = orig_run, orig_fmt


def test_18_markitdown_requires_owner():
    """8.2 convert_to_markdown（markitdown_tool.build_tools 的 _run）：无上下文 / member 被拒，底层 MCP 工具不被调用。"""
    from langchain_core.tools import StructuredTool
    from app.tools import markitdown_tool

    inner_calls = []

    async def _inner(uri: str) -> str:
        inner_calls.append(uri)
        return "MARKDOWN"

    inner = StructuredTool.from_function(coroutine=_inner, name="convert_to_markdown", description="d")
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "a.txt").write_text("hello", encoding="utf-8")
        wrapped = markitdown_tool.build_tools([inner], td)[0]

        async def call():
            return await wrapped.ainvoke({"uri": "/workspace/a.txt"})

        assert "仅管理员可用" in asyncio.run(call()) and inner_calls == []
        with member_context():
            assert "仅管理员可用" in asyncio.run(call()) and inner_calls == []
        with owner_context():
            assert asyncio.run(call()) == "MARKDOWN"
        assert len(inner_calls) == 1


def test_19_browser_action_requires_owner():
    """8.3 execute_complex_browser_action：无上下文 / member 被拒，真实浏览器任务不被触发。"""
    from app.tools import browser_tools

    calls = []

    async def fake_run(instruction):
        calls.append(instruction)
        return "BROWSED"

    orig = browser_tools.run_browser_task
    browser_tools.run_browser_task = fake_run
    try:
        async def call():
            return await browser_tools.execute_complex_browser_action.ainvoke({"instruction": "点一下登录"})

        assert "仅管理员可用" in asyncio.run(call()) and calls == []
        with member_context():
            assert "仅管理员可用" in asyncio.run(call()) and calls == []
        with owner_context():
            assert asyncio.run(call()) == "BROWSED"
        assert calls == ["点一下登录"]
    finally:
        browser_tools.run_browser_task = orig


class _DesktopProbeModel(BaseChatModel):
    """第一次调用发出一个桌面点击；看到工具结果后收尾。"""

    @property
    def _llm_type(self) -> str:
        return "desktop-probe"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content="点击已处理")
        else:
            msg = AIMessage(content="", tool_calls=[{"name": "click_grid_center", "args": {"grid_id": "A1"}, "id": "d1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_20_desktop_node_requires_owner():
    """8.4 DesktopAgent 的 da_tools_node：执行任何工具之前就拒绝（无上下文 / member），点击不会发生；owner 照常执行。"""
    from app.agents.experts import desktop_agent

    clicks = []

    @tool
    def click_grid_center(grid_id: str) -> str:
        """测试用点击工具：记录被真正执行的点击。"""
        clicks.append(grid_id)
        return f"clicked {grid_id}"

    graph = desktop_agent.build_desktop_agent_subgraph(
        desktop_tools=[click_grid_center], system_prompt="x", llm=_DesktopProbeModel()
    )

    def run():
        return asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="点击 A1")]}, config={"recursion_limit": 100}))

    def tool_results(res):
        return [m for m in res["messages"] if isinstance(m, ToolMessage)]

    res = run()                                                      # 没有请求上下文
    assert clicks == [] and tool_results(res) and all("仅管理员可用" in str(m.content) for m in tool_results(res)), res["messages"]
    with member_context():
        res = run()
    assert clicks == [] and tool_results(res) and all("仅管理员可用" in str(m.content) for m in tool_results(res)), res["messages"]
    with owner_context():
        res = run()
    assert clicks == ["A1"] and [str(m.content) for m in tool_results(res)] == ["clicked A1"], (clicks, res["messages"])


def test_21_guard_call_sites_and_layer():
    """8.5 守卫就在文档指定的 4 个位置，而且没有加在 L1 的 host_shell 里（运维脚本与测试要能直接用它）。"""
    expected = {
        "app/agents/experts/file_agent.py": "run_host_command",
        "app/tools/markitdown_tool.py": "_run",
        "app/agents/experts/desktop_agent.py": "da_tools_node",
        "app/tools/browser_tools.py": "execute_complex_browser_action",
    }
    import ast
    for rel, func in expected.items():
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func]
        assert fns, f"{rel} 里找不到 {func}"
        calls_guard = any(
            isinstance(c, ast.Call) and getattr(c.func, "id", None) == "require_owner"
            for fn in fns for c in ast.walk(fn)
        )
        assert calls_guard, f"{rel}:{func} 里没有调用 require_owner()"
    assert "require_owner" not in (ROOT / "app/infra/host_shell.py").read_text(encoding="utf-8"), "守卫不要加在 L1 的 host_shell 里"


def test_22_metrics_record_carries_user():
    """8.6 cli_logger 的 LLM 指标记录带 user 字段：来自 configurable.user_id，主管与子专家（嵌套命名空间）都有；取不到记空串。"""
    from app.runtime import cli_logger

    agent = create_deep_agent(
        model=ReactiveModel(role="main"),
        tools=[sup_probe],
        subagents=[{
            "name": "Probe",
            "description": "探针专家",
            "system_prompt": "你是 Probe，用于测试。",
            "tools": [probe_async, probe_sync, probe_to_thread],
            "model": ReactiveModel(role="sub"),
        }],
        checkpointer=MemorySaver(),
    )
    old_path, old_debug, old_metrics = config.LLM_METRICS_PATH, config.CLI_DEBUG, config.CLI_METRICS
    with tempfile.TemporaryDirectory() as td:
        config.LLM_METRICS_PATH = os.path.join(td, "llm_metrics.jsonl")
        config.CLI_DEBUG, config.CLI_METRICS = "0", "1"
        try:
            def run(thread_id, user_id):
                # 与 chat_service._run_config 的形状一致（那边有单测）：user_id 同时在 configurable 与 metadata 里
                cfg = {
                    "configurable": {"thread_id": thread_id, **({"user_id": user_id} if user_id else {})},
                    "metadata": {"user_id": user_id} if user_id else {},
                    "recursion_limit": 60,
                    "callbacks": [cli_logger.CLIDebugCallbackHandler()],
                }
                asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="go")]}, cfg))

            run("t-a", "memberaaa1")
            run("t-b", "owner")
            run("t-c", "")                                           # 没有 user_id：记空串
            with open(config.LLM_METRICS_PATH, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        finally:
            config.LLM_METRICS_PATH, config.CLI_DEBUG, config.CLI_METRICS = old_path, old_debug, old_metrics
    llm = [r for r in records if r.get("type") == "llm"]
    assert llm and all("user" in r for r in llm), llm
    by_thread = {}
    for r in llm:
        by_thread.setdefault(r["thread"], []).append(r)
    assert {r["user"] for r in by_thread["t-a"]} == {"memberaaa1"}, by_thread["t-a"]
    assert {r["user"] for r in by_thread["t-b"]} == {"owner"}
    assert {r["user"] for r in by_thread["t-c"]} == {""}
    # 主管（单层命名空间）与子专家（含 '|' 的嵌套命名空间）的调用都带上了 user
    assert any("|" not in r["ns"] for r in by_thread["t-a"]) and any("|" in r["ns"] for r in by_thread["t-a"]), by_thread["t-a"]


# ══════════════════════════════════════════════════════════════════════════════
# 组 8（续）：QQ 插件只处理好友私聊（C2C），身份来自 user_openid
# ══════════════════════════════════════════════════════════════════════════════

def _load_chat_plugin():
    """不启动 NoneBot：把 on_message / get_driver 换成桩后，按文件路径加载 plugins/chat.py，拿到真实的 handle_chat。"""
    import importlib.util
    import nonebot

    class _Driver:
        def on_startup(self, func):
            return func

        def on_shutdown(self, func):
            return func

    class _MatcherFactory:
        def handle(self):
            return lambda func: func

    orig = (nonebot.on_message, nonebot.get_driver)
    nonebot.on_message = lambda *a, **k: _MatcherFactory()
    nonebot.get_driver = lambda: _Driver()
    try:
        spec = importlib.util.spec_from_file_location("chat_plugin_under_test", ROOT / "plugins" / "chat.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        nonebot.on_message, nonebot.get_driver = orig
    return mod


def test_23_qq_plugin_c2c_only_and_identity():
    """8.7 插件只接受好友私聊（C2C）：群 @ / 频道 / 频道私信一律忽略且日志限频；openid 格式不合法忽略；
    owner 的 openid 归一为 owner；被 /stop 中止的轮（None）不再回复。"""
    import contextlib
    import io
    from nonebot.adapters.qq.event import (
        C2CMessageCreateEvent,
        DirectMessageCreateEvent,
        GroupAtMessageCreateEvent,
        MessageCreateEvent,
    )
    from nonebot.adapters.qq.message import Message, MessageSegment
    from nonebot.adapters.qq.models import FriendAuthor
    from nonebot.exception import FinishedException
    from app.services.chat_service import ChatReply

    mod = _load_chat_plugin()
    calls = []
    state = {"result": ChatReply("机器人的回复", [])}

    async def fake_process(text, image_url=None, *, identity):
        calls.append((text, image_url, identity))
        return state["result"]

    class FakeMatcher:
        def __init__(self):
            self.sent = []

        async def finish(self, message=None):
            self.sent.append(message)
            raise FinishedException()

    def make_event(cls, text="你好", **fields):
        sub = type(cls.__name__, (cls,), {"get_message": lambda self: Message([MessageSegment.text(text)])})
        return sub.model_construct(**fields)

    def c2c(openid, text="你好"):
        return make_event(C2CMessageCreateEvent, text, author=FriendAuthor.model_construct(id="x", user_openid=openid))

    async def run(event):
        matcher = FakeMatcher()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                await mod.handle_chat(matcher, event)
                finished = False
            except FinishedException:
                finished = True
        return matcher, finished, out.getvalue()

    orig_process, orig_init = mod.chat_service.process_message, mod.chat_service.is_initialized
    mod.chat_service.process_message = fake_process
    mod.chat_service.is_initialized = lambda: True
    try:
        async def scenario():
            # 1. 普通好友：member，原文与 openid 都原样交给 process_message
            matcher, finished, _ = await run(c2c("MEMBERAAA0001", "帮我查天气"))
            assert finished and len(calls) == 1 and "机器人的回复" in str(matcher.sent[0]), (calls, matcher.sent)
            text, image_url, ident = calls[0]
            assert text == "帮我查天气" and image_url is None
            assert ident.user_id == "MEMBERAAA0001" and not ident.is_owner and ident.raw_id == "MEMBERAAA0001", ident

            # 2. 命中 OWNER_USER_IDS：归一为 owner（raw_id 保留平台 ID 供 /whoami 显示）
            with _OwnerIds({"OWNERQQ00000001"}):
                await run(c2c("OWNERQQ00000001"))
            ident = calls[-1][2]
            assert ident.user_id == "owner" and ident.is_owner and ident.raw_id == "OWNERQQ00000001", ident

            # 3. 其他事件：群 @、频道消息、频道私信一律忽略——不调 process_message、不回复；同类日志限频（每类最多一行）
            n_before = len(calls)
            mod._last_ignored_log.clear()
            total_out = ""
            for cls in (GroupAtMessageCreateEvent, DirectMessageCreateEvent, MessageCreateEvent):
                for _ in range(3):
                    matcher, finished, out = await run(make_event(cls))
                    assert not finished and matcher.sent == [], cls
                    total_out += out
            assert len(calls) == n_before, "非 C2C 事件绝不能进入 process_message"
            lines = [ln for ln in total_out.splitlines() if "忽略" in ln]
            assert len(lines) == 3, f"3 类事件各打一行（同类限频），实际 {lines}"
            assert all(name in total_out for name in ("GroupAtMessageCreateEvent", "DirectMessageCreateEvent", "MessageCreateEvent"))

            # 4. openid 格式不合法：忽略，不泄露原始值到日志（只记长度），同样限频
            mod._last_ignored_log.clear()
            for bad in ("abc-123", "short", "x" * 70, "owner", "含中文的openid1"):
                matcher, finished, out = await run(c2c(bad))
                assert not finished and matcher.sent == [] and bad not in out, (bad, out)
            assert len(calls) == n_before

            # 5. 这一轮被 /stop 中止（process_message 返回 None）：不再多发一条回复
            state["result"] = None
            matcher, finished, _ = await run(c2c("MEMBERAAA0001"))
            assert not finished and matcher.sent == [] and len(calls) == n_before + 1
            state["result"] = ChatReply("机器人的回复", [])

            # 6. 服务还没初始化：给固定提示，不进入 process_message
            mod.chat_service.is_initialized = lambda: False
            matcher, finished, _ = await run(c2c("MEMBERAAA0001"))
            assert finished and "尚未初始化" in str(matcher.sent[0]) and len(calls) == n_before + 1

        asyncio.run(scenario())
    finally:
        mod.chat_service.process_message, mod.chat_service.is_initialized = orig_process, orig_init


def test_24_qq_push_error_mapping():
    """8.8 plugins/chat.py 的 _qq_push：无 msg_id 的纯文本主动消息；错误码 / 异常分类；URL 被抹掉；日志里 openid 只显示前 8 位。"""
    import contextlib
    import io
    import nonebot
    from nonebot.adapters.qq import Bot as QQBot
    from nonebot.adapters.qq import exception as qq_exc
    from nonebot.drivers import Response
    from app.services.scheduler_service import PushError

    mod = _load_chat_plugin()
    openid = "OPENIDAAAA00001234"
    sent = []
    state = {"raises": None}

    class FakeQQBot(QQBot):
        async def send_to_c2c(self, openid, message, msg_id=None, msg_seq=None, event_id=None):
            sent.append({"openid": openid, "message": message, "msg_id": msg_id, "msg_seq": msg_seq, "event_id": event_id})
            if state["raises"] is not None:
                raise state["raises"]

    bot = FakeQQBot.__new__(FakeQQBot)
    orig_get_bots = nonebot.get_bots

    def failed(status, code=None, message="m", cls=qq_exc.ActionFailed):
        body = {} if code is None and message is None else {"code": code, "message": message}
        return cls(Response(status, content=json.dumps(body).encode("utf-8")))

    async def push(text="你好"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            await mod._qq_push(openid, text)
        return buf.getvalue()

    async def push_error(raises):
        state["raises"] = raises
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                await mod._qq_push(openid, "你好")
        except PushError as e:
            return e, buf.getvalue()
        finally:
            state["raises"] = None
        raise AssertionError("应当抛出 PushError")

    async def scenario():
        nonebot.get_bots = lambda: {"123": bot}
        # 成功：主动消息不带 msg_id / msg_seq / event_id；文本里的链接被抹掉（主动消息含链接会被 40054010 拒收）
        await push("详情见 https://example.com/a 和 www.example.org 这里")
        assert len(sent) == 1 and sent[0]["openid"] == openid
        assert sent[0]["msg_id"] is None and sent[0]["msg_seq"] is None and sent[0]["event_id"] is None
        assert "https://" not in sent[0]["message"] and "www." not in sent[0]["message"] and "（链接已省略）" in sent[0]["message"], sent[0]
        # 错误码表
        table = {
            40034105: "permanent", 40054013: "permanent",
            40034100: "transient", 50055001: "transient", 40054016: "transient",
            40054005: "dedupe", 40054007: "too_long", 40054010: "url_rejected",
            12345678: "transient",
        }
        assert {k: v for k, v in mod._PUSH_CODE_KINDS.items()} == {k: v for k, v in table.items() if k != 12345678}
        for code, kind in table.items():
            err, _ = await push_error(failed(403, code, "msg"))
            assert (err.kind, err.code) == (kind, code), (code, err)
        err, _ = await push_error(failed(500, None, None))                        # 没有响应体
        assert err.kind == "transient" and err.code is None
        # 其余异常
        err, _ = await push_error(failed(429, 40034100, "limited", cls=qq_exc.RateLimitException))
        assert (err.kind, err.code) == ("transient", 40034100)
        err, _ = await push_error(failed(401, 11255, "bad token", cls=qq_exc.UnauthorizedException))
        assert err.kind == "transient"
        assert (await push_error(qq_exc.NetworkError("down")))[0].kind == "transient"
        assert (await push_error(qq_exc.ApiNotAvailable()))[0].kind == "transient"
        assert (await push_error(RuntimeError("意外")))[0].kind == "transient"
        # 平台已受理的两种情况：不算失败（进入审核 / HTTP 201、202）
        state["raises"] = qq_exc.AuditException("audit-1")
        await push()
        state["raises"] = failed(202, None, None)
        await push()
        state["raises"] = failed(201, None, None)
        await push()
        state["raises"] = None
        # 日志：openid 只显示前 8 位，message 只留前 100 字，不带消息正文
        secret_text = "绝密正文DEF"
        state["raises"] = failed(403, 40034105, f"用户 {openid} 未授权 " + "长" * 300)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                await mod._qq_push(openid, secret_text)
        except PushError as e:
            err = e
        state["raises"] = None
        log = buf.getvalue()
        assert openid not in log and openid[:8] in log and secret_text not in log, log
        assert "status=403" in log and "code=40034105" in log
        assert len(err.detail) <= 200 and openid not in err.detail and err.kind == "permanent"
        # 没有 QQ 机器人在线 / 只有别的 Bot
        nonebot.get_bots = lambda: {}
        e = None
        try:
            await mod._qq_push(openid, "x")
        except PushError as ex:
            e = ex
        assert e is not None and e.kind == "transient" and "offline" in e.detail
        nonebot.get_bots = lambda: {"9": object()}
        try:
            await mod._qq_push(openid, "x")
            raise AssertionError("应当抛出 PushError")
        except PushError as ex:
            assert ex.kind == "transient"

    try:
        asyncio.run(scenario())
    finally:
        nonebot.get_bots = orig_get_bots


def test_25_plugin_scheduler_hooks():
    """8.9 调度器在 init_system 里于 init_service 成功之后启动，在 shutdown_system 里先于 shutdown_service 停止；
    开关关闭、初始化失败、启动 / 停止抛异常时都不影响主流程。"""
    import contextlib
    import io
    from app import config as cfg

    mod = _load_chat_plugin()
    cs, ss = mod.chat_service, mod.scheduler_service
    calls = []
    state = {"init_ok": True, "start_raises": False, "stop_raises": False}
    saved = (cs.init_service, cs.is_initialized, cs.shutdown_service, ss.start, ss.stop, cfg.SCHEDULER_ENABLED)

    async def fake_init():
        calls.append("init")
        if not state["init_ok"]:
            raise RuntimeError("初始化失败")

    async def fake_shutdown():
        calls.append("shutdown_service")

    def fake_start(**kwargs):
        calls.append(("start", kwargs))
        if state["start_raises"]:
            raise RuntimeError("起不来")

    async def fake_stop():
        calls.append("stop")
        if state["stop_raises"]:
            raise RuntimeError("停不下")

    def run(coro_fn):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):     # 插件会 print_exc，别刷屏
            asyncio.run(coro_fn())
        return buf.getvalue()

    try:
        cs.init_service, cs.shutdown_service = fake_init, fake_shutdown
        cs.is_initialized = lambda: state["init_ok"]
        ss.start, ss.stop = fake_start, fake_stop
        # 1. 开关打开：init_service 之后启动（run_turn / push / cli 三个参数），关闭时先停调度器再关服务
        cfg.SCHEDULER_ENABLED = True
        run(mod.init_system)
        assert calls[0] == "init" and calls[1][0] == "start", calls
        kwargs = calls[1][1]
        assert kwargs["run_turn"] is cs.run_scheduled_turn and kwargs["push"] is mod._qq_push and kwargs["cli"] is False, kwargs
        calls.clear()
        run(mod.shutdown_system)
        assert calls == ["stop", "shutdown_service"], calls
        # 2. 开关关闭：不启动
        calls.clear()
        cfg.SCHEDULER_ENABLED = False
        run(mod.init_system)
        assert calls == ["init"], calls
        # 3. init_service 失败：不启动调度器，也不抛
        calls.clear()
        cfg.SCHEDULER_ENABLED = True
        state["init_ok"] = False
        out = run(mod.init_system)
        assert calls == ["init"] and "初始化失败" in out, (calls, out)
        state["init_ok"] = True
        # 4. 调度器启动 / 停止抛异常：只打印，不影响主流程；停止失败后仍然关闭服务
        calls.clear()
        state["start_raises"] = state["stop_raises"] = True
        out = run(mod.init_system)
        assert "调度器启动失败" in out
        calls.clear()
        out = run(mod.shutdown_system)
        assert calls == ["stop", "shutdown_service"] and "调度器停止失败" in out, (calls, out)
    finally:
        cs.init_service, cs.is_initialized, cs.shutdown_service, ss.start, ss.stop, cfg.SCHEDULER_ENABLED = saved


SCENARIOS = [
    ("1.1 resolve_platform_user 格式与 owner 映射", test_1_resolve_platform_user),
    ("1.2 平台 id 无法冒充 owner / cli-", test_2_no_impersonation),
    ("1.3 validate_user_id 白名单", test_3_validate_user_id),
    ("1.4 owner_identity / cli_member_identity", test_4_cli_identities),
    ("1.5 请求上下文与 require_*（失败即关闭）", test_5_request_context),
    ("1.6 并发 task 的上下文互不串", test_6_context_isolated_between_tasks),
    ("1.7 配置解析（OWNER_USER_IDS / 并发上限 / 欢迎语）", test_7_config_parsing),
    ("1.8 OWNER_USER_IDS 非法项检测", test_8_invalid_owner_ids),
    ("2.1 v1 库迁移到 v2（数据归 owner、指针迁移、幂等）", test_9_migrate_v1_database),
    ("2.2 迁移边角（新旧键并存 / 无指针 / 全新库同路径 / 版本过高）", test_10_migrate_edge_cases),
    ("2.3 迁移中途失败整体回滚", test_11_migration_is_atomic),
    ("2.4 用户隔离（会话 / 指针 / 前缀 / 搜索 / 非法 id）", test_12_user_isolation),
    ("2.5 users 表（touch / 备注 / 封禁 / 解析）", test_13_users_table),
    ("2.6 静态检查：INSERT INTO sessions 带 user_id、无 owner 默认值", test_14_static_session_insert_is_explicit),
    ("3.1 回溯工具只读自己的会话", test_15_session_tools_isolation),
    ("7.1 请求上下文穿过 task 子专家 / 线程池 / to_thread，并发互不串", test_16_context_propagation_concurrent),
    ("8.1 run_host_command 仅 owner（含线程池路径）", test_17_run_host_command_requires_owner),
    ("8.2 convert_to_markdown 仅 owner", test_18_markitdown_requires_owner),
    ("8.3 execute_complex_browser_action 仅 owner", test_19_browser_action_requires_owner),
    ("8.4 DesktopAgent da_tools_node 仅 owner", test_20_desktop_node_requires_owner),
    ("8.5 守卫位置正确且不在 host_shell 里", test_21_guard_call_sites_and_layer),
    ("8.6 LLM 指标记录带 user 字段", test_22_metrics_record_carries_user),
    ("8.7 QQ 插件只处理 C2C，身份来自 openid", test_23_qq_plugin_c2c_only_and_identity),
    ("8.8 _qq_push：主动消息、错误码 / 异常分类、日志卫生", test_24_qq_push_error_mapping),
    ("8.9 插件里调度器的启动 / 停止顺序与容错", test_25_plugin_scheduler_hooks),
]


def main():
    print("=" * 70)
    print("开始运行多用户改造（阶段 1）离线冒烟测试...")
    print("=" * 70)
    passed = failed = 0
    for name, fn in SCENARIOS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e!r}")
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
