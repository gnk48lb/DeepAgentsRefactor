"""
tests/smoke_chat_service.py

chat_service 服务层离线冒烟测试：用脚本化假模型驱动真实的 FileAgent / hitl / outbox / chat_service，
并且所有场景都使用生产同款的 AsyncSqliteSaver（checkpoint）+ SessionStore（会话 / 事件日志 / 挂起态）。
覆盖：
  A–G  多轮 HITL、发件箱连续性、并发忙提示、并行中断一次 Y 放行、超时后会话可继续、/reset、图片轮不提取记忆
  H    重启恢复：挂起授权 -> 关闭 saver/store -> 用同一批文件重建 -> 回复 Y 照常放行
  I    多会话：/new /sessions /switch，旧会话授权保持挂起，上下文互不串，只有 /reset 重置沙箱
  J    崩溃遗留：未处理的入站事件被标记放弃，不自动重跑
  K    事件日志：user_message / expert_report / assistant_message，镜像幂等，不含 <long_term_memory>
  L    嵌套 checkpoint 清理：只清嵌套命名空间，根命名空间保留，开关可关
  M    /reset 清上下文但保留事件日志
  N    wake 幂等消费
  O    /stop（锁外中止当前轮）：O1 慢模型 O2 子专家工具（含悬空 tool_calls 探针）O3 空闲 O4 忙提示与忙时 /reset
       O5 授权恢复阶段被中止（P2 探针 + awaiting_approval 不变量）O6 终止原语被触发且在 cancel 之后
不需要任何 API Key / Docker / MCP / 网络：orchestrator / injector / extractor / startup 在 import 前被桩掉。
"""
import asyncio
import contextlib
import io
import json
import operator
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from deepagents import (
    create_deep_agent,
    CompiledSubAgent,
    register_harness_profile,
    HarnessProfile,
    GeneralPurposeSubagentProfile,
)

# ── 桩模块：必须在 import app.services.chat_service 之前安装，避免 import 时需要 API Key / Milvus ──
import app
import app.agents
import app.memory
import app.services


def _stub(pkg, name, **attrs):
    m = types.ModuleType(f"{pkg.__name__}.{name}")
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[m.__name__] = m
    setattr(pkg, name, m)
    return m


mem_calls = []


async def _fake_extract_memory(user_text, *, user_id):
    mem_calls.append(user_text)


_stub(app.agents, "orchestrator", build_main_agents=None)
_stub(
    app.memory,
    "injector",
    build_initial_messages=lambda user_query, history_messages=None, *, user_id: [HumanMessage(content=user_query, name="user")],
)
_extractor_stub = _stub(
    app.memory,
    "extractor",
    extract_and_store_memory=_fake_extract_memory,
    mem_calls=mem_calls,
)
_stub(app.services, "startup", run_global_database_init=lambda: None)

from app import config
from app.services import chat_service as cs
from app.services import session_log
from app.agents.experts import file_agent
from app.infra import checkpointer, modal_sandbox, sandbox, session_store
from app.infra.session_store import SessionStore
from app.runtime import hitl, outbox
from app.utils.context import get_request_context
from app.utils.identity import owner_identity, resolve_platform_user

# 沙箱重置计数：/new /switch 不得触发，只有 /reset 触发（同时避免测试里碰 Docker）
reset_calls = []
sandbox.reset_sandbox = lambda: reset_calls.append(1)

OWNER = "owner"
ID = owner_identity()                       # 单用户场景（A–O）统一以 owner 身份发消息
MEMBER_A = resolve_platform_user("MEMBERAAA0001")
MEMBER_B = resolve_platform_user("MEMBERBBB0002")


def _sid(user_id: str = OWNER) -> str:
    """该用户当前的活动会话 id（chat_service.get_active_session_id 必须带 user_id）。"""
    return cs.get_active_session_id(user_id=user_id)


class ScriptedModel(BaseChatModel):
    """按脚本依次返回 AIMessage；delay 用来模拟慢模型；seen 记录每次调用收到的输入文本。"""
    script: List[AIMessage]
    idx: int = 0
    delay: float = 0.0
    seen: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append("\n".join(str(getattr(m, "content", "")) for m in messages))
        if self.delay:
            time.sleep(self.delay)
        msg = self.script[min(self.idx, len(self.script) - 1)]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


# 与生产一致的全局排除 profile（key = 类名小写）
register_harness_profile(
    "scriptedmodel",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset({"write_file", "edit_file", "delete", "execute", "ls", "glob", "grep"}),
    ),
)


def _tc(name, args, id_):
    return {"name": name, "args": args, "id": id_}


def _task(sub, id_):
    return _tc("task", {"subagent_type": sub, "description": "d"}, id_)


def _write(path, id_):
    return _tc("write_file", {"file_path": path, "content": "x"}, id_)


FA_DONE = AIMessage(content="【内部专家汇报 - 来源：FileAgent】完成")


class _S(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


def _desktop_like():
    """DesktopAgent 替身：手写 interrupt 字符串。"""
    def node(state):
        d = interrupt("🛡️ 桌面高危操作，请回复 Y/N")
        return {"messages": [AIMessage(content=f"desktop 决定={d}")]}
    g = StateGraph(_S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile()


# ── 测试环境：临时目录里的真实 AsyncSqliteSaver + SessionStore ──────────────────────────

class _Env:
    def __init__(self):
        self.base = tempfile.mkdtemp(prefix="chat_")
        self.root = os.path.join(self.base, "ws")                  # FileAgent 的 workspace
        os.makedirs(self.root)
        self.cp_path = os.path.join(self.base, "db", "checkpoints.db")
        self.db_path = os.path.join(self.base, "db", "sessions.db")
        self.saver = None
        self.store = None


_env: Optional[_Env] = None


async def _open_stack(env: _Env):
    env.saver = await checkpointer.open_checkpointer(env.cp_path)
    env.store = SessionStore(env.db_path)
    session_store.set_store(env.store)


def _build(env: _Env, main_script, fa_script, extra=None, delay=0.0):
    fam = ScriptedModel(script=fa_script)
    mm = ScriptedModel(script=main_script, delay=delay)
    subs = [CompiledSubAgent(name="FileAgent", description="f",
                             runnable=file_agent.build_file_agent_runnable(fam, "x", env.root))]
    subs += extra or []
    agent = create_deep_agent(model=mm, subagents=subs, checkpointer=env.saver)
    return agent, fam, mm


async def _install(main_script, fa_script, extra=None, delay=0.0):
    """装配：真实 FileAgent + 脚本化主模型 + 真实 AsyncSqliteSaver / SessionStore，并把它塞进 chat_service。"""
    global _env
    _env = _Env()
    await _open_stack(_env)
    agent, fam, mm = _build(_env, main_script, fa_script, extra, delay)
    # 单图场景：owner 与 member 共用同一张图（只有多用户场景才会给 member 装一张不同的图，见 _install_two_graphs）
    cs._agents = {"owner": agent, "member": agent}
    cs._initialized, cs._checkpointer = True, _env.saver
    mem_calls.clear()
    reset_calls.clear()
    return _env.root, fam, mm


async def _restart(main_script, fa_script, extra=None):
    """模拟进程重启：关闭 saver 与 store，用同一批文件重建 saver / store / agent（全新的脚本化模型）。"""
    await checkpointer.close_checkpointer(_env.saver)
    session_store.close_store()
    await _open_stack(_env)
    agent, fam, mm = _build(_env, main_script, fa_script, extra)
    cs._agents = {"owner": agent, "member": agent}
    cs._checkpointer = _env.saver
    return fam, mm


async def _teardown():
    """先关连接（含 -wal / -shm），再删目录（Windows 要求）。"""
    global _env
    cs._agents, cs._initialized, cs._checkpointer = {}, False, None
    cs._reset_runtime_state()                  # 清掉每用户的锁 / 信号量 / 当前轮句柄：它们不能跨 asyncio.run 复用
    if _env is not None:
        await checkpointer.close_checkpointer(_env.saver)
        session_store.close_store()
        if _env.store is not None:
            _env.store.close()
        shutil.rmtree(_env.base, ignore_errors=True)
        _env = None


def _files(root):
    return sorted(p.name for p in Path(root).rglob("*") if p.is_file())


async def _cp_rows(sid, nested: bool, table: str = "checkpoints") -> int:
    op = "!=" if nested else "="
    async with cs._checkpointer.lock:
        cur = await cs._checkpointer.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE thread_id = ? AND checkpoint_ns {op} ''", (sid,)
        )
        row = await cur.fetchone()
        await cur.close()
    return row[0]


# ══════════════════════════════════════════════════════════════════════════════

async def test_a_multi_turn_hitl():
    """A 多轮：普通消息 → 写文件挂起 → 无关消息只重复提示 → 小写 y 批准 → 再次挂起 → N 拒绝。"""
    root, fam, mm = await _install(
        [AIMessage(content="你好呀"), AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]),
         AIMessage(content="写好了"), AIMessage(content="", tool_calls=[_task("FileAgent", "t2")]),
         AIMessage(content="已取消")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE,
         AIMessage(content="", tool_calls=[_write("workspace/b.txt", "c2")]), FA_DONE],
    )
    r = await cs.process_message("你好", identity=ID)
    await asyncio.sleep(0.05)
    assert r.text == "你好呀" and not r.awaiting_approval, r
    assert mem_calls and mem_calls[0] == "你好", mem_calls

    mem_calls.clear()
    r = await cs.process_message("写个文件", identity=ID)
    assert r.awaiting_approval and "write_file" in r.text and "workspace/a.txt" in r.text and r.images == [], r.text

    idx_before = fam.idx
    r = await cs.process_message("等等我想想", identity=ID)
    assert r.awaiting_approval and "当前正在等待授权确认" in r.text and fam.idx == idx_before, r.text

    r = await cs.process_message("y", identity=ID)
    await asyncio.sleep(0.05)
    assert (not r.awaiting_approval) and r.text == "写好了" and _files(root) == ["a.txt"], (r.text, _files(root))
    assert mem_calls == ["写个文件"], f"HITL 流程结束后只记录一次最初原话，实际: {mem_calls}"

    r = await cs.process_message("再写一个", identity=ID)
    assert r.awaiting_approval and "workspace/b.txt" in r.text, r.text

    r = await cs.process_message("N", identity=ID)
    assert _files(root) == ["a.txt"] and r.text == "已取消" and fam.idx == 4, (_files(root), r.text, fam.idx)
    sid = _sid()
    assert not cs._user_lock(OWNER).locked() and _env.store.load_turn_state(sid) is None


async def test_b_outbox_continuity():
    """B HITL 前已产生的图片，批准后仍随最终回复发出（不丢、不重复）。"""
    @tool
    def make_image() -> str:
        """make"""
        outbox.push({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})
        return "ok"

    cam = ScriptedModel(script=[AIMessage(content="", tool_calls=[_tc("make_image", {}, "m1")]),
                                AIMessage(content="画好了")])
    code_spec = {"name": "CodeAgent", "description": "c", "system_prompt": "x", "model": cam, "tools": [make_image]}
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("CodeAgent", "t1"), _task("FileAgent", "t2")]),
         AIMessage(content="图和文件都好了")],
        [AIMessage(content="", tool_calls=[_write("workspace/p.txt", "c1")]), FA_DONE],
        extra=[code_spec],
    )
    r1 = await cs.process_message("画图并写文件", identity=ID)
    ts = _env.store.load_turn_state(_sid())
    assert r1.awaiting_approval and ts is not None and len(ts.outbox) == 1, r1
    r2 = await cs.process_message("Y", identity=ID)
    assert len(r2.images) == 1 and r2.text == "图和文件都好了" and cam.idx == 2, (len(r2.images), r2.text, cam.idx)
    # 图片字节永远不进事件日志，只记张数
    evs = _env.store.get_events(_sid(), limit=200)
    assert "base64" not in json.dumps([e.payload for e in evs], ensure_ascii=False)
    assert [e.payload["image_count"] for e in evs if e.type == "assistant_message"] == [1]


async def test_c_busy_lock():
    """C 处理中再来一条 → 立即返回忙提示，第一条不受影响。"""
    root, fam, mm = await _install([AIMessage(content="慢回复")], [FA_DONE], delay=0.6)
    t = asyncio.create_task(cs.process_message("慢", identity=ID))
    await asyncio.sleep(0.2)
    rb = await cs.process_message("插队", identity=ID)
    ra = await t
    assert "还在处理中" in rb.text and ra.text == "慢回复" and not cs._user_lock(OWNER).locked(), (rb.text, ra.text)


async def test_d_parallel_interrupts():
    """D 主 Agent 同时派 FileAgent + DesktopAgent：两类授权一起提示，一次 Y 同时放行。"""
    desk = CompiledSubAgent(name="DesktopAgent", description="d", runnable=_desktop_like())
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("FileAgent", "t1"), _task("DesktopAgent", "t2")]),
         AIMessage(content="两件事都办了")],
        [AIMessage(content="", tool_calls=[_write("workspace/q.txt", "c1")]), FA_DONE],
        extra=[desk],
    )
    r = await cs.process_message("同时办两件事", identity=ID)
    assert r.awaiting_approval and "write_file" in r.text and "桌面高危操作" in r.text, r.text
    r = await cs.process_message("Y", identity=ID)
    assert r.text == "两件事都办了" and _files(root) == ["q.txt"], (r.text, _files(root))


async def test_e_timeout_recovery():
    """E 单轮超时后给出友好提示，且当前会话仍可继续对话；事件日志里有 turn_failed(timeout)。"""
    root, fam, mm = await _install([AIMessage(content="很慢"), AIMessage(content="恢复了")], [FA_DONE], delay=1.2)
    old = config.AGENT_TURN_TIMEOUT_SECONDS
    config.AGENT_TURN_TIMEOUT_SECONDS = 0.3
    try:
        r = await cs.process_message("会超时的问题", identity=ID)
    finally:
        config.AGENT_TURN_TIMEOUT_SECONDS = old
    mm.delay = 0.0
    await asyncio.sleep(1.3)  # 等后台线程里被取消的那次模型调用自然结束
    sid = _sid()
    failed = _env.store.get_events(sid, types=["turn_failed"])
    assert [e.payload["reason"] for e in failed] == ["timeout"], failed
    assert _env.store.get_pending_events(sid) == [], "失败的轮也必须推进游标"
    r2 = await cs.process_message("超时之后的新问题", identity=ID)
    assert "我先停下了" in r.text, r.text
    assert r2.text in ("恢复了", "很慢"), r2.text


async def test_f_reset_command():
    """F /reset 丢弃挂起的授权与历史，之后是全新对话。"""
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]), AIMessage(content="新会话回复")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE],
    )
    r = await cs.process_message("写个文件", identity=ID)
    assert r.awaiting_approval, r
    r = await cs.process_message("/reset", identity=ID)
    assert "已清空" in r.text, r.text
    st = await cs._agents[OWNER].aget_state({"configurable": {"thread_id": _sid()}})
    assert not hitl.has_pending_interrupts(st) and not st.values.get("messages"), st
    r = await cs.process_message("你好", identity=ID)
    assert r.text == "新会话回复" and _files(root) == [] and fam.idx == 1, (r.text, _files(root), fam.idx)


async def test_g_image_turn_no_memory():
    """G 图片轮不触发记忆提取。"""
    from app.infra import models
    models._cache["vlm"] = ScriptedModel(script=[AIMessage(content="分析图里的苹果")])
    root, fam, mm = await _install([AIMessage(content="看图回复")], [FA_DONE])
    mem_calls.clear()
    r = await cs.process_message("看这个图", image_url="data:image/png;base64,AAAA", identity=ID)
    await asyncio.sleep(0.05)
    assert r.text == "看图回复"
    assert len(mem_calls) == 0, f"图片轮不应触发记忆提取，实际: {mem_calls}"
    evs = _env.store.get_events(_sid(), types=["user_message"])
    assert evs[0].payload["has_image"] is True and evs[0].payload["processed_text"] == "分析图里的苹果"
    assert "base64" not in json.dumps(evs[0].payload), "图片字节不得进事件日志"


async def test_h_restart_recovery():
    """H 重启恢复：挂起 HITL（含发件箱图片）→ 关闭 saver/store → 同一批文件重建 → Y 放行。"""
    @tool
    def make_image() -> str:
        """make"""
        outbox.push({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})
        return "ok"

    def code_spec(model):
        return {"name": "CodeAgent", "description": "c", "system_prompt": "x", "model": model, "tools": [make_image]}

    cam = ScriptedModel(script=[AIMessage(content="", tool_calls=[_tc("make_image", {}, "m1")]),
                                AIMessage(content="画好了")])
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("CodeAgent", "t1"), _task("FileAgent", "t2")]),
         AIMessage(content="图和文件都好了")],
        [AIMessage(content="", tool_calls=[_write("workspace/p.txt", "c1")]), FA_DONE],
        extra=[code_spec(cam)],
    )
    sid = _sid()
    r1 = await cs.process_message("画图并写文件", identity=ID)
    assert r1.awaiting_approval and cam.idx == 2
    assert await _cp_rows(sid, nested=True) > 0, "挂起期间嵌套 checkpoint 必须保留，否则无法恢复"

    # ── 重启：全新的模型实例，脚本里只有“批准之后”才会用到的回复；若已完成的专家被重跑会立刻暴露 ──
    cam2 = ScriptedModel(script=[AIMessage(content="不应被调用")])
    fam2, mm2 = await _restart([AIMessage(content="图和文件都好了")], [FA_DONE], extra=[code_spec(cam2)])
    assert _sid() == sid
    abandoned = await cs.recover_orphans()
    assert abandoned == 0, "挂起中的授权不受 recover_orphans 影响"
    ts = _env.store.load_turn_state(sid)
    assert ts is not None and len(ts.outbox) == 1 and ts.user_text == "画图并写文件", ts

    mem_calls.clear()
    r2 = await cs.process_message("y", identity=ID)
    await asyncio.sleep(0.05)
    assert r2.text == "图和文件都好了" and not r2.awaiting_approval, r2
    assert _files(root) == ["p.txt"], _files(root)
    assert len(r2.images) == 1, "重启前已生成的图片仍随回复发出"
    assert mem_calls == ["画图并写文件"], f"记忆提取应收到最初原话而不是 'y': {mem_calls}"
    assert cam2.idx == 0, "已完成的专家不应被重跑"
    assert fam2.idx == 1 and mm2.idx == 1, (fam2.idx, mm2.idx)
    types_ = [e.type for e in _env.store.get_events(sid, limit=200)]
    assert "approval_request" in types_ and "approval" in types_ and types_[-1] == "assistant_message", types_


async def test_i_multi_session():
    """I 多会话：/new 保留旧会话授权；/sessions 标记；/switch 序号与前缀；上下文互不串；只有 /reset 重置沙箱。"""
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]),
         AIMessage(content="会话二回复"),
         AIMessage(content="写好了")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE],
    )
    sid1 = _sid()
    r = await cs.process_message("写个文件", identity=ID)
    assert r.awaiting_approval

    r = await cs.process_message("/new 第二个会话", identity=ID)
    sid2 = _sid()
    assert sid2 != sid1 and sid2[:8] in r.text and "第二个会话" in r.text, r.text
    assert sid1[:8] in r.text and "待授权" in r.text, "旧会话还有待授权操作，应附提醒"
    assert not r.awaiting_approval and reset_calls == [], "/new 不动沙箱"

    r = await cs.process_message("/sessions", identity=ID)
    lines = [ln for ln in r.text.splitlines() if ln[:2] in ("1.", "2.")]
    assert len(lines) == 2, r.text
    assert "[当前]" in lines[0] and sid2[:8] in lines[0], lines
    assert "[待授权]" in lines[1] and sid1[:8] in lines[1] and "[当前]" not in lines[1], lines
    assert "/switch" in r.text and "/new" in r.text

    r = await cs.process_message("第二会话专属消息", identity=ID)
    assert r.text == "会话二回复" and not r.awaiting_approval, r.text
    assert "写个文件" not in mm.seen[1], "第二个会话的模型输入里不应有第一个会话的内容"

    r = await cs.process_message("/switch 2", identity=ID)                          # 按 /sessions 的序号
    assert _sid() == sid1 and r.awaiting_approval, r.text
    assert "write_file" in r.text and "workspace/a.txt" in r.text, "切到有待授权的会话时附上授权内容"

    r = await cs.process_message("Y", identity=ID)
    assert r.text == "写好了" and _files(root) == ["a.txt"], (r.text, _files(root))
    assert "第二会话专属消息" not in mm.seen[2], "第一个会话的模型输入里不应有第二个会话的内容"

    r = await cs.process_message(f"/switch {sid2[:8]}", identity=ID)                # 按 id 前缀
    assert _sid() == sid2 and not r.awaiting_approval, r.text
    r = await cs.process_message(f"/switch {sid2[:8]}", identity=ID)
    assert "已经在" in r.text and _sid() == sid2
    r = await cs.process_message("/switch zzzz", identity=ID)
    assert "没有找到" in r.text and _sid() == sid2
    r = await cs.process_message("/switch", identity=ID)
    assert "用法" in r.text
    assert reset_calls == [], "/new、/switch 不触发沙箱重置"

    await cs.process_message("/reset", identity=ID)
    assert reset_calls == [1], "只有 /reset 重置沙箱"

    # 命令按首个空白分隔的词精确匹配；其余以 / 开头的文本（如路径）按普通消息处理
    assert cs._parse_command("/tmp/a.txt 里有什么") is None
    assert cs._parse_command("/resetx") is None and cs._parse_command("你好 /reset") is None
    assert cs._parse_command("/NEW 标题") == ("/new", "标题")
    assert cs._parse_command("/sessions 5") == ("/sessions", "5")


async def test_j_crash_leftover():
    """J 崩溃遗留：未处理的入站事件 → recover_orphans 标记放弃，不调用模型；之后新消息正常处理。"""
    root, fam, mm = await _install([AIMessage(content="正常回复")], [FA_DONE])
    sid = _sid()
    store = _env.store
    seq = store.emit_event(sid, "user_message", {"text": "崩溃前的消息", "processed_text": "崩溃前的消息", "has_image": False})
    assert [e.seq for e in store.get_pending_events(sid)] == [seq]

    n = await cs.recover_orphans()
    assert n == 1 and store.get_pending_events(sid) == []
    ab = store.get_events(sid, types=["turn_abandoned"])
    assert len(ab) == 1 and ab[0].payload == {"reason": "process_restart", "event_seq": seq}, ab
    assert mm.idx == 0, "不自动重跑：模型不应被调用"

    r = await cs.process_message("新消息", identity=ID)
    assert r.text == "正常回复" and mm.idx == 1, (r.text, mm.idx)


async def test_k_event_log():
    """K 事件日志：user_message / expert_report / assistant_message；镜像幂等；日志里没有 <long_term_memory>。"""
    @tool
    def lookup() -> str:
        """lookup"""
        return "查到：数据X"

    cam = ScriptedModel(script=[AIMessage(content="", tool_calls=[_tc("lookup", {}, "l1")]),
                                AIMessage(content="专家结论：数据X")])
    spec = {"name": "CodeAgent", "description": "c", "system_prompt": "x", "model": cam, "tools": [lookup]}
    root, fam, mm = await _install(
        [AIMessage(content="普通回复"), AIMessage(content="", tool_calls=[_task("CodeAgent", "t1")]),
         AIMessage(content="最终回复")],
        [FA_DONE],
        extra=[spec],
    )
    orig = cs.injector.build_initial_messages
    cs.injector.build_initial_messages = lambda q, history_messages=None, *, user_id: [
        HumanMessage(content=f"<long_term_memory>\n我住台北\n</long_term_memory>\n{q}", name="user")
    ]
    try:
        r1 = await cs.process_message("你好", identity=ID)
        r2 = await cs.process_message("查个数据", identity=ID)
    finally:
        cs.injector.build_initial_messages = orig
    assert r1.text == "普通回复" and r2.text == "最终回复", (r1.text, r2.text)
    assert "long_term_memory" in mm.seen[0], "桩应当把记忆块真的拼进了模型输入（否则下面的断言没有意义）"

    sid = _sid()
    store = _env.store
    evs = [e for e in store.get_events(sid, limit=200) if e.type != "session_created"]
    assert [e.type for e in evs] == [
        "user_message", "assistant_message", "user_message", "expert_report", "assistant_message"
    ], [e.type for e in evs]
    assert evs[0].payload["text"] == "你好", "user_message.text 是用户原话"
    er = evs[3].payload
    assert er["expert"] == "CodeAgent" and er["task"] == "d" and "专家结论：数据X" in er["text"], er
    assert evs[4].payload == {"text": "最终回复", "image_count": 0}
    assert "long_term_memory" not in json.dumps([e.payload for e in evs], ensure_ascii=False)

    # 重复镜像不产生重复事件
    st = await cs._agents[OWNER].aget_state({"configurable": {"thread_id": sid}})
    before = store.count_events(sid)
    assert session_log.mirror_messages(store, sid, st.values["messages"]) == 0
    assert store.count_events(sid) == before


async def test_l_nested_prune():
    """L 嵌套 checkpoint 清理：挂起期间保留，正常结束后清空嵌套命名空间、保留根命名空间；开关可关。"""
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]), AIMessage(content="写好了"),
         AIMessage(content="", tool_calls=[_task("FileAgent", "t2")]), AIMessage(content="又写好了")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE,
         AIMessage(content="", tool_calls=[_write("workspace/b.txt", "c2")]), FA_DONE],
    )
    sid = _sid()
    r = await cs.process_message("写个文件", identity=ID)
    assert r.awaiting_approval
    assert await _cp_rows(sid, nested=True) > 0, "挂起期间嵌套 checkpoint 必须保留"
    root_before = await _cp_rows(sid, nested=False)

    r = await cs.process_message("Y", identity=ID)
    assert r.text == "写好了"
    assert await _cp_rows(sid, nested=True) == 0 and await _cp_rows(sid, nested=True, table="writes") == 0
    assert await _cp_rows(sid, nested=False) >= root_before > 0, "根命名空间必须保留"

    st = await cs._agents[OWNER].aget_state({"configurable": {"thread_id": sid}})
    humans = [m.content for m in st.values["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["写个文件"], humans
    # 下一轮仍能看到上一轮的消息
    old = config.CHECKPOINT_PRUNE_NESTED
    config.CHECKPOINT_PRUNE_NESTED = False
    try:
        r = await cs.process_message("再写一个", identity=ID)
        assert r.awaiting_approval
        r = await cs.process_message("Y", identity=ID)
        assert r.text == "又写好了", r.text
        assert await _cp_rows(sid, nested=True) > 0, "CHECKPOINT_PRUNE_NESTED=False 时不清理"
    finally:
        config.CHECKPOINT_PRUNE_NESTED = old
    st = await cs._agents[OWNER].aget_state({"configurable": {"thread_id": sid}})
    humans = [m.content for m in st.values["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["写个文件", "再写一个"], humans


async def test_m_reset_keeps_log():
    """M /reset 清上下文与挂起授权，但事件日志保留并追加 session_reset。"""
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]), AIMessage(content="新对话回复")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE],
    )
    sid = _sid()
    r = await cs.process_message("写个文件", identity=ID)
    assert r.awaiting_approval and _env.store.load_turn_state(sid) is not None

    r = await cs.process_message("/reset", identity=ID)
    assert "已清空" in r.text and "会话记录仍保留" in r.text, r.text
    assert _sid() == sid, "/reset 不换会话"
    st = await cs._agents[OWNER].aget_state({"configurable": {"thread_id": sid}})
    assert not hitl.has_pending_interrupts(st) and not st.values.get("messages"), st
    assert _env.store.load_turn_state(sid) is None
    assert await _cp_rows(sid, nested=False) == 0 and await _cp_rows(sid, nested=True) == 0

    types_ = [e.type for e in _env.store.get_events(sid, limit=200)]
    assert "user_message" in types_ and "approval_request" in types_ and types_[-1] == "session_reset", types_
    assert _env.store.get_pending_events(sid) == []

    r = await cs.process_message("你好", identity=ID)                                # 授权已被丢弃：这是一条全新的普通消息
    assert r.text == "新对话回复" and not r.awaiting_approval and _files(root) == [], (r.text, _files(root))


async def test_n_wake_idempotent():
    """N wake 幂等消费：已处理的事件不会被再次处理；直接 emit + wake 同样可用。"""
    root, fam, mm = await _install([AIMessage(content="回复一"), AIMessage(content="回复二")], [FA_DONE])
    sid = _sid()
    r = await cs.process_message("你好", identity=ID)
    assert r.text == "回复一"
    calls = mm.idx
    assert await cs.wake(sid) is None and await cs.wake(sid) is None
    assert mm.idx == calls, "对同一会话重复 wake 不应再次调用模型"

    _env.store.emit_event(sid, "user_message", {"text": "直接写事件", "processed_text": "直接写事件", "has_image": False})
    r = await cs.wake(sid)
    assert r is not None and r.text == "回复二" and mm.idx == calls + 1, (r, mm.idx)
    assert await cs.wake(sid) is None and mm.idx == calls + 1


async def _state(sid=None):
    return await cs._agents[OWNER].aget_state({"configurable": {"thread_id": sid or _sid()}})


async def _stop_after(delay: float):
    await asyncio.sleep(delay)
    t0 = time.monotonic()
    reply = await cs.process_message("/stop", identity=ID)
    return reply, time.monotonic() - t0


async def test_o1_stop_slow_model():
    """O1 慢模型调用中 /stop：/stop 立即返回，原消息得到 None，锁释放，turn_failed(user_stop)，之后会话可继续。"""
    root, fam, mm = await _install([AIMessage(content="慢回复"), AIMessage(content="之后的回复")], [FA_DONE], delay=1.5)
    sid = _sid()
    t0 = time.monotonic()
    turn = asyncio.create_task(cs.process_message("很慢的问题", identity=ID))
    (stop_reply, stop_cost) = await _stop_after(0.3)
    original = await turn
    total = time.monotonic() - t0

    assert "已中止" in stop_reply.text and not stop_reply.awaiting_approval, stop_reply
    assert stop_cost < 1.0 and total < 1.2, f"/stop 不应等模型：stop={stop_cost:.2f}s total={total:.2f}s"
    assert original is None, original
    assert not cs._user_lock(OWNER).locked() and cs._current_turns.get(OWNER) is None
    failed = _env.store.get_events(sid, types=["turn_failed"])
    assert [e.payload["reason"] for e in failed] == ["user_stop"], failed
    assert _env.store.get_pending_events(sid) == [], "被中止的轮也必须推进游标"
    assert _env.store.load_turn_state(sid) is None
    assert not [e for e in _env.store.get_events(sid, limit=200) if e.type == "assistant_message"]

    mm.delay = 0.0
    await asyncio.sleep(1.6)                # 让被放弃的线程自然结束（参考 test_e）
    r = await cs.process_message("中止之后的新问题", identity=ID)
    assert r is not None and r.text == "之后的回复" and not cs._user_lock(OWNER).locked(), r


async def test_o2_stop_in_subagent_tool():
    """O2 子专家工具执行中 /stop（P1 探针）：async 工具收到 CancelledError，同步线程不被等待，下一轮补 cancelled 的 ToolMessage。"""
    flags: dict = {}

    @tool
    async def hang() -> str:
        """hang"""
        flags["hang_started"] = True
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            flags["hang_cancelled"] = True
            raise
        return "不应到这里"

    @tool
    def slow_sync() -> str:
        """slow"""
        flags["sync_started"] = True
        time.sleep(2)
        flags["sync_done"] = True
        return "done"

    async def _wait_flag(name):
        deadline = time.monotonic() + 5
        while name not in flags and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert name in flags, f"{name} 未出现"

    for variant, tool_fn, started in (("async", hang, "hang_started"), ("sync", slow_sync, "sync_started")):
        flags.clear()
        cam = ScriptedModel(script=[AIMessage(content="", tool_calls=[_tc(tool_fn.name, {}, "x1")]),
                                    AIMessage(content="不应到这里")])
        spec = {"name": "CodeAgent", "description": "c", "system_prompt": "x", "model": cam, "tools": [tool_fn]}
        root, fam, mm = await _install(
            [AIMessage(content="", tool_calls=[_task("CodeAgent", "t1")]), AIMessage(content="继续好了")],
            [FA_DONE], extra=[spec],
        )
        sid = _sid()
        turn = asyncio.create_task(cs.process_message("让专家干活", identity=ID))
        await _wait_flag(started)
        stop_reply, stop_cost = await _stop_after(0.1)
        original = await turn
        assert "已中止" in stop_reply.text and stop_cost < 1.5, (variant, stop_reply.text, stop_cost)
        assert original is None and not cs._user_lock(OWNER).locked() and cs._current_turns.get(OWNER) is None, variant
        if variant == "async":
            assert flags.get("hang_cancelled"), "async 工具应收到 CancelledError"
        else:
            assert "sync_done" not in flags, "/stop 不应等同步线程跑完"
        failed = _env.store.get_events(sid, types=["turn_failed"])
        assert [e.payload["reason"] for e in failed] == ["user_stop"], (variant, failed)

        # P1：悬空的 task 调用在下一轮开头被补成 "...was cancelled..." 的 ToolMessage，会话能继续
        r = await cs.process_message("中止之后继续", identity=ID)
        assert r is not None and r.text == "继续好了", (variant, r)
        assert "cancelled" in mm.seen[-1], f"P1 不成立：下一轮主模型输入里没有 cancelled\n{mm.seen[-1][-400:]}"
        if variant == "sync":
            await asyncio.sleep(2.2)        # 让被放弃的线程结束再清理
        await _teardown()


async def test_o3_idle_stop():
    """O3 空闲 /stop：不取锁、不写事件；有待授权时回复里带 Y/N 提示。"""
    root, fam, mm = await _install(
        [AIMessage(content="普通回复"), AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]), AIMessage(content="写好了")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE],
    )
    sid = _sid()
    before = _env.store.count_events(sid)
    r = await cs.process_message("/stop", identity=ID)
    assert "没有正在运行" in r.text and not r.awaiting_approval, r
    assert _env.store.count_events(sid) == before, "空闲 /stop 不应新增事件"

    # 不取锁：锁被占着（比如正在执行命令）时 /stop 也立即返回
    async with cs._user_lock(OWNER):
        r = await asyncio.wait_for(cs.process_message("/stop", identity=ID), timeout=2)
        assert "没有正在运行" in r.text
    # 并发再发一条普通消息能正常处理
    stop_r, normal_r = await asyncio.gather(cs.process_message("/stop", identity=ID), cs.process_message("你好", identity=ID))
    assert "没有正在运行" in stop_r.text and normal_r.text == "普通回复", (stop_r, normal_r)

    r = await cs.process_message("写个文件", identity=ID)
    assert r.awaiting_approval
    before = _env.store.count_events(sid)
    r = await cs.process_message("/stop", identity=ID)
    assert "没有正在运行" in r.text and "Y" in r.text and "N" in r.text and r.awaiting_approval, r
    assert _env.store.count_events(sid) == before
    r = await cs.process_message("Y", identity=ID)
    assert r.text == "写好了" and _files(root) == ["a.txt"], (r.text, _files(root))


async def test_o4_busy_hint_and_reset_while_busy():
    """O4 忙提示带 /stop 提示；忙时 /reset 先中止当前轮再清空。"""
    root, fam, mm = await _install([AIMessage(content="慢回复"), AIMessage(content="全新对话")], [FA_DONE], delay=1.5)
    sid = _sid()
    turn = asyncio.create_task(cs.process_message("很慢的问题", identity=ID))
    await asyncio.sleep(0.3)
    busy = await cs.process_message("插队", identity=ID)
    assert "还在处理中" in busy.text and "/stop" in busy.text and "已运行" in busy.text, busy.text
    for cmd in ("/new x", "/switch 1", "/sessions"):
        r = await cs.process_message(cmd, identity=ID)
        assert "还在处理中" in r.text, (cmd, r.text)

    reset_reply = await cs.process_message("/reset", identity=ID)
    original = await turn
    assert original is None, original
    assert "已先中止" in reset_reply.text and "已清空" in reset_reply.text, reset_reply.text
    assert not cs._user_lock(OWNER).locked() and cs._current_turns.get(OWNER) is None
    st = await _state(sid)
    assert not st.values.get("messages") and not hitl.has_pending_interrupts(st), st
    types_ = [e.type for e in _env.store.get_events(sid, limit=200)]
    reasons = [e.payload["reason"] for e in _env.store.get_events(sid, types=["turn_failed"])]
    assert reasons == ["user_stop"] and "session_reset" in types_, (reasons, types_)

    mm.delay = 0.0
    await asyncio.sleep(1.6)
    r = await cs.process_message("你好", identity=ID)
    assert r.text == "全新对话", r
    assert "很慢的问题" not in mm.seen[-1], "/reset 之后应是全新对话"


async def test_o5_stop_during_approval_resume():
    """O5 授权恢复阶段被中止（P2 探针 + 不变量）：副作用不回滚；回复与实际授权状态一致；之后会话可继续。"""
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]), AIMessage(content="写好了"),
         AIMessage(content="继续好了")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE, FA_DONE],
    )
    sid = _sid()
    fam.delay = 0.0
    r = await cs.process_message("写个文件", identity=ID)
    assert r.awaiting_approval and _files(root) == [], r

    fam.delay = 1.5                          # 批准后 write_file 立即落盘，随后 FileAgent 的下一次模型调用很慢
    turn = asyncio.create_task(cs.process_message("Y", identity=ID))
    stop_reply, stop_cost = await _stop_after(0.4)
    original = await turn
    assert original is None and "已中止" in stop_reply.text, (original, stop_reply)
    assert _files(root) == ["a.txt"], "已落盘的文件不回滚"
    assert "不会回滚" in stop_reply.text

    st = await _state(sid)
    pending = hitl.has_pending_interrupts(st)
    print(f"    [P2 实测] 授权恢复阶段被中止后 has_pending_interrupts = {pending}")
    assert stop_reply.awaiting_approval == pending, (stop_reply.awaiting_approval, pending)
    if pending:
        assert "回复 Y 重试" in stop_reply.text and "write_file" in stop_reply.text, stop_reply.text
    assert not cs._user_lock(OWNER).locked() and cs._current_turns.get(OWNER) is None
    reasons = [e.payload["reason"] for e in _env.store.get_events(sid, types=["turn_failed"])]
    assert reasons == ["user_stop"], reasons

    fam.delay = 0.0
    await asyncio.sleep(1.6)
    if pending:
        r = await cs.process_message("N", identity=ID)
    else:
        r = await cs.process_message("再说点什么", identity=ID)
    assert r is not None and r.text in ("写好了", "继续好了"), r.text
    r2 = await cs.process_message("最后一条", identity=ID)
    assert r2 is not None and not cs._user_lock(OWNER).locked(), r2


async def test_o6_kill_primitives_triggered():
    """O6 终止原语被正确触发，且在 cancel 之后调用。"""
    from app.infra import host_shell
    root, fam, mm = await _install([AIMessage(content="慢回复"), AIMessage(content="后续")], [FA_DONE], delay=1.2)
    calls = {"host": [], "sandbox": []}

    def fake_kill():
        t = cs._current_turns.get(OWNER)
        calls["host"].append(bool(t is not None and t.stop_requested and t.task.cancelling() > 0))
        return 2

    def fake_interrupt():
        t = cs._current_turns.get(OWNER)
        calls["sandbox"].append(bool(t is not None and t.stop_requested))
        return False

    orig_kill, orig_int = host_shell.kill_all_running, sandbox.interrupt_sandbox
    host_shell.kill_all_running, sandbox.interrupt_sandbox = fake_kill, fake_interrupt
    try:
        turn = asyncio.create_task(cs.process_message("慢", identity=ID))
        stop_reply, _ = await _stop_after(0.3)
        await turn
        # 空闲时的 /stop 不应触发任何终止原语
        await cs.process_message("/stop", identity=ID)
    finally:
        host_shell.kill_all_running, sandbox.interrupt_sandbox = orig_kill, orig_int
    assert calls == {"host": [True], "sandbox": [True]}, calls
    assert "已终止 2 个" in stop_reply.text, stop_reply.text
    mm.delay = 0.0
    await asyncio.sleep(1.3)


# ══════════════════════════════════════════════════════════════════════════════
# 多用户：owner / member 各用各的图、各自的锁；按用户隔离的会话与命令；per-user /stop
# ══════════════════════════════════════════════════════════════════════════════

async def _install_two_graphs(owner_script, member_script, owner_delay=0.0, member_delay=0.0):
    """装配两张不同的图：owner 图（带 FileAgent）与 member 图（没有任何子专家），各用各的脚本化模型，共用同一个 saver / store。"""
    global _env
    _env = _Env()
    await _open_stack(_env)
    fam = ScriptedModel(script=[FA_DONE])
    omm = ScriptedModel(script=owner_script, delay=owner_delay)
    mmm = ScriptedModel(script=member_script, delay=member_delay)
    owner_agent = create_deep_agent(
        model=omm,
        subagents=[CompiledSubAgent(name="FileAgent", description="f",
                                    runnable=file_agent.build_file_agent_runnable(fam, "x", _env.root))],
        checkpointer=_env.saver,
    )
    member_agent = create_deep_agent(model=mmm, subagents=[], checkpointer=_env.saver)
    cs._agents = {"owner": owner_agent, "member": member_agent}
    cs._initialized, cs._checkpointer = True, _env.saver
    mem_calls.clear()
    reset_calls.clear()
    return omm, mmm


def _say(identity, text, **kw):
    return cs.process_message(text, identity=identity, **kw)


def _start(identity, text):
    return asyncio.create_task(cs.process_message(text, identity=identity))


async def test_p1_users_do_not_block_each_other():
    """P1 A 的慢回合不阻塞 B（owner 与 member）；各自只用自己那张图；member 首条回复带欢迎语。"""
    omm, mmm = await _install_two_graphs([AIMessage(content="owner慢回复")], [AIMessage(content="member快回复")], owner_delay=2.0)
    slow = _start(ID, "owner 的慢问题")
    await asyncio.sleep(0.25)
    r = await _say(MEMBER_A, "member 的问题")
    assert r.text.endswith("member快回复"), r.text
    assert not slow.done(), "member 的回复先于 owner 的慢回合返回：没有被 owner 的锁挡住，也没有打断 owner"
    ro = await slow
    assert ro.text == "owner慢回复" and not ro.text.startswith(config.MEMBER_WELCOME_TEXT.strip()[:6]), ro.text
    assert omm.idx == 1 and mmm.idx == 1, "每个用户只用自己的图"
    assert _sid(OWNER) != _sid(MEMBER_A.user_id)
    await asyncio.sleep(0.2)


async def test_p2_same_user_busy_other_users_unaffected():
    """P2 同一用户连发：第二条得到忙提示（带 /stop 用法）；其他用户（owner / 另一个 member）不受这把锁影响。"""
    omm, mmm = await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")], member_delay=2.0)
    slow = _start(MEMBER_A, "A 的慢问题")
    await asyncio.sleep(0.25)
    busy = await _say(MEMBER_A, "A 又发一条")
    assert "还在处理中" in busy.text and "/stop" in busy.text and "已运行" in busy.text, busy.text
    ro = await _say(ID, "owner 问题")                             # owner 走自己的图和自己的锁：不等 A
    assert ro.text == "o" and not slow.done(), ro.text
    rb = await _say(MEMBER_B, "B 的问题")                         # B 与 A 共用同一个慢模型，但没有被 A 的锁挡住（返回的是回复，不是忙提示）
    assert "还在处理中" not in rb.text and rb.text.endswith("m"), rb.text
    ra = await slow
    assert ra.text.endswith("m") and not cs._user_lock(MEMBER_A.user_id).locked()
    await asyncio.sleep(0.2)


async def test_p3_sessions_isolated_between_users():
    """P3 各用户有各自的活动会话；/sessions 只列自己的；/switch 无法通过序号、前缀或完整 id 到别人的会话，且回复与“不存在”无法区分。"""
    await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")])
    await _say(ID, "owner 你好")
    await _say(MEMBER_A, "A 的第一句")
    await _say(MEMBER_B, "B 的第一句")
    sa0, sb0, so0 = _sid(MEMBER_A.user_id), _sid(MEMBER_B.user_id), _sid(OWNER)
    assert len({sa0, sb0, so0}) == 3

    await _say(MEMBER_A, "/new A的第二个会话")
    sa1 = _sid(MEMBER_A.user_id)
    assert sa1 != sa0 and _sid(MEMBER_B.user_id) == sb0 and _sid(OWNER) == so0, "A 的 /new 不动别人的活动会话"

    ra = (await _say(MEMBER_A, "/sessions")).text
    rb = (await _say(MEMBER_B, "/sessions")).text
    ro = (await _say(ID, "/sessions")).text
    assert sa0[:8] in ra and sa1[:8] in ra and sb0[:8] not in ra and so0[:8] not in ra, ra
    assert sb0[:8] in rb and sa0[:8] not in rb and sa1[:8] not in rb and so0[:8] not in rb, rb
    assert so0[:8] in ro and sa0[:8] not in ro and sb0[:8] not in ro, ro

    # B 想切到 A 的会话：完整 id / 前缀 / 序号都找不到，而且回复与“根本不存在的 id”除去回显外完全一致
    fake = "deadbeef" + "0" * 24
    base = (await _say(MEMBER_B, f"/switch {fake}")).text.replace(fake, "<ref>")
    assert "没有找到" in base
    for ref in (sa0, sa1, sa0[:8], sa1[:8], sa0[:12]):
        txt = (await _say(MEMBER_B, f"/switch {ref}")).text
        assert txt.replace(ref, "<ref>") == base, (ref, txt, base)          # 与“根本不存在的 id”除去回显外完全一致
    for idx in ("2", "3", "99"):
        txt = (await _say(MEMBER_B, f"/switch {idx}")).text
        # B 只有 1 个会话，更大的序号走不到序号路径；随后按前缀在 B 自己的会话里找——会话 id 是随机 uuid，
        # 偶尔恰好以这个数字开头，那也只会命中 B 自己的会话，绝不会是别人的
        assert "没有找到" in txt or sb0.startswith(idx), txt
        assert sa0[:8] not in txt and sa1[:8] not in txt and so0[:8] not in txt, txt
    assert _sid(MEMBER_B.user_id) == sb0 and _sid(MEMBER_A.user_id) == sa1, "越权尝试之后各自的活动会话不变"

    # 自己的会话：序号与前缀都能切；owner 同样只在自己的会话里
    assert "已切换" in (await _say(MEMBER_A, "/switch 2")).text and _sid(MEMBER_A.user_id) == sa0
    assert "已切换" in (await _say(MEMBER_A, f"/switch {sa1[:8]}")).text and _sid(MEMBER_A.user_id) == sa1
    assert "没有找到" in (await _say(ID, f"/switch {sb0[:8]}")).text and _sid(OWNER) == so0


async def test_p4_member_reset_never_touches_sandbox():
    """P4 member 的 /reset 清自己的上下文，绝不重置本机 Docker 沙箱；owner 的会调用，且不影响别人的上下文。"""
    await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")])
    await _say(ID, "owner 说话")
    await _say(MEMBER_A, "A 说话")
    r = await _say(MEMBER_A, "/reset")
    assert "已清空" in r.text and reset_calls == [], reset_calls
    st_a = await cs._agents["member"].aget_state({"configurable": {"thread_id": _sid(MEMBER_A.user_id)}})
    st_o = await cs._agents["owner"].aget_state({"configurable": {"thread_id": _sid(OWNER)}})
    assert not st_a.values.get("messages") and st_o.values.get("messages"), "A 的 /reset 只清 A 的上下文"
    r = await _say(ID, "/reset")
    assert "已清空" in r.text and reset_calls == [1], "owner 的 /reset 才会重置沙箱"


async def test_p5_admin_commands():
    """P5 管理命令仅 owner 可用：member 一律固定回复且无副作用；owner 的 /users /label /block /unblock 行为正确；封禁生效且 owner 不可被封禁。"""
    omm, mmm = await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")])
    await _say(ID, "owner 先出现")
    await _say(MEMBER_A, "A 说话")
    await _say(MEMBER_B, "B 说话")
    store = _env.store
    users_before = [(u.user_id, u.label, u.blocked) for u in store.list_users()]
    for cmd in ("/users", "/label 1 乱写备注", "/label 2 乱写备注", "/block 1", "/block 2", "/unblock 2", "/BLOCK 3", "/users extra"):
        r = await _say(MEMBER_B, cmd)
        assert r.text == "该命令仅管理员可用" and not r.awaiting_approval, (cmd, r.text)
        assert A_PREFIX not in r.text
    assert [(u.user_id, u.label, u.blocked) for u in store.list_users()] == users_before, "member 的管理命令不得有任何副作用"

    # owner：/users
    r = await _say(ID, "/users")
    lines = [ln for ln in r.text.splitlines() if ln[:2] in ("1.", "2.", "3.")]
    assert len(lines) == 3, r.text
    assert "owner" in lines[0] and MEMBER_A.user_id[:8] in lines[1] and MEMBER_B.user_id[:8] in lines[2], lines
    assert "个会话" in lines[1] and "最后活跃" in lines[1] and "[已封禁]" not in r.text

    # /label：序号与前缀都行；找不到 / 没参数给用法
    assert "备注" in (await _say(ID, "/label 2 小明 QQ号 12345")).text
    assert "备注" in (await _say(ID, f"/label {MEMBER_B.user_id[:8]} 小红")).text
    assert [store.get_user(MEMBER_A.user_id).label, store.get_user(MEMBER_B.user_id).label] == ["小明 QQ号 12345", "小红"]
    assert "小明" in (await _say(ID, "/users")).text and "小红" in (await _say(ID, "/users")).text
    assert "用法" in (await _say(ID, "/label")).text and "没有找到" in (await _say(ID, "/label 99 x")).text
    assert "没有找到" in (await _say(ID, "/block zzzz")).text and "用法" in (await _say(ID, "/block")).text

    # /block：B 被封后任何输入（含 /whoami /stop）都是固定回复，不进入后续流程；A 不受影响；/unblock 恢复
    calls_before = mmm.idx
    assert "已封禁" in (await _say(ID, "/block 3")).text and store.get_user(MEMBER_B.user_id).blocked
    for text in ("你好", "/whoami", "/stop", "/new x", "/users"):
        r = await _say(MEMBER_B, text)
        assert r.text == "你当前无法使用本机器人。", (text, r.text)
    assert mmm.idx == calls_before, "被封禁的用户不会触发任何模型调用"
    assert "[已封禁]" in (await _say(ID, "/users")).text
    assert (await _say(MEMBER_A, "A 还能用")).text.endswith("m")
    assert "已解封" in (await _say(ID, f"/unblock {MEMBER_B.user_id[:8]}")).text
    assert (await _say(MEMBER_B, "B 恢复了")).text.endswith("m")

    # owner 不可被封禁，也不可能被封禁后锁在门外
    assert "不可被封禁" in (await _say(ID, "/block 1")).text and not store.get_user(OWNER).blocked
    store._conn.execute("UPDATE users SET blocked = 1 WHERE user_id = 'owner'")      # 即使库被改坏，owner 也照常能用
    assert (await _say(ID, "owner 照常")).text == "o"
    store._conn.execute("UPDATE users SET blocked = 0 WHERE user_id = 'owner'")


A_PREFIX = MEMBER_A.user_id[:8]


async def test_p6_whoami():
    """P6 /whoami 所有人可用，输出平台 ID、内部 ID、身份、CodeAgent 后端；QQ owner 与 CLI owner 同为 owner。"""
    await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")])
    r = (await _say(ID, "/whoami")).text
    assert "平台 ID：cli" in r and "内部 ID：owner" in r and "身份：管理员" in r and "本机 Docker" in r, r
    qq_owner_id = "QQOWNER00000001"
    old = config.OWNER_USER_IDS
    config.OWNER_USER_IDS = frozenset({qq_owner_id})
    try:
        qq_owner = resolve_platform_user(qq_owner_id)
        r = (await _say(qq_owner, "/whoami")).text
        assert f"平台 ID：{qq_owner_id}" in r and "内部 ID：owner" in r and "身份：管理员" in r, r
        assert _sid(OWNER) == cs.get_active_session_id(user_id="owner"), "QQ owner 与 CLI owner 共享同一套会话"
    finally:
        config.OWNER_USER_IDS = old
    r = (await _say(MEMBER_A, "/whoami")).text
    assert f"平台 ID：{MEMBER_A.raw_id}" in r and f"内部 ID：{MEMBER_A.user_id}" in r and "身份：成员" in r and "未启用" in r, r
    assert "本机 Docker" not in r and "管理员" not in r.replace("身份：成员", "")


async def test_p7_welcome_once():
    """P7 欢迎语只给 member、每个用户只出现一次（首条回复之前）；owner 永远没有。"""
    await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")])
    welcome = config.MEMBER_WELCOME_TEXT.strip()
    r1 = (await _say(MEMBER_A, "第一条")).text
    r2 = (await _say(MEMBER_A, "第二条")).text
    assert r1.startswith(welcome) and r1.endswith("m") and r2 == "m", (r1, r2)
    rb = (await _say(MEMBER_B, "/whoami")).text                      # 第一条是命令时同样附在这条回复前
    assert rb.startswith(welcome) and "身份：成员" in rb
    assert (await _say(MEMBER_B, "/whoami")).text.startswith("平台 ID")
    ro = (await _say(ID, "你好")).text
    assert ro == "o" and welcome not in ro
    assert (await _say(ID, "/whoami")).text.startswith("平台 ID")


async def test_p8_member_semaphore():
    """P8 member 全局并发上限生效：上限 1 时第二个 member 排队；owner 不占用也不排队。"""
    old = config.MAX_CONCURRENT_MEMBER_TURNS
    config.MAX_CONCURRENT_MEMBER_TURNS = 1
    try:
        omm, mmm = await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")], member_delay=0.8)
        t0 = time.monotonic()
        ta, tb = _start(MEMBER_A, "A 的问题"), _start(MEMBER_B, "B 的问题")
        await asyncio.sleep(0.25)
        ro = await _say(ID, "owner 的问题")
        assert ro.text == "o" and not (ta.done() and tb.done()), "owner 不占用 member 的名额，也不排队"
        done_at = {}
        for name, t in (("a", ta), ("b", tb)):
            r = await t
            done_at[name] = time.monotonic() - t0
            assert r.text.endswith("m"), r.text
        first, second = sorted(done_at.values())
        # 上限 1：两个 member 被串行化，第二个比第一个晚完成约一个模型延迟（并行的话两者几乎同时完成）
        assert second - first >= 0.6, f"第二个应排在第一个之后完成: {done_at}"
        assert mmm.idx == 2
        await asyncio.sleep(0.2)

        # 排队期间被 /stop：取消排队中的那一轮，名额不泄漏
        await cs.process_message("/reset", identity=MEMBER_A)
        ta, tb = _start(MEMBER_A, "A 再问"), _start(MEMBER_B, "B 再问")
        await asyncio.sleep(0.25)
        victim = MEMBER_B if cs._current_turns.get(MEMBER_B.user_id) is not None else MEMBER_A
        await _say(victim, "/stop")
        results = await asyncio.gather(ta, tb)
        assert sum(1 for r in results if r is None) == 1, results
        await asyncio.sleep(0.9)
        r = await _say(MEMBER_A, "之后还能正常用")
        assert r is not None and r.text.endswith("m"), r
    finally:
        config.MAX_CONCURRENT_MEMBER_TURNS = old
        await asyncio.sleep(0.8)


async def test_p9_graph_selection_and_fail_closed():
    """P9 owner 走 owner 图、member 走 member 图；缺 member 图时 member 失败而不是回落到 owner 图；run_config 带 user_id；上下文只在 wake 期间存在。"""
    from app.utils.context import get_request_context
    omm, mmm = await _install_two_graphs([AIMessage(content="来自owner图")], [AIMessage(content="来自member图")])
    assert (await _say(ID, "q")).text == "来自owner图"
    assert (await _say(MEMBER_A, "q")).text.endswith("来自member图")
    assert omm.idx == 1 and mmm.idx == 1
    assert get_request_context() is None, "wake 结束后请求上下文必须还原"
    cfg = cs._run_config("sid123", "owner")
    assert cfg["configurable"] == {"thread_id": "sid123", "user_id": "owner"}
    assert cfg["metadata"] == {"user_id": "owner"}, "cli_logger 靠回调 metadata 里的 user_id 记录每次 LLM 调用属于谁"
    other = cs._run_config("sid123", MEMBER_A.user_id)
    assert other["configurable"]["user_id"] == other["metadata"]["user_id"] == MEMBER_A.user_id

    # 缺 member 图：member 得到通用错误，owner 图一次都不会被调用
    cs._agents = {"owner": cs._agents["owner"]}
    before = omm.idx
    r = await _say(MEMBER_B, "我是 member")
    assert "来自owner图" not in r.text and "处理时出了点问题" in r.text, r.text
    assert omm.idx == before, "绝不回落到 owner 图"
    try:
        cs._agent_for(MEMBER_B.user_id)
        raise AssertionError("缺 member 图应抛 RuntimeError")
    except RuntimeError:
        pass


async def test_p10_identity_validation():
    """P10 identity 不自洽直接拒绝（不触库、不调用模型）；缺 identity 是 TypeError。"""
    from app.utils.identity import Identity
    omm, mmm = await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")])
    users_before = _env.store.list_users()
    bad_identities = [
        Identity(user_id="owner", is_owner=False, raw_id="x"),            # 声称是 owner 却 is_owner=False
        Identity(user_id=MEMBER_A.user_id, is_owner=True, raw_id="x"),     # 普通用户却声称是管理员
        Identity(user_id="bad id", is_owner=False, raw_id="x"),
        Identity(user_id="", is_owner=False, raw_id="x"),
        Identity(user_id="owner\n", is_owner=True, raw_id="x"),
        None, "owner", {"user_id": "owner", "is_owner": True, "raw_id": "x"},
    ]
    for ident in bad_identities:
        r = await cs.process_message("/users", identity=ident)
        assert r.text == "身份无效，已拒绝。", (ident, r.text)
    assert omm.idx == 0 and mmm.idx == 0
    assert _env.store.list_users() == users_before, "身份无效的消息不得写 users 表"
    try:
        await cs.process_message("你好")
        raise AssertionError("缺 identity 应抛 TypeError")
    except TypeError:
        pass


async def test_p11_recover_orphans_multiuser():
    """P11 崩溃遗留：所有用户的未处理入站事件都被标记放弃，不调用任何模型。"""
    omm, mmm = await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")])
    store = _env.store
    so, sa = _sid(OWNER), _sid(MEMBER_A.user_id)
    e1 = store.emit_event(so, "user_message", {"text": "owner 崩溃前", "processed_text": "x", "has_image": False})
    e2 = store.emit_event(sa, "user_message", {"text": "A 崩溃前", "processed_text": "x", "has_image": False})
    assert store.get_pending_events(so) and store.get_pending_events(sa)
    assert await cs.recover_orphans() == 2
    assert store.get_pending_events(so) == [] and store.get_pending_events(sa) == []
    assert [e.payload["event_seq"] for e in store.get_events(so, types=["turn_abandoned"])] == [e1]
    assert [e.payload["event_seq"] for e in store.get_events(sa, types=["turn_abandoned"])] == [e2]
    assert omm.idx == 0 and mmm.idx == 0


async def test_p12_startup_notices():
    """P12 启动提示：OWNER_USER_IDS 为空时给醒目提示；非法项与 MYSQL_USER=root 也有提示；配置齐全时不打扰。"""
    old_ids, old_user = config.OWNER_USER_IDS, config.MYSQL_USER
    try:
        config.OWNER_USER_IDS, config.MYSQL_USER = frozenset(), "root"
        text = "\n".join(cs._startup_notices())
        assert "OWNER_USER_IDS" in text and "/whoami" in text and "root" in text and "只读" in text, text
        config.OWNER_USER_IDS, config.MYSQL_USER = frozenset({"A1B2C3D4E5F60718", "bad-id"}), "readonly"
        notices = cs._startup_notices()
        assert len(notices) == 1 and "bad-id" in notices[0] and "A1B2C3D4E5F60718" not in notices[0], notices
        config.OWNER_USER_IDS = frozenset({"A1B2C3D4E5F60718"})
        assert cs._startup_notices() == []
    finally:
        config.OWNER_USER_IDS, config.MYSQL_USER = old_ids, old_user


async def test_p14_init_and_shutdown_service():
    """P14 init_service：编译两张图装进 _agents、启动恢复处理所有用户的遗留事件；
    shutdown_service：并发中止所有用户在跑的轮并清空全部进程内状态。"""
    global _env
    _env = _Env()
    store = SessionStore(_env.db_path)
    _env.store = store
    session_store.set_store(store)
    so = store.get_active_session_id(user_id=OWNER)
    sa = store.get_active_session_id(user_id=MEMBER_A.user_id)
    store.emit_event(so, "user_message", {"text": "owner 崩溃前", "processed_text": "x", "has_image": False})
    store.emit_event(sa, "user_message", {"text": "A 崩溃前", "processed_text": "x", "has_image": False})

    real_open, real_validate, real_build = checkpointer.open_checkpointer, cs.models.validate_required_keys, cs.orchestrator.build_main_agents
    seen = {}

    async def open_tmp(*args, **kwargs):
        return await real_open(_env.cp_path)          # 绝不碰真实的 storage/checkpoints.db

    async def fake_build(checkpointer):
        seen["saver"] = checkpointer

        def mk(reply):
            return create_deep_agent(model=ScriptedModel(script=[AIMessage(content=reply)], delay=1.5), subagents=[], checkpointer=checkpointer)

        return types.SimpleNamespace(
            owner=mk("o"), member=mk("m"),
            owner_experts=("FileAgent",), member_experts=("KnowledgeAgent", "MediaAgent", "MapAgent"),
        )

    checkpointer.open_checkpointer = open_tmp
    cs.models.validate_required_keys = lambda: None
    cs.orchestrator.build_main_agents = fake_build
    try:
        await cs.init_service()
        _env.saver = cs._checkpointer                  # 让 _teardown 能关闭它
        assert cs.is_initialized() and set(cs._agents) == {"owner", "member"}, cs._agents
        assert cs._agents["owner"] is not cs._agents["member"] and seen["saver"] is cs._checkpointer
        assert store.get_pending_events(so) == [] and store.get_pending_events(sa) == [], "启动恢复：所有用户的遗留事件都被标记放弃"

        # 两个用户各跑一个慢轮：shutdown 把它们都中止，并清空全部进程内状态
        t_owner, t_member = _start(ID, "owner 的慢问题"), _start(MEMBER_A, "A 的慢问题")
        await asyncio.sleep(0.4)
        assert set(cs._current_turns) == {OWNER, MEMBER_A.user_id}
        await cs.shutdown_service()
        assert await t_owner is None and await t_member is None, "关闭服务时所有用户在跑的轮都被中止"
        assert not cs.is_initialized() and cs._agents == {} and cs._current_turns == {} and cs._user_locks == {}
        assert cs._checkpointer is None and cs._member_sem is None
        verify = SessionStore(_env.db_path)             # shutdown_service 已关掉 store：重新打开同一个库来核对事件日志
        try:
            reasons = lambda sid: [e.payload["reason"] for e in verify.get_events(sid, types=["turn_failed"])]
            assert reasons(so) == ["user_stop"] and reasons(sa) == ["user_stop"], (reasons(so), reasons(sa))
        finally:
            verify.close()
    finally:
        checkpointer.open_checkpointer, cs.models.validate_required_keys, cs.orchestrator.build_main_agents = real_open, real_validate, real_build
        await asyncio.sleep(1.2)                        # 让被放弃的模型线程自然结束


class _ProbeModel(BaseChatModel):
    """无状态假模型：没有工具结果就调一次 ctx_probe，有了就收尾。多个用户并发共用同一张图时也不会串。"""
    reply: str

    @property
    def _llm_type(self) -> str:
        return "probe"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content=self.reply)
        else:
            msg = AIMessage(content="", tool_calls=[{"name": "ctx_probe", "args": {}, "id": "probe1"}])
        time.sleep(0.15)                                  # 让并发的几个用户真正交错
        return ChatResult(generations=[ChatGeneration(message=msg)])


register_harness_profile(
    "probemodel",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset({"write_file", "edit_file", "delete", "execute", "ls", "glob", "grep"}),
    ),
)
_ctx_log: list = []


@tool
def ctx_probe(config: RunnableConfig) -> str:
    """记录工具执行时看到的请求上下文与 RunnableConfig。"""
    ctx = get_request_context()
    configurable = (config or {}).get("configurable") or {}
    _ctx_log.append({
        "ctx_user": ctx.user_id if ctx else None,
        "ctx_session": ctx.session_id if ctx else None,
        "ctx_is_owner": ctx.is_owner if ctx else None,
        "cfg_user": configurable.get("user_id"),
        "cfg_thread": configurable.get("thread_id"),
    })
    return "ok"


async def test_p13_wake_sets_request_context():
    """P13 经由 process_message → wake：并发的 owner / 两个 member 各自的工具都看到自己的请求上下文与 RunnableConfig；wake 之外没有上下文。"""
    global _env
    _env = _Env()
    await _open_stack(_env)
    _ctx_log.clear()
    owner_agent = create_deep_agent(model=_ProbeModel(reply="owner图回复"), tools=[ctx_probe], subagents=[], checkpointer=_env.saver)
    member_agent = create_deep_agent(model=_ProbeModel(reply="member图回复"), tools=[ctx_probe], subagents=[], checkpointer=_env.saver)
    cs._agents = {"owner": owner_agent, "member": member_agent}
    cs._initialized, cs._checkpointer = True, _env.saver

    replies = await asyncio.gather(_say(ID, "o"), _say(MEMBER_A, "a"), _say(MEMBER_B, "b"))
    assert [r.text.endswith(t) for r, t in zip(replies, ("owner图回复", "member图回复", "member图回复"))] == [True] * 3, [r.text for r in replies]
    assert len(_ctx_log) == 3 and None not in [r["ctx_user"] for r in _ctx_log], _ctx_log
    by_user = {r["ctx_user"]: r for r in _ctx_log}
    assert set(by_user) == {OWNER, MEMBER_A.user_id, MEMBER_B.user_id}, by_user
    for uid, is_owner in ((OWNER, True), (MEMBER_A.user_id, False), (MEMBER_B.user_id, False)):
        rec = by_user[uid]
        assert rec["ctx_is_owner"] is is_owner and rec["cfg_user"] == uid, rec
        assert rec["ctx_session"] == rec["cfg_thread"] == _sid(uid), "上下文里的会话就是该用户当前的活动会话，thread_id 同一个"
    assert get_request_context() is None, "wake 之外不应残留请求上下文"


# ── /stop：只中止发送者自己的那一轮；宿主机 / 沙箱终止原语只有 owner 的 /stop 才会触发 ──

class _KillSpy:
    """替换 host_shell.kill_all_running 与 sandbox.interrupt_sandbox，记录谁在什么时候触发了它们。"""

    def __enter__(self):
        from app.infra import host_shell
        self.host_shell = host_shell
        self.host_calls, self.sandbox_calls = [], []
        self._orig = (host_shell.kill_all_running, sandbox.interrupt_sandbox)

        def fake_kill():
            self.host_calls.append(1)
            return 2

        def fake_interrupt():
            self.sandbox_calls.append(1)
            return False

        host_shell.kill_all_running, sandbox.interrupt_sandbox = fake_kill, fake_interrupt
        return self

    def __exit__(self, *exc):
        self.host_shell.kill_all_running, sandbox.interrupt_sandbox = self._orig


async def test_q1_member_stop_only_stops_self():
    """Q1 member 的 /stop 只中止自己的轮：owner 正在跑的轮不受影响；member 的 /reset 忙时同理。"""
    omm, mmm = await _install_two_graphs([AIMessage(content="owner慢回复")], [AIMessage(content="member慢回复")],
                                         owner_delay=1.0, member_delay=1.0)
    owner_turn = _start(ID, "owner 的慢问题")
    member_turn = _start(MEMBER_A, "A 的慢问题")
    await asyncio.sleep(0.3)
    assert OWNER in cs._current_turns and MEMBER_A.user_id in cs._current_turns
    stop = await _say(MEMBER_A, "/stop")
    assert "已中止" in stop.text and "宿主机命令" not in stop.text, stop.text
    assert await member_turn is None
    assert OWNER in cs._current_turns and not owner_turn.done(), "member 的 /stop 不得波及 owner 的轮"
    assert MEMBER_A.user_id not in cs._current_turns
    ro = await owner_turn
    assert ro is not None and ro.text == "owner慢回复"
    reasons = lambda sid: [e.payload["reason"] for e in _env.store.get_events(sid, types=["turn_failed"])]
    assert reasons(_sid(MEMBER_A.user_id)) == ["user_stop"] and reasons(_sid(OWNER)) == []

    # member 忙时 /reset：只先中止自己，owner 的轮照常跑完
    await asyncio.sleep(1.1)
    owner_turn = _start(ID, "owner 又一个慢问题")
    member_turn = _start(MEMBER_A, "A 又一个慢问题")
    await asyncio.sleep(0.3)
    r = await _say(MEMBER_A, "/reset")
    assert "已先中止" in r.text and "已清空" in r.text and reset_calls == [], (r.text, reset_calls)
    assert await member_turn is None
    assert (await owner_turn).text == "owner慢回复"
    await asyncio.sleep(1.1)


async def test_q2_kill_primitives_owner_only():
    """Q2 宿主机 / 沙箱的终止原语是进程级全局的：member 的 /stop（不管有没有在跑）都不得触发；owner 的才会。"""
    omm, mmm = await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")], owner_delay=1.0, member_delay=0.8)
    with _KillSpy() as spy:
        # member 空闲时 /stop；member 在跑时 /stop；owner 在跑、member 来 /stop
        assert "没有正在运行" in (await _say(MEMBER_A, "/stop")).text
        owner_turn = _start(ID, "owner 慢问题")
        await asyncio.sleep(0.25)
        assert "没有正在运行" in (await _say(MEMBER_B, "/stop")).text
        member_turn = _start(MEMBER_A, "A 慢问题")
        await asyncio.sleep(0.25)
        assert "已中止" in (await _say(MEMBER_A, "/stop")).text and await member_turn is None
        assert spy.host_calls == [] and spy.sandbox_calls == [], "member 的 /stop 绝不能杀 owner 的宿主机命令或中断沙箱"
        assert not owner_turn.done()
        # owner 自己 /stop：两个原语各触发一次，回复里报告被终止的宿主机命令数
        stop = await _say(ID, "/stop")
        assert "已中止" in stop.text and "已终止 2 个" in stop.text, stop.text
        assert await owner_turn is None
        assert spy.host_calls == [1] and spy.sandbox_calls == [1]
        # owner 空闲时 /stop 不触发任何原语
        assert "没有正在运行" in (await _say(ID, "/stop")).text
        assert spy.host_calls == [1] and spy.sandbox_calls == [1]
    await asyncio.sleep(1.2)


async def test_q3_owner_stop_does_not_stop_member():
    """Q3 owner 的 /stop 与 /reset 不会中止 member 正在跑的轮。"""
    omm, mmm = await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="member慢回复")], member_delay=1.0)
    member_turn = _start(MEMBER_A, "A 的慢问题")
    await asyncio.sleep(0.3)
    assert "没有正在运行" in (await _say(ID, "/stop")).text
    assert "已先中止" not in (await _say(ID, "/reset")).text
    assert not member_turn.done() and MEMBER_A.user_id in cs._current_turns
    r = await member_turn
    assert r is not None and r.text.endswith("member慢回复")
    await asyncio.sleep(0.2)


async def test_q4_block_and_stop_all_turns():
    """Q4 /block 会顺手中止被封用户正在跑的轮；_stop_all_turns（shutdown 用）中止所有用户的轮。"""
    omm, mmm = await _install_two_graphs([AIMessage(content="owner慢回复")], [AIMessage(content="member慢回复")],
                                         owner_delay=1.0, member_delay=1.0)
    await _say(ID, "owner 先出现")
    await _say(MEMBER_A, "A 先出现")
    await asyncio.sleep(0.1)
    mem_turn = _start(MEMBER_A, "A 的慢问题")
    await asyncio.sleep(0.3)
    r = await _say(ID, "/block 2")
    assert "已封禁" in r.text and "中止了它正在运行的任务" in r.text, r.text
    assert await mem_turn is None and _env.store.get_user(MEMBER_A.user_id).blocked
    await asyncio.sleep(0.9)

    # _stop_all_turns：owner 与另一个 member 同时在跑，一次全部中止
    owner_turn = _start(ID, "owner 慢问题")
    b_turn = _start(MEMBER_B, "B 慢问题")
    await asyncio.sleep(0.3)
    assert OWNER in cs._current_turns and MEMBER_B.user_id in cs._current_turns
    with _KillSpy() as spy:
        await cs._stop_all_turns()
    assert await owner_turn is None and await b_turn is None
    assert cs._current_turns == {}
    assert spy.host_calls == [1], "owner 的轮被中止时才触发宿主机终止原语（且只一次）"
    await asyncio.sleep(1.1)


# ── 阶段 2：member 的云端沙箱钩子（用替换 modal_sandbox 模块属性的方式观察；一律离线，池从不被创建）──

class _ModalSpy:
    """替换 modal_sandbox 上的钩子：记录 (名字, 参数, 调用所在的线程)；hooks 可以统一抛异常。"""

    def __init__(self, reset_user_result=0, raises=None, status=(False, "测试里没有配置 Modal")):
        self.reset_user_result, self.raises, self.status = reset_user_result, raises, status
        self.calls = []

    def __enter__(self):
        self.loop_thread = threading.get_ident()
        self._orig = {n: getattr(modal_sandbox, n) for n in
                      ("reset_session", "reset_user", "interrupt_session_nowait", "shutdown_pool", "member_code_status")}

        def make(name, result=None):
            def hook(*args, **kwargs):
                self.calls.append((name, args, threading.get_ident()))
                if self.raises is not None and name != "member_code_status":
                    raise self.raises
                return result() if callable(result) else result
            return hook

        modal_sandbox.reset_session = make("reset_session")
        modal_sandbox.reset_user = make("reset_user", lambda: self.reset_user_result)
        modal_sandbox.interrupt_session_nowait = make("interrupt_session_nowait")
        modal_sandbox.shutdown_pool = make("shutdown_pool")
        modal_sandbox.member_code_status = make("member_code_status", lambda: self.status)
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(modal_sandbox, name, fn)

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


async def test_r1_reset_hook():
    """R1 member 的 /reset → reset_session(member 的 user_id, 当前 session_id) 恰好一次，在事件循环线程之外执行，Docker 沙箱没被碰；
    owner 反过来；钩子抛异常时回复文本不变。"""
    await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")])
    await _say(MEMBER_A, "A 先出现")
    sid = _sid(MEMBER_A.user_id)
    with _ModalSpy() as spy:
        reply = await _say(MEMBER_A, "/reset")
        assert "已清空" in reply.text, reply.text
        hooks = spy.named("reset_session")
        assert len(hooks) == 1 and hooks[0][1] == (MEMBER_A.user_id, sid), hooks
        assert hooks[0][2] != spy.loop_thread, "阻塞钩子必须在事件循环线程之外执行"
        assert reset_calls == [], "member 的 /reset 绝不能重置 owner 的 Docker 沙箱"
        assert spy.named("reset_user") == [] and spy.named("interrupt_session_nowait") == []
    with _ModalSpy(raises=RuntimeError("boom")) as failing:
        again = await _say(MEMBER_A, "/reset")
        assert again.text == reply.text, "钩子失败不改变回复文本"
        assert len(failing.named("reset_session")) == 1
    with _ModalSpy() as spy_owner:
        await _say(ID, "owner 先出现")
        owner_reply = await _say(ID, "/reset")
        assert "已清空" in owner_reply.text and reset_calls == [1], reset_calls
        assert spy_owner.named("reset_session") == [], "owner 的 /reset 不得调用任何 Modal 钩子"


async def test_r2_stop_hook():
    """R2 member 的 /stop → interrupt_session_nowait(member 的 user_id, 那一轮的 session_id) 一次；host_shell 与 Docker 的终止原语都不碰；
    不影响 owner 与其他 member 的轮；owner 的 /stop 不调用任何 modal_sandbox 函数。"""
    await _install_two_graphs([AIMessage(content="owner慢回复")], [AIMessage(content="member慢回复")], owner_delay=1.2, member_delay=1.2)
    with _KillSpy() as kill_spy, _ModalSpy() as spy:
        member_turn, other_turn = _start(MEMBER_A, "A 慢问题"), _start(MEMBER_B, "B 慢问题")
        owner_turn = _start(ID, "owner 慢问题")
        await asyncio.sleep(0.4)
        handle_sid = cs._current_turns[MEMBER_A.user_id].session_id
        stop = await _say(MEMBER_A, "/stop")
        assert "已中止" in stop.text and await member_turn is None
        hooks = spy.named("interrupt_session_nowait")
        assert len(hooks) == 1 and hooks[0][1] == (MEMBER_A.user_id, handle_sid), hooks
        assert kill_spy.host_calls == [] and kill_spy.sandbox_calls == [], "member 的 /stop 绝不能走到进程级全局的终止原语"
        assert not owner_turn.done() and not other_turn.done(), "member 的 /stop 不影响 owner 与其他 member 的轮"
        spy.calls.clear()
        owner_stop = await _say(ID, "/stop")
        assert "已中止" in owner_stop.text and await owner_turn is None
        assert spy.calls == [], f"owner 的 /stop 不得调用任何 modal_sandbox 函数: {spy.calls}"
        assert kill_spy.host_calls == [1] and kill_spy.sandbox_calls == [1]
        assert (await other_turn).text.endswith("member慢回复")
    await asyncio.sleep(1.3)


async def test_r3_failed_turn_interrupts_member_sandbox():
    """R3 member 的回合超时 → interrupt_session_nowait 恰好一次；用户 /stop 引起的失败总共仍然只有一次（来自 _stop_current_turn，不重复）；
    owner 失败 → 没有调用。"""
    await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")], owner_delay=1.5, member_delay=1.5)
    old_timeout = config.AGENT_TURN_TIMEOUT_SECONDS
    try:
        with _ModalSpy() as spy:
            config.AGENT_TURN_TIMEOUT_SECONDS = 0.4
            reply = await _say(MEMBER_A, "A 的超时问题")
            assert "还没完成" in reply.text, reply.text
            sid = _sid(MEMBER_A.user_id)
            hooks = spy.named("interrupt_session_nowait")
            assert len(hooks) == 1 and hooks[0][1] == (MEMBER_A.user_id, sid), hooks

            config.AGENT_TURN_TIMEOUT_SECONDS = old_timeout
            spy.calls.clear()
            turn = _start(MEMBER_A, "A 的慢问题")
            await asyncio.sleep(0.3)
            await _say(MEMBER_A, "/stop")
            assert await turn is None
            assert len(spy.named("interrupt_session_nowait")) == 1, "/stop 引起的失败只调用一次（不重复）"

            config.AGENT_TURN_TIMEOUT_SECONDS = 0.4
            spy.calls.clear()
            await _say(ID, "owner 的超时问题")
            assert spy.calls == [], "owner 的失败不得触发任何 Modal 钩子"
    finally:
        config.AGENT_TURN_TIMEOUT_SECONDS = old_timeout
    await asyncio.sleep(1.6)


async def test_r4_block_recycles_sandboxes():
    """R4 /block：先中止再 reset_user 一次；返回 2 → 回复里有“已回收它的 2 个云端沙箱”；返回 0 → 回复与阶段 1 逐字一致；owner 不可被封禁且无调用。"""
    await _install_two_graphs([AIMessage(content="owner慢回复")], [AIMessage(content="member慢回复")], owner_delay=0.5, member_delay=1.0)
    await _say(ID, "owner 先出现")
    await _say(MEMBER_A, "A 先出现")
    await _say(MEMBER_B, "B 先出现")
    await asyncio.sleep(0.1)
    turn = _start(MEMBER_A, "A 的慢问题")
    await asyncio.sleep(0.3)
    with _ModalSpy(reset_user_result=2) as spy:
        reply = await _say(ID, "/block 2")
        assert "已封禁" in reply.text and "中止了它正在运行的任务" in reply.text and "已回收它的 2 个云端沙箱。" in reply.text, reply.text
        assert await turn is None
        order = [c[0] for c in spy.calls]
        assert order == ["interrupt_session_nowait", "reset_user"], order
        hooks = spy.named("reset_user")
        assert hooks[0][1] == (MEMBER_A.user_id,) and hooks[0][2] != spy.loop_thread, "reset_user 在事件循环线程之外执行"
    with _ModalSpy(reset_user_result=0) as spy:
        reply = await _say(ID, "/block 3")
        assert reply.text == f"已封禁 {MEMBER_B.user_id[:8]}。", reply.text      # 与阶段 1 逐字一致
        assert len(spy.named("reset_user")) == 1
    with _ModalSpy(raises=RuntimeError("boom")):
        reply = await _say(ID, "/unblock 2")
        assert "已解封" in reply.text, reply.text
        reply = await _say(ID, "/block 2")
        assert reply.text.startswith("已封禁") and "云端沙箱" not in reply.text, "钩子失败不改变回复"
    with _ModalSpy() as spy:
        reply = await _say(ID, "/block owner")
        assert reply.text == "管理员不可被封禁。" and spy.calls == [], (reply.text, spy.calls)
    await asyncio.sleep(0.6)


async def test_r5_shutdown_order_and_isolation():
    """R5 shutdown_service 调用 shutdown_pool 恰好一次，且在 mcp_service.shutdown_mcp 之前；
    _stop_all_turns 抛异常或 shutdown_pool 自己抛异常时，后续的关闭步骤（MCP、checkpointer 等）照样执行。"""
    from app.infra import mcp_service
    real_mcp = mcp_service.shutdown_mcp
    real_stop_all = cs._stop_all_turns

    for label, stop_raises, pool_raises in (("正常", False, False), ("_stop_all_turns 抛异常", True, False), ("shutdown_pool 抛异常", False, True)):
        await _install_two_graphs([AIMessage(content="o")], [AIMessage(content="m")])
        order = []

        async def traced_mcp():
            order.append("mcp")
            return await real_mcp()

        async def broken_stop_all():
            raise RuntimeError("stop-all failed")

        mcp_service.shutdown_mcp = traced_mcp
        if stop_raises:
            cs._stop_all_turns = broken_stop_all
        try:
            with _ModalSpy(raises=RuntimeError("pool failed") if pool_raises else None) as spy:
                spy_pool = modal_sandbox.shutdown_pool

                def traced_pool(*args, **kwargs):
                    order.append("pool")
                    return spy_pool(*args, **kwargs)

                modal_sandbox.shutdown_pool = traced_pool
                await cs.shutdown_service()
                assert len(spy.named("shutdown_pool")) == 1, label
        finally:
            mcp_service.shutdown_mcp = real_mcp
            cs._stop_all_turns = real_stop_all
        assert order == ["pool", "mcp"], (label, order)
        assert cs._checkpointer is None and not cs.is_initialized() and cs._agents == {}, label
        await _teardown()


async def _init_service_with_member_experts(member_experts, scheduled=None):
    """照 P14 的做法：用假的 build_main_agents / open_checkpointer 跑真实的 init_service。"""
    global _env
    _env = _Env()
    store = SessionStore(_env.db_path)
    _env.store = store
    session_store.set_store(store)
    real_open, real_validate, real_build = checkpointer.open_checkpointer, cs.models.validate_required_keys, cs.orchestrator.build_main_agents

    async def open_tmp(*args, **kwargs):
        return await real_open(_env.cp_path)

    async def fake_build(checkpointer):
        def mk(reply):
            return create_deep_agent(model=ScriptedModel(script=[AIMessage(content=reply)]), subagents=[], checkpointer=checkpointer)

        extra = {"scheduled": scheduled, "scheduled_experts": ("KnowledgeAgent", "MediaAgent", "MapAgent")} if scheduled is not None else {}
        return types.SimpleNamespace(owner=mk("o"), member=mk("m"), owner_experts=("FileAgent",), member_experts=member_experts, **extra)

    checkpointer.open_checkpointer, cs.models.validate_required_keys, cs.orchestrator.build_main_agents = open_tmp, lambda: None, fake_build
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            await cs.init_service()
        _env.saver = cs._checkpointer
    finally:
        checkpointer.open_checkpointer, cs.models.validate_required_keys, cs.orchestrator.build_main_agents = real_open, real_validate, real_build
    return out.getvalue()


async def test_r6_startup_banner_and_whoami():
    """R6 member_experts 带 CodeAgent → _member_code_enabled 为真、启动横幅已打印、member 的 /whoami 是“云端沙箱（Modal）”、owner 仍是“本机 Docker”；
    不带 CodeAgent → “未启用”并打印 member_code_status 的原因；shutdown_service 之后标志复位。"""
    with _ModalSpy():
        banner = await _init_service_with_member_experts(("KnowledgeAgent", "MediaAgent", "MapAgent", "CodeAgent"))
        assert cs._member_code_enabled is True
        assert "member 云端 CodeAgent：已启用（Modal）" in banner, banner
        assert "云端沙箱（Modal）" in (await _say(MEMBER_A, "/whoami")).text
        assert "CodeAgent 后端：本机 Docker" in (await _say(ID, "/whoami")).text
        await cs.shutdown_service()
        assert cs._member_code_enabled is False, "shutdown_service 之后标志复位"
        await _teardown()

    with _ModalSpy(status=(False, "MODAL_DISABLED=true")) as spy:
        banner = await _init_service_with_member_experts(("KnowledgeAgent", "MediaAgent", "MapAgent"))
        assert cs._member_code_enabled is False
        assert "member 云端 CodeAgent：未启用（MODAL_DISABLED=true）" in banner, banner
        assert len(spy.named("member_code_status")) == 1
        whoami = (await _say(MEMBER_A, "/whoami")).text
        assert "CodeAgent 后端：未启用" in whoami and "云端沙箱" not in whoami, whoami
        assert "CodeAgent 后端：本机 Docker" in (await _say(ID, "/whoami")).text
        await cs.shutdown_service()
        await _teardown()


async def test_r8_scheduled_graph_wiring():
    """R8 init_service 只在 built.scheduled 非 None 时才把第三张图写进 _agents["scheduled"]（旧的假 build 没有这个属性也照常工作），
    启动横幅多一行“定时任务：已启用 / 未启用”；_agent_for 永远只按 owner / member 取图，绝不返回 scheduled 图。"""
    members = ("KnowledgeAgent", "MediaAgent", "MapAgent")
    with _ModalSpy(status=(False, "MODAL_DISABLED=true")):
        banner = await _init_service_with_member_experts(members)
        assert set(cs._agents) == {"owner", "member"}, cs._agents
        assert "⏰ [chat_service] 定时任务：未启用" in banner, banner
        await cs.shutdown_service()
        await _teardown()

        sentinel = object()
        banner = await _init_service_with_member_experts(members, scheduled=sentinel)
        assert set(cs._agents) == {"owner", "member", "scheduled"} and cs._agents["scheduled"] is sentinel, cs._agents
        assert "⏰ [chat_service] 定时任务：已启用" in banner, banner
        assert cs._agent_for("owner") is cs._agents["owner"] and cs._agent_for(MEMBER_A) is cs._agents["member"]
        for uid in ("owner", MEMBER_A, "scheduled"):
            assert cs._agent_for(uid) is not sentinel, uid
        await cs.shutdown_service()
        assert cs._agents == {} and not cs.is_initialized(), "shutdown_service 之后三张图都清掉"
        await _teardown()


async def test_r7_startup_notice_wording():
    """R7 OWNER_USER_IDS 为空时的启动提示：不再说 member 没有“代码能力”，改为说明代码只能在云端沙箱里跑；P12 原有断言的关键词都在。"""
    old_ids, old_user = config.OWNER_USER_IDS, config.MYSQL_USER
    try:
        config.OWNER_USER_IDS, config.MYSQL_USER = frozenset(), "root"
        text = "\n".join(cs._startup_notices())
        assert "代码能力" not in text and "云端沙箱" in text, text
        assert "OWNER_USER_IDS" in text and "/whoami" in text and "root" in text and "只读" in text, text
    finally:
        config.OWNER_USER_IDS, config.MYSQL_USER = old_ids, old_user


SCENARIOS = [
    ("A 多轮 HITL（批准/无关消息/拒绝）", test_a_multi_turn_hitl),
    ("B HITL 前的图片随最终回复发出", test_b_outbox_continuity),
    ("C 并发忙提示", test_c_busy_lock),
    ("D 并行中断一次 Y 放行", test_d_parallel_interrupts),
    ("E 超时后会话可继续", test_e_timeout_recovery),
    ("F /reset 丢弃挂起的授权与历史", test_f_reset_command),
    ("G 图片轮不触发记忆提取", test_g_image_turn_no_memory),
    ("H 重启恢复（挂起授权 + 发件箱图片）", test_h_restart_recovery),
    ("I 多会话 /new /sessions /switch", test_i_multi_session),
    ("J 崩溃遗留事件标记放弃", test_j_crash_leftover),
    ("K 事件日志镜像", test_k_event_log),
    ("L 嵌套 checkpoint 清理", test_l_nested_prune),
    ("M /reset 保留事件日志", test_m_reset_keeps_log),
    ("N wake 幂等消费", test_n_wake_idempotent),
    ("O1 慢模型调用中 /stop", test_o1_stop_slow_model),
    ("O2 子专家工具中 /stop（悬空 tool_calls）", test_o2_stop_in_subagent_tool),
    ("O3 空闲 /stop", test_o3_idle_stop),
    ("O4 忙提示与忙时 /reset", test_o4_busy_hint_and_reset_while_busy),
    ("O5 授权恢复阶段被中止", test_o5_stop_during_approval_resume),
    ("O6 终止原语被正确触发", test_o6_kill_primitives_triggered),
    ("P1 不同用户互不阻塞，各用各的图", test_p1_users_do_not_block_each_other),
    ("P2 同一用户忙提示，其他用户不受影响", test_p2_same_user_busy_other_users_unaffected),
    ("P3 会话按用户隔离（/sessions /switch 越权）", test_p3_sessions_isolated_between_users),
    ("P4 member 的 /reset 不碰沙箱", test_p4_member_reset_never_touches_sandbox),
    ("P5 管理命令仅 owner 可用 / 封禁", test_p5_admin_commands),
    ("P6 /whoami", test_p6_whoami),
    ("P7 欢迎语只出现一次", test_p7_welcome_once),
    ("P8 member 并发上限（信号量）", test_p8_member_semaphore),
    ("P9 选图与失败即关闭", test_p9_graph_selection_and_fail_closed),
    ("P10 identity 校验", test_p10_identity_validation),
    ("P11 崩溃遗留（多用户）", test_p11_recover_orphans_multiuser),
    ("P12 启动提示", test_p12_startup_notices),
    ("P13 wake 设置请求上下文（并发多用户）", test_p13_wake_sets_request_context),
    ("P14 init_service 与 shutdown_service", test_p14_init_and_shutdown_service),
    ("Q1 member 的 /stop 只中止自己", test_q1_member_stop_only_stops_self),
    ("Q2 终止原语只有 owner 的 /stop 才触发", test_q2_kill_primitives_owner_only),
    ("Q3 owner 的 /stop 不中止 member", test_q3_owner_stop_does_not_stop_member),
    ("Q4 /block 与 _stop_all_turns", test_q4_block_and_stop_all_turns),
    ("R1 member 的 /reset 钩子", test_r1_reset_hook),
    ("R2 member 的 /stop 钩子", test_r2_stop_hook),
    ("R3 失败的回合中断 member 沙箱", test_r3_failed_turn_interrupts_member_sandbox),
    ("R4 /block 回收云端沙箱", test_r4_block_recycles_sandboxes),
    ("R5 关闭顺序与隔离", test_r5_shutdown_order_and_isolation),
    ("R6 启动横幅与 /whoami", test_r6_startup_banner_and_whoami),
    ("R7 启动提示文案", test_r7_startup_notice_wording),
    ("R8 scheduled 图的接线与启动横幅", test_r8_scheduled_graph_wiring),
]


async def _run(fn):
    try:
        await fn()
    finally:
        await _teardown()


def main():
    print("=" * 70)
    print("开始运行 chat_service 服务层离线冒烟测试...")
    print("=" * 70)
    passed = failed = 0
    for name, fn in SCENARIOS:
        try:
            asyncio.run(_run(fn))
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e!r}")
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
