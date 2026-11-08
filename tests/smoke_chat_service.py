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
不需要任何 API Key / Docker / MCP / 网络：orchestrator / injector / extractor / startup 在 import 前被桩掉。
"""
import asyncio
import json
import operator
import os
import shutil
import sys
import tempfile
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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
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


async def _fake_extract_memory(user_text):
    mem_calls.append(user_text)


_stub(app.agents, "orchestrator", build_main_agent=None)
_stub(
    app.memory,
    "injector",
    build_initial_messages=lambda user_query, history_messages=None: [HumanMessage(content=user_query, name="user")],
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
from app.infra import checkpointer, sandbox, session_store
from app.infra.session_store import SessionStore
from app.runtime import hitl, outbox

# 沙箱重置计数：/new /switch 不得触发，只有 /reset 触发（同时避免测试里碰 Docker）
reset_calls = []
sandbox.reset_sandbox = lambda: reset_calls.append(1)


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
    cs._agent, cs._initialized, cs._checkpointer = agent, True, _env.saver
    mem_calls.clear()
    reset_calls.clear()
    return _env.root, fam, mm


async def _restart(main_script, fa_script, extra=None):
    """模拟进程重启：关闭 saver 与 store，用同一批文件重建 saver / store / agent（全新的脚本化模型）。"""
    await checkpointer.close_checkpointer(_env.saver)
    session_store.close_store()
    await _open_stack(_env)
    agent, fam, mm = _build(_env, main_script, fa_script, extra)
    cs._agent, cs._checkpointer = agent, _env.saver
    return fam, mm


async def _teardown():
    """先关连接（含 -wal / -shm），再删目录（Windows 要求）。"""
    global _env
    cs._agent, cs._initialized, cs._checkpointer = None, False, None
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
    r = await cs.process_message("你好")
    await asyncio.sleep(0.05)
    assert r.text == "你好呀" and not r.awaiting_approval, r
    assert mem_calls and mem_calls[0] == "你好", mem_calls

    mem_calls.clear()
    r = await cs.process_message("写个文件")
    assert r.awaiting_approval and "write_file" in r.text and "workspace/a.txt" in r.text and r.images == [], r.text

    idx_before = fam.idx
    r = await cs.process_message("等等我想想")
    assert r.awaiting_approval and "当前正在等待授权确认" in r.text and fam.idx == idx_before, r.text

    r = await cs.process_message("y")
    await asyncio.sleep(0.05)
    assert (not r.awaiting_approval) and r.text == "写好了" and _files(root) == ["a.txt"], (r.text, _files(root))
    assert mem_calls == ["写个文件"], f"HITL 流程结束后只记录一次最初原话，实际: {mem_calls}"

    r = await cs.process_message("再写一个")
    assert r.awaiting_approval and "workspace/b.txt" in r.text, r.text

    r = await cs.process_message("N")
    assert _files(root) == ["a.txt"] and r.text == "已取消" and fam.idx == 4, (_files(root), r.text, fam.idx)
    sid = cs.get_active_session_id()
    assert not cs._chat_lock.locked() and _env.store.load_turn_state(sid) is None


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
    r1 = await cs.process_message("画图并写文件")
    ts = _env.store.load_turn_state(cs.get_active_session_id())
    assert r1.awaiting_approval and ts is not None and len(ts.outbox) == 1, r1
    r2 = await cs.process_message("Y")
    assert len(r2.images) == 1 and r2.text == "图和文件都好了" and cam.idx == 2, (len(r2.images), r2.text, cam.idx)
    # 图片字节永远不进事件日志，只记张数
    evs = _env.store.get_events(cs.get_active_session_id(), limit=200)
    assert "base64" not in json.dumps([e.payload for e in evs], ensure_ascii=False)
    assert [e.payload["image_count"] for e in evs if e.type == "assistant_message"] == [1]


async def test_c_busy_lock():
    """C 处理中再来一条 → 立即返回忙提示，第一条不受影响。"""
    root, fam, mm = await _install([AIMessage(content="慢回复")], [FA_DONE], delay=0.6)
    t = asyncio.create_task(cs.process_message("慢"))
    await asyncio.sleep(0.2)
    rb = await cs.process_message("插队")
    ra = await t
    assert "还在处理中" in rb.text and ra.text == "慢回复" and not cs._chat_lock.locked(), (rb.text, ra.text)


async def test_d_parallel_interrupts():
    """D 主 Agent 同时派 FileAgent + DesktopAgent：两类授权一起提示，一次 Y 同时放行。"""
    desk = CompiledSubAgent(name="DesktopAgent", description="d", runnable=_desktop_like())
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("FileAgent", "t1"), _task("DesktopAgent", "t2")]),
         AIMessage(content="两件事都办了")],
        [AIMessage(content="", tool_calls=[_write("workspace/q.txt", "c1")]), FA_DONE],
        extra=[desk],
    )
    r = await cs.process_message("同时办两件事")
    assert r.awaiting_approval and "write_file" in r.text and "桌面高危操作" in r.text, r.text
    r = await cs.process_message("Y")
    assert r.text == "两件事都办了" and _files(root) == ["q.txt"], (r.text, _files(root))


async def test_e_timeout_recovery():
    """E 单轮超时后给出友好提示，且当前会话仍可继续对话；事件日志里有 turn_failed(timeout)。"""
    root, fam, mm = await _install([AIMessage(content="很慢"), AIMessage(content="恢复了")], [FA_DONE], delay=1.2)
    old = config.AGENT_TURN_TIMEOUT_SECONDS
    config.AGENT_TURN_TIMEOUT_SECONDS = 0.3
    try:
        r = await cs.process_message("会超时的问题")
    finally:
        config.AGENT_TURN_TIMEOUT_SECONDS = old
    mm.delay = 0.0
    await asyncio.sleep(1.3)  # 等后台线程里被取消的那次模型调用自然结束
    sid = cs.get_active_session_id()
    failed = _env.store.get_events(sid, types=["turn_failed"])
    assert [e.payload["reason"] for e in failed] == ["timeout"], failed
    assert _env.store.get_pending_events(sid) == [], "失败的轮也必须推进游标"
    r2 = await cs.process_message("超时之后的新问题")
    assert "我先停下了" in r.text, r.text
    assert r2.text in ("恢复了", "很慢"), r2.text


async def test_f_reset_command():
    """F /reset 丢弃挂起的授权与历史，之后是全新对话。"""
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]), AIMessage(content="新会话回复")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE],
    )
    r = await cs.process_message("写个文件")
    assert r.awaiting_approval, r
    r = await cs.process_message("/reset")
    assert "已清空" in r.text, r.text
    st = await cs._agent.aget_state({"configurable": {"thread_id": cs.get_active_session_id()}})
    assert not hitl.has_pending_interrupts(st) and not st.values.get("messages"), st
    r = await cs.process_message("你好")
    assert r.text == "新会话回复" and _files(root) == [] and fam.idx == 1, (r.text, _files(root), fam.idx)


async def test_g_image_turn_no_memory():
    """G 图片轮不触发记忆提取。"""
    from app.infra import models
    models._cache["vlm"] = ScriptedModel(script=[AIMessage(content="分析图里的苹果")])
    root, fam, mm = await _install([AIMessage(content="看图回复")], [FA_DONE])
    mem_calls.clear()
    r = await cs.process_message("看这个图", image_url="data:image/png;base64,AAAA")
    await asyncio.sleep(0.05)
    assert r.text == "看图回复"
    assert len(mem_calls) == 0, f"图片轮不应触发记忆提取，实际: {mem_calls}"
    evs = _env.store.get_events(cs.get_active_session_id(), types=["user_message"])
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
    sid = cs.get_active_session_id()
    r1 = await cs.process_message("画图并写文件")
    assert r1.awaiting_approval and cam.idx == 2
    assert await _cp_rows(sid, nested=True) > 0, "挂起期间嵌套 checkpoint 必须保留，否则无法恢复"

    # ── 重启：全新的模型实例，脚本里只有“批准之后”才会用到的回复；若已完成的专家被重跑会立刻暴露 ──
    cam2 = ScriptedModel(script=[AIMessage(content="不应被调用")])
    fam2, mm2 = await _restart([AIMessage(content="图和文件都好了")], [FA_DONE], extra=[code_spec(cam2)])
    assert cs.get_active_session_id() == sid
    abandoned = await cs.recover_orphans()
    assert abandoned == 0, "挂起中的授权不受 recover_orphans 影响"
    ts = _env.store.load_turn_state(sid)
    assert ts is not None and len(ts.outbox) == 1 and ts.user_text == "画图并写文件", ts

    mem_calls.clear()
    r2 = await cs.process_message("y")
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
    sid1 = cs.get_active_session_id()
    r = await cs.process_message("写个文件")
    assert r.awaiting_approval

    r = await cs.process_message("/new 第二个会话")
    sid2 = cs.get_active_session_id()
    assert sid2 != sid1 and sid2[:8] in r.text and "第二个会话" in r.text, r.text
    assert sid1[:8] in r.text and "待授权" in r.text, "旧会话还有待授权操作，应附提醒"
    assert not r.awaiting_approval and reset_calls == [], "/new 不动沙箱"

    r = await cs.process_message("/sessions")
    lines = [ln for ln in r.text.splitlines() if ln[:2] in ("1.", "2.")]
    assert len(lines) == 2, r.text
    assert "[当前]" in lines[0] and sid2[:8] in lines[0], lines
    assert "[待授权]" in lines[1] and sid1[:8] in lines[1] and "[当前]" not in lines[1], lines
    assert "/switch" in r.text and "/new" in r.text

    r = await cs.process_message("第二会话专属消息")
    assert r.text == "会话二回复" and not r.awaiting_approval, r.text
    assert "写个文件" not in mm.seen[1], "第二个会话的模型输入里不应有第一个会话的内容"

    r = await cs.process_message("/switch 2")                          # 按 /sessions 的序号
    assert cs.get_active_session_id() == sid1 and r.awaiting_approval, r.text
    assert "write_file" in r.text and "workspace/a.txt" in r.text, "切到有待授权的会话时附上授权内容"

    r = await cs.process_message("Y")
    assert r.text == "写好了" and _files(root) == ["a.txt"], (r.text, _files(root))
    assert "第二会话专属消息" not in mm.seen[2], "第一个会话的模型输入里不应有第二个会话的内容"

    r = await cs.process_message(f"/switch {sid2[:8]}")                # 按 id 前缀
    assert cs.get_active_session_id() == sid2 and not r.awaiting_approval, r.text
    r = await cs.process_message(f"/switch {sid2[:8]}")
    assert "已经在" in r.text and cs.get_active_session_id() == sid2
    r = await cs.process_message("/switch zzzz")
    assert "没有找到" in r.text and cs.get_active_session_id() == sid2
    r = await cs.process_message("/switch")
    assert "用法" in r.text
    assert reset_calls == [], "/new、/switch 不触发沙箱重置"

    await cs.process_message("/reset")
    assert reset_calls == [1], "只有 /reset 重置沙箱"

    # 命令按首个空白分隔的词精确匹配；其余以 / 开头的文本（如路径）按普通消息处理
    assert cs._parse_command("/tmp/a.txt 里有什么") is None
    assert cs._parse_command("/resetx") is None and cs._parse_command("你好 /reset") is None
    assert cs._parse_command("/NEW 标题") == ("/new", "标题")
    assert cs._parse_command("/sessions 5") == ("/sessions", "5")


async def test_j_crash_leftover():
    """J 崩溃遗留：未处理的入站事件 → recover_orphans 标记放弃，不调用模型；之后新消息正常处理。"""
    root, fam, mm = await _install([AIMessage(content="正常回复")], [FA_DONE])
    sid = cs.get_active_session_id()
    store = _env.store
    seq = store.emit_event(sid, "user_message", {"text": "崩溃前的消息", "processed_text": "崩溃前的消息", "has_image": False})
    assert [e.seq for e in store.get_pending_events(sid)] == [seq]

    n = await cs.recover_orphans()
    assert n == 1 and store.get_pending_events(sid) == []
    ab = store.get_events(sid, types=["turn_abandoned"])
    assert len(ab) == 1 and ab[0].payload == {"reason": "process_restart", "event_seq": seq}, ab
    assert mm.idx == 0, "不自动重跑：模型不应被调用"

    r = await cs.process_message("新消息")
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
    cs.injector.build_initial_messages = lambda q, history_messages=None: [
        HumanMessage(content=f"<long_term_memory>\n我住台北\n</long_term_memory>\n{q}", name="user")
    ]
    try:
        r1 = await cs.process_message("你好")
        r2 = await cs.process_message("查个数据")
    finally:
        cs.injector.build_initial_messages = orig
    assert r1.text == "普通回复" and r2.text == "最终回复", (r1.text, r2.text)
    assert "long_term_memory" in mm.seen[0], "桩应当把记忆块真的拼进了模型输入（否则下面的断言没有意义）"

    sid = cs.get_active_session_id()
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
    st = await cs._agent.aget_state({"configurable": {"thread_id": sid}})
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
    sid = cs.get_active_session_id()
    r = await cs.process_message("写个文件")
    assert r.awaiting_approval
    assert await _cp_rows(sid, nested=True) > 0, "挂起期间嵌套 checkpoint 必须保留"
    root_before = await _cp_rows(sid, nested=False)

    r = await cs.process_message("Y")
    assert r.text == "写好了"
    assert await _cp_rows(sid, nested=True) == 0 and await _cp_rows(sid, nested=True, table="writes") == 0
    assert await _cp_rows(sid, nested=False) >= root_before > 0, "根命名空间必须保留"

    st = await cs._agent.aget_state({"configurable": {"thread_id": sid}})
    humans = [m.content for m in st.values["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["写个文件"], humans
    # 下一轮仍能看到上一轮的消息
    old = config.CHECKPOINT_PRUNE_NESTED
    config.CHECKPOINT_PRUNE_NESTED = False
    try:
        r = await cs.process_message("再写一个")
        assert r.awaiting_approval
        r = await cs.process_message("Y")
        assert r.text == "又写好了", r.text
        assert await _cp_rows(sid, nested=True) > 0, "CHECKPOINT_PRUNE_NESTED=False 时不清理"
    finally:
        config.CHECKPOINT_PRUNE_NESTED = old
    st = await cs._agent.aget_state({"configurable": {"thread_id": sid}})
    humans = [m.content for m in st.values["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["写个文件", "再写一个"], humans


async def test_m_reset_keeps_log():
    """M /reset 清上下文与挂起授权，但事件日志保留并追加 session_reset。"""
    root, fam, mm = await _install(
        [AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]), AIMessage(content="新对话回复")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE],
    )
    sid = cs.get_active_session_id()
    r = await cs.process_message("写个文件")
    assert r.awaiting_approval and _env.store.load_turn_state(sid) is not None

    r = await cs.process_message("/reset")
    assert "已清空" in r.text and "会话记录仍保留" in r.text, r.text
    assert cs.get_active_session_id() == sid, "/reset 不换会话"
    st = await cs._agent.aget_state({"configurable": {"thread_id": sid}})
    assert not hitl.has_pending_interrupts(st) and not st.values.get("messages"), st
    assert _env.store.load_turn_state(sid) is None
    assert await _cp_rows(sid, nested=False) == 0 and await _cp_rows(sid, nested=True) == 0

    types_ = [e.type for e in _env.store.get_events(sid, limit=200)]
    assert "user_message" in types_ and "approval_request" in types_ and types_[-1] == "session_reset", types_
    assert _env.store.get_pending_events(sid) == []

    r = await cs.process_message("你好")                                # 授权已被丢弃：这是一条全新的普通消息
    assert r.text == "新对话回复" and not r.awaiting_approval and _files(root) == [], (r.text, _files(root))


async def test_n_wake_idempotent():
    """N wake 幂等消费：已处理的事件不会被再次处理；直接 emit + wake 同样可用。"""
    root, fam, mm = await _install([AIMessage(content="回复一"), AIMessage(content="回复二")], [FA_DONE])
    sid = cs.get_active_session_id()
    r = await cs.process_message("你好")
    assert r.text == "回复一"
    calls = mm.idx
    assert await cs.wake(sid) is None and await cs.wake(sid) is None
    assert mm.idx == calls, "对同一会话重复 wake 不应再次调用模型"

    _env.store.emit_event(sid, "user_message", {"text": "直接写事件", "processed_text": "直接写事件", "has_image": False})
    r = await cs.wake(sid)
    assert r is not None and r.text == "回复二" and mm.idx == calls + 1, (r, mm.idx)
    assert await cs.wake(sid) is None and mm.idx == calls + 1


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
