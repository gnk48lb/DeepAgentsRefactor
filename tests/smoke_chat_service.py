"""
tests/smoke_chat_service.py

chat_service 服务层离线冒烟测试：用脚本化假模型驱动真实的 FileAgent / hitl / outbox / chat_service，
覆盖：多轮 HITL（批准 / 拒绝 / 挂起期间发无关消息）、HITL 前已产生图片的发件箱连续性、
并发忙提示、并行中断一次 Y 放行、超时后会话仍可继续、/reset 丢弃挂起授权与历史。
不需要任何 API Key / Docker / MCP / 网络：orchestrator / injector / extractor / startup 在 import 前被桩掉。
"""
import asyncio
import operator
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path
from typing import Annotated, List
from typing_extensions import TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")



    

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
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


async def _fake_extract_memory(recent_chat):
    mem_calls.append(recent_chat)


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
from app.agents.experts import file_agent
from app.runtime import hitl, outbox


class ScriptedModel(BaseChatModel):
    """按脚本依次返回 AIMessage；delay 用来模拟慢模型。"""
    script: List[AIMessage]
    idx: int = 0
    delay: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
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
        excluded_tools=frozenset({"write_file", "edit_file", "delete", "execute"}),
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


def _install(main_script, fa_script, extra=None, delay=0.0):
    """装配：真实 FileAgent + 脚本化主模型 + MemorySaver，并把它塞进 chat_service。"""
    root = tempfile.mkdtemp(prefix="ws_")
    fam = ScriptedModel(script=fa_script)
    mm = ScriptedModel(script=main_script, delay=delay)
    subs = [CompiledSubAgent(name="FileAgent", description="f",
                             runnable=file_agent.build_file_agent_runnable(fam, "x", root))]
    subs += extra or []
    cp = MemorySaver()
    agent = create_deep_agent(model=mm, subagents=subs, checkpointer=cp)
    cs._agent, cs._initialized, cs._pending_outbox = agent, True, None
    if hasattr(cs, "_checkpointer"):
        cs._checkpointer = cp
    mem_calls.clear()
    return root, fam, mm


def _files(root):
    return sorted(p.name for p in Path(root).rglob("*") if p.is_file())


# ══════════════════════════════════════════════════════════════════════════════

async def test_a_multi_turn_hitl():
    """A 多轮：普通消息 → 写文件挂起 → 无关消息只重复提示 → 小写 y 批准 → 再次挂起 → N 拒绝。"""
    root, fam, mm = _install(
        [AIMessage(content="你好呀"), AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]),
         AIMessage(content="写好了"), AIMessage(content="", tool_calls=[_task("FileAgent", "t2")]),
         AIMessage(content="已取消")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE,
         AIMessage(content="", tool_calls=[_write("workspace/b.txt", "c2")]), FA_DONE],
    )
    r = await cs.process_message("你好")
    await asyncio.sleep(0.05)
    assert r.text == "你好呀" and not r.awaiting_approval, r
    assert mem_calls and "human: 你好" in mem_calls[0], mem_calls

    r = await cs.process_message("写个文件")
    assert r.awaiting_approval and "write_file" in r.text and "workspace/a.txt" in r.text and r.images == [], r.text

    idx_before = fam.idx
    r = await cs.process_message("等等我想想")
    assert r.awaiting_approval and "当前正在等待授权确认" in r.text and fam.idx == idx_before, r.text

    r = await cs.process_message("y")
    assert (not r.awaiting_approval) and r.text == "写好了" and _files(root) == ["a.txt"], (r.text, _files(root))

    r = await cs.process_message("再写一个")
    assert r.awaiting_approval and "workspace/b.txt" in r.text, r.text

    r = await cs.process_message("N")
    assert _files(root) == ["a.txt"] and r.text == "已取消" and fam.idx == 4, (_files(root), r.text, fam.idx)
    assert not cs._chat_lock.locked() and cs._pending_outbox is None


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
    root, fam, mm = _install(
        [AIMessage(content="", tool_calls=[_task("CodeAgent", "t1"), _task("FileAgent", "t2")]),
         AIMessage(content="图和文件都好了")],
        [AIMessage(content="", tool_calls=[_write("workspace/p.txt", "c1")]), FA_DONE],
        extra=[code_spec],
    )
    r1 = await cs.process_message("画图并写文件")
    assert r1.awaiting_approval and cs._pending_outbox is not None and len(cs._pending_outbox) == 1, r1
    r2 = await cs.process_message("Y")
    assert len(r2.images) == 1 and r2.text == "图和文件都好了" and cam.idx == 2, (len(r2.images), r2.text, cam.idx)


async def test_c_busy_lock():
    """C 处理中再来一条 → 立即返回忙提示，第一条不受影响。"""
    root, fam, mm = _install([AIMessage(content="慢回复")], [FA_DONE], delay=0.6)
    t = asyncio.create_task(cs.process_message("慢"))
    await asyncio.sleep(0.2)
    rb = await cs.process_message("插队")
    ra = await t
    assert "还在处理中" in rb.text and ra.text == "慢回复" and not cs._chat_lock.locked(), (rb.text, ra.text)


async def test_d_parallel_interrupts():
    """D 主 Agent 同时派 FileAgent + DesktopAgent：两类授权一起提示，一次 Y 同时放行。"""
    desk = CompiledSubAgent(name="DesktopAgent", description="d", runnable=_desktop_like())
    root, fam, mm = _install(
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
    """E 单轮超时后给出友好提示，且同一个固定会话仍可继续对话。"""
    root, fam, mm = _install([AIMessage(content="很慢"), AIMessage(content="恢复了")], [FA_DONE], delay=1.2)
    old = config.AGENT_TURN_TIMEOUT_SECONDS
    config.AGENT_TURN_TIMEOUT_SECONDS = 0.3
    try:
        r = await cs.process_message("会超时的问题")
    finally:
        config.AGENT_TURN_TIMEOUT_SECONDS = old
    mm.delay = 0.0
    await asyncio.sleep(1.3)  # 等后台线程里被取消的那次模型调用自然结束
    r2 = await cs.process_message("超时之后的新问题")
    assert "我先停下了" in r.text, r.text
    assert r2.text in ("恢复了", "很慢"), r2.text


async def test_f_reset_command():
    """F /reset 丢弃挂起的授权与历史，之后是全新会话。"""
    root, fam, mm = _install(
        [AIMessage(content="", tool_calls=[_task("FileAgent", "t1")]), AIMessage(content="新会话回复")],
        [AIMessage(content="", tool_calls=[_write("workspace/a.txt", "c1")]), FA_DONE],
    )
    r = await cs.process_message("写个文件")
    assert r.awaiting_approval, r
    r = await cs.process_message("/reset")
    assert "已清空" in r.text, r.text
    st = await cs._agent.aget_state({"configurable": {"thread_id": cs.THREAD_ID}})
    assert not hitl.has_pending_interrupts(st) and not st.values.get("messages"), st
    r = await cs.process_message("你好")
    assert r.text == "新会话回复" and _files(root) == [] and fam.idx == 1, (r.text, _files(root), fam.idx)


SCENARIOS = [
    ("A 多轮 HITL（批准/无关消息/拒绝）", test_a_multi_turn_hitl),
    ("B HITL 前的图片随最终回复发出", test_b_outbox_continuity),
    ("C 并发忙提示", test_c_busy_lock),
    ("D 并行中断一次 Y 放行", test_d_parallel_interrupts),
    ("E 超时后会话可继续", test_e_timeout_recovery),
    ("F /reset 丢弃挂起的授权与历史", test_f_reset_command),
]


def main():
    print("=" * 70)
    print("开始运行 chat_service 服务层离线冒烟测试...")
    print("=" * 70)
    passed = failed = 0
    for name, fn in SCENARIOS:
        try:
            asyncio.run(fn())
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
