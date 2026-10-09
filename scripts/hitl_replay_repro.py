"""
hitl_replay_repro.py —— 复现并验证 "interrupt() 恢复时整个节点从头重跑" 带来的重复执行问题。

背景：LangGraph 恢复中断时，会把被中断的那个【节点】从头重新执行一遍（interrupt() 立刻返回已有的恢复值）。
所以"同一个节点里先执行了有副作用的工具、后面才遇到 interrupt()"，副作用就会被重复执行。

OLD = 复制自 app/graph.py 的现有结构（节点内循环处理同一条 AIMessage 里的全部 tool_calls）
NEW = 重放安全结构（每次节点执行只处理 1 个待处理的 tool_call；路由按"尚未回复的 tool_call"决定下一步）

运行：python scripts/hitl_replay_repro.py      （只需要 langgraph / langchain-core，不联网、不碰 MCP、不碰桌面）
参考结果（langgraph 1.2.12 / langchain-core 1.6.6）：
  S1 OLD: write_file 3 次 / create_directory 2 次 / move_file 1 次；NEW: 各 1 次
  S2 OLD: edit_file 执行 2 次，第一次成功的结果被第二次的"找不到匹配"报错覆盖；NEW: 各 1 次
  S3 OLD: 混合调用里的安全工具 read_text_file 从未执行、也没有 ToolMessage；NEW: 正常
  Desktop OLD: click / safe_input_text 各执行 2 次；NEW: 各 1 次
注意：判断"是否还有待批准的中断"请看 state.tasks[*].interrupts，不要只看 state.next
（直接对子图调用时，第二个中断处我遇到过 next 为空但 tasks 里仍挂着 interrupt 的情况）。
"""

import asyncio, operator
from collections import Counter
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

DANGEROUS = {"write_file", "edit_file", "create_directory", "move_file"}


class FileAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_tool_call_id: str
    instruction_to_worker: str
    user_query: str


# ───────────────────────── OLD (copied structure) ─────────────────────────
def build_old(tools, llm, checkpointer=None):
    tool_map = {t.name: t for t in tools}

    async def fa_llm_node(state):
        resp = await llm.bind_tools(tools).ainvoke(state["messages"])
        return {"messages": [resp]}

    async def fa_safe_tools_node(state):
        last_ai = state["messages"][-1]
        out = []
        for tc in last_ai.tool_calls:
            if tc["name"] in DANGEROUS:
                continue
            tool = tool_map.get(tc["name"])
            try:
                result = str(await tool.ainvoke(tc["args"]))
            except Exception as e:
                result = f"工具执行出错: {e}"
            out.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": out}

    async def fa_dangerous_tools_node(state):
        last_ai = state["messages"][-1]
        out = []
        for tc in last_ai.tool_calls:
            if tc["name"] not in DANGEROUS:
                continue
            decision = interrupt({"tool_name": tc["name"], "tool_args": tc["args"], "tool_call_id": tc["id"]})
            if isinstance(decision, str) and decision.strip().upper() == "Y":
                try:
                    result = str(await tool_map[tc["name"]].ainvoke(tc["args"]))
                except Exception as e:
                    result = f"工具执行出错: {e}"
            else:
                result = "CRITICAL ERROR: USER PERMISSION DENIED."
            out.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": out}

    async def fa_summarize_node(state):
        return {"messages": [AIMessage(content="summary")]}

    def _route_after_llm(state):
        last_ai = state["messages"][-1]
        if not getattr(last_ai, "tool_calls", None):
            return "fa_summarize"
        for tc in last_ai.tool_calls:
            if tc["name"] in DANGEROUS:
                return "fa_dangerous_tools"
        return "fa_safe_tools"

    def _route_after_tools(state):
        return "fa_llm"

    g = StateGraph(FileAgentState)
    g.add_node("fa_llm", fa_llm_node)
    g.add_node("fa_safe_tools", fa_safe_tools_node)
    g.add_node("fa_dangerous_tools", fa_dangerous_tools_node)
    g.add_node("fa_summarize", fa_summarize_node)
    g.add_edge(START, "fa_llm")
    g.add_conditional_edges("fa_llm", _route_after_llm)
    g.add_conditional_edges("fa_safe_tools", _route_after_tools)
    g.add_conditional_edges("fa_dangerous_tools", _route_after_tools)
    g.add_edge("fa_summarize", END)
    return g.compile(checkpointer=checkpointer)


# ───────────────────────── NEW (replay-safe) ─────────────────────────
def _last_ai_with_calls(messages):
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            return i, m
    return -1, None


def _unanswered(messages):
    i, ai = _last_ai_with_calls(messages)
    if ai is None:
        return []
    answered = {m.tool_call_id for m in messages[i + 1:] if isinstance(m, ToolMessage)}
    return [tc for tc in ai.tool_calls if tc["id"] not in answered]


def build_new(tools, llm, checkpointer=None):
    tool_map = {t.name: t for t in tools}

    async def fa_llm_node(state):
        resp = await llm.bind_tools(tools).ainvoke(state["messages"])
        return {"messages": [resp]}

    async def fa_safe_tools_node(state):
        out = []
        for tc in _unanswered(state["messages"]):
            if tc["name"] in DANGEROUS:
                continue
            tool = tool_map.get(tc["name"])
            if tool is None:
                result = f"工具 '{tc['name']}' 未找到。"
            else:
                try:
                    result = str(await tool.ainvoke(tc["args"]))
                except Exception as e:
                    result = f"工具执行出错: {e}"
            out.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": out}

    async def fa_dangerous_tools_node(state):
        pending = [tc for tc in _unanswered(state["messages"]) if tc["name"] in DANGEROUS]
        tc = pending[0]                      # exactly ONE per node execution
        decision = interrupt({"tool_name": tc["name"], "tool_args": tc["args"], "tool_call_id": tc["id"]})
        if isinstance(decision, str) and decision.strip().upper() == "Y":
            try:
                result = str(await tool_map[tc["name"]].ainvoke(tc["args"]))
            except Exception as e:
                result = f"工具执行出错: {e}"
        else:
            result = "CRITICAL ERROR: USER PERMISSION DENIED."
        return {"messages": [ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"])]}

    async def fa_summarize_node(state):
        return {"messages": [AIMessage(content="summary")]}

    def _next_step(state):
        _, ai = _last_ai_with_calls(state["messages"])
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            return "fa_summarize"
        pending = _unanswered(state["messages"])
        if any(tc["name"] not in DANGEROUS for tc in pending):
            return "fa_safe_tools"
        if any(tc["name"] in DANGEROUS for tc in pending):
            return "fa_dangerous_tools"
        return "fa_llm"

    g = StateGraph(FileAgentState)
    g.add_node("fa_llm", fa_llm_node)
    g.add_node("fa_safe_tools", fa_safe_tools_node)
    g.add_node("fa_dangerous_tools", fa_dangerous_tools_node)
    g.add_node("fa_summarize", fa_summarize_node)
    g.add_edge(START, "fa_llm")
    g.add_conditional_edges("fa_llm", _next_step)
    g.add_conditional_edges("fa_safe_tools", _next_step)
    g.add_conditional_edges("fa_dangerous_tools", _next_step)
    g.add_edge("fa_summarize", END)
    return g.compile(checkpointer=checkpointer)


# ───────────────────────── harness ─────────────────────────
class ScriptedLLM:
    def __init__(self, scripted):
        self._it = iter(scripted)

    def bind_tools(self, tools, **kw):
        return self

    async def ainvoke(self, messages, **kw):
        try:
            return next(self._it)
        except StopIteration:
            return AIMessage(content="done")


def make_tools(counts, edit_once=False):
    def mk(name):
        def fn(**kwargs) -> str:
            counts[name] += 1
            if edit_once and name == "edit_file" and counts[name] > 1:
                raise RuntimeError("Could not find exact match for edit (oldText already replaced)")
            return f"{name} ok"
        return StructuredTool.from_function(func=fn, name=name, description=name, args_schema={"type": "object", "properties": {}})
    return [mk(n) for n in ["read_text_file", "write_file", "edit_file", "create_directory", "move_file"]]


def call(name, i):
    return {"name": name, "args": {}, "id": f"c{i}"}


async def scenario(build, title, calls, answers, edit_once=False):
    counts = Counter()
    tools = make_tools(counts, edit_once)
    llm = ScriptedLLM([AIMessage(content="", tool_calls=calls)])
    g = build(tools, llm, MemorySaver())
    cfg = {"configurable": {"thread_id": title}}
    init = {"messages": [HumanMessage(content="go")], "current_tool_call_id": "x", "instruction_to_worker": "", "user_query": ""}
    await g.ainvoke(init, cfg)
    asked, i = [], 0
    while True:
        st = await g.aget_state(cfg)
        pend = [it.value for t in st.tasks for it in t.interrupts]
        if not pend:
            break
        payload = pend[0]
        asked.append(payload["tool_name"])
        await g.ainvoke(Command(resume=answers[i]), cfg)
        i += 1
    msgs = (await g.aget_state(cfg)).values["messages"]
    tms = {m.tool_call_id: m.content for m in msgs if isinstance(m, ToolMessage)}
    return dict(executions=dict(counts), asked=asked, answered_ids=sorted(tms), tool_results=tms)


async def main_file():
    scenarios = [
        ("S1 三个危险调用并行 (write/mkdir/move)，全部批准",
         [call("write_file", 1), call("create_directory", 2), call("move_file", 3)], ["Y", "Y", "Y"], False),
        ("S2 非幂等: edit_file + move_file 并行，全部批准",
         [call("edit_file", 1), call("move_file", 2)], ["Y", "Y"], True),
        ("S3 混合: read(安全) + edit(危险) 并行",
         [call("read_text_file", 1), call("edit_file", 2)], ["Y"], False),
        ("S4 三个危险调用，第二个拒绝",
         [call("write_file", 1), call("edit_file", 2), call("move_file", 3)], ["Y", "N", "Y"], False),
    ]
    for title, calls, answers, edit_once in scenarios:
        print("=" * 78)
        print(title)
        for label, b in (("OLD", build_old), ("NEW", build_new)):
            try:
                r = await scenario(b, f"{title}-{label}", calls, answers, edit_once)
                print(f"  [{label}] 实际执行次数: {r['executions']}")
                print(f"  [{label}] 询问顺序    : {r['asked']}")
                print(f"  [{label}] 已回复的 tool_call_id: {r['answered_ids']}  (应为 {sorted(c['id'] for c in calls)})")
                for k, v in r["tool_results"].items():
                    print(f"        {k}: {str(v)[:70]}")
            except Exception as e:
                print(f"  [{label}] 异常: {type(e).__name__}: {str(e)[:120]}")




# ───────────────────────── DesktopAgent ─────────────────────────
DesktopAgentState = FileAgentState

HIGH = ["enter", "return", "delete", "alt", "win", "ctrl"]

def desktop_risky(tc):
    return tc["name"] == "press_hotkey" and any(k in tc.get("args", {}).get("keys", "").lower() for k in HIGH)

def build_desktop(kind, tools, llm, ckpt):
    tool_map = {t.name: t for t in tools}
    async def da_llm(state):
        return {"messages": [await llm.bind_tools(tools).ainvoke(state["messages"])]}

    async def run_one(tc):
        if desktop_risky(tc):
            d = interrupt(f"高危按键: {tc['args']}")
            if not (isinstance(d, str) and d.strip().upper() == "Y"):
                return ToolMessage(content="CRITICAL ERROR: USER PERMISSION DENIED.", tool_call_id=tc["id"], name=tc["name"])
        try:
            r = str(await tool_map[tc["name"]].ainvoke(tc["args"]))
        except Exception as e:
            r = f"工具执行出错: {e}"
        return ToolMessage(content=r, tool_call_id=tc["id"], name=tc["name"])

    async def da_tools_old(state):                     # copied structure: loop over ALL calls in one execution
        return {"messages": [await run_one(tc) for tc in state["messages"][-1].tool_calls]}

    async def da_tools_new(state):                     # exactly ONE unanswered call per execution
        return {"messages": [await run_one(_unanswered(state["messages"])[0])]}

    async def da_summarize(state):
        return {"messages": [AIMessage(content="summary")]}

    def after_llm(state):
        return "da_tools" if getattr(state["messages"][-1], "tool_calls", None) else "da_summarize"

    def after_tools_new(state):
        return "da_tools" if _unanswered(state["messages"]) else "da_llm"

    g = StateGraph(DesktopAgentState)
    g.add_node("da_llm", da_llm); g.add_node("da_summarize", da_summarize)
    g.add_node("da_tools", da_tools_old if kind == "OLD" else da_tools_new)
    g.add_edge(START, "da_llm"); g.add_conditional_edges("da_llm", after_llm)
    if kind == "OLD": g.add_edge("da_tools", "da_llm")
    else: g.add_conditional_edges("da_tools", after_tools_new)
    g.add_edge("da_summarize", END)
    return g.compile(checkpointer=ckpt)

def make_desktop_tools(counts):
    def mk(name):
        def fn(**kw) -> str:
            counts[name] += 1
            return f"{name} ok"
        return StructuredTool.from_function(func=fn, name=name, description=name, args_schema={"type": "object", "properties": {}})
    return [mk(n) for n in ["click_grid_center", "safe_input_text", "press_hotkey"]]

async def run_desktop(kind, calls, answers):
    counts = Counter()
    g = build_desktop(kind, make_desktop_tools(counts), ScriptedLLM([AIMessage(content="", tool_calls=calls)]), MemorySaver())
    cfg = {"configurable": {"thread_id": f"{kind}-{len(answers)}-{answers}"}}
    await g.ainvoke({"messages": [HumanMessage(content="go")], "current_tool_call_id": "x", "instruction_to_worker": "", "user_query": ""}, cfg)
    i = 0
    while True:
        st = await g.aget_state(cfg)
        if not [it for t in st.tasks for it in t.interrupts]: break
        await g.ainvoke(Command(resume=answers[i]), cfg); i += 1
    msgs = (await g.aget_state(cfg)).values["messages"]
    return dict(counts), sorted(m.tool_call_id for m in msgs if isinstance(m, ToolMessage))

async def main_desktop():
    calls = [{"name": "click_grid_center", "args": {"grid_id": "A1"}, "id": "c1"},
             {"name": "safe_input_text", "args": {"text": "hello"}, "id": "c2"},
             {"name": "press_hotkey", "args": {"keys": "ctrl+s"}, "id": "c3"}]
    for answers in (["Y"], ["N"]):
        print(f"[DesktopAgent] 一条消息里: 点击 + 输入文字 + ctrl+s(高危)，用户回答 {answers}")
        for kind in ("OLD", "NEW"):
            counts, ids = await run_desktop(kind, calls, answers)
            print(f"   {kind}: 实际执行次数 {counts} | 已回复 id {ids}")



async def main():
    print("#" * 78); print("# FileAgent"); print("#" * 78)
    await main_file()
    print(); print("#" * 78); print("# DesktopAgent (假工具，不碰真实桌面)"); print("#" * 78)
    await main_desktop()

if __name__ == "__main__":
    asyncio.run(main())
