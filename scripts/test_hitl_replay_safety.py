import asyncio
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from app import graph


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


def make_file_tools(counts, edit_throw_on_second=False):
    def mk(name):
        def fn(**kwargs) -> str:
            counts[name] += 1
            if edit_throw_on_second and name == "edit_file" and counts[name] > 1:
                raise RuntimeError("Could not find exact match for edit (oldText already replaced)")
            return f"{name} ok"
        return StructuredTool.from_function(func=fn, name=name, description=name, args_schema={"type": "object", "properties": {}})
    return [mk(n) for n in ["read_text_file", "write_file", "edit_file", "create_directory", "move_file"]]


def make_desktop_tools(counts):
    def mk(name):
        def fn(**kwargs) -> str:
            counts[name] += 1
            return f"{name} ok"
        return StructuredTool.from_function(func=fn, name=name, description=name, args_schema={"type": "object", "properties": {}})
    return [mk(n) for n in ["click_grid_center", "safe_input_text", "press_hotkey"]]


async def run_subgraph(subgraph, initial_input, config, answers):
    await subgraph.ainvoke(initial_input, config)
    asked_tools = []
    ans_idx = 0
    while True:
        state = await subgraph.aget_state(config)
        interrupts = []
        for t in state.tasks:
            if hasattr(t, "interrupts") and t.interrupts:
                interrupts.extend(t.interrupts)
        if not interrupts:
            break
        payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        if isinstance(payload, dict):
            asked_tools.append(payload.get("tool_name"))
        else:
            asked_tools.append(str(payload))
        resume_val = answers[ans_idx]
        ans_idx += 1
        await subgraph.ainvoke(Command(resume=resume_val), config)

    final_state = await subgraph.aget_state(config)
    msgs = final_state.values.get("messages", [])
    # 提取所有工具调用消息（过滤掉最后的总结 ToolMessage）
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage) and m.name != "FileAgent" and m.name != "DesktopAgent"]
    return asked_tools, tool_msgs


def check_tool_messages(tool_msgs, expected_call_ids):
    # 每个 tool_call_id 恰有一条 ToolMessage
    ids = [m.tool_call_id for m in tool_msgs]
    id_counts = Counter(ids)
    for cid in expected_call_ids:
        assert id_counts[cid] == 1, f"tool_call_id {cid} 应该恰有 1 条 ToolMessage，实际: {id_counts[cid]}"
    assert len(ids) == len(expected_call_ids), f"预期 ToolMessage 数量 {len(expected_call_ids)}，实际: {len(ids)}"


async def test_fa_1():
    print("=" * 60)
    print("Running FA-1: [write_file, create_directory, move_file], 全部 Y")
    counts = Counter()
    tools = make_file_tools(counts)
    calls = [
        {"name": "write_file", "args": {"path": "a.txt"}, "id": "c1"},
        {"name": "create_directory", "args": {"path": "dir"}, "id": "c2"},
        {"name": "move_file", "args": {"src": "a", "dst": "b"}, "id": "c3"},
    ]
    llm = ScriptedLLM([AIMessage(content="", tool_calls=calls)])
    subgraph = graph.build_file_agent_subgraph(tools, "prompt", llm=llm, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "fa-1"}}
    asked, tool_msgs = await run_subgraph(
        subgraph,
        {"messages": [HumanMessage(content="go")], "current_tool_call_id": "call_root", "instruction_to_worker": "go", "user_query": "go"},
        cfg,
        ["Y", "Y", "Y"]
    )
    print(f"  执行次数: {dict(counts)}")
    print(f"  询问顺序: {asked}")
    print(f"  回复 ToolMessage IDs: {[m.tool_call_id for m in tool_msgs]}")
    assert counts == {"write_file": 1, "create_directory": 1, "move_file": 1}, f"执行次数不符合预期: {counts}"
    assert asked == ["write_file", "create_directory", "move_file"], f"询问顺序不符合预期: {asked}"
    check_tool_messages(tool_msgs, ["c1", "c2", "c3"])
    print("  -> FA-1 PASS")


async def test_fa_2():
    print("=" * 60)
    print("Running FA-2: [edit_file, move_file] 全部 Y, edit_file 第二次调用抛异常")
    counts = Counter()
    tools = make_file_tools(counts, edit_throw_on_second=True)
    calls = [
        {"name": "edit_file", "args": {"path": "a.txt"}, "id": "c1"},
        {"name": "move_file", "args": {"src": "a", "dst": "b"}, "id": "c2"},
    ]
    llm = ScriptedLLM([AIMessage(content="", tool_calls=calls)])
    subgraph = graph.build_file_agent_subgraph(tools, "prompt", llm=llm, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "fa-2"}}
    asked, tool_msgs = await run_subgraph(
        subgraph,
        {"messages": [HumanMessage(content="go")], "current_tool_call_id": "call_root", "instruction_to_worker": "go", "user_query": "go"},
        cfg,
        ["Y", "Y"]
    )
    print(f"  执行次数: {dict(counts)}")
    print(f"  询问顺序: {asked}")
    assert counts == {"edit_file": 1, "move_file": 1}, f"执行次数不符合预期: {counts}"
    check_tool_messages(tool_msgs, ["c1", "c2"])
    c1_msg = [m for m in tool_msgs if m.tool_call_id == "c1"][0]
    print(f"  edit_file 结果: {c1_msg.content}")
    assert c1_msg.content == "edit_file ok", f"edit_file 应该成功，但得到: {c1_msg.content}"
    print("  -> FA-2 PASS")


async def test_fa_3():
    print("=" * 60)
    print("Running FA-3: [read_text_file(安全), edit_file(危险)], 回答 Y")
    counts = Counter()
    tools = make_file_tools(counts)
    calls = [
        {"name": "read_text_file", "args": {"path": "a.txt"}, "id": "c1"},
        {"name": "edit_file", "args": {"path": "b.txt"}, "id": "c2"},
    ]
    llm = ScriptedLLM([AIMessage(content="", tool_calls=calls)])
    subgraph = graph.build_file_agent_subgraph(tools, "prompt", llm=llm, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "fa-3"}}
    asked, tool_msgs = await run_subgraph(
        subgraph,
        {"messages": [HumanMessage(content="go")], "current_tool_call_id": "call_root", "instruction_to_worker": "go", "user_query": "go"},
        cfg,
        ["Y"]
    )
    print(f"  执行次数: {dict(counts)}")
    print(f"  询问顺序: {asked}")
    assert counts == {"read_text_file": 1, "edit_file": 1}, f"执行次数不符合预期: {counts}"
    assert asked == ["edit_file"], f"只有 edit_file 应该触发询问，实际: {asked}"
    check_tool_messages(tool_msgs, ["c1", "c2"])
    print("  -> FA-3 PASS")


async def test_fa_4():
    print("=" * 60)
    print("Running FA-4: [write_file, edit_file, move_file], 回答 Y, N, Y")
    counts = Counter()
    tools = make_file_tools(counts)
    calls = [
        {"name": "write_file", "args": {"path": "a.txt"}, "id": "c1"},
        {"name": "edit_file", "args": {"path": "b.txt"}, "id": "c2"},
        {"name": "move_file", "args": {"src": "a", "dst": "b"}, "id": "c3"},
    ]
    llm = ScriptedLLM([AIMessage(content="", tool_calls=calls)])
    subgraph = graph.build_file_agent_subgraph(tools, "prompt", llm=llm, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "fa-4"}}
    asked, tool_msgs = await run_subgraph(
        subgraph,
        {"messages": [HumanMessage(content="go")], "current_tool_call_id": "call_root", "instruction_to_worker": "go", "user_query": "go"},
        cfg,
        ["Y", "N", "Y"]
    )
    print(f"  执行次数: {dict(counts)}")
    print(f"  询问顺序: {asked}")
    assert counts == {"write_file": 1, "move_file": 1}, f"edit_file 被拒应不执行，实际: {counts}"
    check_tool_messages(tool_msgs, ["c1", "c2", "c3"])
    c2_msg = [m for m in tool_msgs if m.tool_call_id == "c2"][0]
    print(f"  c2(edit_file) 回复: {c2_msg.content[:50]}...")
    assert "USER PERMISSION DENIED" in c2_msg.content, f"拒绝操作应包含 USER PERMISSION DENIED"
    print("  -> FA-4 PASS")


async def test_da_1():
    print("=" * 60)
    print("Running DA-1: [click_grid_center, safe_input_text, press_hotkey(keys='ctrl+s')]")
    for ans in (["Y"], ["N"]):
        print(f"  --- 测试输入 {ans} ---")
        counts = Counter()
        tools = make_desktop_tools(counts)
        calls = [
            {"name": "click_grid_center", "args": {"grid_id": "A1"}, "id": "c1"},
            {"name": "safe_input_text", "args": {"text": "hello"}, "id": "c2"},
            {"name": "press_hotkey", "args": {"keys": "ctrl+s"}, "id": "c3"},
        ]
        llm = ScriptedLLM([AIMessage(content="", tool_calls=calls)])
        subgraph = graph.build_desktop_agent_subgraph(tools, "prompt", llm=llm, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": f"da-1-{ans[0]}"}}
        asked, tool_msgs = await run_subgraph(
            subgraph,
            {"messages": [HumanMessage(content="go")], "current_tool_call_id": "call_root", "instruction_to_worker": "go", "user_query": "go"},
            cfg,
            ans
        )
        print(f"  执行次数: {dict(counts)}")
        print(f"  询问拦截: {len(asked)} 次")
        check_tool_messages(tool_msgs, ["c1", "c2", "c3"])
        if ans == ["Y"]:
            assert counts == {"click_grid_center": 1, "safe_input_text": 1, "press_hotkey": 1}, f"Y 时三个工具各1次，实际: {counts}"
            c3_msg = [m for m in tool_msgs if m.tool_call_id == "c3"][0]
            assert c3_msg.content == "press_hotkey ok"
        else:
            assert counts == {"click_grid_center": 1, "safe_input_text": 1}, f"N 时press_hotkey为0次，实际: {counts}"
            c3_msg = [m for m in tool_msgs if m.tool_call_id == "c3"][0]
            assert "USER PERMISSION DENIED" in c3_msg.content
    print("  -> DA-1 PASS")


async def main():
    print("############################################################")
    print("Starting HITL Replay Safety Unit Tests")
    print("############################################################")
    await test_fa_1()
    await test_fa_2()
    await test_fa_3()
    await test_fa_4()
    await test_da_1()
    print("############################################################")
    print("ALL UNIT TESTS PASSED!")
    print("############################################################")


if __name__ == "__main__":
    asyncio.run(main())
