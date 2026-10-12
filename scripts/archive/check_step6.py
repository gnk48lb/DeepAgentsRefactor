import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import deep_agent, database, mcp_service
from app.tools import rag
from langchain_core.messages import HumanMessage


async def check_a():
    print("=" * 70)
    print("Step 6(a): 工具盘点 (Main Agent + Subagents)")
    print("=" * 70)
    agent = await deep_agent.build_main_agent()
    
    # 1. 主 Agent 工具盘点
    tools_node = agent.nodes.get("tools")
    main_tools = []
    if tools_node:
        t_bound = getattr(tools_node, "tools_by_name", None) or getattr(tools_node, "bound", None)
        if hasattr(t_bound, "tools_by_name"):
            main_tools = list(t_bound.tools_by_name.keys())
        elif isinstance(t_bound, dict):
            main_tools = list(t_bound.keys())
        elif hasattr(tools_node, "tools"):
            main_tools = [t.name for t in tools_node.tools]
    print(f"\n[Main Agent] 拥有的工具 ({len(main_tools)} 个):")
    for t in sorted(main_tools):
        print(f"  - {t}")

    # 2. 从 task 工具提取子 Agent 的 graph 并盘点其工具
    task_tool = None
    if tools_node and hasattr(tools_node, "tools_by_name"):
        task_tool = tools_node.tools_by_name.get("task")
    elif tools_node and hasattr(tools_node, "bound") and hasattr(tools_node.bound, "tools_by_name"):
        task_tool = tools_node.bound.tools_by_name.get("task")

    subagent_graphs = {}
    if task_tool:
        coro = getattr(task_tool, "coroutine", None) or getattr(task_tool, "func", None)
        if coro and hasattr(coro, "__code__") and hasattr(coro, "__closure__") and coro.__closure__:
            closure_dict = dict(zip(coro.__code__.co_freevars, [c.cell_contents for c in coro.__closure__]))
            if "subagent_graphs" in closure_dict:
                subagent_graphs = closure_dict["subagent_graphs"]

    print(f"\n[Subagents] 识别到 {len(subagent_graphs)} 个子 Agent graph:")
    builtin_tools_set = {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "delete", "execute", "write_todos"}

    subagent_tools_map = {}
    for name, sub_g in sorted(subagent_graphs.items()):
        sub_tools = []
        if hasattr(sub_g, "nodes") and "tools" in sub_g.nodes:
            s_tn = sub_g.nodes["tools"]
            s_bound = getattr(s_tn, "tools_by_name", None) or getattr(s_tn, "bound", None)
            if hasattr(s_bound, "tools_by_name"):
                sub_tools = list(s_bound.tools_by_name.keys())
            elif isinstance(s_bound, dict):
                sub_tools = list(s_bound.keys())
            elif hasattr(s_tn, "tools"):
                sub_tools = [t.name for t in s_tn.tools]
        subagent_tools_map[name] = sub_tools
        print(f"\n  * {name} ({len(sub_tools)} 个工具):")
        for st in sorted(sub_tools):
            is_builtin = " [deepagents 内置]" if st in builtin_tools_set else ""
            print(f"      - {st}{is_builtin}")

    # 特别报告
    print("\n" + "-" * 70)
    print("【特别报告：deepagents 内置工具出现情况】")
    print(f"内置候选列表: {sorted(builtin_tools_set)}")
    main_builtins = [t for t in main_tools if t in builtin_tools_set]
    print(f"主 Agent 包含的内置工具: {main_builtins if main_builtins else '无'}")
    for name, t_list in subagent_tools_map.items():
        sub_builtins = [t for t in t_list if t in builtin_tools_set]
        print(f"子 Agent [{name}] 包含的内置工具: {sub_builtins if sub_builtins else '无'}")
    print("-" * 70)
    return agent


async def check_b():
    print("\n" + "=" * 70)
    print("Step 6(b): 直接调用 rag 查 '火神战姬 Q技能'")
    print("=" * 70)
    res_rag = rag.invoke({"query": "火神战姬 Q技能"})
    print(f"rag 返回类型: {type(res_rag)}")
    if isinstance(res_rag, list):
        for idx, part in enumerate(res_rag):
            if isinstance(part, dict) and part.get("type") == "text":
                print(f"\n--- Part {idx} [Text Content] ---\n{part.get('text')}")
            elif isinstance(part, dict) and part.get("type") == "image_url":
                b64_url = part.get("image_url", {}).get("url", "")
                print(f"\n--- Part {idx} [Image URL] --- (base64 length: {len(b64_url)})")
    else:
        print(f"原始返回:\n{res_rag}")


async def check_c(agent):
    print("\n" + "=" * 70)
    print("Step 6(c): 通过主 Agent 问 '火神战姬的技能有哪些？'")
    print("=" * 70)
    query = "火神战姬的技能有哪些？"
    cfg = {"configurable": {"thread_id": "test_huoshen_read_check"}}
    inputs = {"messages": deep_agent.build_initial_messages(user_query=query, user_id="console_user")}
    
    final_text = ""
    async for event in agent.astream(inputs, config=cfg):
        if not isinstance(event, dict):
            continue
        for node_name, state_update in event.items():
            if "Middleware" in node_name or node_name.startswith("__"):
                continue
            if not isinstance(state_update, dict):
                continue
            msgs = state_update.get("messages", [])
            if not isinstance(msgs, list):
                msgs = [msgs]
            for m in msgs:
                if node_name == "model" and type(m).__name__ == "AIMessage":
                    tc = getattr(m, "tool_calls", None) or []
                    if not tc and getattr(m, "content", ""):
                        from app.utils import extract_text
                        final_text = extract_text(m.content)

    print("\n================== [FINAL_ANSWER_FULL] ==================")
    print(final_text)
    print("================== [END_FINAL_ANSWER] ==================\n")


async def main():
    database.run_global_database_init()
    agent = await check_a()
    await check_b()
    await check_c(agent)


if __name__ == "__main__":
    asyncio.run(main())
