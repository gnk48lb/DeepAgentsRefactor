import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 设置标准输出编码为 utf-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from app import database, deep_agent, outbox

async def test_subgraphs():
    database.run_global_database_init()
    agent = await deep_agent.build_main_agent()
    outbox.begin()

    query = "请用 matplotlib 画 x=[1,2,3], y=[1,4,9] 的折线图并保存"
    messages = deep_agent.build_initial_messages(user_query=query, user_id="test_user")
    inputs = {"messages": messages}
    config_dict = {
        "recursion_limit": 30,
        "configurable": {"thread_id": "test_subgraphs_trace"},
    }

    print("\n--- 监听 subgraphs=True 的 astream 事件 ---")
    async for item in agent.astream(inputs, config=config_dict, subgraphs=True):
        # item 通常是 (namespace, event) 元组
        namespace = item[0] if isinstance(item, tuple) else None
        event = item[1] if isinstance(item, tuple) else item
        if not isinstance(event, dict):
            continue
        for node_name, state_update in event.items():
            if not isinstance(state_update, dict):
                continue
            msgs = state_update.get("messages", [])
            if not isinstance(msgs, list):
                msgs = [msgs]
            for m in msgs:
                tool_calls = getattr(m, "tool_calls", None) or []
                if tool_calls:
                    for tc in tool_calls:
                        tc_name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                        print(f"[astream (ns={namespace}, node={node_name})] tool_call: {tc_name}")

if __name__ == "__main__":
    asyncio.run(test_subgraphs())
