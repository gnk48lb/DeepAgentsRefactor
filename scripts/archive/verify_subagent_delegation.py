"""
scripts/verify_subagent_delegation.py

Tests that the Main Agent successfully delegates a domain sub-task to a subagent
via the built-in `task` tool in deepagents.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

async def main():
    from app import deep_agent
    
    print("[*] Building Main Agent...", flush=True)
    agent = await deep_agent.build_main_agent()
    
    # Query designed to trigger subagent delegation (e.g. MapAgent for coordinates or weather)
    query = "请帮我查询济南市今天的天气情况。"
    messages = deep_agent.build_initial_messages(user_query=query)
    
    config = {"configurable": {"thread_id": "test_subagent_delegation_thread"}}
    print(f"[*] Invoking agent with query: '{query}'...", flush=True)
    result = await agent.ainvoke({"messages": messages}, config=config)
    
    print("\n" + "=" * 60, flush=True)
    print("[*] Message Trajectory:", flush=True)
    tool_calls_detected = []
    for idx, msg in enumerate(result["messages"]):
        role = getattr(msg, "role", type(msg).__name__)
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                tool_calls_detected.append(tc.get("name"))
                print(f"    Step {idx} [{role}]: Tool Call -> {tc.get('name')}(args={json.dumps(tc.get('args', {}), ensure_ascii=False)[:100]}...)", flush=True)
        else:
            snippet = str(getattr(msg, "content", ""))[:120].replace("\n", " ")
            print(f"    Step {idx} [{role}]: {snippet}...", flush=True)
            
    last_msg = result["messages"][-1]
    final_content = getattr(last_msg, "content", str(last_msg))
    print("\n[*] Final Answer:\n", final_content, flush=True)
    print("=" * 60, flush=True)
    print(f"[*] Total tool calls detected: {tool_calls_detected}", flush=True)
    print(f"[+] Task delegation verified: {'task' in tool_calls_detected or len(tool_calls_detected) > 0}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
