import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 设置标准输出编码为 utf-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from app import database, deep_agent, outbox
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                t = p.get("text", "")
                parts.append(t if isinstance(t, str) else str(t))
        return "".join(parts)
    return "" if content is None else str(content)

async def run_query(query: str, agent, thread_id: str):
    config_dict = {
        "recursion_limit": 30,
        "configurable": {"thread_id": thread_id},
    }
    outbox.begin()
    messages = deep_agent.build_initial_messages(
        user_query=query,
        user_id="test_user",
    )
    inputs = {"messages": messages}
    print(f"\n==================================================", flush=True)
    print(f"--- 🚀 Running DeepAgent for: '{query}' ---", flush=True)
    print(f"==================================================", flush=True)

    final_text = ""
    async for event in agent.astream(inputs, config=config_dict):
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
                m_type = type(m).__name__
                tool_calls = getattr(m, "tool_calls", None) or []
                content = getattr(m, "content", "")
                text = extract_text(content)

                if node_name == "model" and m_type == "AIMessage":
                    if tool_calls:
                        for tc in tool_calls:
                            tc_name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                            tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                            if tc_name == "task":
                                subagent = tc_args.get("subagent_type", "?")
                                desc = tc_args.get("description", "")
                                print(f"\n🔵 [派发任务给 {subagent}]: {desc}", flush=True)
                            else:
                                args_str = json.dumps(tc_args, ensure_ascii=False)[:150]
                                print(f"\n🔧 [调用工具 {tc_name}]: {args_str}", flush=True)
                    elif text.strip():
                        final_text = text
                        print(f"\n🌟 [最终回答]:\n{text}\n", flush=True)
                elif node_name == "tools" and m_type == "ToolMessage":
                    tool_name = getattr(m, "name", "unknown")
                    disp = text[:200] + ("..." if len(text) > 200 else "")
                    if tool_name == "task":
                        print(f"\n🟡 [专家战报]:\n{disp}", flush=True)
                    else:
                        print(f"\n🟢 [{tool_name} 返回]:\n{disp}", flush=True)

    # 检查发件箱
    lst = outbox._outbox.get()
    print(f"📦 [发件箱状态]: 收集到 {len(lst) if lst is not None else 0} 张图片", flush=True)
    return final_text

async def main():
    database.run_global_database_init()
    
    print("\n--- 插入记忆 ---")
    database.insert_memory("用户家住在济南市历下区文化东路42号")
    print("已插入记忆: 用户家住在济南市历下区文化东路42号")

    agent = await deep_agent.build_main_agent()

    # 验证 1: 菜市场查询 (验证 B2-1: description 带上具体地址)
    await run_query("我家附近有菜市场吗", agent, "test_market_001")

    # 验证 2: 济南天气查询 (验证仍路由给 MapAgent，且最终回答纯文本无 markdown)
    await run_query("济南今天天气怎么样", agent, "test_weather_001")

    # 验证 3: CodeAgent 画图 (验证 B2-2, B2-3 纯文本、无 markdown 图片语法)
    await run_query("帮我画一张折线图，x=[1,2,3], y=[1,4,9]", agent, "test_chart_001")

if __name__ == "__main__":
    asyncio.run(main())
