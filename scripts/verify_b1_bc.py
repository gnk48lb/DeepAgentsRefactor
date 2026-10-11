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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def count_image_urls(messages):
    count = 0
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part):
                    count += 1
        elif isinstance(content, dict):
            if content.get("type") == "image_url" or "image_url" in content:
                count += 1
    return count

async def main():
    print("=== 开始执行 B1(b)/(c) 完整验证 ===")
    database.run_global_database_init()
    agent = await deep_agent.build_main_agent()

    thread_id = "test_verify_b1_bc_final"
    config_dict = {
        "recursion_limit": 30,
        "configurable": {"thread_id": thread_id},
    }

    # 1. 在同一个协程里调用 outbox.begin()
    outbox.begin()
    query = "请用 matplotlib 画 x=[1,2,3], y=[1,4,9] 的折线图并 savefig 到 /workspace/outputs/a.png"
    messages = deep_agent.build_initial_messages(user_query=query, user_id="test_user")
    inputs = {"messages": messages}

    print(f"\n--- 🚀 Running Main Agent with query: '{query}' ---")

    final_text = ""
    astream_tool_calls = []

    # 2. 遍历 astream 事件（开启 subgraphs=True 以捕获子图内所有事件）
    async for item in agent.astream(inputs, config=config_dict, subgraphs=True):
        namespace = item[0] if isinstance(item, tuple) else ()
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
                m_type = type(m).__name__
                tool_calls = getattr(m, "tool_calls", None) or []
                content = getattr(m, "content", "")

                if m_type == "AIMessage" and tool_calls:
                    for tc in tool_calls:
                        tc_name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                        raw_line = f"[astream (ns={namespace}, node={node_name})] tool_call: {tc_name}"
                        print(raw_line)
                        astream_tool_calls.append((namespace, node_name, tc_name))

                if m_type == "AIMessage" and not tool_calls and not namespace:
                    text_str = deep_agent.extract_text(content)
                    if text_str.strip():
                        final_text = text_str

    # 3. 统计发件箱张数和 base64 长度
    outbox_list = outbox._outbox.get() or []
    print(f"\n--- 发件箱统计 (outbox) ---")
    print(f"发件箱张数: {len(outbox_list)}")
    debug_dir = os.path.join("storage", "outbox_debug")
    os.makedirs(debug_dir, exist_ok=True)
    for idx, item in enumerate(outbox_list):
        url = item.get("image_url", {}).get("url", "")
        if url.startswith("data:image/jpeg;base64,"):
            b64_str = url[len("data:image/jpeg;base64,"):]
        elif "base64," in url:
            b64_str = url.split("base64,", 1)[1]
        else:
            b64_str = url
        print(f"  图片 #{idx+1}: base64 长度 = {len(b64_str)}")
        try:
            import base64
            img_bytes = base64.b64decode(b64_str)
            save_path = os.path.join(debug_dir, f"b1_final_img_{idx+1}.png")
            with open(save_path, "wb") as f:
                f.write(img_bytes)
            print(f"  已解码保存至: {save_path}")
        except Exception as e:
            print(f"  保存失败: {e}")

    # 4. 遍历整个 result['messages']（含嵌套 list content）统计 image_url 出现次数
    final_state = await agent.aget_state(config_dict)
    all_msgs = final_state.values.get("messages", [])
    print(f"\n--- 消息链检查 (result['messages']) ---")
    print(f"消息总数: {len(all_msgs)}")
    msg_image_count = count_image_urls(all_msgs)
    print(f"消息链中出现 image_url 次数: {msg_image_count}")

    # 5. astream 事件里 CodeAgent 实际调用的工具名
    print(f"\n--- astream 事件中 CodeAgent 实际调用的工具名 ---")
    for ns, node, name in astream_tool_calls:
        if name != "task":
            print(f"CodeAgent 调用工具: {name}")

    # 6. 主 Agent 最终回答原文
    print(f"\n--- 主 Agent 最终回答原文 ---")
    print(final_text)

if __name__ == "__main__":
    asyncio.run(main())
