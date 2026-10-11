import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 设置标准输出编码为 utf-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import config
from app import chat_service

async def test_7_restricted_agent():
    print("\n" + "=" * 60)
    print("【测试 7: 受限 Agent 与特权 Agent 隔离】 OWNER_QQ_IDS=111")
    print("=" * 60)

    # 动态配置主人为 111
    config.OWNER_QQ_IDS = {"111"}
    await chat_service.init_service()

    f_path = "workspace/test_hitl1.txt"
    with open(f_path, "w", encoding="utf-8") as f:
        f.write("hello from chat_service")
    print(f"准备文件 {f_path}，当前内容: 'hello from chat_service'")

    # 1. 普通用户 222 (受限版)
    print("\n--- 7.1 非主人用户 222 (无 FileAgent 权限) ---")
    q222 = "请把 workspace/test_hitl1.txt 的内容改写成 'x'"
    print(f"用户 222 发送: {q222}")
    r222 = await chat_service.process_message(user_id="222", text=q222)
    print(f"r222.awaiting_approval: {r222.awaiting_approval}")
    print(f"r222.text:\n{r222.text}")
    with open(f_path, "r", encoding="utf-8") as f:
        c222 = f.read()
    print(f"用户 222 操作后文件内容: '{c222}' (预期内容未被改写: 'hello from chat_service')")

    # 2. 主人用户 111 (特权版)
    print("\n--- 7.2 主人用户 111 (有 FileAgent 权限) ---")
    q111 = "请把 workspace/test_hitl1.txt 的内容改写成 'owner_updated'"
    print(f"主人 111 发送: {q111}")
    r111 = await chat_service.process_message(user_id="111", text=q111)
    print(f"r111.awaiting_approval: {r111.awaiting_approval}")
    print(f"r111.text:\n{r111.text}")

    if r111.awaiting_approval:
        print("\n主人 111 回复: Y 授权操作")
        r111_y = await chat_service.process_message(user_id="111", text="Y")
        print(f"r111_y.text:\n{r111_y.text}")
        while r111_y and r111_y.awaiting_approval:
            print("\n主人 111 继续回复: Y 授权后续操作")
            r111_y = await chat_service.process_message(user_id="111", text="Y")
            print(f"r111_y.text:\n{r111_y.text}")


    with open(f_path, "r", encoding="utf-8") as f:
        c111 = f.read()
    print(f"主人 111 操作后文件内容: '{c111}' (预期已改写为: 'owner_updated')")

if __name__ == "__main__":
    asyncio.run(test_7_restricted_agent())
