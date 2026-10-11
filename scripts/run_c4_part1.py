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
from scripts.chat_cli import save_images

async def test_1_pure_text():
    print("\n" + "=" * 60)
    print("【测试 1: 纯文本】 1+1等于几")
    print("=" * 60)
    reply = await chat_service.process_message(user_id="user_test_1", text="1+1等于几")
    print(f"Reply Text: {reply.text}")
    print(f"Images count: {len(reply.images)}")
    print(f"Awaiting approval: {reply.awaiting_approval}")

async def test_2_geo_context():
    print("\n" + "=" * 60)
    print("【测试 2: 地理与上下文接力】")
    print("=" * 60)
    user_id = "user_geo_relay"
    print("-> 消息 1: 济南今天天气怎么样")
    r1 = await chat_service.process_message(user_id=user_id, text="济南今天天气怎么样")
    print(f"Reply 1 Text: {r1.text}")

    print("\n-> 消息 2: 那明天呢")
    r2 = await chat_service.process_message(user_id=user_id, text="那明天呢")
    print(f"Reply 2 Text: {r2.text}")

async def test_3_file_agent_hitl():
    print("\n" + "=" * 60)
    print("【测试 3: FileAgent 授权 Y / N】")
    print("=" * 60)
    user_id = "user_hitl_test"

    # 3.1 授权同意 Y
    f1_path = "workspace/test_hitl1.txt"
    with open(f1_path, "w", encoding="utf-8") as f:
        f.write("initial content 1")
    print(f"准备测试文件 {f1_path}，初始内容: 'initial content 1'")

    q1 = "请把 workspace/test_hitl1.txt 的内容改写成 'hello from chat_service'"
    print(f"-> 发送: {q1}")
    r1 = await chat_service.process_message(user_id=user_id, text=q1)
    print(f"r1.awaiting_approval: {r1.awaiting_approval}")
    print(f"r1.text:\n{r1.text}")

    print("\n-> 回复: Y")
    r2 = await chat_service.process_message(user_id=user_id, text="Y")
    print(f"r2.awaiting_approval: {r2.awaiting_approval}")
    print(f"r2.text:\n{r2.text}")
    with open(f1_path, "r", encoding="utf-8") as f:
        content_after = f.read()
    print(f"文件 {f1_path} 最终内容: '{content_after}'")

    # 3.2 授权拒绝 N
    f2_path = "workspace/test_hitl2.txt"
    with open(f2_path, "w", encoding="utf-8") as f:
        f.write("initial content 2")
    print(f"\n准备测试文件 {f2_path}，初始内容: 'initial content 2'")

    user_id_n = "user_hitl_test_n"
    q2 = "请把 workspace/test_hitl2.txt 的内容改写成 'should not change'"
    print(f"-> 发送: {q2}")
    rn1 = await chat_service.process_message(user_id=user_id_n, text=q2)
    print(f"rn1.awaiting_approval: {rn1.awaiting_approval}")
    print(f"rn1.text:\n{rn1.text}")

    print("\n-> 回复: N")
    rn2 = await chat_service.process_message(user_id=user_id_n, text="N")
    print(f"rn2.awaiting_approval: {rn2.awaiting_approval}")
    print(f"rn2.text:\n{rn2.text}")
    with open(f2_path, "r", encoding="utf-8") as f:
        content_after_2 = f.read()
    print(f"文件 {f2_path} 最终内容: '{content_after_2}'")

async def test_4_irrelevant_during_hitl():
    print("\n" + "=" * 60)
    print("【测试 4: 授权等待期间发无关消息】")
    print("=" * 60)
    user_id = "user_irrelevant_test"
    q = "请把 workspace/test_hitl1.txt 的内容改写成 'temporary test'"
    print(f"-> 发送: {q}")
    r1 = await chat_service.process_message(user_id=user_id, text=q)
    print(f"r1.awaiting_approval: {r1.awaiting_approval}")

    print("\n-> 发送无关消息: '今天天气真好啊'")
    r2 = await chat_service.process_message(user_id=user_id, text="今天天气真好啊")
    print(f"r2.awaiting_approval: {r2.awaiting_approval}")
    print(f"r2.text:\n{r2.text}")

    print("\n-> 随后回复 N 取消中断")
    r3 = await chat_service.process_message(user_id=user_id, text="N")
    print(f"r3.text:\n{r3.text}")

async def test_5_image_input():
    print("\n" + "=" * 60)
    print("【测试 5: 图片输入】 下载 -> VLM 意图翻译 -> Agent")
    print("=" * 60)
    img_path = "data/pipeline/images/148e7e2ea6184312606c8bdd0997a81a94c6d4fb9cf884efefa48b2aa5d31783.png"
    print(f"使用图片: {img_path}")
    reply = await chat_service.process_message(
        user_id="user_image_test",
        text="这是什么",
        image_url=img_path,
    )
    print(f"Reply Text:\n{reply.text}")
    print(f"Awaiting approval: {reply.awaiting_approval}")

async def test_6_concurrency():
    print("\n" + "=" * 60)
    print("【测试 6: 并发拦截】 同一 user_id 同时发两条")
    print("=" * 60)
    user_id = "user_concurrency"

    # 同时发起两个 process_message 请求
    t1 = asyncio.create_task(chat_service.process_message(user_id=user_id, text="帮我数到 100"))
    t2 = asyncio.create_task(chat_service.process_message(user_id=user_id, text="帮我算 2+2"))

    results = await asyncio.gather(t1, t2)
    print(f"Task 1 Reply: {results[0].text if results[0] else None}")
    print(f"Task 2 Reply: {results[1].text if results[1] else None}")

async def test_9_chart():
    print("\n" + "=" * 60)
    print("【测试 9: 图表折线图】")
    print("=" * 60)
    user_id = "user_chart_test"
    q = "帮我画一张折线图，x=[1,2,3], y=[1,4,9]"
    print(f"-> 发送: {q}")
    reply = await chat_service.process_message(user_id=user_id, text=q)
    print(f"Reply Text 原文:\n{reply.text}")
    print(f"Images count: {len(reply.images)}")
    save_images(reply.images, user_id)

async def main():
    await chat_service.init_service()
    await test_1_pure_text()
    await test_2_geo_context()
    await test_3_file_agent_hitl()
    await test_4_irrelevant_during_hitl()
    await test_5_image_input()
    await test_6_concurrency()
    await test_9_chart()

if __name__ == "__main__":
    asyncio.run(main())
