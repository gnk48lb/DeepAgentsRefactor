"""GNK48-Agent 控制台交互入口 (CLI)"""

from app import bootstrap

bootstrap.apply()

import argparse
import asyncio
import base64
import logging
import os

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

from app.services import chat_service
from app import config
from app.utils.images import guess_mime_type


def save_images(images: list):
    if not images:
        return
    debug_dir = config.OUTBOX_DEBUG_DIR
    os.makedirs(debug_dir, exist_ok=True)
    for idx, item in enumerate(images, 1):
        url = item.get("image_url", {}).get("url", "")
        if url.startswith("data:image/"):
            try:
                header, b64data = url.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                save_path = os.path.join(debug_dir, f"cli_{idx}.png")
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
                print(f"  [图片已保存] {save_path} (大小: {len(img_bytes)} 字节)")
            except Exception as e:
                print(f"  [图片保存失败] {e}")


def _local_image_to_data_uri(path_str: str) -> str:
    if not path_str:
        return path_str
    if path_str.startswith("http://") or path_str.startswith("https://") or path_str.startswith("data:"):
        return path_str
    if os.path.exists(path_str):
        with open(path_str, "rb") as f:
            data = f.read()
        mime = guess_mime_type(data, default="image/png")
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    return path_str


async def main():
    parser = argparse.ArgumentParser(description="GNK48-Agent Chat CLI")
    parser.add_argument("--image", type=str, default=None, help="图片路径或 URL (附带在第一条消息)")
    parser.add_argument("--rebuild", action="store_true", help="强制重建知识库索引")
    args = parser.parse_args()

    if args.rebuild:
        config.REFRESH_COLLECTION = True
        print("🚀 检测到 --rebuild 参数，将强制重建知识库索引！")

    first_image = args.image

    print("=" * 60)
    print("启动 Chat CLI...")
    print("输入 'exit' 或 'quit' 退出。")
    print("=" * 60)

    try:
        await chat_service.init_service()

        is_first = True
        while True:
            try:
                prompt_str = "\n请输入消息: "
                user_input = await asyncio.to_thread(input, prompt_str)
                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit"):
                    print("👋 退出 CLI。")
                    break

                current_img = first_image if is_first else None
                is_first = False

                if current_img:
                    print(f"📷 附带图片: {current_img}")
                    current_img = _local_image_to_data_uri(current_img)

                reply = await chat_service.process_message(
                    text=user_input,
                    image_url=current_img,
                )

                if reply is None:
                    print("🤖 [Reply]: (空)")
                    continue

                print("\n🤖 [Reply]:")
                print(reply.text)
                if reply.awaiting_approval:
                    print("⚠️ [状态: 等待人工授权 (awaiting_approval=True)]")

                if reply.images:
                    print(f"📦 [收到 {len(reply.images)} 张图片]:")
                    save_images(reply.images)

            except (KeyboardInterrupt, EOFError):
                print("\n👋 退出 CLI。")
                break
            except Exception as e:
                print(f"❌ 运行出错: {e}")
    finally:
        await chat_service.shutdown_service()


if __name__ == "__main__":
    asyncio.run(main())
