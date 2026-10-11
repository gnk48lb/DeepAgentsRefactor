import asyncio
import os
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from app import chat_service

class SilentHTTPHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # 静默或只打印简短日志
        print(f"  [HTTP Server] {self.address_string()} - {format % args}")

def run_server(server: HTTPServer):
    server.serve_forever()

async def main():
    print("=" * 60)
    print("【测试 HTTP 远程图片服务输入】")
    print("=" * 60)

    # 1. 创建一张测试 PNG
    test_dir = Path("storage/test_http")
    test_dir.mkdir(parents=True, exist_ok=True)
    img_path = test_dir / "x.png"

    img = Image.new("RGB", (200, 100), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((20, 40), "HTTP IMAGE TEST", fill=(255, 255, 0))
    img.save(img_path, format="PNG")
    print(f"生成测试图片: {img_path} ({os.path.getsize(img_path)} 字节)")

    # 2. 启动本地 http.server
    import socket
    # 动态获取空闲端口
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    handler = lambda *args, **kwargs: SilentHTTPHandler(*args, directory=str(test_dir), **kwargs)
    httpd = HTTPServer(("127.0.0.1", port), handler)
    server_thread = threading.Thread(target=run_server, args=(httpd,), daemon=True)
    server_thread.start()
    print(f"本地 HTTP 服务器已启动: http://127.0.0.1:{port}/")

    try:
        # 3. 初始化并调用 chat_service.process_message
        print("初始化 chat_service...")
        await chat_service.init_service()

        test_url = f"http://127.0.0.1:{port}/x.png"
        print(f"调用 process_message(user_id='t_http', text='这是什么', image_url='{test_url}')...")

        reply = await chat_service.process_message(
            user_id="t_http",
            text="这是什么",
            image_url=test_url,
        )

        print("\n=== process_message 返回结果 ===")
        if reply is None:
            print("reply: None")
        else:
            print(f"reply.awaiting_approval: {reply.awaiting_approval}")
            print(f"reply.images count: {len(reply.images)}")
            print("reply.text:")
            print(reply.text)

    finally:
        httpd.shutdown()
        print("\n本地 HTTP 服务器已关闭。")

if __name__ == "__main__":
    asyncio.run(main())
