"""GNK48-Agent 控制台交互入口 (CLI)"""

from app import bootstrap

bootstrap.apply()

import argparse
import asyncio
import base64
import collections
import logging
import os
import threading
from typing import Optional

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

from app.services import chat_service
from app import config
from app.utils.identity import Identity, cli_member_identity, owner_identity
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


def _start_stdin_reader(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> threading.Thread:
    """全程只开一个 daemon 读线程：多个 input() 线程会抢 stdin 丢行；非 daemon 则退出时会被挂着的 input() 拖住。

    每读到一行就投进 queue，EOF 投 None。这样任务运行期间主 task 也能读到 /stop。
    """
    def _post(item) -> bool:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
            return True
        except RuntimeError:        # 事件循环已关闭
            return False

    def _reader() -> None:
        while True:
            try:
                line = input()
            except Exception:       # EOFError 或 stdin 被关闭
                _post(None)
                return
            if not _post(line):
                return

    thread = threading.Thread(target=_reader, name="cli-stdin", daemon=True)
    thread.start()
    return thread


def _print_reply(reply) -> None:
    print("\n🤖 [Reply]:")
    print(reply.text)
    if reply.awaiting_approval:
        print("⚠️ [状态: 等待人工授权 (awaiting_approval=True)]")

    if reply.images:
        print(f"📦 [收到 {len(reply.images)} 张图片]:")
        save_images(reply.images)


def _is_stop_like(text: str) -> bool:
    """/stop 与 /reset 都会使正在运行的那一轮返回 None（被中止），此时不应再打印“(空)”。"""
    parts = text.split(None, 1)
    return bool(parts) and parts[0].lower() in ("/stop", "/reset")


def _report_turn(turn: asyncio.Task, stop_sent: bool) -> None:
    """打印一轮 process_message 的结果。"""
    try:
        reply = turn.result()
    except asyncio.CancelledError:
        return
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return
    if reply is None:
        if not stop_sent:           # 被 /stop 中止的那一轮也返回 None，不是“空回复”
            print("🤖 [Reply]: (空)")
        return
    _print_reply(reply)


async def _wait_line_or_turn(turn: asyncio.Task, queue: asyncio.Queue) -> tuple[Optional[str], bool]:
    """任务运行期间并发等“这一轮结束”或“又来一行输入”。返回 (行, 是否取到了一行)；EOF 的行是 None。"""
    getter = asyncio.ensure_future(queue.get())
    try:
        await asyncio.wait({turn, getter}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        getter.cancel()
        raise
    if getter.done() and not getter.cancelled():
        return getter.result(), True
    getter.cancel()                 # Queue.get 被取消不会丢数据
    return None, False


async def main():
    parser = argparse.ArgumentParser(description="GNK48-Agent Chat CLI")
    parser.add_argument("--image", type=str, default=None, help="图片路径或 URL (附带在第一条消息)")
    parser.add_argument("--rebuild", action="store_true", help="强制重建知识库索引")
    parser.add_argument(
        "--member",
        type=str,
        default=None,
        metavar="NAME",
        help="以 member 身份运行（不开 QQ 就能在终端体验 member：没有文件 / 命令 / 桌面 / 浏览器 / 代码能力）；"
             "NAME 只能由字母、数字、下划线组成。不写则以 owner 身份运行",
    )
    args = parser.parse_args()

    # 注意用 is not None 判断：--member "" 必须报错，而不是悄悄变成 owner
    identity: Identity
    try:
        identity = cli_member_identity(args.member) if args.member is not None else owner_identity()
    except ValueError as e:
        parser.error(str(e))

    if args.rebuild:
        config.REFRESH_COLLECTION = True
        print("🚀 检测到 --rebuild 参数，将强制重建知识库索引！")

    first_image = args.image

    print("=" * 60)
    print("启动 Chat CLI...")
    print(f"当前身份：{'管理员 (owner)' if identity.is_owner else f'成员 (member: {identity.user_id})'}")
    print("输入 'exit' 或 'quit' 退出；任务运行中输入 /stop 可中止。")
    print("=" * 60)

    turn: Optional[asyncio.Task] = None
    try:
        await chat_service.init_service()

        queue: asyncio.Queue = asyncio.Queue()
        backlog: collections.deque = collections.deque()   # 运行结束时恰好已取出、还没处理的行
        _start_stdin_reader(asyncio.get_running_loop(), queue)

        is_first = True
        stop_sent = False
        while True:
            try:
                if turn is None:
                    # ── 空闲：等下一行输入 ──
                    print("\n请输入消息: ", end="", flush=True)
                    line = backlog.popleft() if backlog else await queue.get()
                    if line is None:                        # EOF
                        print("\n👋 退出 CLI。")
                        break
                    user_input = line.strip()

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

                    turn = asyncio.create_task(
                        chat_service.process_message(text=user_input, image_url=current_img, identity=identity)
                    )
                    stop_sent = _is_stop_like(user_input)
                    if not user_input.startswith("/"):
                        print("（任务运行中，输入 /stop 可中止）")
                    continue

                # ── 运行中：同时等“这一轮结束”和“又来一行输入” ──
                line, got_line = await _wait_line_or_turn(turn, queue)

                if turn.done():
                    _report_turn(turn, stop_sent)
                    turn = None
                    if got_line:                            # 先打印这一轮的结果，这一行留给空闲阶段
                        backlog.appendleft(line)
                    continue
                if not got_line:
                    continue

                if line is None:                            # 运行期间读到 EOF：视为 /stop 后退出
                    reply = await chat_service.process_message(text="/stop", identity=identity)
                    if reply is not None:
                        _print_reply(reply)
                    await asyncio.wait({turn}, timeout=config.AGENT_STOP_WAIT_SECONDS)
                    print("\n👋 退出 CLI。")
                    break

                text = line.strip()
                if not text:
                    continue
                if text.lower() in ("exit", "quit"):
                    print("⚠️ 任务运行中，请先 /stop 或等待完成。")
                    continue
                if _is_stop_like(text):
                    stop_sent = True
                # /stop、/reset、忙提示都由 chat_service 处理
                reply = await chat_service.process_message(text=text, identity=identity)
                if reply is not None:
                    _print_reply(reply)

            except (KeyboardInterrupt, EOFError):
                print("\n👋 退出 CLI。")
                break
            except Exception as e:
                print(f"❌ 运行出错: {e}")
    finally:
        await chat_service.shutdown_service()
        if turn is not None and not turn.done():
            turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
