import base64
import traceback
from nonebot import on_message, get_driver
from nonebot.adapters.qq.event import DirectMessageCreateEvent, MessageEvent
from nonebot.adapters.qq.message import MessageSegment, Message
from nonebot.matcher import Matcher
from nonebot.exception import FinishedException

from app import chat_service

driver = get_driver()

def _make_image_segment(url: str) -> MessageSegment:
    """
    智能图片 Segment 工厂函数。
    - 如果 url 是 data URI（base64 内嵌图片），解码为 bytes 并使用 file_image()。
    - 否则直接使用 image() 发送普通 URL。
    """
    if url.startswith("data:"):
        try:
            header, b64data = url.split(",", 1)
            img_bytes = base64.b64decode(b64data)
            return MessageSegment.file_image(img_bytes)
        except Exception as e:
            print(f"[WARN] _make_image_segment 解码失败: {e}")
            return MessageSegment.text("[图片发送失败]")
    return MessageSegment.image(url)

@driver.on_startup
async def init_system():
    """在 NoneBot 启动时初始化 chat_service 服务"""
    try:
        await chat_service.init_service()
    except Exception as e:
        print(f"[ERROR] chat_service 初始化失败: {e}")
        traceback.print_exc()

# 注册事件响应器，监听群消息和私聊消息
chat_matcher = on_message(priority=10, block=False)

@chat_matcher.handle()
async def handle_chat(matcher: Matcher, event: MessageEvent | DirectMessageCreateEvent):
    if not chat_service.is_initialized():
        await matcher.finish("Agent 系统尚未初始化完成，请稍后再试。")

    user_id = str(event.get_user_id())
    print(f"[chat] user_id={user_id}")

    # 提取消息中的图片和文本
    image_url = None
    text_parts = []
    message = event.get_message()

    for msg_seg in message:
        if msg_seg.type in ["image", "attachment"]:
            url = msg_seg.data.get("url")
            if url and not image_url:
                image_url = url
        elif msg_seg.type == "text":
            text_parts.append(msg_seg.data.get("text", ""))

    raw_text = "".join(text_parts).strip()

    try:
        reply = await chat_service.process_message(user_id=user_id, text=raw_text, image_url=image_url)
        if reply is None:
            return

        out_msg = Message()
        if reply.text:
            out_msg += MessageSegment.text(reply.text)
        for img_item in reply.images:
            url = img_item.get("image_url", {}).get("url", "")
            if url:
                out_msg += _make_image_segment(url)

        await matcher.finish(out_msg)

    except FinishedException:
        raise
    except Exception as e:
        traceback.print_exc()
        await matcher.finish("处理时出了点问题，请稍后再试")
