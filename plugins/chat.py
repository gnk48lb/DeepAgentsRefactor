import base64
import re
import traceback
from nonebot import on_message, get_driver
from nonebot.adapters.qq.event import DirectMessageCreateEvent, MessageEvent
from nonebot.adapters.qq.message import MessageSegment, Message
from nonebot.matcher import Matcher
from nonebot.exception import FinishedException

from app import config
from app.services import chat_service

_URL_REGEX = re.compile(r"(?:https?://|www\.)[^\s<>\"'）)】」》，。；、！？]+", re.IGNORECASE)

def _strip_urls(text: str) -> str:
    """对文本中的显式 URL 进行脱敏替换，避免 QQ 官方平台因域名白名单拒收消息"""
    if not text:
        return text
    return _URL_REGEX.sub("（链接已省略）", text)

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
        except Exception:
            print(f"[WARN] _make_image_segment 解码失败: {e}")
            return MessageSegment.text("[图片发送失败]")
    return MessageSegment.image(url)

@driver.on_startup
async def init_system():
    """在 NoneBot 启动时初始化 chat_service 服务"""
    try:
        await chat_service.init_service()
    except Exception:
        print(f"[ERROR] chat_service 初始化失败: {e}")
        traceback.print_exc()

@driver.on_shutdown
async def shutdown_system():
    """在 NoneBot 关闭时释放 chat_service 与 MCP 资源"""
    try:
        await chat_service.shutdown_service()
    except Exception:
        print(f"[ERROR] chat_service 关闭失败: {e}")
        traceback.print_exc()

# 注册事件响应器，监听群消息和私聊消息
chat_matcher = on_message(priority=10, block=False)

@chat_matcher.handle()
async def handle_chat(matcher: Matcher, event: MessageEvent | DirectMessageCreateEvent):
    if not chat_service.is_initialized():
        await matcher.finish("Agent 系统尚未初始化完成，请稍后再试。")

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
        reply = await chat_service.process_message(text=raw_text, image_url=image_url)
        if reply is None:
            return

        out_msg = Message()
        if reply.text:
            reply_text = reply.text
            if len(reply_text) > config.QQ_MAX_REPLY_CHARS:
                reply_text = reply_text[:config.QQ_MAX_REPLY_CHARS] + "\n（内容过长已截断）"
            out_msg += MessageSegment.text(_strip_urls(reply_text))
        for img_item in reply.images:
            url = img_item.get("image_url", {}).get("url", "")
            if url:
                out_msg += _make_image_segment(url)

        await matcher.finish(out_msg)

    except FinishedException:
        raise
    except Exception:
        traceback.print_exc()
        await matcher.finish("处理时出了点问题，请稍后再试")
