import base64
import time
import traceback
from nonebot import on_message, get_driver
from nonebot.adapters.qq.event import C2CMessageCreateEvent, MessageEvent
from nonebot.adapters.qq.message import MessageSegment, Message
from nonebot.matcher import Matcher
from nonebot.exception import FinishedException

from app import config
from app.services import chat_service
from app.utils.identity import resolve_platform_user
from app.utils.text import URL_REGEX, defang_urls

def _strip_urls(text: str) -> str:
    """对文本中的显式 URL 进行脱敏替换，避免 QQ 官方平台因域名白名单拒收消息"""
    if not text:
        return text
    return URL_REGEX.sub("（链接已省略）", text)

driver = get_driver()

# 同类“被忽略”的日志限频：群聊 / 频道消息可能很多，每类每 60 秒最多打一行，别刷屏
_IGNORED_LOG_INTERVAL = 60.0
_last_ignored_log: dict[str, float] = {}


def _log_ignored(kind: str, message: str) -> None:
    now = time.monotonic()
    last = _last_ignored_log.get(kind)
    if last is not None and now - last < _IGNORED_LOG_INTERVAL:
        return
    _last_ignored_log[kind] = now
    print(message)

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

@driver.on_shutdown
async def shutdown_system():
    """在 NoneBot 关闭时释放 chat_service 与 MCP 资源"""
    try:
        await chat_service.shutdown_service()
    except Exception as e:
        print(f"[ERROR] chat_service 关闭失败: {e}")
        traceback.print_exc()

# 注册事件响应器，接收所有消息事件，再在处理函数里只放行好友私聊（C2C）
chat_matcher = on_message(priority=10, block=False)

@chat_matcher.handle()
async def handle_chat(matcher: Matcher, event: MessageEvent):
    # 只处理好友私聊（C2C）：群 @、频道、频道私信等其他事件一律忽略，只打一行限频日志。
    # 这里用 isinstance 而不是靠类型标注过滤：标注不匹配时 NoneBot 会静默跳过，打不出日志。
    if not isinstance(event, C2CMessageCreateEvent):
        _log_ignored(type(event).__name__, f"[plugins.chat] 忽略非私聊（C2C）事件: {type(event).__name__}")
        return

    if not chat_service.is_initialized():
        await matcher.finish("Agent 系统尚未初始化完成，请稍后再试。")

    # 身份：C2C 事件的 get_user_id() 是 user_openid（只在当前机器人 AppID 范围内有效）。格式不合法就忽略，不进入任何流程。
    raw_id = str(event.get_user_id()).strip()
    try:
        identity = resolve_platform_user(raw_id)
    except ValueError:
        _log_ignored("bad_user_id", f"[plugins.chat] 忽略一条用户 ID 格式不合法的私聊消息（长度 {len(raw_id)}）")
        return

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
        reply = await chat_service.process_message(text=raw_text, image_url=image_url, identity=identity)
        if reply is None:
            return

        out_msg = Message()
        if reply.text:
            reply_text = reply.text
            if len(reply_text) > config.QQ_MAX_REPLY_CHARS:
                reply_text = reply_text[:config.QQ_MAX_REPLY_CHARS] + "\n（内容过长已截断）"
            sanitize = defang_urls if reply.awaiting_approval else _strip_urls
            out_msg += MessageSegment.text(sanitize(reply_text))
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
