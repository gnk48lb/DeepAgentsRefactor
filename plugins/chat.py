import base64
import time
import traceback
from nonebot import on_message, get_driver
from nonebot.adapters.qq import Bot as QQBot
from nonebot.adapters.qq import exception as qq_exc
from nonebot.adapters.qq.event import C2CMessageCreateEvent, MessageEvent
from nonebot.adapters.qq.message import MessageSegment, Message
from nonebot.matcher import Matcher
from nonebot.exception import FinishedException

from app import config
from app.services import chat_service, scheduler_service
from app.utils.files import incoming_files_from_attachments
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

# ── 主动推送（定时任务到点后由 scheduler_service 调用）──
# 错误码表：ActionFailed.code -> PushError.kind。“来源”是依据，“实测”指 tests/live_qq_push.py 在真机上是否触发过：
# 2026-11-26 的第一轮真机探针里，无 msg_id 的主动文本能发出，没有触发过下面任何一个错误码，所以全部是平台文档的说法、未实测；
# 用户关闭“允许主动发送消息”之后的错误码那一项（用例 ⑥）那次也没量到（发送仍然成功），permanent 组仍是文档值。
_PUSH_CODE_KINDS = {
    40034105: "permanent",     # 无主动消息权限                来源：平台文档    实测：否
    40054013: "permanent",     # 用户拒绝接收                  来源：平台文档    实测：否
    40034100: "transient",     # 频率限制                      来源：平台文档    实测：否（连发 8 条没有触发）
    50055001: "transient",     # 服务端要求重试                来源：平台文档    实测：否
    40054016: "transient",     # 机器人离线                    来源：平台文档    实测：否
    40054005: "dedupe",        # 内容重复                      来源：平台文档    实测：否（间隔 1 秒的相同内容没有被去重）
    40054007: "too_long",      # 内容过长                      来源：平台文档    实测：否（4000 字可发）
    40054010: "url_rejected",  # 含不允许的链接                来源：平台文档    实测：否（https 链接与裸域名都没被拒）
}
# 表外的 ActionFailed 一律按 transient 处理（有次数与时间上限）。适配器里另有几类不是 ActionFailed 的情况：
#   AuditException（2xx 且带 audit_id，消息进入审核）、HTTP 201 / 202（适配器统一抛成 ActionFailed）：都按“平台已受理”处理；
#   RateLimitException（429）/ UnauthorizedException（401，适配器已刷新过一次令牌）/ NetworkError / ApiNotAvailable：transient。
_PUSH_LOG_MESSAGE_CHARS = 100


async def _qq_push(openid: str, text: str) -> None:
    """经 QQ C2C 向用户主动发一条纯文本（不带 msg_id）；失败一律抛 scheduler_service.PushError。

    get_bots 在函数里惰性导入：smoke_multiuser 的桩加载器只替换了 on_message / get_driver。
    文本先过 _strip_urls（主动消息里的链接会被 40054010 拒收）。日志里 openid 只显示前 8 位，不记录正文。"""
    from nonebot import get_bots

    PushError = scheduler_service.PushError
    bot = next((b for b in get_bots().values() if isinstance(b, QQBot)), None)
    if bot is None:
        raise PushError("transient", None, "bot offline")
    tag = openid[:8]

    def scrub(value) -> str:
        return " ".join(str(value or "").replace(openid, tag).split())[:_PUSH_LOG_MESSAGE_CHARS]

    try:
        await bot.send_to_c2c(openid=openid, message=_strip_urls(text))
    except qq_exc.AuditException:
        return                                              # 进入审核：平台已受理
    except qq_exc.RateLimitException as e:
        print(f"[plugins.chat] 主动推送被限频 openid={tag} code={e.code}")
        raise PushError("transient", e.code, "rate limited") from None
    except qq_exc.UnauthorizedException:
        print(f"[plugins.chat] 主动推送未授权（令牌刷新后仍被拒）openid={tag}")
        raise PushError("transient", None, "unauthorized") from None
    except qq_exc.ActionFailed as e:
        if e.status_code in (201, 202):
            return
        print(f"[plugins.chat] 主动推送失败 openid={tag} status={e.status_code} code={e.code} message={scrub(e.message)!r}")
        raise PushError(_PUSH_CODE_KINDS.get(e.code, "transient"), e.code, f"status={e.status_code} {scrub(e.message)}") from None
    except (qq_exc.NetworkError, qq_exc.ApiNotAvailable) as e:
        print(f"[plugins.chat] 主动推送网络 / 接口不可用 openid={tag} {type(e).__name__}")
        raise PushError("transient", None, type(e).__name__) from None
    except Exception as e:
        print(f"[plugins.chat] 主动推送异常 openid={tag} {type(e).__name__}")
        raise PushError("transient", None, type(e).__name__) from None


@driver.on_startup
async def init_system():
    """在 NoneBot 启动时初始化 chat_service 服务"""
    try:
        await chat_service.init_service()
    except Exception as e:
        print(f"[ERROR] chat_service 初始化失败: {e}")
        traceback.print_exc()
    # 定时任务调度器必须在 init_service 成功之后启动（NoneBot 的 startup 钩子按注册顺序串行）
    if chat_service.is_initialized() and config.SCHEDULER_ENABLED:
        try:
            scheduler_service.start(run_turn=chat_service.run_scheduled_turn, push=_qq_push, cli=False)
        except Exception as e:
            print(f"[ERROR] 定时任务调度器启动失败: {e}")
            traceback.print_exc()

@driver.on_shutdown
async def shutdown_system():
    """在 NoneBot 关闭时释放 chat_service 与 MCP 资源"""
    # shutdown 钩子倒序执行：先停调度器（取消在途的定时任务），再关服务
    try:
        await scheduler_service.stop()
    except Exception as e:
        print(f"[ERROR] 定时任务调度器停止失败: {e}")
        traceback.print_exc()
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

    # 文件：QQ 的 file 段只留下载地址（丢了文件名和大小），所以文件名 / 大小取自 event.attachments；
    # 下载地址带签名，等同凭证，这里不打印、不记录。只有确实带文件时才传 files，其余消息的调用与改动前完全一致
    incoming_files = incoming_files_from_attachments(getattr(event, "attachments", None), message)
    kwargs = {"files": incoming_files} if incoming_files else {}

    try:
        reply = await chat_service.process_message(text=raw_text, image_url=image_url, identity=identity, **kwargs)
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
