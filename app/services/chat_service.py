"""多用户多模态会话状态机与消息调度服务 (所在层: L5)

会话状态不在进程内存里：会话 / 事件日志 / 挂起态在 SessionStore（sessions.db），
模型上下文与 HITL 中断在 checkpoint（checkpoints.db），thread_id == session_id。
process_message 只负责“写入入站事件 + wake(session_id)”，wake 不带载荷，载荷全部从事件日志里读。

多用户：
- 身份由调用方（QQ 插件 / CLI）解析成 Identity 后传入 process_message；owner 与 member 各用一张编译好的主图
  （member 图在编译期就没有任何通往宿主机的专家），能力门控不靠运行时判断。
- 每个用户一把锁：同一用户串行，不同用户互不阻塞；member 另有全局并发上限（信号量），owner 不占用。
- 所有会话读写都带 user_id；wake 按会话归属设置 RequestContext，工具层的 require_owner() 靠它做纵深防御。
- /stop 只中止发送者自己的那一轮；宿主机命令与 Docker 沙箱的终止原语是进程级全局的，只有 owner 的 /stop 才会触发。
- member 的云端 CodeAgent（Modal 沙箱，按 (用户, 会话) 一个）的生命周期钩子都经 modal_sandbox 的模块属性调用：
  /reset 终止当前会话的沙箱，/stop 与失败的回合只中断它自己会话里的命令，/block 回收该用户的全部沙箱，关闭服务时全部终止。
  这些钩子都是尽力而为（失败只打印一行警告，不改变命令或回合的结果），owner 路径上没有任何 Modal 调用。
"""

import asyncio
import base64
import re
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Sequence, Set, Any
import httpx

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.errors import GraphRecursionError

from app import config
from app.agents import orchestrator
from app.memory import injector, extractor
from app.services import startup, session_log, file_staging
from app.runtime import outbox, hitl, cli_logger
from app.infra import models, session_store, checkpointer, modal_sandbox
from app.utils.text import extract_text
from app.utils.files import IncomingFile
from app.utils.images import guess_mime_type
from app.utils.context import RequestContext, set_request_context, reset_request_context
from app.utils.identity import OWNER_USER_ID, Identity, invalid_owner_ids, validate_user_id


@dataclass
class ChatReply:
    text: str
    images: list                  # [{"type": "image_url", "image_url": {"url": "data:image/..."}}]
    awaiting_approval: bool = False


# ── 全局状态：两张编译好的主图 + 每用户一把锁；会话状态全部在 SessionStore / checkpoint 里 ──
_initialized: bool = False
# {"owner": 7 个专家的图, "member": 只有 Knowledge / Media / Map 的图}
_agents: dict = {}
# init_service 里保存的 AsyncSqliteSaver：/reset 删线程、每轮结束清理嵌套 checkpoint 时用
_checkpointer = None
# 同一用户同一时刻只处理一条消息（防止连发消息并发踩同一会话 / HITL 中断状态）；不同用户互不影响
_user_locks: dict[str, asyncio.Lock] = {}
# member 全局并发上限（owner 不占用）；惰性创建，容量取 config.MAX_CONCURRENT_MEMBER_TURNS
_member_sem: Optional[asyncio.Semaphore] = None
# member 图是否编译进了云端 CodeAgent（init_service 之后以真正编译出来的结果为准）：只用于启动提示与 /whoami，不用来门控任何钩子
_member_code_enabled: bool = False
_background_tasks: Set[asyncio.Task] = set()

_SLASH_COMMANDS = {
    "/new", "/sessions", "/switch", "/reset", "/stop",
    "/whoami", "/users", "/label", "/block", "/unblock",
}
_ADMIN_COMMANDS = {"/users", "/label", "/block", "/unblock"}
_SESSIONS_CMD_MAX = 30

_BLOCKED_TEXT = "你当前无法使用本机器人。"
_ADMIN_ONLY_TEXT = "该命令仅管理员可用"


@dataclass
class _TurnHandle:
    """正在运行的那一轮对话的句柄：/stop 靠它 cancel 内层 task（不能 cancel 调用方自己的 task）。"""
    task: asyncio.Task
    session_id: str
    user_id: str
    started_at: float            # time.monotonic()
    stop_requested: bool = False
    cancel_sent_at: float = 0.0


# 每个用户当前正在运行的一轮（仅普通消息 / Y-N 审批；斜杠命令在锁内同步执行，不可被 /stop 取消）
_current_turns: dict[str, _TurnHandle] = {}


def _role_of(user_id: str) -> str:
    return "owner" if user_id == OWNER_USER_ID else "member"


def _agent_for(user_id: str):
    """该用户对应的主图。缺图就抛 RuntimeError：绝不回落到 owner 图。"""
    agent = _agents.get(_role_of(user_id))
    if agent is None:
        raise RuntimeError("Agent 尚未初始化")
    return agent


def _user_lock(user_id: str) -> asyncio.Lock:
    lock = _user_locks.get(user_id)
    if lock is None:
        lock = _user_locks[user_id] = asyncio.Lock()
    return lock


def _member_semaphore() -> asyncio.Semaphore:
    global _member_sem
    if _member_sem is None:
        _member_sem = asyncio.Semaphore(config.MAX_CONCURRENT_MEMBER_TURNS)
    return _member_sem


def _reset_runtime_state() -> None:
    """清掉进程内的并发状态（锁 / 信号量 / 正在跑的轮的句柄）。shutdown 与测试共用：
    asyncio 的锁和信号量一旦在某个事件循环里发生过等待，就绑定了那个循环，不能跨 asyncio.run 复用。"""
    global _member_sem, _member_code_enabled
    _user_locks.clear()
    _current_turns.clear()
    _member_sem = None
    _member_code_enabled = False
    file_staging.reset_runtime_state()


def _identity_consistent(identity: Any) -> bool:
    """入口校验：user_id 合法，且 is_owner 恰好等于 user_id 是否为 owner。不自洽一律拒绝（失败即关闭）。"""
    try:
        if not isinstance(identity, Identity):
            return False
        validate_user_id(identity.user_id)
        return identity.is_owner == (identity.user_id == OWNER_USER_ID)
    except ValueError:
        return False


async def _load_image_as_data_uri(image_url: str) -> Optional[str]:
    """
    下载或读取图片，转成 data: URI。
    超时 15 秒，大小上限 5MB。
    """
    try:
        if image_url.startswith("http://") or image_url.startswith("https://"):
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(image_url)
                resp.raise_for_status()
                img_bytes = resp.content
                if len(img_bytes) > 5 * 1024 * 1024:
                    print(f"[WARN] 远程图片超过 5MB 限制: {len(img_bytes)} 字节")
                    return None
        elif image_url.startswith("data:"):
            return image_url
        else:
            print(f"[WARN] 未知的图片路径或 URL: {image_url}")
            return None

        mime = guess_mime_type(img_bytes)
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64_str}"
    except Exception as e:
        print(f"[ERROR] 读取/下载图片失败: {e}")
        return None


def get_active_session_id(*, user_id: str) -> str:
    """该用户当前活动会话 id（供斜杠命令与测试使用）。"""
    return session_store.get_store().get_active_session_id(user_id=user_id)


def _state_config(session_id: str) -> dict:
    """只读查询状态用的最小 config。"""
    return {"configurable": {"thread_id": session_id}}


def _run_config(session_id: str, user_id: str) -> dict:
    return {
        "recursion_limit": config.AGENT_RECURSION_LIMIT,
        # configurable.user_id：会话回溯工具靠它限定只读自己的会话
        "configurable": {"thread_id": session_id, "user_id": user_id},
        # metadata.user_id：cli_logger 靠它记录每次 LLM 调用属于谁。configurable 里的自定义键不会自动进回调 metadata
        # （实测：当前 langchain-core 只带 model / checkpoint_ns；thread_id 是 LangGraph 自己加的），必须显式写；
        # metadata 会被子图 / 子专家的调用继承
        "metadata": {"user_id": user_id},
        "callbacks": [cli_logger.CLIDebugCallbackHandler()],
    }


async def _has_pending_approval(session_id: str, user_id: str) -> bool:
    """该会话的 checkpoint 里是否有未处理的 interrupt（按会话所属用户的角色选图）。"""
    agent = _agents.get(_role_of(user_id))
    if agent is None:
        return False
    state = await agent.aget_state(_state_config(session_id))
    return hitl.has_pending_interrupts(state)


def _startup_notices() -> list[str]:
    """启动时需要提醒运营者的事项（纯函数，便于测试）。"""
    notices: list[str] = []
    if not config.OWNER_USER_IDS:
        notices.append(
            "⚠️⚠️ 未配置 OWNER_USER_IDS：此时 QQ 上没有 owner，所有 QQ 用户都是 member（没有文件 / 命令 / 桌面 / 浏览器能力；代码只能在云端沙箱里跑）。\n"
            "   请用主号私聊机器人发 /whoami 取得平台 ID，写入 .env 的 OWNER_USER_IDS 后重启。本地 CLI 始终是 owner，不受影响。"
        )
    bad = invalid_owner_ids()
    if bad:
        notices.append(f"⚠️ OWNER_USER_IDS 里有格式不合法的项（永远不会命中任何 QQ 用户，多半是写错了）: {', '.join(bad)}")
    if str(config.MYSQL_USER).strip().lower() == "root":
        notices.append(
            "⚠️ MYSQL_USER 是 root：MediaAgent 的 execute_sql 虽然只放行单条 SELECT，仍建议改用只读专用账号，"
            "member 现在也能驱动它生成 SQL。"
        )
    return notices


async def init_service() -> None:
    """初始化数据库、会话存储、checkpointer、MCP 服务并编译 owner / member 两张主图。"""
    global _initialized, _agents, _checkpointer, _member_code_enabled
    models.validate_required_keys()
    print("🚀 [chat_service] 正在初始化数据环境与 Agent 服务...")
    saver = None
    try:
        await asyncio.to_thread(startup.run_global_database_init)
        store = await asyncio.to_thread(session_store.get_store)
        sid = await asyncio.to_thread(store.get_active_session_id, user_id=OWNER_USER_ID)
        saver = await checkpointer.open_checkpointer()
        _checkpointer = saver
        built = await orchestrator.build_main_agents(checkpointer=saver)
        _agents = {"owner": built.owner, "member": built.member}
        await recover_orphans()
        info = store.get_session(sid, user_id=OWNER_USER_ID)
        if config.QQ_FILE_ENABLED:
            try:
                removed = await asyncio.to_thread(file_staging.cleanup_expired_inbox)
                if removed > 0:
                    print(f"🧹 [chat_service] 清理过期 inbox 目录 {removed} 个")
            except Exception as e:
                print(f"⚠️ 警告：清理过期 inbox 失败 (非致命): {type(e).__name__}")
        _initialized = True
    except Exception:
        _agents = {}
        _checkpointer = None
        await checkpointer.close_checkpointer(saver)
        session_store.close_store()
        raise
    print(f"🧩 [chat_service] owner 图专家: {', '.join(built.owner_experts)}")
    print(f"🧩 [chat_service] member 图专家: {', '.join(built.member_experts)}")
    # 以真正编译出来的结果为准，不要自己再算一遍门控
    _member_code_enabled = "CodeAgent" in built.member_experts
    if _member_code_enabled:
        print("☁️ [chat_service] member 云端 CodeAgent：已启用（Modal）")
    else:
        try:
            reason = modal_sandbox.member_code_status()[1]          # 静态判断：不联网、不抛，只用于打印
        except Exception:
            reason = ""
        print("☁️ [chat_service] member 云端 CodeAgent：未启用" + (f"（{reason}）" if reason else ""))
    if info is not None:
        print(f"💬 [chat_service] owner 当前会话 {info.short_id}「{info.title or '（无标题）'}」，事件 {info.events} 条。")
    for notice in _startup_notices():
        print(notice)
    print("✅ [chat_service] 初始化完成。")


def is_initialized() -> bool:
    return _initialized


async def shutdown_service() -> None:
    """关闭系统服务并释放 MCP 资源"""
    global _initialized, _agents, _checkpointer
    print("🛑 [chat_service] 正在关闭服务...")
    # 先中止所有用户还在跑的一轮，避免关闭 MCP / 浏览器时它仍在操作桌面或宿主机命令（尽力而为）
    try:
        await _stop_all_turns()
    except Exception as e:
        print(f"⚠️ 警告：关闭前中止当前任务失败 (非致命): {e}")
    # 云端沙箱按时计费：放在后面的 MCP / 浏览器 / Milvus 关闭之前，它们万一抛异常也不能把它漏掉（atexit 兜底只管正常退出）
    try:
        await asyncio.to_thread(modal_sandbox.shutdown_pool)
    except Exception as e:
        print(f"⚠️ 警告：云端沙箱销毁失败 (非致命): {e}")
    from app.infra import mcp_service, browser_service, milvus_client
    from app.tools import amap
    await mcp_service.shutdown_mcp()
    await browser_service.close_browser()
    milvus_client.close_client()
    await amap.aclose()
    try:
        from app.infra import sandbox
        await asyncio.to_thread(sandbox.shutdown_sandbox)
    except Exception as e:
        print(f"⚠️ 警告：沙箱销毁失败 (非致命): {e}")
    _agents = {}
    _initialized = False
    _reset_runtime_state()
    saver, _checkpointer = _checkpointer, None
    await checkpointer.close_checkpointer(saver)
    session_store.close_store()
    print("✅ [chat_service] 服务已关闭。")


# ══════════════════════════════════════════════════════════════════════════════
# 斜杠命令（在 HITL 判断之前拦截；回复一律纯文本，QQ 不渲染 Markdown）
# ══════════════════════════════════════════════════════════════════════════════

def _command_reply(text: str, turn_start: float, awaiting: bool = False) -> ChatReply:
    cli_logger.log_turn_end(text, 0, awaiting, time.perf_counter() - turn_start)
    return ChatReply(text, [], awaiting_approval=awaiting)


def _parse_command(text: str) -> Optional[tuple[str, str]]:
    """按首个空白分隔的词精确匹配；以 / 开头但不是命令的文本（如路径）按普通消息处理。"""
    parts = text.strip().split(None, 1)
    if not parts:
        return None
    cmd = parts[0].lower()
    if cmd not in _SLASH_COMMANDS:
        return None
    return cmd, (parts[1].strip() if len(parts) > 1 else "")


async def _handle_command(cmd: str, arg: str, turn_start: float, identity: Identity) -> ChatReply:
    if cmd in _ADMIN_COMMANDS and not identity.is_owner:
        # 非 owner：固定回复，不泄露任何信息（连“有没有这个用户”都不透露）
        return _command_reply(_ADMIN_ONLY_TEXT, turn_start)
    if cmd == "/new":
        return await _cmd_new(arg, turn_start, identity)
    if cmd == "/sessions":
        return await _cmd_sessions(arg, turn_start, identity)
    if cmd == "/switch":
        return await _cmd_switch(arg, turn_start, identity)
    if cmd == "/whoami":
        return _cmd_whoami(turn_start, identity)
    if cmd == "/users":
        return _cmd_users(turn_start)
    if cmd == "/label":
        return _cmd_label(arg, turn_start)
    if cmd == "/block":
        return await _cmd_block(arg, turn_start, blocked=True)
    if cmd == "/unblock":
        return await _cmd_block(arg, turn_start, blocked=False)
    return await _cmd_reset(turn_start, identity)     # /stop 不会走到这里：它在 process_message 里锁外直接处理


async def _cmd_new(title: str, turn_start: float, identity: Identity) -> ChatReply:
    uid = identity.user_id
    store = session_store.get_store()
    old_sid = store.get_active_session_id(user_id=uid)
    old_pending = await _has_pending_approval(old_sid, uid)
    info = store.create_session(title, activate=True, user_id=uid)
    msg = f"已新建会话 {info.short_id}" + (f"「{info.title}」" if info.title else "（发出第一条消息后自动命名）") + "，旧会话已原样保留。"
    if old_pending:
        msg += f"\n提醒：上一个会话 {old_sid[:8]} 还有待授权的操作，用 /switch {old_sid[:8]} 切回去后回复 Y 或 N。"
    return _command_reply(msg, turn_start)


async def _cmd_sessions(arg: str, turn_start: float, identity: Identity) -> ChatReply:
    uid = identity.user_id
    store = session_store.get_store()
    limit = config.SESSION_LIST_LIMIT
    if arg.strip().isdigit():
        limit = max(1, min(int(arg.strip()), _SESSIONS_CMD_MAX))
    infos = await asyncio.to_thread(store.list_sessions, limit, user_id=uid)
    active = store.get_active_session_id(user_id=uid)
    now = time.time()
    lines = []
    for i, info in enumerate(infos, 1):
        try:
            pending = await _has_pending_approval(info.id, uid)
        except Exception:
            pending = False
        lines.append(session_log.format_session_line(i, info, is_active=info.id == active, is_pending=pending, now=now))
    msg = "最近的会话：\n" + "\n".join(lines) + "\n用 /switch <序号或id前缀> 切换会话，/new [标题] 新建会话。"
    return _command_reply(msg, turn_start)


async def _cmd_switch(arg: str, turn_start: float, identity: Identity) -> ChatReply:
    uid = identity.user_id
    store = session_store.get_store()
    arg = arg.strip()
    if not arg:
        return _command_reply("用法：/switch <序号或id前缀>（先用 /sessions 查看列表）。", turn_start)

    # 序号与前缀都只在当前用户自己的会话里解析：找不到就说没找到，不透露别人的会话是否存在
    target = None
    if arg.isdigit():
        # 序号与 /sessions 的顺序一致：它们都是同一个“最近更新”排序的前缀
        infos = await asyncio.to_thread(store.list_sessions, _SESSIONS_CMD_MAX, user_id=uid)
        idx = int(arg)
        if 1 <= idx <= len(infos):
            target = infos[idx - 1]
    if target is None:
        try:
            target = await asyncio.to_thread(store.resolve_session, arg, user_id=uid)
        except ValueError as e:
            return _command_reply(f"{e}，请多写几位 id 前缀。", turn_start)
        if target is None:
            return _command_reply(f"没有找到会话「{arg}」，用 /sessions 查看列表。", turn_start)

    if target.id == store.get_active_session_id(user_id=uid):
        return _command_reply(f"已经在会话 {target.short_id}「{target.title or '（无标题）'}」里了。", turn_start)

    store.set_active_session(target.id, user_id=uid)
    msg = f"已切换到会话 {target.short_id}「{target.title or '（无标题）'}」（{target.turns}轮）。"
    state = await _agent_for(uid).aget_state(_state_config(target.id))
    if hitl.has_pending_interrupts(state):
        msg += "\n该会话有待授权的操作：\n\n" + hitl.format_hitl_warning(state)
        return _command_reply(msg, turn_start, awaiting=True)
    return _command_reply(msg, turn_start)


async def _cmd_reset(turn_start: float, identity: Identity) -> ChatReply:
    """只清上下文：删 checkpoint 线程 + 挂起态 + 重置沙箱（仅 owner）；不删事件日志，日志里追加 session_reset。"""
    uid = identity.user_id
    store = session_store.get_store()
    sid = store.get_active_session_id(user_id=uid)
    if _checkpointer is not None:
        await _checkpointer.adelete_thread(sid)
    store.clear_turn_state(sid)
    store.emit_event(sid, "session_reset", {})
    store.mark_processed(sid)
    if identity.is_owner:
        # 本机 Docker 沙箱是进程级全局、只属于 owner：member 的 /reset 绝不能去重置它
        try:
            from app.infra import sandbox
            await asyncio.to_thread(sandbox.reset_sandbox)
        except Exception as e:
            print(f"⚠️ 警告：沙箱重置失败 (非致命): {e}")
        try:
            await asyncio.to_thread(file_staging.clear_session_inbox, sid)         # 该会话上传的文件一并清掉
        except Exception as e:
            print(f"⚠️ 警告：清理会话 inbox 失败 (非致命): {type(e).__name__}")
    else:
        # member 的云端沙箱按 (用户, 会话) 隔离：只终止当前会话的那一个（/new、/switch 不动它）
        try:
            await asyncio.to_thread(modal_sandbox.reset_session, uid, sid)
        except Exception as e:
            print(f"⚠️ 警告：云端沙箱重置失败 (非致命): {e}")
    return _command_reply("对话上下文已清空，会话记录仍保留。已开始新的对话。", turn_start)


def _cmd_whoami(turn_start: float, identity: Identity) -> ChatReply:
    """所有人可用的纯文本自查：平台 ID（拿去写 OWNER_USER_IDS）、内部 ID、身份、CodeAgent 后端。"""
    msg = (
        f"平台 ID：{identity.raw_id}\n"
        f"内部 ID：{identity.user_id}\n"
        f"身份：{'管理员' if identity.is_owner else '成员'}\n"
        f"CodeAgent 后端：{'本机 Docker' if identity.is_owner else ('云端沙箱（Modal）' if _member_code_enabled else '未启用')}"
    )
    return _command_reply(msg, turn_start)


def _resolve_user_ref(store, ref: str):
    """/label /block /unblock 的目标：序号（与 /users 一致）或 user_id 前缀。返回 (用户, 错误文案)。"""
    ref = ref.strip()
    if ref.isdigit():
        users = store.list_users()
        idx = int(ref)
        if 1 <= idx <= len(users):
            return users[idx - 1], None
    try:
        user = store.resolve_user(ref)
    except ValueError as e:
        return None, f"{e}，请多写几位 id 前缀。"
    if user is None:
        return None, f"没有找到用户「{ref}」，用 /users 查看列表。"
    return user, None


def _user_display(user) -> str:
    return f"{user.label}（{user.user_id[:8]}）" if user.label else user.user_id[:8]


def _cmd_users(turn_start: float) -> ChatReply:
    store = session_store.get_store()
    users = store.list_users()
    if not users:
        return _command_reply("还没有任何用户记录。", turn_start)
    now = time.time()
    lines = []
    for i, u in enumerate(users, 1):
        sessions = store.count_sessions(user_id=u.user_id)
        lines.append(
            f"{i}. {u.label or '（无备注）'} | {u.user_id[:8]} | 最后活跃 {session_log.format_relative_time(u.last_seen, now)}"
            f" | {sessions} 个会话" + (" [已封禁]" if u.blocked else "")
        )
    msg = "用户列表：\n" + "\n".join(lines) + "\n用 /label <序号|id前缀> <备注> 加备注，/block 或 /unblock <序号|id前缀> 封禁 / 解封。"
    return _command_reply(msg, turn_start)


def _cmd_label(arg: str, turn_start: float) -> ChatReply:
    parts = arg.strip().split(None, 1)
    if not parts:
        return _command_reply("用法：/label <序号|id前缀> <备注>（备注留空会清除）。", turn_start)
    store = session_store.get_store()
    user, err = _resolve_user_ref(store, parts[0])
    if user is None:
        return _command_reply(err, turn_start)
    label = parts[1] if len(parts) > 1 else ""
    store.set_user_label(user.user_id, label)
    saved = store.get_user(user.user_id)
    return _command_reply(
        f"已给 {user.user_id[:8]} 设置备注「{saved.label}」。" if saved is not None and saved.label else f"已清除 {user.user_id[:8]} 的备注。",
        turn_start,
    )


async def _cmd_block(arg: str, turn_start: float, *, blocked: bool) -> ChatReply:
    cmd = "/block" if blocked else "/unblock"
    ref = arg.strip()
    if not ref:
        return _command_reply(f"用法：{cmd} <序号|id前缀>。", turn_start)
    store = session_store.get_store()
    user, err = _resolve_user_ref(store, ref)
    if user is None:
        return _command_reply(err, turn_start)
    try:
        store.set_user_blocked(user.user_id, blocked)
    except ValueError:
        return _command_reply("管理员不可被封禁。", turn_start)
    if not blocked:
        return _command_reply(f"已解封 {_user_display(user)}。", turn_start)
    msg = f"已封禁 {_user_display(user)}。"
    status, _killed = await _stop_current_turn(user.user_id)      # 顺手中止它正在跑的轮（member 不会触发任何宿主机终止原语）
    if status != "idle":
        msg += "并中止了它正在运行的任务。"
    recycled = 0
    try:
        recycled = await asyncio.to_thread(modal_sandbox.reset_user, user.user_id)     # 回收它的全部云端沙箱
    except Exception as e:
        print(f"⚠️ 警告：回收云端沙箱失败 (非致命): {e}")
    if recycled and recycled > 0:
        msg += f"已回收它的 {recycled} 个云端沙箱。"
    return _command_reply(msg, turn_start)


# ══════════════════════════════════════════════════════════════════════════════
# 消息入口：写入入站事件 -> wake(session_id)
# ══════════════════════════════════════════════════════════════════════════════

async def _prepare_user_input(
    text: str,
    image_url: Optional[str],
    turn_start: float,
) -> tuple[Optional[str], Optional[ChatReply]]:
    """新消息前置准备：图片 VLM 意图翻译。返回 (processed_text, 提前结束的回复)。"""
    processed_text = text
    if image_url:
        data_uri = await _load_image_as_data_uri(image_url)
        if not data_uri:
            cli_logger.log_turn_end("图片处理失败，请稍后重试。", 0, False, time.perf_counter() - turn_start)
            return None, ChatReply("图片处理失败，请稍后重试。", [])

        sys_prompt = (
            "你现在是一个意图翻译官。用户的系统中有一套纯文本的工具链（包括菜谱检索、高德地图、资料库等）。"
            "请观察用户上传的图片，结合用户的文字描述，识别出图片中的核心实体（如具体食材、地名、人物等），并将其转化为一句精简的纯文本请求。\n"
            "【示例】用户发了西红柿和鸡蛋的图片+文字'我能吃啥' -> 你的输出：'查询使用西红柿和鸡蛋制作的菜谱'。\n"
            "请只输出翻译后的纯文本指令，不要包含任何多余的解释。"
        )
        content = []
        if processed_text.strip():
            content.append({"type": "text", "text": processed_text.strip()})
        else:
            content.append({"type": "text", "text": "请分析这张图片，提取其中的核心实体或需求"})
        content.append({"type": "image_url", "image_url": {"url": data_uri}})

        vlm_start = time.perf_counter()
        try:
            vlm_res = await models.vlm.ainvoke(
                [
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=content)
                ],
                config={"callbacks": [cli_logger.CLIDebugCallbackHandler()]},
            )
            processed_text = extract_text(vlm_res.content).strip()
            cli_logger.log_vlm_translation(processed_text, time.perf_counter() - vlm_start)
        except Exception as e:
            print(f"[ERROR] VLM 意图翻译失败: {e}")
            traceback.print_exc()
            cli_logger.log_turn_end("图片理解失败，请稍后重试。", 0, False, time.perf_counter() - turn_start)
            return None, ChatReply("图片理解失败，请稍后重试。", [])

    if not processed_text.strip():
        cli_logger.log_turn_end("（空消息）", 0, False, time.perf_counter() - turn_start)
        return None, None
    return processed_text, None


async def _run_turn(
    uid: str, sid: str, text: str, image_url: Optional[str], turn_start: float
) -> Optional[ChatReply]:
    """一轮普通对话（HITL 检查 -> Y/N 恢复 / 图片转文字 -> 写入入站事件 -> wake）。

    跑在 _run_cancellable 创建的独立 task 里，/stop 只取消这个 task，不会波及调用方。
    """
    store = session_store.get_store()

    # 检查当前会话是否在等授权
    current_state = await _agent_for(uid).aget_state(_state_config(sid))
    if hitl.has_pending_interrupts(current_state):
        user_reply = text.strip().upper()
        if user_reply in ("Y", "N"):
            store.emit_event(sid, "approval", {"answer": user_reply})
            return await wake(sid, turn_start=turn_start)
        cli_logger.log_hitl_status("REMIND", user_reply)
        warn = hitl.format_hitl_warning(current_state)
        msg = f"⚠️ 当前正在等待授权确认，请回复 Y 允许或 N 拒绝。\n\n{warn}"
        cli_logger.log_turn_end(msg, 0, True, time.perf_counter() - turn_start)
        return ChatReply(msg, [], awaiting_approval=True)

    # 新消息流程：图片转文字（可选）-> 写入 user_message 事件 -> wake
    processed_text, early_reply = await _prepare_user_input(text, image_url, turn_start)
    if early_reply is not None or processed_text is None:
        return early_reply
    store.emit_event(
        sid,
        "user_message",
        {"text": text, "processed_text": processed_text, "has_image": bool(image_url)},
    )
    return await wake(sid, turn_start=turn_start)


async def _run_cancellable(coro, user_id: str, session_id: str, turn_start: float) -> Optional[ChatReply]:
    """把一轮对话放进独立 task 并登记该用户的句柄，使 /stop 能在用户锁之外取消它。

    被 /stop 取消的那一轮返回 None（plugins/chat.py 对 None 直接 return，不会多发一条回复）。
    禁止用 asyncio.current_task().cancel() 实现 /stop：当前 task 是 NoneBot handler / CLI 主 task，会误杀调用方。
    """
    task = asyncio.create_task(coro, name="chat-turn")
    handle = _TurnHandle(task=task, session_id=session_id, user_id=user_id, started_at=time.monotonic())
    _current_turns[user_id] = handle            # 与 create_task 之间不能有 await
    try:
        return await task
    except asyncio.CancelledError:
        me = asyncio.current_task()
        if handle.stop_requested and me is not None and me.cancelling() == 0:
            # 是内层 task 被 /stop 取消，而不是调用方自己被取消
            cli_logger.log_turn_end("（已被用户中止）", 0, False, time.perf_counter() - turn_start)
            return None
        raise
    finally:
        if _current_turns.get(user_id) is handle:
            del _current_turns[user_id]


def _kill_side_effects() -> int:
    """终止这一轮已在宿主机 / 沙箱里起的进程（阻塞调用，放线程里跑）。返回被终止的宿主机命令数。

    两个终止原语都是进程级全局的，调用方必须保证只有 owner 的 /stop 才走到这里。
    """
    from app.infra import host_shell, sandbox
    killed = 0
    try:
        killed = int(host_shell.kill_all_running())
    except Exception as e:
        print(f"⚠️ [/stop] 终止宿主机命令失败 (非致命): {e}")
    try:
        sandbox.interrupt_sandbox()
    except Exception as e:
        print(f"⚠️ [/stop] 中断沙箱失败 (非致命): {e}")
    return killed


async def _stop_current_turn(user_id: str) -> tuple[str, int]:
    """中止该用户当前正在运行的一轮（/stop、/reset、/block、关闭服务共用）。返回 (状态, 被终止的宿主机命令数)。

    状态：idle（没有在跑的轮）/ stopped（已收尾）/ timeout（已发出中止，但 AGENT_STOP_WAIT_SECONDS 内没收尾）。
    只取消 user_id 自己的那一轮，绝不碰别人的。顺序是先 cancel 再杀进程：反过来的话，被杀进程的“失败结果”会让图再往前走一步。
    """
    handle = _current_turns.get(user_id)
    if handle is None or handle.task.done():
        return "idle", 0
    now = time.monotonic()
    # 同一轮重复调用不再 cancel（会打断收尾代码），除非距上次已超过等待时限
    if not handle.stop_requested or now - handle.cancel_sent_at > config.AGENT_STOP_WAIT_SECONDS:
        handle.stop_requested = True
        handle.cancel_sent_at = now
        handle.task.cancel()
    killed = 0
    if user_id == OWNER_USER_ID:
        # 宿主机命令与 Docker 沙箱只有 owner 能用，而这两个终止原语是进程级全局的：member 的 /stop 绝不能触发它们，
        # 否则会杀掉 owner 正在跑的命令。
        killed = await asyncio.to_thread(_kill_side_effects)
    else:
        # member：只中断它自己这个会话的云端命令（起守护线程，立即返回）；绝不走到 _kill_side_effects，
        # 否则会杀掉 owner 正在跑的命令
        try:
            modal_sandbox.interrupt_session_nowait(user_id, handle.session_id)
        except Exception as e:
            print(f"⚠️ [/stop] 中断云端沙箱失败 (非致命): {e}")
    # 不用 wait_for：它超时会再取消一次
    await asyncio.wait({handle.task}, timeout=config.AGENT_STOP_WAIT_SECONDS)
    return ("stopped" if handle.task.done() else "timeout"), killed


async def _stop_all_turns() -> None:
    """关闭服务时中止所有用户还在跑的轮（并发进行，不让一个慢的拖住其他人）。"""
    uids = list(_current_turns)
    if uids:
        await asyncio.gather(*(_stop_current_turn(uid) for uid in uids), return_exceptions=True)


async def _cmd_stop(identity: Identity, turn_start: float) -> ChatReply:
    """/stop：在用户锁之外运行，只中止发送者自己的那一轮；不写 assistant_message、不提取记忆、不清 checkpoint。"""
    uid = identity.user_id
    handle = _current_turns.get(uid)
    status, killed = await _stop_current_turn(uid)
    store = session_store.get_store()
    sid = handle.session_id if handle is not None else store.get_active_session_id(user_id=uid)

    async def _pending_state():
        agent = _agents.get(_role_of(uid))
        if agent is None:
            return None
        try:
            state = await agent.aget_state(_state_config(sid))
        except Exception as e:
            print(f"⚠️ [/stop] 读取会话状态失败 (非致命): {e}")
            return None
        return state if hitl.has_pending_interrupts(state) else None

    if status == "idle":
        msg = "当前没有正在运行的任务。"
        pending = await _pending_state()
        if pending is not None:
            return _command_reply(msg + "有待授权的操作，回复 Y 允许 / N 拒绝，或 /reset 清空会话。", turn_start, awaiting=True)
        return _command_reply(msg, turn_start)

    if status == "timeout":
        return _command_reply(
            "已发送中止指令，但任务还在收尾，请稍后再发消息；仍无响应可再发一次 /stop。", turn_start
        )

    # 让日志留下中止前已落盘的进度（非致命）
    try:
        state = await _agent_for(uid).aget_state(_state_config(sid))
        await asyncio.to_thread(session_log.mirror_messages, store, sid, state.values.get("messages", []))
    except Exception as e:
        print(f"⚠️ [会话日志镜像失败]: {e}")

    msg = "已中止当前任务。"
    if killed:
        msg += f"已终止 {killed} 个正在运行的宿主机命令。"
    msg += "已经执行完的操作（点击、已写入的文件、已运行完的命令）不会回滚。"
    pending = await _pending_state()
    if pending is not None:
        # 授权恢复阶段被中止后图里仍有待处理的中断：回复必须与实际状态一致
        msg += "\n仍有待授权的操作：\n\n" + hitl.format_hitl_warning(pending) + "\n回复 Y 重试 / N 放弃 / 或 /reset 清空会话。"
        return _command_reply(msg, turn_start, awaiting=True)
    return _command_reply(msg, turn_start)


async def process_message(
    text: str,
    image_url: Optional[str] = None,
    *,
    identity: Identity,
    files: Optional[Sequence[IncomingFile]] = None,
) -> Optional[ChatReply]:
    """处理单条用户消息。不依赖 NoneBot，纯业务逻辑层。

    identity 由调用方（QQ 插件 / CLI）解析；它必须自洽，否则直接拒绝（失败即关闭，绝不回落成 owner）。
    files 是 QQ 私聊里随消息到达的文件（只有 QQ 插件在非空时才传）：落地与回执见 app/services/file_staging.py。
    返回 None 表示这一轮被 /stop 中止（调用方不应再回复）。
    """
    if not _initialized:
        return ChatReply("Agent 系统尚未初始化完成，请稍后再试。", [])

    if not _identity_consistent(identity):
        print(f"[WARN] 拒绝一条身份不自洽的消息: {identity!r:.200}")
        return ChatReply("身份无效，已拒绝。", [])

    uid = identity.user_id
    user, created = session_store.get_store().touch_user(uid)
    welcome = ""
    if created:
        print(
            f"👤 [chat_service] 新用户首次出现：平台 ID={identity.raw_id}，内部 ID={uid}，"
            f"身份={'管理员' if identity.is_owner else '成员'}"
        )
        if not identity.is_owner:
            welcome = config.MEMBER_WELCOME_TEXT.strip()

    # 被 /block 的 member：固定回复，不进入任何后续流程（owner 不可被封禁，库里即使被改坏也不会把 owner 锁在门外）
    if user.blocked and not identity.is_owner:
        return ChatReply(_BLOCKED_TEXT, [])

    reply = await _process_with_files(text, image_url, identity, files)
    if welcome and reply is not None:
        reply = ChatReply(welcome + "\n\n" + reply.text, reply.images, reply.awaiting_approval)
    return reply


async def _process_with_files(
    text: str, image_url: Optional[str], identity: Identity, files: Optional[Sequence[IncomingFile]]
) -> Optional[ChatReply]:
    """文件接收的分流：纯文件消息在锁外落地并回确定性回执（不调 LLM、不占用户锁）；
    其余消息先等本用户在途的文件下载（/stop 除外），之后与没有文件时完全一致。"""
    uid = identity.user_id
    parsed = _parse_command(text)
    incoming = list(files or ())
    pure_file = bool(incoming) and parsed is None and not text.strip() and not image_url

    # 1) 在途闸门：QQ 里"发文件 → 紧接着发文字"只隔 1~2 秒，而下载落地可能要几秒。无在途下载时同步返回
    if not pure_file and not (parsed is not None and parsed[0] == "/stop"):
        await file_staging.wait_for_pending(uid)

    # 2) 没有文件，或是命令（命令消息附带的文件直接忽略）：与改动前完全一致
    if not incoming or parsed is not None:
        return await _process_unblocked(text, image_url, identity)

    # 3) 门控：功能关闭 / member 未开放 / 云端 CodeAgent 未启用时，不下载、不落盘
    ok, why = file_staging.gate(identity, member_code_enabled=_member_code_enabled)
    if not ok:
        if pure_file:
            return ChatReply(why, [])
        reply = await _process_unblocked(text, image_url, identity)        # 带文字：忽略文件，照常回答，并在前面说明
        return reply if reply is None else ChatReply(why + "\n\n" + reply.text, reply.images, reply.awaiting_approval)

    # 4) 落地（锁外：一个慢下载不会让该用户后续消息都收到"忙"，正在跑的回合也不会让文件收不进来）
    sid = session_store.get_store().get_active_session_id(user_id=uid)
    outcome = await file_staging.stage_incoming(incoming, identity=identity, session_id=sid)

    # 5) 纯文件：确定性回执
    if pure_file:
        pending = False
        if outcome.staged:
            try:
                pending = await _has_pending_approval(sid, uid)
            except Exception:
                pending = False
        return ChatReply(file_staging.format_receipts(outcome, is_owner=identity.is_owner, approval_pending=pending), [])

    # 6) 带文字（少见）：成功的文件靠 collect / inject 自动进入本回合；失败的逐条说明放在回答前面
    reply = await _process_unblocked(text, image_url, identity)
    if reply is not None and outcome.errors:
        reply = ChatReply("\n".join(outcome.errors) + "\n\n" + reply.text, reply.images, reply.awaiting_approval)
    return reply


async def _process_unblocked(text: str, image_url: Optional[str], identity: Identity) -> Optional[ChatReply]:
    uid = identity.user_id
    parsed = _parse_command(text)

    # /stop 必须在用户锁之外：锁被正在运行的那一轮占着，等锁就永远停不下来
    if parsed is not None and parsed[0] == "/stop":
        stop_start = time.perf_counter()
        cli_logger.log_turn_start(text, None)     # 附带的图片忽略
        return await _cmd_stop(identity, stop_start)

    lock = _user_lock(uid)

    # /reset 忙时不干等：先中止自己的当前轮，再走原有 /reset 流程
    reset_prefix = ""
    if parsed is not None and parsed[0] == "/reset" and uid in _current_turns:
        await _stop_current_turn(uid)
        for _ in range(20):                       # 被取消的那一轮释放锁可能晚一拍，最多再等 1 秒
            if not lock.locked():
                break
            await asyncio.sleep(0.05)
        reset_prefix = "已先中止正在运行的任务。"

    # 该用户的锁已被占用：直接返回占用提示，不排队；不同用户互不影响
    if lock.locked():
        busy = "上一条消息还在处理中，请稍候再发。"
        turn = _current_turns.get(uid)
        if turn is not None:
            busy += f"（已运行 {int(time.monotonic() - turn.started_at)} 秒）"
        return ChatReply(busy + "如需中止请发送 /stop", [])

    async with lock:
        turn_start = time.perf_counter()
        cli_logger.log_turn_start(text, image_url)
        try:
            # 斜杠命令：放在 HITL 判断之前，这样 /reset 能一并丢弃挂起中的授权，/switch 也能在授权挂起时使用
            # 命令在锁内同步执行，不可被 /stop 取消
            if parsed is not None:
                reply = await _handle_command(parsed[0], parsed[1], turn_start, identity)
                if reset_prefix:
                    reply = ChatReply(reset_prefix + reply.text, reply.images, reply.awaiting_approval)
                return reply

            sid = session_store.get_store().get_active_session_id(user_id=uid)
            return await _run_cancellable(_run_turn(uid, sid, text, image_url, turn_start), uid, sid, turn_start)

        except GraphRecursionError as e:
            print(f"[WARN] 任务步骤超限 (GraphRecursionError): {e}")
            msg = "这个任务步骤太多，我先停下了，可以拆成几步再试。"
            cli_logger.log_turn_end(msg, 0, False, time.perf_counter() - turn_start)
            return ChatReply(
                text=msg,
                images=[],
                awaiting_approval=False,
            )
        except asyncio.TimeoutError:
            timeout_minutes = max(1, config.AGENT_TURN_TIMEOUT_SECONDS // 60)
            print(f"[WARN] 任务运行超时 (超过 {config.AGENT_TURN_TIMEOUT_SECONDS} 秒)")
            msg = f"这次处理超过 {timeout_minutes} 分钟还没完成，我先停下了，可以换个说法或拆成几步再试。"
            cli_logger.log_turn_end(msg, 0, False, time.perf_counter() - turn_start)
            return ChatReply(
                text=msg,
                images=[],
                awaiting_approval=False,
            )
        except Exception:
            traceback.print_exc()
            cli_logger.log_turn_end("处理时出了点问题，请稍后再试", 0, False, time.perf_counter() - turn_start)
            return ChatReply("处理时出了点问题，请稍后再试", [])


# ══════════════════════════════════════════════════════════════════════════════
# wake：无载荷唤醒，处理会话里所有未处理的入站事件
# ══════════════════════════════════════════════════════════════════════════════

def _failure_reason(exc: BaseException, user_id: str) -> tuple[str, str]:
    if isinstance(exc, asyncio.CancelledError):
        turn = _current_turns.get(user_id)
        if turn is not None and turn.stop_requested:
            return "user_stop", "用户发送 /stop 中止"
        return "cancelled", "任务被取消"
    if isinstance(exc, GraphRecursionError):
        return "recursion_limit", str(exc)[:200]
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout", f"超过 {config.AGENT_TURN_TIMEOUT_SECONDS} 秒"
    return "error", f"{type(exc).__name__}: {str(exc)[:200]}"


async def wake(session_id: str, *, turn_start: Optional[float] = None) -> Optional[ChatReply]:
    """按需触发（用户消息 / Y-N 审批）：读取该会话 seq > processed_seq 的入站事件，逐条处理。

    不携带载荷；没有待处理事件返回 None（幂等）；返回最后一条事件的 ChatReply。
    会话归属的用户决定三件事：用哪张图（owner / member）、请求上下文里的身份（工具层门控用）、
    member 是否要先过全局并发上限。
    处理失败时追加 turn_failed 并把异常抛给 process_message 转成友好文案；
    无论成功失败，最后都推进游标（不做崩溃自动重试）。
    """
    store = session_store.get_store()
    info = store.get_session(session_id)          # 不带 user_id：这里就是要查出会话归属于谁
    if info is None:
        return None
    user_id = validate_user_id(info.user_id)
    agent = _agent_for(user_id)
    pending = store.get_pending_events(session_id)
    if not pending:
        return None
    started = turn_start if turn_start is not None else time.perf_counter()
    cfg = _run_config(session_id, user_id)
    is_owner = user_id == OWNER_USER_ID
    token = set_request_context(RequestContext(user_id=user_id, session_id=session_id, is_owner=is_owner))
    sem = None if is_owner else _member_semaphore()
    reply: Optional[ChatReply] = None
    try:
        failed_at = 0
        try:
            if sem is not None:
                await sem.acquire()                # 排队期间可被 /stop 取消
            try:
                for i, ev in enumerate(pending):
                    failed_at = i
                    if ev.type == "user_message":
                        reply = await _run_user_message(session_id, user_id, ev, agent, cfg, started)
                    elif ev.type == "approval":
                        reply = await _run_approval(session_id, user_id, ev, agent, cfg, started)
            finally:
                if sem is not None:
                    sem.release()
        except (Exception, asyncio.CancelledError) as e:
            reason, detail = _failure_reason(e, user_id)
            if not is_owner and reason != "user_stop":
                # 超时 / 递归上限 / 异常 / 非 /stop 的取消：已经在云端沙箱里起了的命令不会因为 asyncio 任务被取消而停下（它跑在线程里）。
                # user_stop 已由 _stop_current_turn 处理，不重复
                try:
                    modal_sandbox.interrupt_session_nowait(user_id, session_id)
                except Exception as hook_err:
                    print(f"⚠️ 警告：中断云端沙箱失败 (非致命): {hook_err}")
            try:
                store.emit_event(session_id, "turn_failed", {"reason": reason, "detail": detail})
                for rest in pending[failed_at + 1:]:
                    store.emit_event(
                        session_id, "turn_abandoned", {"reason": "earlier_event_failed", "event_seq": rest.seq}
                    )
            except Exception as log_err:
                print(f"⚠️ [会话日志写入失败]: {log_err}")
            raise
    finally:
        try:
            store.mark_processed(session_id)
        except Exception as e:
            print(f"⚠️ [推进会话游标失败]: {e}")
        reset_request_context(token)
    return reply


async def _run_user_message(session_id: str, user_id: str, ev: Any, agent: Any, cfg: dict, turn_start: float) -> ChatReply:
    payload = ev.payload
    text = str(payload.get("text", ""))
    processed_text = str(payload.get("processed_text") or text)
    has_image = bool(payload.get("has_image"))

    outbox.begin()
    messages = await asyncio.to_thread(injector.build_initial_messages, processed_text, user_id=user_id)
    try:
        first_content = getattr(messages[-1], "content", "") if messages else ""
        if isinstance(first_content, str) and "<long_term_memory>" in first_content:
            mem_m = re.search(r"<long_term_memory>([\s\S]*?)</long_term_memory>", first_content)
            mem_text = mem_m.group(1).strip() if mem_m else ""
            cli_logger.log_memory_status(True, mem_text)
        else:
            cli_logger.log_memory_status(False)
    except Exception:
        pass

    messages = await _attach_uploaded_files(messages, session_id, user_id, ev)
    await asyncio.wait_for(
        agent.ainvoke({"messages": messages}, cfg),
        timeout=config.AGENT_TURN_TIMEOUT_SECONDS,
    )
    # 有图片的轮不提取记忆（用户原话不含图片内容）
    memory_text = None if has_image else (text.strip() or None)
    return await _finalize_round(session_id, user_id, agent, cfg, turn_start, memory_text)


async def _attach_uploaded_files(messages: list, session_id: str, user_id: str, ev: Any) -> list:
    """把本条消息之前上传、尚未注入过的文件清单加到最后一条 HumanMessage 前面。永不抛异常；
    只在 build_initial_messages 之后动 messages：RAG 查询、记忆抽取和事件日志里的文本都不含清单块。"""
    if not config.QQ_FILE_ENABLED:
        return messages
    try:
        notices = await file_staging.collect_pending_notices(session_id, user_id, upto_seq=ev.seq)
        if not notices:
            return messages
        block = file_staging.format_uploaded_files_block(notices, is_owner=(user_id == OWNER_USER_ID))
        return file_staging.inject_files_block(messages, block) if block else messages
    except Exception as e:                       # CancelledError 是 BaseException，不会被吞：/stop 照常生效
        print(f"⚠️ [file] 注入文件清单失败（非致命）: {type(e).__name__}")
        return messages


async def _run_approval(session_id: str, user_id: str, ev: Any, agent: Any, cfg: dict, turn_start: float) -> ChatReply:
    store = session_store.get_store()
    answer = str(ev.payload.get("answer", "")).strip().upper()
    state = await agent.aget_state(cfg)
    if not hitl.has_pending_interrupts(state):
        # 授权已过期 / 被 /reset 丢弃：不调用模型
        msg = "当前没有待授权的操作。"
        store.emit_event(session_id, "turn_failed", {"reason": "no_pending_approval", "detail": answer})
        cli_logger.log_turn_end(msg, 0, False, time.perf_counter() - turn_start)
        return ChatReply(msg, [])

    cli_logger.log_hitl_status("RESUMED", answer)
    turn_state = store.load_turn_state(session_id)
    outbox.resume(list(turn_state.outbox) if turn_state else [])
    store.clear_turn_state(session_id)
    resume_value = hitl.build_resume(state, answer)
    await asyncio.wait_for(
        agent.ainvoke(Command(resume=resume_value), cfg),
        timeout=config.AGENT_TURN_TIMEOUT_SECONDS,
    )
    memory_text = turn_state.user_text if turn_state else None
    return await _finalize_round(session_id, user_id, agent, cfg, turn_start, memory_text)


async def _finalize_round(
    session_id: str,
    user_id: str,
    agent: Any,
    cfg: dict,
    turn_start: float,
    memory_text: Optional[str],
) -> ChatReply:
    """回合收尾：镜像事件日志、检查中断状态、提取最终回复、收集发件箱图片并调度后台记忆提取。"""
    store = session_store.get_store()
    after_state = await agent.aget_state(cfg)
    state_msgs = after_state.values.get("messages", [])

    try:
        await asyncio.to_thread(session_log.mirror_messages, store, session_id, state_msgs)
    except Exception as e:
        print(f"⚠️ [会话日志镜像失败]: {e}")

    if hitl.has_pending_interrupts(after_state):
        pending_images = outbox.current()
        warning = hitl.format_hitl_warning(after_state)
        # 挂起态（原话 + 已产生的图片）落库，重启后用户回复 Y/N 仍能恢复；不做任何 checkpoint 清理
        store.save_turn_state(session_id, memory_text, pending_images)
        store.emit_event(session_id, "approval_request", {"text": warning})
        cli_logger.log_hitl_status("SUSPEND", warning)
        cli_logger.log_turn_end(warning, len(pending_images), True, time.perf_counter() - turn_start)
        return ChatReply(
            text=warning,
            images=[],
            awaiting_approval=True,
        )

    final_text = ""
    for m in reversed(state_msgs):
        if isinstance(m, AIMessage):
            t = extract_text(getattr(m, "content", "")).strip()
            if t:
                final_text = t
                break

    if not final_text:
        final_text = "（没有生成回复）"

    round_images = outbox.current()
    # 图片字节永远不进事件日志，只记张数
    store.emit_event(session_id, "assistant_message", {"text": final_text, "image_count": len(round_images)})
    store.clear_turn_state(session_id)

    try:
        if memory_text:
            mem_task = asyncio.create_task(extractor.extract_and_store_memory(memory_text, user_id=user_id))
            _background_tasks.add(mem_task)
            mem_task.add_done_callback(_background_tasks.discard)
    except Exception as bg_err:
        print(f"⚠️ [记忆提取调度异常]: {bg_err}")

    # 清理只在“正常完成”时做：失败 / 超时 / 挂起时被取消的后台线程可能还在写 checkpoint
    if config.CHECKPOINT_PRUNE_NESTED and _checkpointer is not None:
        await checkpointer.prune_nested_checkpoints(_checkpointer, session_id)

    cli_logger.log_turn_end(final_text, len(round_images), False, time.perf_counter() - turn_start)
    return ChatReply(
        text=final_text,
        images=round_images,
        awaiting_approval=False,
    )


async def recover_orphans() -> int:
    """启动恢复：处理崩溃前遗留的未处理入站事件（所有用户的会话），返回被标记放弃的事件数。

    先把已落盘 checkpoint 里的消息镜像进日志（按会话所属用户的角色选图），再对每条待处理事件追加 turn_abandoned 并推进游标。
    不自动重跑：回复通道已断，重放还可能让已批准的宿主机命令至少再执行一次。
    挂起中的授权不受影响：它的状态在 checkpoint 和 turn_state 里，用户回复 Y/N 照常恢复。
    """
    store = session_store.get_store()
    abandoned = 0
    for sid in await asyncio.to_thread(store.sessions_with_pending):
        info = store.get_session(sid)
        agent = _agents.get(_role_of(info.user_id)) if info is not None else None
        if agent is not None:
            try:
                state = await agent.aget_state(_state_config(sid))
                await asyncio.to_thread(session_log.mirror_messages, store, sid, state.values.get("messages", []))
            except Exception as e:
                print(f"⚠️ [启动恢复] 会话 {sid[:8]} 镜像失败 (非致命): {e}")
        for ev in store.get_pending_events(sid):
            store.emit_event(sid, "turn_abandoned", {"reason": "process_restart", "event_seq": ev.seq})
            abandoned += 1
        store.mark_processed(sid)
    if abandoned:
        print(f"⚠️ [chat_service] 启动恢复：{abandoned} 条上次崩溃时未处理的消息已标记放弃（不自动重跑）。")
    return abandoned
