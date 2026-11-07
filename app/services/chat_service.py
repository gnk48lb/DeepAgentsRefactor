"""单用户多模态会话状态机与消息调度服务 (所在层: L5)

会话状态不在进程内存里：会话 / 事件日志 / 挂起态在 SessionStore（sessions.db），
模型上下文与 HITL 中断在 checkpoint（checkpoints.db），thread_id == session_id。
process_message 只负责“写入入站事件 + wake(session_id)”，wake 不带载荷，载荷全部从事件日志里读。
"""

import asyncio
import base64
import re
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Set, Any
import httpx

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.errors import GraphRecursionError

from app import config
from app.agents import orchestrator
from app.memory import injector, extractor
from app.services import startup, session_log
from app.runtime import outbox, hitl, cli_logger
from app.infra import models, session_store, checkpointer
from app.utils.text import extract_text
from app.utils.images import guess_mime_type


@dataclass
class ChatReply:
    text: str
    images: list                  # [{"type": "image_url", "image_url": {"url": "data:image/..."}}]
    awaiting_approval: bool = False

# ── 单用户模式：全局只有一个 Agent、一把锁；会话状态全部在 SessionStore / checkpoint 里 ──
_initialized: bool = False
_agent = None
# init_service 里保存的 AsyncSqliteSaver：/reset 删线程、每轮结束清理嵌套 checkpoint 时用
_checkpointer = None
# 同一时刻只处理一条消息（防止连发消息并发踩同一会话 / HITL 中断状态）
_chat_lock = asyncio.Lock()
_background_tasks: Set[asyncio.Task] = set()

_SLASH_COMMANDS = {"/new", "/sessions", "/switch", "/reset"}
_SESSIONS_CMD_MAX = 30


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


def get_active_session_id() -> str:
    """当前活动会话 id（供斜杠命令与测试使用）。"""
    return session_store.get_store().get_active_session_id()


def _state_config(session_id: str) -> dict:
    """只读查询状态用的最小 config。"""
    return {"configurable": {"thread_id": session_id}}


def _run_config(session_id: str) -> dict:
    return {
        "recursion_limit": config.AGENT_RECURSION_LIMIT,
        "configurable": {"thread_id": session_id},
        "callbacks": [cli_logger.CLIDebugCallbackHandler()],
    }


async def _has_pending_approval(session_id: str) -> bool:
    """该会话的 checkpoint 里是否有未处理的 interrupt。"""
    if _agent is None:
        return False
    state = await _agent.aget_state(_state_config(session_id))
    return hitl.has_pending_interrupts(state)


async def init_service() -> None:
    """初始化数据库、会话存储、checkpointer、MCP 服务并构建主 Agent（单用户：只构建一个全功能 Agent）"""
    global _initialized, _agent, _checkpointer
    models.validate_required_keys()
    print("🚀 [chat_service] 正在初始化数据环境与 Agent 服务...")
    saver = None
    try:
        await asyncio.to_thread(startup.run_global_database_init)
        store = await asyncio.to_thread(session_store.get_store)
        sid = await asyncio.to_thread(store.get_active_session_id)
        saver = await checkpointer.open_checkpointer()
        _checkpointer = saver
        _agent = await orchestrator.build_main_agent(checkpointer=saver)
        await recover_orphans()
        info = store.get_session(sid)
        _initialized = True
    except Exception:
        _agent = None
        _checkpointer = None
        await checkpointer.close_checkpointer(saver)
        session_store.close_store()
        raise
    if info is not None:
        print(f"💬 [chat_service] 当前会话 {info.short_id}「{info.title or '（无标题）'}」，事件 {info.events} 条。")
    print("✅ [chat_service] 初始化完成。")


def is_initialized() -> bool:
    return _initialized


async def shutdown_service() -> None:
    """关闭系统服务并释放 MCP 资源"""
    global _initialized, _agent, _checkpointer
    print("🛑 [chat_service] 正在关闭服务...")
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
    _agent = None
    _initialized = False
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


async def _handle_command(cmd: str, arg: str, turn_start: float) -> ChatReply:
    if cmd == "/new":
        return await _cmd_new(arg, turn_start)
    if cmd == "/sessions":
        return await _cmd_sessions(arg, turn_start)
    if cmd == "/switch":
        return await _cmd_switch(arg, turn_start)
    return await _cmd_reset(turn_start)


async def _cmd_new(title: str, turn_start: float) -> ChatReply:
    store = session_store.get_store()
    old_sid = store.get_active_session_id()
    old_pending = await _has_pending_approval(old_sid)
    info = store.create_session(title, activate=True)
    msg = f"已新建会话 {info.short_id}" + (f"「{info.title}」" if info.title else "（发出第一条消息后自动命名）") + "，旧会话已原样保留。"
    if old_pending:
        msg += f"\n提醒：上一个会话 {old_sid[:8]} 还有待授权的操作，用 /switch {old_sid[:8]} 切回去后回复 Y 或 N。"
    return _command_reply(msg, turn_start)


async def _cmd_sessions(arg: str, turn_start: float) -> ChatReply:
    store = session_store.get_store()
    limit = config.SESSION_LIST_LIMIT
    if arg.strip().isdigit():
        limit = max(1, min(int(arg.strip()), _SESSIONS_CMD_MAX))
    infos = await asyncio.to_thread(store.list_sessions, limit)
    active = store.get_active_session_id()
    now = time.time()
    lines = []
    for i, info in enumerate(infos, 1):
        try:
            pending = await _has_pending_approval(info.id)
        except Exception:
            pending = False
        lines.append(session_log.format_session_line(i, info, is_active=info.id == active, is_pending=pending, now=now))
    msg = "最近的会话：\n" + "\n".join(lines) + "\n用 /switch <序号或id前缀> 切换会话，/new [标题] 新建会话。"
    return _command_reply(msg, turn_start)


async def _cmd_switch(arg: str, turn_start: float) -> ChatReply:
    store = session_store.get_store()
    arg = arg.strip()
    if not arg:
        return _command_reply("用法：/switch <序号或id前缀>（先用 /sessions 查看列表）。", turn_start)

    target = None
    if arg.isdigit():
        # 序号与 /sessions 的顺序一致：它们都是同一个“最近更新”排序的前缀
        infos = await asyncio.to_thread(store.list_sessions, _SESSIONS_CMD_MAX)
        idx = int(arg)
        if 1 <= idx <= len(infos):
            target = infos[idx - 1]
    if target is None:
        try:
            target = await asyncio.to_thread(store.resolve_session, arg)
        except ValueError as e:
            return _command_reply(f"{e}，请多写几位 id 前缀。", turn_start)
        if target is None:
            return _command_reply(f"没有找到会话「{arg}」，用 /sessions 查看列表。", turn_start)

    if target.id == store.get_active_session_id():
        return _command_reply(f"已经在会话 {target.short_id}「{target.title or '（无标题）'}」里了。", turn_start)

    store.set_active_session(target.id)
    msg = f"已切换到会话 {target.short_id}「{target.title or '（无标题）'}」（{target.turns}轮）。"
    state = await _agent.aget_state(_state_config(target.id))
    if hitl.has_pending_interrupts(state):
        msg += "\n该会话有待授权的操作：\n\n" + hitl.format_hitl_warning(state)
        return _command_reply(msg, turn_start, awaiting=True)
    return _command_reply(msg, turn_start)


async def _cmd_reset(turn_start: float) -> ChatReply:
    """只清上下文：删 checkpoint 线程 + 挂起态 + 重置沙箱；不删事件日志，日志里追加 session_reset。"""
    store = session_store.get_store()
    sid = store.get_active_session_id()
    if _checkpointer is not None:
        await _checkpointer.adelete_thread(sid)
    store.clear_turn_state(sid)
    store.emit_event(sid, "session_reset", {})
    store.mark_processed(sid)
    try:
        from app.infra import sandbox
        await asyncio.to_thread(sandbox.reset_sandbox)
    except Exception as e:
        print(f"⚠️ 警告：沙箱重置失败 (非致命): {e}")
    return _command_reply("对话上下文已清空，会话记录仍保留。已开始新的对话。", turn_start)


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


async def process_message(text: str, image_url: Optional[str] = None) -> Optional[ChatReply]:
    """处理单条用户消息（单用户模式：不区分用户、无权限校验）。不依赖 NoneBot，纯业务逻辑层。"""
    if not _initialized:
        return ChatReply("Agent 系统尚未初始化完成，请稍后再试。", [])

    # 全局一把锁：上一条还在处理时直接返回占用提示，不排队
    if _chat_lock.locked():
        return ChatReply("上一条消息还在处理中，请稍候再发。", [])

    async with _chat_lock:
        turn_start = time.perf_counter()
        cli_logger.log_turn_start(text, image_url)
        try:
            # 斜杠命令：放在 HITL 判断之前，这样 /reset 能一并丢弃挂起中的授权，/switch 也能在授权挂起时使用
            parsed = _parse_command(text)
            if parsed is not None:
                return await _handle_command(parsed[0], parsed[1], turn_start)

            store = session_store.get_store()
            sid = store.get_active_session_id()

            # 检查当前会话是否在等授权
            current_state = await _agent.aget_state(_state_config(sid))
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

def _failure_reason(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, GraphRecursionError):
        return "recursion_limit", str(exc)[:200]
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout", f"超过 {config.AGENT_TURN_TIMEOUT_SECONDS} 秒"
    return "error", f"{type(exc).__name__}: {str(exc)[:200]}"


async def wake(session_id: str, *, turn_start: Optional[float] = None) -> Optional[ChatReply]:
    """按需触发（用户消息 / Y-N 审批）：读取该会话 seq > processed_seq 的入站事件，逐条处理。

    不携带载荷；没有待处理事件返回 None（幂等）；返回最后一条事件的 ChatReply。
    处理失败时追加 turn_failed 并把异常抛给 process_message 转成友好文案；
    无论成功失败，最后都推进游标（不做崩溃自动重试）。
    """
    store = session_store.get_store()
    agent = _agent
    if agent is None:
        raise RuntimeError("Agent 尚未初始化")
    pending = store.get_pending_events(session_id)
    if not pending:
        return None
    started = turn_start if turn_start is not None else time.perf_counter()
    cfg = _run_config(session_id)
    reply: Optional[ChatReply] = None
    try:
        for i, ev in enumerate(pending):
            try:
                if ev.type == "user_message":
                    reply = await _run_user_message(session_id, ev, agent, cfg, started)
                elif ev.type == "approval":
                    reply = await _run_approval(session_id, ev, agent, cfg, started)
            except Exception as e:
                reason, detail = _failure_reason(e)
                try:
                    store.emit_event(session_id, "turn_failed", {"reason": reason, "detail": detail})
                    for rest in pending[i + 1:]:
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
    return reply


async def _run_user_message(session_id: str, ev: Any, agent: Any, cfg: dict, turn_start: float) -> ChatReply:
    payload = ev.payload
    text = str(payload.get("text", ""))
    processed_text = str(payload.get("processed_text") or text)
    has_image = bool(payload.get("has_image"))

    outbox.begin()
    messages = await asyncio.to_thread(injector.build_initial_messages, processed_text)
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

    await asyncio.wait_for(
        agent.ainvoke({"messages": messages}, cfg),
        timeout=config.AGENT_TURN_TIMEOUT_SECONDS,
    )
    # 有图片的轮不提取记忆（用户原话不含图片内容）
    memory_text = None if has_image else (text.strip() or None)
    return await _finalize_round(session_id, agent, cfg, turn_start, memory_text)


async def _run_approval(session_id: str, ev: Any, agent: Any, cfg: dict, turn_start: float) -> ChatReply:
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
    return await _finalize_round(session_id, agent, cfg, turn_start, memory_text)


async def _finalize_round(
    session_id: str,
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
            mem_task = asyncio.create_task(extractor.extract_and_store_memory(memory_text))
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
    """启动恢复：处理崩溃前遗留的未处理入站事件，返回被标记放弃的事件数。

    先把已落盘 checkpoint 里的消息镜像进日志，再对每条待处理事件追加 turn_abandoned 并推进游标。
    不自动重跑：回复通道已断，重放还可能让已批准的宿主机命令至少再执行一次。
    挂起中的授权不受影响：它的状态在 checkpoint 和 turn_state 里，用户回复 Y/N 照常恢复。
    """
    store = session_store.get_store()
    abandoned = 0
    for sid in await asyncio.to_thread(store.sessions_with_pending):
        if _agent is not None:
            try:
                state = await _agent.aget_state(_state_config(sid))
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
