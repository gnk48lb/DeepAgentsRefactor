"""单用户多模态会话状态机与消息调度服务 (所在层: L5)"""

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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from app import config
from app.agents import orchestrator
from app.memory import injector, extractor
from app.services import startup
from app.runtime import outbox, hitl, cli_logger
from app.infra import models
from app.utils.text import extract_text
from app.utils.images import guess_mime_type


@dataclass
class ChatReply:
    text: str
    images: list                  # [{"type": "image_url", "image_url": {"url": "data:image/..."}}]
    awaiting_approval: bool = False

# ── 单用户模式：全局只有一个 Agent、一把锁、一个固定会话 ──
THREAD_ID = "main"                       # 固定 thread_id：私聊 / 群聊共用同一份对话上下文
_initialized: bool = False
_agent = None
_chat_lock = asyncio.Lock()              # 同一时刻只处理一条消息（防止连发消息并发踩同一会话 / HITL 中断状态）
_pending_outbox: Optional[list] = None   # HITL 中断期间暂存本轮已产生的图片
_checkpointer = None                     # init_service 里保存，/reset 时用它清空固定会话
RESET_COMMANDS = {"/reset", "/new", "/新对话"}
_background_tasks: Set[asyncio.Task] = set()


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


async def init_service() -> None:
    """初始化数据库、MCP 服务并构建主 Agent（单用户：只构建一个全功能 Agent）"""
    global _initialized, _agent, _checkpointer
    models.validate_required_keys()
    print("🚀 [chat_service] 正在初始化数据环境与 Agent 服务...")
    await asyncio.to_thread(startup.run_global_database_init)
    _checkpointer = MemorySaver()
    _agent = await orchestrator.build_main_agent(checkpointer=_checkpointer)
    _initialized = True
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
    _agent = None
    _checkpointer = None
    _initialized = False
    print("✅ [chat_service] 服务已关闭。")


async def _handle_reset(turn_start: float) -> ChatReply:
    """清空固定会话：丢弃挂起中的授权与历史上下文。"""
    global _pending_outbox
    if _checkpointer is not None:
        await _checkpointer.adelete_thread(THREAD_ID)
    _pending_outbox = None
    cli_logger.log_turn_end("已清空对话上下文，开始新的对话。", 0, False, time.perf_counter() - turn_start)
    return ChatReply("已清空对话上下文，开始新的对话。", [])


async def _handle_hitl(
    text: str,
    current_state: Any,
    agent: Any,
    agent_config: dict,
    turn_start: float,
) -> Optional[ChatReply]:
    """处理挂起中的 HITL 授权状态：Y/N 则恢复执行；其他输入则给出提醒。"""
    global _pending_outbox
    user_reply = text.strip().upper()
    if user_reply in ("Y", "N"):
        cli_logger.log_hitl_status("RESUMED", user_reply)
        resumed_outbox = _pending_outbox or []
        _pending_outbox = None
        outbox.resume(resumed_outbox)
        await asyncio.wait_for(
            agent.ainvoke(
                Command(resume=hitl.build_resume(current_state, user_reply)),
                agent_config,
            ),
            timeout=config.AGENT_TURN_TIMEOUT_SECONDS,
        )
        return None
    else:
        cli_logger.log_hitl_status("REMIND", user_reply)
        warn = hitl.format_hitl_warning(current_state)
        msg = f"⚠️ 当前正在等待授权确认，请回复 Y 允许或 N 拒绝。\n\n{warn}"
        cli_logger.log_turn_end(msg, 0, True, time.perf_counter() - turn_start)
        return ChatReply(msg, [], awaiting_approval=True)


async def _prepare_round_input(
    text: str,
    image_url: Optional[str],
    turn_start: float,
) -> tuple[Optional[list], Optional[ChatReply]]:
    """新消息前置准备：图片 VLM 意图翻译、记忆注入与初次消息拼装。"""
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

    outbox.begin()
    messages = await asyncio.to_thread(
        injector.build_initial_messages,
        processed_text,
    )
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

    return messages, None


async def _finalize_round(
    agent: Any,
    agent_config: dict,
    turn_start: float,
) -> ChatReply:
    """回合收尾：检查中断状态、提取最终回复、收集发件箱图片并调度后台记忆提取。"""
    global _pending_outbox
    after_state = await agent.aget_state(agent_config)
    if hitl.has_pending_interrupts(after_state):
        _pending_outbox = outbox.current()
        warning = hitl.format_hitl_warning(after_state)
        cli_logger.log_hitl_status("SUSPEND", warning)
        cli_logger.log_turn_end(warning, len(_pending_outbox or []), True, time.perf_counter() - turn_start)
        return ChatReply(
            text=warning,
            images=[],
            awaiting_approval=True,
        )

    state_msgs = after_state.values.get("messages", [])
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

    try:
        last_human_text = ""
        for m in reversed(state_msgs):
            if isinstance(m, HumanMessage):
                last_human_text = extract_text(getattr(m, "content", "")).strip()
                break

        if last_human_text and final_text:
            recent_chat = f"human: {last_human_text}\nai: {final_text}"
            mem_task = asyncio.create_task(extractor.extract_and_store_memory(recent_chat))
            _background_tasks.add(mem_task)
            mem_task.add_done_callback(_background_tasks.discard)
    except Exception as bg_err:
        print(f"⚠️ [记忆提取调度异常]: {bg_err}")

    cli_logger.log_turn_end(final_text, len(round_images), False, time.perf_counter() - turn_start)
    return ChatReply(
        text=final_text,
        images=round_images,
        awaiting_approval=False,
    )


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
            # 清空固定会话：放在 HITL 判断之前，这样挂起中的授权也能被一并丢弃
            if text.strip().lower() in RESET_COMMANDS:
                return await _handle_reset(turn_start)

            agent = _agent
            agent_config = {
                "recursion_limit": config.AGENT_RECURSION_LIMIT,
                "configurable": {"thread_id": THREAD_ID},
                "callbacks": [cli_logger.CLIDebugCallbackHandler()],
            }

            # 2. 检查是否在等授权
            current_state = await agent.aget_state(agent_config)
            if hitl.has_pending_interrupts(current_state):
                hitl_reply = await _handle_hitl(text, current_state, agent, agent_config, turn_start)
                if hitl_reply is not None:
                    return hitl_reply
            else:
                # 3. 新消息流程：图片转文字（可选）-> 拼装初次输入 -> 触发 Agent
                messages, early_reply = await _prepare_round_input(text, image_url, turn_start)
                if early_reply is not None or messages is None:
                    return early_reply

                await asyncio.wait_for(
                    agent.ainvoke({"messages": messages}, agent_config),
                    timeout=config.AGENT_TURN_TIMEOUT_SECONDS,
                )

            # 4 & 5. 检查后续中断或完成收尾与记忆提取
            return await _finalize_round(agent, agent_config, turn_start)

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
