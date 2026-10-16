import asyncio
import base64
import os
import traceback
from dataclasses import dataclass
from typing import Optional, List, Dict, Set
import httpx

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

import config
from app import database, deep_agent, outbox, models
from app.utils import extract_text

@dataclass
class ChatReply:
    text: str
    images: list                  # [{"type": "image_url", "image_url": {"url": "data:image/..."}}]
    awaiting_approval: bool = False

# 全局服务状态与锁
_initialized: bool = False
_owner_agent = None
_normal_agent = None
_user_locks: Dict[str, asyncio.Lock] = {}
_pending_outbox: Dict[str, list] = {}
_background_tasks: Set[asyncio.Task] = set()

def _guess_mime_type(data: bytes) -> str:
    """根据字节头判断图片的 MIME 类型"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    elif data.startswith(b"GIF8"):
        return "image/gif"
    elif data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"

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

        mime = _guess_mime_type(img_bytes)
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64_str}"
    except Exception as e:
        print(f"[ERROR] 读取/下载图片失败: {e}")
        return None

def _format_hitl_warning(state) -> str:
    """从图状态中提取即将执行的危险工具信息，格式化为用户友好的授权拦截提示。"""
    try:
        tasks = getattr(state, "tasks", [])
        if not tasks:
            return "⚠️ 请求执行一项敏感操作，请回复 Y 允许或 N 拒绝。"

        item = None
        for task in reversed(tasks):
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_item = task.interrupts[-1]
                if isinstance(interrupt_item, str):
                    return interrupt_item

                val = interrupt_item.value if hasattr(interrupt_item, "value") else interrupt_item
                if isinstance(val, str):
                    return val

                item = val[0] if isinstance(val, list) and val else val
                break

        if not item or not isinstance(item, dict):
            return "⚠️ 请求执行一项敏感操作，请回复 Y 允许或 N 拒绝。"

        t_name = item.get("tool_name", "未知工具")
        t_args = item.get("tool_args", {})
        target = t_args.get("path") or t_args.get("destination") or t_args.get("directory") or str(t_args)

        return (
            f"⚠️ 敏感操作授权拦截\n"
            f"──────────────────\n"
            f"操作: [{t_name}]\n"
            f"目标: {target}\n"
            f"──────────────────\n"
            f"回复 Y 允许，回复 N 拒绝。"
        )
    except Exception as e:
        print(f"⚠️ 格式化警告失败: {e}")
    return "⚠️ 请求执行一项敏感操作，请回复 Y 允许或 N 拒绝。"

def _has_pending_interrupts(state) -> bool:
    """判断状态中是否有待处理的中断"""
    tasks = getattr(state, "tasks", [])
    for t in tasks:
        if hasattr(t, "interrupts") and t.interrupts:
            return True
    return False

async def init_service() -> None:
    """初始化数据库、MCP服务并构建主 Agent"""
    global _initialized, _owner_agent, _normal_agent
    print("🚀 [chat_service] 正在初始化数据环境与 Agent 服务...")

    # 数据库同步初始化放入线程
    await asyncio.to_thread(database.run_global_database_init)

    owner_ids = getattr(config, "OWNER_QQ_IDS", set())
    shared_checkpointer = MemorySaver()

    if not owner_ids:
        print("\n" + "!" * 60)
        print("⚠️ 警告：config.OWNER_QQ_IDS 为空！所有用户均享有特权专家权限！")
        print("!" * 60 + "\n")
        _owner_agent = await deep_agent.build_main_agent(
            checkpointer=shared_checkpointer,
            include_privileged=True,
        )
        _normal_agent = _owner_agent
    else:
        print(f"🔒 [chat_service] 主人 QQ 列表配置: {owner_ids}，正在构建特权版与受限版 Agent...")
        _owner_agent = await deep_agent.build_main_agent(
            checkpointer=shared_checkpointer,
            include_privileged=True,
        )
        _normal_agent = await deep_agent.build_main_agent(
            checkpointer=shared_checkpointer,
            include_privileged=False,
        )

    _initialized = True
    print("✅ [chat_service] 初始化完成。")

def is_initialized() -> bool:
    return _initialized

def _select_agent(user_id: str):
    owner_ids = getattr(config, "OWNER_QQ_IDS", set())
    if not owner_ids or str(user_id).strip() in owner_ids:
        return _owner_agent
    return _normal_agent

async def process_message(user_id: str, text: str, image_url: Optional[str] = None) -> Optional[ChatReply]:
    """
    处理单条用户消息。
    不依赖 NoneBot，纯纯的业务逻辑层。
    """
    if not _initialized:
        return ChatReply("Agent 系统尚未初始化完成，请稍后再试。", [])

    owner_ids = getattr(config, "OWNER_QQ_IDS", set())
    if owner_ids and str(user_id).strip() not in owner_ids:
        print(f"[chat] 非主人用户 {user_id}")

    # 1. 每个 user_id 一把锁，并发时直接返回占用提示不排队
    lock = _user_locks.setdefault(user_id, asyncio.Lock())
    if lock.locked():
        return ChatReply("上一条消息还在处理中，请稍候再发。", [])

    async with lock:
        try:
            agent = _select_agent(user_id)
            thread_id = str(user_id).strip()
            agent_config = {
                "recursion_limit": config.AGENT_RECURSION_LIMIT,
                "configurable": {"thread_id": thread_id},
            }

            # 2. 检查是否在等授权
            current_state = await agent.aget_state(agent_config)
            if _has_pending_interrupts(current_state):
                user_reply = text.strip().upper()
                if user_reply in ("Y", "N"):
                    resumed_outbox = _pending_outbox.pop(user_id, [])
                    outbox.resume(resumed_outbox)
                    await agent.ainvoke(Command(resume=user_reply), agent_config)
                else:
                    warning = _format_hitl_warning(current_state)
                    return ChatReply(
                        text=f"⚠️ 当前正在等待授权确认，请回复 Y 允许或 N 拒绝。\n\n{warning}",
                        images=[],
                        awaiting_approval=True,
                    )
            else:
                # 3. 新消息流程
                processed_text = text
                if image_url:
                    data_uri = await _load_image_as_data_uri(image_url)
                    if not data_uri:
                        return ChatReply("图片处理失败，请稍后重试。", [])

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

                    try:
                        vlm_res = await models.vlm.ainvoke([
                            SystemMessage(content=sys_prompt),
                            HumanMessage(content=content)
                        ])
                        processed_text = extract_text(vlm_res.content).strip()
                    except Exception as e:
                        print(f"[ERROR] VLM 意图翻译失败: {e}")
                        traceback.print_exc()
                        return ChatReply("图片理解失败，请稍后重试。", [])

                if not processed_text.strip():
                    return None

                outbox.begin()
                messages = await asyncio.to_thread(
                    deep_agent.build_initial_messages,
                    processed_text,
                    user_id,
                )
                await agent.ainvoke({"messages": messages}, agent_config)

            # 4. 一轮 ainvoke 结束后再次检查是否有中断
            after_state = await agent.aget_state(agent_config)
            if _has_pending_interrupts(after_state):
                curr_lst = outbox.current()
                _pending_outbox[user_id] = curr_lst
                warning = _format_hitl_warning(after_state)
                return ChatReply(
                    text=warning,
                    images=[],
                    awaiting_approval=True,
                )

            # 5. 没有中断：正常回复
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
            elif len(final_text) > config.QQ_MAX_REPLY_CHARS:
                final_text = final_text[:config.QQ_MAX_REPLY_CHARS] + "\n（内容过长已截断）"

            # 收集本轮发件箱图片
            round_images = outbox.current()

            # 调度后台记忆提取
            try:
                last_human_text = ""
                for m in reversed(state_msgs):
                    if isinstance(m, HumanMessage):
                        last_human_text = extract_text(getattr(m, "content", "")).strip()
                        break

                if last_human_text and final_text:
                    recent_chat = f"human: {last_human_text}\nai: {final_text}"
                    mem_task = asyncio.create_task(deep_agent._background_extract_memory(recent_chat))
                    _background_tasks.add(mem_task)
                    mem_task.add_done_callback(_background_tasks.discard)
            except Exception as bg_err:
                print(f"⚠️ [记忆提取调度异常]: {bg_err}")

            return ChatReply(
                text=final_text,
                images=round_images,
                awaiting_approval=False,
            )

        except GraphRecursionError as e:
            print(f"[WARN] 任务步骤超限 (GraphRecursionError): {e}")
            return ChatReply(
                text="这个任务步骤太多，我先停下了，可以拆成几步再试。",
                images=[],
                awaiting_approval=False,
            )
        except Exception as e:
            traceback.print_exc()
            return ChatReply("处理时出了点问题，请稍后再试", [])
