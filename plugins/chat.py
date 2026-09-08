import traceback
import base64
from nonebot import on_message, get_driver
from nonebot.adapters.qq.event import DirectMessageCreateEvent, MessageEvent
from nonebot.adapters.qq.message import MessageSegment, Message
from nonebot.matcher import Matcher
from nonebot.exception import FinishedException
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command

# 从项目中导入核心逻辑，不修改其内部实现
from app import graph, database, mcp_service, browser_service
from app.models import vlm

driver = get_driver()
langgraph_app = None

def _make_image_segment(url: str) -> MessageSegment:
    """
    智能图片 Segment 工厂函数。
    - 如果 url 是 data URI（base64 内嵌图片），解码为 bytes 并使用 file_image()。
    - 否则直接使用 image() 发送普通 URL。
    """
    if url.startswith("data:"):
        # 格式: data:image/jpeg;base64,<b64data>
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
    """
    在 NoneBot 启动时初始化 LangGraph 环境，包括数据库 and MCP 服务，并构建计算图
    """
    global langgraph_app
    print("Initializing Database and MCP for NoneBot...")
    
    # 数据库全家桶一键初始化
    database.run_global_database_init()
    
    # 初始化 MCP 服务
    await mcp_service.initialize_mcp()
    # 构建计算图实例
    langgraph_app = graph.build_graph()
    print("LangGraph app built successfully for NoneBot.")


# 注册事件响应器，监听群消息和私聊消息
chat_matcher = on_message(priority=10, block=False)

@chat_matcher.handle()
async def handle_chat(matcher: Matcher, event: MessageEvent | DirectMessageCreateEvent):
    global langgraph_app
    print(f"\n[DEBUG] 收到消息事件 ID: {event.id}")
    
    if not langgraph_app:
        await matcher.finish("Agent 系统尚未初始化完成，请稍后再试。")
        
    user_id = event.get_user_id()
    config = {
        "recursion_limit": 25,
        "configurable": {"thread_id": user_id}
    }

    # ── 提取消息中的图片和文本 ─────────────────────────────────────────────────
    image_url = None
    text_parts = []
    message = event.get_message()
    
    seg_types = [seg.type for seg in message]
    print(f"[DEBUG] 消息分段类型: {seg_types}")

    for msg_seg in message:
        if msg_seg.type in ["image", "attachment"]:
            url = msg_seg.data.get("url")
            if url and not image_url:
                image_url = url
        elif msg_seg.type == "text":
            text_parts.append(msg_seg.data.get("text", ""))
            
    raw_text = "".join(text_parts).strip()
    print(f"[DEBUG] 提取文本: {raw_text[:50]}{'...' if len(raw_text)>50 else ''}, 是否包含图片: {bool(image_url)}")

    try:
        # ── 检查是否处于 HITL 中断等待状态 ─────────────────────────────────────
        current_state = langgraph_app.get_state(config)
        is_interrupted = bool(current_state.next)  # next 非空 = 图在等待恢复

        if is_interrupted:
            # 图已经暂停，等待用户对危险操作做出 Y/N 判断
            user_reply = raw_text.strip().upper()
            
            if user_reply == "Y":
                print(f"[HITL] 用户授权，恢复执行...")
                async for event in langgraph_app.astream(Command(resume="Y"), config=config):
                    for node_name, state_update in event.items():
                        if node_name == "Supervisor":
                            analysis = state_update.get("analysis", "思考中...")
                            print(f"\n🟢 \033[92m[Supervisor 反思]\033[0m: {analysis}")
                        elif node_name in ["MapAgent", "KnowledgeAgent", "MediaAgent", "FileAgent", "CodeAgent", "BrowserAgent"]:
                            print(f"\n🟡 \033[93m[{node_name} 执行完毕]\033[0m")
            elif user_reply == "N":
                print(f"[HITL] 用户拒绝，注入 DENIED 并恢复执行...")
                await matcher.send("🚫 已拦截该敏感操作，正在通知 AI 调整后续计划...")
                async for event in langgraph_app.astream(Command(resume="N"), config=config):
                    for node_name, state_update in event.items():
                        if node_name == "Supervisor":
                            analysis = state_update.get("analysis", "思考中...")
                            print(f"\n🟢 \033[92m[Supervisor 反思]\033[0m: {analysis}")
            else:
                # 用户回复了其他内容，提示他当前处于授权等待中
                # 从状态中提取待确认的工具信息，再次显示警告
                warning = _format_hitl_warning(current_state)
                await matcher.finish(
                    f"⚠️ 当前正在等待授权确认，请回复 Y 允许或 N 拒绝。\n\n{warning}"
                )
                return

            # 恢复后检查图是否再次中断
            final_state = langgraph_app.get_state(config)
            await _handle_result_and_possibly_interrupt(matcher, langgraph_app, final_state.values, config)
            return

        # ── 正常流程：处理新消息 ─────────────────────────────────────────────────

        # 如果消息包含图片，调用 VLM 进行意图翻译
        if image_url:
            print(f"[DEBUG] 启动 VLM 意图翻译层 (处理单张图片)...")
            sys_prompt = (
                "你现在是一个意图翻译官。用户的系统中有一套纯文本的工具链（包括菜谱检索、高德地图、资料库等）。"
                "请观察用户上传的图片，结合用户的文字描述，识别出图片中的核心实体（如具体食材、地名、人物等），并将其转化为一句精简的纯文本请求。\n"
                "【示例】用户发了西红柿和鸡蛋的图片+文字'我能吃啥' -> 你的输出：'查询使用西红柿和鸡蛋制作的菜谱'。\n"
                "请只输出翻译后的纯文本指令，不要包含任何多余的解释。"
            )
            content = []
            if raw_text:
                content.append({"type": "text", "text": raw_text})
            else:
                content.append({"type": "text", "text": "请分析这张图片，提取其中的核心实体或需求"})
            content.append({"type": "image_url", "image_url": {"url": image_url}})
                
            try:
                vlm_res = await vlm.ainvoke([
                    SystemMessage(content=sys_prompt), 
                    HumanMessage(content=content)
                ])
                raw_text = vlm_res.content.strip()
                print(f"[VLM 意图翻译] 转换后指令: {raw_text}")
            except Exception as e:
                print(f"[ERROR] VLM 调用失败:")
                traceback.print_exc()
                await matcher.finish(f"多模态意图翻译失败: {str(e)}")

        # 过滤空消息
        if not raw_text:
            return
            
        inputs = {"messages": [HumanMessage(content=raw_text, name="user")]}
        
        # 使用 astream 替换 ainvoke 以便在终端实时打印 Supervisor 的思考和 Worker 的战报
        async for event in langgraph_app.astream(inputs, config=config):
            for node_name, state_update in event.items():
                if node_name == "Supervisor":
                    # 打印 Supervisor 反思
                    analysis = state_update.get("analysis", "思考中...")
                    print(f"\n🟢 \033[92m[Supervisor 反思]\033[0m: {analysis}")
                    
                    next_agent = state_update.get("next_agent")
                    messages = state_update.get("messages", [])
                    
                    if next_agent == "FINISH":
                        if messages and hasattr(messages[-1], "name") and messages[-1].name == "Supervisor_Final":
                            print(f"\n🌟 \033[96m[最终回答]\033[0m: {messages[-1].content}\n")
                    elif next_agent:
                        # 尝试从最新消息中提取分发的指令
                        instruction = ""
                        if messages and hasattr(messages[-1], "name") and messages[-1].name == "Supervisor":
                            content = messages[-1].content
                            # 兼容新旧格式
                            instruction = content.replace("【决策】指派 ", "").replace(f"{next_agent} 执行任务：", "").replace(f"指派 {next_agent} 执行任务：", "")
                        print(f"\n🔵 \033[94m[派发任务给 {next_agent}]\033[0m: {instruction}")
                        
                elif node_name in ["MapAgent", "KnowledgeAgent", "MediaAgent", "FileAgent", "CodeAgent", "BrowserAgent"]:
                    messages = state_update.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        content = last_msg.content
                        
                        if isinstance(content, list):
                            has_image = any(isinstance(c, dict) and c.get("type") == "image_url" for c in content)
                            text_parts = []
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    t = c.get("text", "")
                                    text_parts.append(str(t) if not isinstance(t, list) else str(t[0]))
                            display_text = " ".join(text_parts)
                            if len(display_text) > 800: display_text = display_text[:800] + "..."
                            label = f"{node_name} 战报 (图文多模态)" if has_image else f"{node_name} 战报"
                            print(f"\n🟡 \033[93m[{label}]\033[0m: \n{display_text}")
                        else:
                            display_text = str(content)
                            if len(display_text) > 800: display_text = display_text[:800] + "..."
                            print(f"\n🟡 \033[93m[{node_name} 战报]\033[0m: \n{display_text}")

        # 获取最终完整的 state 作为 result 传给结果处理器
        final_state = langgraph_app.get_state(config)
        await _handle_result_and_possibly_interrupt(matcher, langgraph_app, final_state.values, config)

    except FinishedException:
        raise
    except Exception as e:
        traceback.print_exc()
        await matcher.finish(f"Agent 大脑过载了: {str(e)}")


def _format_hitl_warning(state) -> str:
    """从图状态中提取即将执行的危险工具信息，格式化为用户友好的 QQ 警告消息。"""
    try:
        tasks = getattr(state, "tasks", [])
        if not tasks:
            return "⚠️ FileAgent 请求执行一项敏感操作，请回复 Y 允许或 N 拒绝。"
        
        # 寻找最近的一个中断数据
        item = None
        for task in reversed(tasks):
            if task.interrupts:
                interrupt_item = task.interrupts[-1]
                # 如果中断信息本身就是字符串（例如 raise GraphInterrupt("msg") 抛出的信息）
                if isinstance(interrupt_item, str):
                    return interrupt_item
                
                # 兼容不同版本的 LangGraph 中断类型（含有 value 属性）
                val = interrupt_item.value if hasattr(interrupt_item, "value") else interrupt_item
                
                if isinstance(val, str):
                    return val
                
                # 如果是列表取第一个，如果是字典直接用
                item = val[0] if isinstance(val, list) and val else val
                break

        if not item or not isinstance(item, dict):
            return "⚠️ FileAgent 请求执行一项敏感操作，请回复 Y 允许或 N 拒绝。"

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
    return "⚠️ FileAgent 请求执行一项敏感操作，请回复 Y 允许或 N 拒绝。"


async def _handle_result_and_possibly_interrupt(matcher, app, result, config):
    """
    处理 ainvoke 的返回结果：
    - 如果图再次中断（危险工具待确认），发送 HITL 警告并等待用户回复。
    - 否则提取最后一条消息作为正常回复，并附带 Worker 战报中收集到的图片。
    """
    # 检查图是否在返回后仍处于中断状态
    after_state = app.get_state(config)
    if after_state.next:
        # 图暂停了，发送授权请求给用户
        warning = _format_hitl_warning(after_state)
        await matcher.finish(warning)
        return

    # 正常结束，提取最后一条消息
    messages = result.get("messages", [])
    if not messages:
        await matcher.finish("Agent 没有返回任何消息。")
        return

    # ── 收集 Worker 战报中的图片 ──────────────────────────────────────────────
    # 从所有消息中逆序扫描，找到最近一轮 Worker HumanMessage 里包含的 image_url
    collected_images = []
    for msg in reversed(messages):
        # 如果遇到了当前轮次用户发出的消息，说明当前轮次没有产生图片，立刻停止回溯，防止跨轮次泄漏
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "user":
            break

        # Supervisor 最终回答是 AIMessage，Worker 战报是 HumanMessage
        # 找到 Supervisor 最终回答的 AIMessage 后才开始搜集图片（从它往前）
        if isinstance(msg, AIMessage) and not collected_images:
            # 还未找到图片，继续往前看 Worker 消息
            continue
        if isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    collected_images.append(item)
        # Worker 战报是单独的 HumanMessage，找完一层即止（不回溯到上上轮）
        if collected_images:
            break

    # ── 构建最终回复 ──────────────────────────────────────────────────────────
    last_msg = messages[-1]
    content = last_msg.content

    # 提取纯文本回答
    if isinstance(content, str):
        text_answer = content
    elif isinstance(content, list):
        text_answer = " ".join(
            str(item.get("text", "")) for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        text_answer = str(content)

    if len(text_answer) > 600:
        text_answer = text_answer[:600] + "\n... (内容过长已截断)"

    # 如果有图片，组合文本 + 图片一起发送
    if collected_images:
        reply = Message()
        if text_answer:
            reply += MessageSegment.text(text_answer)
        for img_item in collected_images:
            url = img_item.get("image_url", {}).get("url", "")
            if url:
                reply += _make_image_segment(url)
        await matcher.finish(reply)
    else:
        # 纯文本回复（兼容原先的 list content 格式，如 RAG 返回的多模态）
        if isinstance(content, list):
            reply = Message()
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_part = item.get("text", "")
                        if len(text_part) > 600:
                            text_part = text_part[:600] + "\n... (部分文本过长已截断)"
                        reply += MessageSegment.text(text_part)
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            reply += _make_image_segment(url)
                elif isinstance(item, str):
                    if len(item) > 600:
                        item = item[:600] + "\n... (内容过长已截断)"
                    reply += MessageSegment.text(item)
            await matcher.finish(reply)
        else:
            await matcher.finish(text_answer)

