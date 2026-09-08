from typing import Annotated, List, Literal, Optional, Any, Dict
from typing_extensions import TypedDict
import operator

# Reducer：用于在 LangGraph 状态更新时累加 Agent 调用次数（而非覆盖）
def add_counts(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    """将两个调用计数字典合并，对相同 key 的值进行累加。"""
    res = left.copy()
    for k, v in right.items():
        res[k] = res.get(k, 0) + v
    return res

import asyncio
import copy
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt

from . import models
from . import tools
from .tools import code_tools
from . import mcp_service
from . import database
import config
from langgraph.checkpoint.memory import MemorySaver

# 1. 定义状态 (Agent State)
class AgentState(TypedDict):
    # 使用 operator.add 保证消息队列能持续累加，实现状态保留与防火墙
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    analysis: str  # 记录主管的反思过程，供日志展示
    long_term_memories: str
    instruction_to_worker: str
    current_tool_call_id: str
    user_query: str
    # Agent 调用计数器：记录每个 Agent 本轮被调用次数，用于死循环检测
    agent_call_counts: Annotated[Dict[str, int], add_counts]

from .models import Router

# 3. 创建 Supervisor 节点
SUPERVISOR_PROMPT = """你是一个多智能体系统的主管（Supervisor）。
你负责分析用户的问题，并且你现在拥有一个名为 `Router` 的工具。请分析上下文，并**调用该工具**来指派下属专家或直接给出最终回答。

【决策逻辑】
1. **优先检索**：如果用户的问题涉及知识查询（如做法、地址、特定事实），必须先派发给相关专家。
2. **直接回答**：只有当目前的对话历史或长期记忆中已经包含了完美回答用户所需的全部信息时，才选择 FINISH 并给出 final_answer。
   【禁令 - 严禁偷懒】：用户**完全看不见**专家汇报的任何中间过程。你必须像一个“信息搬运工”，将专家为你找到的**每一项细节数据**（如完整文件路径列表、检索到的知识原文、代码运行结果）全部亲手拷贝到 final_answer 中。
   【禁令 - 严禁脑补】：**绝对禁止凭空编造任何数据！** 如果你在下属专家的汇报中没有看到明确的原始内容（例如：FileAgent 没给文件正文、CodeAgent 没给执行结果、KnowledgeAgent 没给检索原文），你必须如实回答“未能获取到相关内容”或继续指派专家深挖。**严禁将专家的“操作计划”或“步骤描述”误认为执行结果！**
   【身份识别】：历史记录中任何带有 **“【内部专家汇报】”** 头部的消息都是你的下属专家提供的，**绝非用户在说话**。你必须提取其中的数据并复述给用户。
   【孤岛效应】：不要假设用户看过之前的战报。每一份 final_answer 必须是**自包含且完整**的报告。严禁概括性回答“已为您列出”，必须在 final_answer 里把列表再写一遍！如果你只给结论不给数据，将被视为任务失败。
3. **去代词化**：在给专家的指令中，必须将“他”、“那里”、“那个”等代词替换为具体名词。

【长期记忆注入】
以下是关于该用户的长期背景信息（如有）：
{long_term_memories}

【信息防火墙与多模态兼容说明】
- 下属专家（Worker）的汇报将会以普通消息（ToolMessage）的形式返回给你。
- 专家有时会返回包含多模态（如图片）的数据。你（Supervisor）作为多模态大模型，可以直接“看”到这些图片，请综合图文信息来思考。

【下属专家清单与任务分工 - 必须精准分流】
- KnowledgeAgent：【知识百科专家】负责处理菜谱、游戏攻略、历史、科学等纯知识性咨询。
  * 检索原则：优先使用 rag 检索本地知识，无果时才用 web_search。
  * 禁区：严禁处理任何涉及“在哪买”、“怎么去”、“天气如何”等地理位置相关的请求。
  * 禁区：严禁处理任何涉及项目代码、本地文件读写、目录浏览的请求，此类任务属于 FileAgent 的职责。
- MapAgent：【地理出行专家】挂载高德地图工具，负责处理：
  * 地理编码（经纬度查询）。
  * 路线规划（步行、骑行、驾车导航）。
  * 周边搜索（POI 搜索，如“附近的超市”、“济南的农贸市场”）。
  * 天气查询。
- MediaAgent：【成人影视专家】（高优先级）负责 AV、女优、番号等综合性或模糊搜索/关系网络咨询（基于知识图谱）。
- SQLAgent：【关系数据库专家】专门用于精确过滤、统计或多条件检索本地 MySQL 数据库中女优 (actresses)、作品 (works) 及其关联表关系。
  * 任务范围：所有需要根据精确条件（如“身高大于165且罩杯为G的女优”、“某个女优在2020年后发行过的作品”、“统计某演员的作品数量”等）进行查询或跨表统计计算的问题。如果问题不涉及精确条件, 直接用MediaAgent; 查到精确信息后如果答案还不完善, 可以再交给MediaAgent
- BrowserAgent：【浏览器操作专家】负责处理所有涉及网页点击、动态数据抓取、文件上传与自动化发帖（如B站/小红书等社交平台发布动态）的操作。对于复杂规划（如携程查完再高德导航），必须先指派 BrowserAgent 拿地址，再指派 MapAgent。
  * 【信任原则 - 极端重要】：BrowserAgent 挂载的浏览器自动化工具完全有权限且能够直接处理本地文件上传（它只需接收路径字符串）。你（Supervisor）绝对不需要指派 FileAgent 去读取图片，直接将包含图片路径（如 'workspace/1.png'）的发布任务打包指派给 BrowserAgent 即可。
  * 【禁止摆烂】：不要盲目猜测系统没有权限或无法自动化。只要涉及网页端发帖、发布、登录等交互，果断指派给 BrowserAgent 让它调用工具去执行，严禁提前选择 FINISH 让用户手动操作！
- FileAgent：【本地文件专家】挂载 Filesystem MCP 工具，是唯一能访问本地磁盘的 Agent。负责处理：
  * 读取、写入、编辑项目内的文件（如查看某个 .py/.json/.md 文件等）。
  * 极端重要：严禁指派 FileAgent 去“读取（read_file）”任何图像或音视频文件，它无法解析二进制数据，这会导致系统死循环！
- DesktopAgent：【桌面操作专家】负责通过网格坐标系统截屏、模拟点击、安全输入和快捷键来操控 Windows 桌面，执行本地 GUI 软件交互。
- CodeAgent：【代码执行专家】负责使用 Python 解决数学计算、数据处理和绘图（如保存图表）任务。只能编写代码，会自动在沙箱内运行。
  * 注意：你必须明确指示 CodeAgent 绘制图片，它无法帮你查资料。图片会自动通过工具回传，你不必自己处理图片。

【任务指派规范】
1. 如果需要组合能力（如查到某个地点后导航），请分步指派：先派某专家获取信息，得到结果后再派导航专家。
2. instruction_to_worker 必须包含足够明确的上下文！例如指派 MapAgent 时，指令必
须是“用户当前位置在济南市历下区文化东路42号，请搜索附近的菜市场”，绝不能仅仅是“帮用户查去哪买菜”。
3. 防死循环与强制终止策略：当收集到的信息已足够完美回答用户，或者发现 Worker 反复查找不到新信息时，必须立刻选择 FINISH。严禁为了过度确认 engagement 而反复调用 Agent！
4. 【图文剥离原则】：如果下属专家传回了图片（你能在上下文中看到），你【绝对不要】在 final_answer 中尝试生成图片链接、Base64或Markdown图片语法！底层系统会自动将图片发送给QQ用户。你的 final_answer 只需要输出基于图片分析出的纯文本结论即可！

【强制回复原则】
- 所有的最终回复必须保持精简、干脆，严禁啰嗦。
- 最终回答的内容字数必须控制在 500 字以内，以适应 QQ 机器人的显示限制。
"""

MEMORY_EXTRACTOR_PROMPT = """你是一个极其冷酷、挑剔且吝啬的“个人隐私与偏好提取官”。
你的目标是：只有当用户透露了【足以跨越数月甚至数年都有参考价值】的个人核心信息时，才进行记录。
绝大多数对话都应该是 NONE。

【严禁记录 - 只要包含以下特征，立即回复 NONE】
1. 任何关于“明天”、“今天”、“周几”、“几点”的行程、天气或临时打算。
2. 任何从工具、知识库、地图中查出的客观数据（如英雄技能、经纬度、路线指引、百科知识）。
3. 任何临时的询问意图（如“我想吃...”、“帮我查...”）。
4. 任何不涉及用户“本人”特征的客观事实。

【仅允许记录 - 必须符合以下条件之一】
1. 永久性的个人基本信息：家/公司的精确地址（需具体到门牌号或小区名）、真实姓名、生日、手机号。
   ⚠️ 特别注意：用户哪怕只是陈述性地说"我家在XX"或"我住在XX"，也必须立即提取并记录，无需等待用户提出具体需求。
2. 核心且长期的身体/习惯特征：如“对海鲜严重过敏”、“平时只喝冰美式”、“不吃任何辣的东西”。
3. 强烈的、相对稳定的个人关系或身份：如“我老婆叫小红”、“我是一名程序员”。

【输出要求】
- 符合条件：总结成一句话（如“用户家住在济南市文化东路42号”）。
- 不符合条件：**必须回复且只能回复** NONE。禁止解释原因，禁止加标点。

对话内容：
{recent_chat}
"""

def _sanitize_messages_for_llm(messages: list) -> list:
    """
    统一脱敏函数：将消息列表中的多模态内容转换为所有 LLM 兼容的稳健格式。
    """
    sanitized = []
    for msg in messages:
        # 提取核心内容
        content = getattr(msg, "content", "")
        msg_name = getattr(msg, "name", None)
        
        # 处理内容格式
        if isinstance(content, list):
            # ... (保持原有的多模态处理逻辑) ...
            clean_parts = []
            for item in content:
                if not isinstance(item, dict): continue
                if item.get("type") == "text":
                    text_val = item.get("text", "")
                    clean_parts.append({"type": "text", "text": str(text_val)})
                elif item.get("type") == "image_url":
                    clean_parts.append(item)
            content = clean_parts if clean_parts else ""

        # 重构消息对象
        if isinstance(msg, HumanMessage):
            sanitized.append(HumanMessage(content=content, name=msg_name))
        elif isinstance(msg, AIMessage):
            sanitized.append(AIMessage(content=content, name=msg_name, tool_calls=getattr(msg, "tool_calls", [])))
        elif isinstance(msg, ToolMessage):
            sanitized.append(ToolMessage(content=str(content), tool_call_id=getattr(msg, "tool_call_id", ""), name=msg_name))
        elif isinstance(msg, SystemMessage):
            sanitized.append(SystemMessage(content=str(content)))
            
    return sanitized

async def supervisor_node(state: AgentState) -> dict:
    messages = state["messages"]

    # 统一净化消息格式
    sanitized_messages = _sanitize_messages_for_llm(messages)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    # 绑定 Router 工具实现 Native Tool Calling
    bound_llm = models.supervisor_llm.bind_tools([Router], tool_choice="Router")
    
    try:
        response = await (prompt | bound_llm).ainvoke({
            "messages": sanitized_messages,
            "long_term_memories": state.get("long_term_memories", "无")
        })
        
        if not response.tool_calls:
            raise ValueError("LLM 没有调用工具，可能是被拦截或格式错误。")
            
        tool_call = response.tool_calls[0]
        result_dict = tool_call["args"]
        tool_call_id = tool_call["id"]
            
    except Exception as e:
        print(f"⚠️ [Supervisor] 彻底解析失败或 LLM 输出异常: {e}")
        return {
            "next_agent": "FINISH",
            "analysis": "由于模型输出异常或被拦截，强制结束任务。",
            "messages": [AIMessage(content="抱歉，刚才的处理过程中触发了模型的内容审核或发生了系统异常，无法为您提供该话题的详细解答。", name="Supervisor_Final")]
        }
    
    updates = {
        "next_agent": result_dict.get("next_agent", "FINISH"),
        "analysis": result_dict.get("analysis", ""),
        "instruction_to_worker": result_dict.get("instruction_to_worker", ""),
        "current_tool_call_id": tool_call_id  # 动态覆盖更新 id
    }
    
    new_messages = []

    # ── 死循环拦截：若同一 Agent 已被调用 >= 2 次则强制终止 ──────────────────
    target_agent = updates["next_agent"]
    current_counts = state.get("agent_call_counts", {})

    if target_agent != "FINISH":
        if current_counts.get(target_agent, 0) >= 2:
            print(f"\n🚫 \033[91m[强制干预]\033[0m: 检测到 {target_agent} 陷入死循环（已调用 {current_counts[target_agent]} 次），强制终止任务。")
            updates["next_agent"] = "FINISH"
            updates["analysis"] = f"强制干预：检测到 {target_agent} 无法获取有效信息并陷入死循环，任务被系统强制结束。"
            error_msg = f"抱歉，在尝试多次后，底层系统（{target_agent}）仍未能成功获取到有效数据，任务已自动终止。"
            new_messages.append(response)
            new_messages.append(AIMessage(content=error_msg, name="Supervisor_Final"))
            updates["messages"] = new_messages
            return updates
        else:
            # 记录本次调用 (+1)，通过 Reducer 累加到全局状态
            updates["agent_call_counts"] = {target_agent: 1}
    # ────────────────────────────────────────────────────────────────────────

    if updates["next_agent"] == "FINISH":
        final_msg = result_dict.get("final_answer") or "任务完成。"
        # 原封不动保存原生工具调用 AIMessage
        new_messages.append(response)
        new_messages.append(AIMessage(content=final_msg, name="Supervisor_Final"))
    else:
        # 指派任务时同样需要将原始 response 入列
        new_messages.append(response)
        
    updates["messages"] = new_messages
    return updates

async def retrieve_memory_node(state: AgentState) -> dict:
    messages = state["messages"]
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.type == "human" and isinstance(msg.content, str):
            last_user_msg = msg.content
            break
            
    updates = {"user_query": last_user_msg}
    
    if last_user_msg:
        print(f"\n🔍 \033[95m[长期记忆检索]\033[0m: '{last_user_msg}'")
        memories = database.search_memory(last_user_msg, top_k=1)
        if memories:
            long_term_memories = "\n".join([f"- {m}" for m in memories])
            print(f"✅ \033[92m[唤醒记忆]\033[0m: 找到 {len(memories)} 条相关内容:")
            for i, m in enumerate(memories):
                print(f"   \033[90m{i+1}. {m}\033[0m")
            updates["long_term_memories"] = long_term_memories
            return updates
        else:
            print("⚪ \033[90m[检索结果]: 无相关长期记忆\033[0m")
            
    updates["long_term_memories"] = "无"
    return updates

async def _background_extract_memory(recent_chat: str):
    """后台异步执行记忆提取与保存，不阻塞主流程"""
    try:
        # 结合 Prompt 模板
        final_prompt = MEMORY_EXTRACTOR_PROMPT.format(recent_chat=recent_chat)
        
        # 调用已经解耦的 models.extractor_llm
        res = await models.extractor_llm.ainvoke(final_prompt)
        content = res.content.strip()
        
        # 严格 NONE 判断机制
        if content and content.upper() != "NONE" and "NONE" not in content.upper():
            database.insert_memory(content)
            print(f"--- \033[92m[后台记忆保存]\033[0m: {content} ---")
        else:
            print("--- \033[90m[后台记忆跳过]: 未发现长期价值信息\033[0m ---")
    except Exception as e:
        print(f"⚠️ \033[91m[后台记忆提取异常]\033[0m: {str(e)}")

async def extract_memory_node(state: AgentState) -> dict:
    print("\n🧠 \033[95m[长期记忆提取中 (后台任务已启动)...]\033[0m")
    
    messages = state["messages"]
    if len(messages) >= 2:
        chat_history = []
        # 只看最近 2-3 条对话，避免过度总结
        for msg in messages[-3:]:
            content = msg.content
            if isinstance(content, list):
                text_part = " ".join([
                    str(item.get("text", "")) if not isinstance(item.get("text", ""), list)
                    else " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ])
                chat_history.append(f"{msg.type}: {text_part}")
            elif isinstance(content, str):
                chat_history.append(f"{msg.type}: {content}")
        
        recent_chat = "\n".join(chat_history)
        
        # 触发后台任务，立即返回，不阻塞主 Graph
        asyncio.create_task(_background_extract_memory(recent_chat))
            
    return {}

# 4. 构建 Worker 与拦截器 (Node Wrapper)
def make_worker_node(worker_name: str, worker_tools: list, system_prompt: str, llm=None):
    """
    Worker 节点工厂函数。
    【核心修正】：使用原生的 create_react_agent。
    通过 prompt 参数传入一个 Callable，在模型调用前拦截并处理 ToolMessage 中的图片，
    将图片剥离到 HumanMessage 中以绕过 OpenAI 400 错误。
    """
    if llm is None:
        llm = models.worker_llm
    
    def multimodal_prompt_handler(state: Any):
        """
        动态构建发送给 LLM 的消息序列。
        如果历史记录中有 ToolMessage 包含了 List 格式的多模态内容（图片），
        则将其拆分为：1个纯文本 ToolMessage + 1个包含图片的 HumanMessage。
        """
        # create_react_agent 传入的 state 通常有 messages 键
        msgs = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        
        sanitized = [SystemMessage(content=system_prompt)]
        
        for msg in msgs:
            # 只有 ToolMessage 包含图片列表时需要特殊处理
            if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
                text_parts = []
                image_parts = []
                for item in msg.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            image_parts.append(item)
                    else:
                        text_parts.append(str(item))
                
                # 1. 添加合法的纯文本 ToolMessage
                sanitized.append(ToolMessage(
                    content="\n".join(text_parts),
                    tool_call_id=msg.tool_call_id
                ))
                # 2. 紧跟一条 HumanMessage 承载图片，使 LLM 能看见
                if image_parts:
                    sanitized.append(HumanMessage(content=[
                        {"type": "text", "text": "[系统消息：以下是工具获取的相关图片证据]"}
                    ] + image_parts))
            else:
                sanitized.append(msg)
        return sanitized

    # 使用原生引擎构建 worker agent
    agent_executor = create_react_agent(
        llm,
        tools=worker_tools,
        prompt=multimodal_prompt_handler
    )
    
    async def worker_node_wrapper(state: AgentState) -> dict:
        """
        Worker 节点的统一包装器：
        1. 状态隔离：不传递全部全局消息，只传递 SystemPrompt + Task Instruction + 用户初始问题
        2. 执行 Agent 决策循环。
        3. 收集图片并使用原生的 ToolMessage 返回。
        """
        # 🔥 直接从 state 获取，稳如老狗
        user_query = state.get("user_query", "")
        task_instruction = state.get("instruction_to_worker", "")
        tool_call_id = state.get("current_tool_call_id", "")
        
        # 隔离状态：重新组装纯净的输入历史
        isolated_messages = [
            HumanMessage(content=f"【用户原始请求背景】：{user_query}\n\n【当前核心任务】：{task_instruction}")
        ]
        
        # 调用原生 Agent
        result = await agent_executor.ainvoke({"messages": isolated_messages})
        
        # 提取最后一条 AI 回复，并确保转换为纯文本字符串
        final_ai_message = result["messages"][-1]
        worker_text = final_ai_message.content
        if isinstance(worker_text, list):
            worker_text = "\n".join(str(item.get("text", "")) for item in worker_text if isinstance(item, dict) and item.get("type") == "text")
        
        # 💡【优化】：因为 isolated_messages 长度固定为 1，[1:] 是最安全且纯粹的本轮变动增量
        new_messages = result["messages"][1:]
        collected_images = []
        for msg in new_messages:
            # 过滤1：排除 RAG 工具提取的辅助图片（仅供 VLM 看，不给用户看）
            if isinstance(msg, ToolMessage) and getattr(msg, "name", "") in ["rag", "av_graph_rag"]:
                continue
            
            # 过滤2：排除 multimodal_prompt_handler 注入的辅助 HumanMessage
            if isinstance(msg, HumanMessage):
                text_content = ""
                if isinstance(msg.content, str):
                    text_content = msg.content
                elif isinstance(msg.content, list):
                    text_content = "".join(str(i.get("text", "")) for i in msg.content if isinstance(i, dict) and i.get("type") == "text")
                if "[系统消息：以下是工具获取的相关图片证据]" in text_content:
                    continue

            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        collected_images.append(item)
        
        # 战报内容：处理 LLM 可能因安全审核（Block）而返回空响应的情况
        if not worker_text and not collected_images:
            worker_text = "抱歉，由于内容安全限制，该专家节点无法回答此问题或讨论此话题。"
            
        report_header = f"【内部专家汇报 - 来源：{worker_name}】\n"
        
        if collected_images:
            report_content = [{"type": "text", "text": report_header + (worker_text or "[图文信息]")}] + collected_images
        else:
            report_content = report_header + (worker_text or "")
            
        # 返回标准的 ToolMessage
        return {"messages": [ToolMessage(content=report_content, tool_call_id=tool_call_id, name=worker_name)]}
    
    return worker_node_wrapper



# ══════════════════════════════════════════════════════════════════════════════
# FileAgent Subgraph —— 带 HITL 的文件系统 Agent
# ══════════════════════════════════════════════════════════════════════════════

class FileAgentState(TypedDict):
    """FileAgent 子图专用状态；messages 字段与主图共享 operator.add 语义。"""
    messages: Annotated[List[BaseMessage], operator.add]
    current_tool_call_id: str
    instruction_to_worker: str
    user_query: str


def build_file_agent_subgraph(filesystem_tools: list, system_prompt: str, llm=None):
    """
    构建带 Human-in-the-loop 的 FileAgent 子图。

    子图节点：
      fa_llm            — LLM 决策节点，产出 AIMessage（含 tool_calls）
      fa_safe_tools     — 执行安全工具（read/list/search 等），直接执行不拦截
      fa_dangerous_tools— 执行危险工具前调用 interrupt() 等待用户授权

    子图不携带 Checkpointer，由父图（主图）的 MemorySaver 统一管理状态持久化。
    interrupt() 触发时，父图的 ainvoke() 会自动返回，主图状态被保存。
    用户回复后通过 Command(resume=value) 恢复执行。
    """
    if llm is None:
        llm = models.worker_llm

    # 工具字典，供节点按名查找
    tool_map = {t.name: t for t in filesystem_tools}
    dangerous_names = mcp_service.DANGEROUS_FS_TOOLS

    # ── 节点函数 ─────────────────────────────────────────────────────────────

    async def fa_llm_node(state: FileAgentState) -> dict:
        """调用 LLM，决定下一步工具调用或直接生成回答。"""
        msgs = state["messages"]
        sys_msg = SystemMessage(content=system_prompt)
        
        target_tool_call_id = state.get("current_tool_call_id")
        
        task_instruction = state.get("instruction_to_worker", "")
        user_query = state.get("user_query", "")
        
        # 寻找子图内部产生的交互
        split_idx = 0
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], AIMessage) and msgs[i].tool_calls:
                if any(tc["id"] == target_tool_call_id for tc in msgs[i].tool_calls):
                    split_idx = i
                    break
                
        isolated_start_msg = HumanMessage(content=f"【用户原始请求背景】：{user_query}\n\n【当前核心任务】：{task_instruction}")
        
        # 取该次调用之后的所有子图内部产生的交互
        internal_history = msgs[split_idx+1:]
        sanitized_msgs = _sanitize_messages_for_llm(internal_history)
        
        # 绑定工具 schema 给 LLM
        bound_llm = llm.bind_tools(filesystem_tools)
        response: AIMessage = await bound_llm.ainvoke([sys_msg, isolated_start_msg] + sanitized_msgs)
        return {"messages": [response]}

    async def fa_safe_tools_node(state: FileAgentState) -> dict:
        """执行所有安全工具调用，不需要用户确认。"""
        last_ai: AIMessage = state["messages"][-1]
        tool_messages = []
        for tc in last_ai.tool_calls:
            if tc["name"] in dangerous_names:
                continue  # 危险工具不在这里执行
            print(f"  🤖 [FileAgent] 正在调用安全工具: \033[94m{tc['name']}\033[0m, 参数: {tc['args']}")
            tool = tool_map.get(tc["name"])
            if tool is None:
                result = f"工具 '{tc['name']}' 未找到。"
            else:
                try:
                    result = await tool.ainvoke(tc["args"])
                    result = str(result)
                except Exception as e:
                    result = f"工具执行出错: {e}"
            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
        return {"messages": tool_messages}

    async def fa_dangerous_tools_node(state: FileAgentState) -> dict:
        """
        执行危险工具节点（回归单步确认模式）。
        """
        last_ai: AIMessage = state["messages"][-1]
        tool_messages = []

        for tc in last_ai.tool_calls:
            if tc["name"] not in dangerous_names:
                continue

            print(f"\n🔴 \033[91m[FileAgent HITL]\033[0m: 危险工具 '{tc['name']}' 请求授权...")

            # 每次只处理一个危险工具，遇到就暂停
            decision: str = interrupt({
                "tool_name": tc["name"],
                "tool_args": tc["args"],
                "tool_call_id": tc["id"]
            })

            if isinstance(decision, str) and decision.strip().upper() == "Y":
                tool = tool_map.get(tc["name"])
                try:
                    result = await tool.ainvoke(tc["args"])
                    result = str(result)
                    print(f"  ✅ [FileAgent HITL] 执行成功")
                except Exception as e:
                    result = f"工具执行出错: {e}"
            else:
                # 注入一个极强语义的错误，防止模型（尤其是 Gemini）产生“再试一次”的幻觉
                result = (
                    "CRITICAL ERROR: USER PERMISSION DENIED. "
                    f"The user has explicitly forbidden the execution of '{tc['name']}'. "
                    "DO NOT attempt to retry this tool call or any similar calls in this turn. "
                    "If this operation is essential for the remaining task, you MUST stop and report "
                    "to the supervisor that the task was cancelled by the user."
                )
                print(f"  ⛔ [FileAgent HITL] 用户已拒绝工具 '{tc['name']}'")

            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tc["id"],
                name=tc["name"],
            ))

        return {"messages": tool_messages}

    async def fa_summarize_node(state: FileAgentState) -> dict:
        """子图收尾节点：将所有工具执行结果汇总为一条战报格式的消息回传主图。"""
        msgs = state["messages"]
        # 提取当前子图运行产生的最后一条 AI 总结消息
        worker_text = ""
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage) and msg.content:
                worker_text = msg.content
                break
        
        # 如果是列表且只包含文本，强转为字符串以净化上下文
        if isinstance(worker_text, list):
            has_image = any(isinstance(c, dict) and c.get("type") == "image_url" for c in worker_text)
            if not has_image:
                worker_text = "\n".join(c.get("text", "") for c in worker_text if isinstance(c, dict) and c.get("type") == "text")
        
        # 如果最后一条消息只是计划（Plan）而没产生实际结果，强制增加警告提示
        if "【执行】" in worker_text or "请开始执行" in worker_text:
            if not any(isinstance(m, ToolMessage) for m in state["messages"][-5:]):
                worker_text += "\n⚠️ [警告] 专家未能通过工具获取到实际数据，以上仅为计划描述。"
            
        # 【稳健重构】：构造显式的“内部汇报”头部
        report_header = f"【内部专家汇报 - 来源：FileAgent】\n"
        report_content = report_header + worker_text
                
        # 返回 ToolMessage 与 Supervisor 产生闭环
        tool_call_id = state.get("current_tool_call_id")
        return {"messages": [ToolMessage(content=report_content, tool_call_id=tool_call_id, name="FileAgent")]}

    # ── 路由函数 ──────────────────────────────────────────────────────────────

    def _route_after_llm(state: FileAgentState) -> str:
        """根据 LLM 输出的 tool_calls 决定下一步节点。"""
        last_ai: AIMessage = state["messages"][-1]
        if not getattr(last_ai, "tool_calls", None):
            return "fa_summarize"  # 无工具调用，走总结节点收尾
        # 有危险工具调用则先走危险节点，有安全工具则走安全节点
        for tc in last_ai.tool_calls:
            if tc["name"] in dangerous_names:
                return "fa_dangerous_tools"
        return "fa_safe_tools"

    def _route_after_tools(state: FileAgentState) -> str:
        """工具执行完毕后，总是回到 LLM 决策节点继续。"""
        return "fa_llm"

    # ── 子图组装 ──────────────────────────────────────────────────────────────

    subgraph = StateGraph(FileAgentState)
    subgraph.add_node("fa_llm", fa_llm_node)
    subgraph.add_node("fa_safe_tools", fa_safe_tools_node)
    subgraph.add_node("fa_dangerous_tools", fa_dangerous_tools_node)
    subgraph.add_node("fa_summarize", fa_summarize_node)

    subgraph.add_edge(START, "fa_llm")
    subgraph.add_conditional_edges("fa_llm", _route_after_llm)
    subgraph.add_conditional_edges("fa_safe_tools", _route_after_tools)
    subgraph.add_conditional_edges("fa_dangerous_tools", _route_after_tools)
    subgraph.add_edge("fa_summarize", END)

    # 编译时不指定 checkpointer，由父图（主图）的 MemorySaver 统一负责状态持久化
    return subgraph.compile()

    def _route_after_llm(state: FileAgentState) -> str:
        """根据 LLM 输出的 tool_calls 决定下一步节点。"""
        last_ai: AIMessage = state["messages"][-1]
        if not getattr(last_ai, "tool_calls", None):
            return "fa_summarize"  # 无工具调用，走总结节点收尾
        # 有危险工具调用则先走危险节点，有安全工具则走安全节点
        for tc in last_ai.tool_calls:
            if tc["name"] in dangerous_names:
                return "fa_dangerous_tools"
        return "fa_safe_tools"

    def _route_after_tools(state: FileAgentState) -> str:
        """工具执行完毕后，总是回到 LLM 决策节点继续。"""
        return "fa_llm"

    # ── 子图组装 ──────────────────────────────────────────────────────────────

    subgraph = StateGraph(FileAgentState)
    subgraph.add_node("fa_llm", fa_llm_node)
    subgraph.add_node("fa_safe_tools", fa_safe_tools_node)
    subgraph.add_node("fa_dangerous_tools", fa_dangerous_tools_node)
    subgraph.add_node("fa_summarize", fa_summarize_node)

    subgraph.add_edge(START, "fa_llm")
    subgraph.add_conditional_edges("fa_llm", _route_after_llm)
    subgraph.add_conditional_edges("fa_safe_tools", _route_after_tools)
    subgraph.add_conditional_edges("fa_dangerous_tools", _route_after_tools)
    subgraph.add_edge("fa_summarize", END)

    # 编译时不指定 checkpointer，由父图（主图）的 MemorySaver 统一负责状态持久化
    return subgraph.compile()

# ══════════════════════════════════════════════════════════════════════════════
# DesktopAgent Subgraph —— 带动态风险拦截的桌面操作 Agent
# ══════════════════════════════════════════════════════════════════════════════

class DesktopAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_tool_call_id: str
    instruction_to_worker: str
    user_query: str

def build_desktop_agent_subgraph(desktop_tools: list, system_prompt: str, llm=None):
    if llm is None:
        llm = models.desktop_llm

    tool_map = {t.name: t for t in desktop_tools}
    
    async def da_llm_node(state: DesktopAgentState) -> dict:
        import re
        msgs = state["messages"]
        sys_msg = SystemMessage(content=system_prompt)
        
        target_tool_call_id = state.get("current_tool_call_id")
        task_instruction = state.get("instruction_to_worker", "")
        user_query = state.get("user_query", "")
        
        split_idx = 0
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], AIMessage) and msgs[i].tool_calls:
                if any(tc["id"] == target_tool_call_id for tc in msgs[i].tool_calls):
                    split_idx = i
                    break
                    
        isolated_start_msg = HumanMessage(content=f"【用户原始请求背景】：{user_query}\n\n【当前核心任务】：{task_instruction}")
        internal_history = msgs[split_idx+1:]
        sanitized_msgs = _sanitize_messages_for_llm(internal_history)
        
        # 🚀 核心大招：动态拦截并处理 take_grid_screenshot 的 base64 返回值，转换成真实的图片格式
        processed_msgs = []
        for msg in sanitized_msgs:
            if isinstance(msg, ToolMessage) and msg.name == "take_grid_screenshot":
                # 1. 插入一个纯文本 ToolMessage 告诉 LLM 工具执行成功
                processed_msgs.append(ToolMessage(
                    content="截图获取成功，已通过下方系统图片消息展示给您。",
                    tool_call_id=msg.tool_call_id,
                    name=msg.name
                ))
                # 2. 紧跟一条 HumanMessage，通过数据 URL 协议把 Base64 数据封装为多模态图片节点，让 VLM 能直接“看”到！
                base64_data = str(msg.content).strip()
                if "base64," in base64_data:
                    base64_data = base64_data.split("base64,")[1]
                # 滤除非 base64 字符
                base64_data = re.sub(r"\s+", "", base64_data)
                
                img_url = f"data:image/jpeg;base64,{base64_data}"
                processed_msgs.append(HumanMessage(content=[
                    {"type": "text", "text": "[系统消息：以下是 DesktopAgent 获取的当前 Windows 屏幕带红色网格线坐标的截图，请通过该截图分析并确定你需要点击或操作的图标的 A1-Z16 坐标]"},
                    {"type": "image_url", "image_url": {"url": img_url}}
                ]))
            else:
                processed_msgs.append(msg)
        
        bound_llm = llm.bind_tools(desktop_tools)
        response = await bound_llm.ainvoke([sys_msg, isolated_start_msg] + processed_msgs)
        return {"messages": [response]}

    async def da_tools_node(state: DesktopAgentState) -> dict:
        last_ai: AIMessage = state["messages"][-1]
        tool_messages = []
        for tc in last_ai.tool_calls:
            tool_name = tc["name"]
            args = tc.get("args", {})
            
            is_high_risk = False
            warning_msg = ""
            if tool_name == "press_hotkey":
                keys = args.get("keys", "").lower()
                high_risk_keys = ["enter", "return", "delete", "alt", "win", "ctrl"]
                if any(k in keys for k in high_risk_keys):
                    is_high_risk = True
                    warning_msg = f"检测到高危按键组合：'{keys}'"
                    
            if is_high_risk:
                print(f"\n🔴 \033[91m[DesktopAgent HITL]\033[0m: 危险工具 '{tool_name}' 请求授权...")
                # 使用 interrupt 函数进行安全拦截，并传入友好的提示信息给前端
                decision: str = interrupt(
                    f"🛡️ 安全系统拦截：DesktopAgent 尝试高危操作。\n"
                    f"原因：{warning_msg}\n"
                    f"工具：{tool_name}\n"
                    f"参数：{args}\n"
                    f"请回复 Y 授权继续，或回复 N 阻断该操作。"
                )
                
                if isinstance(decision, str) and decision.strip().upper() == "Y":
                    print(f"  ✅ [DesktopAgent HITL] 用户已授权执行高危按键：{keys}")
                    # 执行高危工具
                    tool = tool_map.get(tool_name)
                    if tool is None:
                        result = f"工具 '{tool_name}' 未找到。"
                    else:
                        try:
                            result = await tool.ainvoke(args)
                            result = str(result)
                        except Exception as e:
                            result = f"工具执行出错: {e}"
                else:
                    print(f"  🚫 [DesktopAgent HITL] 用户拒绝了高危按键：{keys}")
                    # 注入安全拦截的错误结果，通知 AI 调整计划
                    result = (
                        "CRITICAL ERROR: USER PERMISSION DENIED. "
                        "The user actively rejected this operation. "
                        "DO NOT try to execute this tool or target again. "
                        "Please reply to the user and explain that the operation was denied, "
                        "or try another completely different, non-dangerous way."
                    )
            else:
                # 正常非高危操作
                print(f"  🤖 [DesktopAgent] 正在执行操作: \033[94m{tool_name}\033[0m, 参数: {args}")
                tool = tool_map.get(tool_name)
                if tool is None:
                    result = f"工具 '{tool_name}' 未找到。"
                else:
                    try:
                        result = await tool.ainvoke(args)
                        result = str(result)
                    except Exception as e:
                        result = f"工具执行出错: {e}"
                        
            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tc["id"],
                name=tool_name,
            ))
        return {"messages": tool_messages}

    async def da_summarize_node(state: DesktopAgentState) -> dict:
        msgs = state["messages"]
        worker_text = ""
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage) and msg.content:
                worker_text = msg.content
                break
                
        if isinstance(worker_text, list):
            has_image = any(isinstance(c, dict) and c.get("type") == "image_url" for c in worker_text)
            if not has_image:
                worker_text = "\n".join(c.get("text", "") for c in worker_text if isinstance(c, dict) and c.get("type") == "text")
                
        if "【执行】" in worker_text or "请开始执行" in worker_text:
            if not any(isinstance(m, ToolMessage) for m in state["messages"][-5:]):
                worker_text += "\n⚠️ [警告] 专家未能通过工具获取到实际数据，以上仅为计划描述。"
                
        report_header = f"【内部专家汇报 - 来源：DesktopAgent】\n"
        report_content = report_header + worker_text
                
        tool_call_id = state.get("current_tool_call_id")
        return {"messages": [ToolMessage(content=report_content, tool_call_id=tool_call_id, name="DesktopAgent")]}

    def _route_after_llm(state: DesktopAgentState) -> str:
        last_ai: AIMessage = state["messages"][-1]
        if not getattr(last_ai, "tool_calls", None):
            return "da_summarize"
        return "da_tools"

    subgraph = StateGraph(DesktopAgentState)
    subgraph.add_node("da_llm", da_llm_node)
    subgraph.add_node("da_tools", da_tools_node)
    subgraph.add_node("da_summarize", da_summarize_node)

    subgraph.add_edge(START, "da_llm")
    subgraph.add_conditional_edges("da_llm", _route_after_llm)
    subgraph.add_edge("da_tools", "da_llm")
    subgraph.add_edge("da_summarize", END)

    return subgraph.compile()

# 5. 编排 StateGraph (图的组装)
def build_graph():
    # ── 精准工具分流（基于 __mcp_server__ 来源标签）────────────────────────────
    amap_tools = mcp_service.get_tools_by_server("amap")
    filesystem_tools = mcp_service.get_tools_by_server("filesystem")
    desktop_tools = mcp_service.get_tools_by_server("desktop")
    from .tools import tools as local_tools
    all_mcp_tools = amap_tools + filesystem_tools + desktop_tools

    # 兼容性预处理：工具名中的冒号替换为下划线
    for t in all_mcp_tools:
        if hasattr(t, "name") and ":" in t.name:
            t.name = t.name.replace(":", "__")
    for t in local_tools:
        if hasattr(t, "name") and ":" in t.name:
            t.name = t.name.replace(":", "__")

    WORKER_BASE_PROMPT = (
        "你是一个底层领域专家。你的汇报对象是主管（Supervisor）。你的任务是严格执行主管交代的明确指令。\n"
        "【执行规范 - 强制】：**严禁在调用工具前输出任何“计划”、“思考过程”或“闲聊”文字！** 如果你需要使用工具，请直接输出 tool_calls。只有当工具执行完毕且你拿到了最终结果后，你才进行数据总结和汇报。\n"
        "【职能防火墙 - 极端重要】：你只能且必须只回答主管指派给你的具体子任务。严禁利用你的通用知识去回答用户提问中涉及其他领域的子问题！\n"
        "（例如：如果你是 MapAgent，主管让你找菜市场，你只需返回菜市场信息，绝对严禁顺便提供菜谱或食材清单，即使你知道答案也必须闭嘴，让相关领域的专家去处理）。\n"
        "请只返回客观、精确的工具执行结果或数据总结，禁止说客套话，禁止反问主管。\n"
    )

    knowledge_agent = make_worker_node(
        "KnowledgeAgent",
        [tools.rag, tools.web_search],
        WORKER_BASE_PROMPT + (
            "你是 KnowledgeAgent，负责使用 rag 检索本地知识库以及使用 web_search 获取网络信息。\n"
            "【检索顺序原则】：你必须首先且优先使用 rag 工具。只有当 rag 工具返回的结果明确表示找不到信息，且该问题具有较强的时效性时，才允许使用 web_search 作为最后的兜底手段。\n"
            "【极端重要】：当你收到包含多个子问题的综合指令（如同时查询多个属性或技能）且目标是同一个实体时，"
            "你必须将它们合并为一个涵盖所有关键词的单次 RAG 查询（例如：'火神战姬 Q技能 W技能 力量智力敏捷成长'），"
            "绝对严禁拆分成多次细碎的单独查询调用！"
        ),
        llm=models.vlm
    )

    media_agent = make_worker_node(
        "MediaAgent",
        [tools.av_graph_rag],
        WORKER_BASE_PROMPT + "你是 MediaAgent，负责使用 av_graph_rag 工具检索相关专有图谱数据。"
    )

    map_agent = make_worker_node(
        "MapAgent",
        amap_tools,
        WORKER_BASE_PROMPT + "你是 MapAgent，负责调用高德地图服务工具。必须先获取经纬度再进行周边/坐标类搜索。"
    )

    # ── FileAgent：带 HITL 拦截的子图 ───────────────────────────────────────
    FILE_AGENT_PROMPT = (
        WORKER_BASE_PROMPT
        + "你是 FileAgent，负责操作本地文件系统（Filesystem MCP）。\n"
        + "你可以读取文件内容、查看目录结构、搜索文件、写入/编辑文件、移动文件等。\n"
        + "【安全红线 - 严重警告】\n"
        + f"  · 所有操作必须限定在项目工作区内：{config.WORKSPACE_DIR}\n"
        + "  · 绝对禁止传入含 '../' 的路径或系统绝对路径（如 /etc, /home 等）。\n"
        + "  · 如果你在执行危险操作时触发了 HITL 授权，且用户返回了 'DENIED' 或拒绝信息，"
        + "    你必须立即停止对该工具的尝试。严禁循环请求同一个工具！\n"
        + "    在这种情况下，你应该礼貌地向主管报告该步骤已被用户取消，并根据情况继续其他安全步骤或直接结束任务。\n"
        + "【效率原则】\n"
        + "  · 优先使用 directory_tree 或 list_directory 获取目录全貌，再按需读取具体文件。\n"
        + "  · 读取大文件时，优先使用 search_files 或 get_file_info 确认内容再决定是否 read_file。\n"
        + "【路径适配规范】\n"
        + "  · 如果主管提供的路径包含反斜杠（如 data\\files），你必须将其转换为正斜杠（data/files）后再传给工具。\n"
        + "【关于删除操作的特殊说明】\n"
        + "  · 当前工具集不直接提供 delete_file 工具。如果主管要求你“删除”文件，"
        + "    你必须使用 move_file 工具将目标文件移动到项目根目录下的 'archive_trash' 文件夹中（如果该文件夹不存在，请先用 create_directory 创建它）。"
    )
    file_agent_subgraph = build_file_agent_subgraph(
        filesystem_tools=filesystem_tools,
        system_prompt=FILE_AGENT_PROMPT,
    )

    # 🌟 新增：子图隔离包装器，防止子图内部的琐碎消息污染主图历史
    async def file_agent_node_wrapper(state: AgentState) -> dict:
        task_instruction = state.get("instruction_to_worker", "")
        user_query = state.get("user_query", "")
        
        # 1. 严格控制塞入子图的初始消息，使其与普通 Worker 保持一致的纯净度
        subgraph_input = {
            "messages": [HumanMessage(content=f"【用户原始请求背景】：{user_query}\n\n【当前核心任务】：{task_instruction}")],
            "current_tool_call_id": state.get("current_tool_call_id", ""),
            "instruction_to_worker": task_instruction,
            "user_query": user_query
        }
        
        # 2. 异步调用子图并等待其完成（包括中途可能触发的 HITL 挂起和恢复）
        subgraph_output = await file_agent_subgraph.ainvoke(subgraph_input)
        
        # 3. 核心大招：子图内部打得再热火朝天，我们也只摘取它最后一项由 fa_summarize_node 产出的终极战报
        final_report_msg = subgraph_output["messages"][-1]
        
        # 4. 只将这一条 ToolMessage 吐回主图，完美守住信息防火墙！
        return {"messages": [final_report_msg]}

    # ── DesktopAgent：带动态拦截的桌面操作专家 ───────────────────────────────
    DESKTOP_AGENT_PROMPT = (
        WORKER_BASE_PROMPT
        + "你是 DesktopAgent，负责通过纯视觉和模拟键鼠操作 Windows 桌面。\n"
        + "【⚠️ 桌面操作核心防坑准则】\n"
        + "1. **切回桌面**：如果第一步 take_grid_screenshot 发现屏幕被应用窗口（如 VS Code、浏览器）占满，而你的目标在桌面上，**你必须首先**调用 `press_hotkey(keys='win+d')` 来显示桌面！切换后**必须再次调用 `take_grid_screenshot`** 观察新的桌面网格！\n"
        + "2. **精细坐标校准（防偏左偏右）**：\n"
        + "   - 网格横向为 A-Z（26列），纵向为 1-16（16行）。A在最左，Z在最右。\n"
        + "   - 如果目标在“最右上角”（例如右上角的文件夹），它通常位于 Z1、Y1、Z2、Y2 等最右侧的格子里。W1、P1 明显偏左，请仔细核对 Z, Y, X, W, V 的顺序，避免数错列！\n"
        + "   - 每次决定点击前，在脑海中从最右侧（Z列）向左倒数，核实目标到底在第几列。\n"
        + "【操作流程】\n"
        + "1. 使用 take_grid_screenshot 获取带网格的屏幕截图。\n"
        + "2. 观察截图，若未在桌面则调用 `press_hotkey(keys='win+d')` 切回桌面并重新截图。\n"
        + "3. 仔细对照网格字母和数字，确定目标所在的网格（如 Z1），调用 click_grid_center(grid_id='Z1', click_type='double') 双击打开文件夹。\n"
        + "4. 需要输入中文时，优先使用 safe_input_text，避免拼音输入法干扰。\n"
        + "5. 需要按快捷键时，使用 press_hotkey（如 ctrl+a）。\n"
        + "【重要安全限制】：部分高危按键操作会触发用户授权拦截，请勿滥用。"
    )
    desktop_agent_subgraph = build_desktop_agent_subgraph(
        desktop_tools=desktop_tools,
        system_prompt=DESKTOP_AGENT_PROMPT,
    )
    
    async def desktop_agent_node_wrapper(state: AgentState) -> dict:
        task_instruction = state.get("instruction_to_worker", "")
        user_query = state.get("user_query", "")
        subgraph_input = {
            "messages": [HumanMessage(content=f"【用户原始请求背景】：{user_query}\n\n【当前核心任务】：{task_instruction}")],
            "current_tool_call_id": state.get("current_tool_call_id", ""),
            "instruction_to_worker": task_instruction,
            "user_query": user_query
        }
        subgraph_output = await desktop_agent_subgraph.ainvoke(subgraph_input)
        return {"messages": [subgraph_output["messages"][-1]]}

    # ── CodeAgent：Docker 代码沙箱专家 ──────────────────────────────────────
    CODE_AGENT_PROMPT = (
        WORKER_BASE_PROMPT
        + "你是 CodeAgent，专职编写和执行 Python 代码。"
        + "【画图强制要求】：如果需要绘图（如 matplotlib），必须在代码中显式将图片 savefig 保存到 '/workspace/outputs/' 目录下，且以 .png 结尾。禁止调用 plt.show()。"
        + "沙箱已预装 pandas, numpy, matplotlib。"
    )
    code_agent = make_worker_node(
        "CodeAgent",
        code_tools,
        CODE_AGENT_PROMPT,
    )

    # ── BrowserAgent：网页操作专家 ───────────────────────────────────────────
    BROWSER_AGENT_PROMPT = (
        WORKER_BASE_PROMPT
        + "你是 BrowserAgent，负责处理所有涉及网页内容读取、动态数据抓取、文件上传等网页端操作的任务。\n"
        + "【工具选择策略 - 极端重要】：\n"
        + "1. 优先使用 read_webpage_content 处理用户发来的普通新闻/文章/静态网页链接（仅做纯文本内容提取）。\n"
        + "2. 【特定局限】：只有当遇到 B站用户个人主页/空间链接（如含有 UID 或 space.bilibili.com）且任务是纯粹“查询/分析该UP主资料”时，才允许提取UID并调用 get_bilibili_profile。\n"
        + "3. 【强制执行】：当任务涉及“发布动态”、“发帖”、“上传文件/配图”、“点击”、“登录”、“签到”或输入文本等任何需要真人交互、改变网页状态的操作时，**必须且只能**调用 execute_complex_browser_action！\n"
        + "   - 严禁在调用工具前向主管口头汇报你的计划或分析文字！\n"
        + "   - 直接将整个操作任务（包含用户提供的本地文件路径，如 'workspace/1.png'）作为全局 instruction 传入该工具，由真实的可见浏览器去执行完整的登录和上传流程。"

        + "【⚠️ 动态网页操作红线准则 - 极端重要】：\n"
        + "1. 【严禁原地复读】：如果执行 `click` 或 `input` 工具后提示 `Element index not available`（索引不可用）或没有任何反应，**绝对禁止**在下一步重复尝试同一个 index！\n"
        + "2. 【自救策略】：一旦遇到索引失效、网页无响应或被弹窗遮挡，你必须立即采取以下行动之一：\n"
        + "   - 尝试使用 `scroll_to_element` 滚动页面刷新视图。\n"
        + "   - 放弃盲目点击 index，直接改用 `evaluate` 动作编写一小段强力的 JavaScript 代码进行文本精准匹配点击（例如：`document.querySelectorAll('span').find(el => el.textContent.includes('原创')).click()`）。\n"
        + "3. 【处理弹窗】：上传完封面或勾选协议时，网页会弹出“确认/确定”的蒙版弹窗。你必须先优先寻找并点击这个弹窗里的“确定”按钮（如果点击失效，用 JS evaluate 强行清除或点击），直到弹窗彻底消失，才能去操作底部的“提交稿件”按钮！\n"
        + "4. 所有的发帖、配图交互任务必须直接调用 `execute_complex_browser_action` 进去炸街，严禁向主管口头画饼。"
    )
    browser_agent = make_worker_node(
        "BrowserAgent",
        tools.browser_tools,
        BROWSER_AGENT_PROMPT,
    )

    # ── SQLAgent：MySQL 关系数据库专家 ───────────────────────────────────────────
    SQL_AGENT_PROMPT = (
        WORKER_BASE_PROMPT
        + "你是 SQLAgent，负责根据提供的数据库结构，将用户的自然语言问题转化为 SQL 语句并执行，最后根据查询结果回答用户。\n\n"
        + "【数据库 Schema】\n"
        + "数据库 `av_db` 包含日本AV资料数据，具有以下三张核心表：\n"
        + "1. `actresses` (演员表)\n"
        + "   - `name` (VARCHAR, 主键)\n"
        + "   - `birth_year` (INT, 出生年份)\n"
        + "   - `cup_size` (VARCHAR, 罩杯)\n"
        + "   - `height` (INT, 身高)\n\n"
        + "2. `works` (作品表)\n"
        + "   - `code` (VARCHAR, 主键，如番号)\n"
        + "   - `title` (VARCHAR, 标题)\n"
        + "   - `year` (INT, 发行年份)\n\n"
        + "3. `work_actress` (关联表)\n"
        + "   - `work_code` (VARCHAR, 外键关联 works.code)\n"
        + "   - `actress_name` (VARCHAR, 外键关联 actresses.name)\n\n"
        + "【核心规则】\n"
        + "1. **工具使用**：必须使用 `execute_sql` 工具来执行生成的 SQL。\n"
        + "2. **安全性**：只能生成并执行 `SELECT` 查询，严禁使用 `INSERT`, `UPDATE`, `DELETE`, `DROP` 等操作。\n"
        + "3. **模糊匹配**：如果遇到不知道的精确匹配词，或者用户提供的名字可能不完全准确，请考虑使用 `LIKE` 进行模糊匹配 (如 `name LIKE '%xxx%'`)。\n"
        + "4. **自我纠错**：如果 `execute_sql` 工具返回了错误信息，请仔细阅读错误堆栈，修复你的 SQL 语法或逻辑，并重新调用工具（你最多重试 3 次）。\n"
        + "5. **最终输出**：当获取到查询结果后，请综合用户的原始问题，给出一份清晰、准确的自然语言回答，不要只贴冰冷的数据。\n"
    )
    sql_agent = make_worker_node(
        "SQLAgent",
        [tools.execute_sql],
        SQL_AGENT_PROMPT,
    )

    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve_memory_node", retrieve_memory_node)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("KnowledgeAgent", knowledge_agent)
    workflow.add_node("MediaAgent", media_agent)
    workflow.add_node("MapAgent", map_agent)
    workflow.add_node("FileAgent", file_agent_node_wrapper)
    workflow.add_node("DesktopAgent", desktop_agent_node_wrapper)
    workflow.add_node("BrowserAgent", browser_agent)
    workflow.add_node("CodeAgent", code_agent)
    workflow.add_node("SQLAgent", sql_agent)
    workflow.add_node("extract_memory_node", extract_memory_node)
    
    workflow.add_edge(START, "retrieve_memory_node")
    workflow.add_edge("retrieve_memory_node", "Supervisor")
    
    workflow.add_edge("KnowledgeAgent", "Supervisor")
    workflow.add_edge("MediaAgent", "Supervisor")
    workflow.add_edge("MapAgent", "Supervisor")
    workflow.add_edge("FileAgent", "Supervisor")
    workflow.add_edge("DesktopAgent", "Supervisor")
    workflow.add_edge("BrowserAgent", "Supervisor")
    workflow.add_edge("CodeAgent", "Supervisor")
    workflow.add_edge("SQLAgent", "Supervisor")
    
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next_agent"],
        {
            "KnowledgeAgent": "KnowledgeAgent",
            "MediaAgent": "MediaAgent",
            "MapAgent": "MapAgent",
            "FileAgent": "FileAgent",
            "DesktopAgent": "DesktopAgent",
            "BrowserAgent": "BrowserAgent",
            "CodeAgent": "CodeAgent",
            "SQLAgent": "SQLAgent",
            "FINISH": "extract_memory_node"
        }
    )
    
    workflow.add_edge("extract_memory_node", END)
    
    memory_saver = MemorySaver()
    return workflow.compile(checkpointer=memory_saver)
