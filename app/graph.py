"""
app/graph.py

Subgraph and prompts for human-in-the-loop specialist agent (DesktopAgent).
FileAgent has been migrated to deepagents native FilesystemMiddleware + HumanInTheLoopMiddleware in app/file_agent.py.
"""

from typing import Annotated, List, Optional, Any, Dict
from typing_extensions import TypedDict
import operator
import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from app.infra import models
from app.utils.text import extract_text


def _sanitize_messages_for_llm(messages: list) -> list:
    """
    统一脱敏函数：将消息列表中的多模态内容转换为所有 LLM 兼容的稳健格式。
    保留消息的 additional_kwargs（含 thought_signature）、response_metadata、tool_calls 等关键元数据。
    """
    sanitized = []
    for msg in messages:
        # 提取核心内容
        content = getattr(msg, "content", "")
        
        # 处理内容格式（仅对 list 形式的多模态 content 做清洗，非 list 维持原样）
        if isinstance(content, list):
            clean_parts = []
            for item in content:
                if not isinstance(item, dict): continue
                if item.get("type") == "text":
                    text_val = item.get("text", "")
                    clean_parts.append({"type": "text", "text": str(text_val)})
                elif item.get("type") == "image_url":
                    clean_parts.append(item)
            content = clean_parts if clean_parts else ""

        # 使用 model_copy 仅更新 content，保留其余所有元数据（thought_signature, tool_calls, id 等）
        if hasattr(msg, "model_copy"):
            sanitized.append(msg.model_copy(update={"content": content}))
        else:
            sanitized.append(msg)
            
    return sanitized


def _last_ai_with_calls(messages):
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            return i, m
    return -1, None


def _unanswered_tool_calls(messages):
    """最近一条带 tool_calls 的 AIMessage 里，后面还没有对应 ToolMessage 的调用。"""
    i, ai = _last_ai_with_calls(messages)
    if ai is None:
        return []
    answered = {m.tool_call_id for m in messages[i + 1:] if isinstance(m, ToolMessage)}
    return [tc for tc in ai.tool_calls if tc["id"] not in answered]


# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════

WORKER_BASE_PROMPT = (
    "你是一个底层领域专家。你的汇报对象是主管（Supervisor）。你的任务是严格执行主管交代的明确指令。\n"
    "【执行规范 - 强制】：**严禁在调用工具前输出任何“计划”、“思考过程”或“闲聊”文字！** 如果你需要使用工具，请直接输出 tool_calls。只有当工具执行完毕且你拿到了最终结果后，你才进行数据总结和汇报。\n"
    "【职能防火墙 - 极端重要】：你只能且必须只回答主管指派给你的具体子任务。严禁利用你的通用知识去回答用户提问中涉及其他领域的子问题！\n"
    "（例如：如果你是 MapAgent，主管让你找菜市场，你只需返回菜市场信息，绝对严禁顺便提供菜谱或食材清单，即使你知道答案也必须闭嘴，让相关领域的专家去处理）。\n"
    "请只返回客观、精确的工具执行结果或数据总结，禁止说客套话，禁止反问主管。\n"
)

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

HIGH_RISK_HOTKEYS = ["enter", "return", "delete", "alt", "win", "ctrl"]


# ══════════════════════════════════════════════════════════════════════════════
# DesktopAgent Subgraph —— 带动态风险拦截的桌面操作 Agent
# ══════════════════════════════════════════════════════════════════════════════

class DesktopAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


HISTORICAL_SCREENSHOT_PLACEHOLDER = "（历史截图已省略，请以最新截图为准）"
LATEST_SCREENSHOT_TOOL_MESSAGE = "截图获取成功，已通过下方系统图片消息展示给您。"


def _process_desktop_messages_for_llm(messages: list) -> list:
    """处理 DesktopAgent 消息历史中的截图：
    只对最后一条 take_grid_screenshot 的 ToolMessage 保留多模态图片展示；
    更早的历史截图只保留固定占位文案的 ToolMessage，防止 token 暴涨及坐标混淆。
    """
    last_screenshot_idx = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "take_grid_screenshot":
            last_screenshot_idx = idx

    processed = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "take_grid_screenshot":
            if idx == last_screenshot_idx:
                raw_content = str(msg.content).strip()
                base64_data = raw_content
                if "base64," in base64_data:
                    base64_data = base64_data.split("base64,")[1]
                base64_data = re.sub(r"\s+", "", base64_data)

                # 判断是否为合法 base64（字母数字+/= 且长度 >= 100）
                if bool(re.fullmatch(r"[A-Za-z0-9+/=]+", base64_data)) and len(base64_data) >= 100:
                    # 最新截图且合法：短文本 ToolMessage + 紧随其后的带图 HumanMessage
                    processed.append(ToolMessage(
                        content=LATEST_SCREENSHOT_TOOL_MESSAGE,
                        tool_call_id=getattr(msg, "tool_call_id", ""),
                        name=msg.name,
                    ))
                    img_url = f"data:image/jpeg;base64,{base64_data}"
                    processed.append(HumanMessage(content=[
                        {"type": "text", "text": "[系统消息：以下是 DesktopAgent 获取的当前 Windows 屏幕带红色网格线坐标的截图，请通过该截图分析并确定你需要点击或操作的图标的 A1-Z16 坐标]"},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ]))
                else:
                    # 截图失败（如报错信息）：保持原 ToolMessage 文本不变，不追加带坏图片的 HumanMessage
                    processed.append(msg)
            else:
                # 历史截图：只保留固定占位文案 ToolMessage，不带图片
                processed.append(ToolMessage(
                    content=HISTORICAL_SCREENSHOT_PLACEHOLDER,
                    tool_call_id=getattr(msg, "tool_call_id", ""),
                    name=msg.name,
                ))
        else:
            processed.append(msg)

    return processed


def build_desktop_agent_subgraph(desktop_tools: list, system_prompt: str, llm=None, checkpointer=None):
    if llm is None:
        llm = models.desktop_llm

    tool_map = {t.name: t for t in desktop_tools}
    
    async def da_llm_node(state: DesktopAgentState) -> dict:
        msgs = state["messages"]
        sys_msg = SystemMessage(content=system_prompt)
        sanitized_msgs = _sanitize_messages_for_llm(msgs)
        # 🚀 核心大招：动态拦截并处理 take_grid_screenshot 的 base64 返回值，仅最新截图保留真实图片
        processed_msgs = _process_desktop_messages_for_llm(sanitized_msgs)
        bound_llm = llm.bind_tools(desktop_tools)
        response = await bound_llm.ainvoke([sys_msg] + processed_msgs)
        return {"messages": [response]}

    async def da_tools_node(state: DesktopAgentState) -> dict:
        pending = _unanswered_tool_calls(state["messages"])
        tc = pending[0]
        tool_name = tc["name"]
        args = tc.get("args", {})
        
        is_high_risk = False
        warning_msg = ""
        if tool_name == "press_hotkey":
            keys = args.get("keys", "").lower()
            if any(k in keys for k in HIGH_RISK_HOTKEYS):
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
                    
        return {"messages": [ToolMessage(
            content=result,
            tool_call_id=tc["id"],
            name=tool_name,
        )]}

    async def da_summarize_node(state: DesktopAgentState) -> dict:
        msgs = state["messages"]
        worker_text = ""
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage) and msg.content:
                worker_text = extract_text(msg.content)
                break

        if "【执行】" in worker_text or "请开始执行" in worker_text:
            if not any(isinstance(m, ToolMessage) for m in state["messages"][-5:]):
                worker_text += "\n⚠️ [警告] 专家未能通过工具获取到实际数据，以上仅为计划描述。"

        report_header = "【内部专家汇报 - 来源：DesktopAgent】\n"
        report_content = report_header + worker_text

        return {"messages": [AIMessage(content=report_content)]}

    def _route_after_llm(state: DesktopAgentState) -> str:
        last_ai: AIMessage = state["messages"][-1]
        if not getattr(last_ai, "tool_calls", None):
            return "da_summarize"
        return "da_tools"

    def _after_tools(state: DesktopAgentState) -> str:
        return "da_tools" if _unanswered_tool_calls(state["messages"]) else "da_llm"

    subgraph = StateGraph(DesktopAgentState)
    subgraph.add_node("da_llm", da_llm_node)
    subgraph.add_node("da_tools", da_tools_node)
    subgraph.add_node("da_summarize", da_summarize_node)

    subgraph.add_edge(START, "da_llm")
    subgraph.add_conditional_edges("da_llm", _route_after_llm)
    subgraph.add_conditional_edges("da_tools", _after_tools)
    subgraph.add_edge("da_summarize", END)

    return subgraph.compile(checkpointer=checkpointer)
