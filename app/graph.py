"""
app/graph.py

Subgraphs and prompts for human-in-the-loop specialist agents (FileAgent and DesktopAgent).
The legacy Supervisor/Worker graph has been decommissioned in favor of deepagents architecture.
"""

from typing import Annotated, List, Optional, Any, Dict
from typing_extensions import TypedDict
import operator
import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from . import models
from . import mcp_service
import config


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

FILE_AGENT_PROMPT = (
    WORKER_BASE_PROMPT
    + "你是 FileAgent，负责操作本地文件系统（Filesystem MCP）。\n"
    + "你可以读取文件内容、查看目录结构、搜索文件、写入/编辑文件、移动文件等。\n"
    + "【安全红线 - 严重警告】\n"
    + f"  · 所有操作必须限定在项目工作区内：{config.WORKSPACE_DIR}\n"
    + "  · 绝对禁止传入含 '../' 的路径或系统绝对路径（如 /etc, /home 等）。\n"
    + "  · 文件路径一律相对于工作区根目录。用户或主管写的 \"workspace/xxx\" 指的就是工作区里的 \"xxx\"，传给工具时必须去掉 \"workspace/\" 前缀。\n"
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
# FileAgent Subgraph —— 带 HITL 的文件系统 Agent
# ══════════════════════════════════════════════════════════════════════════════

class FileAgentState(TypedDict):
    """FileAgent 子图专用状态；messages 字段与主图共享 operator.add 语义。"""
    messages: Annotated[List[BaseMessage], operator.add]
    current_tool_call_id: str
    instruction_to_worker: str
    user_query: str


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


def _normalize_fs_args(args: dict) -> dict:
    """对 path、source、destination、directory 字段及 paths 列表，去掉开头的 workspace/ 或 workspace\\"""
    if not isinstance(args, dict):
        return args
    new_args = dict(args)
    def _strip_ws(val: str) -> str:
        return re.sub(r"^workspace[/\\]", "", val, flags=re.IGNORECASE)

    for key in ("path", "source", "destination", "directory"):
        if key in new_args and isinstance(new_args[key], str):
            new_args[key] = _strip_ws(new_args[key])

    if "paths" in new_args and isinstance(new_args["paths"], list):
        new_args["paths"] = [_strip_ws(p) if isinstance(p, str) else p for p in new_args["paths"]]

    return new_args


def build_file_agent_subgraph(filesystem_tools: list, system_prompt: str, llm=None, checkpointer=None):
    """
    构建带 Human-in-the-loop 的 FileAgent 子图。

    子图节点：
      fa_llm            — LLM 决策节点，产出 AIMessage（含 tool_calls）
      fa_safe_tools     — 执行安全工具（read/list/search 等），直接执行不拦截
      fa_dangerous_tools— 执行危险工具前调用 interrupt() 等待用户授权

    子图不携带 Checkpointer（默认），由父图（主图）的 MemorySaver 统一管理状态持久化。
    也可传入 checkpointer 供单测独立测试。
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
        tool_messages = []
        for tc in _unanswered_tool_calls(state["messages"]):
            if tc["name"] in dangerous_names:
                continue  # 危险工具不在这里执行
            norm_args = _normalize_fs_args(tc.get("args") or {})
            print(f"  🤖 [FileAgent] 正在调用安全工具: \033[94m{tc['name']}\033[0m, 参数: {norm_args}")
            tool = tool_map.get(tc["name"])
            if tool is None:
                result = f"工具 '{tc['name']}' 未找到。"
            else:
                try:
                    result = await tool.ainvoke(norm_args)
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
        执行危险工具节点（单步确认模式）。
        取尚未回复的危险调用里的第一个，interrupt() 作为第一个动作，批准则执行、拒绝则返回现有的 CRITICAL ERROR 文本。
        """
        pending = [tc for tc in _unanswered_tool_calls(state["messages"]) if tc["name"] in dangerous_names]
        tc = pending[0]
        norm_args = _normalize_fs_args(tc.get("args") or {})

        print(f"\n🔴 \033[91m[FileAgent HITL]\033[0m: 危险工具 '{tc['name']}' 请求授权...")

        # 每次只处理一个危险工具，遇到就暂停
        decision: str = interrupt({
            "tool_name": tc["name"],
            "tool_args": norm_args,
            "tool_call_id": tc["id"]
        })

        if isinstance(decision, str) and decision.strip().upper() == "Y":
            tool = tool_map.get(tc["name"])
            try:
                result = await tool.ainvoke(norm_args)
                result = str(result)
                if (result.startswith("工具 '") and "执行出错" in result) or result.startswith("工具执行出错"):
                    print(f"  ❌ [FileAgent HITL] 执行失败: {result[:200]}")
                else:
                    print(f"  ✅ [FileAgent HITL] 执行成功")
            except Exception as e:
                result = f"工具执行出错: {e}"
                print(f"  ❌ [FileAgent HITL] 执行失败: {result[:200]}")
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

        return {"messages": [ToolMessage(
            content=result,
            tool_call_id=tc["id"],
            name=tc["name"],
        )]}

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

    def _next_step(state: FileAgentState) -> str:
        msgs = state["messages"]
        if isinstance(msgs[-1], AIMessage) and not msgs[-1].tool_calls:
            return "fa_summarize"
        pending = _unanswered_tool_calls(msgs)
        if any(tc["name"] not in dangerous_names for tc in pending):
            return "fa_safe_tools"
        if any(tc["name"] in dangerous_names for tc in pending):
            return "fa_dangerous_tools"
        return "fa_llm"

    # ── 子图组装 ──────────────────────────────────────────────────────────────

    subgraph = StateGraph(FileAgentState)
    subgraph.add_node("fa_llm", fa_llm_node)
    subgraph.add_node("fa_safe_tools", fa_safe_tools_node)
    subgraph.add_node("fa_dangerous_tools", fa_dangerous_tools_node)
    subgraph.add_node("fa_summarize", fa_summarize_node)

    subgraph.add_edge(START, "fa_llm")
    subgraph.add_conditional_edges("fa_llm", _next_step)
    subgraph.add_conditional_edges("fa_safe_tools", _next_step)
    subgraph.add_conditional_edges("fa_dangerous_tools", _next_step)
    subgraph.add_edge("fa_summarize", END)

    # 编译时支持 checkpointer 参数
    return subgraph.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════════════════════
# DesktopAgent Subgraph —— 带动态风险拦截的桌面操作 Agent
# ══════════════════════════════════════════════════════════════════════════════

class DesktopAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_tool_call_id: str
    instruction_to_worker: str
    user_query: str


def build_desktop_agent_subgraph(desktop_tools: list, system_prompt: str, llm=None, checkpointer=None):
    if llm is None:
        llm = models.desktop_llm

    tool_map = {t.name: t for t in desktop_tools}
    
    async def da_llm_node(state: DesktopAgentState) -> dict:
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
