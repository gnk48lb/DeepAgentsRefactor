"""
app/deep_agent.py

Core implementation of the Main Agent ReAct loop and declarative subagent architecture
using the deepagents framework (Phase 1).

Architecture:
- Main Agent: Built via create_deep_agent with MAIN_AGENT_PROMPT.
- Subagents: 6 tool-specialist subagents (KnowledgeAgent, MediaAgent, MapAgent, CodeAgent, BrowserAgent, SQLAgent)
  delegated to via the deepagents built-in `task` tool.
- Middleware: ToolCallLimitMiddleware guarding the `task` tool (run_limit=12, exit_behavior="continue").
- HITL Subgraphs: FileAgent and DesktopAgent are deferred to subsequent phases.
"""

from typing import List, Optional, Sequence, Any, Dict
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent

# ToolCallLimitMiddleware guard
try:
    from langchain.agents.middleware import ToolCallLimitMiddleware
except ImportError:
    ToolCallLimitMiddleware = None

from . import models
from . import tools
from . import mcp_service
from . import database
import config

# ============================================================================
# Prompts
# ============================================================================

MAIN_AGENT_PROMPT = """你是一个智能中枢助手（Main Agent），拥有强大的工具调用与多专家协同能力。

【运行模式 - ReAct 循环】：
1. 仔细阅读用户的请求与上下文历史。
2. 拆解任务：判断需要调用哪些下属专家或工具。如果任务复杂，先调用专家收集数据，不要凭空猜测。
3. 动态委派：使用 task 工具将明确、具体的子任务委派给专门的专家。每次委派请给出清晰指令。
4. 综合总结：收到专家返回的结果后，判断信息是否完整。如果不足，继续委派；如果已充足，向用户提供最终答复。

【下属专家清单】：
- KnowledgeAgent: 负责检索本地知识库(rag)及网络搜索(web_search)。当需要查找专业资料、历史知识或最新网络资讯时委派。
- MediaAgent: 负责音视频专有图谱检索(av_graph_rag)。
- MapAgent: 负责地理位置、路线规划及周边搜索（高德地图工具）。
- SQLAgent: 负责数据库查询(execute_sql)。只读查询。
- BrowserAgent: 负责控制可见浏览器执行网页操作、多步复杂网页交互。
- CodeAgent: 负责编写和执行 Python 代码及数据可视化。

【专家协同与职能防火墙原则】：
- 严禁越俎代庖：每个专家仅负责自身领域。如果需要多个领域配合（例如“查地图后写代码画图”），请按步骤依次委派给对应专家。
- 工具型专家只能通过 task 工具调用，请在 task 的 instruction 中写明具体要查什么。

【最终回答规范 - 极端重要】：
- 这是用户唯一能看到的内容。你必须完整、详尽地复述专家为你找到的所有核心数据（如文件清单、知识点、代码结果、地图路线）。绝对禁止只给结论不给数据！
- 回答风格亲切、客观、严谨，直接呈现最终综合结果。
"""

WORKER_BASE_PROMPT = (
    "你是一个底层领域专家。你的汇报对象是主管（Supervisor）。你的任务是严格执行主管交代的明确指令。\n"
    "【执行规范 - 强制】：**严禁在调用工具前输出任何\"计划\"、\"思考过程\"或\"闲聊\"文字！** 如果你需要使用工具，请直接输出 tool_calls。只有当工具执行完毕且你拿到了最终结果后，你才进行数据总结和汇报。\n"
    "【职能防火墙 - 极端重要】：你只能且必须只回答主管指派给你的具体子任务。严禁利用你的通用知识去回答用户提问中涉及其他领域的子问题！\n"
    "（例如：如果你是 MapAgent，主管让你找菜市场，你只需返回菜市场信息，绝对严禁顺便提供菜谱或食材清单，即使你知道答案也必须闭嘴，让相关领域的专家去处理）。\n"
    "请只返回客观、精确的工具执行结果或数据总结，禁止说客套话，禁止反问主管。\n"
)

KNOWLEDGE_AGENT_PROMPT = (
    WORKER_BASE_PROMPT + (
        "你是 KnowledgeAgent，负责使用 rag 检索本地知识库以及使用 web_search 获取网络信息。\n"
        "【检索顺序原则】：你必须首先且优先使用 rag 工具。只有当 rag 工具返回的结果明确表示找不到信息，且该问题具有较强的时效性时，才允许使用 web_search 作为最后的兜底手段。\n"
        "【极端重要】：当你收到包含多个子问题的综合指令（如同时查询多个属性或技能）且目标是同一个实体时，"
        "你必须将它们合并为一个涵盖所有关键词的单次 RAG 查询（例如：'火神战姬 Q技能 W技能 力量智力敏捷成长'），"
        "绝对严禁拆分成多次细碎的单独查询调用！"
    )
)

MEDIA_AGENT_PROMPT = (
    WORKER_BASE_PROMPT + "你是 MediaAgent，负责使用 av_graph_rag 工具检索相关专有图谱数据。"
)

MAP_AGENT_PROMPT = (
    WORKER_BASE_PROMPT + "你是 MapAgent，负责调用高德地图服务工具。必须先获取经纬度再进行周边/坐标类搜索。"
)

CODE_AGENT_PROMPT = (
    WORKER_BASE_PROMPT
    + "你是 CodeAgent，专职编写和执行 Python 代码。"
    + "【画图强制要求】：如果需要绘图（如 matplotlib），必须在代码中显式将图片 savefig 保存到 '/workspace/outputs/' 目录下，且以 .png 结尾。禁止调用 plt.show()。"
    + "沙箱已预装 pandas, numpy, matplotlib。"
)

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

# ============================================================================
# Memory Helpers
# ============================================================================

def get_user_long_term_memories(user_query: str, limit: int = 3) -> str:
    """从数据库中检索与用户查询相关的长期记忆"""
    try:
        memories = database.retrieve_memories(user_query, limit=limit)
        if memories:
            return "\n".join([f"- {m}" for m in memories])
    except Exception as e:
        print(f"⚠️ [记忆检索异常]: {e}")
    return "无"

async def _background_extract_memory(recent_chat: str):
    """后台异步执行记忆提取与保存，不阻塞主流程"""
    try:
        final_prompt = MEMORY_EXTRACTOR_PROMPT.format(recent_chat=recent_chat)
        res = await models.extractor_llm.ainvoke(final_prompt)
        content = res.content.strip()
        if content and content.upper() != "NONE" and "NONE" not in content.upper():
            database.insert_memory(content)
            print(f"--- \033[92m[后台记忆保存]\033[0m: {content} ---")
        else:
            print("--- \033[90m[后台记忆跳过]: 未发现长期价值信息\033[0m ---")
    except Exception as e:
        print(f"⚠️ \033[91m[后台记忆提取异常]\033[0m: {str(e)}")

def build_initial_messages(
    user_query: str,
    long_term_memories: str = "无",
    history_messages: Optional[List[BaseMessage]] = None,
) -> List[BaseMessage]:
    """
    构造 deepagents 的初始消息列表。
    注入长期记忆与多轮对话历史。
    """
    messages: List[BaseMessage] = []
    
    if long_term_memories and long_term_memories != "无":
        memory_content = f"【用户长期记忆与偏好】：\n{long_term_memories}"
        messages.append(SystemMessage(content=memory_content))
    
    if history_messages:
        messages.extend(history_messages)
        
    messages.append(HumanMessage(content=user_query))
    return messages

# ============================================================================
# Main Agent Builder
# ============================================================================

async def build_main_agent(checkpointer: Optional[BaseCheckpointSaver] = None):
    """
    构建基于 deepagents 的主 Agent 实例。
    - 主模型: models.worker_llm (ChatGoogleGenerativeAI)
    - 子专家: 6 个工具型专家声明式接入 (task 委派)
    - 中间件: ToolCallLimitMiddleware 防御 task 工具死循环
    - 持久化: 默认使用 MemorySaver
    """
    # 确保 MCP 服务已初始化
    await mcp_service.initialize_mcp()
    amap_tools = mcp_service.get_tools_by_server("amap")

    # 配置中间件
    middleware = []
    if ToolCallLimitMiddleware is not None:
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name="task",
                run_limit=12,
                exit_behavior="continue",
            )
        )
    else:
        print("[-] Warning: ToolCallLimitMiddleware not available, task tool limit disabled.")

    # 声明式 subagents 列表
    subagents = [
        {
            "name": "KnowledgeAgent",
            "description": "负责检索本地知识库(rag)及网络搜索(web_search)。当需要查找专业资料、历史知识或最新网络资讯时委派。",
            "system_prompt": KNOWLEDGE_AGENT_PROMPT,
            "tools": [tools.rag, tools.web_search],
            "model": models.worker_llm,
        },
        {
            "name": "MediaAgent",
            "description": "负责使用 av_graph_rag 工具检索音视频专有图谱数据。",
            "system_prompt": MEDIA_AGENT_PROMPT,
            "tools": [tools.av_graph_rag],
            "model": models.worker_llm,
        },
        {
            "name": "MapAgent",
            "description": "负责调用高德地图服务工具查询地理位置、路线规划及周边搜索。",
            "system_prompt": MAP_AGENT_PROMPT,
            "tools": amap_tools,
            "model": models.worker_llm,
        },
        {
            "name": "CodeAgent",
            "description": "专职编写和执行 Python 代码及数据可视化。",
            "system_prompt": CODE_AGENT_PROMPT,
            "tools": tools.code_tools,
            "model": models.worker_llm,
        },
        {
            "name": "BrowserAgent",
            "description": "负责处理所有涉及网页内容读取、动态数据抓取、文件上传等网页端操作的任务。",
            "system_prompt": BROWSER_AGENT_PROMPT,
            "tools": tools.browser_tools,
            "model": models.worker_llm,
        },
        {
            "name": "SQLAgent",
            "description": "负责根据提供的数据库结构，将用户的自然语言问题转化为 SQL 语句并执行查询。",
            "system_prompt": SQL_AGENT_PROMPT,
            "tools": [tools.execute_sql],
            "model": models.worker_llm,
        },
        # ====================================================================
        # NOTE [Subsequent Phases]:
        # FileAgent and DesktopAgent involve interactive HITL (Human-in-the-loop)
        # compiled subgraphs and approval mechanisms. They will be integrated
        # in Phase 2/3. Do NOT add them here in Phase 1 to avoid runtime schema mismatches.
        # ====================================================================
    ]

    if checkpointer is None:
        checkpointer = MemorySaver()

    agent = create_deep_agent(
        model=models.worker_llm,
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=subagents,
        middleware=middleware,
        checkpointer=checkpointer,
    )

    return agent
