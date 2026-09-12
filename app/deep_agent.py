"""
app/deep_agent.py

Core implementation of the Main Agent ReAct loop and declarative subagent architecture
using the deepagents framework (Phase 1).

Architecture:
- Main Agent: Built via create_deep_agent with MAIN_AGENT_PROMPT, driven by models.supervisor_llm.
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
# Prompts & Descriptions
# ============================================================================

MAIN_AGENT_PROMPT = """你是一个多智能体系统的主 Agent，负责理解用户的问题，在需要时通过 task 工具委派给下属专家，收集到足够信息后直接给用户答案。

【信息完整性 —— 最高优先级】
用户完全看不到任何专家的中间汇报过程。你必须像"信息搬运工"，把专家汇报里的每一项具体数据（完整路径、检索原文、执行结果、数字等）原文复制进最终回答，绝不能只给结论不给数据。
如果专家没有给出明确的原始内容，如实回答"未能获取到相关内容"，绝不能凭空编造。绝不能把专家的"计划/步骤描述"误当成"执行结果"。
task 工具返回的所有内容都是专家的汇报，不是用户在说话。

【自包含原则】
不要假设用户看过之前的战报，每一次回答都必须独立、完整。禁止用"已为您列出""如上所述"这类概括性描述代替实际内容。

【何时委派 vs 直接回答】
只有当对话历史或系统消息里提供的长期记忆已经包含完整回答所需的全部信息时，才可以不委派、直接回答。涉及知识查询（做法、地址、事实、代码、数据库等）必须先委派给对应专家——每个专家的职责范围写在它们各自的描述里，派发前先确认问题落在谁的范围内。

【委派规范】
- 给专家的任务描述里禁止使用"他""那里""那个"等代词，必须替换成具体名词。
- 需要组合能力时（比如先查到地点再导航），分步委派：先拿到一个专家的结果，再据此委派下一个，不要指望一个专家做复合任务。
- 同一个专家已经尝试过仍拿不到新信息时，不要反复重新委派同一件事，直接基于已有信息回答，或如实告知未能获取。

【过程不可见】
在决定委派或继续收集信息的过程中，不要输出你的思考过程、计划或"我准备去问一下 XX"这类文字——这些不会展示给用户。只有当你不再调用 task、准备给出最终结论时，你输出的内容才是用户会看到的内容。

【图文分离】
专家汇报中如果包含图片，你能在上下文里直接看到并据此分析，但绝对不要在最终回答里生成图片链接、Base64 或 Markdown 图片语法——下游系统会自动把图片发给用户，你只需要输出基于图片分析出的文字结论。

【长期记忆】
如果这位用户有长期记忆（家庭住址、过敏史、固定习惯等），会在对话最前面以一条系统消息的形式提供给你，请结合它理解和回答问题，不要在回答里生硬地复述"我记得你..."。

【回复要求】
最终回答必须精简、干脆，控制在 500 字以内，以适配 QQ 机器人的显示限制。
"""

SUBAGENT_DESCRIPTIONS = {
    "KnowledgeAgent": "知识百科专家。处理菜谱、游戏攻略、历史、科学等纯知识性咨询，优先检索本地知识库。不处理地理位置相关问题（在哪买/怎么去/天气），也不处理代码、本地文件、目录浏览。",
    "MediaAgent": "成人影视专家（高优先级）。处理 AV、女优、番号等综合性或模糊搜索、关系网络咨询。不涉及精确条件过滤的问题优先派给这个，而不是 SQLAgent。",
    "MapAgent": "地理出行专家，挂载高德地图工具。处理地理编码、路线规划（步行/骑行/驾车）、周边搜索（POI）、天气查询。",
    "SQLAgent": "关系数据库专家，用于精确过滤、统计或多条件检索女优、作品及关联表关系。不涉及精确条件的模糊/综合查询交给 MediaAgent。",
    "BrowserAgent": "浏览器操作专家。处理网页点击、动态数据抓取、文件上传与自动化发帖。复杂规划中，需要先派这个拿到地址等信息，再派 MapAgent。",
    "CodeAgent": "代码执行专家，用 Python 解决数学计算、数据处理和绘图任务。只能写代码运行，不能帮你查资料；需要绘图时必须明确指示它绘图。",
}

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

def build_initial_messages(
    user_query: str,
    user_id: str = "",
    history_messages: Optional[List[BaseMessage]] = None,
) -> List[BaseMessage]:
    """
    构造 deepagents 的初始消息列表。
    自动查询 Milvus 长期记忆库并注入，同时支持多轮对话历史。
    """
    messages: List[BaseMessage] = []
    memories = database.search_memory(user_query, top_k=1)
    if memories:
        messages.append(SystemMessage(content="【用户长期记忆】\n" + "\n".join(memories)))
    if history_messages:
        messages.extend(history_messages)
    messages.append(HumanMessage(content=user_query, name="user"))
    return messages

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

# ============================================================================
# Main Agent Builder
# ============================================================================

async def build_main_agent(checkpointer: Optional[BaseCheckpointSaver] = None):
    """
    构建基于 deepagents 的主 Agent 实例。
    - 主模型: models.supervisor_llm (ChatGoogleGenerativeAI / gemini-3.1-flash-lite)
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
            "description": SUBAGENT_DESCRIPTIONS["KnowledgeAgent"],
            "system_prompt": KNOWLEDGE_AGENT_PROMPT,
            "tools": [tools.rag, tools.web_search],
            "model": models.vlm,
        },
        {
            "name": "MediaAgent",
            "description": SUBAGENT_DESCRIPTIONS["MediaAgent"],
            "system_prompt": MEDIA_AGENT_PROMPT,
            "tools": [tools.av_graph_rag],
            "model": models.worker_llm,
        },
        {
            "name": "MapAgent",
            "description": SUBAGENT_DESCRIPTIONS["MapAgent"],
            "system_prompt": MAP_AGENT_PROMPT,
            "tools": amap_tools,
            "model": models.worker_llm,
        },
        {
            "name": "CodeAgent",
            "description": SUBAGENT_DESCRIPTIONS["CodeAgent"],
            "system_prompt": CODE_AGENT_PROMPT,
            "tools": tools.code_tools,
            "model": models.worker_llm,
        },
        {
            "name": "BrowserAgent",
            "description": SUBAGENT_DESCRIPTIONS["BrowserAgent"],
            "system_prompt": BROWSER_AGENT_PROMPT,
            "tools": tools.browser_tools,
            "model": models.worker_llm,
        },
        {
            "name": "SQLAgent",
            "description": SUBAGENT_DESCRIPTIONS["SQLAgent"],
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
        model=models.supervisor_llm,
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=subagents,
        middleware=middleware,
        checkpointer=checkpointer,
    )

    return agent
