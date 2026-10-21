import time
"""
app/deep_agent.py

Core implementation of the Main Agent ReAct loop and declarative subagent architecture
using the deepagents framework (Phase 1).

Architecture:
- Main Agent: Built via create_deep_agent with MAIN_AGENT_PROMPT.
- Subagents:
  * 基础专家与本地操作类专家（FileAgent / DesktopAgent / BrowserAgent）始终挂载
- Middleware: ToolCallLimitMiddleware guarding the `task` tool (run_limit=12, exit_behavior="continue").
- HITL: FileAgent 为 create_agent 编译图（原生 HITL），DesktopAgent 仍为手写子图。
"""

import asyncio
import re
from typing import List, Optional
from app.utils.text import extract_text
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

# 长期记忆块标签常量
MEMORY_TAG_OPEN = "<long_term_memory>"
MEMORY_TAG_CLOSE = "</long_term_memory>"
_MEMORY_BLOCK_REGEX = re.compile(
    rf"{re.escape(MEMORY_TAG_OPEN)}\s*[\s\S]*?\s*{re.escape(MEMORY_TAG_CLOSE)}\s*",
    re.MULTILINE,
)


def strip_memory_block(text: str) -> str:
    """去除文本中由系统注入的 <long_term_memory> 块，还原用户原话。"""
    return _MEMORY_BLOCK_REGEX.sub("", text)

from deepagents import (
    create_deep_agent,
    CompiledSubAgent,
    register_harness_profile,
    HarnessProfile,
    GeneralPurposeSubagentProfile,
)

# 确保关掉默认的 general-purpose subagent（由专属专家全面承接任务，避免不可控兜底）
# 同时隐藏内置的危险/修改工具（write_file, edit_file, delete, execute）
# 注：FileAgent 由 create_agent 构建，不受此处 excluded_tools 影响
for _provider in ("google_genai", "openai"):
    try:
        register_harness_profile(
            _provider,
            HarnessProfile(
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                excluded_tools=frozenset({"write_file", "edit_file", "delete", "execute"}),
            ),
        )
    except Exception as e:
        print(f"[-] Warning: Failed to register harness profile for {_provider}: {e}")

# ToolCallLimitMiddleware guard
try:
    from langchain.agents.middleware import ToolCallLimitMiddleware
except ImportError:
    ToolCallLimitMiddleware = None

from app.tools import amap
from app.tools import markitdown_tool
from app.tools import knowledge_tools
from app.tools import media_tools
from app.tools import sql_tools
from app.tools import code_tools
from app.tools import browser_tools
from app.infra import models
from app.infra import mcp_service
from app.memory import store as memory_store
from app import file_agent
from app.graph import (
    build_desktop_agent_subgraph,
    DESKTOP_AGENT_PROMPT,
)
from app import config
from app.skills import build_skills_backend, SKILLS_ROUTE


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
你自己无法得知当前时间：任何涉及“今天/现在/几点/星期几/还有几天”的问题都必须委派给 KnowledgeAgent（它能查询当前时间）。

【委派规范】
- 给专家的任务描述里禁止使用"他""那里""那个"等代词，必须替换成具体名词。
- 需要组合能力时（比如先查到地点再导航），分步委派：先拿到一个专家的结果，再据此委派下一个，不要指望一个专家做复合任务。
- 同一个专家已经尝试过仍拿不到新信息时，不要反复重新委派同一件事，直接基于已有信息回答，或如实告知未能获取。
- 你自带的 ls/read_file/glob/grep 只作用于一个临时的虚拟空间，访问不到用户本机的文件。任何本地文件的读取、修改，一律委派给 FileAgent。
- 任何涉及 Windows 桌面、屏幕图标、打开桌面文件夹、双击桌面文件或应用、GUI 窗口交互的任务，必须且只能委派给 DesktopAgent，绝不能自己给出指导性文字，绝不能要求用户手动操作。
- 不要在 description 里要求专家提供工具拿不到的信息（例如工具没有返回的距离）。
- task 的 description 是子专家能看到的全部信息——它看不到用户的原话、对话历史和长期记忆。完成任务所需的上下文（用户位置/地址、具体名称、约束条件、本地文件路径、上一步专家返回的关键数据）都必须写进 description。例如派 MapAgent 搜附近的菜市场，要写"用户当前位置在济南市历下区文化东路42号，请搜索附近的菜市场"，不能只写"帮用户查去哪买菜"。
- 成人影视领域的任务分工：
  * 关系网络、共同出演、合作过相同男优/女优、剧情题材反差标签（如熟女、人妻）归 MediaAgent（图谱/语义检索）；
  * 女优精确身体属性（身高 height、罩杯 cup_size、出生年份/年龄 birth_year）、作品发行年份及数量统计归 SQLAgent（数据库精确过滤）。
  * 遇到复合问题时分工协作：不要让 MediaAgent 去猜数据库里的身高/年龄数值，需要精确数据过滤或统计时委派给 SQLAgent 执行。
  * 【严格数据边界】：数据库 works 表只有番号 code、标题 title 和年份 year，绝无任何题材标签或剧情分类表！剧情题材、反差萌标签（如大姐姐、熟女、人妻等）100% 仅存在于 MediaAgent 的图谱中。严禁委派 SQLAgent 去查询题材标签！若 MediaAgent 未检索到匹配标签的作品，直接如实向用户说明，严禁转派给 SQLAgent 盲查。

【过程不可见】
在决定委派或继续收集信息的过程中，不要输出你的思考过程、计划或"我准备去问一下 XX"这类文字——这些不会展示给用户。只有当你不再调用 task、准备给出最终结论时，你输出的内容才是用户会看到的内容。

【图文分离】
专家汇报只会以文字形式回到你这里，你看不到图片本身。如果汇报提到已生成图表或图片，系统会自动把图片发给用户：你只需要用文字交代结论，不要生成图片链接、Base64 或 Markdown 图片语法，也不要说"无法展示图片"。

【长期记忆】
如果这位用户有长期记忆（家庭住址、过敏史、固定习惯等），会附在用户消息开头的 <long_term_memory> 块里，请结合它理解和回答问题，不要在回答里生硬地复述"我记得你..."。

【回复要求】
最终回答要精简、干脆，通常不超过 500 字，不要寒暄、重复或解释推理过程；但专家汇报里的具体数据（步骤、列表、数字、路径等）必须完整保留，数据较多时可以更长，最多不超过 2000 字。
网址/链接不要原样复制（QQ 会拒收含链接的消息），需要标注来源时只写站点或文章名称。
最终回答使用纯文本，不要使用 Markdown 语法（不要 **加粗**、# 标题、代码围栏、行首的 * 列表符）；需要分点时写"1. 2. 3."或换行加"- "。专家汇报里的 Markdown 标记要去掉，但数据和数字必须原样保留。
不要向用户提及内部专家或子 Agent 的名字（如 FileAgent、CodeAgent、MapAgent 等），需要说明时只说"我"或"系统"。
不要向用户提及沙箱内部的文件路径（如 /workspace/outputs/...）；图片由系统另行发送，你只用文字交代结论。
"""

SUBAGENT_DESCRIPTIONS = {
    "KnowledgeAgent": "知识百科专家。处理菜谱、游戏攻略、历史、科学等纯知识性咨询，优先检索本地知识库；同时负责回答当前日期、时间、星期几，以及以“现在”为基准的日期推算（如距离某天还有几天）。不处理地理位置相关问题（在哪买/怎么去/天气），也不处理代码、本地文件、目录浏览。",
    "MediaAgent": "成人影视与图谱专家。处理 AV、女优/男优合作关系网络（共同出演、合作过相同男优/女优）、番号查询、以及剧情题材标签（如人妻、熟女、剧情等）的语义和图谱检索。注意：它只有作品、演员和题材标签，绝无女优的身体数值属性（身高、罩杯、出生年份）；凡涉及具体身高、罩杯、出生年份/年龄区间等精确属性过滤或统计，属于 SQLAgent 的职责范围。",
    "MapAgent": "地理出行专家，挂载高德地图工具。处理地理编码、路线规划（步行/骑行/驾车）、周边搜索（POI）、天气查询。",
    "SQLAgent": "关系数据库专家，用于精确过滤、统计或多条件检索女优及作品。数据库 actresses 表包含精确的姓名 name、身高 height、罩杯 cup_size、出生年份 birth_year；works 表包含番号 code、标题 title、发行年份 year。注意：数据库没有任何题材、标签或反差萌字段，严禁派它查询题材标签！凡涉及女优具体身体数值属性（如身高在某区间、指定罩杯、某年份前后出生）或数量统计，必须派给它执行 SQL 精确查询。",
    "BrowserAgent": "浏览器操作专家。处理网页点击、登录、动态数据抓取、文件上传与自动化发帖。它可以直接接收本地文件路径（如 workspace/1.png）完成网页上传，无需先派 FileAgent 读取；只要任务涉及网页端发布/登录/点击等交互，就直接派给它，不要提前结束并让用户手动操作。复杂规划中，需要先派这个拿到地址等信息，再派 MapAgent。",
    "CodeAgent": "代码执行专家，用 Python 解决数学计算、数据处理和绘图任务。只能写代码运行，不能帮你查资料；需要绘图时必须明确指示它绘图。",
    "FileAgent": (
        "本地文件专家，是唯一能访问本地磁盘（workspace 工作区）的专家：列目录、搜索、读取文本类文件，"
        "支持用 convert_to_markdown 读取 PDF、Word、PPT、Excel、HTML、EPub、CSV、ZIP 等文档，"
        "写入、编辑、新建目录、移动或重命名文件（“删除”一律改为移动到 archive_trash 归档）。"
        "写入/编辑/新建目录/移动会暂停并请求人工授权。"
        "不要让它读取纯图片、音视频等媒体文件；"
        "网页上传文件的任务不要先派它读取，直接把路径交给 BrowserAgent。"
    ),
}

WORKER_BASE_PROMPT = (
    "你是一个底层领域专家。你的汇报对象是主管（Supervisor）。你的任务是严格执行主管交代的明确指令。\n"
    "【执行规范 - 强制】：**严禁在调用工具前输出任何\"计划\"、\"思考过程\"或\"闲聊\"文字！** 如果你需要使用工具，请直接输出 tool_calls。只有当工具执行完毕且你拿到了最终结果后，你才进行数据总结和汇报。\n"
    "【职能防火墙 - 极端重要】：你只能且必须只回答主管指派给你的具体子任务。严禁利用你的通用知识去回答用户提问中涉及其他领域的子问题！\n"
    "（例如：如果主管让你找菜市场，你只需返回菜市场信息，绝对严禁顺便提供菜谱或食材清单，即使你知道答案也必须闭嘴，让相关领域的专家去处理）。\n"
    "请只返回客观、精确的工具执行结果或数据总结，禁止说客套话，禁止反问主管。\n"
)

FILE_AGENT_PROMPT = (
    WORKER_BASE_PROMPT
    + "你是 FileAgent，负责操作本地工作区（workspace）里的文件。\n"
    + "【工具】浏览/读取：ls、glob、grep、read_file；写入类（会暂停等待用户授权）：write_file、edit_file、create_directory、move_file。\n"
    + "【路径规范 - 强制】\n"
    + "  · 所有路径一律写成 /workspace/xxx（也接受 workspace/xxx）。用户或主管写的 \"workspace/xxx\" 指的就是工作区里的 xxx，原样传给工具，不要去掉前缀。\n"
    + "  · 只能在工作区内操作：不要使用 ../、Windows 盘符路径（如 C:\\...）或其他系统绝对路径。路径含反斜杠时先改成正斜杠。\n"
    + "  · ls / glob / grep 必须显式传 path，从 /workspace（或其子目录）开始；不要对 \"/\" 列目录或搜索。\n"
    + "【效率原则】\n"
    + "  · 先用 ls 或 glob(pattern=\"**/*\", path=\"/workspace\") 了解目录全貌，再按需 read_file。\n"
    + "  · 大文件用 read_file 的 offset / limit 分页读取（每次不超过 200 行），不要一次读完整个大文件；找内容优先用 grep。\n"
    + "  · 纯文本与代码文件用 read_file（严禁对二进制或富文档调用 read_file）；遇到图片、音视频等媒体文件直接汇报无法读取。\n"
    + "【写入规范】\n"
    + "  · write_file 会整体覆盖已存在的文件，并自动创建父目录；覆盖前先 read_file 确认现有内容。局部修改优先 edit_file。\n"
    + "  · 没有删除工具。主管要求“删除”文件时，用 move_file 把它移动到 /workspace/archive_trash/ 下（目标同名已存在时，在文件名后加序号或时间戳）。\n"
    + "  · 需要新建空文件夹时用 create_directory。\n"
    + "【授权被拒】写入类操作会暂停等待用户授权。如果被用户拒绝（收到“用户明确拒绝/权限被拒绝”的提示），你必须立即停止对该操作及类似操作的尝试，严禁循环请求；礼貌地向主管汇报该步骤已被用户取消，并根据情况继续其他安全步骤或直接结束任务。\n"
    + "【汇报格式】最终回答以“【内部专家汇报 - 来源：FileAgent】”开头，然后给出工具实际返回的结果（路径、文件内容、执行结果等）；没有拿到工具结果时如实说明，禁止把计划当成结果。\n"
)

FILE_AGENT_MARKITDOWN_PROMPT = (
    "【文档转换】PDF、Word、PowerPoint、Excel、HTML、EPub、CSV、ZIP 等需要转成 Markdown 才能阅读的文件，"
    "调用 convert_to_markdown；纯文本、代码文件直接用 read_file 更省 token。图片、音视频仍不要读取。\n"
    "【转换路径】convert_to_markdown 的 uri 直接写成 /workspace/<相对路径>（或 file:///workspace/<相对路径>），"
    "例如 /workspace/报告/a.pdf。不要传 http(s) 链接，不要传工作区外的路径。\n"
    "【转换结果过大】结果超过阈值时，系统不会直接返回正文，而是把它存到 /_artifacts/large_tool_results/<id> 并告诉你这个路径。"
    "这个路径是合法的（前面“路径一律写成 /workspace/xxx”的规则对它不适用）：原样传给 read_file"
    "（用 offset / limit 分页，每次不超过 200 行）或 grep，不要改写成 /workspace/...，也不要去掉 /_artifacts 前缀。\n"
)

KNOWLEDGE_AGENT_PROMPT = (
    WORKER_BASE_PROMPT + (
        "你是 KnowledgeAgent，负责使用 rag 检索本地知识库、使用 web_search 获取网络信息，并使用 get_current_time 回答当前日期/时间类问题。\n"
        "【时间类问题】：凡是询问当前日期、现在几点、今天星期几、今年是哪一年，或需要以“现在”为基准推算（还有几天、几天前/后、下周几是几号）的问题，直接调用 get_current_time，不要先用 rag 或 web_search。推算只基于该工具返回的时间；汇报时同时写出工具返回的原始时间和你的推算结果。\n"
        "【检索顺序原则】：（时间类问题除外）你必须首先且优先使用 rag 工具。只有当 rag 工具返回的结果明确表示找不到信息，且该问题具有较强的时效性时，才允许使用 web_search 作为最后的兜底手段。\n"
        "【极端重要】：当你收到包含多个子问题的综合指令（如同时查询多个属性或技能）且目标是同一个实体时，"
        "你必须将它们合并为一个涵盖所有关键词的单次 RAG 查询（例如：'火神战姬 Q技能 W技能 力量智力敏捷成长'），"
        "绝对严禁拆分成多次细碎的单独查询调用！"
    )
)

MEDIA_AGENT_PROMPT = (
    WORKER_BASE_PROMPT
    + "你是 MediaAgent，专职使用 av_graph_rag 工具检索成人影视作品、番号、女优/男优合作关系网络及题材标签（如人妻、熟女、反差等）。\n"
    + "【能力边界 - 强制】：av_graph_rag 工具仅返回作品番号、标题、参演演员和题材标签；图谱库中**绝不包含**女优的身高、罩杯、出生年份等精确身体数值！\n"
    + "严禁凭借预训练记忆编造女优的身高或年龄。若主管指派的任务中包含过滤具体身高、罩杯或出生年份，只汇报你通过工具实际检索到的演员与作品信息，并明确提示主管：女优的具体身体属性（身高/罩杯/出生年份）属于数据库结构化数据，需由主管派发给 SQLAgent 查询。\n"
    + "【输出规范 - 强制】：当 av_graph_rag 返回检索结果后，必须直接总结工具返回的番号、标题与演员信息，严禁返回空白内容！若工具未检索到符合条件的特定标签作品，请明确陈述“未检索到带有 XX 标签的相关作品”，禁止输出空字符串。"
)

MAP_AGENT_PROMPT = (
    WORKER_BASE_PROMPT
    + "你是 MapAgent，负责调用高德地图服务工具。必须先获取经纬度再进行周边/坐标类搜索。\n"
    + "只报告工具实际返回的字段。工具没有返回距离就不要写距离，更不要给不同地点填同一个距离；确实需要距离时，对每个地点分别调用距离/路线工具。"
)

CODE_AGENT_PROMPT = (
    WORKER_BASE_PROMPT
    + "你是 CodeAgent，专职编写和执行 Python 代码解决纯数学计算、数据处理和绘图任务。\n"
    + "【运行环境限制 - 极端重要】：你运行在完全隔离且每次调用后即销毁的临时 Docker 沙箱中，绝对无法读写或修改用户宿主机的本地文件（如 workspace/ 目录下的文件）。严禁执行 Python 代码尝试读取或改写用户宿主机文件！若主管要求你读写或改写本地文件，请直接汇报无法访问或修改宿主机本地文件，严禁调用 execute_python_code 反复尝试重试！\n"
    + "【绘图】凡是需要输出图片的任务，写代码之前必须先用 read_file 读取技能文件 /skills/matplotlib-chart/SKILL.md，并严格遵循其中的规范（固定开头、保存路径、中文字体、统一样式）。read_file 只用于读取 /skills/ 下的技能文件，不能读取用户本地文件；技能里的 /workspace/outputs/ 是沙箱内路径，与 read_file 的虚拟路径是两个不同的空间。不需要出图的纯计算任务无需读取技能。\n"
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
    + "   - 直接将整个操作任务（包含用户提供的本地文件路径，如 'workspace/1.png'）作为全局 instruction 传入该工具，由真实的可见浏览器去执行完整的登录和上传流程。\n"
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
    + "3. **字段与能力防火墙**：数据库中没有任何题材、风格、标签（Tag）或反差萌分类表/字段！works 表只有标题 title 和年份 year。严禁在数据库中查找题材或标签；若主管询问题材/标签，只需查询 works 标题，若标题未体现即客观告知数据库无此类标签记录。\n"
    + "4. **止损与防循环（极端重要）**：\n"
    + "   - 执行合理的 SQL 查询后，若返回空结果 `()`，最多允许尝试 1 次合理的名字模糊匹配放宽，严禁为了“求证”而连续发起多次查询！\n"
    + "   - 严禁为了确认记录数而无意义地查询 `COUNT(*)`，严禁将 `IN` 换成 `OR` 进行语法等价重复查询，严禁反复添加 `LIMIT` 重试！\n"
    + "   - 只要已经查出了目标演员的全部作品清单，就立刻根据该清单判断并汇报结论，严禁反复空转！\n"
    + "5. **自我纠错**：仅当 `execute_sql` 返回语法错误或异常堆栈时才修复 SQL 语法并重试（最多重试 2 次）；查询成功返回空列表不是错误，切勿当成语法错误重试！\n"
    + "6. **最终输出**：当获取到查询结果后，请综合主管的指令，给出一份清晰、准确的自然语言战报，不要只贴冰冷的数据。\n"
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
    history_messages: Optional[List[BaseMessage]] = None,
) -> List[BaseMessage]:
    """
    构造 deepagents 的初始消息列表。

    前缀静态原则（为了 Gemini 前缀缓存命中）：
    - 绝不在此产生任何 SystemMessage。若在历史中引入 SystemMessage，langchain-google-genai
      会在多轮中把中间的 SystemMessage extend 进 system_instruction，导致系统指令动态变长、
      请求前缀每轮改变，前缀缓存整段失效。
    - 检索到的长期记忆（top_k=1）固定以 <long_term_memory> 块拼在当轮 HumanMessage 开头；
      未检索到记忆时保持纯 user_query。
    """
    messages: List[BaseMessage] = []
    memories = memory_store.search_memory(user_query, top_k=1)
    if memories:
        query_content = f"{MEMORY_TAG_OPEN}\n" + "\n".join(memories) + f"\n{MEMORY_TAG_CLOSE}\n{user_query}"
    else:
        query_content = user_query

    if history_messages:
        messages.extend(history_messages)
    messages.append(HumanMessage(content=query_content, name="user"))
    return messages

async def _background_extract_memory(recent_chat: str):
    """后台异步执行记忆提取与保存，不阻塞主流程"""
    try:
        clean_chat = strip_memory_block(recent_chat)
        final_prompt = MEMORY_EXTRACTOR_PROMPT.format(recent_chat=clean_chat)
        from app.runtime import cli_logger
        mem_start = time.perf_counter()
        res = await models.extractor_llm.ainvoke(
            final_prompt,
            config={"callbacks": [cli_logger.CLIDebugCallbackHandler()]},
        )
        mem_elapsed = time.perf_counter() - mem_start
        content = extract_text(res.content).strip()
        if content and content.upper() != "NONE" and "NONE" not in content.upper():
            await asyncio.to_thread(memory_store.insert_memory, content)
            print(f"--- \033[92m[后台记忆保存] (耗时 {mem_elapsed:.2f}s)\033[0m: {content} ---", flush=True)
        else:
            print(f"--- \033[90m[后台记忆跳过] (耗时 {mem_elapsed:.2f}s): 未发现长期价值信息\033[0m ---", flush=True)
    except Exception as e:
        print(f"⚠️ \033[91m[后台记忆提取异常]\033[0m: {str(e)}", flush=True)

# ============================================================================
# Main Agent Builder
# ============================================================================

async def build_main_agent(
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """
    构建基于 deepagents 的主 Agent 实例（单用户：本地操作类专家始终挂载）。
    - 主模型: models.supervisor_llm (ChatGoogleGenerativeAI / gemini-3.1-flash-lite)
    - 子专家: 声明式接入 (task 委派)，包含 FileAgent / DesktopAgent / BrowserAgent 等全部 8 个专家
    - 中间件: ToolCallLimitMiddleware 防御 task 工具死循环
    - 持久化: 默认使用 MemorySaver
    """
    # 校验高德配置并获取本地工具
    amap.validate_config()
    amap_tools = amap.get_tools()
    if not amap_tools:
        print("⚠️ 警告：MapAgent 没有任何高德工具，地图/天气查询将不可用！")

    # 确保 MCP 服务已初始化
    await mcp_service.initialize_mcp()

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

    # 声明式基础专家列表
    subagents = [
        {
            "name": "KnowledgeAgent",
            "description": SUBAGENT_DESCRIPTIONS["KnowledgeAgent"],
            "system_prompt": KNOWLEDGE_AGENT_PROMPT,
            "tools": knowledge_tools.TOOLS,
            "model": models.vlm,
        },
        {
            "name": "MediaAgent",
            "description": SUBAGENT_DESCRIPTIONS["MediaAgent"],
            "system_prompt": MEDIA_AGENT_PROMPT,
            "tools": media_tools.TOOLS,
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
            "tools": code_tools.TOOLS,
            "model": models.worker_llm,
            "skills": [SKILLS_ROUTE],
        },
        {
            "name": "SQLAgent",
            "description": SUBAGENT_DESCRIPTIONS["SQLAgent"],
            "system_prompt": SQL_AGENT_PROMPT,
            "tools": sql_tools.TOOLS,
            "model": models.worker_llm,
        },
    ]

    # 本地操作类专家：FileAgent / DesktopAgent / BrowserAgent 始终挂载
    desktop_tools = mcp_service.get_tools_by_server("desktop")

    markitdown_tools = mcp_service.get_tools_by_server("markitdown")
    file_extra = markitdown_tool.build_tools(markitdown_tools, config.WORKSPACE_DIR)
    if not file_extra:
        print("⚠️ 警告：FileAgent 未获取到 convert_to_markdown 工具，请检查 uvx 与 markitdown-mcp 环境！")

    file_agent_runnable = file_agent.build_file_agent_runnable(
        llm=models.worker_llm,
        system_prompt=FILE_AGENT_PROMPT + (FILE_AGENT_MARKITDOWN_PROMPT if file_extra else ""),
        workspace_dir=config.WORKSPACE_DIR,
        extra_tools=file_extra,
    )
    file_subagent = CompiledSubAgent(
        name="FileAgent",
        description=SUBAGENT_DESCRIPTIONS["FileAgent"],
        runnable=file_agent_runnable,
    )
    desktop_agent_subgraph = build_desktop_agent_subgraph(
        desktop_tools=desktop_tools,
        system_prompt=DESKTOP_AGENT_PROMPT,
    )
    desktop_subagent = CompiledSubAgent(
        name="DesktopAgent",
        description="桌面操作专家。负责通过屏幕截图和网格坐标操作 Windows 桌面 GUI。只要任务涉及打开桌面文件夹、点击屏幕图标、双击打开本地文件或应用、选择打开方式窗口等任何桌面端 GUI 交互，必须直接派给它执行，严禁提前结束并让用户手动操作！",
        runnable=desktop_agent_subgraph,
    )

    subagents.append({
        "name": "BrowserAgent",
        "description": SUBAGENT_DESCRIPTIONS["BrowserAgent"],
        "system_prompt": BROWSER_AGENT_PROMPT,
        "tools": browser_tools.TOOLS,
        "model": models.worker_llm,
    })
    subagents.append(file_subagent)
    subagents.append(desktop_subagent)

    # 显式覆盖框架内置的 general-purpose subagent
    subagents.append({
        "name": "general-purpose",
        "description": "仅当用户请求明显不属于以上任何专家范围时才使用，绝大多数情况不应该被调用。",
        "system_prompt": (
            "如果你被调用，说明主 Agent 判断这个任务不属于任何专属领域专家的范围。"
            "请如实告知用户无法处理这类请求，不要尝试执行任何工具操作或猜测性回答。"
        ),
        "tools": [],
        "model": models.worker_llm,
    })

    # 断言除 FileAgent 外，任何声明式子 Agent 的工具列表都不得包含 convert_to_markdown
    for sa in subagents:
        if isinstance(sa, dict) and sa.get("name") != "FileAgent":
            tool_names = [getattr(t, "name", str(t)) for t in sa.get("tools", [])]
            assert "convert_to_markdown" not in tool_names, (
                f"子 Agent '{sa.get('name')}' 违规持有 convert_to_markdown 工具！"
            )

    if checkpointer is None:
        checkpointer = MemorySaver()

    agent = create_deep_agent(
        model=models.supervisor_llm,
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=subagents,
        middleware=middleware,
        backend=build_skills_backend(),
        checkpointer=checkpointer,
    )

    return agent

