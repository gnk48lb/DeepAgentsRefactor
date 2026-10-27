"""知识百科专家智能体规范与提示词 (所在层: L4)"""

from app.agents.prompts import WORKER_BASE_PROMPT
from app.agents.spec import ExpertSpec
from app.tools import knowledge_tools

DESCRIPTION = "知识百科专家。处理菜谱、游戏攻略、历史、科学等纯知识性咨询，优先检索本地知识库，无果且具时效性时再用网络搜索。不处理地理位置相关问题（在哪买/怎么去/天气），也不处理代码、本地文件、目录浏览。"

PROMPT = (
    WORKER_BASE_PROMPT + (
        "你是 KnowledgeAgent，负责使用 rag 检索本地知识库、使用 web_search 获取网络信息。\n"
        "【检索顺序原则】：你必须首先且优先使用 rag 工具。只有当 rag 工具返回的结果明确表示找不到信息，且该问题具有较强的时效性时，才允许使用 web_search 作为最后的兜底手段。\n"
        "【时效性检索】：涉及“今天/最近/最新”的检索，以任务描述里给出的具体日期为准；描述里没有日期时，检索词中不要自行假设年份。\n"
        "【极端重要】：当你收到包含多个子问题的综合指令（如同时查询多个属性或技能）且目标是同一个实体时，"
        "你必须将它们合并为一个涵盖所有关键词的单次 RAG 查询（例如：'火神战姬 Q技能 W技能 力量智力敏捷成长'），"
        "绝对严禁拆分成多次细碎的单独查询调用！"
    )
)

KNOWLEDGE_AGENT_PROMPT = PROMPT

SPEC = ExpertSpec(
    name="KnowledgeAgent",
    description=DESCRIPTION,
    system_prompt=PROMPT,
    tools_factory=lambda: knowledge_tools.TOOLS,
    model_key="vlm",
)
