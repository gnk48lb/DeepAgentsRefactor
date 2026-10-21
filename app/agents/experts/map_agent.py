"""地理出行高德专家智能体规范与提示词 (所在层: L4)"""

from app.agents.prompts import WORKER_BASE_PROMPT
from app.agents.spec import ExpertSpec
from app.tools import amap

DESCRIPTION = "地理出行专家，挂载高德地图工具。处理地理编码、路线规划（步行/骑行/驾车）、周边搜索（POI）、天气查询。"

PROMPT = (
    WORKER_BASE_PROMPT
    + "你是 MapAgent，负责调用高德地图服务工具。必须先获取经纬度再进行周边/坐标类搜索。\n"
    + "只报告工具实际返回的字段。工具没有返回距离就不要写距离，更不要给不同地点填同一个距离；确实需要距离时，对每个地点分别调用距离/路线工具。"
)

MAP_AGENT_PROMPT = PROMPT

SPEC = ExpertSpec(
    name="MapAgent",
    description=DESCRIPTION,
    system_prompt=PROMPT,
    tools_factory=lambda: amap.get_tools(),
    model_key="worker_llm",
)
