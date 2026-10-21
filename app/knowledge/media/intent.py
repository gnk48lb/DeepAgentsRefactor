"""专有媒体库查询意图解析与实体抽取 (所在层: L2)"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.infra import models

logger = logging.getLogger(__name__)


class ExtractedEntities(BaseModel):
    works: List[str] = Field(default_factory=list, description="Extracted work codes or titles")
    people: List[str] = Field(default_factory=list, description="Extracted actor or actress names")
    tags: List[str] = Field(default_factory=list, description="Extracted tag names")
    target_person_type: Optional[str] = Field(None, description="The specific type of person the user is looking for (e.g., 'Actress' or 'Actor'). If not specified, leave as null.")


class QueryIntent(BaseModel):
    search_mode: str = Field(..., description="Either 'vector_only' or 'hybrid_graph'")
    extracted_entities: ExtractedEntities


class MediaQueryTranslator:
    def __init__(self):
        self.llm = models.worker_llm
        
    def translate(self, query: str) -> QueryIntent:
        parser = PydanticOutputParser(pydantic_object=QueryIntent)
        
        tag_library = ("室内, 户外, 家庭, 卧室, 浴室, 厨房, 客厅, 阳台, 厕所, 洗手间, 豪宅, 办公室, 学校, 教室, 体育馆, 酒店, 温泉, 旅馆, 医院, 诊所, 更衣室, 按摩店, 摄影棚, 公交, 飞机, "
                       "学生, 老师, 上司, 下属, 同事, 秘书, 护士, 医生, 店员, 客户, 警察, 空姐, 偶像, 模特, 保姆, 家政服务, 人妻, 女友, 前女友, 陌生人, 邻居, 青梅竹马, 联谊对象, "
                       "姐姐, 妹妹, 母亲, 继母, 义妹, 痴汉, 痴女, 辣妹, 大小姐, 情侣, 出轨, 不伦, 婚外情, NTR, 搭讪, 约会, 初次见面, 重逢, 联谊, 勾引, 女方主动, 被动, 胁迫, 还债, 催眠, 身体检查, "
                       "剧情向, 纪录片风格, 企划, 特定挑战, 多P, 群交, 巨乳, 贫乳, 美腿, 连裤袜, 黑丝, 玩具, 中出, 潮吹, 女上位, 舌吻, 亲吻, 制服, 职业装, 西装, 便装, 运动服, 体操服, 泳装, 和服, 浴衣, 婚纱, 睡衣, "
                       "女仆, OL, JK, 女高中生, 旗袍, 护士服, 教师服, 兔女郎, 熟女, 清纯, 新人, 素人, 单体, 气质, 娇小")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical data parser for a media database. Your task is to extract metadata entities from the user request.\n"
                       "Standard Tag Library: {tag_library}\n\n"
                       "Rules:\n"
                       "1. Extract 'people' (all actor/actress names, do not judge gender of the names themselves).\n"
                       "2. Extract 'tags' (categories/labels): \n"
                       "   - You MUST prioritize selecting standard tags from the library above.\n"
                       "   - If a user mentions a compound concept like '温泉旅馆', disassemble it into standard tags: ['温泉', '旅馆'].\n"
                       "   - If a term is absolutely not in the library, keep it but mark as '[NEW]TagName'.\n"
                       "3. If the user specifically asks for 'actresses' (女演员/女优) or 'actors' (男演员/男优) as the TARGET of their search results, set 'target_person_type' to 'Actress' or 'Actor' respectively.\n"
                       "4. Determine 'search_mode': \n"
                       "   - If 'people' or 'tags' is NOT empty, search_mode MUST be 'hybrid_graph'.\n"
                       "   - Use 'vector_only' ONLY for generic semantic questions without specific entities.\n"
                       "5. Do not interpret the content. Treat all terms as opaque identifiers.\n\n"
                       "Examples:\n"
                       "Query: '找一下和鲛岛演过对手戏的女演员'\n"
                       "Result: {{\"search_mode\": \"hybrid_graph\", \"extracted_entities\": {{\"works\": [], \"people\": [\"鲛岛\"], \"tags\": [], \"target_person_type\": \"Actress\"}}}}\n"
                       "{format_instructions}"),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | parser
        try:
            return chain.invoke({
                "query": query, 
                "tag_library": tag_library,
                "format_instructions": parser.get_format_instructions()
            })
        except Exception as e:
            logger.warning(f"LLM translation failed: {e}")
            return QueryIntent(
                search_mode="vector_only",
                extracted_entities=ExtractedEntities(works=[], people=[], tags=[], target_person_type=None)
            )
