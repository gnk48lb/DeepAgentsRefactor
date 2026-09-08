from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from . import models
from .models import QueryIntent, ExtractedEntities
import logging

logger = logging.getLogger(__name__)

def route_query(query: str) -> str:
    """
    使用 LLM 决定该查询最适合哪种转换策略。
    返回: 'hyde', 'step_back', 'multi_query', 或 'none'
    """
    sys_prompt = """You are an expert query router for a RAG system. Given the user query, classify it into EXACTLY ONE of the following strategies:
- "hyde": The query is looking for specific facts, explanations, or domain knowledge where hallucinating a hypothetical answer might contain useful keywords. (e.g. "What are the effects of Anubis's W skill?")
- "step_back": The query asks about a highly specific or complex scenario that relies on broader underlying principles or mechanics. (e.g. "Why does my character take damage when passing through the wall?")
- "multi_query": The query is vague, ambiguous, or could be phrased in multiple different ways. (e.g. "Anubis changes")
- "none": The query is simple, direct, and clear enough that it doesn't need transformation.

Return ONLY the strategy name (hyde, step_back, multi_query, or none). Do not output any other text.
"""
    try:
        response = models.llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        strategy = response.content.strip().lower()
        if strategy not in ["hyde", "step_back", "multi_query", "none"]:
            return "none"
        return strategy
    except Exception as e:
        print(f"[Query Router Error] {e}")
        return "none"

def generate_hyde(query: str) -> str:
    """
    生成一个假设性答案 (HyDE)。
    """
    sys_prompt = """Please write a hypothetical passage that answers the following query. 
The passage should sound like it came from a game wiki or official patch notes.
Do not include any introductory or concluding remarks, just the hypothetical answer."""
    try:
        response = models.llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        return response.content.strip()
    except Exception:
        return query

def generate_step_back(query: str) -> str:
    """
    生成一个更宏观的“退回一步”问题。
    """
    sys_prompt = """You are an expert at information retrieval. Given a specific user question, generate a more abstract, broader "step-back" question that helps understand the underlying concepts.
Do not include any introductory remarks. Just the question itself."""
    try:
        response = models.llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        return response.content.strip()
    except Exception:
        return query

def generate_multi_query(query: str) -> list[str]:
    """
    生成 3 个多视角的相似查询。
    """
    sys_prompt = """You are an expert at query expansion. Given a user query, generate 3 different alternative ways to phrase the same question in Chinese.
These alternative queries should capture different keywords or perspectives.
Output exactly 3 lines, each line containing one query. Do not include numbering, bullet points, or introductory text."""
    try:
        response = models.llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        lines = [line.strip() for line in response.content.split('\n') if line.strip()]
        # Remove any leading numbers or dashes that the LLM might have added despite instructions
        clean_lines = []
        for line in lines:
            import re
            cleaned = re.sub(r'^[0-9\.\-\*]+\s*', '', line)
            if cleaned:
                clean_lines.append(cleaned)
        
        # We only want at most 3
        return clean_lines[:3]
    except Exception:
        return []

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
