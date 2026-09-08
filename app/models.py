from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pymilvus.model.sparse import BM25EmbeddingFunction
import config
import logging

# 屏蔽无用的 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("jieba").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)

# --- Embedding & Reranking Models ---

# Initialize HuggingFace Dense Embeddings (API) for GraphRAG
from langchain_huggingface import HuggingFaceEndpointEmbeddings
av_embeddings = HuggingFaceEndpointEmbeddings(
    model=config.EMBEDDING_MODEL,
    huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN
)

# Initialize SiliconFlow Dense Embeddings (Custom class to avoid tiktoken SSL errors)
from langchain_core.embeddings import Embeddings
class SiliconFlowEmbeddings(BaseModel, Embeddings):
    model: str
    api_key: str
    base_url: str = config.SILICON_FLOW_BASE_URL
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": self.model, "input": batch}
            resp = requests.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
            data = resp.json()
            if "data" not in data or data["data"] is None:
                # 记录详细错误，判断是否被模型网关拦截 (内容审核)
                error_msg = data.get("error", data.get("message", "Unknown error"))
                raise Exception(f"SiliconFlow Embedding Failed (Possible Censorship?): {error_msg}\nFull Response: {data}")
            all_embeddings.extend([item["embedding"] for item in data["data"]])
            
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

embeddings = SiliconFlowEmbeddings(
    model=config.EMBEDDING_MODEL,
    api_key=config.SILICON_FLOW_API_KEY
)



# Initialize BM25 Sparse Embeddings (Local)
# Note: BM25 needs to be fitted with the corpus before calling encode()
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
analyzer = build_default_analyzer(language="zh")
bm25_ef = BM25EmbeddingFunction(analyzer=analyzer)
av_bm25_ef = BM25EmbeddingFunction(analyzer=analyzer)

# # Initialize Local Reranker (CPU)
# # from sentence_transformers import CrossEncoder
# # reranker = CrossEncoder(config.RERANKER_MODEL, device='cpu', max_length=512)

# Initialize SiliconFlow Reranker
import requests
class SiliconFlowReranker:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        self.url = f"{config.SILICON_FLOW_BASE_URL}/rerank"
        
    def predict(self, pairs):
        """兼容 CrossEncoder 的接口"""
        if not pairs: return []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # 假设 pairs 格式为 [[query, doc1], [query, doc2], ...]
        payload = {
            "model": self.model_name,
            "query": pairs[0][0],
            "documents": [p[1] for p in pairs],
            "top_n": len(pairs)
        }
        resp = requests.post(self.url, json=payload, headers=headers)
        data = resp.json()
        
        # SiliconFlow 返回格式: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        # 需要还原到原始 list 的顺序
        scores = [0.0] * len(pairs)
        for item in data.get("results", []):
            scores[item["index"]] = item["relevance_score"]
        return scores

reranker = SiliconFlowReranker(config.RERANKER_MODEL, config.SILICON_FLOW_API_KEY)


# Initialize Worker LLM (Text only)
worker_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=config.GEMINI_API_KEY,
    temperature=0
)

# worker_llm = ChatOpenAI(
#     model=config.WORKER_LLM_MODEL,
#     base_url=config.GITHUB_BASE_URL,
#     api_key=config.GITHUB_TOKEN,
# )

supervisor_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    # model="gemini-3.1-flash-lite-preview",
    google_api_key=config.GEMINI_API_KEY,
    temperature=0
)


# supervisor_llm = ChatOpenAI(
#     model=config.SUPERVISOR_LLM_MODEL,
#     base_url=config.GITHUB_BASE_URL,
#     api_key=config.GITHUB_TOKEN,
#     model_kwargs={"response_format": {"type": "json_object"}}
# )

# Initialize Supervisor LLM (Text only, must support Structured Output)

# Initialize Browser VLM
browser_vlm = ChatGoogleGenerativeAI(
    model=config.BROWSER_VLM_MODEL,
    google_api_key=config.GEMINI_API_KEY,
    temperature=0
)

# Initialize Desktop VLM
desktop_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=config.GEMINI_API_KEY,
    temperature=0
)

# Initialize VLM (Multimodal)
vlm = ChatOpenAI(
    model=config.VLM_MODEL,
    base_url=config.GITHUB_BASE_URL,
    api_key=config.GITHUB_TOKEN
)

# Initialize Extractor LLM (Memory Extraction)
extractor_llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=config.GITHUB_BASE_URL,
    api_key=config.GITHUB_TOKEN
)


# --- Pydantic Models for Structured Output & RAG ---

class Router(BaseModel):
    analysis: str = Field(description="反思空间：1.用户最初的问题是什么？ 2.目前的战报已收集到哪些信息？ 3.信息是否已经足以完美回答用户？ 4.如果不足，还缺什么？")
    next_agent: Literal["KnowledgeAgent", "MediaAgent", "MapAgent", "BrowserAgent", "FileAgent", "DesktopAgent", "CodeAgent", "SQLAgent", "FINISH"] = Field(description="选择下一个执行任务的专家，或者选择 FINISH 结束任务。")
    instruction_to_worker: Optional[str] = Field(description="如果指派给专家，请给出具体、明确的单步指令（例如：'请查询某某的经纬度'）。")
    final_answer: Optional[str] = Field(description="【极端重要】：这是用户唯一能看到的内容。如果任务已结束，你必须在此完整、详尽地复述专家为你找到的所有核心数据（如文件清单、知识点、代码结果）。绝对禁止只给结论不给数据！")

class ExtractedEntities(BaseModel):
    works: List[str] = Field(default_factory=list, description="Extracted work codes or titles")
    people: List[str] = Field(default_factory=list, description="Extracted actor or actress names")
    tags: List[str] = Field(default_factory=list, description="Extracted tag names")
    target_person_type: Optional[str] = Field(None, description="The specific type of person the user is looking for (e.g., 'Actress' or 'Actor'). If not specified, leave as null.")

class QueryIntent(BaseModel):
    search_mode: str = Field(..., description="Either 'vector_only' or 'hybrid_graph'")
    extracted_entities: ExtractedEntities
