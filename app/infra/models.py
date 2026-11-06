"""模型客户端惰性单例与校验管理 (L1)"""

from typing import List
import logging
import threading
import requests
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from pymilvus.model.sparse import BM25EmbeddingFunction
from langchain_core.embeddings import Embeddings
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from app import config

# 屏蔽无用的 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("jieba").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("langchain_anthropic").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Embedding & Reranking Models ---

SILICONFLOW_EMBEDDING_TIMEOUT = 60
SILICONFLOW_RERANK_TIMEOUT = 30


# Initialize Cloudflare Workers AI Dense Embeddings (Serverless API) for GraphRAG / AV Search
class CloudflareWorkersAIEmbeddings(BaseModel, Embeddings):
    account_id: str
    api_token: str
    model: str = config.CLOUDFLARE_EMBEDDING_MODEL
    base_url: str = "https://api.cloudflare.com/client/v4/accounts"
    timeout: int = 60

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.account_id:
            raise ValueError(
                "未配置 CLOUDFLARE_ACCOUNT_ID！"
                "请登录 Cloudflare 控制台 (dash.cloudflare.com -> Workers 和 Pages) 获取 32 位的 Account ID，"
                "并在 .env 文件中填入 CLOUDFLARE_ACCOUNT_ID"
            )
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ") for t in texts[i : i + batch_size]]
            url = f"{self.base_url}/{self.account_id}/ai/run/{self.model}"
            resp = requests.post(url, headers=headers, json={"text": batch}, timeout=self.timeout)
            data = resp.json()
            if not data.get("success") or "result" not in data or "data" not in data["result"]:
                raise Exception(f"Cloudflare Workers AI Embedding Failed: {data}")
            all_embeddings.extend(data["result"]["data"])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# Initialize SiliconFlow Dense Embeddings (Custom class to avoid tiktoken SSL errors)
class SiliconFlowEmbeddings(BaseModel, Embeddings):
    model: str
    api_key: str
    base_url: str = config.SILICON_FLOW_BASE_URL
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": self.model, "input": batch}
            resp = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=SILICONFLOW_EMBEDDING_TIMEOUT,
            )
            data = resp.json()
            if "data" not in data or data["data"] is None:
                # 记录详细错误，判断是否被模型网关拦截 (内容审核)
                error_msg = data.get("error", data.get("message", "Unknown error"))
                raise Exception(f"SiliconFlow Embedding Failed (Possible Censorship?): {error_msg}\nFull Response: {data}")
            all_embeddings.extend([item["embedding"] for item in data["data"]])
            
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class RerankerUnavailable(Exception):
    """当重排模型 (Reranker) 服务不可用、网络异常或响应非预期时抛出。"""
    pass


# Initialize SiliconFlow Reranker
class SiliconFlowReranker:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        self.url = f"{config.SILICON_FLOW_BASE_URL}/rerank"
        
    def predict(self, pairs):
        """兼容 CrossEncoder 的接口"""
        if not pairs:
            return []
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
        try:
            resp = requests.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=SILICONFLOW_RERANK_TIMEOUT,
            )
            if resp.status_code != 200:
                raise RerankerUnavailable(
                    f"[SiliconFlowReranker] HTTP {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            if "results" not in data:
                raise RerankerUnavailable(
                    f"[SiliconFlowReranker] API 未返回 results (状态码 {resp.status_code}): "
                    f"{data.get('message', '未知错误')}"
                )
        except requests.RequestException as e:
            raise RerankerUnavailable(f"[SiliconFlowReranker] 网络请求异常: {e}") from e
        except Exception as e:
            if isinstance(e, RerankerUnavailable):
                raise
            raise RerankerUnavailable(f"[SiliconFlowReranker] 响应解析异常: {e}") from e
        
        # SiliconFlow 返回格式: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        # 需要还原到原始 list 的顺序
        scores = [0.0] * len(pairs)
        for item in data["results"]:
            scores[item["index"]] = item["relevance_score"]
        return scores


# 默认宽松安全设置（放开成人影视领域涉及的关键词，防止触发静默拦截或 0 token 阻断）
DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_NONE,
}


def _require_gemini_key():
    if not config.GEMINI_API_KEY:
        raise ValueError("未配置 GEMINI_API_KEY（或 GOOGLE_API_KEY）")


_MIGRATED_ROLE_CONFIG = {
    "WORKER_LLM": (config.WORKER_LLM_PROVIDER, config.WORKER_LLM_MODEL),
    "ORCHESTRATOR_LLM": (config.ORCHESTRATOR_LLM_PROVIDER, config.ORCHESTRATOR_LLM_MODEL),
    "KNOWLEDGE_LLM": (config.KNOWLEDGE_LLM_PROVIDER, config.KNOWLEDGE_LLM_MODEL),
    "INGESTION_VLM": (config.INGESTION_VLM_PROVIDER, config.INGESTION_VLM_MODEL),
    "BROWSER_VLM": (config.BROWSER_VLM_PROVIDER, config.BROWSER_VLM_MODEL),
    "DESKTOP_VLM": (config.DESKTOP_VLM_PROVIDER, config.DESKTOP_VLM_MODEL),
}


def _validate_role_config(role: str, provider: str, model_name: str):
    if provider not in ("gemini", "magpie"):
        raise ValueError(f"{role}_PROVIDER 仅支持 gemini 或 magpie，当前为 {provider!r}")
    if not model_name.strip():
        raise ValueError(f"{role}_MODEL 不能为空")
    if provider == "magpie":
        if not config.MAGPIE_BASE_URL.strip():
            raise ValueError("使用 Magpie 时 MAGPIE_BASE_URL 不能为空")
        if not config.MAGPIE_API_KEY.strip():
            raise ValueError("使用 Magpie 时 MAGPIE_API_KEY 不能为空（本地网关可填 local）")


def validate_required_keys():
    # 原实现（保留用于迁移对照）：
    # if not config.GEMINI_API_KEY:
    #     raise ValueError("未配置 GEMINI_API_KEY（或 GOOGLE_API_KEY）")
    # if not config.SILICON_FLOW_API_KEY:
    #     raise ValueError("未配置 SILICON_FLOW_API_KEY（知识库/记忆的嵌入与重排依赖它）")
    if not config.GEMINI_API_KEY:
        raise ValueError("未配置 GEMINI_API_KEY（或 GOOGLE_API_KEY）；轻量工具、聊天 VLM 与记忆提取仍依赖它")
    if not config.SILICON_FLOW_API_KEY:
        raise ValueError("未配置 SILICON_FLOW_API_KEY（知识库/记忆的嵌入与重排依赖它）")
    for role, (provider, model_name) in _MIGRATED_ROLE_CONFIG.items():
        _validate_role_config(role, provider, model_name)


_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = build_default_analyzer(language="zh")
    return _analyzer


def _make_av_embeddings():
    if config.CLOUDFLARE_API_TOKEN:
        return CloudflareWorkersAIEmbeddings(
            account_id=config.CLOUDFLARE_ACCOUNT_ID or "",
            api_token=config.CLOUDFLARE_API_TOKEN,
            model=config.CLOUDFLARE_EMBEDDING_MODEL,
        )
    else:
        return HuggingFaceEndpointEmbeddings(
            model=config.EMBEDDING_MODEL,
            huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN,
        )


def _make_embeddings():
    return SiliconFlowEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.SILICON_FLOW_API_KEY or "",
    )


def _make_bm25_ef():
    return BM25EmbeddingFunction(analyzer=_get_analyzer())


def _make_av_bm25_ef():
    return BM25EmbeddingFunction(analyzer=_get_analyzer())


def _make_reranker():
    return SiliconFlowReranker(config.RERANKER_MODEL, config.SILICON_FLOW_API_KEY or "")


def _make_gemini_chat(model_name: str):
    _require_gemini_key()
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=config.GEMINI_API_KEY,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def _make_magpie_chat(model_name: str):
    return ChatAnthropic(
        model=model_name,
        base_url=config.MAGPIE_BASE_URL,
        api_key=config.MAGPIE_API_KEY,
        max_tokens=config.MAGPIE_MAX_TOKENS,
        timeout=config.MAGPIE_TIMEOUT_SECONDS,
        max_retries=config.MAGPIE_MAX_RETRIES,
    )


def _make_role_chat(role: str):
    provider, model_name = _MIGRATED_ROLE_CONFIG[role]
    _validate_role_config(role, provider, model_name)
    if provider == "magpie":
        return _make_magpie_chat(model_name)
    return _make_gemini_chat(model_name)


def _make_worker_llm():
    # Gemini 原实现（保留用于回退对照）：
    # _require_gemini_key()
    # return ChatGoogleGenerativeAI(
    #     model=config.WORKER_LLM_MODEL,
    #     google_api_key=config.GEMINI_API_KEY,
    #     safety_settings=DEFAULT_SAFETY_SETTINGS,
    # )
    return _make_role_chat("WORKER_LLM")


def _make_orchestrator_llm():
    # Gemini 原实现（保留用于回退对照）：
    # _require_gemini_key()
    # return ChatGoogleGenerativeAI(
    #     model=config.ORCHESTRATOR_LLM_MODEL,
    #     google_api_key=config.GEMINI_API_KEY,
    #     safety_settings=DEFAULT_SAFETY_SETTINGS,
    # )
    return _make_role_chat("ORCHESTRATOR_LLM")


def _make_desktop_llm():
    # Gemini 原实现（保留用于回退对照）：
    # _require_gemini_key()
    # return ChatGoogleGenerativeAI(
    #     model=config.DESKTOP_VLM_MODEL,
    #     google_api_key=config.GEMINI_API_KEY,
    #     safety_settings=DEFAULT_SAFETY_SETTINGS,
    # )
    return _make_role_chat("DESKTOP_VLM")


def _make_knowledge_llm():
    return _make_role_chat("KNOWLEDGE_LLM")


def _make_ingestion_vlm():
    return _make_role_chat("INGESTION_VLM")


def _make_utility_llm():
    return _make_gemini_chat(config.UTILITY_LLM_MODEL)


def _make_vlm():
    return _make_gemini_chat(config.VLM_MODEL)


def _make_extractor_llm():
    return _make_gemini_chat(config.EXTRACTOR_LLM_MODEL)


_lock = threading.RLock()
_cache: dict = {}
_FACTORIES = {
    "av_embeddings": _make_av_embeddings,
    "embeddings": _make_embeddings,
    "bm25_ef": _make_bm25_ef,
    "av_bm25_ef": _make_av_bm25_ef,
    "reranker": _make_reranker,
    "worker_llm": _make_worker_llm,
    "orchestrator_llm": _make_orchestrator_llm,
    "desktop_llm": _make_desktop_llm,
    "knowledge_llm": _make_knowledge_llm,
    "ingestion_vlm": _make_ingestion_vlm,
    "utility_llm": _make_utility_llm,
    "vlm": _make_vlm,
    "extractor_llm": _make_extractor_llm,
}


def __getattr__(name):
    f = _FACTORIES.get(name)
    if f is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    with _lock:
        if name not in _cache:
            _cache[name] = f()
        return _cache[name]



