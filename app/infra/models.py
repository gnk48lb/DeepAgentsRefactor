"""模型客户端惰性单例与校验管理 (L1)"""

from typing import Optional, List
import logging
import threading
import requests
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEndpointEmbeddings
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
                error_msg = data.get("error", data.get("message", "Unknown error"))
                raise Exception(f"SiliconFlow Embedding Failed (Possible Censorship?): {error_msg}\nFull Response: {data}")
            all_embeddings.extend([item["embedding"] for item in data["data"]])
            
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


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
        payload = {
            "model": self.model_name,
            "query": pairs[0][0],
            "documents": [p[1] for p in pairs],
            "top_n": len(pairs)
        }
        resp = requests.post(
            self.url,
            json=payload,
            headers=headers,
            timeout=SILICONFLOW_RERANK_TIMEOUT,
        )
        data = resp.json()
        if "results" not in data:
            logger.warning(
                f"[SiliconFlowReranker] API 未返回 results (状态码 {resp.status_code}): "
                f"{data.get('message', '未知错误')}"
            )
            return [0.0] * len(pairs)
        
        scores = [0.0] * len(pairs)
        for item in data["results"]:
            scores[item["index"]] = item["relevance_score"]
        return scores


# 默认宽松安全设置
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


def validate_required_keys():
    if not config.GEMINI_API_KEY:
        raise ValueError("未配置 GEMINI_API_KEY（或 GOOGLE_API_KEY）")
    if not config.SILICON_FLOW_API_KEY:
        raise ValueError("未配置 SILICON_FLOW_API_KEY（知识库/记忆的嵌入与重排依赖它）")


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


def _make_worker_llm():
    _require_gemini_key()
    return ChatGoogleGenerativeAI(
        model=config.WORKER_LLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def _make_orchestrator_llm():
    _require_gemini_key()
    return ChatGoogleGenerativeAI(
        model=config.ORCHESTRATOR_LLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def _make_browser_vlm():
    _require_gemini_key()
    return ChatGoogleGenerativeAI(
        model=config.BROWSER_VLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def _make_desktop_llm():
    _require_gemini_key()
    return ChatGoogleGenerativeAI(
        model=config.DESKTOP_VLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def _make_vlm():
    _require_gemini_key()
    return ChatGoogleGenerativeAI(
        model=config.VLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def _make_extractor_llm():
    _require_gemini_key()
    return ChatGoogleGenerativeAI(
        model=config.EXTRACTOR_LLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


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
    "browser_vlm": _make_browser_vlm,
    "desktop_llm": _make_desktop_llm,
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



