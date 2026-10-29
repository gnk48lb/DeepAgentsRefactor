"""全局配置模块，负责加载环境变量与解析路径常量 (L0)"""

import os
from dotenv import load_dotenv

# Project Root (absolute path string)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(value: str) -> str:
    """含 :// 或已是绝对路径 -> 原样返回，否则相对 PROJECT_ROOT 规范化"""
    if not value:
        return value
    if "://" in value or os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(PROJECT_ROOT, value))


# Load environment variables from .env file located at PROJECT_ROOT
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# API Configuration
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_EMBEDDING_MODEL = os.getenv("CLOUDFLARE_EMBEDDING_MODEL") or "@cf/baai/bge-m3"
JINA_API_KEY = os.getenv("JINA_API_KEY")

# CLI Logger / Debug Configuration
CLI_DEBUG = os.getenv("CLI_DEBUG", "1")
CLI_DEBUG_MAX_CHARS = os.getenv("CLI_DEBUG_MAX_CHARS", "600")

# Workspace / Filesystem MCP 配置
WORKSPACE_DIR = _resolve_path(
    os.getenv("WORKSPACE_DIR") or os.path.join(PROJECT_ROOT, "workspace")
)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# 路径常量（全部绝对路径 str）
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "source")
PENDING_DIR = os.path.join(PROJECT_ROOT, "data", "pending")
PIPELINE_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "pipeline", "images")
PIPELINE_SUMMARIES_DIR = os.path.join(PROJECT_ROOT, "data", "pipeline", "summaries")
MEDIA_SOURCE_FILE = os.path.join(PROJECT_ROOT, "data", "graphrag", "work.jsonl")
BM25_MODEL_PATH = os.path.join(PROJECT_ROOT, "storage", "bm25_model.json")
AV_BM25_MODEL_PATH = os.path.join(PROJECT_ROOT, "storage", "av_bm25_model.json")
BROWSER_PROFILE_DIR = os.path.join(PROJECT_ROOT, "storage", "browser_profile")
OUTBOX_DEBUG_DIR = os.path.join(PROJECT_ROOT, "storage", "outbox_debug")
SANDBOX_TMP_DIR = os.path.join(STORAGE_DIR, "sandbox_tmp")

# 设置环境变量供 Hugging Face 和 Tavily 等库自动识别
if HUGGINGFACEHUB_API_TOKEN:
    os.environ["HF_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Milvus Configuration
KNOWLEDGE_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME") or "knowledge"
MEDIA_COLLECTION_NAME = os.getenv("MILVUS_MEDIA_COLLECTION_NAME") or "av_works"
MEMORY_COLLECTION_NAME = os.getenv("MILVUS_MEMORY_COLLECTION_NAME") or "user_memory"

# Milvus Local DB Path
MILVUS_DB_PATH = _resolve_path(
    os.getenv("MILVUS_DB_PATH") or "storage/gnk48_agent_collection.db"
)

# 长期记忆配置 (Schema v2 两阶段版本化记忆)
MEMORY_SCHEMA_VERSION = 2
MEMORY_MIN_SCORE = 0.45
MEMORY_TOP_K = 3  # 每轮最多注入的长期记忆条数
MEMORY_RECALL_TOP_K = 5  # 阶段2对账候选召回 top-k
MEMORY_TEXT_MAX_CHARS = 200  # 单条记忆最大字符数
MEMORY_EXTRACT_MAX_USER_CHARS = 800  # 用户单轮原话提取上限，超过视为粘贴文本不提取
MEMORY_MAX_STATEMENTS_PER_TURN = 4  # 单轮最多提取并处理的陈述条数

# MySQL Configuration
MYSQL_HOST = os.getenv("MYSQL_HOST") or "localhost"
MYSQL_PORT = os.getenv("MYSQL_PORT") or "8964"
MYSQL_USER = os.getenv("MYSQL_USER") or "root"
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD") or "8964"
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE") or "av_db"

MYSQL_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER") or "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or "password"

# Model Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RAG_MIN_RERANK_SCORE = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.1"))

# 文字模型 LLM (Worker)
WORKER_LLM_MODEL = "gemini-3.5-flash-lite"

# 文字模型 LLM (Orchestrator)
ORCHESTRATOR_LLM_MODEL = "gemini-3.5-flash-lite"

# 多模态模型 VLM
VLM_MODEL = "gemini-3.5-flash-lite"

# 浏览器多模态模型 (BrowserAgent)
BROWSER_VLM_MODEL = "gemini-3.5-flash-lite"

# 桌面多模态模型 (DesktopAgent)
DESKTOP_VLM_MODEL = "gemini-3.5-flash-lite"

# 记忆提取模型 (Memory Extraction)
EXTRACTOR_LLM_MODEL = os.getenv("EXTRACTOR_LLM_MODEL") or "gemini-3.1-flash-lite"

# SiliconFlow 相关配置 (暂时注释)
SILICON_FLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# Gemini 相关配置
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Advanced RAG Configurations
ENABLE_QUERY_TRANSFORM = False

# Docker Sandbox Configuration
DOCKER_IMAGE_NAME = "python-ds:latest"
DOCKER_MAX_MEM = "256m"
DOCKER_PIDS_LIMIT = int(os.getenv("DOCKER_PIDS_LIMIT", "128"))
DOCKER_NANO_CPUS = int(os.getenv("DOCKER_NANO_CPUS", "1000000000"))
SANDBOX_MAX_IMAGES = int(os.getenv("SANDBOX_MAX_IMAGES", "4"))

REFRESH_COLLECTION = False
CLEAR_MEMORY_ON_STARTUP = False
REFRESH_BM25 = False

# QQ 回复最大字符数限制
QQ_MAX_REPLY_CHARS = 10000

# Agent 执行 recursion_limit
AGENT_RECURSION_LIMIT = 60

# 专家独立停止条件：每个专家在“一次 task 委派”内最多可调用的工具次数（实现见 app/agents/budget.py）。
# 第 N+1 次起的工具调用不再执行（软停止，提示模型立即收尾汇报）；模型仍不收手则在第 N+2 次模型调用后强制结束（硬停止）。
# 这些是“止损阈值”而不是常规用量：按各专家真实流程留足余量，调参只改这里。
# 约束：DesktopAgent 是手写子图，没有自己的 recursion_limit，继承 AGENT_RECURSION_LIMIT，
# 且每个工具步骤占 2 个 superstep，所以必须满足 2 * N + 4 <= AGENT_RECURSION_LIMIT（tests/smoke_budget.py 会校验）。
SUBAGENT_TOOL_LIMIT_DEFAULT = 8
SUBAGENT_TOOL_LIMITS = {
    "KnowledgeAgent": 5,
    "MediaAgent": 10,
    "MapAgent": 12,
    "CodeAgent": 8,
    "BrowserAgent": 6,
    "FileAgent": 20,
    "DesktopAgent": 20,
}

# 单轮 Agent 运行总时限（秒）
AGENT_TURN_TIMEOUT_SECONDS = 600

# MCP 单次工具调用超时（秒）
MCP_CALL_TIMEOUT_SECONDS = 60

# MarkItDown MCP 配置
MARKITDOWN_ALLOW_HTTP = os.getenv("MARKITDOWN_ALLOW_HTTP", "false").lower() in ("1", "true", "yes")
MARKITDOWN_MAX_FILE_MB = 50
MCP_CALL_TIMEOUT_SECONDS_BY_SERVER = {"markitdown": 180}
MCP_STARTUP_TIMEOUT_SECONDS = 180

# 网页抓取与 B 站个人主页抓取文本上限
WEBPAGE_MAX_CHARS = 30000
BILI_PROFILE_MAX_CHARS = 8000
