import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY")

# Workspace / Filesystem MCP 配置
# 默认为当前工作目录下的 workspace 文件夹，允许通过 .env 中的 WORKSPACE_DIR 覆盖
WORKSPACE_DIR = os.getenv(
    "WORKSPACE_DIR",
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "workspace")
)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# 设置环境变量供 Hugging Face 和 Tavily 等库自动识别
if HUGGINGFACEHUB_API_TOKEN:
    os.environ["HF_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Milvus Configuration
KNOWLEDGE_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "knowledge")
MEDIA_COLLECTION_NAME = os.getenv("MILVUS_MEDIA_COLLECTION_NAME", "av_works")
MEMORY_COLLECTION_NAME = os.getenv("MILVUS_MEMORY_COLLECTION_NAME", "user_memory")

# Milvus Local DB Path (Reads from MILVUS_DB_PATH to avoid environment variable naming conflict with PyMilvus legacy system)
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", "storage/gnk48_agent_collection.db")

# MySQL Configuration (Pre-configured for future Text-to-SQL integration)
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "8964")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "8964")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "my_database")

MYSQL_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Model Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 文字模型 LLM (Worker)
WORKER_LLM_MODEL = "gpt-4.1"

# 文字模型 LLM (Supervisor)
SUPERVISOR_LLM_MODEL = "gpt-4o"
# SUPERVISOR_LLM_MODEL = "gemini-3.1-flash-lite-preview"

# 多模态模型 VLM
# VLM_MODEL = "gpt-4.1"
VLM_MODEL = "gpt-4.1"

# 浏览器多模态模型 (BrowserAgent)
BROWSER_VLM_MODEL = "gemini-3.1-flash-lite"

# SiliconFlow 相关配置 (暂时注释)
SILICON_FLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# Gemini 相关配置
# 注意：Google Gemini 的 OpenAI 兼容接口需要在 v1beta 后加 /openai/
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6Ju8-FZijZ9dgjaUg6311SntPaKJpyjW-kNdeCYA0bSAA")

GITHUB_BASE_URL = "https://models.inference.ai.azure.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


# Advanced RAG Configurations
ENABLE_QUERY_TRANSFORM = False # 控制是否开启查询转换增强模式 (HyDE, Step-Back, Multi-Query)

# Docker Sandbox Configuration
DOCKER_IMAGE_NAME = "python-ds:latest"
DOCKER_MAX_MEM = "256m"

REFRESH_COLLECTION = False # 本次需设为 True 以重建 Schema
CLEAR_MEMORY_ON_STARTUP = False # 是否在启动时清空长期记忆集合
REFRESH_BM25 = False # 手动触发 BM25 模型的重新拟合与旧数据稀疏向量的局部更新
