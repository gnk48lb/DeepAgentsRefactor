import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_EMBEDDING_MODEL = os.getenv("CLOUDFLARE_EMBEDDING_MODEL") or "@cf/baai/bge-m3"

# Workspace / Filesystem MCP 配置
# 默认为当前工作目录下的 workspace 文件夹，允许通过 .env 中的 WORKSPACE_DIR 覆盖
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR") or os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "workspace"
)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# 设置环境变量供 Hugging Face 和 Tavily 等库自动识别
if HUGGINGFACEHUB_API_TOKEN:
    os.environ["HF_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Milvus Configuration
KNOWLEDGE_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME") or "knowledge"
MEDIA_COLLECTION_NAME = os.getenv("MILVUS_MEDIA_COLLECTION_NAME") or "av_works"
MEMORY_COLLECTION_NAME = os.getenv("MILVUS_MEMORY_COLLECTION_NAME") or "user_memory"

# Milvus Local DB Path (Reads from MILVUS_DB_PATH to avoid environment variable naming conflict with PyMilvus legacy system)
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH") or "storage/gnk48_agent_collection.db"

# MySQL Configuration (Pre-configured for future Text-to-SQL integration)
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

# 文字模型 LLM (Worker)
WORKER_LLM_MODEL = "gemini-3.5-flash-lite"

# 文字模型 LLM (Supervisor)
SUPERVISOR_LLM_MODEL = "gemini-3.5-flash-lite"

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
ENABLE_QUERY_TRANSFORM = False # 控制是否开启查询转换增强模式 (HyDE, Step-Back, Multi-Query)

# Docker Sandbox Configuration
DOCKER_IMAGE_NAME = "python-ds:latest"
DOCKER_MAX_MEM = "256m"

REFRESH_COLLECTION = False # 本次需设为 True 以重建 Schema
CLEAR_MEMORY_ON_STARTUP = False # 是否在启动时清空长期记忆集合
REFRESH_BM25 = False # 手动触发 BM25 模型的重新拟合与旧数据稀疏向量的局部更新

# 权限配置：主人 QQ 号列表（英文逗号分隔）
OWNER_QQ_IDS = {x.strip() for x in (os.getenv("OWNER_QQ_IDS") or "").split(",") if x.strip()}

# QQ 回复最大字符数限制
# 依据：官方未公布字符上限；实测 API 基本不拦，移动客户端折叠阈值最低约 15000（如 HarmonyOS 16000 起折叠），取 10000 留安全余量
QQ_MAX_REPLY_CHARS = 10000

# Agent 执行 recursion_limit（递归深度/超步限制）
# 主图每次委派约占 3 个 superstep，子图 llm<->tools 工具调用各占 2 个 superstep；60 允许约 18 次委派或约 29 次子图工具调用
AGENT_RECURSION_LIMIT = 60

# 单轮 Agent 运行总时限（秒，含工具与子 Agent 调用）
# 依据：防止外部接口卡死导致用户会话锁永不释放；人工授权等待不计入（interrupt 会让 ainvoke 立即返回）
# 若主要在群聊使用，QQ 群被动回复窗口通常为 5 分钟，可调整为 280 左右
AGENT_TURN_TIMEOUT_SECONDS = 600

# MCP 单次工具调用超时（秒）
# 若 MCP 服务进程卡死未返回，超过此时间将触发服务重启回收并报错
MCP_CALL_TIMEOUT_SECONDS = 60

# MCP 服务启动/连接等待超时（秒）
# 各服务启动时拉取工具的最大允许等待时间，超时按启动失败处理
MCP_STARTUP_TIMEOUT_SECONDS = 180

# 网页抓取与 B 站个人主页抓取文本上限
# 依据：deepagents 中间件会对超过约 8 万字符的工具结果落盘到 state['files'] 导致 checkpoint 膨胀
WEBPAGE_MAX_CHARS = 30000
BILI_PROFILE_MAX_CHARS = 8000

