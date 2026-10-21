"""系统全局服务与数据库流式初始化 (所在层: L5)"""

from app import config
from app.knowledge.knowledge_base import init_knowledge_base, process_incremental_knowledge
from app.knowledge.media.builder import init_av_graph_rag
from app.memory.store import init_memory_collection


def run_global_database_init():
    """一键运行系统所有数据库及知识库的流式初始化，供所有入口（控制台/微信/QQ）统一调用"""
    print("🚀 [Database] 开始执行全局多模态知识库联合初始化...")
    init_knowledge_base(config.SOURCE_DIR)
    process_incremental_knowledge()
    init_av_graph_rag()
    init_memory_collection()
    print("✅ [Database] 全局数据环境初始化就绪。")
