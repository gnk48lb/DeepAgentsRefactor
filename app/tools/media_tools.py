"""专有媒体库知识图谱检索工具 (所在层: L3)"""

import traceback
from langchain_core.tools import tool
from app.knowledge.media.retriever import get_media_retriever


@tool("av_graph_rag")
def av_graph_rag(query: str) -> str:
    """专门用于检索日本AV作品、番号、女优、男优、题材标签（如人妻、温泉、熟女等）相关信息的工具。当用户询问任何关于成人视频、特定番号作品或相关演员时，必须优先且仅能调用此工具。"""
    print(f"\n[Tool Call: av_graph_rag] Query: {query}")
    try:
        retriever = get_media_retriever()
        return retriever.search(query)
    except Exception as e:
        traceback.print_exc()
        return f"Error executing AV graph search: {e}"


TOOLS = [av_graph_rag]
