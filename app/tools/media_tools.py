"""专有媒体库知识图谱检索工具 (所在层: L3)"""

import traceback
from langchain_core.tools import tool
from app.knowledge.media.retriever import get_media_retriever


@tool("av_graph_rag")
def av_graph_rag(query: str) -> str:
    """检索日本AV作品、番号、女优/男优合作关系网络（共同出演、搭档）和剧情题材标签（如人妻、温泉、熟女等）的知识图谱工具。只返回作品番号、标题、参演演员与题材标签，不含女优的身高、罩杯、出生年份等身体数值；这类精确属性过滤和数量统计请改用 execute_sql 查数据库。"""
    print(f"\n[Tool Call: av_graph_rag] Query: {query}")
    try:
        retriever = get_media_retriever()
        return retriever.search(query)
    except Exception as e:
        traceback.print_exc()
        return f"Error executing AV graph search: {e}"


TOOLS = [av_graph_rag]
