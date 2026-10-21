"""知识库检索查询改写与路由策略 (所在层: L2)"""

import re
from langchain_core.messages import HumanMessage, SystemMessage
from app.infra import models
from app.utils.text import extract_text


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
        response = models.worker_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        strategy = extract_text(response.content).strip().lower()
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
        response = models.worker_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        return extract_text(response.content).strip()
    except Exception:
        return query


def generate_step_back(query: str) -> str:
    """
    生成一个更宏观的“退回一步”问题。
    """
    sys_prompt = """You are an expert at information retrieval. Given a specific user question, generate a more abstract, broader "step-back" question that helps understand the underlying concepts.
Do not include any introductory remarks. Just the question itself."""
    try:
        response = models.worker_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        return extract_text(response.content).strip()
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
        response = models.worker_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"Query: {query}")
        ])
        text_content = extract_text(response.content)
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        # Remove any leading numbers or dashes that the LLM might have added despite instructions
        clean_lines = []
        for line in lines:
            cleaned = re.sub(r'^[0-9\.\-\*]+\s*', '', line)
            if cleaned:
                clean_lines.append(cleaned)
        
        # We only want at most 3
        return clean_lines[:3]
    except Exception:
        return []
