"""对话历史与长期记忆注入器 (所在层: L2)"""

import re
from typing import List, Optional
from langchain_core.messages import HumanMessage, BaseMessage
from app.memory import store as memory_store

# 长期记忆块标签常量
MEMORY_TAG_OPEN = "<long_term_memory>"
MEMORY_TAG_CLOSE = "</long_term_memory>"
_MEMORY_BLOCK_REGEX = re.compile(
    rf"{re.escape(MEMORY_TAG_OPEN)}\s*[\s\S]*?\s*{re.escape(MEMORY_TAG_CLOSE)}\s*",
    re.MULTILINE,
)


def strip_memory_block(text: str) -> str:
    """去除文本中由系统注入的 <long_term_memory> 块，还原用户原话。"""
    return _MEMORY_BLOCK_REGEX.sub("", text)


def build_initial_messages(
    user_query: str,
    history_messages: Optional[List[BaseMessage]] = None,
) -> List[BaseMessage]:
    """
    构造 deepagents 的初始消息列表。

    前缀静态原则（为了 Gemini 前缀缓存命中）：
    - 绝不在此产生任何 SystemMessage。若在历史中引入 SystemMessage，langchain-google-genai
      会在多轮中把中间的 SystemMessage extend 进 system_instruction，导致系统指令动态变长、
      请求前缀每轮改变，前缀缓存整段失效。
    - 检索到的长期记忆（top_k=1）固定以 <long_term_memory> 块拼在当轮 HumanMessage 开头；
      未检索到记忆时保持纯 user_query。
    """
    messages: List[BaseMessage] = []
    memories = memory_store.search_memory(user_query, top_k=1)
    if memories:
        query_content = f"{MEMORY_TAG_OPEN}\n" + "\n".join(memories) + f"\n{MEMORY_TAG_CLOSE}\n{user_query}"
    else:
        query_content = user_query

    if history_messages:
        messages.extend(history_messages)
    messages.append(HumanMessage(content=query_content, name="user"))
    return messages
