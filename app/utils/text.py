"""文本提取与处理工具 (L0)"""


def extract_text(content) -> str:
    """Extract plain text from a LangChain message content field.

    Handles three formats:
    - str: returned as-is
    - list[dict|str]: Gemini/OpenAI multimodal format, joins all type=="text" parts
    - None or other: converted to str (None -> "")
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                t = p.get("text", "")
                parts.append(t if isinstance(t, str) else str(t))
        return "".join(parts)
    return "" if content is None else str(content)
