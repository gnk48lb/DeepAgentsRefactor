"""文本提取与处理工具 (L0)"""
import re

URL_REGEX = re.compile(r"(?:https?://|www\.)[^\s<>\"'）)】」》，。；、！？]+", re.IGNORECASE)


def defang_urls(text: str) -> str:
    """把 URL 改写成不会被识别为链接、但人眼仍能看清的形式（https[:]//a[.]com/x）。
    用于授权提示：这类文本不能像普通回复那样把 URL 直接删掉，否则用户看不到自己在批准什么。"""
    if not text:
        return text
    return URL_REGEX.sub(lambda m: m.group(0).replace("://", "[:]//").replace(".", "[.]"), text)


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
