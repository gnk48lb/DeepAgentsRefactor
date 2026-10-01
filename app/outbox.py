import contextvars
from typing import Optional

_outbox: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("image_outbox", default=None)

def begin() -> list:
    """处理每条用户消息前调用，返回本轮的图片收集丞表。"""
    lst: list = []
    _outbox.set(lst)
    return lst

def push(item: dict) -> None:
    lst = _outbox.get()
    if lst is not None:
        lst.append(item)
