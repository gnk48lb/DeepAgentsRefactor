import contextvars
from typing import Optional

_outbox: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("image_outbox", default=None)

def begin() -> list:
    """处理每条用户消息前调用，返回本轮的图片收集列表。"""
    lst: list = []
    _outbox.set(lst)
    return lst

def push(item: dict) -> None:
    """没有调用过 begin() 的上下文（旧入口、单元测试）里是空操作。"""
    lst = _outbox.get()
    if lst is not None:
        lst.append(item)

def resume(lst: list) -> None:
    """同一条用户请求跨越 HITL 中断时，要把之前的发件箱重新绑定回当前上下文。"""
    _outbox.set(lst)

def current() -> list:
    """返回当前上下文的发件箱列表，没有则返回 []。"""
    lst = _outbox.get()
    return lst if lst is not None else []
