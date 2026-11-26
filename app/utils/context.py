"""请求上下文：当前这一轮是谁在说话 (所在层: L0，只用标准库)

chat_service.wake 在进入图之前设置 RequestContext，工具层靠它做能力门控（require_owner）。
基于 contextvars：asyncio 任务、asyncio.to_thread 以及 LangChain 线程池执行同步工具（copy_context）都会继承它。

失败即关闭：没有上下文、上下文不对，一律按无权限处理或报错，绝不回落成 owner。
"""

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class RequestContext:
    """本轮的身份。新字段只能加在末尾并带默认值：这个类被按位置构造。

    raw_id：平台原始 ID（QQ openid；CLI 为 "cli"），定时任务据此取推送目标，由代码填写，模型无法指定。
    is_scheduled：这一轮是无人值守的定时任务，没有任何人在对话里；require_owner() 对它一律拒绝。
    """

    user_id: str
    session_id: str
    is_owner: bool
    raw_id: str = ""
    is_scheduled: bool = False


_current: contextvars.ContextVar[Optional[RequestContext]] = contextvars.ContextVar(
    "request_context", default=None
)


def set_request_context(ctx: RequestContext) -> contextvars.Token:
    """设置当前上下文，返回 Token；调用方必须在 finally 里用 reset_request_context 还原。"""
    return _current.set(ctx)


def reset_request_context(token: contextvars.Token) -> None:
    _current.reset(token)


@contextmanager
def request_context(ctx: RequestContext) -> Iterator[RequestContext]:
    token = set_request_context(ctx)
    try:
        yield ctx
    finally:
        reset_request_context(token)


def get_request_context() -> Optional[RequestContext]:
    return _current.get()


def require_request_context() -> RequestContext:
    """没有请求上下文就抛 RuntimeError（失败即关闭）。"""
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError("没有请求上下文，已拒绝")
    return ctx


def require_owner() -> RequestContext:
    """只有 owner 能过：没有上下文抛 RuntimeError，非 owner 抛 PermissionError。

    无人值守的定时任务（is_scheduled）即使是 owner 的任务也一律拒绝：没有人在场可以授权。"""
    ctx = require_request_context()
    if ctx.is_scheduled:
        raise PermissionError("无人值守的定时任务不能执行需要管理员权限的操作")
    if not ctx.is_owner:
        raise PermissionError("该操作仅管理员可用")
    return ctx
