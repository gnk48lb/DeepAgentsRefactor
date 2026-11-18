"""测试公用辅助（只给 tests/ 下的脚本用，不属于 app/）。

这些脚本都是直接 `python tests/smoke_xxx.py` 运行的，此时 tests/ 在 sys.path 里，可以 `from _support import ...`。
"""
from app.utils.context import RequestContext, request_context


def owner_context(session_id: str = "test-session"):
    """owner 的请求上下文：run_host_command / 桌面 / 浏览器 / 文档转换这些 require_owner() 门控的工具只在它里面可用。"""
    return request_context(RequestContext(user_id="owner", session_id=session_id, is_owner=True))


def member_context(user_id: str = "member001", session_id: str = "test-session"):
    """member 的请求上下文（user_id 默认是一个合法的 openid 形状）。"""
    return request_context(RequestContext(user_id=user_id, session_id=session_id, is_owner=False))
