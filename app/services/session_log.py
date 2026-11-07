"""会话事件日志镜像与命令输出格式化 (所在层: L5)

纯函数，便于离线测试：
- mirror_messages：把图状态（checkpoint）里的消息按 event_key 幂等地镜像进事件日志
- format_*：/sessions 等斜杠命令用的相对时间与会话行
"""

import json
import re
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app import config
from app.utils.text import extract_text

# 回溯工具自己的结果不镜像：否则“查到的历史”会被当成新事实再次入库，污染搜索
_MIRROR_SKIP_TOOLS = {"get_session_events", "search_session_history", "write_todos"}
_TASK_TOOL = "task"
_ARGS_MAX_CHARS = 2000

_DATA_URI_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=_-]+")
_IMAGE_OMITTED = "[图片已省略]"


def _clean_text(text: str) -> tuple[str, bool]:
    """去掉 data:image URI，并按单段上限截断。返回 (文本, 是否被截断)。"""
    text = _DATA_URI_RE.sub(_IMAGE_OMITTED, text or "")
    limit = config.SESSION_EVENT_TEXT_MAX_CHARS
    if len(text) > limit:
        return text[:limit] + f"…[已截断，原文共 {len(text)} 字]", True
    return text, False


def _clean_args(args: Any) -> str:
    try:
        raw = json.dumps(args, ensure_ascii=False, default=str)
    except Exception:
        raw = str(args)
    raw = _DATA_URI_RE.sub(_IMAGE_OMITTED, raw)
    if len(raw) > _ARGS_MAX_CHARS:
        raw = raw[:_ARGS_MAX_CHARS] + f"…[已截断，原文共 {len(raw)} 字]"
    return raw


def _msg_key(msg: Any, fallback: str) -> str:
    mid = getattr(msg, "id", None)
    return f"msg:{mid}" if mid else fallback


def mirror_messages(store, session_id: str, messages: list) -> int:
    """把图状态里的消息镜像进事件日志，返回新写入的事件数。

    每轮收尾和启动恢复时都对“全部消息”整体重放：靠 key 幂等去重，
    所以摘要压缩、重试都不会造成重复或丢失。事件 ts 是写入时间，不是原始发生时间。
    不镜像：HumanMessage / SystemMessage（用户输入已由 user_message 事件记录，且含 <long_term_memory> 块）、
    不带 tool_calls 的 AIMessage（最终回复由 assistant_message 记录）。
    """
    # tool_call_id -> 发起它的 tool_call，用于给 ToolMessage 补全 name / task 信息
    calls_by_id: dict[str, dict] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                if tc.get("id"):
                    calls_by_id[tc["id"]] = tc

    written = 0
    for m in messages:
        if isinstance(m, (HumanMessage, SystemMessage)):
            continue

        if isinstance(m, AIMessage):
            tcs = getattr(m, "tool_calls", None) or []
            if not tcs:
                continue
            kept = [tc for tc in tcs if tc.get("name") != _TASK_TOOL and tc.get("name") not in _MIRROR_SKIP_TOOLS]
            if not kept:
                continue
            fallback = "calls:" + ",".join(str(tc.get("id") or tc.get("name")) for tc in kept)
            payload = {"calls": [{"name": tc.get("name", ""), "args": _clean_args(tc.get("args"))} for tc in kept]}
            if store.emit_event(session_id, "tool_call", payload, key=_msg_key(m, fallback)) is not None:
                written += 1
            continue

        if isinstance(m, ToolMessage):
            call = calls_by_id.get(getattr(m, "tool_call_id", "") or "", {})
            name = getattr(m, "name", None) or call.get("name") or ""
            if name in _MIRROR_SKIP_TOOLS:
                continue
            text, truncated = _clean_text(extract_text(getattr(m, "content", "")))
            status = getattr(m, "status", None) or "success"
            key = _msg_key(m, f"tool:{getattr(m, 'tool_call_id', '')}")
            if name == _TASK_TOOL:
                args = call.get("args") or {}
                payload = {
                    "expert": str(args.get("subagent_type", "")),
                    "task": _clean_text(str(args.get("description", "")))[0],
                    "text": text,
                    "status": status,
                }
                etype = "expert_report"
            else:
                payload = {"name": name, "status": status, "text": text}
                etype = "tool_result"
            if truncated:
                payload["truncated"] = True
            if store.emit_event(session_id, etype, payload, key=key) is not None:
                written += 1
    return written


# ── 斜杠命令输出用的格式化小函数 ───────────────────────────────


def format_relative_time(ts: float, now: Optional[float] = None) -> str:
    """刚刚 / N分钟前 / N小时前 / 昨天 HH:MM / MM-DD HH:MM"""
    now = time.time() if now is None else now
    delta = now - ts
    if delta < 60:
        return "刚刚"
    if delta < 3600:
        return f"{int(delta // 60)}分钟前"
    t, n = time.localtime(ts), time.localtime(now)
    if (t.tm_year, t.tm_yday) == (n.tm_year, n.tm_yday):
        return f"{int(delta // 3600)}小时前"
    yesterday = time.localtime(now - 86400)
    if (t.tm_year, t.tm_yday) == (yesterday.tm_year, yesterday.tm_yday):
        return time.strftime("昨天 %H:%M", t)
    return time.strftime("%m-%d %H:%M", t)


def format_session_line(
    index: int,
    info,
    *,
    is_active: bool = False,
    is_pending: bool = False,
    now: Optional[float] = None,
) -> str:
    marks = ""
    if is_active:
        marks += "[当前]"
    if is_pending:
        marks += "[待授权]"
    title = info.title or "（无标题）"
    head = f"{index}. {marks} " if marks else f"{index}. "
    return f"{head}{info.id[:8]} {title} · {info.turns}轮 · {format_relative_time(info.updated_at, now)}"
