"""会话记录回溯工具（只读，直接挂在主 Agent 上） (所在层: L3)

查的是事件日志（用户原话 + 专家完整汇报），不受上下文压缩影响。
注意：工具函数里的 `config: RunnableConfig` 参数会遮蔽模块级 config，
所以所有读取 app.config 的逻辑都放在模块级辅助函数里。
"""

import time
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app import config
from app.infra import session_store

_GET_MAX_LIMIT = 20
_SEARCH_MAX_LIMIT = 10
_SNIPPET_RADIUS = 60


def _out_max() -> int:
    return config.SESSION_TOOL_OUTPUT_MAX_CHARS


def _event_max(single: bool) -> int:
    return config.SESSION_TOOL_OUTPUT_MAX_CHARS if single else config.SESSION_TOOL_EVENT_MAX_CHARS


def _fmt_ts(ts: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def _search_text(payload: dict) -> str:
    parts: list[str] = []

    def walk(o):
        if isinstance(o, str):
            if o:
                parts.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(payload)
    return " ".join(parts)


def _event_label(ev) -> str:
    if ev.type == "expert_report" and ev.payload.get("expert"):
        return f"{ev.type}/{ev.payload['expert']}"
    return ev.type


def _event_body(ev) -> str:
    """把事件 payload 渲染成给模型看的一段文本。"""
    p = ev.payload
    t = ev.type
    if t in ("user_message", "assistant_message", "approval_request"):
        body = str(p.get("text", ""))
        if t == "assistant_message" and p.get("image_count"):
            body += f"\n[随回复发出图片 {p['image_count']} 张]"
        return body
    if t == "approval":
        return f"用户回复：{p.get('answer', '')}"
    if t == "expert_report":
        task = str(p.get("task", ""))
        task = task[:100] + ("…" if len(task) > 100 else "")
        head = f"（任务：{task}）\n" if task else ""
        return head + str(p.get("text", ""))
    if t == "tool_call":
        return "; ".join(f"{c.get('name', '')}({c.get('args', '')})" for c in p.get("calls", []))
    if t == "tool_result":
        return f"[{p.get('name', '')}/{p.get('status', '')}] {p.get('text', '')}"
    if t == "turn_failed":
        return f"{p.get('reason', '')} {p.get('detail', '')}".strip()
    if t == "turn_abandoned":
        return str(p.get("reason", ""))
    if t == "session_created":
        return str(p.get("title", ""))
    return _search_text(p)


def _squeeze(text: str, limit: int, seq: int) -> str:
    """超长时头尾各留一半，并提示用 start_seq=N limit=1 看全文。"""
    text = text.strip()
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n…[中间省略 {omitted} 字，用 start_seq={seq} limit=1 看全文]…\n{text[-half:]}"


def _snippet(text: str, tokens: list[str]) -> str:
    low = text.lower()
    pos = -1
    for tok in tokens:
        pos = low.find(tok.lower())
        if pos >= 0:
            break
    if pos < 0:
        return text[: _SNIPPET_RADIUS * 2].replace("\n", " ")
    start = max(0, pos - _SNIPPET_RADIUS)
    end = min(len(text), pos + _SNIPPET_RADIUS)
    return ("…" if start > 0 else "") + text[start:end].replace("\n", " ") + ("…" if end < len(text) else "")


def _resolve(store, session_id: str, config_: Optional[RunnableConfig]):
    """会话解析优先级：session_id 参数 > RunnableConfig.thread_id > 活动会话。"""
    if session_id and session_id.strip():
        info = store.resolve_session(session_id.strip())
        if info is None:
            raise LookupError(f"找不到会话 {session_id}。可以先用 search_session_history 不带关键词查看会话列表。")
        return info
    thread_id = ((config_ or {}).get("configurable") or {}).get("thread_id")
    if thread_id:
        info = store.get_session(str(thread_id))
        if info is not None:
            return info
    info = store.get_session(store.get_active_session_id())
    if info is None:
        raise LookupError("当前没有任何会话记录。")
    return info


@tool
def get_session_events(
    start_seq: int = 0,
    limit: int = 10,
    types: str = "",
    session_id: str = "",
    config: RunnableConfig = None,
) -> str:
    """读取系统保存的完整会话记录（用户原话、专家的完整汇报、工具结果），不受上下文压缩影响。
    用于回顾“刚才 / 之前 / 上次”的内容，通常先用 search_session_history 找到位置，再用本工具读取该位置前后的原文。

    参数：
    - start_seq: 起始事件序号。<=0 表示取该会话最新的 limit 条；>=1 表示从该序号（含）向后取 limit 条。
    - limit: 返回条数，默认 10，最大 20。limit=1 时单条内容不再截断到短上限，适合读一条事件的全文。
    - types: 事件类型过滤，逗号分隔，可选 user_message, assistant_message, expert_report, tool_call, tool_result。留空表示不过滤。
    - session_id: 会话 id（可只写前几位）。留空表示当前对话所在的会话。
    """
    try:
        store = session_store.get_store()
        info = _resolve(store, session_id, config)
        limit = max(1, min(int(limit or 10), _GET_MAX_LIMIT))
        type_list = [t.strip() for t in (types or "").split(",") if t.strip()]
        start_seq = int(start_seq or 0)
        if start_seq <= 0:
            events = store.get_last_events(info.id, limit=limit, types=type_list)
        else:
            events = store.get_events(info.id, start_seq=start_seq, limit=limit, types=type_list)

        total = store.count_events(info.id)
        title = info.title or "（无标题）"
        if not events:
            return f"会话 {info.id[:8]}「{title}」共 {total} 条事件；这个范围内没有符合条件的记录。"

        first, last = events[0].seq, events[-1].seq
        header = f"会话 {info.id[:8]}「{title}」共 {total} 条事件，本次显示 #{first}~#{last}（{len(events)} 条）"
        single = limit == 1
        lines: list[str] = []
        used = len(header)
        shown_last = first - 1
        cut_off = False
        for ev in events:
            line = f"#{ev.seq} {_fmt_ts(ev.ts)} {_event_label(ev)} " + _squeeze(_event_body(ev), _event_max(single), ev.seq)
            if lines and used + len(line) + 1 > _out_max():
                cut_off = True
                break
            if not lines and used + len(line) + 1 > _out_max():
                line = line[: max(_out_max() - used - 40, 100)] + "…[输出过长已截断]"
            lines.append(line)
            used += len(line) + 1
            shown_last = ev.seq

        tail: list[str] = []
        if cut_off:
            tail.append(f"（输出达到长度上限，从 start_seq={shown_last + 1} 继续读取）")
        elif start_seq <= 0:
            if first > 1:
                tail.append(f"（更早的记录：start_seq={max(1, first - limit)} limit={limit}）")
        elif last < store.last_seq(info.id):
            tail.append(f"（继续往后翻：start_seq={last + 1} limit={limit}）")
        return "\n".join([header, *lines, *tail])
    except Exception as e:  # 工具异常一律转成文本，不抛给图
        return f"读取会话记录失败：{type(e).__name__}: {e}"


@tool
def search_session_history(
    query: str = "",
    limit: int = 5,
    session_id: str = "",
    config: RunnableConfig = None,
) -> str:
    """在系统保存的全部会话记录里按关键词搜索（用户原话、专家完整汇报、工具结果），最新的排在前面。
    用于定位“之前查到的XX / 上次聊过的XX”出现在哪个会话的哪个位置，之后可用 get_session_events 读取该位置前后的原文。

    参数：
    - query: 关键词，多个词用空格分隔，必须同时命中；不区分大小写。留空则返回最近若干会话的概览。
    - limit: 最多返回条数，默认 5，最大 10。
    - session_id: 会话 id（可只写前几位）。留空表示在全部会话里搜索；填写则只搜该会话。
    """
    try:
        store = session_store.get_store()
        limit = max(1, min(int(limit or 5), _SEARCH_MAX_LIMIT))
        tokens = [t for t in (query or "").split() if t]

        if not tokens:
            sessions = store.list_sessions(limit=limit)
            if not sessions:
                return "目前没有任何会话记录。"
            rows = [
                f"- {s.id[:8]}「{s.title or '（无标题）'}」 {s.turns} 轮，最后活跃 {_fmt_ts(s.updated_at)}"
                for s in sessions
            ]
            return "最近的会话：\n" + "\n".join(rows) + "\n（给出关键词可以搜索具体内容）"

        scope_id = None
        if session_id and session_id.strip():
            info = store.resolve_session(session_id.strip())
            if info is None:
                return f"找不到会话 {session_id}。可以不带关键词调用本工具查看会话列表。"
            scope_id = info.id

        events = store.search_events(" ".join(tokens), session_id=scope_id, limit=limit)
        if not events:
            return f"没有找到同时包含「{' '.join(tokens)}」的记录。可以换个关键词（更短、更口语化，或只用其中一个词）再试。"

        titles: dict[str, str] = {}
        lines: list[str] = []
        used = 0
        for ev in events:
            if ev.session_id not in titles:
                s = store.get_session(ev.session_id)
                titles[ev.session_id] = (s.title if s else "") or "（无标题）"
            line = (
                f"[{ev.session_id[:8]}「{titles[ev.session_id]}」 #{ev.seq} {_fmt_ts(ev.ts)} {_event_label(ev)}] "
                + _snippet(_search_text(ev.payload), tokens)
            )
            if lines and used + len(line) + 1 > _out_max():
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines) + "\n（用 get_session_events 的 session_id 与 start_seq 读取原文）"
    except Exception as e:
        return f"搜索会话记录失败：{type(e).__name__}: {e}"


TOOLS = [get_session_events, search_session_history]
