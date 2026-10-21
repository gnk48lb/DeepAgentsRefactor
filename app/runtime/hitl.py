"""
app/runtime/hitl.py - 人工授权（HITL）公共辅助 (L1)

人工授权（HITL）公共辅助。同时兼容两种中断 payload：
  1. deepagents / langchain 原生：{"action_requests": [...], "review_configs": [...]}（FileAgent）
  2. 手写 interrupt("提示字符串")（DesktopAgent）
"""
from typing import Any

# reject 时发给模型的文案：保持旧版“严禁重试”的强语义（Gemini 被拒后容易再试一次）
REJECT_MESSAGE = (
    "用户明确拒绝了该操作（权限被拒绝）。禁止重试该操作或任何类似操作；"
    "如果该操作对完成剩余任务必不可少，请立即停止并向主管汇报：任务已被用户取消。"
)

_FOOTER = "回复 Y 允许，回复 N 拒绝。"


def pending_interrupts(state) -> list:
    """StateSnapshot 里所有未处理的中断（包含嵌套子图里的）。"""
    return list(getattr(state, "interrupts", ()) or ())


def has_pending_interrupts(state) -> bool:
    return bool(pending_interrupts(state))


def _target(name: str, args: dict) -> str:
    if name == "move_file":
        return f"{args.get('source')} -> {args.get('destination')}"
    return str(args.get("file_path") or args.get("path") or args.get("destination") or args)


def format_hitl_warning(state) -> str:
    blocks: list[str] = []  # 原生 payload 的“操作/目标”块
    texts: list[str] = []   # 手写字符串 payload（DesktopAgent）
    for intr in pending_interrupts(state):
        val = getattr(intr, "value", intr)
        if isinstance(val, dict) and "action_requests" in val:
            for a in val["action_requests"]:
                name = a.get("name", "未知工具")
                blocks.append(f"操作: [{name}]\n目标: {_target(name, a.get('args') or {})}")
        elif isinstance(val, dict):  # 兼容旧格式 {"tool_name","tool_args"}
            name = val.get("tool_name", "未知工具")
            blocks.append(f"操作: [{name}]\n目标: {_target(name, val.get('tool_args') or {})}")
        elif isinstance(val, str):
            texts.append(val)
    parts: list[str] = []
    if blocks:
        note = "（Y/N 将同时作用于以上全部操作）\n" if len(blocks) > 1 else ""
        parts.append(
            "⚠️ 敏感操作授权拦截\n──────────────────\n"
            + "\n\n".join(blocks)
            + f"\n──────────────────\n{note}{_FOOTER}"
        )
    parts.extend(texts)
    return "\n\n".join(parts) if parts else f"⚠️ 请求执行一项敏感操作，{_FOOTER}"


def build_resume(state, answer: str) -> dict[str, Any]:
    """把用户的 Y/N 转成 {interrupt_id: resume_value}。
    必须用 id 映射：并行派发多个会中断的专家时，直接 Command(resume="Y") 会抛 RuntimeError。"""
    approved = str(answer).strip().upper() == "Y"
    resume: dict[str, Any] = {}
    for intr in pending_interrupts(state):
        val = getattr(intr, "value", intr)
        if isinstance(val, dict) and "action_requests" in val:
            n = len(val["action_requests"])
            decision = {"type": "approve"} if approved else {"type": "reject", "message": REJECT_MESSAGE}
            resume[intr.id] = {"decisions": [dict(decision) for _ in range(n)]}
        else:
            resume[intr.id] = "Y" if approved else "N"
    return resume
