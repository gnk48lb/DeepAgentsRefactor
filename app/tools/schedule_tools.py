"""定时任务工具：创建 / 查看 / 取消，只挂在 owner 与 member 的主管上（SCHEDULER_ENABLED 打开时） (所在层: L3)

推送目标由代码在创建时从请求上下文取（RequestContext.user_id / raw_id），工具不接受任何 target / user 参数，
模型既指定不了、也改不了；无人值守的定时任务（is_scheduled）里一律拒绝，所以定时任务不能再创建定时任务。
预期内的失败（参数不对、超限、身份不明）一律返回带纠正建议的中文文本，不抛异常给模型。
参数类型故意宽松（不用 Literal / Optional 联合 / 带时区偏移的 ISO）：在函数里自己校验，错了告诉模型怎么改。
注意：工具函数在 LangChain 的线程池里执行，请求上下文靠 contextvars 传进来（见 app/utils/context.py）。
"""

import time
import unicodedata
from typing import Optional

from langchain_core.tools import tool

from app import config
from app.infra import schedule_store
from app.utils import schedule_time
from app.utils.context import RequestContext, get_request_context
from app.utils.identity import OWNER_USER_ID, resolve_platform_user

TITLE_MAX_CHARS = 40
PROMPT_MAX_CHARS = {"reminder": 200, "agent": 800}
LIST_MAX_ROWS = 20
_PREVIEW_CHARS = 60

_FIRST_HINT = (
    "首次创建提示：到点的消息由机器人主动发出，如果你在 QQ 里关闭了对机器人的“允许主动发送消息”，就收不到；"
    "收不到时可以发 /tasks 查看任务状态。"
)
_CLI_NOTE = "（这个任务的目标是命令行，只有 main.py --run-scheduler 运行时才会触发）"
_STATUS_TEXT = {"active": "进行中", "done": "已完成"}
_PAUSE_REASON_TEXT = {
    "user": "你暂停的",
    "admin": "被管理员暂停",
    "push_refused": "推送被拒收，请确认没有关闭“允许主动发送消息”，再发 /tasks resume 恢复",
    "failures": "连续失败，问题解决后发 /tasks resume 恢复",
}
_LAST_TEXT = {"ok": "成功", "failed": "失败", "skipped": "已跳过", "missed": "错过了"}
_MODE_TEXT = {"reminder": "固定提醒", "agent": "到点执行"}


def _clean(text, *, keep_newlines: bool) -> str:
    """去掉控制字符（含零宽等格式字符）；多余空白折叠，需要时保留换行。"""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for ch in text:
        if ch == "\n":
            out.append("\n" if keep_newlines else " ")
        elif ch == "\t":
            out.append(" ")
        elif unicodedata.category(ch)[0] == "C":
            continue
        else:
            out.append(ch)
    cleaned = "".join(out)
    if keep_newlines:
        lines = [" ".join(line.split()) for line in cleaned.split("\n")]
        return "\n".join(lines).strip()
    return " ".join(cleaned.split())


def _caller() -> tuple[Optional[RequestContext], str]:
    """当前请求的身份。返回 (上下文, 拒绝原因)：失败即关闭，没有上下文 / 无人值守 / 身份不明一律拒绝。"""
    ctx = get_request_context()
    if ctx is None:
        return None, "无法确定当前用户，已拒绝。"
    if ctx.is_scheduled:
        return None, "无人值守的定时任务里不能再创建、查看或取消定时任务。"
    if not ctx.user_id or ctx.user_id.startswith("cli-"):
        return None, "命令行里的测试成员不支持定时任务。"
    return ctx, ""


def _target_openid(ctx: RequestContext) -> tuple[Optional[str], str]:
    """推送目标：只取自请求上下文。命令行里的 owner 目标是 cli；其余必须是合法的平台 openid，且能映射回当前用户。"""
    if not ctx.raw_id:
        return None, "无法确定推送目标，已拒绝创建定时任务。"
    if ctx.user_id == OWNER_USER_ID and ctx.raw_id == schedule_store.CLI_TARGET:
        return schedule_store.CLI_TARGET, ""
    try:
        identity = resolve_platform_user(ctx.raw_id)
    except ValueError:
        return None, "无法确定推送目标，已拒绝创建定时任务。"
    if identity.user_id != ctx.user_id:
        return None, "无法确定推送目标，已拒绝创建定时任务。"
    return ctx.raw_id, ""


def _store_error(action: str, e: Exception) -> str:
    print(f"[schedule_tools] {action}失败: {type(e).__name__}")
    return f"{action}失败：定时任务存储暂时不可用，请稍后再试。"


def _preview(text: str) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= _PREVIEW_CHARS else one_line[:_PREVIEW_CHARS] + "…"


@tool
def create_scheduled_task(
    title: str,
    mode: str,
    prompt: str,
    in_minutes: int = 0,
    run_at: str = "",
    cron: str = "",
) -> str:
    """为当前用户创建一个定时任务：到点后系统会通过 QQ 私聊主动把消息发给该用户（推送对象由系统自动确定，不能指定）。
    只在用户明确要求“提醒我 / 每天……告诉我 / N 分钟后叫我 / 定时……”时才调用；一个任务只做一件事，多件事分别创建。

    参数：
    - title: 任务名，40 字以内，例如“带伞提醒”。
    - mode: 只能是 "reminder" 或 "agent"。
      reminder = 到点把 prompt 原样发给用户（不调用任何模型），适合固定的话：喝水、开会、吃药、关火。
      agent = 到点由系统按 prompt 里的指令执行一次，再把最终文字发给用户，适合需要查资料的：查天气再提醒带不带伞。
    - prompt: reminder 时，是要原样发给用户的那一句话（200 字以内）；agent 时，是到点要执行的指令（800 字以内）。
      agent 的指令必须自包含：到点执行时没有当前对话，看不到上下文和长期记忆，城市、对象、要求、语气都要写进去。
      例如“查询北京今天的天气，用一两句话提醒用户出门要不要带伞、穿什么，语气简短”。
      到点执行时只能用资料库、媒体、地图（含天气）这几类能力，做不到的需求不要创建 agent 任务。
    - 时间参数 in_minutes、run_at、cron 三选一，只填一个，其余留空（in_minutes 填 0）：
      * in_minutes: 整数，几分钟后触发一次（1～525600），例如 30 表示 30 分钟后。
      * run_at: 用户本地时间 "YYYY-MM-DD HH:MM"，触发一次，例如 "2026-11-27 08:30"；必须是未来、一年之内；不要带时区。
        “明天 / 下周一”之类要先调用 get_current_time 看清今天是几号再换算。
      * cron: 5 段 "分 时 日 月 周"，重复触发，按用户本地时区。每天 8:30 是 "30 8 * * *"，工作日 9 点是 "0 9 * * 1-5"，
        每周一三五 9 点是 "0 9 * * 1,3,5"，每月 1 号 10 点是 "0 10 1 * *"。相邻两次触发至少隔 30 分钟，不支持秒与 @daily 这类别名。
    返回创建结果或失败原因；失败时按原因修正参数，不要用同样的参数重试。
    """
    print("\n[Tool Call: create_scheduled_task]")
    ctx, refused = _caller()
    if ctx is None:
        return f"创建失败：{refused}"
    target, refused = _target_openid(ctx)
    if target is None:
        return f"创建失败：{refused}"

    title = _clean(title, keep_newlines=False)
    if not title:
        return "创建失败：任务名不能为空。"
    if len(title) > TITLE_MAX_CHARS:
        return f"创建失败：任务名太长（{len(title)} 字），请缩到 {TITLE_MAX_CHARS} 字以内。"
    mode = (mode or "").strip().lower() if isinstance(mode, str) else ""
    if mode not in PROMPT_MAX_CHARS:
        return "创建失败：mode 只能是 reminder（到点原样发一句话）或 agent（到点执行一次指令再发结果）。"
    prompt = _clean(prompt, keep_newlines=True)
    if not prompt:
        return "创建失败：prompt 不能为空。"
    limit = PROMPT_MAX_CHARS[mode]
    if len(prompt) > limit:
        return f"创建失败：{mode} 模式的 prompt 最多 {limit} 字，现在有 {len(prompt)} 字，请精简。"

    now = time.time()
    try:
        spec = schedule_time.parse_spec(in_minutes, run_at, cron, now=now, tz_name=config.DEFAULT_TIMEZONE)
    except schedule_time.ScheduleError as e:
        return f"创建失败：{e}"

    max_jobs = config.SCHEDULER_MAX_JOBS_OWNER if ctx.user_id == OWNER_USER_ID else config.SCHEDULER_MAX_JOBS_MEMBER
    try:
        store = schedule_store.get_schedule_store()
        job, created = store.create_job(
            user_id=ctx.user_id, target_openid=target, title=title, mode=mode, prompt=prompt, kind=spec.kind,
            cron=spec.cron, timezone=spec.timezone, next_run_at=spec.next_run_at, spec_key=spec.key,
            max_jobs=max_jobs, now=now,
        )
    except schedule_store.ScheduleStoreError as e:
        return f"创建失败：{e}"
    except Exception as e:
        return _store_error("创建", e)

    when = schedule_time.describe(job.kind, job.cron, job.next_run_at, job.timezone)
    if job.kind == "cron":
        when += f"，下次 {schedule_time.fmt_local(job.next_run_at, job.timezone)}"
    what = (f"到点会发送「{_preview(job.prompt)}」" if job.mode == "reminder"
            else f"到点会按这条指令执行，并把结果发给你：「{_preview(job.prompt)}」")
    if not created:
        return f"这个定时任务刚刚已经创建过了（编号 {job.short_id}），没有重复创建：{when}。{what}。"
    text = f"已创建定时任务「{job.title}」（编号 {job.short_id}）：{when}。{what}。到点我会通过 QQ 主动发给你。"
    if target == schedule_store.CLI_TARGET:
        text += _CLI_NOTE
    try:
        if store.take_first_hint(ctx.user_id):
            text += "\n" + _FIRST_HINT
    except Exception as e:
        print(f"[schedule_tools] 记录首次提示失败: {type(e).__name__}")
    return text


def _status_text(job: schedule_store.Job) -> str:
    if job.status == "paused":
        return "已暂停（" + _PAUSE_REASON_TEXT.get(job.paused_reason or "", "已暂停") + "）"
    return _STATUS_TEXT.get(job.status, job.status)


def format_job_line(index: int, job: schedule_store.Job) -> str:
    """一行任务描述：序号、标题、模式、编号前 8 位、状态（含暂停原因）、周期、下次时间、上次结果。"""
    when = schedule_time.describe(job.kind, job.cron, job.next_run_at, job.timezone)
    parts = [
        f"{index}. 「{job.title}」（{_MODE_TEXT.get(job.mode, job.mode)}）编号 {job.short_id}",
        _status_text(job),
        when,
    ]
    if job.status == "active":
        parts.append(f"下次 {schedule_time.fmt_local(job.next_run_at, job.timezone)}")
    parts.append("上次：" + (_LAST_TEXT.get(job.last_status, job.last_status) if job.last_status else "还没执行过"))
    return "｜".join(parts)


@tool
def list_scheduled_tasks(placeholder: str = "default") -> str:
    """列出当前用户的全部定时任务：序号、任务名、编号、状态、周期、下次触发时间与上次结果（最多 20 条）。
    用户问“我有哪些提醒 / 定时任务”，或要取消某个任务之前先看一遍编号时调用。只能看到用户自己的任务。
    本工具不需要任何参数，placeholder 可忽略。"""
    print("\n[Tool Call: list_scheduled_tasks]")
    ctx, refused = _caller()
    if ctx is None:
        return refused
    try:
        jobs = schedule_store.get_schedule_store().list_jobs(ctx.user_id)
    except Exception as e:
        return _store_error("查看", e)
    if not jobs:
        return "你现在没有定时任务。"
    lines = [format_job_line(i, job) for i, job in enumerate(jobs[:LIST_MAX_ROWS], start=1)]
    if len(jobs) > LIST_MAX_ROWS:
        lines.append(f"（另有 {len(jobs) - LIST_MAX_ROWS} 个任务没有显示）")
    return "\n".join(lines)


@tool
def cancel_scheduled_task(task_ref: str) -> str:
    """取消（删除）当前用户的一个定时任务。task_ref 是 list_scheduled_tasks 里的序号（如 "2"），
    或编号的前 4 位以上（如 "ab12cd34"）。取消前先调用 list_scheduled_tasks 确认是哪一个，不要凭记忆猜编号。
    只能取消用户自己的任务；正在执行的那一次仍会照常跑完，之后不再触发。"""
    print("\n[Tool Call: cancel_scheduled_task]")
    ctx, refused = _caller()
    if ctx is None:
        return f"取消失败：{refused}"
    try:
        store = schedule_store.get_schedule_store()
        job = store.resolve(str(task_ref), ctx.user_id, is_owner=False)       # 工具只管用户自己的任务，owner 也一样
        deleted = store.cancel_job(job.id, user_id=ctx.user_id)
    except schedule_store.ScheduleStoreError as e:
        return f"取消失败：{e}"
    except Exception as e:
        return _store_error("取消", e)
    if not deleted:
        return "取消失败：这个任务已经不存在了（可能刚被取消）。"
    return f"已取消定时任务「{job.title}」（编号 {job.short_id}）。"


TOOLS = [create_scheduled_task, list_scheduled_tasks, cancel_scheduled_task]
