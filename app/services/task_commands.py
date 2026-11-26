"""/tasks 斜杠命令：查看与管理定时任务 (所在层: L5)

只依赖存储与时间工具（不 import chat_service：循环依赖且太重）。handle 是同步函数，与 touch_user 同风格直接读写存储；
回复一律是纯文本（QQ 不渲染 Markdown），日志里不记录任务正文。

  /tasks                  列出我的任务
  /tasks rm <序号|编号前缀>      取消（别名 del）
  /tasks pause <…>        暂停          /tasks resume <…>        恢复
  /tasks all              仅 owner：列出所有用户的任务（每行带用户显示名与编号前缀）
member 只能管自己的任务；owner 另可用编号前缀对任何用户的任务做 rm / pause / resume（序号永远指自己的列表）。
被管理员暂停（paused_reason=admin）的任务，member 不能自己恢复。
"""

from app import config
from app.infra import schedule_store, session_store
from app.utils import schedule_time
from app.utils.identity import OWNER_USER_ID, Identity

DISABLED_TEXT = "定时任务功能未启用。"
ADMIN_ONLY_TEXT = "该命令仅管理员可用"          # 与 chat_service 里管理员命令的固定回复一致
LIST_MAX_ROWS = 50
_ERROR_TEXT = "定时任务功能暂时出错，请稍后再试。"

_USAGE = (
    "用法：\n"
    "/tasks  查看我的定时任务\n"
    "/tasks rm <序号|编号前缀>  取消（也可写 del）\n"
    "/tasks pause <序号|编号前缀>  暂停\n"
    "/tasks resume <序号|编号前缀>  恢复"
)
_ADMIN_USAGE = "\n/tasks all  查看所有用户的任务（管理员）；rm / pause / resume 可用编号前缀操作任何人的任务"
_LIST_HINT = "用 /tasks rm <序号> 取消，/tasks pause <序号> 暂停，/tasks resume <序号> 恢复。"
_EMPTY = "你现在没有定时任务。直接告诉我“每天早上 8 点提醒我……”这样的需求就能创建。"


def handle(arg: str, identity: Identity) -> str:
    """执行一条 /tasks 命令，返回要回复给用户的纯文本。永不抛异常。"""
    if not config.SCHEDULER_ENABLED:
        return DISABLED_TEXT
    parts = (arg or "").strip().split(None, 1)
    sub = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""
    try:
        store = schedule_store.get_schedule_store()
        if sub == "":
            return _list_mine(store, identity)
        if sub == "all":
            if not identity.is_owner:
                return ADMIN_ONLY_TEXT
            return _list_all(store)
        if sub in ("rm", "del"):
            return _cmd_rm(store, identity, rest)
        if sub == "pause":
            return _cmd_pause(store, identity, rest)
        if sub == "resume":
            return _cmd_resume(store, identity, rest)
        return _usage(identity)
    except schedule_store.ScheduleStoreError as e:
        return str(e)
    except Exception as e:
        print(f"[task_commands] /tasks 出错: {type(e).__name__}")
        return _ERROR_TEXT


def _usage(identity: Identity) -> str:
    return _USAGE + (_ADMIN_USAGE if identity.is_owner else "")


def _who(user_id: str) -> str:
    """用户显示名：备注（编号前缀）或编号前缀；查不到备注就只显示前缀。"""
    short = user_id[:8]
    try:
        user = session_store.get_store().get_user(user_id)
    except Exception:
        user = None
    return f"{user.label}（{short}）" if user is not None and user.label else short


def _list_mine(store, identity: Identity) -> str:
    jobs = store.list_jobs(identity.user_id)
    if not jobs:
        return _EMPTY
    lines = [schedule_time.format_job_line(i, job) for i, job in enumerate(jobs[:LIST_MAX_ROWS], start=1)]
    if len(jobs) > LIST_MAX_ROWS:
        lines.append(f"（另有 {len(jobs) - LIST_MAX_ROWS} 个任务没有显示）")
    tail = _LIST_HINT + ("用 /tasks all 查看所有人的任务。" if identity.is_owner else "")
    return "你的定时任务：\n" + "\n".join(lines) + "\n" + tail


def _list_all(store) -> str:
    jobs = store.list_jobs(None)
    if not jobs:
        return "现在没有任何定时任务。"
    names: dict[str, str] = {}
    lines = []
    for i, job in enumerate(jobs[:LIST_MAX_ROWS], start=1):
        who = names.get(job.user_id)
        if who is None:
            who = names[job.user_id] = _who(job.user_id)
        lines.append(schedule_time.format_job_line(i, job, who=who))
    if len(jobs) > LIST_MAX_ROWS:
        lines.append(f"（另有 {len(jobs) - LIST_MAX_ROWS} 个任务没有显示）")
    return ("所有用户的定时任务（共 %d 个）：\n" % len(jobs)) + "\n".join(lines) \
        + "\n对别人的任务请用编号前缀：/tasks rm <编号前缀>、/tasks pause <编号前缀>、/tasks resume <编号前缀>。"


def _need_ref(sub: str, rest: str) -> str:
    return f"用法：/tasks {sub} <序号|编号前缀>。"


def _resolve(store, identity: Identity, rest: str):
    return store.resolve(rest, identity.user_id, is_owner=identity.is_owner)


def _owner_note(job, identity: Identity) -> str:
    """owner 操作别人的任务时，回复里带上是谁的。"""
    return f"（{_who(job.user_id)} 的任务）" if identity.is_owner and job.user_id != identity.user_id else ""


def _cmd_rm(store, identity: Identity, rest: str) -> str:
    if not rest:
        return _need_ref("rm", rest)
    job = _resolve(store, identity, rest)
    scope = None if identity.is_owner else identity.user_id
    if not store.cancel_job(job.id, user_id=scope):
        return "没有找到这个任务（可能刚被取消）。"
    return f"已取消定时任务「{job.title}」（编号 {job.id[:8]}）{_owner_note(job, identity)}。"


def _cmd_pause(store, identity: Identity, rest: str) -> str:
    if not rest:
        return _need_ref("pause", rest)
    job = _resolve(store, identity, rest)
    # owner 暂停别人的任务记成 admin（member 之后不能自己恢复）；暂停自己的任务就是普通的 user
    reason = "admin" if identity.is_owner and job.user_id != OWNER_USER_ID else "user"
    scope = None if identity.is_owner else identity.user_id
    store.pause_job(job.id, reason, user_id=scope)
    return f"已暂停定时任务「{job.title}」（编号 {job.id[:8]}）{_owner_note(job, identity)}。恢复请发 /tasks resume {job.id[:8]}。"


def _cmd_resume(store, identity: Identity, rest: str) -> str:
    if not rest:
        return _need_ref("resume", rest)
    job = _resolve(store, identity, rest)
    scope = None if identity.is_owner else identity.user_id
    resumed = store.resume_job(job.id, is_owner=identity.is_owner, user_id=scope)
    return (
        f"已恢复定时任务「{resumed.title}」（编号 {resumed.id[:8]}）{_owner_note(job, identity)}，"
        f"{schedule_time.describe(resumed.kind, resumed.cron, resumed.next_run_at, resumed.timezone)}，"
        f"下次 {schedule_time.fmt_local(resumed.next_run_at, resumed.timezone)}。"
    )
