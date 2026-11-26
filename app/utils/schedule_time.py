"""定时任务的时间解析与计算 (所在层: L0，只依赖标准库、cronsim、app.config)

全部是纯函数。业务时区是 config.DEFAULT_TIMEZONE；库里只存 epoch 秒、cron 字符串和创建任务时的时区名，
所以之后改配置不会悄悄挪动已有任务。调度计算一律用“epoch + 时区名”，不用 time.localtime。
import 时不读时区库、不建对象：ZoneInfo 在第一次 get_tz() 时才创建。

cronsim 的实测边界（tests/smoke_scheduler.py 场景 A 把它们固化成了断言）：
  - 它对“严格晚于”的处理是对的：基准时刻恰好等于触发时刻时，返回的是下一次；
  - 它接受 6 段（带秒）、L / 5L / MON#5 / LW，拒绝 @daily 这类别名（报“字段数不对”）、不可能的日期（0 0 30 2 * 直接抛错）；
  - 月 / 周的英文缩写（JAN、MON-FRI）可用，星期 0 和 7 都是周日；
  - 夏令时：缺失的小时里，固定时刻的任务顺延到当天第一个存在的时刻（30 2 * * * 变成 03:00），
    “每 30 分钟”这种带步长的任务当天直接跳过；重复的小时里，固定时刻的任务只触发一次，带步长的任务触发两次。
我们自己的校验在它前面，冲突时以我们的为准：恰好 5 段、不接受 @ 别名与秒 / 年字段、字符集受限。
"""

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from typing import Optional
from zoneinfo import ZoneInfo

from cronsim import CronSim, CronSimError

from app import config

MAX_IN_MINUTES = 525600                 # 一年
RUN_AT_MIN_LEAD_SECONDS = 10            # run_at 必须比现在晚至少这么多秒
HORIZON_SECONDS = 366 * 86400           # 一次性任务最远 / cron 至少要在这个范围内触发一次
_PROBE_MAX_FIRES = 400                  # 校验 cron 最小间隔时最多取多少次触发（一年内触发不到 400 次的表达式会被完整检查）

_WEEKDAY_FULL = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
_WEEKDAY_SHORT = ("一", "二", "三", "四", "五", "六", "日")
_DOW_NAMES = {"MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6, "SUN": 7}
_CRON_FIELD_RE = re.compile(r"[0-9A-Za-z*/,#\-]{1,40}")
_RUN_AT_RE = re.compile(r"([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})[ T]([0-9]{1,2}):([0-9]{2})(?::([0-9]{2}))?")
_RUN_AT_OFFSET_RE = re.compile(r"(?:[zZ]|[+-][0-9]{2}:?[0-9]{2})$")

_CRON_REASONS = {
    "Bad minute": "分钟（第 1 段）不合法，取值 0-59",
    "Bad hour": "小时（第 2 段）不合法，取值 0-23",
    "Bad day-of-month": "日（第 3 段）不合法：取值 1-31 或 L（月末）；2 月没有 30、31 日，4/6/9/11 月没有 31 日",
    "Bad month": "月（第 4 段）不合法，取值 1-12 或 JAN-DEC",
    "Bad day-of-week": "星期（第 5 段）不合法，取值 0-7（0 和 7 都是周日）或 MON-SUN",
    "Wrong number of fields": "段数不对，必须恰好 5 段",
}


class ScheduleError(ValueError):
    """时间参数不合法。message 是给模型 / 用户看的中文说明：哪里错、怎么改，并带上当前业务时间。"""


@dataclass(frozen=True)
class ScheduleSpec:
    kind: str                 # "once" | "cron"
    cron: Optional[str]       # kind == "cron" 时的规范化 5 段表达式（单空格、大写）
    next_run_at: float        # 第一次触发的 epoch 秒
    timezone: str             # 任务自己的时区名（创建时的 DEFAULT_TIMEZONE）
    key: str                  # 去重用的规格键：in:30 / at:<epoch> / cron:<表达式>


# ── 时区 ───────────────────────────────────────────────────

_tz_cache: dict[str, tzinfo] = {}
_tz_lock = threading.Lock()


def get_tz(name: Optional[str] = None) -> tzinfo:
    """名字（默认 DEFAULT_TIMEZONE）对应的时区；名字无效或系统缺 tzdata 时回落系统本地时区，同名只警告一次。按名字缓存。"""
    key = (name or config.DEFAULT_TIMEZONE or "").strip()
    with _tz_lock:
        tz = _tz_cache.get(key)
        if tz is not None:
            return tz
        try:
            tz = ZoneInfo(key)
        except Exception:   # ZoneInfoNotFoundError / 非法 key 的 ValueError / 缺 tzdata 的 OSError
            tz = datetime.now().astimezone().tzinfo or timezone.utc
            print(f"[WARN] schedule_time: 时区 {key[:60]!r} 不可用（名字无效或缺少 tzdata），回落到系统本地时区")
        _tz_cache[key] = tz
        return tz


def clear_tz_cache() -> None:
    """测试用：清掉 get_tz 的缓存（含“已警告过”的状态）。"""
    with _tz_lock:
        _tz_cache.clear()


def _at(epoch: float, tz_name: Optional[str]) -> datetime:
    return datetime.fromtimestamp(epoch, get_tz(tz_name))


def fmt_local(epoch: float, tz_name: Optional[str] = None) -> str:
    """“MM-DD HH:MM”，按给定时区（默认业务时区）。"""
    return _at(epoch, tz_name).strftime("%m-%d %H:%M")


def fmt_full(epoch: float, tz_name: Optional[str] = None) -> str:
    """“YYYY-MM-DD HH:MM 星期X”，按给定时区（默认业务时区）。"""
    dt = _at(epoch, tz_name)
    return f"{dt:%Y-%m-%d %H:%M} {_WEEKDAY_FULL[dt.weekday()]}"


def fmt_now(now: float, tz_name: Optional[str] = None) -> str:
    """错误信息里带的“当前业务时间”，帮模型自我纠正（它必须先 get_current_time 再换算 run_at）。"""
    dt = _at(now, tz_name)
    return f"当前业务时间 {dt:%Y-%m-%d %H:%M}，{_WEEKDAY_FULL[dt.weekday()]}，时区 {tz_name or config.DEFAULT_TIMEZONE}"


# ── 解析 ───────────────────────────────────────────────────

def _err(message: str, now: float, tz_name: Optional[str]) -> ScheduleError:
    return ScheduleError(f"{message}（{fmt_now(now, tz_name)}）")


def _given(value) -> bool:
    """工具的默认值是 0 / ""，它们都表示“没填”。bool 一律算填了（随后会被拒绝）。"""
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (int, float)):
        return value != 0
    return True


def _parse_minutes(value, now: float, tz_name: Optional[str]) -> int:
    hint = "in_minutes 必须是 1～525600 的整数（多少分钟后触发）。几小时后请自己换算成分钟；更久以后请改用 run_at"
    if isinstance(value, bool):
        raise _err(hint, now, tz_name)
    if isinstance(value, int):
        minutes = value
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")) or not value.is_integer():
            raise _err(hint, now, tz_name)
        minutes = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[0-9]{1,7}", text):
            minutes = int(text)
        elif re.fullmatch(r"[0-9]{1,7}\.0+", text):
            minutes = int(float(text))
        else:
            raise _err(hint, now, tz_name)
    else:
        raise _err(hint, now, tz_name)
    if not 1 <= minutes <= MAX_IN_MINUTES:
        raise _err(hint, now, tz_name)
    return minutes


def _parse_run_at(value, now: float, tz_name: Optional[str]) -> float:
    hint = "run_at 要写本地时间，格式 YYYY-MM-DD HH:MM，例如 2026-11-27 08:30（不要带时区或 Z）"
    if not isinstance(value, str):
        raise _err(hint, now, tz_name)
    text = value.strip()
    if _RUN_AT_OFFSET_RE.search(text):
        raise _err("run_at 不要带时区（Z 或 +08:00），一律按用户本地时间写。" + hint, now, tz_name)
    m = _RUN_AT_RE.fullmatch(text)
    if m is None:
        raise _err(hint, now, tz_name)
    year, month, day, hour, minute, second = (int(g) if g is not None else 0 for g in m.groups())
    try:
        naive = datetime(year, month, day, hour, minute, second)
    except ValueError:
        raise _err("run_at 不是一个有效的日期时间。" + hint, now, tz_name) from None
    tz = get_tz(tz_name)
    aware = naive.replace(tzinfo=tz)
    # 夏令时“跳过”的时刻（如春季 02:30）按墙上时间往返一次会变样：这种时刻不存在，直接报错而不是悄悄挪动
    if aware.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None) != naive:
        raise _err("这个本地时间因夏令时调整而不存在，请换一个时间", now, tz_name)
    epoch = aware.timestamp()
    if epoch < now + RUN_AT_MIN_LEAD_SECONDS:
        raise _err("run_at 已经过去，或离现在不足 10 秒。请先调用 get_current_time 确认现在几点，再换算", now, tz_name)
    if epoch > now + HORIZON_SECONDS:
        raise _err("run_at 太远：一次性任务最多设在 366 天以内", now, tz_name)
    return epoch


def _cron_hint(minutes: int) -> str:
    if minutes <= 59:
        return f"每 {minutes} 分钟一次可写 */{minutes} * * * *，每小时一次可写 0 * * * *"
    if minutes <= 60:
        return "每小时一次可写 0 * * * *"
    return "每天一次可写 0 8 * * *（每天 8:00）"


def _parse_cron(value, now: float, tz_name: Optional[str]) -> tuple[str, float]:
    format_hint = "cron 必须是 5 段：分 时 日 月 周，用空格隔开。每天 8:30 是 30 8 * * *，工作日 9 点是 0 9 * * 1-5"
    if not isinstance(value, str):
        raise _err(format_hint, now, tz_name)
    text = value.strip()
    if text.startswith("@"):
        raise _err("不支持 @daily 这类别名。" + format_hint, now, tz_name)
    if len(text) > 100:
        raise _err("cron 太长。" + format_hint, now, tz_name)
    parts = text.split()
    if len(parts) != 5:
        raise _err(f"cron 恰好要 5 段，你写了 {len(parts)} 段（不支持秒和年）。" + format_hint, now, tz_name)
    if not all(_CRON_FIELD_RE.fullmatch(p) for p in parts):
        raise _err("cron 里有不支持的字符。" + format_hint, now, tz_name)
    expr = " ".join(parts).upper()

    tz = get_tz(tz_name)
    base = datetime.fromtimestamp(now, tz)
    horizon = now + HORIZON_SECONDS
    fires: list[float] = []
    try:
        for dt in CronSim(expr, base):
            ts = dt.timestamp()
            if ts > horizon:
                break
            fires.append(ts)
            if len(fires) >= _PROBE_MAX_FIRES:
                break
    except CronSimError as e:
        reason = _CRON_REASONS.get(str(e), str(e))
        raise _err(f"cron 表达式不合法：{reason}。" + format_hint, now, tz_name) from None
    except (ValueError, OverflowError, OSError):
        raise _err("cron 表达式无法计算触发时间。" + format_hint, now, tz_name) from None
    if not fires:
        raise _err("这个 cron 在未来 366 天内一次都不会触发（比如日期不存在，或要等很多年）。" + format_hint, now, tz_name)

    limit_minutes = config.SCHEDULER_MIN_INTERVAL_MINUTES
    if len(fires) >= 2:
        shortest = min(b - a for a, b in zip(fires, fires[1:]))
        if shortest < limit_minutes * 60:
            raise _err(
                f"重复任务相邻两次触发至少要隔 {limit_minutes} 分钟，这个 cron 最短只隔 {max(1, int(shortest // 60))} 分钟。"
                + _cron_hint(limit_minutes),
                now, tz_name,
            )
    return expr, fires[0]


def parse_spec(
    in_minutes=None,
    run_at=None,
    cron=None,
    now: Optional[float] = None,
    tz_name: Optional[str] = None,
) -> ScheduleSpec:
    """把三选一的时间参数解析成一个规格；不合法抛 ScheduleError（中文，含当前业务时间）。

    in_minutes（1～525600，接受数字字符串与整数值的浮点，拒绝 bool）与 run_at（本地时间 YYYY-MM-DD HH:MM，
    允许 T 分隔与 :SS，必须晚于现在 10 秒以上且在 366 天内）得到一次性规格；cron（5 段）得到重复规格。
    0 和空字符串表示“没填”。now 默认取当前时刻；tz_name 默认业务时区，会原样记进规格。
    """
    now = time.time() if now is None else float(now)
    tz_name = (tz_name or config.DEFAULT_TIMEZONE).strip()
    given = [name for name, v in (("in_minutes", in_minutes), ("run_at", run_at), ("cron", cron)) if _given(v)]
    if not given:
        raise _err(
            "没有给出时间：in_minutes（N 分钟后）、run_at（本地时间 YYYY-MM-DD HH:MM）、cron（5 段表达式）三选一",
            now, tz_name,
        )
    if len(given) > 1:
        raise _err(f"in_minutes、run_at、cron 只能填一个，你同时填了 {' 和 '.join(given)}", now, tz_name)

    if given[0] == "in_minutes":
        minutes = _parse_minutes(in_minutes, now, tz_name)
        return ScheduleSpec("once", None, float(int(now) + minutes * 60), tz_name, f"in:{minutes}")
    if given[0] == "run_at":
        epoch = _parse_run_at(run_at, now, tz_name)
        return ScheduleSpec("once", None, epoch, tz_name, f"at:{int(epoch)}")
    expr, first = _parse_cron(cron, now, tz_name)
    return ScheduleSpec("cron", expr, first, tz_name, f"cron:{expr}")


# ── 计算 ───────────────────────────────────────────────────

def next_fire(cron: str, after: float, tz_name: Optional[str] = None) -> Optional[float]:
    """cron 在 after（epoch 秒）之后的下一次触发，严格晚于 after；没有下一次或表达式不合法返回 None。"""
    try:
        it = CronSim(cron, _at(after, tz_name))
        for _ in range(4):      # 防御：即便夏令时让它吐出不晚于 after 的值，也最多多看几个
            ts = next(it).timestamp()
            if ts > after:
                return ts
    except (CronSimError, StopIteration, ValueError, OverflowError, OSError):
        pass
    return None


# ── 描述 ───────────────────────────────────────────────────

def _int_list(field: str, low: int, high: int) -> Optional[list[int]]:
    """只含十进制整数、逗号分隔、都在 [low, high] 里 -> 升序去重列表；否则 None。"""
    if not re.fullmatch(r"[0-9]{1,2}(,[0-9]{1,2})*", field):
        return None
    values = sorted({int(p) for p in field.split(",")})
    return values if all(low <= v <= high for v in values) else None


def _dow_days(field: str) -> Optional[list[int]]:
    """星期字段 -> 升序的 1..7（周日为 7）；含步长 / # / L 等写法返回 None。"""
    days: set[int] = set()
    for token in field.split(","):
        bounds = token.split("-")
        if len(bounds) > 2:
            return None
        numbers = []
        for position, b in enumerate(bounds):
            if b == "SUN":      # 范围的起点按 0（SUN-TUE），其余按 7（FRI-SUN、单独的 SUN）
                numbers.append(0 if len(bounds) == 2 and position == 0 else 7)
            elif b in _DOW_NAMES:
                numbers.append(_DOW_NAMES[b])
            elif re.fullmatch(r"[0-7]", b):
                numbers.append(int(b))
            else:
                return None
        if len(numbers) == 2:
            if numbers[0] > numbers[1]:
                return None
            days.update(range(numbers[0], numbers[1] + 1))
        else:
            days.add(numbers[0])
    return sorted({7 if d == 0 else d for d in days})


def _describe_cron(cron: str) -> str:
    parts = cron.upper().split()
    if len(parts) != 5:
        return cron
    minute, hour, dom, month, dow = parts
    minutes = _int_list(minute, 0, 59)
    hours = _int_list(hour, 0, 23)
    if minutes and len(minutes) == 1 and hours:
        clock = "、".join(f"{h:02d}:{minutes[0]:02d}" for h in hours)
        if dom == "*" and month == "*":
            if dow == "*":
                return f"每天 {clock}"
            days = _dow_days(dow)
            if days:
                if len(days) == 7:
                    return f"每天 {clock}"
                if days == [1, 2, 3, 4, 5]:
                    return f"每周一至周五 {clock}"
                return "每周" + "、".join(_WEEKDAY_SHORT[d - 1] for d in days) + f" {clock}"
        elif dow == "*":
            dom_list = _int_list(dom, 1, 31)
            if dom_list and month == "*":
                return "每月 " + "、".join(str(d) for d in dom_list) + f" 日 {clock}"
            month_list = _int_list(month, 1, 12)
            if dom_list and month_list and len(dom_list) == 1 and len(month_list) == 1:
                return f"每年 {month_list[0]} 月 {dom_list[0]} 日 {clock}"
    elif dom == "*" and month == "*" and dow == "*":
        step = re.fullmatch(r"\*/([0-9]{1,2})", minute)
        if step and hour == "*":
            return f"每 {int(step.group(1))} 分钟"
        if minutes and len(minutes) == 1 and hour == "*":
            return f"每小时第 {minutes[0]} 分钟"
        hour_step = re.fullmatch(r"\*/([0-9]{1,2})", hour)
        if minutes and len(minutes) == 1 and hour_step:
            return f"每 {int(hour_step.group(1))} 小时（第 {minutes[0]} 分钟）"
    return cron


_JOB_STATUS_TEXT = {"active": "进行中", "done": "已完成"}
_JOB_PAUSE_TEXT = {
    "user": "你暂停的",
    "admin": "被管理员暂停",
    "push_refused": "推送被拒收，请确认没有关闭“允许主动发送消息”，再发 /tasks resume 恢复",
    "failures": "连续失败，问题解决后发 /tasks resume 恢复",
}
_JOB_LAST_TEXT = {"ok": "成功", "failed": "失败", "skipped": "已跳过", "missed": "错过了"}
_JOB_MODE_TEXT = {"reminder": "固定提醒", "agent": "到点执行"}


def job_status_text(job) -> str:
    """任务状态的一句话：进行中 / 已完成 / 已暂停（原因）。job 是任何带 status / paused_reason 属性的对象。"""
    if job.status == "paused":
        return "已暂停（" + _JOB_PAUSE_TEXT.get(job.paused_reason or "", "已暂停") + "）"
    return _JOB_STATUS_TEXT.get(job.status, job.status)


def format_job_line(index: int, job, *, who: str = "") -> str:
    """一行任务描述：序号、（谁）、标题、模式、编号前 8 位、状态（含暂停原因）、周期、下次时间、上次结果。
    job 是 schedule_store.Job 或任何带同名属性的对象（这里是 L0，不 import 存储层）。"""
    when = describe(job.kind, job.cron, job.next_run_at, job.timezone)
    owner = f"{who} " if who else ""
    parts = [
        f"{index}. {owner}「{job.title}」（{_JOB_MODE_TEXT.get(job.mode, job.mode)}）编号 {job.id[:8]}",
        job_status_text(job),
        when,
    ]
    if job.status == "active":
        parts.append(f"下次 {fmt_local(job.next_run_at, job.timezone)}")
    parts.append("上次：" + (_JOB_LAST_TEXT.get(job.last_status, job.last_status) if job.last_status else "还没执行过"))
    return "｜".join(parts)


def describe(kind: str, cron: Optional[str], next_run_at: Optional[float], tz_name: Optional[str] = None) -> str:
    """“每天 08:30”“每周一、三、五 09:00”“仅一次 11-27 08:30”；认不出的 cron 原样显示。"""
    if kind == "once":
        return "仅一次" if next_run_at is None else f"仅一次 {fmt_local(next_run_at, tz_name)}"
    return _describe_cron(cron or "")
