"""
tests/smoke_scheduler.py

定时任务与主动推送的离线冒烟测试（不联网、不碰真实 QQ / Modal / LLM；SQLite 都在临时目录里）：
  A1 时区 / 解析 / cron 校验 / next_fire / describe   A2 配置块与 import 卫生   A3 请求上下文 / require_owner / get_current_time
  B1 建库与单例   B2 创建 / 上限 / 去重   B3 列表与编号解析   B4 取消   B5 暂停 / 恢复 / 管理员暂停
  B6 认领（原子性 / 错过窗口 / cli 分流）   B7 收尾   B8 重启恢复   B9 变更监听器   B10 杂项
  C 调度循环（reminder 端到端 / 错过 / 唤醒 / 停机 / 并发 / 节流 / CLI 分流 / agent 成功）
  D 失败策略与错误分类   E 事件写入（幂等、不是入站事件、日志卫生）
所有时间断言都基于固定的 NOW（2026-11-26 08:30:00 +08:00，周四），不依赖真实的今天。
"""
import asyncio
import contextlib
import io
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app import config
from app.infra import schedule_store
from app.infra.schedule_store import (
    CLI_TARGET, JobLimitError, JobStateError, RefError, ScheduleStore,
)
from app.infra import session_store
from app.infra.session_store import SessionStore
from app.services import scheduler_service
from app.services.scheduler_service import AgentRunResult, PushError
from app.utils import schedule_time
from app.utils.schedule_time import ScheduleError

ROOT = Path(__file__).resolve().parents[1]
SH = "Asia/Shanghai"
NY = "America/New_York"
NOW = datetime(2026, 11, 26, 8, 30, 0, tzinfo=ZoneInfo(SH)).timestamp()       # 周四
U1 = "A1B2C3D4E5F6A7B8"        # member 的 user_id 就是 openid
U2 = "B1B2C3D4E5F6A7B8"
OWNER = "owner"
OPENID_OWNER = "C1B2C3D4E5F6A7B8"
DAY = 86400


def local_epoch(text: str, tz: str = SH) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(tz)).timestamp()


@contextlib.contextmanager
def patched(**values):
    """临时改 config 里的常量（调用时现读的那些），退出时还原。"""
    old = {k: getattr(config, k) for k in values}
    for k, v in values.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        return e
    raise AssertionError(f"应当抛出 {exc.__name__}")


class _Env:
    """临时目录里的 ScheduleStore；退出时先关连接（含 -wal / -shm）再删目录（Windows 要求）。"""

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="sched_")
        self.path = os.path.join(self.dir, "sub", "schedules.db")     # 父目录不存在，验证自动创建
        self.store = ScheduleStore(self.path)
        schedule_store.set_schedule_store(self.store)
        return self

    def reopen(self):
        self.store.close()
        self.store = ScheduleStore(self.path)
        schedule_store.set_schedule_store(self.store)
        return self.store

    def __exit__(self, *exc):
        schedule_store.close_schedule_store()
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def mk(store, *, user=U1, target=None, title="喝水", kind="cron", cron="30 8 * * *", next_run_at=None, mode="reminder",
       prompt="该喝水了", tz=SH, spec_key=None, now=NOW, max_jobs=10):
    """在库里造一个任务（直接走 create_job）。cron 任务默认下次是 NOW 之后的第一个 08:30。"""
    if kind == "once":
        cron = None
    if next_run_at is None:
        next_run_at = NOW + 3600 if kind == "once" else schedule_time.next_fire(cron, now, tz)
    if spec_key is None:
        spec_key = f"cron:{cron}" if kind == "cron" else f"at:{int(next_run_at)}"
    if target is None:
        target = OPENID_OWNER if user == OWNER else user
    job, created = store.create_job(
        user_id=user, target_openid=target, title=title, mode=mode, prompt=prompt, kind=kind, cron=cron,
        timezone=tz, next_run_at=next_run_at, spec_key=spec_key, max_jobs=max_jobs, now=now,
    )
    assert created, "mk 造任务时不应命中去重"
    return job


# ════════════════════════ A 时间 ════════════════════════

def test_a1_time():
    # —— get_tz：按名字缓存，无效名字回落并只警告一次 ——
    schedule_time.clear_tz_cache()
    assert schedule_time.get_tz(SH) is schedule_time.get_tz(SH)
    assert str(schedule_time.get_tz(NY)) == NY
    with patched(DEFAULT_TIMEZONE=NY):
        assert schedule_time.get_tz() is schedule_time.get_tz(NY), "name=None 现读 config.DEFAULT_TIMEZONE"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bad1 = schedule_time.get_tz("Not/AZone")
        bad2 = schedule_time.get_tz("Not/AZone")
        schedule_time.get_tz("../../etc/passwd")      # 路径穿越式的名字也只会回落
    assert bad1 is bad2 and buf.getvalue().count("回落到系统本地时区") == 2, buf.getvalue()   # 每个坏名字警告一次
    assert datetime.now(bad1).utcoffset() is not None

    # —— parse_spec：in_minutes ——
    s = schedule_time.parse_spec(in_minutes=30, now=NOW, tz_name=SH)
    assert (s.kind, s.cron, s.next_run_at, s.timezone, s.key) == ("once", None, NOW + 1800, SH, "in:30")
    for ok in ("30", " 30 ", 30.0, "30.0"):
        assert schedule_time.parse_spec(in_minutes=ok, now=NOW, tz_name=SH).next_run_at == NOW + 1800, ok
    assert schedule_time.parse_spec(in_minutes=1, now=NOW, tz_name=SH).next_run_at == NOW + 60
    assert schedule_time.parse_spec(in_minutes=525600, now=NOW, tz_name=SH).next_run_at == NOW + 525600 * 60
    for bad in (True, False, 0.5, "abc", "1e2", -5, 525601, "525601", [30], "３０"):    # bool 一律拒绝（含 False），全角数字也拒绝
        err = raises(ScheduleError, schedule_time.parse_spec, in_minutes=bad, now=NOW, tz_name=SH)
        assert "in_minutes" in str(err) and "当前业务时间 2026-11-26 08:30" in str(err) and "星期四" in str(err), err

    # —— 三选一 ——
    e = raises(ScheduleError, schedule_time.parse_spec, now=NOW, tz_name=SH)
    assert "三选一" in str(e)
    e = raises(ScheduleError, schedule_time.parse_spec, in_minutes=0, run_at="", cron="  ", now=NOW, tz_name=SH)
    assert "三选一" in str(e), "0 / 空串 / 空白都表示没填"
    e = raises(ScheduleError, schedule_time.parse_spec, in_minutes=5, cron="0 9 * * *", now=NOW, tz_name=SH)
    assert "只能填一个" in str(e) and "in_minutes" in str(e) and "cron" in str(e)

    # —— run_at ——
    s = schedule_time.parse_spec(run_at="2026-11-27 08:30", now=NOW, tz_name=SH)
    assert s.kind == "once" and s.next_run_at == local_epoch("2026-11-27 08:30") and s.key == f"at:{int(s.next_run_at)}"
    assert schedule_time.parse_spec(run_at="2026-11-27T08:30:15", now=NOW, tz_name=SH).next_run_at == s.next_run_at + 15
    assert schedule_time.parse_spec(run_at="2026-11-27 8:30", now=NOW, tz_name=SH).next_run_at == s.next_run_at
    near = NOW + 10
    ok_edge = schedule_time.parse_spec(run_at="2026-11-26 08:30:10", now=NOW, tz_name=SH)
    assert ok_edge.next_run_at == near, "恰好晚 10 秒允许"
    for bad in ("2026-11-26 08:30:09", "2026-11-26 08:00", "2025-01-01 08:00", "2027-12-01 08:00",
                "2026-11-27 08:30Z", "2026-11-27 08:30+08:00", "2026-11-27T08:30:00+0800", "明天早上八点",
                "2026-02-30 08:00", "2026-11-27", "2026-11-27 25:00", 20261127):
        raises(ScheduleError, schedule_time.parse_spec, run_at=bad, now=NOW, tz_name=SH)
    assert "时区" in str(raises(ScheduleError, schedule_time.parse_spec, run_at="2026-11-27 08:30+08:00", now=NOW, tz_name=SH))
    # 夏令时：2027-03-14 02:30 在纽约不存在；2027-11-07 01:30 出现两次，取第一次（-04:00）
    e = raises(ScheduleError, schedule_time.parse_spec, run_at="2027-03-14 02:30", now=NOW, tz_name=NY)
    assert "不存在" in str(e) and "夏令时" in str(e)
    amb = schedule_time.parse_spec(run_at="2027-11-07 01:30", now=NOW, tz_name=NY)
    assert datetime.fromtimestamp(amb.next_run_at, ZoneInfo(NY)).utcoffset().total_seconds() == -4 * 3600
    # 同一个时刻换时区，epoch 不同
    assert schedule_time.parse_spec(run_at="2026-11-27 08:30", now=NOW, tz_name=NY).next_run_at != s.next_run_at

    # —— cron：合法 ——
    s = schedule_time.parse_spec(cron="30 8 * * *", now=NOW, tz_name=SH)
    assert s.kind == "cron" and s.cron == "30 8 * * *" and s.next_run_at == NOW + DAY, "now 恰好是触发时刻时，下一次严格晚于 now"
    assert s.key == "cron:30 8 * * *"
    s = schedule_time.parse_spec(cron="0 9 * * 1,3,5", now=NOW, tz_name=SH)
    assert s.next_run_at == local_epoch("2026-11-27 09:00"), "周四之后的第一个周一三五是周五"
    s = schedule_time.parse_spec(cron="  0   9 *  * mon-fri ", now=NOW, tz_name=SH)
    assert s.cron == "0 9 * * MON-FRI", "空白折叠、名字大写"
    for ok in ("0 9 * jan *", "0 0 L * *", "0 0 * * 5L", "0 0 * * MON#5", "0 9 * * 7", "0 9 * * 0", "*/30 * * * *",
               "0 8,20 * * *", "0 9 15 * 1", "0-30/30 8 * * *", "0 0 LW * *"):
        schedule_time.parse_spec(cron=ok, now=NOW, tz_name=SH)
    # —— cron：拒绝 ——
    reasons = {
        "* * * * * *": "5 段", "0 30 8 * * *": "5 段", "0 0 0 1 1 * 2027": "5 段", "0 9 * *": "5 段",
        "@daily": "别名", "@hourly": "别名", "@reboot": "别名",
        "0 0 30 2 *": "日", "0 0 31 4 *": "日", "60 8 * * *": "分钟", "0 24 * * *": "小时", "0 0 * 13 *": "月",
        "0 9 * * 8": "星期", "0 9 * * ?": "星期", "0 9 ? * 1": "日", "5-1 * * * *": "分钟", "*/0 * * * *": "分钟",
        "0 9 * * *;": "字符", "0 9 * * * $(x)": "5 段", "0 9 * * `x`": "字符", "0 0 15W * *": "日",
        "0 0 29 2 *": "366 天",
    }
    for expr, word in reasons.items():
        e = raises(ScheduleError, schedule_time.parse_spec, cron=expr, now=NOW, tz_name=SH)
        assert word in str(e) and "当前业务时间" in str(e), (expr, str(e))
    raises(ScheduleError, schedule_time.parse_spec, cron="0 9 * * *" + " " * 5 + "x" * 200, now=NOW, tz_name=SH)
    raises(ScheduleError, schedule_time.parse_spec, cron=12345, now=NOW, tz_name=SH)
    # 最小间隔：校验会扫过午夜（23:55 -> 00:05 只隔 10 分钟）
    with patched(SCHEDULER_MIN_INTERVAL_MINUTES=30):
        for bad in ("*/5 * * * *", "* * * * *", "0,10 8 * * *", "5,55 0,23 * * *", "*/29 * * * *"):
            e = raises(ScheduleError, schedule_time.parse_spec, cron=bad, now=NOW, tz_name=SH)
            assert "30 分钟" in str(e) and "*/30 * * * *" in str(e), (bad, str(e))
        schedule_time.parse_spec(cron="*/30 * * * *", now=NOW, tz_name=SH)
        schedule_time.parse_spec(cron="0 8,9 * * *", now=NOW, tz_name=SH)
    with patched(SCHEDULER_MIN_INTERVAL_MINUTES=5):
        schedule_time.parse_spec(cron="*/5 * * * *", now=NOW, tz_name=SH)
        raises(ScheduleError, schedule_time.parse_spec, cron="*/4 * * * *", now=NOW, tz_name=SH)
    with patched(SCHEDULER_MIN_INTERVAL_MINUTES=120):
        e = raises(ScheduleError, schedule_time.parse_spec, cron="0 * * * *", now=NOW, tz_name=SH)
        assert "120 分钟" in str(e) and "0 8 * * *" in str(e)

    # —— next_fire：严格晚于；夏令时；不可能 / 非法 ——
    assert schedule_time.next_fire("30 8 * * *", NOW, SH) == NOW + DAY
    assert schedule_time.next_fire("30 8 * * *", NOW - 0.5, SH) == NOW, "晚于 08:29:59.5 的第一个触发是 08:30:00"
    assert schedule_time.next_fire("30 8 * * *", NOW + 0.5, SH) == NOW + DAY
    assert schedule_time.next_fire("0 0 29 2 *", NOW, SH) == local_epoch("2028-02-29 00:00")
    assert schedule_time.next_fire("not a cron", NOW, SH) is None
    assert schedule_time.next_fire("0 0 30 2 *", NOW, SH) is None
    ts = local_epoch("2026-03-07 12:00", NY)
    assert schedule_time.next_fire("30 2 * * *", ts, NY) == local_epoch("2026-03-08 03:00", NY), "缺失的小时：顺延到 03:00"
    assert schedule_time.next_fire("*/30 2 * * *", ts, NY) == local_epoch("2026-03-09 02:00", NY), "带步长的任务当天跳过"
    assert schedule_time.next_fire("30 2 * * *", ts, SH) == local_epoch("2026-03-08 02:30", SH), "上海没有夏令时"
    ts = local_epoch("2026-10-31 12:00", NY)
    first = schedule_time.next_fire("30 1 * * *", ts, NY)
    assert first == local_epoch("2026-11-01 01:30", NY)
    assert schedule_time.next_fire("30 1 * * *", first, NY) == first + DAY + 3600, "重复的小时：固定时刻的任务只触发一次"
    step_fires = []
    cursor = ts
    for _ in range(4):
        cursor = schedule_time.next_fire("*/30 1 * * *", cursor, NY)
        step_fires.append(cursor)
    assert step_fires[1] - step_fires[0] == 1800 and step_fires[2] - step_fires[1] == 1800, "带步长的任务在重复的小时里触发两遍"
    assert step_fires == sorted(set(step_fires)), "next_fire 的结果严格递增"

    # —— describe / fmt ——
    cases = {
        "30 8 * * *": "每天 08:30",
        "0 9 * * 1,3,5": "每周一、三、五 09:00",
        "0 9 * * mon,wed,fri": "每周一、三、五 09:00",
        "0 9 * * 1-5": "每周一至周五 09:00",
        "0 9 * * MON-FRI": "每周一至周五 09:00",
        "0 9 * * 0": "每周日 09:00",
        "0 9 * * 7": "每周日 09:00",
        "0 9 * * 6,0": "每周六、日 09:00",
        "0 9 * * 0-6": "每天 09:00",
        "0 8,20 * * *": "每天 08:00、20:00",
        "0 9 15 * *": "每月 15 日 09:00",
        "0 9 1,15 * *": "每月 1、15 日 09:00",
        "0 9 15 3 *": "每年 3 月 15 日 09:00",
        "*/15 * * * *": "每 15 分钟",
        "5 * * * *": "每小时第 5 分钟",
        "0 */2 * * *": "每 2 小时（第 0 分钟）",
        "0 9 15 * 1": "0 9 15 * 1",
        "0 0 L * *": "0 0 L * *",
        "0 0 * * MON#5": "0 0 * * MON#5",
    }
    for expr, want in cases.items():
        got = schedule_time.describe("cron", expr, NOW, SH)
        assert got == want, (expr, got, want)
    assert schedule_time.describe("once", None, local_epoch("2026-11-27 08:30"), SH) == "仅一次 11-27 08:30"
    assert schedule_time.fmt_local(NOW, SH) == "11-26 08:30"
    assert schedule_time.fmt_local(NOW, NY) == "11-25 19:30"
    assert "星期四" in schedule_time.fmt_now(NOW, SH) and "2026-11-26 08:30" in schedule_time.fmt_now(NOW, SH)


def test_a2_config_and_import_hygiene():
    """配置块的夹紧 / 回退，以及 import 时不建库、不读时区库（都在子进程里测，环境变量可控）。"""
    def probe(env_extra: dict, code: str) -> str:
        env = dict(os.environ)
        for k in [k for k in env if k.startswith("SCHEDULER_") or k in ("DEFAULT_TIMEZONE", "SCHEDULES_DB_PATH")]:
            env.pop(k)
        env.update(env_extra)
        res = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, %r)\n%s" % (str(ROOT), code)],
                             env=env, cwd=str(ROOT), capture_output=True, text=True)
        assert res.returncode == 0, f"子进程失败:\n{res.stdout}\n{res.stderr}"
        return res.stdout.strip()

    out = probe({}, "from app import config as c; print(c.SCHEDULER_ENABLED, c.DEFAULT_TIMEZONE, c.SCHEDULER_MAX_JOBS_OWNER,"
                    " c.SCHEDULER_MAX_JOBS_MEMBER, c.SCHEDULER_MIN_INTERVAL_MINUTES, c.SCHEDULER_MISFIRE_GRACE_SECONDS,"
                    " c.SCHEDULER_LOCK_WAIT_SECONDS, c.SCHEDULER_MAX_CONCURRENT_RUNS, c.SCHEDULER_PUSH_MAX_CHARS,"
                    " c.SCHEDULER_PUSH_MIN_INTERVAL_SECONDS, c.SCHEDULER_MAX_CONSECUTIVE_FAILURES, c.SCHEDULER_RETRY_MAX,"
                    " c.SCHEDULER_RETRY_DELAY_SECONDS, c.SCHEDULES_DB_PATH.replace(chr(92), '/').endswith('storage/schedules.db'))")
    assert out == "False Asia/Shanghai 30 5 30 900 60 2 1500 0.5 3 3 30 True", out
    # 开关写法：1 / true / yes（不分大小写）为真，其余为假
    for raw, want in (("1", True), ("True", True), (" YES ", True), ("0", False), ("no", False), ("", False), ("enable", False)):
        got = probe({"SCHEDULER_ENABLED": raw}, "from app import config as c; print(c.SCHEDULER_ENABLED)")
        assert got == str(want), (raw, got)
    # 写错的值回退，越界值夹紧，import 不崩
    out = probe(
        {"DEFAULT_TIMEZONE": "  ", "SCHEDULER_MAX_JOBS_OWNER": "abc", "SCHEDULER_MAX_JOBS_MEMBER": "-3",
         "SCHEDULER_MIN_INTERVAL_MINUTES": "0", "SCHEDULER_MISFIRE_GRACE_SECONDS": "5",
         "SCHEDULER_LOCK_WAIT_SECONDS": "999999", "SCHEDULER_MAX_CONCURRENT_RUNS": "0",
         "SCHEDULER_PUSH_MAX_CHARS": "99999", "SCHEDULER_PUSH_MIN_INTERVAL_SECONDS": "-1",
         "SCHEDULER_MAX_CONSECUTIVE_FAILURES": "x", "SCHEDULER_RETRY_MAX": "-1", "SCHEDULER_RETRY_DELAY_SECONDS": "0"},
        "from app import config as c; print(c.DEFAULT_TIMEZONE, c.SCHEDULER_MAX_JOBS_OWNER, c.SCHEDULER_MAX_JOBS_MEMBER,"
        " c.SCHEDULER_MIN_INTERVAL_MINUTES, c.SCHEDULER_MISFIRE_GRACE_SECONDS, c.SCHEDULER_LOCK_WAIT_SECONDS,"
        " c.SCHEDULER_MAX_CONCURRENT_RUNS, c.SCHEDULER_PUSH_MAX_CHARS, c.SCHEDULER_PUSH_MIN_INTERVAL_SECONDS,"
        " c.SCHEDULER_MAX_CONSECUTIVE_FAILURES, c.SCHEDULER_RETRY_MAX, c.SCHEDULER_RETRY_DELAY_SECONDS)")
    assert out == "Asia/Shanghai 30 5 1 30 3600 1 4000 0.5 3 0 1", out
    out = probe({"SCHEDULER_MIN_INTERVAL_MINUTES": "99999", "SCHEDULER_MAX_JOBS_OWNER": "99999"},
                "from app import config as c; print(c.SCHEDULER_MIN_INTERVAL_MINUTES, c.SCHEDULER_MAX_JOBS_OWNER)")
    assert out == "1440 1000", out

    # import 卫生：不建库、不读时区库、不带进受限模块
    tmp = tempfile.mkdtemp(prefix="sched_imp_")
    try:
        db = os.path.join(tmp, "nope", "schedules.db")
        out = probe(
            {"SCHEDULES_DB_PATH": db},
            "import sys, os, threading\n"
            "before = threading.active_count()\n"
            "import app.utils.schedule_time as t, app.infra.schedule_store as s, app.tools.knowledge_tools as k\n"
            "assert not t._tz_cache, 'import 时不得创建时区对象'\n"
            "assert not os.path.exists(%r) and not os.path.exists(os.path.dirname(%r)), 'import 时不得建库'\n"
            "assert threading.active_count() == before, 'import 时不得起线程'\n"
            "bad = [m for m in ('app.infra.models', 'app.infra.milvus_client', 'modal') if m in sys.modules]\n"
            "assert not bad or 'app.tools.knowledge_tools' in sys.modules, bad\n"
            "print('ok')" % (db, db),
        )
        assert out == "ok", out
        out = probe(
            {"SCHEDULES_DB_PATH": db},
            "import sys\nimport app.utils.schedule_time, app.infra.schedule_store\n"
            "bad = [m for m in ('app.infra.models', 'app.infra.milvus_client', 'app.knowledge', 'app.memory', 'modal') if m in sys.modules]\n"
            "assert not bad, bad\nprint('ok')",
        )
        assert out == "ok", out
        assert not os.path.exists(db)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a3_context_and_current_time():
    """RequestContext 的新字段只在末尾且有默认值；scheduled 上下文下 require_owner 一律拒绝；get_current_time 用业务时区。"""
    from app.tools import knowledge_tools
    from app.utils import context
    from app.utils.context import RequestContext

    old = RequestContext("owner", "sid", True)                     # 旧的按位置构造
    assert (old.raw_id, old.is_scheduled) == ("", False)
    assert RequestContext("u", "s", False, "rawid", True).is_scheduled is True
    assert [f for f in RequestContext.__dataclass_fields__][:3] == ["user_id", "session_id", "is_owner"], "已有字段的顺序不变"
    raises(Exception, setattr, old, "is_scheduled", True)           # frozen

    with context.request_context(RequestContext("owner", "s", True)):
        assert context.require_owner().user_id == "owner"
    with context.request_context(RequestContext("owner", "s", True, "C1B2C3D4E5F6A7B8", False)):
        assert context.require_owner().raw_id == "C1B2C3D4E5F6A7B8"
    with context.request_context(RequestContext("owner", "s", True, "C1B2C3D4E5F6A7B8", True)):
        e = raises(PermissionError, context.require_owner)
        assert "无人值守" in str(e), "owner 的定时任务也拒绝"
    with context.request_context(RequestContext(U1, "s", False)):
        assert "仅管理员可用" in str(raises(PermissionError, context.require_owner))
    raises(RuntimeError, context.require_owner)                      # 没有上下文：失败即关闭

    # get_current_time：格式不变，时区取业务时区
    import re
    pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} 星期[一二三四五六日] \(UTC[+-]\d{2}:\d{2}\)")
    for tz_name, offset in ((SH, "+08:00"), (NY, None)):
        with patched(DEFAULT_TIMEZONE=tz_name):
            schedule_time.clear_tz_cache()
            out = knowledge_tools.get_current_time.invoke({})
            assert pattern.fullmatch(out), out
            if offset:
                assert out.endswith(f"(UTC{offset})"), out
            else:
                assert out.endswith("(UTC-05:00)") or out.endswith("(UTC-04:00)"), out
    schedule_time.clear_tz_cache()
    assert "业务时区" in knowledge_tools.get_current_time.description and "本机时区" not in knowledge_tools.get_current_time.description


# ════════════════════════ B 存储 ════════════════════════

def test_b1_open_and_singleton():
    with _Env() as env:
        s = env.store
        assert os.path.exists(env.path) and s.schema_version() == 1
        assert s._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        tables = {r["name"] for r in s._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables >= {"jobs", "runs", "meta"}, tables
        job = mk(s)
        s2 = env.reopen()
        assert s2.get_job(job.id) == job, "关闭重开后数据完整"
        assert schedule_store.get_schedule_store() is s2
        # schema 版本高于当前代码支持的：拒绝打开，改回去后又能打开
        s2.close()
        raw = sqlite3.connect(env.path)
        raw.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        raw.commit()
        assert "99" in str(raises(RuntimeError, ScheduleStore, env.path))
        raw.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
        raw.commit()
        raw.close()
        assert env.reopen().get_job(job.id) == job
    # 单例：惰性创建、按 config 路径、close 之后可重建
    tmp = tempfile.mkdtemp(prefix="sched_single_")
    try:
        path = os.path.join(tmp, "x", "schedules.db")
        schedule_store.close_schedule_store()
        with patched(SCHEDULES_DB_PATH=path):
            assert not os.path.exists(path), "没人要之前不建库"
            got = schedule_store.get_schedule_store()
            assert got is schedule_store.get_schedule_store() and os.path.exists(path)
            schedule_store.close_schedule_store()
            assert got._closed
            assert schedule_store.get_schedule_store() is not got
    finally:
        schedule_store.close_schedule_store()
        shutil.rmtree(tmp, ignore_errors=True)


def test_b2_create_limit_dedupe():
    with _Env() as env:
        s = env.store
        a = mk(s, user=U1, title="A", max_jobs=2)
        assert a.status == "active" and a.user_id == U1 and a.target_openid == U1 and a.run_count == 0
        assert a.next_run_at == NOW + DAY and a.created_at == NOW and a.paused_reason is None and len(a.id) == 32
        b = mk(s, user=U1, title="B", max_jobs=2)
        e = raises(JobLimitError, mk, s, user=U1, title="C", max_jobs=2)
        assert "2 个" in str(e) and "上限" in str(e)
        assert s.count_user_jobs(U1) == 2 and s.list_jobs(U1) == [a, b]
        assert s.list_jobs(U2) == [], "别的用户不受影响"
        mk(s, user=U2, title="A", max_jobs=2)
        # paused 占名额，done 不占
        s.pause_job(a.id)
        raises(JobLimitError, mk, s, user=U1, title="C", max_jobs=2)
        once = mk(s, user=U2, title="一次", kind="once", max_jobs=3)
        assert s.count_user_jobs(U2) == 2
        claims, _ = s.claim_due(once.next_run_at, cli=False)
        assert [c.job.id for c in claims] == [once.id]
        assert s.get_job(once.id).status == "done" and s.count_user_jobs(U2) == 1
        mk(s, user=U2, title="D", max_jobs=2)

        # —— 重复创建保护：5 分钟内 user + title + prompt + mode + 时间规格完全相同 -> 返回已有任务 ——
        base = mk(s, user=OWNER, title="吃药", prompt="吃药", cron="0 9 * * *")

        def create(**over):
            args = dict(user_id=OWNER, target_openid=OPENID_OWNER, title="吃药", mode="reminder", prompt="吃药", kind="cron",
                        cron="0 9 * * *", timezone=SH, next_run_at=base.next_run_at, spec_key="cron:0 9 * * *",
                        max_jobs=50, now=NOW + 10)
            args.update(over)
            return s.create_job(**args)

        assert create(now=NOW + 299) == (base, False), "5 分钟内完全相同：返回已有任务"
        for over in (dict(title="吃药2"), dict(prompt="吃药 "), dict(mode="agent"),
                     dict(spec_key="cron:0 10 * * *", cron="0 10 * * *"), dict(user_id=U2, target_openid=U2)):
            job, created = create(**over)
            assert created and job.id != base.id, over
        fresh, created = create(now=NOW + 301)
        assert created and fresh.id != base.id, "超过 5 分钟就是新任务"
        s.pause_job(fresh.id)
        assert create(now=NOW + 320) == (s.get_job(fresh.id), False), "暂停中的任务也算已有任务"
        s.cancel_job(fresh.id)
        again_job, created = create(now=NOW + 330)
        assert created and again_job.id not in (base.id, fresh.id), "已取消的不算（base 已在 5 分钟窗口之外）"
        # 额度已满时重复创建仍然是“返回已有”，而不是报超限
        full = mk(s, user=U1, title="满", prompt="满", cron="0 7 * * *", max_jobs=10)
        n = s.count_user_jobs(U1)
        again = s.create_job(user_id=U1, target_openid=U1, title="满", mode="reminder", prompt="满", kind="cron", cron="0 7 * * *",
                             timezone=SH, next_run_at=full.next_run_at, spec_key="cron:0 7 * * *", max_jobs=n, now=NOW + 5)
        assert again == (full, False)

        # —— 参数校验：失败即关闭 ——
        good = dict(user_id=U1, target_openid=U1, title="x", mode="reminder", prompt="x", kind="cron", cron="30 8 * * *",
                    timezone=SH, next_run_at=NOW + 1, spec_key="k", max_jobs=99, now=NOW)
        for over in (dict(user_id="bad id"), dict(user_id="cli-x", target_openid="bad id"), dict(target_openid="x"),
                     dict(target_openid="a" * 65), dict(mode="shell"), dict(kind="weekly"), dict(cron=None),
                     dict(kind="once", cron="30 8 * * *")):
            raises(ValueError, s.create_job, **{**good, **over})
        cli_job, created = s.create_job(**{**good, "target_openid": CLI_TARGET, "user_id": OWNER, "title": "cli任务"})
        assert created and cli_job.is_cli and cli_job.short_id == cli_job.id[:8]

    # —— 上限检查与插入在同一个事务里：几个连接同时创建，恰好只有 max_jobs 个成功 ——
    with _Env() as env:
        stores = [env.store] + [ScheduleStore(env.path) for _ in range(3)]
        outcomes: list[str] = []
        barrier = threading.Barrier(len(stores) * 2)

        def worker(store, n):
            barrier.wait()
            try:
                store.create_job(user_id=U1, target_openid=U1, title=f"并发{n}", mode="reminder", prompt="x", kind="cron",
                                 cron="30 8 * * *", timezone=SH, next_run_at=NOW + 1, spec_key=f"k{n}", max_jobs=3, now=NOW)
                outcomes.append("ok")
            except JobLimitError:
                outcomes.append("limit")

        threads = [threading.Thread(target=worker, args=(stores[n % len(stores)], n)) for n in range(len(stores) * 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert outcomes.count("ok") == 3 and outcomes.count("limit") == 5, outcomes
        assert env.store.count_user_jobs(U1) == 3
        for st in stores[1:]:
            st.close()


def test_b3_list_and_resolve():
    with _Env() as env:
        s = env.store
        jobs = [mk(s, user=U1, title=f"任务{i}", cron=f"{i} 8 * * *") for i in range(1, 4)]
        other = mk(s, user=U2, title="别人的")
        owner_job = mk(s, user=OWNER, title="主人的")
        assert [j.id for j in s.list_jobs(U1)] == [j.id for j in jobs]
        assert [j.title for j in s.list_jobs()] == ["任务1", "任务2", "任务3", "别人的", "主人的"], "None = 全部，按创建顺序"
        assert len(s.list_jobs(U1, limit=2)) == 2
        # 序号
        assert s.resolve("1", U1).id == jobs[0].id and s.resolve(" 3 ", U1).id == jobs[2].id
        for bad in ("0", "4", "999"):
            e = raises(RefError, s.resolve, bad, U1)
            assert "一共有 3 个" in str(e) or "没有第" in str(e)
        # id 前缀：≥4 位，大小写不敏感，只在自己的任务里
        j = jobs[1]
        assert s.resolve(j.id[:4], U1).id == j.id and s.resolve(j.id[:8].upper(), U1).id == j.id and s.resolve(j.id, U1).id == j.id
        for bad in ("", "  ", "ab", "xyz!", j.id[:3] + "g", "a" * 33):
            raises(RefError, s.resolve, bad, U1)
        e = raises(RefError, s.resolve, other.id[:6], U1)
        assert "没有找到" in str(e), "member 看不到别人的任务，错误信息也不泄露存在性"
        assert s.resolve(other.id[:6], OWNER, is_owner=True).id == other.id, "owner 可以跨用户用 id 前缀"
        raises(RefError, s.resolve, other.id[:6], OWNER, False)
        assert s.resolve("1", OWNER, True).id == owner_job.id, "序号永远指自己的列表"
        # 已完成的排在后面，序号随之移动
        once = mk(s, user=U1, title="一次性", kind="once", next_run_at=NOW + 5)
        assert s.list_jobs(U1)[-1].id == once.id
        s.claim_due(NOW + 5, cli=False)
        assert [x.title for x in s.list_jobs(U1)] == ["任务1", "任务2", "任务3", "一次性"] and s.list_jobs(U1)[-1].status == "done"
        mk(s, user=U1, title="新的", cron="5 9 * * *", now=NOW + 6)
        assert [x.title for x in s.list_jobs(U1)] == ["任务1", "任务2", "任务3", "新的", "一次性"], "done 永远排在最后"

    # 前缀冲突：伪造 id 让两个任务共享前 6 位
    with _Env() as env:
        s = env.store
        ids = iter(["abcdef" + "0" * 26, "abcdef" + "1" * 26, "abc123" + "2" * 26])

        class _Fake:
            def __init__(self, hex_):
                self.hex = hex_
        original = schedule_store.uuid.uuid4
        schedule_store.uuid.uuid4 = lambda: _Fake(next(ids))
        try:
            a = mk(s, user=U1, title="a", cron="1 8 * * *")
            b = mk(s, user=U1, title="b", cron="2 8 * * *")
            c = mk(s, user=U1, title="c", cron="3 8 * * *")
        finally:
            schedule_store.uuid.uuid4 = original
        e = raises(RefError, s.resolve, "abcdef", U1)
        assert "2 个" in str(e)
        assert s.resolve("abcdef0", U1).id == a.id and s.resolve("abcdef1", U1).id == b.id and s.resolve("abc12", U1).id == c.id
        raises(RefError, s.resolve, "abcd", U1)        # abcdef0… 与 abcdef1… 都以 abcd 开头
        assert s.resolve("abc1", U1).id == c.id
        # 纯数字：≤3 位按序号，≥4 位按 id 前缀
        raises(RefError, s.resolve, "1234", U1)


def test_b4_cancel():
    with _Env() as env:
        s = env.store
        a = mk(s, user=U1, title="a")
        b = mk(s, user=U2, title="b")
        claims, _ = s.claim_due(a.next_run_at, cli=False)
        assert len(claims) == 2 and len(s.get_runs(a.id)) == 1
        assert s.cancel_job(a.id, user_id=U2) is False and s.get_job(a.id) is not None, "不能删别人的任务"
        assert s.cancel_job("0" * 32) is False
        assert s.cancel_job(a.id, user_id=U1) is True
        assert s.get_job(a.id) is None and s.get_runs(a.id) == [] and s.get_run(claims[0].run_id) is None, "取消 = 删除任务及其 run"
        assert s.get_job(b.id) is not None and len(s.get_runs(b.id)) == 1
        assert s.cancel_job(b.id) is True and s.cancel_job(b.id) is False
        raises(ValueError, s.cancel_job, b.id, user_id="bad id")


def test_b5_pause_resume():
    with _Env() as env:
        s = env.store
        cron_job = mk(s, user=U1, title="c", cron="0 9 * * *")
        once_job = mk(s, user=U1, title="o", kind="once", next_run_at=NOW + 3600)
        # 暂停 / 恢复
        p = s.pause_job(cron_job.id, "user", user_id=U1)
        assert (p.status, p.paused_reason) == ("paused", "user")
        assert s.pause_job(cron_job.id, "user").paused_reason == "user", "重复暂停无副作用"
        raises(JobStateError, s.pause_job, cron_job.id, "user", user_id=U2)
        raises(JobStateError, s.pause_job, "0" * 32)
        raises(ValueError, s.pause_job, cron_job.id, "whatever")
        later = NOW + 5 * DAY + 100
        r = s.resume_job(cron_job.id, now=later, user_id=U1)
        assert r.status == "active" and r.paused_reason is None and r.consecutive_failures == 0
        assert r.next_run_at == schedule_time.next_fire("0 9 * * *", later, SH) and r.next_run_at > later, "cron 从现在重算下次"
        raises(JobStateError, s.resume_job, cron_job.id, now=later)       # 没在暂停
        raises(JobStateError, s.resume_job, cron_job.id, now=later, user_id=U2)
        # 一次性任务：没过期可恢复，过期拒绝
        s.pause_job(once_job.id)
        assert s.resume_job(once_job.id, now=NOW + 10).next_run_at == once_job.next_run_at
        s.pause_job(once_job.id)
        e = raises(JobStateError, s.resume_job, once_job.id, now=NOW + 7200)
        assert "重新创建" in str(e) and s.get_job(once_job.id).status == "paused"
        # 管理员暂停：member 不能恢复，owner 可以；user 的暂停可被 admin 升级
        s.resume_job(once_job.id, now=NOW + 10)
        assert s.pause_job(cron_job.id, "admin").paused_reason == "admin"
        e = raises(JobStateError, s.resume_job, cron_job.id, now=later, is_owner=False)
        assert "管理员" in str(e) and s.get_job(cron_job.id).status == "paused"
        assert s.resume_job(cron_job.id, now=later, is_owner=True).status == "active"
        s.pause_job(cron_job.id, "user")
        assert s.pause_job(cron_job.id, "admin").paused_reason == "admin", "已暂停的任务被 admin 升级"
        assert s.pause_job(cron_job.id, "user").paused_reason == "admin", "反过来不降级"
        # push_refused / failures 暂停：member 可以自己恢复
        s.resume_job(cron_job.id, now=later, is_owner=True)
        s.pause_job(cron_job.id, "push_refused")
        assert s.resume_job(cron_job.id, now=later).status == "active"
        # done 的任务
        done = mk(s, user=U2, title="d", kind="once", next_run_at=NOW + 5)
        s.claim_due(NOW + 5, cli=False)
        for fn in (lambda: s.pause_job(done.id), lambda: s.resume_job(done.id, now=NOW + 6)):
            raises(JobStateError, fn)
        # pause_user_jobs
        a = mk(s, user=U2, title="a", cron="1 8 * * *")
        b = mk(s, user=U2, title="b", cron="2 8 * * *")
        s.pause_job(b.id, "user")
        c = mk(s, user=U1, title="别人", cron="3 8 * * *")
        assert s.pause_user_jobs(U2, "admin") == 1, "只暂停进行中的"
        assert s.get_job(a.id).paused_reason == "admin" and s.get_job(b.id).paused_reason == "user"
        assert s.get_job(c.id).status == "active", "别的用户不受影响"
        assert s.pause_user_jobs(U2, "admin") == 0
        raises(ValueError, s.pause_user_jobs, U2, "nope")
        # /unblock 不自动恢复：库里没有任何自动恢复的路径
        assert s.get_job(a.id).status == "paused"


def test_b6_claim_due():
    with _Env() as env:
        s = env.store
        cron_job = mk(s, user=U1, title="c", cron="30 8 * * *")          # 下次 11-27 08:30
        once_job = mk(s, user=U2, title="o", kind="once", next_run_at=NOW + 100)
        later = mk(s, user=U1, title="later", kind="once", next_run_at=NOW + 5000)
        paused = mk(s, user=U1, title="p", kind="once", next_run_at=NOW + 100)
        s.pause_job(paused.id)
        assert s.next_due_at(cli=False) == NOW + 100, "paused 不算，done 不算"
        # 没到点：什么都不领
        assert s.claim_due(NOW + 50, cli=False) == ([], [])
        # 到点：一次性 -> done，cron -> 推进到严格晚于 now 的下一次；run 记为 claimed
        claims, missed = s.claim_due(NOW + 100, cli=False)
        assert missed == [] and [c.job.id for c in claims] == [once_job.id]
        c = claims[0]
        assert c.scheduled_for == NOW + 100 and c.job.next_run_at == NOW + 100
        run = s.get_run(c.run_id)
        assert (run.job_id, run.status, run.scheduled_for, run.started_at, run.finished_at) == (once_job.id, "claimed", NOW + 100, NOW + 100, None)
        j = s.get_job(once_job.id)
        assert j.status == "done" and j.next_run_at == NOW + 100
        assert s.claim_due(NOW + 100, cli=False) == ([], []), "原子：同一次到点只认领一次"
        assert s.claim_due(NOW + 4000, cli=False) == ([], [])
        # cron 到点（later 的一次性任务此时已经远远超过宽限，会被判为错过；错过的另在下面单独断言）
        t = cron_job.next_run_at
        claims, missed = s.claim_due(t + 2, cli=False)
        assert [x.job.id for x in claims] == [cron_job.id] and claims[0].scheduled_for == t
        assert [x.id for x in missed] == [later.id]
        j = s.get_job(cron_job.id)
        assert j.status == "active" and j.next_run_at == t + DAY, "认领时就推进，不等执行完"
        assert s.next_due_at(cli=False) == t + DAY

    with _Env() as env:
        s = env.store
        o = mk(s, user=U1, title="o", kind="once", next_run_at=NOW + 100)
        c = mk(s, user=U2, title="c", cron="0 9 * * *")
        t = c.next_run_at
        # 恰好在宽限内（含边界）算按时，超过一秒算错过
        claims, missed = s.claim_due(NOW + 100 + 900, cli=False, grace=900)
        assert [x.job.id for x in claims] == [o.id] and missed == []
        far = t + 901
        claims, missed = s.claim_due(far, cli=False, grace=900)
        assert claims == [] and missed == [], "cron 错过：静默推进，不进 missed_once"
        j = s.get_job(c.id)
        assert j.next_run_at == schedule_time.next_fire("0 9 * * *", far, SH) and j.next_run_at > far and j.last_status == "missed"
        assert s.get_runs(c.id)[0].status == "missed" and s.get_runs(c.id)[0].finished_at == far
        m = mk(s, user=U1, title="m", kind="once", next_run_at=far + 100)
        claims, missed = s.claim_due(far + 100 + 901, cli=False, grace=900)
        assert claims == [] and [x.id for x in missed] == [m.id]
        j = s.get_job(m.id)
        assert j.status == "done" and j.last_status == "missed" and s.get_runs(m.id)[0].status == "missed"
        assert s.claim_due(far + 5000, cli=False, grace=900) == ([], []), "错过的任务不会再被领一次"
        # 默认宽限读 config
        m2 = mk(s, user=U1, title="m2", kind="once", next_run_at=far + 6000)
        with patched(SCHEDULER_MISFIRE_GRACE_SECONDS=60):
            claims, missed = s.claim_due(far + 6000 + 61, cli=False)
        assert claims == [] and [x.id for x in missed] == [m2.id]

    # cli 分流：两个调度器互不“偷”任务
    with _Env() as env:
        s = env.store
        q = mk(s, user=U1, title="qq", kind="once", next_run_at=NOW + 10)
        k = mk(s, user=OWNER, title="cli", kind="once", next_run_at=NOW + 10, target=CLI_TARGET)
        assert s.next_due_at(cli=True) == NOW + 10 and s.next_due_at(cli=False) == NOW + 10
        claims, _ = s.claim_due(NOW + 10, cli=True)
        assert [x.job.id for x in claims] == [k.id]
        assert s.get_job(q.id).status == "active", "CLI 调度器没有碰 QQ 用户的任务"
        claims, _ = s.claim_due(NOW + 10, cli=False)
        assert [x.job.id for x in claims] == [q.id]
        assert s.next_due_at(cli=True) is None and s.next_due_at(cli=False) is None

    # limit 与顺序：先到点的先领
    with _Env() as env:
        s = env.store
        ids = [mk(s, user=U1, title=f"j{i}", kind="once", next_run_at=NOW + 10 + i).id for i in range(5)]
        claims, _ = s.claim_due(NOW + 100, cli=False, limit=2)
        assert [x.job.id for x in claims] == ids[:2]
        claims, _ = s.claim_due(NOW + 100, cli=False, limit=10)
        assert [x.job.id for x in claims] == ids[2:]

    # UNIQUE(job_id, scheduled_for) 是最后一道保险：已有同一时刻的 run，就只推进、不重复执行
    with _Env() as env:
        s = env.store
        job = mk(s, user=U1, title="u", cron="0 9 * * *")
        s._conn.execute("INSERT INTO runs(job_id, scheduled_for, started_at, status) VALUES (?,?,?, 'ok')",
                        (job.id, job.next_run_at, NOW))
        claims, missed = s.claim_due(job.next_run_at + 1, cli=False)
        assert claims == [] and missed == []
        assert s.get_job(job.id).next_run_at == job.next_run_at + DAY, "仍然推进了，不会卡在原地反复尝试"
        assert len(s.get_runs(job.id)) == 1
        raises(Exception, s._conn.execute, "INSERT INTO runs(job_id, scheduled_for, started_at, status) VALUES (?,?,?, 'ok')",
               (job.id, job.next_run_at, NOW))

    # 多个连接 / 线程同时认领：每个任务恰好被领一次
    with _Env() as env:
        stores = [env.store] + [ScheduleStore(env.path) for _ in range(3)]
        n = 12
        want = {mk(env.store, user=U1, title=f"并发{i}", kind="once", next_run_at=NOW + 10, spec_key=f"k{i}", max_jobs=50).id
                for i in range(n)}
        got: list[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(len(stores))

        def grab(store):
            barrier.wait()
            while True:
                claims, _ = store.claim_due(NOW + 10, cli=False, limit=3)
                if not claims:
                    return
                with lock:
                    got.extend(c.job.id for c in claims)
        threads = [threading.Thread(target=grab, args=(st,)) for st in stores]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(got) == sorted(want) and len(got) == n, "每个任务恰好被认领一次"
        runs = env.store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert runs == n
        for st in stores[1:]:
            st.close()

    # cron 的 run 历史只留最近 20 条
    with _Env() as env:
        s = env.store
        job = mk(s, user=U1, title="h", cron="0 * * * *")
        with patched(SCHEDULER_MISFIRE_GRACE_SECONDS=86400):
            t = job.next_run_at
            for _ in range(30):
                claims, _ = s.claim_due(t, cli=False)
                assert len(claims) == 1
                s.finish_run(claims[0].run_id, "ok", now=t)
                t = s.get_job(job.id).next_run_at
        runs = s.get_runs(job.id)
        assert len(runs) == 20 and runs[-1].scheduled_for == t - 3600, len(runs)


def test_b7_finish_run():
    with _Env() as env:
        s = env.store
        job = mk(s, user=U1, title="f", cron="0 9 * * *")

        def fire(job_id):
            t = s.get_job(job_id).next_run_at
            claims, _ = s.claim_due(t, cli=False)
            assert len(claims) == 1
            return claims[0].run_id, t
        run_id, t = fire(job.id)
        j = s.finish_run(run_id, "ok", now=t + 5)
        assert (j.last_status, j.last_run_at, j.run_count, j.consecutive_failures, j.status) == ("ok", t + 5, 1, 0, "active")
        r = s.get_run(run_id)
        assert (r.status, r.finished_at, r.error) == ("ok", t + 5, None)
        # 只有 claimed 的 run 会被收尾：第二次收尾（如 stop() 已写成 cancelled）不重复计数
        j2 = s.finish_run(run_id, "failed", error="late", now=t + 9)
        assert j2.run_count == 1 and j2.last_status == "ok" and s.get_run(run_id).status == "ok"
        # 失败：计数 +1；ok 清零；count_failure=False（busy / stopped）不计
        run_id, t = fire(job.id)
        j = s.finish_run(run_id, "failed", error="x" * 500, now=t + 1, max_failures=3)
        assert j.consecutive_failures == 1 and j.run_count == 2 and j.last_status == "failed" and j.status == "active"
        assert len(s.get_run(run_id).error) == 200, "error 截断"
        run_id, t = fire(job.id)
        j = s.finish_run(run_id, "failed", error="busy", count_failure=False, now=t + 1, max_failures=3)
        assert j.consecutive_failures == 1 and j.run_count == 3
        run_id, t = fire(job.id)
        j = s.finish_run(run_id, "skipped", now=t + 1, max_failures=3)
        assert j.consecutive_failures == 1 and j.run_count == 3 and j.last_status == "skipped", "skipped 不计数也不清零"
        run_id, t = fire(job.id)
        j = s.finish_run(run_id, "ok", now=t + 1, max_failures=3)
        assert j.consecutive_failures == 0
        # 连续失败达到上限 -> 暂停(failures)
        for i in range(3):
            run_id, t = fire(job.id)
            j = s.finish_run(run_id, "failed", error="boom", now=t + 1, max_failures=3)
        assert (j.status, j.paused_reason, j.consecutive_failures) == ("paused", "failures", 3)
        assert s.next_due_at(cli=False) is None
        j = s.resume_job(job.id, now=t + 100)
        assert j.status == "active" and j.consecutive_failures == 0
        # pause_reason：推送被拒
        run_id, t = fire(job.id)
        j = s.finish_run(run_id, "failed", error="push", now=t + 1, pause_reason="push_refused")
        assert (j.status, j.paused_reason) == ("paused", "push_refused")
        # 一次性任务已是 done：只记录结果，不会变成 paused
        once = mk(s, user=U2, title="o", kind="once", next_run_at=NOW + 5)
        claims, _ = s.claim_due(NOW + 5, cli=False)
        j = s.finish_run(claims[0].run_id, "failed", error="push", now=NOW + 6, pause_reason="push_refused", max_failures=1)
        assert j.status == "done" and j.last_status == "failed" and j.run_count == 1
        # 任务在运行中被取消：收尾不报错，返回 None
        gone = mk(s, user=U2, title="g", cron="0 10 * * *")
        t = gone.next_run_at
        claims, _ = s.claim_due(t, cli=False)
        s.cancel_job(gone.id)
        assert s.finish_run(claims[0].run_id, "ok", now=t + 1) is None
        assert s.finish_run(10**9, "ok") is None, "不存在的 run 也不报错"
        raises(ValueError, s.finish_run, run_id, "claimed")
        raises(ValueError, s.finish_run, run_id, "ok", pause_reason="bogus")


def test_b8_recover_stale_runs():
    with _Env() as env:
        s = env.store
        threshold = config.AGENT_TURN_TIMEOUT_SECONDS + 300
        stale_once = mk(s, user=U1, title="o1", kind="once", next_run_at=NOW + 10)
        cli_once = mk(s, user=OWNER, title="o3", kind="once", next_run_at=NOW + 10, target=CLI_TARGET)
        fresh_once = mk(s, user=U2, title="o2", kind="once", next_run_at=NOW + 500)
        cron_job = mk(s, user=U1, title="c1", cron="0 9 * * *")
        assert [c.job.id for c in s.claim_due(NOW + 10, cli=False)[0]] == [stale_once.id]
        assert [c.job.id for c in s.claim_due(NOW + 10, cli=True)[0]] == [cli_once.id]
        assert [c.job.id for c in s.claim_due(NOW + 500, cli=False)[0]] == [fresh_once.id]
        # 刚好没到阈值：谁都不动
        assert s.recover_stale_runs(NOW + 10 + threshold, cli=False) == []
        affected = s.recover_stale_runs(NOW + 10 + threshold + 1, cli=False)
        assert [j.id for j in affected] == [stale_once.id], "QQ 调度器只处理 QQ 任务；没到阈值的（fresh）不动"
        r = s.get_runs(stale_once.id)[0]
        assert (r.status, r.error, r.finished_at) == ("failed", "process_restart", NOW + 10 + threshold + 1)
        j = s.get_job(stale_once.id)
        assert j.last_status == "failed" and j.status == "done" and j.consecutive_failures == 0, "重启不计入连续失败，也不重跑"
        assert s.get_runs(cli_once.id)[0].status == "claimed", "cli 过滤：CLI 的任务没被 QQ 调度器碰"
        assert s.get_runs(fresh_once.id)[0].status == "claimed"
        # cron 的 run 也会被恢复，但不会返回（重复任务静默）
        cron_run = s.claim_due(cron_job.next_run_at, cli=False)[0][0]
        far = cron_job.next_run_at + threshold + 1
        affected = s.recover_stale_runs(far)       # cli=None：都处理
        assert {j.id for j in affected} == {cli_once.id, fresh_once.id}
        assert s.get_run(cron_run.run_id).status == "failed" and s.get_job(cron_job.id).status == "active"
        assert s.recover_stale_runs(far + 10) == [], "幂等"
        # 自定义阈值
        mk(s, user=U1, title="o9", kind="once", next_run_at=far + 20, spec_key="k9")
        s.claim_due(far + 20, cli=False)
        assert s.recover_stale_runs(far + 30, stale_after=5)[0].title == "o9"


def test_b9_listeners_and_misc():
    with _Env() as env:
        s = env.store
        calls = []
        cb = lambda: calls.append(threading.current_thread().name)       # noqa: E731
        s.add_change_listener(cb)
        s.add_change_listener(cb)       # 重复注册只算一次
        a = mk(s, user=U1, title="a")
        assert len(calls) == 1, calls
        # 去重命中、失败的创建、没有变化的操作不回调
        s.create_job(user_id=U1, target_openid=U1, title="a", mode="reminder", prompt="该喝水了", kind="cron", cron="30 8 * * *",
                     timezone=SH, next_run_at=a.next_run_at, spec_key="cron:30 8 * * *", max_jobs=10, now=NOW + 1)
        raises(JobLimitError, mk, s, user=U1, title="b", max_jobs=1)
        s.cancel_job("0" * 32)
        s.pause_user_jobs(U2, "admin")
        assert len(calls) == 1, "没有变化就不回调"
        s.pause_job(a.id)
        s.resume_job(a.id, now=NOW)
        s.pause_user_jobs(U1, "admin")
        s.cancel_job(a.id)
        assert len(calls) == 5, calls
        # 调度器自己的写操作（认领 / 收尾 / 恢复 / 清理）不回调，免得自己唤醒自己
        b = mk(s, user=U1, title="b", kind="once", next_run_at=NOW + 5)
        n = len(calls)
        claims, _ = s.claim_due(NOW + 5, cli=False)
        s.finish_run(claims[0].run_id, "ok", now=NOW + 6)
        s.recover_stale_runs(NOW + 99999)
        s.purge_done(NOW + 99 * DAY)
        assert len(calls) == n
        # 回调出错不影响写操作，也不影响其他回调
        broken = lambda: 1 / 0       # noqa: E731
        s.add_change_listener(broken)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            c = mk(s, user=U1, title="c", cron="1 9 * * *")
        assert s.get_job(c.id) is not None and len(calls) == n + 1 and "ZeroDivisionError" in buf.getvalue()
        s.remove_change_listener(broken)
        # 从别的线程创建（LangChain 工具线程）时，回调在那个线程里执行
        out = []
        t = threading.Thread(target=lambda: out.append(mk(s, user=U2, title="t", cron="2 9 * * *")), name="tool-thread")
        t.start()
        t.join()
        assert calls[-1] == "tool-thread" and out
        s.remove_change_listener(cb)
        s.remove_change_listener(cb)
        before = len(calls)
        mk(s, user=U2, title="t2", cron="3 9 * * *")
        assert len(calls) == before


def test_b10_first_hint_and_purge():
    with _Env() as env:
        s = env.store
        assert s.take_first_hint(U1) is True and s.take_first_hint(U1) is False
        assert s.take_first_hint(U2) is True and s.take_first_hint(OWNER) is True
        raises(ValueError, s.take_first_hint, "bad id")
        env.reopen()
        assert env.store.take_first_hint(U1) is False, "标记持久化在 meta 表里"
        s = env.store
        # purge_done：只删完成超过保留期的一次性任务及其 run
        old = mk(s, user=U1, title="old", kind="once", next_run_at=NOW + 5)
        new = mk(s, user=U1, title="new", kind="once", next_run_at=NOW + 5 + 40 * DAY, spec_key="k2")
        live = mk(s, user=U1, title="live", cron="0 9 * * *")
        claims, _ = s.claim_due(NOW + 5, cli=False)
        s.finish_run(claims[0].run_id, "ok", now=NOW + 6)
        claims, _ = s.claim_due(NOW + 5 + 40 * DAY, cli=False, grace=10**9)
        s.finish_run(claims[0].run_id, "ok", now=NOW + 5 + 40 * DAY)
        assert s.purge_done(NOW + 31 * DAY) == 1
        assert s.get_job(old.id) is None and s.get_runs(old.id) == []
        assert s.get_job(new.id) is not None and s.get_job(live.id) is not None
        assert s.purge_done(NOW + 31 * DAY) == 0
        assert s.purge_done(NOW + 80 * DAY, retention=DAY) == 1
        assert s.get_job(live.id) is not None, "进行中的任务永远不清理"


# ════════════════════════ C / D / E 调度循环、失败策略、事件 ════════════════════════
# 这几组场景在真实的事件循环里跑调度器：任务设在“现在 + 零点几秒”，推送与 run_turn 都是假的；
# 所有等待都带超时，重试 / 节流间隔用 patched() 压到几十毫秒。

FAST = dict(
    SCHEDULER_PUSH_MIN_INTERVAL_SECONDS=0.01,
    SCHEDULER_RETRY_DELAY_SECONDS=0.05,
    SCHEDULER_RETRY_MAX=2,
    SCHEDULER_MISFIRE_GRACE_SECONDS=900,
    SCHEDULER_MAX_CONCURRENT_RUNS=2,
    SCHEDULER_MAX_CONSECUTIVE_FAILURES=3,
    SCHEDULER_LOCK_WAIT_SECONDS=1,
)


class _SchedEnv(_Env):
    """调度器场景用的环境：schedules.db 与 sessions.db 都在临时目录里，退出时还原全局单例并重置调度器。"""

    def __enter__(self):
        super().__enter__()
        self.sessions = SessionStore(os.path.join(self.dir, "sessions.db"))
        session_store.set_store(self.sessions)
        scheduler_service.reset_for_tests()
        return self

    def __exit__(self, *exc):
        scheduler_service.reset_for_tests()
        session_store.close_store()
        self.sessions.close()
        super().__exit__(*exc)


class FakePush:
    """假的推送：记录每次调用；script 里按顺序给出每次调用的结果（异常实例会被抛出，其余视为成功）。"""

    def __init__(self, script=None):
        self.calls: list[tuple[str, str, float]] = []
        self.script = list(script or [])

    async def __call__(self, openid, text):
        self.calls.append((openid, text, time.monotonic()))
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item

    @property
    def texts(self):
        return [c[1] for c in self.calls]


class FakeRun:
    """假的 run_turn：results 里每项是 AgentRunResult / 异常 / 无参协程函数；用完后重复最后一项。"""

    def __init__(self, *results):
        self.results = list(results) or [AgentRunResult(True, "好的")]
        self.calls: list[dict] = []
        self.active = 0
        self.max_active = 0

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if callable(item):
                item = await item()
            if isinstance(item, BaseException):
                raise item
            return item
        finally:
            self.active -= 1


async def wait_until(cond, timeout=5.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(step)
    return False


def soon(seconds=0.25):
    return time.time() + seconds


def run_scenario(coro_fn):
    asyncio.run(coro_fn())


def _runs(store, job):
    return store.get_runs(job.id)


def test_c1_reminder_end_to_end():
    with _SchedEnv() as env, patched(**FAST):
        s = env.store
        once = mk(s, user=U1, title="喝水", kind="once", next_run_at=soon(), prompt="该喝水了")
        cron = mk(s, user=U2, title="吃药", cron="0 9 * * 1,3,5", next_run_at=soon(0.3), prompt="吃药", spec_key="c")
        push = FakePush()

        async def scenario():
            scheduler_service.start(run_turn=FakeRun(), push=push, cli=False)
            assert scheduler_service.is_running()
            assert await wait_until(lambda: len(push.calls) == 2)
            await asyncio.sleep(0.5)              # 多等一会儿：到点只会推一次，不会重复
            assert len(push.calls) == 2, push.texts
            await scheduler_service.stop()

        run_scenario(scenario)
        by_target = {c[0]: c[1] for c in push.calls}
        assert by_target == {U1: "⏰ 该喝水了", U2: "⏰ 吃药"}, by_target
        j = s.get_job(once.id)
        assert (j.status, j.last_status, j.run_count, j.consecutive_failures) == ("done", "ok", 1, 0)
        r = _runs(s, once)
        assert len(r) == 1 and r[0].status == "ok" and r[0].finished_at is not None and r[0].error is None
        # cron：认领时就推进到严格晚于现在的下一次触发
        j = s.get_job(cron.id)
        assert j.status == "active" and j.last_status == "ok" and j.run_count == 1
        assert j.next_run_at == schedule_time.next_fire("0 9 * * 1,3,5", cron.next_run_at, SH) and j.next_run_at > time.time()
        assert not scheduler_service.is_running()


def test_c2_missed_window_and_restart_recovery():
    with _SchedEnv() as env, patched(**FAST):
        s = env.store
        base = time.time()
        missed_once = mk(s, user=U1, title="开会", kind="once", next_run_at=base - 2000, prompt="开会了")
        missed_cron = mk(s, user=U2, title="日报", cron="0 9 * * *", next_run_at=base - 2000, prompt="日报", spec_key="m")
        # 另一个一次性任务：在上一个进程里已被认领（run 仍是 claimed），而且已经过去很久 -> 启动时按“被中断”处理，不重跑
        interrupted = mk(s, user=U1, title="取快递", kind="once", next_run_at=base - 5000, prompt="取快递", spec_key="i")
        claimed, _ = s.claim_due(base - 4999, cli=False, grace=900)
        assert [c.job.id for c in claimed] == [interrupted.id]
        push = FakePush()

        async def scenario():
            scheduler_service.start(run_turn=FakeRun(), push=push, cli=False)
            assert await wait_until(lambda: len(push.calls) == 2)
            await asyncio.sleep(0.3)
            await scheduler_service.stop()

        run_scenario(scenario)
        texts = {c[1] for c in push.calls}
        hhmm = lambda job: schedule_time.fmt_local(job.next_run_at, SH)[-5:]       # noqa: E731
        assert texts == {f"「开会」原定 {hhmm(missed_once)} 触发，当时机器人不在线，没有执行。",
                         f"「取快递」原定 {hhmm(interrupted)} 触发，当时机器人不在线，没有执行。"}, texts
        assert all("开会了" not in t and "取快递" not in t.replace("「取快递」", "") for t in texts), "没有执行，也就不会推送任务内容"
        assert len(push.calls) == 2, "每个一次性任务最多一条通知；错过的 cron 静默"
        assert s.get_job(missed_once.id).last_status == "missed" and s.get_runs(missed_once.id)[0].status == "missed"
        j = s.get_job(missed_cron.id)
        assert j.status == "active" and j.last_status == "missed" and j.next_run_at > base and len(s.get_runs(missed_cron.id)) == 1
        r = s.get_runs(interrupted.id)[0]
        assert (r.status, r.error) == ("failed", "process_restart")
        # 再启动一次：不会重复通知，也不会补跑
        push2 = FakePush()

        async def again():
            scheduler_service.reset_for_tests()
            scheduler_service.start(run_turn=FakeRun(), push=push2, cli=False)
            await asyncio.sleep(0.4)
            await scheduler_service.stop()

        run_scenario(again)
        assert push2.calls == []


def test_c3_wakeup_stop_and_listener():
    with _SchedEnv() as env, patched(**FAST):
        s = env.store
        push = FakePush()
        started = {}

        async def scenario():
            scheduler_service.start(run_turn=FakeRun(), push=push, cli=False)
            await asyncio.sleep(0.2)                 # 循环此时在睡（没有任务时最长睡 30 秒）
            t0 = time.monotonic()
            # 从别的线程创建任务（LangChain 的工具线程）：存储的变更监听器 -> notify_changed -> 唤醒循环
            th = threading.Thread(target=lambda: mk(s, user=U1, title="喝水", kind="once", next_run_at=soon(0.3), prompt="喝水"))
            th.start()
            th.join()
            assert await wait_until(lambda: len(push.calls) == 1, timeout=3.0), "没有被唤醒"
            started["wake_delay"] = time.monotonic() - t0
            await scheduler_service.stop()
            await scheduler_service.stop()           # 幂等
            scheduler_service.notify_changed()       # 停止之后：什么都不做
            assert s._listeners == [], "stop 之后监听器已摘掉"

        run_scenario(scenario)
        assert started["wake_delay"] < 3.0, started

        # notify_changed：没启动时是空操作；loop 已关闭时吞掉异常
        scheduler_service.reset_for_tests()
        scheduler_service.notify_changed()
        push2 = FakePush()

        async def start_only():
            scheduler_service.start(run_turn=FakeRun(), push=push2, cli=False)
            await asyncio.sleep(0.05)
        run_scenario(start_only)                      # asyncio.run 结束时 loop 被关闭，调度器没来得及 stop
        scheduler_service.notify_changed()            # 不抛
        raises(RuntimeError, scheduler_service.start, run_turn=FakeRun(), push=push2)       # 没有运行中的事件循环
        scheduler_service.reset_for_tests()

        async def twice():
            scheduler_service.start(run_turn=FakeRun(), push=push2)
            raises(RuntimeError, scheduler_service.start, run_turn=FakeRun(), push=push2)
            await scheduler_service.stop()
            scheduler_service.start(run_turn=FakeRun(), push=push2)       # 停止之后可以再启动
            await scheduler_service.stop()
        run_scenario(twice)


def test_c4_stop_cancels_inflight_runs():
    with _SchedEnv() as env, patched(**FAST):
        s = env.store
        job = mk(s, user=U1, title="查天气", kind="once", next_run_at=soon(0.1), mode="agent", prompt="查天气")
        gate = {}

        async def hang():
            gate["entered"] = True
            await asyncio.sleep(60)

        run = FakeRun(hang)
        push = FakePush()

        async def scenario():
            scheduler_service.start(run_turn=run, push=push, cli=False)
            assert await wait_until(lambda: gate.get("entered"), timeout=3.0)
            await scheduler_service.stop()

        run_scenario(scenario)
        r = s.get_runs(job.id)[0]
        assert (r.status, r.error) == ("failed", "cancelled")
        j = s.get_job(job.id)
        assert j.consecutive_failures == 0 and j.last_status == "failed" and push.calls == []

        # 认领了但还没来得及开跑就被取消：stop() 兜底补写
        job2 = mk(s, user=U1, title="第二个", kind="once", next_run_at=soon(0.1), mode="agent", prompt="x", spec_key="2")

        async def cancel_before_run():
            scheduler_service.reset_for_tests()
            scheduler_service.start(run_turn=FakeRun(hang), push=push, cli=False)
            sched = scheduler_service._current
            for _ in range(200):
                await asyncio.sleep(0.01)
                if sched.inflight:
                    break
            await scheduler_service.stop()
        run_scenario(cancel_before_run)
        assert s.get_runs(job2.id)[0].status == "failed" and s.get_runs(job2.id)[0].error == "cancelled"


def test_c5_concurrency_throttle_and_blocked():
    with _SchedEnv() as env:
        s = env.store
        # 并发上限：名额满了其余的等着，每个任务最终都执行
        with patched(**{**FAST, "SCHEDULER_MAX_CONCURRENT_RUNS": 1}):
            jobs = [mk(s, user=U1, title=f"任务{i}", kind="once", next_run_at=soon(0.1), mode="agent", prompt=f"p{i}", spec_key=str(i))
                    for i in range(3)]

            async def slow():
                await asyncio.sleep(0.15)
                return AgentRunResult(True, "完成")
            run, push = FakeRun(slow), FakePush()

            async def scenario():
                scheduler_service.start(run_turn=run, push=push, cli=False)
                assert await wait_until(lambda: len(push.calls) == 3, timeout=6.0), push.texts
                await scheduler_service.stop()
            run_scenario(scenario)
            assert run.max_active == 1, "SCHEDULER_MAX_CONCURRENT_RUNS=1：同一时刻只跑一个"
            assert all(s.get_job(j.id).last_status == "ok" for j in jobs)

        # 推送节流：相邻两条至少隔 SCHEDULER_PUSH_MIN_INTERVAL_SECONDS
        with patched(**{**FAST, "SCHEDULER_PUSH_MIN_INTERVAL_SECONDS": 0.2, "SCHEDULER_MAX_CONCURRENT_RUNS": 4}):
            scheduler_service.reset_for_tests()
            for i in range(3):
                mk(s, user=U2, title=f"提醒{i}", kind="once", next_run_at=soon(0.1), prompt=f"r{i}", spec_key=f"t{i}")
            push = FakePush()

            async def throttled():
                scheduler_service.start(run_turn=FakeRun(), push=push, cli=False)
                assert await wait_until(lambda: len(push.calls) == 3, timeout=5.0)
                await scheduler_service.stop()
            run_scenario(throttled)
            times = [c[2] for c in push.calls]
            assert all(b - a >= 0.18 for a, b in zip(times, times[1:])), [round(b - a, 3) for a, b in zip(times, times[1:])]

        # 被 /block 的 member：记 skipped，不推送；owner 不受封禁影响
        with patched(**FAST):
            scheduler_service.reset_for_tests()
            env.sessions.touch_user(U1)
            env.sessions.set_user_blocked(U1, True)
            blocked = mk(s, user=U1, title="被封的", kind="once", next_run_at=soon(0.1), prompt="x", spec_key="b")
            fine = mk(s, user=OWNER, title="主人的", kind="once", next_run_at=soon(0.1), prompt="y", spec_key="o")
            push = FakePush()

            async def blocked_run():
                scheduler_service.start(run_turn=FakeRun(), push=push, cli=False)
                assert await wait_until(lambda: len(push.calls) == 1, timeout=3.0)
                await asyncio.sleep(0.3)
                await scheduler_service.stop()
            run_scenario(blocked_run)
            assert push.calls[0][0] == OPENID_OWNER and len(push.calls) == 1
            r = s.get_runs(blocked.id)[0]
            assert (r.status, r.error) == ("skipped", "blocked")
            j = s.get_job(blocked.id)
            assert j.last_status == "skipped" and j.run_count == 0 and j.consecutive_failures == 0
            assert s.get_runs(fine.id)[0].status == "ok"


def test_c6_cli_scheduler_only_takes_cli_jobs():
    with _SchedEnv() as env, patched(**FAST):
        s = env.store
        qq = mk(s, user=U1, title="qq", kind="once", next_run_at=soon(0.1), prompt="q", spec_key="q")
        cli = mk(s, user=OWNER, title="cli", kind="once", next_run_at=soon(0.1), prompt="c", spec_key="c", target=schedule_store.CLI_TARGET)
        push = FakePush()

        async def scenario():
            scheduler_service.start(run_turn=FakeRun(), push=push, cli=True)
            assert await wait_until(lambda: len(push.calls) == 1, timeout=3.0)
            await asyncio.sleep(0.3)
            await scheduler_service.stop()
        run_scenario(scenario)
        assert push.calls[0][0] == schedule_store.CLI_TARGET and len(push.calls) == 1
        assert s.get_job(qq.id).status == "active" and s.get_job(cli.id).status == "done"


def test_c7_agent_success_and_run_turn_contract():
    with _SchedEnv() as env, patched(**FAST):
        s = env.store
        job = mk(s, user=U1, title="天气", kind="once", next_run_at=soon(0.1), mode="agent", prompt="查询北京今天的天气", tz=NY)
        long_job = mk(s, user=U2, title="长", kind="once", next_run_at=soon(0.1), mode="agent", prompt="写长文", spec_key="l")
        calls_text = {job.id: "北京今天晴，带伞不用。", long_job.id: "字" * 2000}
        push = FakePush()

        class _Run(FakeRun):
            async def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return AgentRunResult(True, calls_text[kwargs["job_id"]])
        run = _Run()

        async def scenario():
            scheduler_service.start(run_turn=run, push=push, cli=False)
            assert await wait_until(lambda: len(push.calls) == 2, timeout=3.0)
            await scheduler_service.stop()
        run_scenario(scenario)
        by_target = {c[0]: c[1] for c in push.calls}
        assert by_target[U1] == "北京今天晴，带伞不用。", "agent 的推送正文不带 ⏰ 前缀"
        assert by_target[U2] == "字" * config.SCHEDULER_PUSH_MAX_CHARS + "（内容过长已截断）"
        call = next(c for c in run.calls if c["job_id"] == job.id)
        run_id = s.get_runs(job.id)[0].id
        assert call == {"job_id": job.id, "user_id": U1, "target_openid": U1, "title": "天气",
                        "prompt": "查询北京今天的天气", "timezone": NY, "run_ref": f"{run_id}-0"}, call
        assert s.get_job(job.id).last_status == "ok"


# ════════════════════════ D 失败策略 ════════════════════════

def _one_job(env, *, mode="reminder", kind="once", user=U1, prompt="该喝水了", title="喝水", cron="0 9 * * *", **kw):
    kw.setdefault("next_run_at", soon(0.1))
    kw.setdefault("max_jobs", 100)
    return mk(env.store, user=user, title=title, kind=kind, mode=mode, prompt=prompt, cron=cron, **kw)


def _run_one(env, job, push, run=None, *, expect_calls=1, settle=0.3, timeout=5.0):
    async def scenario():
        scheduler_service.reset_for_tests()
        scheduler_service.start(run_turn=run or FakeRun(), push=push, cli=False)
        assert await wait_until(lambda: len(push.calls) >= expect_calls and not scheduler_service._current.inflight, timeout=timeout), push.texts
        await asyncio.sleep(settle)
        await scheduler_service.stop()
    run_scenario(scenario)
    return env.store.get_job(job.id)


def test_d1_transient_retries_and_deadline():
    with _SchedEnv() as env, patched(**FAST):
        # transient 一直失败：最多 1 次首发 + SCHEDULER_RETRY_MAX 次重试，然后记失败
        job = _one_job(env, kind="cron")
        push = FakePush([PushError("transient", 40034100, "x")] * 10)
        j = _run_one(env, job, push, expect_calls=3, settle=0.4)
        assert len(push.calls) == 3, push.texts
        assert (j.last_status, j.consecutive_failures, j.status) == ("failed", 1, "active")
        r = env.store.get_runs(job.id)[0]
        assert r.status == "failed" and r.error == "push_transient"
        assert len({c[1] for c in push.calls}) == 1, "重试用的是同一段正文"
        # 重试后成功：只算一次成功，事件只写一次
        job2 = _one_job(env, prompt="再来一次", title="第二个", spec_key="d1b")
        push2 = FakePush([PushError("transient", None, "bot offline"), OSError("boom")])
        j = _run_one(env, job2, push2, expect_calls=3)
        assert len(push2.calls) == 3 and j.status == "done" and j.last_status == "ok", (push2.texts, j)
        # 不超过 scheduled_for + 宽限：时钟已经走到宽限边缘，不再重试
        job3 = mk(env.store, user=U2, title="边缘", kind="once", next_run_at=1000.0, prompt="边缘", spec_key="d1c")
        push3 = FakePush([PushError("transient", None, "x")] * 5)

        async def at_edge():
            scheduler_service.reset_for_tests()
            scheduler_service.start(run_turn=FakeRun(), push=push3, cli=False, clock=lambda: 1000.0 + config.SCHEDULER_MISFIRE_GRACE_SECONDS - 0.01)
            assert await wait_until(lambda: len(push3.calls) >= 1 and not scheduler_service._current.inflight)
            await asyncio.sleep(0.3)
            await scheduler_service.stop()
        run_scenario(at_edge)
        assert len(push3.calls) == 1, "离宽限截止不足一个重试间隔：不重试"
        assert env.store.get_job(job3.id).last_status == "failed"


def test_d2_dedupe_toolong_url():
    with _SchedEnv() as env, patched(**FAST):
        # dedupe：正文末尾加 [MM-DD HH:MM] 后重试一次；再次 dedupe 就放弃
        job = _one_job(env, prompt="该喝水了")
        push = FakePush([PushError("dedupe", 40054005, "dup")])
        j = _run_one(env, job, push, expect_calls=2)
        assert push.texts[0] == "⏰ 该喝水了" and len(push.texts) == 2
        assert push.texts[1].startswith("⏰ 该喝水了\n[") and push.texts[1].endswith("]") and re.fullmatch(r"\[\d\d-\d\d \d\d:\d\d\]", push.texts[1].split("\n")[-1]), push.texts
        assert j.last_status == "ok"
        job = _one_job(env, prompt="又一条", title="b", spec_key="d2b")
        push = FakePush([PushError("dedupe", 40054005, "dup")] * 5)
        j = _run_one(env, job, push, expect_calls=2, settle=0.3)
        assert len(push.calls) == 2 and j.last_status == "failed" and env.store.get_runs(job.id)[0].error == "push_dedupe"
        # too_long：截断后重试一次
        job = _one_job(env, prompt="长" * 150, title="c", spec_key="d2c")
        push = FakePush([PushError("too_long", 40054007, "x")])
        j = _run_one(env, job, push, expect_calls=2)
        assert len(push.texts[0]) == 152 and push.texts[1].endswith("（内容过长已截断）") and len(push.texts[1]) < len(push.texts[0]), [len(t) for t in push.texts]
        assert j.last_status == "ok"
        # url_rejected：去掉链接后重试一次
        job = _one_job(env, prompt="详情见 https://example.com/a?b=1 和 www.example.org 这里", title="d", spec_key="d2d")
        push = FakePush([PushError("url_rejected", 40054010, "x")])
        j = _run_one(env, job, push, expect_calls=2)
        assert "https://" in push.texts[0] and "https://" not in push.texts[1] and "www." not in push.texts[1] and "（链接已省略）" in push.texts[1], push.texts
        assert j.last_status == "ok"


def test_d3_permanent_pauses_and_stays_silent():
    with _SchedEnv() as env, patched(**FAST):
        # 重复任务：暂停，reason=push_refused，不计入连续失败，也不再推任何失败通知
        job = _one_job(env, kind="cron", prompt="提醒")
        push = FakePush([PushError("permanent", 40034105, "no permission")] * 5)
        j = _run_one(env, job, push, expect_calls=1, settle=0.4)
        assert len(push.calls) == 1, "permanent 立即放弃，不重试"
        assert (j.status, j.paused_reason, j.consecutive_failures, j.last_status) == ("paused", "push_refused", 0, "failed")
        assert env.store.get_runs(job.id)[0].error == "push_refused"
        # 用户在 QQ 里重新打开允许主动消息之后，自己 /tasks resume 就能恢复
        assert env.store.resume_job(job.id).status == "active"
        # 一次性任务已是 done：只记录结果，仍是 done
        job = _one_job(env, kind="once", prompt="一次性", title="o", spec_key="d3b")
        push = FakePush([PushError("permanent", 40054013, "refused")] * 5)
        j = _run_one(env, job, push, expect_calls=1)
        assert (j.status, j.last_status, j.paused_reason) == ("done", "failed", None) and len(push.calls) == 1
        # agent 任务失败后推失败通知时被拒收：也暂停，且只推了这一条
        job = _one_job(env, kind="cron", mode="agent", prompt="查天气", title="a", spec_key="d3c")
        push = FakePush([PushError("permanent", 40034105, "x")] * 5)
        run = FakeRun(AgentRunResult(False, "", "timeout"))
        j = _run_one(env, job, push, run, expect_calls=1, settle=0.4)
        assert len(push.calls) == 1 and (j.status, j.paused_reason) == ("paused", "push_refused"), (push.texts, j)


def test_d4_agent_failures():
    with _SchedEnv() as env, patched(**FAST):
        expectations = {
            "hitl": "定时任务「天气」执行时需要你确认一项操作，无人值守的任务不会执行需要授权的操作，已放弃本次执行。",
            "timeout": "⚠️ 定时任务「天气」这次没能完成（执行超时）",
            "empty": "⚠️ 定时任务「天气」这次没能完成（没有得到可以发送的结果）",
            "error": "⚠️ 定时任务「天气」这次没能完成（执行时出了错）",
            "unavailable": "⚠️ 定时任务「天气」这次没能完成（定时任务功能暂时不可用）",
        }
        for i, (error, want) in enumerate(expectations.items()):
            job = _one_job(env, kind="cron", mode="agent", prompt="查天气", title="天气", spec_key=f"d4{i}")
            push = FakePush()
            j = _run_one(env, job, push, FakeRun(AgentRunResult(False, "", error)), expect_calls=1, settle=0.3)
            assert push.texts == [want], (error, push.texts)
            assert (j.consecutive_failures, j.status, j.last_status) == (1, "active", "failed"), (error, j)
            assert env.store.get_runs(job.id)[0].error == error
        # agent 返回 ok 但正文为空：按 empty 处理
        job = _one_job(env, kind="cron", mode="agent", prompt="查天气", title="天气", spec_key="d4e")
        push = FakePush()
        _run_one(env, job, push, FakeRun(AgentRunResult(True, "  \n ")), expect_calls=1)
        assert push.texts == [expectations["empty"]]
        # run_turn 抛异常：按 error 处理（不让调度循环崩）；返回了不认识的东西同样
        for i, bad in enumerate((RuntimeError("boom"), "not a result")):
            job = _one_job(env, kind="cron", mode="agent", prompt="查天气", title="天气", spec_key=f"d4x{i}")
            push = FakePush()
            _run_one(env, job, push, FakeRun(bad), expect_calls=1)
            assert push.texts == [expectations["error"]], push.texts
        # stopped：用户自己 /stop 的，不推通知，也不计失败
        job = _one_job(env, kind="cron", mode="agent", prompt="查天气", title="天气", spec_key="d4s")
        push = FakePush()
        j = _run_one(env, job, push, FakeRun(AgentRunResult(False, "", "stopped")), expect_calls=0, settle=0.5)
        assert push.calls == [] and j.consecutive_failures == 0 and j.last_status == "failed"
        # busy：按间隔重试 SCHEDULER_RETRY_MAX 次；仍 busy 才放弃，不计失败，但会说明原因
        job = _one_job(env, kind="cron", mode="agent", prompt="查天气", title="天气", spec_key="d4b")
        push = FakePush()
        run = FakeRun(AgentRunResult(False, "", "busy"))
        j = _run_one(env, job, push, run, expect_calls=1, settle=0.3)
        assert len(run.calls) == 1 + config.SCHEDULER_RETRY_MAX, len(run.calls)
        assert len({c["run_ref"] for c in run.calls}) == len(run.calls), "每次尝试的 run_ref 不同（临时线程 id 不会撞）"
        assert push.texts == ["⚠️ 定时任务「天气」这次没能完成（你当时正在和我对话，一直没排上）"]
        assert j.consecutive_failures == 0
        # busy 之后重试成功
        job = _one_job(env, kind="cron", mode="agent", prompt="查天气", title="天气", spec_key="d4c")
        push = FakePush()
        run = FakeRun(AgentRunResult(False, "", "busy"), AgentRunResult(True, "晴天"))
        j = _run_one(env, job, push, run, expect_calls=1)
        assert push.texts == ["晴天"] and len(run.calls) == 2 and j.last_status == "ok"


def test_d5_consecutive_failures_pause():
    with _SchedEnv() as env, patched(**{**FAST, "SCHEDULER_MAX_CONSECUTIVE_FAILURES": 3}):
        job = _one_job(env, kind="cron", mode="agent", prompt="查天气", title="天气")
        run = FakeRun(AgentRunResult(False, "", "error"))
        pushes = []
        for round_no in range(3):
            push = FakePush()
            j = _run_one(env, job, push, run, expect_calls=1)
            pushes.append(push.texts)
            assert j.consecutive_failures == round_no + 1
            if round_no < 2:
                assert j.status == "active"
                env.store._conn.execute("UPDATE jobs SET next_run_at = ? WHERE id = ?", (soon(0.1), job.id))      # 让它再到点
        assert (j.status, j.paused_reason) == ("paused", "failures")
        assert "/tasks resume" in pushes[2][0] and "已暂停" in pushes[2][0]
        assert "/tasks resume" not in pushes[0][0]
        # 成功一次就清零（先恢复）
        env.store.resume_job(job.id)
        env.store._conn.execute("UPDATE jobs SET next_run_at = ? WHERE id = ?", (soon(0.1), job.id))
        j = _run_one(env, job, FakePush(), FakeRun(AgentRunResult(True, "好")), expect_calls=1)
        assert j.consecutive_failures == 0 and j.last_status == "ok"
        # reminder 推送连续失败同样会暂停（但没法再通知用户）
        job2 = _one_job(env, kind="cron", prompt="提醒", title="r", spec_key="d5r")
        for _ in range(3):
            push = FakePush([PushError("transient", None, "x")] * 5)
            j = _run_one(env, job2, push, expect_calls=3, settle=0.2)
            env.store._conn.execute("UPDATE jobs SET next_run_at = ? WHERE id = ?", (soon(0.1), job2.id))
        assert (j.status, j.paused_reason, j.consecutive_failures) == ("paused", "failures", 3)


def test_d6_push_error_and_result_types():
    e = PushError("transient", 40034100, "x" * 500)
    assert (e.kind, e.code) == ("transient", 40034100) and len(e.detail) == 200 and "transient" in str(e)
    assert PushError("permanent").code is None and PushError("permanent").detail == ""
    assert AgentRunResult(True, "hi") == AgentRunResult(ok=True, text="hi", error="")
    raises(Exception, setattr, AgentRunResult(False), "ok", True)        # frozen


# ════════════════════════ E 事件写入 ════════════════════════

def test_e1_push_event_is_written_and_idempotent():
    from app.tools import session_tools
    with _SchedEnv() as env, patched(**FAST):
        sa = env.sessions.get_active_session_id(user_id=U1)
        sb = env.sessions.get_active_session_id(user_id=U2)
        job = _one_job(env, prompt="该喝水了", title="喝水")
        _one_job(env, user=U2, prompt="别人的提醒", title="别人", spec_key="e1b")
        push = FakePush()

        async def scenario():
            scheduler_service.start(run_turn=FakeRun(), push=push, cli=False)
            assert await wait_until(lambda: len(push.calls) == 2, timeout=3.0)
            await asyncio.sleep(0.2)
            await scheduler_service.stop()
        run_scenario(scenario)
        evs = env.sessions.get_events(sa, types=(session_store.EVENT_SCHEDULED_PUSH,))
        assert len(evs) == 1
        run_id = env.store.get_runs(job.id)[0].id
        assert evs[0].payload == {"job_id": job.id[:8], "title": "喝水", "text": "该喝水了", "mode": "reminder", "run_id": run_id}, evs[0].payload
        assert evs[0].event_key == f"sp:{run_id}"
        evs_b = env.sessions.get_events(sb, types=(session_store.EVENT_SCHEDULED_PUSH,))
        assert len(evs_b) == 1 and evs_b[0].payload["text"] == "别人的提醒" and "该喝水了" not in str(evs_b[0].payload), "各写各的会话"
        # 不是入站事件：不触发 wake，也不进 pending
        assert session_store.EVENT_SCHEDULED_PUSH == "scheduled_push" and session_store.EVENT_SCHEDULED_PUSH not in session_store.INBOUND_EVENT_TYPES
        assert env.sessions.get_pending_events(sa) == []
        # 同一个 run 重复写：幂等
        assert env.sessions.emit_event(sa, session_store.EVENT_SCHEDULED_PUSH, {"x": 1}, key=f"sp:{run_id}") is None
        # 回溯工具里的渲染
        assert session_tools._event_body(evs[0]) == "系统定时推送给用户：「喝水」该喝水了"
        assert "scheduled_push" in session_tools.get_session_events.description

        # 用户没有任何会话时，先建默认会话再写；写事件失败不影响推送结果
        job2 = _one_job(env, user=OWNER, prompt="owner的", title="o", spec_key="e1c")
        orig = session_store.get_store

        def broken():
            raise RuntimeError("db gone")          # owner 不做封禁检查，所以第一次碰 get_store 的就是写事件
        session_store.get_store = broken
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                j = _run_one(env, job2, FakePush(), expect_calls=1)
        finally:
            session_store.get_store = orig
        assert j.last_status == "ok" and "写入定时推送事件失败" in buf.getvalue()


def test_e2_log_hygiene_and_layering():
    """日志里不出现任务正文与完整 openid / job id；scheduler_service 不依赖 chat_service / task_commands。"""
    import ast
    src = (ROOT / "app" / "services" / "scheduler_service.py").read_text(encoding="utf-8")
    imported = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    imported |= {f"{n.module}.{a.name}" for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module for a in n.names}
    assert not any("chat_service" in m or "task_commands" in m for m in imported), imported
    with _SchedEnv() as env, patched(**FAST):
        secret = "绝密正文ABC"
        job = _one_job(env, prompt=secret, title="绝密标题XYZ")
        push = FakePush([PushError("transient", 40034100, "限频"), PushError("permanent", 40034105, "拒绝")])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _run_one(env, job, push, expect_calls=2)
        log = buf.getvalue()
        assert secret not in log and "绝密标题XYZ" not in log, log
        assert job.id not in log and U1 not in log, "job id / openid 只显示前 8 位"
        assert job.id[:8] in log and U1[:8] in log


SCENARIOS = [
    ("A1 时区 / 解析 / cron 校验 / next_fire / describe", test_a1_time),
    ("A2 配置块的夹紧回退与 import 卫生", test_a2_config_and_import_hygiene),
    ("A3 请求上下文新字段 / require_owner / get_current_time", test_a3_context_and_current_time),
    ("B1 建库 / 版本 / 单例", test_b1_open_and_singleton),
    ("B2 创建 / 数量上限 / 重复创建保护", test_b2_create_limit_dedupe),
    ("B3 列表与编号解析", test_b3_list_and_resolve),
    ("B4 取消", test_b4_cancel),
    ("B5 暂停 / 恢复 / 管理员暂停", test_b5_pause_resume),
    ("B6 认领（原子性 / 错过窗口 / cli 分流 / 并发）", test_b6_claim_due),
    ("B7 收尾（失败计数 / 暂停 / 任务已删除）", test_b7_finish_run),
    ("B8 重启恢复", test_b8_recover_stale_runs),
    ("B9 变更监听器", test_b9_listeners_and_misc),
    ("B10 首个任务提示标记与清理", test_b10_first_hint_and_purge),
    ("C1 reminder 端到端：到点只推一次、once 变 done、cron 推进", test_c1_reminder_end_to_end),
    ("C2 错过窗口 / 重启恢复（不补跑、每个一次性任务最多一条通知）", test_c2_missed_window_and_restart_recovery),
    ("C3 被 notify 唤醒（来自别的线程）/ 停机 / 重复启动", test_c3_wakeup_stop_and_listener),
    ("C4 stop 取消在途任务并写 failed(cancelled)", test_c4_stop_cancels_inflight_runs),
    ("C5 并发上限 / 推送节流 / 被封禁的 member 跳过", test_c5_concurrency_throttle_and_blocked),
    ("C6 CLI 调度器只领 CLI 的任务", test_c6_cli_scheduler_only_takes_cli_jobs),
    ("C7 agent 成功路径与 run_turn 调用约定", test_c7_agent_success_and_run_turn_contract),
    ("D1 transient 重试与宽限截止", test_d1_transient_retries_and_deadline),
    ("D2 dedupe / too_long / url_rejected 各重试一次", test_d2_dedupe_toolong_url),
    ("D3 permanent 暂停且不再通知", test_d3_permanent_pauses_and_stays_silent),
    ("D4 agent 失败的各种原因与通知", test_d4_agent_failures),
    ("D5 连续失败暂停与恢复", test_d5_consecutive_failures_pause),
    ("D6 PushError / AgentRunResult", test_d6_push_error_and_result_types),
    ("E1 推送事件写入、隔离、幂等、不影响推送结果", test_e1_push_event_is_written_and_idempotent),
    ("E2 日志卫生与分层", test_e2_log_hygiene_and_layering),
]


def main():
    print("=" * 70)
    print("开始运行定时任务离线冒烟测试...")
    print("=" * 70)
    passed = failed = 0
    for name, fn in SCENARIOS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e!r}")
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
