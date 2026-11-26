"""定时任务存储：jobs / runs / meta (所在层: L1)

独立的 SQLite 文件（config.SCHEDULES_DB_PATH，默认 storage/schedules.db），标准库 sqlite3 实现，
单连接 + RLock，WAL，所有写入都是 BEGIN IMMEDIATE 事务。风格同 session_store.py。
这里不依赖 chat_service / scheduler_service：调度器通过 claim_due / finish_run 与它交互，工具与 /tasks 通过
create_job / list_jobs / resolve / cancel_job / pause_job / resume_job 与它交互。

“至多一次”靠两层保证：claim_due 在一个事务里取到点的任务、插入 claimed 的 run 并立刻把 next_run_at 推进
（一次性任务置 done），所以运行中崩溃也不会重跑；runs 上的 UNIQUE(job_id, scheduled_for) 是最后一道保险。

日志与错误信息里不放任务正文；对外展示的 job id 只取前 8 位。
"""

import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

from app import config
from app.utils import schedule_time
from app.utils.identity import validate_user_id

_SCHEMA_VERSION = 1
CLI_TARGET = "cli"                      # target_openid 的保留值：只由 main.py --run-scheduler 的 CLI 调度器领取
RUN_HISTORY_PER_JOB = 20                # 每个任务只留最近这么多条 run
DEDUPE_WINDOW_SECONDS = 300             # 同一用户这段时间内完全相同的创建请求视为模型重复调用
DONE_RETENTION_SECONDS = 30 * 86400     # 已完成的一次性任务保留多久
_OPENID_RE = re.compile(r"[A-Za-z0-9]{8,64}")
_REF_INDEX_MAX_DIGITS = 3               # 不超过 3 位的纯数字按“序号”解释，否则按 id 前缀
_PAUSE_REASONS = ("user", "admin", "push_refused", "failures")
_MODES = ("reminder", "agent")
_KINDS = ("once", "cron")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    target_openid TEXT NOT NULL,
    title TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('reminder', 'agent')),
    prompt TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('once', 'cron')),
    cron TEXT,
    timezone TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'done')),
    paused_reason TEXT CHECK (paused_reason IS NULL OR paused_reason IN ('user', 'admin', 'push_refused', 'failures')),
    next_run_at REAL NOT NULL,
    last_run_at REAL,
    last_status TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    run_count INTEGER NOT NULL DEFAULT 0,
    spec_key TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_due ON jobs(status, next_run_at);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, created_at);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    scheduled_for REAL NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL CHECK (status IN ('claimed', 'ok', 'failed', 'missed', 'skipped')),
    error TEXT,
    UNIQUE(job_id, scheduled_for)
);
CREATE INDEX IF NOT EXISTS idx_runs_job ON runs(job_id, id);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class ScheduleStoreError(Exception):
    """存储层的预期内错误；message 是中文，可以直接转述给用户。"""


class JobLimitError(ScheduleStoreError):
    """该用户的任务数已到上限。"""


class RefError(ScheduleStoreError):
    """任务编号（序号 / id 前缀）解析不出唯一的任务。"""


class JobStateError(ScheduleStoreError):
    """任务当前状态不允许这个操作。"""


@dataclass(frozen=True)
class Job:
    id: str
    user_id: str
    target_openid: str
    title: str
    mode: str
    prompt: str
    kind: str
    cron: Optional[str]
    timezone: str
    status: str
    paused_reason: Optional[str]
    next_run_at: float
    last_run_at: Optional[float]
    last_status: Optional[str]
    consecutive_failures: int
    run_count: int
    spec_key: str
    created_at: float
    updated_at: float

    @property
    def short_id(self) -> str:
        return self.id[:8]

    @property
    def is_cli(self) -> bool:
        return self.target_openid == CLI_TARGET


@dataclass(frozen=True)
class Run:
    id: int
    job_id: str
    scheduled_for: float
    started_at: float
    finished_at: Optional[float]
    status: str
    error: Optional[str]


@dataclass(frozen=True)
class Claim:
    """claim_due 认领到的一次执行：job 是认领那一刻的快照（next_run_at 已在库里推进，快照里仍是本次的 scheduled_for）。"""
    job: Job
    run_id: int
    scheduled_for: float


def _job_from_row(row: sqlite3.Row) -> Job:
    return Job(**{name: row[name] for name in Job.__dataclass_fields__})


def _run_from_row(row: sqlite3.Row) -> Run:
    return Run(**{name: row[name] for name in Run.__dataclass_fields__})


class ScheduleStore:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.RLock()
        self._closed = False
        self._listeners: list[Callable[[], None]] = []
        self._listeners_lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        # isolation_level=None：自动提交模式，多语句事务由 _tx() 显式 BEGIN IMMEDIATE / COMMIT
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    # ── 基础设施 ──────────────────────────────────────────────

    @property
    def path(self) -> str:
        return self._path

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                with self._tx() as c:
                    c.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(_SCHEMA_VERSION),))
            elif int(row["value"]) > _SCHEMA_VERSION:
                raise RuntimeError(
                    f"schedules.db 的 schema 版本 {row['value']} 高于当前代码支持的 {_SCHEMA_VERSION}，请升级代码后再打开"
                )

    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return int(row["value"]) if row else 0

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.close()
            except Exception:
                pass

    @contextmanager
    def _tx(self):
        """持锁的写事务：BEGIN IMMEDIATE ... COMMIT，异常 ROLLBACK。"""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # ── 变更监听器 ────────────────────────────────────────────

    def add_change_listener(self, callback: Callable[[], None]) -> None:
        """创建 / 取消 / 暂停 / 恢复提交之后回调（不带参数）。回调可能来自 LangChain 的工具线程，所以必须线程安全。"""
        with self._listeners_lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_change_listener(self, callback: Callable[[], None]) -> None:
        with self._listeners_lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _notify(self) -> None:
        """在事务提交之后、不持有存储锁时调用；单个回调出错不影响其他回调，也不影响写操作的结果。"""
        with self._listeners_lock:
            listeners = list(self._listeners)
        for callback in listeners:
            try:
                callback()
            except Exception as e:
                print(f"[WARN] schedule_store: 变更监听器出错: {type(e).__name__}")

    # ── 创建 / 查询 ───────────────────────────────────────────

    def create_job(
        self,
        *,
        user_id: str,
        target_openid: str,
        title: str,
        mode: str,
        prompt: str,
        kind: str,
        cron: Optional[str],
        timezone: str,
        next_run_at: float,
        spec_key: str,
        max_jobs: int,
        now: Optional[float] = None,
    ) -> tuple[Job, bool]:
        """创建任务，返回 (任务, 是否新建)。

        数量上限检查与插入在同一个事务里（active 与 paused 一起算，done 不算）。
        同一用户在 DEDUPE_WINDOW_SECONDS 内 title / prompt / mode / 时间规格完全相同的创建请求，返回已有任务而不新建
        （防模型重复调用），此时第二项是 False。
        """
        validate_user_id(user_id)
        if target_openid != CLI_TARGET and not _OPENID_RE.fullmatch(target_openid or ""):
            raise ValueError("推送目标 openid 不合法")
        if mode not in _MODES or kind not in _KINDS:
            raise ValueError("mode / kind 不合法")
        if (kind == "cron") != bool(cron):
            raise ValueError("cron 任务必须带 cron 表达式，一次性任务不能带")
        now = time.time() if now is None else float(now)
        with self._tx() as c:
            dup = c.execute(
                "SELECT * FROM jobs WHERE user_id = ? AND title = ? AND prompt = ? AND mode = ? AND kind = ?"
                " AND spec_key = ? AND status IN ('active', 'paused') AND created_at >= ?"
                " ORDER BY created_at DESC LIMIT 1",
                (user_id, title, prompt, mode, kind, spec_key, now - DEDUPE_WINDOW_SECONDS),
            ).fetchone()
            if dup is not None:
                return _job_from_row(dup), False
            count = c.execute(
                "SELECT COUNT(*) FROM jobs WHERE user_id = ? AND status IN ('active', 'paused')", (user_id,)
            ).fetchone()[0]
            if count >= max_jobs:
                raise JobLimitError(f"你已经有 {count} 个定时任务，上限是 {max_jobs} 个。请先取消不用的任务再创建")
            job_id = uuid.uuid4().hex
            c.execute(
                "INSERT INTO jobs(id, user_id, target_openid, title, mode, prompt, kind, cron, timezone, status,"
                " next_run_at, spec_key, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?, 'active', ?,?,?,?)",
                (job_id, user_id, target_openid, title, mode, prompt, kind, cron, timezone, float(next_run_at),
                 spec_key, now, now),
            )
            job = _job_from_row(c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())
        self._notify()
        return job, True

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _job_from_row(row) if row else None

    def list_jobs(self, user_id: Optional[str] = None, *, limit: Optional[int] = None) -> list[Job]:
        """user_id 给了只列该用户的，None 列全部（仅 owner 的 /tasks all 与调度器用）。

        顺序固定：进行中 / 暂停的在前、已完成的在后，同组按创建时间。序号（resolve 的 1 开始的编号）就是这个顺序里的位置。"""
        sql = "SELECT * FROM jobs"
        args: list = []
        if user_id is not None:
            sql += " WHERE user_id = ?"
            args.append(validate_user_id(user_id))
        sql += " ORDER BY (status = 'done'), created_at, rowid"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_job_from_row(r) for r in rows]

    def count_user_jobs(self, user_id: str) -> int:
        """进行中与暂停的任务数（占用配额的那部分）。"""
        validate_user_id(user_id)
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE user_id = ? AND status IN ('active', 'paused')", (user_id,)
            ).fetchone()[0]

    def resolve(self, ref: str, user_id: str, is_owner: bool = False) -> Job:
        """序号（1 开始，不超过 3 位的纯数字）或 ≥4 位的 id 前缀 -> 唯一的任务；解析不出抛 RefError。

        序号永远指 user_id 自己的列表；id 前缀只在 user_id 自己的任务里匹配，owner（is_owner=True）可以跨用户匹配。"""
        validate_user_id(user_id)
        text = (ref or "").strip().lower()
        if not text:
            raise RefError("请给出任务的序号（列表里的第几个）或编号前 4 位以上")
        if text.isdigit() and len(text) <= _REF_INDEX_MAX_DIGITS:
            index = int(text)
            jobs = self.list_jobs(user_id)
            if 1 <= index <= len(jobs):
                return jobs[index - 1]
            raise RefError(f"没有第 {index} 个任务，你现在一共有 {len(jobs)} 个")
        if not re.fullmatch(r"[0-9a-f]{4,32}", text):
            raise RefError("任务编号要写列表里的序号，或者编号的前 4 位以上（只含 0-9 和 a-f）")
        sql = "SELECT * FROM jobs WHERE substr(id, 1, ?) = ?"
        args: list = [len(text), text]
        if not is_owner:
            sql += " AND user_id = ?"
            args.append(user_id)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY created_at, rowid", args).fetchall()
        if not rows:
            raise RefError(f"没有找到编号以 {text} 开头的任务")
        if len(rows) > 1:
            raise RefError(f"编号 {text} 匹配到 {len(rows)} 个任务，请多写几位")
        return _job_from_row(rows[0])

    # ── 取消 / 暂停 / 恢复 ────────────────────────────────────

    def cancel_job(self, job_id: str, *, user_id: Optional[str] = None) -> bool:
        """取消 = 删除该任务及其全部 run。传了 user_id 时只删该用户自己的。返回是否删到了。"""
        sql, args = "DELETE FROM jobs WHERE id = ?", [job_id]
        if user_id is not None:
            sql += " AND user_id = ?"
            args.append(validate_user_id(user_id))
        with self._tx() as c:
            deleted = c.execute(sql, args).rowcount
            if deleted:
                c.execute("DELETE FROM runs WHERE job_id = ?", (job_id,))
        if deleted:
            self._notify()
        return bool(deleted)

    def _owned_row(self, c: sqlite3.Connection, job_id: str, user_id: Optional[str]) -> sqlite3.Row:
        row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or (user_id is not None and row["user_id"] != validate_user_id(user_id)):
            raise JobStateError("没有找到这个任务")
        return row

    def pause_job(self, job_id: str, reason: str = "user", *, user_id: Optional[str] = None) -> Job:
        """暂停；已暂停的任务保持暂停（reason 为 admin 时升级成 admin，之后 member 不能自己恢复）。已完成的任务抛 JobStateError。"""
        if reason not in _PAUSE_REASONS:
            raise ValueError("暂停原因不合法")
        now = time.time()
        with self._tx() as c:
            row = self._owned_row(c, job_id, user_id)
            if row["status"] == "done":
                raise JobStateError("这个任务已经执行完了，没有需要暂停的")
            if row["status"] == "active":
                c.execute("UPDATE jobs SET status = 'paused', paused_reason = ?, updated_at = ? WHERE id = ?",
                          (reason, now, job_id))
            elif reason == "admin":
                c.execute("UPDATE jobs SET paused_reason = 'admin', updated_at = ? WHERE id = ?", (now, job_id))
            job = _job_from_row(c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())
        self._notify()
        return job

    def resume_job(
        self,
        job_id: str,
        *,
        now: Optional[float] = None,
        is_owner: bool = False,
        user_id: Optional[str] = None,
    ) -> Job:
        """恢复一个暂停的任务：cron 从现在重算下次触发（并清零连续失败）；已过期的一次性任务拒绝并提示重建。

        被管理员暂停（paused_reason=admin）的任务只有 owner（is_owner=True）能恢复。"""
        now = time.time() if now is None else float(now)
        with self._tx() as c:
            row = self._owned_row(c, job_id, user_id)
            if row["status"] == "done":
                raise JobStateError("这个任务已经执行完了，无法恢复")
            if row["status"] != "paused":
                raise JobStateError("这个任务没有处于暂停状态")
            if row["paused_reason"] == "admin" and not is_owner:
                raise JobStateError("该任务被管理员暂停，请联系管理员")
            if row["kind"] == "once":
                if row["next_run_at"] <= now:
                    raise JobStateError("这个一次性任务的时间已经过了，无法恢复，请重新创建")
                next_run_at = row["next_run_at"]
            else:
                next_run_at = schedule_time.next_fire(row["cron"], now, row["timezone"])
                if next_run_at is None:
                    raise JobStateError("这个任务之后不会再触发，无法恢复")
            c.execute(
                "UPDATE jobs SET status = 'active', paused_reason = NULL, next_run_at = ?, consecutive_failures = 0,"
                " updated_at = ? WHERE id = ?",
                (next_run_at, now, job_id),
            )
            job = _job_from_row(c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())
        self._notify()
        return job

    def pause_user_jobs(self, user_id: str, reason: str) -> int:
        """把某个用户所有进行中的任务暂停（/block 用，reason 通常是 admin）；返回暂停了几个。已暂停的不动。"""
        validate_user_id(user_id)
        if reason not in _PAUSE_REASONS:
            raise ValueError("暂停原因不合法")
        with self._tx() as c:
            changed = c.execute(
                "UPDATE jobs SET status = 'paused', paused_reason = ?, updated_at = ? WHERE user_id = ? AND status = 'active'",
                (reason, time.time(), user_id),
            ).rowcount
        if changed:
            self._notify()
        return changed

    # ── 调度器接口 ────────────────────────────────────────────

    @staticmethod
    def _target_clause(cli: bool) -> str:
        return "target_openid = ?" if cli else "target_openid != ?"

    def claim_due(
        self,
        now: float,
        *,
        cli: bool,
        limit: int = 20,
        grace: Optional[float] = None,
    ) -> tuple[list[Claim], list[Job]]:
        """在一个 BEGIN IMMEDIATE 事务里认领到点的任务，返回 (认领列表, 被判错过的一次性任务列表)。

        到点 = status 为 active 且 next_run_at <= now。超过 misfire 宽限（默认 SCHEDULER_MISFIRE_GRACE_SECONDS）的记成
        missed 的 run；其余插入 claimed 的 run。两种情况都立刻推进 next_run_at：cron 取严格晚于 max(now, scheduled_for)
        的下一次（没有下一次就置 done），一次性任务置 done。所以认领之后进程崩溃也不会重跑。
        cli=True 只领 target_openid='cli' 的任务，cli=False 只领其余的，两个进程永远不会互相“偷”任务。
        """
        grace_seconds = config.SCHEDULER_MISFIRE_GRACE_SECONDS if grace is None else float(grace)
        claims: list[Claim] = []
        missed_once: list[Job] = []
        with self._tx() as c:
            rows = c.execute(
                "SELECT * FROM jobs WHERE status = 'active' AND next_run_at <= ? AND "
                + self._target_clause(cli) + " ORDER BY next_run_at, rowid LIMIT ?",
                (now, CLI_TARGET, int(limit)),
            ).fetchall()
            for row in rows:
                job = _job_from_row(row)
                scheduled_for = job.next_run_at
                missed = now - scheduled_for > grace_seconds
                if job.kind == "cron":
                    upcoming = schedule_time.next_fire(job.cron or "", max(now, scheduled_for), job.timezone)
                    new_status, new_next = ("active", upcoming) if upcoming is not None else ("done", scheduled_for)
                else:
                    new_status, new_next = "done", scheduled_for
                cur = c.execute(
                    "INSERT OR IGNORE INTO runs(job_id, scheduled_for, started_at, finished_at, status) VALUES (?,?,?,?,?)",
                    (job.id, scheduled_for, now, now if missed else None, "missed" if missed else "claimed"),
                )
                inserted = cur.rowcount == 1
                run_id = cur.lastrowid
                c.execute(
                    "UPDATE jobs SET status = ?, next_run_at = ?, updated_at = ?,"
                    " last_status = CASE WHEN ? THEN 'missed' ELSE last_status END WHERE id = ?",
                    (new_status, new_next, now, 1 if (missed and inserted) else 0, job.id),
                )
                c.execute(
                    "DELETE FROM runs WHERE job_id = ? AND id NOT IN"
                    " (SELECT id FROM runs WHERE job_id = ? ORDER BY id DESC LIMIT ?)",
                    (job.id, job.id, RUN_HISTORY_PER_JOB),
                )
                if not inserted:        # 同一 (job, scheduled_for) 已经有记录：只推进，不重复执行
                    continue
                if missed:
                    if job.kind == "once":
                        missed_once.append(job)
                else:
                    claims.append(Claim(job=job, run_id=run_id, scheduled_for=scheduled_for))
        return claims, missed_once

    def finish_run(
        self,
        run_id: int,
        status: str,
        *,
        error: Optional[str] = None,
        now: Optional[float] = None,
        count_failure: bool = True,
        max_failures: Optional[int] = None,
        pause_reason: Optional[str] = None,
    ) -> Optional[Job]:
        """收尾一次 claimed 的 run，并更新任务的 last_run_at / last_status / run_count / consecutive_failures。

        status：ok / failed / skipped。只有 claimed 的 run 会被更新：已经被别处收尾的（如 stop() 写成了 cancelled）
        直接返回任务、不重复计数。任务已被删除（取消）时返回 None，不算错。
        failed 且 count_failure 时连续失败 +1，ok 清零，skipped 与 count_failure=False 的 failed 不动它。
        pause_reason 给了就把仍是 active 的任务暂停（如推送被拒）；连续失败达到 max_failures 也暂停（reason=failures）。
        一次性任务已经是 done，不会被改成 paused，只记录结果。返回更新后的任务。
        """
        if status not in ("ok", "failed", "skipped"):
            raise ValueError("finish_run 的 status 只能是 ok / failed / skipped")
        if pause_reason is not None and pause_reason not in _PAUSE_REASONS:
            raise ValueError("暂停原因不合法")
        now = time.time() if now is None else float(now)
        with self._tx() as c:
            run = c.execute("SELECT job_id FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            updated = c.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE id = ? AND status = 'claimed'",
                (status, now, str(error)[:200] if error else None, run_id),
            ).rowcount
            row = c.execute("SELECT * FROM jobs WHERE id = ?", (run["job_id"],)).fetchone()
            if row is None:
                return None
            if not updated:
                return _job_from_row(row)
            failures = row["consecutive_failures"]
            if status == "ok":
                failures = 0
            elif status == "failed" and count_failure:
                failures += 1
            new_status, new_reason = row["status"], row["paused_reason"]
            if new_status == "active":
                if pause_reason is not None:
                    new_status, new_reason = "paused", pause_reason
                elif status == "failed" and count_failure and max_failures and failures >= max_failures:
                    new_status, new_reason = "paused", "failures"
            c.execute(
                "UPDATE jobs SET last_run_at = ?, last_status = ?, run_count = run_count + ?, consecutive_failures = ?,"
                " status = ?, paused_reason = ?, updated_at = ? WHERE id = ?",
                (now, status, 1 if status in ("ok", "failed") else 0, failures, new_status, new_reason, now, row["id"]),
            )
            return _job_from_row(c.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone())

    def next_due_at(self, *, cli: bool) -> Optional[float]:
        """这个调度器（cli 或 QQ）最早的下一次触发时刻；没有进行中的任务返回 None。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(next_run_at) FROM jobs WHERE status = 'active' AND " + self._target_clause(cli),
                (CLI_TARGET,),
            ).fetchone()
        return row[0] if row and row[0] is not None else None

    def recover_stale_runs(
        self,
        now: float,
        *,
        stale_after: Optional[float] = None,
        cli: Optional[bool] = None,
    ) -> list[Job]:
        """启动时把超过 AGENT_TURN_TIMEOUT_SECONDS + 300 秒仍是 claimed 的 run 标成 failed(process_restart)，不重跑。

        这个阈值保证第二个进程启动时不会误伤第一个进程正在跑的任务。返回受影响的一次性任务（用来通知用户“当时没有执行”），
        重复任务静默。cli 给了就只处理对应调度器的任务（True 只处理 CLI 的，False 只处理 QQ 的）。"""
        threshold = (config.AGENT_TURN_TIMEOUT_SECONDS + 300) if stale_after is None else float(stale_after)
        sql = ("SELECT r.id AS run_id, r.started_at AS run_started, j.* FROM runs r JOIN jobs j ON j.id = r.job_id"
               " WHERE r.status = 'claimed' AND r.started_at < ?")
        args: list = [now - threshold]
        if cli is not None:
            sql += " AND " + ("j.target_openid = ?" if cli else "j.target_openid != ?")
            args.append(CLI_TARGET)
        affected: list[Job] = []
        with self._tx() as c:
            for row in c.execute(sql, args).fetchall():
                c.execute("UPDATE runs SET status = 'failed', finished_at = ?, error = 'process_restart' WHERE id = ?",
                          (now, row["run_id"]))
                c.execute("UPDATE jobs SET last_status = 'failed', last_run_at = ?, updated_at = ? WHERE id = ?",
                          (row["run_started"], now, row["id"]))
                if row["kind"] == "once":
                    affected.append(_job_from_row(row))
        return affected

    def purge_done(self, now: float, *, retention: Optional[float] = None) -> int:
        """删除完成超过 retention 秒（默认 30 天）的一次性任务及其 run；返回删了几个。"""
        cutoff = now - (DONE_RETENTION_SECONDS if retention is None else float(retention))
        with self._tx() as c:
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM jobs WHERE status = 'done' AND updated_at < ?", (cutoff,)).fetchall()]
            for job_id in ids:
                c.execute("DELETE FROM runs WHERE job_id = ?", (job_id,))
                c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return len(ids)

    def get_runs(self, job_id: str) -> list[Run]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM runs WHERE job_id = ? ORDER BY id", (job_id,)).fetchall()
        return [_run_from_row(r) for r in rows]

    def get_run(self, run_id: int) -> Optional[Run]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _run_from_row(row) if row else None

    # ── “首个任务提示已显示”标记 ──────────────────────────────

    def take_first_hint(self, user_id: str) -> bool:
        """第一次调用返回 True（并记下），之后对同一个用户都返回 False。用来只给每个用户的第一个任务附一次提示。"""
        key = f"first_hint:{validate_user_id(user_id)}"
        with self._tx() as c:
            return c.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, '1')", (key,)).rowcount == 1


# ── 模块级单例（惰性创建；测试用 set_schedule_store 注入，风格同 session_store.set_store）──

_store: Optional[ScheduleStore] = None
_store_lock = threading.Lock()


def get_schedule_store() -> ScheduleStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = ScheduleStore(config.SCHEDULES_DB_PATH)
        return _store


def set_schedule_store(store: Optional[ScheduleStore]) -> None:
    global _store
    with _store_lock:
        _store = store


def close_schedule_store() -> None:
    global _store
    with _store_lock:
        store, _store = _store, None
    if store is not None:
        store.close()
