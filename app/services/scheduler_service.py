"""定时任务调度器：到点认领、无人值守地执行、经 QQ 主动消息推送 (所在层: L5)

只依赖存储（schedule_store / session_store）与时间工具，不 import chat_service（循环依赖且太重）：
调度器通过 start() 的参数接收两个可调用对象——
  run_turn(*, job_id, user_id, target_openid, title, prompt, timezone, run_ref) -> AgentRunResult   跑一次受限 Agent
  push(openid, text) -> None                                                                       主动推送，失败抛 PushError
单实例：asyncio 对象（Event / Lock / Task）只在 start() 里创建、stop() 里清掉。

保证：
  - 至多一次：认领在 claim_due 的一个事务里完成并立刻推进 next_run_at，崩溃后不重跑；
  - 错过窗口：超过 SCHEDULER_MISFIRE_GRACE_SECONDS 的任务记为错过，一次性任务通知用户一次，重复任务静默跳过；
  - 推送失败按类型处理（见 _push_with_policy）；被用户拒收就暂停该任务，且不再向他推任何失败通知；
  - 日志里 openid 与 job id 只显示前 8 位，不记录任务正文。
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from app import config
from app.infra import schedule_store, session_store
from app.utils import schedule_time
from app.utils.identity import OWNER_USER_ID
from app.utils.text import URL_REGEX

_MAX_IDLE_SLEEP = 30.0          # 没有更近的到点时刻时，最久睡多久（也是外部写入没通知到时的兜底）
_CLAIM_BATCH_MAX = 20
_AGENT_TIMEOUT_MARGIN = 30      # 整体超时 = AGENT_TURN_TIMEOUT_SECONDS + SCHEDULER_LOCK_WAIT_SECONDS + 这个余量
_TRUNCATED_SUFFIX = "（内容过长已截断）"

_MISSED_NOTICE = "「{title}」原定 {clock} 触发，当时机器人不在线，没有执行。"
_HITL_NOTICE = "定时任务「{title}」执行时需要你确认一项操作，无人值守的任务不会执行需要授权的操作，已放弃本次执行。"
_FAIL_NOTICE = "⚠️ 定时任务「{title}」这次没能完成（{reason}）"
_PAUSED_SUFFIX = "。连续失败次数过多，已暂停这个任务，问题解决后可发 /tasks resume 恢复"
_FAIL_REASONS = {
    "busy": "你当时正在和我对话，一直没排上",
    "timeout": "执行超时",
    "empty": "没有得到可以发送的结果",
    "unavailable": "定时任务功能暂时不可用",
    "error": "执行时出了错",
}


class PushError(Exception):
    """推送失败。kind：permanent（无权限 / 用户拒收）| transient（临时，可重试）| dedupe | too_long | url_rejected。

    code 是平台错误码（可为 None），detail 是简短说明（不含任务正文）。"""

    def __init__(self, kind: str, code: Optional[int] = None, detail: str = ""):
        super().__init__(kind, code, detail)
        self.kind = kind
        self.code = code
        self.detail = str(detail)[:200]

    def __str__(self) -> str:
        return f"PushError({self.kind}, code={self.code}, {self.detail})"


@dataclass(frozen=True)
class AgentRunResult:
    """run_turn 的结果。error：busy | hitl | empty | timeout | stopped | unavailable | error（ok 为 True 时为空串）。"""
    ok: bool
    text: str = ""
    error: str = ""


@dataclass(frozen=True)
class _PushOutcome:
    ok: bool
    kind: str = ""          # 失败时：permanent / transient / dedupe / too_long / url_rejected / error


RunTurn = Callable[..., Awaitable[AgentRunResult]]
Push = Callable[[str, str], Awaitable[None]]


def _tag(value: str) -> str:
    return (value or "")[:8]


def _clip(text: str) -> str:
    limit = config.SCHEDULER_PUSH_MAX_CHARS
    return text if len(text) <= limit else text[:limit] + _TRUNCATED_SUFFIX


class _Scheduler:
    def __init__(self, *, run_turn: RunTurn, push: Push, cli: bool, clock: Callable[[], float]):
        self.loop = asyncio.get_running_loop()
        self.run_turn = run_turn
        self.push = push
        self.cli = cli
        self.clock = clock
        self.store = schedule_store.get_schedule_store()
        self.wake = asyncio.Event()
        self.push_lock = asyncio.Lock()
        self.last_push = 0.0                           # time.monotonic()，全局推送节流用
        self.main_task: Optional[asyncio.Task] = None
        self.tasks: set[asyncio.Task] = set()          # 正在执行的任务（占并发名额）
        self.side_tasks: set[asyncio.Task] = set()     # “错过 / 被中断”的通知（不占名额）
        self.inflight: dict[int, schedule_store.Claim] = {}
        self.startup_notices: list[schedule_store.Job] = []
        self.stopped = False

    # ── 生命周期 ──────────────────────────────────────────────

    def start(self) -> None:
        now = self.clock()
        try:
            self.startup_notices = self.store.recover_stale_runs(now, cli=self.cli)
            self.store.purge_done(now)
        except Exception as e:
            print(f"[scheduler] 启动时的恢复 / 清理失败: {type(e).__name__}")
        self.store.add_change_listener(self.notify)
        self.main_task = asyncio.create_task(self._run_loop(), name="scheduler-loop")
        print(f"⏰ [scheduler] 已启动（{'CLI' if self.cli else 'QQ'}），恢复了 {len(self.startup_notices)} 个被中断的一次性任务")

    async def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        self.detach()
        pending = [t for t in (self.main_task, *self.tasks, *self.side_tasks) if t is not None]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        # 还没来得及开跑就被取消的认领：_execute 的 CancelledError 处理不会触发，在这里补写（尽力而为，不抛）
        for run_id in list(self.inflight):
            self._finish(run_id, "failed", error="cancelled", count_failure=False)
        self.inflight.clear()
        print("⏰ [scheduler] 已停止")

    def detach(self) -> None:
        try:
            self.store.remove_change_listener(self.notify)
        except Exception:
            pass

    def notify(self) -> None:
        """存储发生变化（创建 / 取消 / 暂停 / 恢复）：唤醒循环重算下一个到点时刻。可能来自 LangChain 的工具线程，必须线程安全。"""
        try:
            self.loop.call_soon_threadsafe(self.wake.set)
        except RuntimeError:            # loop 已经关闭
            pass

    # ── 主循环 ────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        for job in self.startup_notices:
            self._spawn_notice(job)
        self.startup_notices = []
        while not self.stopped:
            try:
                delay = self._tick()
            except Exception as e:          # 存储暂时出错不能让循环死掉
                print(f"[scheduler] 调度一轮出错: {type(e).__name__}")
                delay = 5.0
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            self.wake.clear()

    def _tick(self) -> float:
        """认领一批到点的任务并起执行任务；返回下一次醒来前最久睡多少秒。"""
        free = config.SCHEDULER_MAX_CONCURRENT_RUNS - len(self.tasks)
        if free > 0:
            claims, missed = self.store.claim_due(self.clock(), cli=self.cli, limit=min(free, _CLAIM_BATCH_MAX))
            for job in missed:
                self._spawn_notice(job)
            for claim in claims:
                self._spawn_run(claim)
            free = config.SCHEDULER_MAX_CONCURRENT_RUNS - len(self.tasks)
        if free <= 0:
            return _MAX_IDLE_SLEEP         # 名额满了：任何一个执行任务收尾都会唤醒循环
        due = self.store.next_due_at(cli=self.cli)
        if due is None:
            return _MAX_IDLE_SLEEP
        return min(_MAX_IDLE_SLEEP, max(0.01, due - self.clock()))

    def _spawn_run(self, claim: schedule_store.Claim) -> None:
        self.inflight[claim.run_id] = claim
        task = asyncio.create_task(self._execute(claim), name=f"scheduled-{_tag(claim.job.id)}")
        self.tasks.add(task)
        task.add_done_callback(self._run_done)

    def _run_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if not self.stopped:
            self.wake.set()                # 名额空出来了：有排队的到点任务就接着认领

    def _spawn_notice(self, job: schedule_store.Job) -> None:
        task = asyncio.create_task(self._send_missed_notice(job), name=f"scheduled-notice-{_tag(job.id)}")
        self.side_tasks.add(task)
        task.add_done_callback(self.side_tasks.discard)

    # ── 执行一次 ──────────────────────────────────────────────

    def _finish(self, run_id: int, status: str, **kwargs) -> Optional[schedule_store.Job]:
        """收尾一个 run；存储出错只打印，不向上抛（任务已被删除时 finish_run 本身返回 None）。"""
        if status == "failed" and "max_failures" not in kwargs:
            kwargs["max_failures"] = config.SCHEDULER_MAX_CONSECUTIVE_FAILURES
        try:
            return self.store.finish_run(run_id, status, now=self.clock(), **kwargs)
        except Exception as e:
            print(f"[scheduler] 收尾 run 失败: {type(e).__name__}")
            return None

    async def _execute(self, claim: schedule_store.Claim) -> None:
        job = claim.job
        try:
            await self._execute_inner(claim)
        except asyncio.CancelledError:
            self._finish(claim.run_id, "failed", error="cancelled", count_failure=False)
            raise
        except Exception as e:
            print(f"[scheduler] job={_tag(job.id)} 执行异常: {type(e).__name__}")
            self._finish(claim.run_id, "failed", error=f"exception:{type(e).__name__}")
        finally:
            self.inflight.pop(claim.run_id, None)

    def _is_blocked(self, job: schedule_store.Job) -> bool:
        if job.user_id == OWNER_USER_ID:
            return False
        try:
            info = session_store.get_store().get_user(job.user_id)
        except Exception as e:                      # 查不到就不推：宁可少发一条，也不向被封禁的用户推送
            print(f"[scheduler] 查询用户封禁状态失败，本次跳过: {type(e).__name__}")
            return True
        return bool(info and info.blocked)

    async def _execute_inner(self, claim: schedule_store.Claim) -> None:
        job, run_id = claim.job, claim.run_id
        print(f"⏰ [scheduler] 触发 job={_tag(job.id)} user={_tag(job.user_id)} mode={job.mode}")
        if self._is_blocked(job):
            self._finish(run_id, "skipped", error="blocked")
            return
        deadline = claim.scheduled_for + config.SCHEDULER_MISFIRE_GRACE_SECONDS
        if job.mode == "agent":
            result = await self._run_agent(claim, deadline)
            if result.ok and not result.text.strip():
                result = AgentRunResult(False, "", "empty")
            if not result.ok:
                await self._handle_agent_failure(claim, result.error or "error", deadline)
                return
            body = _clip(result.text.strip())
            text = body
        else:
            body = _clip(job.prompt)
            text = "⏰ " + body
        outcome = await self._push_with_policy(job.target_openid, text, deadline)
        if outcome.ok:
            self._record_push_event(job, run_id, body)
            self._finish(run_id, "ok")
        elif outcome.kind == "permanent":
            print(f"[scheduler] 推送被拒收，暂停任务 job={_tag(job.id)} openid={_tag(job.target_openid)}")
            self._finish(run_id, "failed", error="push_refused", pause_reason="push_refused", count_failure=False)
        else:
            print(f"[scheduler] 推送失败 job={_tag(job.id)} kind={outcome.kind}")
            self._finish(run_id, "failed", error=f"push_{outcome.kind}")

    async def _run_agent(self, claim: schedule_store.Claim, deadline: float) -> AgentRunResult:
        """跑一次受限 Agent。busy（用户的锁一直拿不到）按间隔重试，次数与时间都有上限。"""
        job = claim.job
        total_timeout = config.AGENT_TURN_TIMEOUT_SECONDS + config.SCHEDULER_LOCK_WAIT_SECONDS + _AGENT_TIMEOUT_MARGIN
        attempt = 0
        while True:
            try:
                result = await asyncio.wait_for(
                    self.run_turn(
                        job_id=job.id, user_id=job.user_id, target_openid=job.target_openid, title=job.title,
                        prompt=job.prompt, timezone=job.timezone, run_ref=f"{claim.run_id}-{attempt}",
                    ),
                    timeout=total_timeout,
                )
            except asyncio.TimeoutError:
                return AgentRunResult(False, "", "timeout")
            except Exception as e:
                print(f"[scheduler] job={_tag(job.id)} run_turn 抛出异常: {type(e).__name__}")
                return AgentRunResult(False, "", "error")
            if not isinstance(result, AgentRunResult):
                return AgentRunResult(False, "", "error")
            if (result.error == "busy" and attempt < config.SCHEDULER_RETRY_MAX
                    and self.clock() + config.SCHEDULER_RETRY_DELAY_SECONDS <= deadline):
                attempt += 1
                await asyncio.sleep(config.SCHEDULER_RETRY_DELAY_SECONDS)
                continue
            return result

    async def _handle_agent_failure(self, claim: schedule_store.Claim, error: str, deadline: float) -> None:
        job = claim.job
        counted = error not in ("busy", "stopped")
        updated = self._finish(claim.run_id, "failed", error=error, count_failure=counted)
        print(f"[scheduler] agent 任务失败 job={_tag(job.id)} error={error}")
        if error == "stopped":
            return                                  # 用户自己 /stop 的，不必再通知
        paused = bool(updated and updated.status == "paused" and updated.paused_reason == "failures")
        if error == "hitl":
            text = _HITL_NOTICE.format(title=job.title)
        else:
            text = _FAIL_NOTICE.format(title=job.title, reason=_FAIL_REASONS.get(error, _FAIL_REASONS["error"]))
        if paused:
            text += _PAUSED_SUFFIX
        outcome = await self._push_with_policy(job.target_openid, text, deadline)
        if not outcome.ok and outcome.kind == "permanent":
            self._pause_refused(job)                # 他拒收了：暂停，之后不再向他推任何失败通知

    def _pause_refused(self, job: schedule_store.Job) -> None:
        try:
            self.store.pause_job(job.id, "push_refused")
        except schedule_store.ScheduleStoreError:
            pass                                    # 任务已被删除 / 已完成
        except Exception as e:
            print(f"[scheduler] 暂停被拒收的任务失败: {type(e).__name__}")

    async def _send_missed_notice(self, job: schedule_store.Job) -> None:
        clock_text = schedule_time.fmt_local(job.next_run_at, job.timezone)[-5:]
        outcome = await self._push_with_policy(
            job.target_openid, _MISSED_NOTICE.format(title=job.title, clock=clock_text),
            self.clock() + config.SCHEDULER_MISFIRE_GRACE_SECONDS,
        )
        if not outcome.ok:
            print(f"[scheduler] 错过通知没有发出 job={_tag(job.id)} kind={outcome.kind}")

    def _record_push_event(self, job: schedule_store.Job, run_id: int, body: str) -> None:
        """把推送内容写进用户当前活动会话的事件日志（不是入站事件，不触发 wake）。写失败只打印。"""
        try:
            store = session_store.get_store()
            session_id = store.get_active_session_id(user_id=job.user_id)
            store.emit_event(
                session_id, session_store.EVENT_SCHEDULED_PUSH,
                {"job_id": job.short_id, "title": job.title, "text": body, "mode": job.mode, "run_id": run_id},
                key=f"sp:{run_id}",
            )
        except Exception as e:
            print(f"[scheduler] 写入定时推送事件失败: {type(e).__name__}")

    # ── 推送 ──────────────────────────────────────────────────

    async def _throttled_push(self, openid: str, text: str) -> None:
        """全局节流：相邻两次推送至少隔 SCHEDULER_PUSH_MIN_INTERVAL_SECONDS（持锁串行发送）。"""
        async with self.push_lock:
            wait = self.last_push + config.SCHEDULER_PUSH_MIN_INTERVAL_SECONDS - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                await self.push(openid, text)
            finally:
                self.last_push = time.monotonic()

    async def _push_with_policy(self, openid: str, text: str, deadline: float) -> _PushOutcome:
        """按错误类型处理一次推送：
        transient 每隔 SCHEDULER_RETRY_DELAY_SECONDS 重试，最多 SCHEDULER_RETRY_MAX 次且不超过 deadline；
        dedupe 在末尾加 [MM-DD HH:MM] 后重试一次；too_long 截断后重试一次；url_rejected 去链接后重试一次；
        permanent 立即放弃。"""
        sent = text
        retries = 0
        adjusted: set[str] = set()
        while True:
            try:
                await self._throttled_push(openid, sent)
                return _PushOutcome(True)
            except asyncio.CancelledError:
                raise
            except PushError as e:
                kind = e.kind
                print(f"[scheduler] 推送失败 openid={_tag(openid)} kind={kind} code={e.code}")
            except Exception as e:                  # push 实现里没归类的异常按临时错误处理
                kind = "transient"
                print(f"[scheduler] 推送异常 openid={_tag(openid)} {type(e).__name__}")
            if kind == "permanent":
                return _PushOutcome(False, "permanent")
            if kind in ("dedupe", "too_long", "url_rejected"):
                if kind in adjusted:
                    return _PushOutcome(False, kind)
                adjusted.add(kind)
                if kind == "dedupe":
                    sent = f"{sent}\n[{schedule_time.fmt_local(self.clock())}]"
                elif kind == "too_long":
                    sent = sent[: max(100, len(sent) // 2)] + _TRUNCATED_SUFFIX
                else:
                    sent = URL_REGEX.sub("（链接已省略）", sent)
                continue
            # transient（含未知类型）
            if retries >= config.SCHEDULER_RETRY_MAX or self.clock() + config.SCHEDULER_RETRY_DELAY_SECONDS > deadline:
                return _PushOutcome(False, "transient")
            retries += 1
            await asyncio.sleep(config.SCHEDULER_RETRY_DELAY_SECONDS)


# ── 模块级接口 ──────────────────────────────────────────────

_current: Optional[_Scheduler] = None


def start(*, run_turn: RunTurn, push: Push, cli: bool = False, clock: Callable[[], float] = time.time) -> None:
    """在运行中的事件循环里启动调度器：先恢复被中断的 run，再起主循环。同一进程只能有一个。"""
    global _current
    if _current is not None and not _current.stopped:
        raise RuntimeError("调度器已经在运行")
    scheduler = _Scheduler(run_turn=run_turn, push=push, cli=cli, clock=clock)
    _current = scheduler
    scheduler.start()


async def stop() -> None:
    """取消主循环与在途任务；被取消的 claimed run 立即写成 failed(cancelled)。没启动过就什么都不做。"""
    scheduler = _current
    if scheduler is not None:
        await scheduler.stop()


def notify_changed() -> None:
    """线程安全：唤醒调度循环重算下一个到点时刻（loop 已关闭时吞掉异常）。没启动就什么都不做。"""
    scheduler = _current
    if scheduler is not None:
        scheduler.notify()


def is_running() -> bool:
    return _current is not None and not _current.stopped


def reset_for_tests() -> None:
    """测试用：丢弃模块级状态。asyncio 对象绑定在创建它们的事件循环上，每次 asyncio.run 之前都要先调用。"""
    global _current
    scheduler, _current = _current, None
    if scheduler is not None:
        scheduler.detach()
