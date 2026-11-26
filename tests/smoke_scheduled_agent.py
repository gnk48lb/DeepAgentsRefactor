"""
tests/smoke_scheduled_agent.py

无人值守的 agent 定时任务（chat_service.run_scheduled_turn）离线冒烟测试：假 agent、假 checkpointer，不碰任何模型 / 网络 / Modal。
  A 成功路径：输入块、config、请求上下文（is_scheduled，require_owner 抛 PermissionError）、临时线程、结束后删线程
  B 失败分类：unavailable / empty / hitl / timeout / error / busy / stopped
  C 不产生任何会话副作用：不访问 SessionStore、不查记忆、不镜像日志、不 _finalize_round、不提取记忆、不清理嵌套 checkpoint
  D _agent_for 永远不返回 scheduled 图；用户锁与 /stop 句柄（source=scheduled）
  E /stop 短路与“忙”提示（不碰任何终止原语）
  F member 信号量（先锁后信号量，owner 不占用）与取消
"""
import asyncio
import contextlib
import io
import os
import shutil
import sys
import tempfile
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from app import config
from app.infra import checkpointer, modal_sandbox, session_store
from app.memory import extractor, injector
from app.runtime import outbox
from app.services import chat_service as cs
from app.services import scheduler_service, session_log
from app.utils.context import get_request_context, require_owner
from app.utils.identity import Identity

U1 = "A1B2C3D4E5F6A7B8"
U2 = "B1B2C3D4E5F6A7B8"
OWNER_RAW = "C1B2C3D4E5F6A7B8"
JOB = "0123456789abcdef0123456789abcdef"
AgentRunResult = scheduler_service.AgentRunResult


class _State:
    def __init__(self, messages, interrupts=()):
        self.values = {"messages": messages}
        self.interrupts = tuple(interrupts)


class FakeAgent:
    """假的 scheduled 图：记录每次 ainvoke 看到的输入 / config / 请求上下文，行为由 behavior 协程控制。"""

    def __init__(self, final="北京今天晴，不用带伞。", *, interrupts=(), behavior=None, raises=None):
        self.final = final
        self.interrupts = interrupts
        self.behavior = behavior
        self.raises = raises
        self.invocations: list[dict] = []
        self.state_calls: list[dict] = []
        self.active = 0
        self.max_active = 0

    async def ainvoke(self, inputs, cfg):
        owner_error = None
        try:
            require_owner()
        except (PermissionError, RuntimeError) as e:
            owner_error = e
        self.invocations.append({"inputs": inputs, "cfg": cfg, "ctx": get_request_context(), "owner_error": owner_error})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            outbox.push({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})      # 图片一律丢弃，只推文本
            if self.behavior is not None:
                await self.behavior()
            if self.raises is not None:
                raise self.raises
            return {"messages": []}
        finally:
            self.active -= 1

    async def aget_state(self, cfg):
        self.state_calls.append(cfg)
        messages = [HumanMessage(content="x")]
        if self.final is not None:
            messages.append(AIMessage(content=self.final))
        return _State(messages, self.interrupts)


class FakeSaver:
    def __init__(self):
        self.deleted: list[str] = []

    async def adelete_thread(self, thread_id):
        self.deleted.append(thread_id)


class _Forbidden:
    """把一批“不应被调用”的函数换成会炸的替身（C 组：无人值守的任务不产生任何会话副作用）。"""

    TARGETS = [
        (session_store, "get_store"),
        (injector, "build_initial_messages"),
        (extractor, "extract_and_store_memory"),
        (session_log, "mirror_messages"),
        (checkpointer, "prune_nested_checkpoints"),
        (cs, "_finalize_round"),
    ]

    def __enter__(self):
        self.originals = [(mod, name, getattr(mod, name)) for mod, name in self.TARGETS]
        self.hits: list[str] = []
        for mod, name in self.TARGETS:
            def boom(*args, _name=name, **kwargs):
                self.hits.append(_name)
                raise AssertionError(f"无人值守的定时任务不应调用 {_name}")
            setattr(mod, name, boom)
        return self

    def __exit__(self, *exc):
        for mod, name, original in self.originals:
            setattr(mod, name, original)


@contextlib.contextmanager
def env(agent=None, *, patched=None):
    """干净的 chat_service 运行时状态 + 假图 + 假 checkpointer；退出时还原。"""
    saved = (cs._agents, cs._checkpointer, {k: getattr(config, k) for k in (patched or {})})
    cs._reset_runtime_state()
    saver = FakeSaver()
    cs._agents = {"owner": object(), "member": object()}
    if agent is not None:
        cs._agents["scheduled"] = agent
    cs._checkpointer = saver
    for k, v in (patched or {}).items():
        setattr(config, k, v)
    try:
        with contextlib.redirect_stdout(io.StringIO()):          # cli_logger 的回合日志太吵；需要看输出的用例自己再套一层 redirect
            yield saver
    finally:
        cs._agents, cs._checkpointer = saved[0], saved[1]
        for k, v in saved[2].items():
            setattr(config, k, v)
        cs._reset_runtime_state()


def run_turn(user_id=U1, **overrides):
    kwargs = dict(job_id=JOB, user_id=user_id, target_openid=OWNER_RAW if user_id == "owner" else user_id,
                  title="带伞提醒", prompt="查询北京今天的天气，用一两句话提醒用户出门要不要带伞", timezone="Asia/Shanghai", run_ref="7-0")
    kwargs.update(overrides)
    return cs.run_scheduled_turn(**kwargs)


async def wait_until(cond, timeout=3.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if cond():
            return True
        await asyncio.sleep(0.01)
    return False


def raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        return e
    raise AssertionError(f"应当抛出 {exc.__name__}")


# ════════════════════════ A 成功路径 ════════════════════════

def test_a_success_path():
    for user_id in ("owner", U1):
        agent = FakeAgent()
        with env(agent) as saver, _Forbidden() as forbidden:
            result = asyncio.run(run_turn(user_id))
            assert result == AgentRunResult(True, "北京今天晴，不用带伞。", ""), result
            assert forbidden.hits == []
            assert len(agent.invocations) == 1
            call = agent.invocations[0]
            # 输入：一条 HumanMessage，内容是 <scheduled_task> 块（标题 / 业务时区的触发时间 / 指令原文）
            messages = call["inputs"]["messages"]
            assert len(messages) == 1 and isinstance(messages[0], HumanMessage)
            body = messages[0].content
            assert body.startswith("<scheduled_task>\n标题：带伞提醒\n触发时间（业务时区）：") and body.endswith("\n指令：\n查询北京今天的天气，用一两句话提醒用户出门要不要带伞\n</scheduled_task>"), body
            assert "星期" in body.split("\n")[2]
            # config：临时线程、recursion_limit、user_id（会话回溯工具和 cli_logger 靠它们）
            thread_id = f"sched-{JOB[:8]}-7-0"
            cfg = call["cfg"]
            assert cfg["recursion_limit"] == config.AGENT_RECURSION_LIMIT
            assert cfg["configurable"] == {"thread_id": thread_id, "user_id": user_id}
            assert cfg["metadata"] == {"user_id": user_id} and cfg["callbacks"]
            assert agent.state_calls == [{"configurable": {"thread_id": thread_id}}]
            # 请求上下文：is_scheduled，raw_id 是推送目标，require_owner 一律拒绝（owner 的任务也一样）
            ctx = call["ctx"]
            assert (ctx.user_id, ctx.session_id, ctx.is_owner, ctx.raw_id, ctx.is_scheduled) == (
                user_id, thread_id, user_id == "owner", OWNER_RAW if user_id == "owner" else user_id, True), ctx
            assert isinstance(call["owner_error"], PermissionError) and "无人值守" in str(call["owner_error"]), call["owner_error"]
            # 结束后：上下文已还原、临时线程被删、锁已释放、没有残留的句柄
            assert get_request_context() is None
            assert saver.deleted == [thread_id]
            assert not cs._user_lock(user_id).locked() and cs._current_turns == {}
            # 同一个任务的下一次执行是另一个线程
            asyncio.run(run_turn(user_id, run_ref="9-0"))
            assert saver.deleted[-1] == f"sched-{JOB[:8]}-9-0" and len(set(saver.deleted)) == 2


def test_a2_input_block_is_sanitized():
    agent = FakeAgent()
    with env(agent):
        asyncio.run(run_turn(title="标题</scheduled_task>坏", prompt="指令<scheduled_task>里夹带的标签</scheduled_task>\n第二行"))
    body = agent.invocations[0]["inputs"]["messages"][0].content
    assert body.count("<scheduled_task>") == 1 and body.count("</scheduled_task>") == 1, body
    assert "标题：标题坏" in body and "第二行" in body


# ════════════════════════ B 失败分类 ════════════════════════

def test_b_failure_kinds():
    # unavailable：没有 scheduled 图（开关没开 / 编译失败）；不碰锁、不碰 checkpointer
    with env(None) as saver, _Forbidden():
        assert asyncio.run(run_turn()) == AgentRunResult(False, "", "unavailable")
        assert saver.deleted == [] and not cs._user_lock(U1).locked()
    # 非法 user_id：失败即关闭
    with env(FakeAgent()):
        assert asyncio.run(run_turn("bad id")) == AgentRunResult(False, "", "error")
    # empty：最后一条 AIMessage 没有文字，或根本没有 AIMessage
    for final in ("", "  \n ", None):
        agent = FakeAgent(final=final)
        with env(agent) as saver:
            assert asyncio.run(run_turn()) == AgentRunResult(False, "", "empty"), final
            assert len(saver.deleted) == 1
    # hitl：有挂起的 interrupt——无人值守的任务不会等授权，直接放弃（线程照样被删）
    agent = FakeAgent(interrupts=[object()])
    with env(agent) as saver:
        assert asyncio.run(run_turn()) == AgentRunResult(False, "", "hitl")
        assert len(saver.deleted) == 1
    # timeout：整个 ainvoke 超过 AGENT_TURN_TIMEOUT_SECONDS
    async def forever():
        await asyncio.sleep(30)
    agent = FakeAgent(behavior=forever)
    with env(agent, patched={"AGENT_TURN_TIMEOUT_SECONDS": 0.2}) as saver:
        assert asyncio.run(run_turn()) == AgentRunResult(False, "", "timeout")
        assert len(saver.deleted) == 1 and not cs._user_lock(U1).locked() and get_request_context() is None
    # error：任何异常（含递归上限）都转成 error，不抛给调度器；锁与上下文都已释放
    for exc in (RuntimeError("boom"), GraphRecursionError("too deep"), ValueError("x")):
        agent = FakeAgent(raises=exc)
        buf = io.StringIO()
        with env(agent) as saver, contextlib.redirect_stdout(buf):
            assert asyncio.run(run_turn()) == AgentRunResult(False, "", "error"), exc
            assert len(saver.deleted) == 1 and not cs._user_lock(U1).locked()
        assert "boom" not in buf.getvalue() and "too deep" not in buf.getvalue(), "日志里只记异常类型，不记内容"
    # checkpointer 删线程失败：不影响结果
    class BrokenSaver(FakeSaver):
        async def adelete_thread(self, thread_id):
            raise RuntimeError("db locked")
    agent = FakeAgent()
    with env(agent):
        cs._checkpointer = BrokenSaver()
        assert asyncio.run(run_turn()).ok is True
    # busy：用户的锁一直被占用（用户正在聊天）：等 SCHEDULER_LOCK_WAIT_SECONDS 之后返回 busy，图一次都没被调用
    agent = FakeAgent()
    with env(agent, patched={"SCHEDULER_LOCK_WAIT_SECONDS": 0.2}) as saver:
        async def scenario():
            lock = cs._user_lock(U1)
            await lock.acquire()
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            result = await run_turn()
            assert result == AgentRunResult(False, "", "busy") and 0.15 <= loop.time() - t0 < 2.0
            assert agent.invocations == [] and saver.deleted == []
            assert lock.locked(), "busy 时不得释放别人持有的锁"
            lock.release()
            assert (await run_turn()).ok is True
        asyncio.run(scenario())
        # 锁在等待期间被释放：拿到锁就跑
        async def released_in_time():
            lock = cs._user_lock(U2)
            await lock.acquire()
            asyncio.get_running_loop().call_later(0.05, lock.release)
            return await run_turn(U2)
        assert asyncio.run(released_in_time()).ok is True


# ════════════════════════ C 不产生会话副作用 ════════════════════════

def test_c_no_session_side_effects():
    """A 组已经在 _Forbidden 里跑过成功 / 失败路径；这里再确认：真实的 SessionStore 一个字节都没动，也没有任何会话被创建。"""
    tmp = tempfile.mkdtemp(prefix="sched_agent_")
    store = session_store.SessionStore(os.path.join(tmp, "sessions.db"))
    session_store.set_store(store)
    try:
        sid = store.get_active_session_id(user_id=U1)
        before = (store.count_sessions(), store.count_events(sid), store.get_session(sid, user_id=U1).updated_at)
        agent = FakeAgent()
        with env(agent):
            for user_id in ("owner", U1):
                assert asyncio.run(run_turn(user_id)).ok is True
        after = (store.count_sessions(), store.count_events(sid), store.get_session(sid, user_id=U1).updated_at)
        assert before == after, (before, after)
        assert store.count_sessions(user_id="owner") == 0, "owner 的任务没有给 owner 建出会话"
        assert store.sessions_with_pending() == []
    finally:
        session_store.close_store()
        store.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ════════════════════════ D 句柄与 _agent_for ════════════════════════

def test_d_agent_for_never_returns_scheduled_and_handle_source():
    scheduled, owner_graph, member_graph = FakeAgent(), object(), object()
    with env(scheduled):
        cs._agents.update({"owner": owner_graph, "member": member_graph})
        assert set(cs._agents) == {"owner", "member", "scheduled"}
        for user_id in ("owner", U1, U2, "scheduled", "cli-x"):
            graph = cs._agent_for(user_id)
            assert graph is not scheduled and graph in (owner_graph, member_graph), user_id
        assert cs._agent_for("owner") is owner_graph and cs._agent_for(U1) is member_graph
        assert {cs._role_of(u) for u in ("owner", U1, "scheduled", "cli-x")} == {"owner", "member"}
        cs._agents.pop("owner")
        cs._agents.pop("member")
        raises(RuntimeError, cs._agent_for, "owner")            # 缺图就报错，绝不回落成 scheduled（也不会回落成 owner）
        raises(RuntimeError, cs._agent_for, U1)
    # 句柄：默认 chat；定时任务运行期间登记的是 scheduled，结束后清掉
    handle = cs._TurnHandle(task=None, session_id="s", user_id=U1, started_at=0.0)
    assert handle.source == "chat"
    seen = {}

    async def watch():
        seen["handle"] = cs._current_turns.get(U1)
    agent = FakeAgent(behavior=watch)
    with env(agent):
        asyncio.run(run_turn(U1))
        assert seen["handle"].source == "scheduled" and seen["handle"].user_id == U1 and seen["handle"].session_id == f"sched-{JOB[:8]}-7-0"
        assert cs._current_turns == {}


# ════════════════════════ E /stop 短路与忙提示 ════════════════════════

class _KillSpy:
    """owner 的 /stop 会杀宿主机命令 / 本机沙箱，member 的会中断云端沙箱：定时任务的 /stop 一个都不该碰。"""

    def __enter__(self):
        self.calls: list[str] = []
        self.orig = (cs._kill_side_effects, modal_sandbox.interrupt_session_nowait, session_log.mirror_messages)

        def spy(name):
            def fn(*args, **kwargs):
                self.calls.append(name)
                return 0
            return fn
        cs._kill_side_effects = spy("kill_side_effects")
        modal_sandbox.interrupt_session_nowait = spy("modal_interrupt")
        session_log.mirror_messages = spy("mirror_messages")
        return self

    def __exit__(self, *exc):
        cs._kill_side_effects, modal_sandbox.interrupt_session_nowait, session_log.mirror_messages = self.orig


def test_e_stop_short_circuit_and_busy_hint():
    for user_id in ("owner", U1):
        identity = Identity(user_id=user_id, is_owner=(user_id == "owner"), raw_id=OWNER_RAW if user_id == "owner" else user_id)
        started = {}

        async def hang():
            started["yes"] = True
            await asyncio.sleep(60)
        agent = FakeAgent(behavior=hang)
        with env(agent) as saver, _KillSpy() as spy:
            async def scenario():
                task = asyncio.create_task(run_turn(user_id))
                assert await wait_until(lambda: started.get("yes")), "定时任务没跑起来"
                # 用户在任务运行期间发普通消息：忙提示改成定时任务的措辞，不排队
                reply = await cs._process_unblocked("你好", None, identity)
                assert reply.text == "正在执行你的定时任务，请稍候再发；需要中止请发送 /stop", reply.text
                # /stop：取消、等待收尾，回固定文案；不碰终止原语、不镜像会话、不读待授权状态
                reply = await cs._process_unblocked("/stop", None, identity)
                assert reply.text == "已中止正在运行的定时任务。" and not reply.awaiting_approval, reply
                result = await asyncio.wait_for(task, 3.0)
                assert result == AgentRunResult(False, "", "stopped"), result
                return result
            asyncio.run(scenario())
            assert spy.calls == [], f"scheduled 句柄的 /stop 不得调用 {spy.calls}"
            assert agent.state_calls == [], "也不读取待授权状态"
            assert saver.deleted == [f"sched-{JOB[:8]}-7-0"] and cs._current_turns == {} and not cs._user_lock(user_id).locked()
    # /stop 之后用户可以马上正常使用；/reset 忙时也会先中止定时任务
    identity = Identity(user_id=U1, is_owner=False, raw_id=U1)
    started = {}

    async def hang2():
        started["yes"] = True
        await asyncio.sleep(60)
    agent = FakeAgent(behavior=hang2)
    with env(agent):
        async def after_stop():
            task = asyncio.create_task(run_turn(U1))
            assert await wait_until(lambda: started.get("yes"))
            await cs._stop_current_turn(U1)                 # 与 /stop 同一个入口
            assert (await asyncio.wait_for(task, 3.0)).error == "stopped"
            assert not cs._user_lock(U1).locked()
            assert cs._current_turns == {} and get_request_context() is None
        asyncio.run(after_stop())
    # 普通对话的忙提示保持原样
    with env(FakeAgent()):
        async def normal_busy():
            lock = cs._user_lock(U1)
            await lock.acquire()
            reply = await cs._process_unblocked("你好", None, identity)
            lock.release()
            return reply.text
        text = asyncio.run(normal_busy())
        assert text == "上一条消息还在处理中，请稍候再发。如需中止请发送 /stop", text


# ════════════════════════ F 信号量与取消 ════════════════════════

def test_f_member_semaphore_and_cancellation():
    gate = {"n": 0}

    async def hold():
        gate["n"] += 1
        await asyncio.sleep(0.15)
    # member：全局并发上限 1，两个 member 的定时任务串行；owner 不占名额
    agent = FakeAgent(behavior=hold)
    with env(agent, patched={"MAX_CONCURRENT_MEMBER_TURNS": 1}):
        async def scenario():
            return await asyncio.gather(run_turn(U1), run_turn(U2), run_turn("owner"))
        results = asyncio.run(scenario())
        assert all(r.ok for r in results), results
        by_user = {c["ctx"].user_id: c for c in agent.invocations}
        assert set(by_user) == {U1, U2, "owner"}
        assert agent.max_active == 2, "member 之间串行（信号量 1），owner 与某个 member 可以同时跑"
    # 信号量被占满时 member 的定时任务排队等（持有用户锁期间），释放后继续
    agent = FakeAgent()
    with env(agent, patched={"MAX_CONCURRENT_MEMBER_TURNS": 1}):
        async def blocked():
            sem = cs._member_semaphore()
            await sem.acquire()
            task = asyncio.create_task(run_turn(U1))
            await asyncio.sleep(0.1)
            assert not task.done() and cs._user_lock(U1).locked() and agent.invocations == [], "先锁后信号量：排队期间已持有用户锁"
            sem.release()
            return await asyncio.wait_for(task, 3.0)
        assert asyncio.run(blocked()).ok is True
    # 调度器停机（取消调用方）：CancelledError 往外传，锁释放、上下文还原、临时线程删除
    started = {}

    async def hang():
        started["yes"] = True
        await asyncio.sleep(60)
    agent = FakeAgent(behavior=hang)
    with env(agent) as saver:
        async def cancelled():
            task = asyncio.create_task(run_turn(U1))
            assert await wait_until(lambda: started.get("yes"))
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return "cancelled"
            return "finished"
        assert asyncio.run(cancelled()) == "cancelled"
        assert not cs._user_lock(U1).locked() and saver.deleted == [f"sched-{JOB[:8]}-7-0"]


SCENARIOS = [
    ("A 成功路径：输入块 / config / 请求上下文 / 临时线程", test_a_success_path),
    ("A2 输入块里的标签被清理", test_a2_input_block_is_sanitized),
    ("B 失败分类：unavailable / empty / hitl / timeout / error / busy", test_b_failure_kinds),
    ("C 不产生任何会话副作用", test_c_no_session_side_effects),
    ("D _agent_for 永不返回 scheduled 图；句柄 source", test_d_agent_for_never_returns_scheduled_and_handle_source),
    ("E /stop 短路与忙提示", test_e_stop_short_circuit_and_busy_hint),
    ("F member 信号量与取消", test_f_member_semaphore_and_cancellation),
]


def main():
    print("=" * 70)
    print("开始运行无人值守 agent 定时任务离线冒烟测试...")
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
