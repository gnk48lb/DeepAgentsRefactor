"""
tests/smoke_budget.py

专家独立停止条件（单次委派内的工具调用预算）离线冒烟测试：全程脚本化假模型，无需 API Key / 网络 / Docker。
  A 纯函数与装配
    A1 build_budget_middleware 构造与参数校验
    A2 config 预算表完整性与 DesktopAgent 递归上限约束
    A3 ExpertSpec.to_subagent 挂载预算中间件
  B 声明式子 Agent（KnowledgeAgent / MediaAgent / MapAgent / CodeAgent / BrowserAgent 的装配方式）
    B1 软停止  B2 硬停止  B3 计数器隔离  B4 生产递归上限下失控
  C FileAgent（create_agent 预编译 + HITL）
    C1 软停止  C2 硬停止  C3 超预算写入不弹授权  C4 预算内写入仍要授权  C5 生产嵌套结构下失控
  D DesktopAgent（手写 StateGraph）
    D1 预算截断  D2 正常路径不变  D3 默认预算 + 生产递归上限

这些数字（模型调用次数、工具执行次数）都是按库源码推导出来的预期值；对不上时要查清原因，不要改断言迁就。
"""
import asyncio
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import List

# 保证在项目环境下统一 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from deepagents import (  # noqa: E402
    CompiledSubAgent,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app import config  # noqa: E402
from app.agents import budget  # noqa: E402
from app.agents.experts import desktop_agent, file_agent  # noqa: E402
from app.agents.spec import ExpertSpec  # noqa: E402
from app.runtime import hitl  # noqa: E402

# 与 smoke_file_agent.py 同款：给 ScriptedModel 注册生产同款 harness profile
register_harness_profile(
    "scriptedmodel",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset({"write_file", "edit_file", "delete", "execute", "ls", "glob", "grep"}),
    ),
)

LIMIT_ERR = "Tool call limit exceeded"
MODEL_LIMIT_ERR = "Model call limits exceeded"
FA_REPORT = "【内部专家汇报 - 来源：FileAgent】"
DA_REPORT = "【内部专家汇报 - 来源：DesktopAgent】"


# ══════════════════════════════════════════════════════════════════════════════
# 公共工具
# ══════════════════════════════════════════════════════════════════════════════

class ScriptedModel(BaseChatModel):
    """按脚本依次返回 AIMessage（超出脚本长度后重复最后一条）；idx = 模型被调用的次数。"""
    script: List[AIMessage]
    idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kw) -> ChatResult:
        msg = self.script[min(self.idx, len(self.script) - 1)]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, cid) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def probe_calls(n, prefix="p"):
    """n 条各带 1 个工具调用的 AIMessage。id 必须全局唯一（重复 id 会被 add_messages 当成“更新”而吞掉）。"""
    return [call("probe", {"q": f"{prefix}{i}"}, f"{prefix}{i}") for i in range(n)]


def ls_calls(n, prefix="l"):
    return [call("ls", {"path": "/workspace"}, f"{prefix}{i}") for i in range(n)]


def click_calls(n, prefix="c"):
    return [call("click_grid_center", {"grid_id": f"A{i}"}, f"{prefix}{i}") for i in range(n)]


def make_probe():
    hits = []

    @tool
    def probe(q: str) -> str:
        """测试探针：记录被真正执行的调用。"""
        hits.append(q)
        return f"ok:{q}"

    return probe, hits


def declarative_sub(model, probe, limit, name="Probe"):
    return {
        "name": name,
        "description": "测试专家",
        "system_prompt": "你是 Probe，用于测试。",
        "tools": [probe],
        "model": model,
        "middleware": budget.build_budget_middleware(limit),
    }


def task_call(sub_name, cid):
    return call("task", {"subagent_type": sub_name, "description": "do it"}, cid)


def run_parent(main_script, subagents, thread, middleware=()):
    """与生产一致：用 AGENT_RECURSION_LIMIT、走异步 ainvoke（chat_service 就是这样调用主图的）。返回 (主模型, 结果)。"""
    main_m = ScriptedModel(script=main_script)
    agent = create_deep_agent(model=main_m, subagents=subagents, middleware=list(middleware), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": config.AGENT_RECURSION_LIMIT}
    res = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="go")]}, config=cfg))
    return main_m, res


def task_results(res):
    return [m for m in res["messages"] if isinstance(m, ToolMessage) and getattr(m, "name", "") == "task"]


def tool_msgs(res, call_id):
    return [m for m in res["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == call_id]


def expect_value_error(bad):
    try:
        budget.build_budget_middleware(bad)
    except ValueError:
        return
    raise AssertionError(f"build_budget_middleware({bad!r}) 应抛 ValueError")


# ══════════════════════════════════════════════════════════════════════════════
# Group A: 纯函数与装配
# ══════════════════════════════════════════════════════════════════════════════

def test_a1_build_budget_middleware():
    """A1 返回 [软停止, 硬停止]；run_limit / exit_behavior 正确；非法参数抛 ValueError；每次返回新实例。"""
    mws = budget.build_budget_middleware(3)
    assert [type(m) for m in mws] == [ToolCallLimitMiddleware, ModelCallLimitMiddleware], mws
    soft, hard = mws
    assert soft.run_limit == 3 and soft.exit_behavior == "continue", (soft.run_limit, soft.exit_behavior)
    assert soft.tool_name is None and soft.thread_limit is None
    assert hard.run_limit == 5 and hard.exit_behavior == "end", (hard.run_limit, hard.exit_behavior)
    assert hard.thread_limit is None
    for bad in (0, -1, None, "3", 2.5, True):
        expect_value_error(bad)
    again = budget.build_budget_middleware(3)
    assert again[0] is not soft and again[1] is not hard, "每次调用必须返回新实例"


def test_a2_config_table():
    """A2 预算表键恰好是 7 个专家；值是 >=1 的 int；tool_limit_for 读表/回落默认；DesktopAgent 不超递归上限。"""
    limits = config.SUBAGENT_TOOL_LIMITS
    expected = {"KnowledgeAgent", "MediaAgent", "MapAgent", "CodeAgent", "BrowserAgent", "FileAgent", "DesktopAgent"}
    assert set(limits) == expected, set(limits) ^ expected
    for name, v in [*limits.items(), ("DEFAULT", config.SUBAGENT_TOOL_LIMIT_DEFAULT)]:
        assert isinstance(v, int) and not isinstance(v, bool) and v >= 1, (name, v)
    assert budget.tool_limit_for("KnowledgeAgent") == limits["KnowledgeAgent"]
    assert budget.tool_limit_for("NoSuchAgent") == config.SUBAGENT_TOOL_LIMIT_DEFAULT
    assert 2 * limits["DesktopAgent"] + 4 <= config.AGENT_RECURSION_LIMIT, "DesktopAgent 预算会撞上继承来的递归上限"


def test_a3_expert_spec_wires_budget():
    """A3 ExpertSpec.to_subagent 产出的字典带 "middleware"：登记过的名字用表值，未登记的用默认值。"""
    fake_models = SimpleNamespace(worker_llm=object())
    spec = ExpertSpec(name="MapAgent", description="d", system_prompt="p", tools_factory=lambda: [], model_key="worker_llm")
    mws = spec.to_subagent(fake_models)["middleware"]
    assert len(mws) == 2, mws
    assert mws[0].run_limit == config.SUBAGENT_TOOL_LIMITS["MapAgent"]
    assert mws[1].run_limit == config.SUBAGENT_TOOL_LIMITS["MapAgent"] + 2
    other = ExpertSpec(name="NoSuchAgent", description="d", system_prompt="p", tools_factory=lambda: [], model_key="worker_llm")
    mws2 = other.to_subagent(fake_models)["middleware"]
    assert mws2[0].run_limit == config.SUBAGENT_TOOL_LIMIT_DEFAULT
    assert mws2[1].run_limit == config.SUBAGENT_TOOL_LIMIT_DEFAULT + 2


# ══════════════════════════════════════════════════════════════════════════════
# Group B: 声明式子 Agent
# ══════════════════════════════════════════════════════════════════════════════

def test_b1_declarative_soft_stop():
    """B1 N=3：第 4 次工具调用被拦截（没有真正执行），模型仍能写出汇报；子模型共调用 N+2=5 次。"""
    probe, hits = make_probe()
    sub_m = ScriptedModel(script=probe_calls(4) + [AIMessage(content="【内部专家汇报 - 来源：Probe】部分结果")])
    main_m, res = run_parent(
        [task_call("Probe", "t1"), AIMessage(content="主任务完成")],
        [declarative_sub(sub_m, probe, 3)], "b1",
    )
    assert sub_m.idx == 5, f"子模型调用数应为 5，实际 {sub_m.idx}"
    assert hits == ["p0", "p1", "p2"], hits
    tr = task_results(res)
    assert len(tr) == 1 and "【内部专家汇报 - 来源：Probe】" in str(tr[0].content), tr
    assert main_m.idx == 2, main_m.idx


def test_b2_declarative_hard_stop():
    """B2 N=3：模型永不收手 → 第 5 次模型调用后被强制结束，task 返回 Model call limits exceeded。"""
    probe, hits = make_probe()
    sub_m = ScriptedModel(script=probe_calls(13))
    _, res = run_parent(
        [task_call("Probe", "t1"), AIMessage(content="主任务完成")],
        [declarative_sub(sub_m, probe, 3)], "b2",
    )
    assert sub_m.idx == 5, f"子模型调用数应为 5，实际 {sub_m.idx}"
    assert hits == ["p0", "p1", "p2"], hits
    tr = task_results(res)
    assert len(tr) == 1 and MODEL_LIMIT_ERR in str(tr[0].content), tr


def test_b3_counter_isolation():
    """B3 N=3，主管另挂 task 工具上限 2：两次委派各自拿到完整的 3 次预算；第三次委派被主管拦下。"""
    probe, hits = make_probe()
    sub_m = ScriptedModel(
        script=probe_calls(3, "a") + [AIMessage(content="r1 done")] + probe_calls(3, "b") + [AIMessage(content="r2 done")]
    )
    parent_limit = ToolCallLimitMiddleware(tool_name="task", run_limit=2, exit_behavior="continue")
    _, res = run_parent(
        [task_call("Probe", "t1"), task_call("Probe", "t2"), task_call("Probe", "t3"), AIMessage(content="完成")],
        [declarative_sub(sub_m, probe, 3)], "b3", middleware=[parent_limit],
    )
    assert len(hits) == 6, hits
    assert sub_m.idx == 8, f"第三次委派应被主管拦下，子模型调用数应为 8，实际 {sub_m.idx}"
    tr = task_results(res)
    assert len(tr) == 3, [str(m.content)[:60] for m in tr]
    assert "r1 done" in str(tr[0].content) and "r2 done" in str(tr[1].content), tr
    assert LIMIT_ERR in str(tr[2].content), tr[2].content


def test_b4_declarative_runaway_under_production_recursion_limit():
    """B4 用生产 recursion_limit=60 调用主图，子专家失控也只会被预算止住，不抛 GraphRecursionError（实测前提 P5）。"""
    n = max(config.SUBAGENT_TOOL_LIMITS.values())
    probe, hits = make_probe()
    sub_m = ScriptedModel(script=probe_calls(n + 15))
    _, res = run_parent(
        [task_call("Probe", "t1"), AIMessage(content="主任务完成")],
        [declarative_sub(sub_m, probe, n)], "b4",
    )
    assert sub_m.idx == n + 2, f"子模型调用数应为 {n + 2}，实际 {sub_m.idx}"
    assert len(hits) == n, len(hits)
    tr = task_results(res)
    assert len(tr) == 1 and MODEL_LIMIT_ERR in str(tr[0].content), tr


# ══════════════════════════════════════════════════════════════════════════════
# Group C: FileAgent
# ══════════════════════════════════════════════════════════════════════════════

def test_c1_file_agent_soft_stop():
    """C1 N=2：第 3 次 ls 被拦截；最终汇报由脚本给出；模型共调用 4 次。"""
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "a.txt").write_text("x", encoding="utf-8")
        fam = ScriptedModel(script=ls_calls(3) + [AIMessage(content=f"{FA_REPORT}已汇总")])
        runnable = file_agent.build_file_agent_runnable(fam, "prompt", td, max_tool_calls=2)
        res = runnable.invoke({"messages": [HumanMessage(content="go")]}, config={"recursion_limit": 100})
    assert fam.idx == 4, f"模型调用数应为 4，实际 {fam.idx}"
    for cid in ("l0", "l1"):
        ms = tool_msgs(res, cid)
        assert len(ms) == 1 and ms[0].status != "error" and LIMIT_ERR not in str(ms[0].content), ms
    blocked = tool_msgs(res, "l2")
    assert len(blocked) == 1 and LIMIT_ERR in str(blocked[0].content), blocked
    assert FA_REPORT in str(res["messages"][-1].content), res["messages"][-1]


def test_c2_file_agent_hard_stop():
    """C2 N=2：模型永不收手 → 第 4 次模型调用后被强制结束；成功执行的 ls 恰好 2 条。"""
    with tempfile.TemporaryDirectory() as td:
        fam = ScriptedModel(script=ls_calls(10))
        runnable = file_agent.build_file_agent_runnable(fam, "prompt", td, max_tool_calls=2)
        res = runnable.invoke({"messages": [HumanMessage(content="go")]}, config={"recursion_limit": 100})
    assert fam.idx == 4, f"模型调用数应为 4，实际 {fam.idx}"
    assert MODEL_LIMIT_ERR in str(res["messages"][-1].content), res["messages"][-1]
    ok = [m for m in res["messages"] if isinstance(m, ToolMessage) and m.name == "ls" and m.status != "error"]
    assert len(ok) == 2, [(m.tool_call_id, m.status) for m in res["messages"] if isinstance(m, ToolMessage)]


def test_c3_over_budget_write_does_not_prompt():
    """C3 N=2：第 3 次调用恰好是 write_file → 被预算拦下，不弹授权、不落盘、同一个 tool_call_id 只有 1 条应答。"""
    with tempfile.TemporaryDirectory() as td:
        script = ls_calls(2) + [
            call("write_file", {"file_path": "/workspace/blocked.txt", "content": "x"}, "w1"),
            AIMessage(content=f"{FA_REPORT}已汇总"),
        ]
        fam = ScriptedModel(script=script)
        runnable = file_agent.build_file_agent_runnable(fam, "prompt", td, checkpointer=MemorySaver(), max_tool_calls=2)
        cfg = {"configurable": {"thread_id": "c3"}, "recursion_limit": 100}
        res = runnable.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)
        assert not hitl.has_pending_interrupts(runnable.get_state(cfg)), "超预算的写入不应再弹授权"
        assert not (Path(td) / "blocked.txt").exists(), "超预算的写入不应落盘"
    ms = tool_msgs(res, "w1")
    assert len(ms) == 1 and LIMIT_ERR in str(ms[0].content), ms
    assert fam.idx == 4, f"模型调用数应为 4，实际 {fam.idx}"


def test_c4_in_budget_write_still_requires_approval():
    """C4 对照组：预算内的写入照常暂停等待授权，批准后落盘。"""
    with tempfile.TemporaryDirectory() as td:
        script = [
            call("write_file", {"file_path": "/workspace/ok.txt", "content": "x"}, "w1"),
            AIMessage(content=f"{FA_REPORT}写入成功"),
        ]
        fam = ScriptedModel(script=script)
        runnable = file_agent.build_file_agent_runnable(fam, "prompt", td, checkpointer=MemorySaver(), max_tool_calls=2)
        cfg = {"configurable": {"thread_id": "c4"}, "recursion_limit": 100}
        runnable.invoke({"messages": [HumanMessage(content="go")]}, config=cfg)
        assert hitl.has_pending_interrupts(runnable.get_state(cfg)), "预算内的写入必须弹授权"
        assert not (Path(td) / "ok.txt").exists(), "授权前不应落盘"
        runnable.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=cfg)
        assert (Path(td) / "ok.txt").read_text(encoding="utf-8") == "x"
    assert fam.idx == 2, f"模型调用数应为 2，实际 {fam.idx}"


def test_c5_file_agent_runaway_nested_under_production_recursion_limit():
    """C5 FileAgent 作为 CompiledSubAgent 嵌在主图里、recursion_limit=60：失控也只会被默认预算止住。"""
    limit = config.SUBAGENT_TOOL_LIMITS["FileAgent"]
    with tempfile.TemporaryDirectory() as td:
        fam = ScriptedModel(script=ls_calls(40))
        sub = CompiledSubAgent(
            name="FileAgent", description="本地文件专家",
            runnable=file_agent.build_file_agent_runnable(fam, "prompt", td),
        )
        _, res = run_parent([task_call("FileAgent", "t1"), AIMessage(content="完成")], [sub], "c5")
    assert fam.idx == limit + 2, f"模型调用数应为 {limit + 2}，实际 {fam.idx}"
    tr = task_results(res)
    assert len(tr) == 1 and MODEL_LIMIT_ERR in str(tr[0].content), tr


# ══════════════════════════════════════════════════════════════════════════════
# Group D: DesktopAgent（手写 StateGraph，没有 middleware）
# ══════════════════════════════════════════════════════════════════════════════

def _desktop_tools():
    clicks = []

    @tool
    def take_grid_screenshot() -> str:
        """测试用截图工具：返回普通文本（不要返回 base64）。"""
        return "shot"

    @tool
    def click_grid_center(grid_id: str, click_type: str = "single") -> str:
        """测试用点击工具：记录被真正执行的点击。"""
        clicks.append(grid_id)
        return f"clicked {grid_id}"

    return [take_grid_screenshot, click_grid_center], clicks


def _run_desktop(script, **kwargs):
    tools, clicks = _desktop_tools()
    llm = ScriptedModel(script=script)
    graph = desktop_agent.build_desktop_agent_subgraph(desktop_tools=tools, system_prompt="x", llm=llm, **kwargs)
    res = asyncio.run(graph.ainvoke(
        {"messages": [HumanMessage(content="go")]},
        config={"recursion_limit": config.AGENT_RECURSION_LIMIT},
    ))
    return llm, clicks, res


def test_d1_desktop_budget_cut():
    """D1 N=3：第 4 次想继续点击时不再执行也不再调模型，直接返回确定性汇报。"""
    llm, clicks, res = _run_desktop(click_calls(10), max_tool_calls=3)
    assert llm.idx == 4, f"模型调用数应为 4，实际 {llm.idx}"
    assert clicks == ["A0", "A1", "A2"], clicks
    last = res["messages"][-1]
    assert isinstance(last, AIMessage), last
    text = str(last.content)
    assert text.startswith(DA_REPORT), text
    assert "步骤上限" in text and "3 步" in text and "click_grid_center" in text, text


def test_d2_desktop_normal_path_unchanged():
    """D2 预算没用完时汇报文案与改动前完全一致。"""
    llm, clicks, res = _run_desktop([call("click_grid_center", {"grid_id": "A0"}, "c0"), AIMessage(content="已点击 A0")], max_tool_calls=3)
    assert clicks == ["A0"], clicks
    assert res["messages"][-1].content == f"{DA_REPORT}\n已点击 A0", res["messages"][-1].content
    assert "步骤上限" not in res["messages"][-1].content


def test_d3_desktop_default_budget_under_production_recursion_limit():
    """D3 不传 max_tool_calls、recursion_limit=60：失控也只会被默认预算止住，不抛 GraphRecursionError。"""
    limit = config.SUBAGENT_TOOL_LIMITS["DesktopAgent"]
    llm, clicks, res = _run_desktop(click_calls(40))
    assert llm.idx == limit + 1, f"模型调用数应为 {limit + 1}，实际 {llm.idx}"
    assert len(clicks) == limit, len(clicks)
    assert "步骤上限" in str(res["messages"][-1].content), res["messages"][-1].content


# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("开始运行专家独立停止条件（工具调用预算）离线冒烟测试...")
    print("=" * 70)
    scenarios = [
        ("A1 中间件构造与参数校验", test_a1_build_budget_middleware),
        ("A2 预算表完整性", test_a2_config_table),
        ("A3 ExpertSpec 挂载预算", test_a3_expert_spec_wires_budget),
        ("B1 声明式·软停止", test_b1_declarative_soft_stop),
        ("B2 声明式·硬停止", test_b2_declarative_hard_stop),
        ("B3 计数器隔离", test_b3_counter_isolation),
        ("B4 声明式·生产递归上限下失控", test_b4_declarative_runaway_under_production_recursion_limit),
        ("C1 FileAgent·软停止", test_c1_file_agent_soft_stop),
        ("C2 FileAgent·硬停止", test_c2_file_agent_hard_stop),
        ("C3 FileAgent·超预算写入不弹授权", test_c3_over_budget_write_does_not_prompt),
        ("C4 FileAgent·预算内写入仍要授权", test_c4_in_budget_write_still_requires_approval),
        ("C5 FileAgent·生产嵌套结构下失控", test_c5_file_agent_runaway_nested_under_production_recursion_limit),
        ("D1 DesktopAgent·预算截断", test_d1_desktop_budget_cut),
        ("D2 DesktopAgent·正常路径不变", test_d2_desktop_normal_path_unchanged),
        ("D3 DesktopAgent·默认预算+生产递归上限", test_d3_desktop_default_budget_under_production_recursion_limit),
    ]
    passed = 0
    for name, fn in scenarios:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception:
            print(f"  [FAIL] {name}")
            traceback.print_exc()
    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(scenarios)}, FAIL: {len(scenarios) - passed}/{len(scenarios)}")
    print("=" * 70)
    sys.exit(0 if passed == len(scenarios) else 1)


if __name__ == "__main__":
    main()
