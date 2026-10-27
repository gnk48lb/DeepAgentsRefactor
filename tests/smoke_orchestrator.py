"""
tests/smoke_orchestrator.py

编排层装配离线冒烟测试:桩掉 MCP 子进程,用真实的 prompt / 工具 / 专家规格装配主 Agent。
  A subagents 顺序(task 工具向 LLM 展示的顺序)
  B 各声明式专家的工具名顺序 / skills 与规格表一致
  C 生产模型(ChatGoogleGenerativeAI)解析到的 HarnessProfile 仍排除 write_file/edit_file/delete/execute,
    并且禁用了内置 general-purpose。这是防止 LLM 通过 /skills/ 路由写真实磁盘的唯一屏障,
    deepagents 升级后一旦失效必须立刻暴露。
  D 主管直接挂载 get_current_time;各声明式专家挂载[软停止, 硬停止]预算中间件且数值与 config 一致。
不需要真实 API Key / 网络:Gemini 客户端用占位 key 构造,构造时不会发请求。
"""
import asyncio
import os
import sys
import traceback

for _k in ("GEMINI_API_KEY", "SILICON_FLOW_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "AMAP_API_KEY"):
    os.environ[_k] = "dummy"   # 显式覆盖:本测试绝不应使用真实 key

from app.infra import mcp_service, models  # noqa: E402
from app.agents import orchestrator  # noqa: E402

EXPECTED_ORDER = ["KnowledgeAgent", "MediaAgent", "MapAgent", "CodeAgent",
                  "BrowserAgent", "FileAgent", "DesktopAgent"]
EXPECTED_TOOLS = {
    "KnowledgeAgent": ["rag", "web_search"],
    "MediaAgent": ["av_graph_rag", "execute_sql"],
    "CodeAgent": ["execute_python_code"],
    "BrowserAgent": ["read_webpage_content", "get_bilibili_profile", "execute_complex_browser_action"],
}
BANNED = {"write_file", "edit_file", "delete", "execute", "ls", "glob", "grep"}

_captured = {}


def _build():
    async def _fake_init():
        return None

    mcp_service.initialize_mcp = _fake_init                 # 真实环境会拉起 uvx / uv 子进程
    mcp_service.get_tools_by_server = lambda name: []
    real = orchestrator.create_deep_agent

    def spy(*args, **kwargs):
        _captured.update(kwargs)
        return real(*args, **kwargs)

    orchestrator.create_deep_agent = spy
    agent = asyncio.run(orchestrator.build_main_agent())
    assert hasattr(agent, "ainvoke"), "build_main_agent 应返回可调用的已编译图"


def _name(sub):
    return sub["name"]


def test_a_order():
    assert [_name(s) for s in _captured["subagents"]] == EXPECTED_ORDER, [_name(s) for s in _captured["subagents"]]


def test_b_specs():
    by_name = {_name(s): s for s in _captured["subagents"]}
    for name, tools in EXPECTED_TOOLS.items():
        got = [t.name for t in by_name[name]["tools"]]
        assert got == tools, f"{name} 工具不一致: {got} != {tools}"
    assert len(by_name["MapAgent"]["tools"]) == 12, "MapAgent 应挂载 12 个高德工具"
    assert by_name["CodeAgent"].get("skills") == ["/skills/"], by_name["CodeAgent"].get("skills")
    for name in ("KnowledgeAgent", "MediaAgent", "MapAgent", "BrowserAgent"):
        assert not by_name[name].get("skills"), f"{name} 不应挂 skills"
    for name in ("FileAgent", "DesktopAgent"):
        assert "runnable" in by_name[name], f"{name} 应是 CompiledSubAgent"


def test_c_harness_profile():
    try:
        from deepagents.graph import _harness_profile_for_model
    except ImportError as e:   # deepagents 没有公开的 profile 查询接口,只能依赖这个私有函数
        raise AssertionError(f"deepagents 内部 API 已变化({e!r}):请人工复核 excluded_tools 是否仍然生效!")
    for label in ("orchestrator_llm", "worker_llm"):
        prof = _harness_profile_for_model(getattr(models, label), None)
        missing = BANNED - set(prof.excluded_tools)
        assert not missing, f"{label} 的 HarnessProfile 未排除 {sorted(missing)}:LLM 将重新获得写工具!"
        assert "read_file" not in prof.excluded_tools, f"{label} 的 HarnessProfile 误排除了 read_file!"
        gp = prof.general_purpose_subagent
        assert gp is not None and gp.enabled is False, f"{label} 应禁用内置 general-purpose 子 Agent"


def test_d_time_tool_and_budgets():
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
    from app import config

    # 主管直接挂载 get_current_time
    assert [t.name for t in _captured["tools"]] == ["get_current_time"], _captured.get("tools")

    # 配置表覆盖全部专家；声明式专家各挂 [软停止, 硬停止]
    by_name = {_name(s): s for s in _captured["subagents"]}
    expert_names = set(EXPECTED_ORDER)
    assert set(config.SUBAGENT_TOOL_LIMITS) == expert_names, set(config.SUBAGENT_TOOL_LIMITS) ^ expert_names
    for name in ("KnowledgeAgent", "MediaAgent", "MapAgent", "CodeAgent", "BrowserAgent"):
        mws = by_name[name]["middleware"]
        assert [type(m) for m in mws] == [ToolCallLimitMiddleware, ModelCallLimitMiddleware], name
        limit = config.SUBAGENT_TOOL_LIMITS[name]
        assert mws[0].run_limit == limit and mws[1].run_limit == limit + 2, name


def main():
    print("=" * 70)
    print("开始运行编排层装配离线冒烟测试...")
    print("=" * 70)
    try:
        _build()
    except Exception:
        print("  [FAIL] 装配主 Agent 失败")
        traceback.print_exc()
        sys.exit(1)
    scenarios = [("A subagents 顺序", test_a_order), ("B 专家规格与工具", test_b_specs),
                 ("C HarnessProfile 排除写工具", test_c_harness_profile),
                 ("D 主管时间工具与专家预算装配", test_d_time_tool_and_budgets)]
    passed = failed = 0
    for name, fn in scenarios:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(scenarios)}, FAIL: {failed}/{len(scenarios)}")
    print("=" * 70)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
