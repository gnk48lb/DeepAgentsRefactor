"""
tests/smoke_orchestrator.py

编排层装配离线冒烟测试:桩掉 MCP 子进程,用真实的 prompt / 工具 / 专家规格装配主 Agent。
  A subagents 顺序(task 工具向 LLM 展示的顺序)
  B 各声明式专家的工具名顺序 / skills 与规格表一致
  C 生产模型(ChatAnthropic / ChatGoogleGenerativeAI)解析到的 HarnessProfile 仍排除 write_file/edit_file/delete/execute,
    并且禁用了内置 general-purpose。这是防止 LLM 通过 /skills/ 路由写真实磁盘的唯一屏障,
    deepagents 升级后一旦失效必须立刻暴露。
  D 主管直接挂载 get_current_time;各声明式专家挂载[软停止, 硬停止]预算中间件且数值与 config 一致。
  A–D 针对 owner 图（构造路径与单用户时代完全等价）。多用户新增（member 图只在编译期就没有宿主机路径）:
  E owner 图仍是 7 个专家 + MAIN_AGENT_PROMPT;member 图恰为 Knowledge / Media / Map,主管工具与预算齐全
  F 启动期断言:黑名单工具（13 个）逐个注入主管 / 声明式子专家、非法专家名、预编译子专家都会 RuntimeError
  G member 构建从不读取 desktop / markitdown 的 MCP 工具,也不构造 CodeAgent / FileAgent / DesktopAgent
  H MEMBER_MAIN_AGENT_PROMPT:含识别短语、能力边界,不含任何宿主机相关词
不需要真实 API Key / 网络:模型客户端用占位 key 构造,构造时不会发请求。
"""
import asyncio
import os
import sys
import traceback

for _k in ("GEMINI_API_KEY", "SILICON_FLOW_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "AMAP_API_KEY", "MAGPIE_API_KEY"):
    os.environ[_k] = "dummy"   # 显式覆盖:本测试绝不应使用真实 key
for _role in ("WORKER_LLM", "ORCHESTRATOR_LLM", "KNOWLEDGE_LLM", "INGESTION_VLM", "BROWSER_VLM", "DESKTOP_VLM"):
    os.environ[f"{_role}_PROVIDER"] = "magpie"
    os.environ.pop(f"{_role}_MODEL", None)

from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.infra import mcp_service, models  # noqa: E402
from app.agents import orchestrator  # noqa: E402
from app.agents.prompts import MAIN_AGENT_PROMPT, MEMBER_MAIN_AGENT_PROMPT  # noqa: E402

EXPECTED_ORDER = ["KnowledgeAgent", "MediaAgent", "MapAgent", "CodeAgent",
                  "BrowserAgent", "FileAgent", "DesktopAgent"]
MEMBER_ORDER = ["KnowledgeAgent", "MediaAgent", "MapAgent"]
# 文档规定的 13 个宿主机相关工具名：member 图里不得出现
FORBIDDEN_TOOLS = {
    "run_host_command", "move_file", "create_directory", "convert_to_markdown",
    "take_grid_screenshot", "click_grid_center", "safe_input_text", "press_hotkey",
    "execute_complex_browser_action", "execute_python_code",
    "execute", "write_file", "edit_file",
}
EXPECTED_TOOLS = {
    "KnowledgeAgent": ["rag", "web_search"],
    "MediaAgent": ["av_graph_rag", "execute_sql"],
    "BrowserAgent": ["read_webpage_content", "execute_complex_browser_action"],
}
BANNED = {"write_file", "edit_file", "delete", "execute", "ls", "glob", "grep"}

_captured = {}              # owner 图的 create_deep_agent 参数（A–D 针对它）
_captured_member = {}       # member 图的 create_deep_agent 参数
_mcp_requests = []          # get_tools_by_server 被问过哪些服务
_agents = None
_real_create = orchestrator.create_deep_agent


def _build():
    async def _fake_init():
        return None

    def _fake_get_tools_by_server(name):
        _mcp_requests.append(name)
        return []

    global _agents
    mcp_service.initialize_mcp = _fake_init                 # 真实环境会拉起 uvx / uv 子进程
    mcp_service.get_tools_by_server = _fake_get_tools_by_server

    def spy(*args, **kwargs):
        # build_main_agents 会调两次 create_deep_agent：按 system_prompt 的身份区分是哪一张图
        prompt = kwargs.get("system_prompt")
        if prompt is MAIN_AGENT_PROMPT:
            _captured.update(kwargs)
        elif prompt is MEMBER_MAIN_AGENT_PROMPT:
            _captured_member.update(kwargs)
        else:
            raise AssertionError("未知的主图 system_prompt")
        return _real_create(*args, **kwargs)

    orchestrator.create_deep_agent = spy
    _agents = asyncio.run(orchestrator.build_main_agents())
    assert hasattr(_agents.owner, "ainvoke") and hasattr(_agents.member, "ainvoke"), "build_main_agents 应返回两张可调用的已编译图"


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
    for name in ("KnowledgeAgent", "MediaAgent", "MapAgent", "BrowserAgent"):
        assert not by_name[name].get("skills"), f"{name} 不应挂 skills"
    for name in ("CodeAgent", "FileAgent", "DesktopAgent"):
        assert "runnable" in by_name[name], f"{name} 应是 CompiledSubAgent"


def test_c_harness_profile():
    try:
        from deepagents.graph import _harness_profile_for_model
    except ImportError as e:   # deepagents 没有公开的 profile 查询接口,只能依赖这个私有函数
        raise AssertionError(f"deepagents 内部 API 已变化({e!r}):请人工复核 excluded_tools 是否仍然生效!")
    for label in ("orchestrator_llm", "worker_llm", "knowledge_llm"):
        llm = getattr(models, label)
        assert llm._get_ls_params()["ls_provider"] == "anthropic", f"{label} 应走 Anthropic 协议"
        prof = _harness_profile_for_model(llm, None)
        missing = BANNED - set(prof.excluded_tools)
        assert not missing, f"{label} 的 HarnessProfile 未排除 {sorted(missing)}:LLM 将重新获得写工具!"
        assert "read_file" not in prof.excluded_tools, f"{label} 的 HarnessProfile 误排除了 read_file!"
        gp = prof.general_purpose_subagent
        assert gp is not None and gp.enabled is False, f"{label} 应禁用内置 general-purpose 子 Agent"


def test_d_time_tool_and_budgets():
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
    from app import config

    # 主管直接挂载 get_current_time + 两个只读会话回溯工具（顺序固定）
    from app.tools import session_tools
    assert [t.name for t in session_tools.TOOLS] == ["get_session_events", "search_session_history"]
    assert [t.name for t in _captured["tools"]] == [
        "get_current_time", "get_session_events", "search_session_history"
    ], _captured.get("tools")

    # 配置表覆盖全部专家；声明式专家各挂 [软停止, 硬停止]
    by_name = {_name(s): s for s in _captured["subagents"]}
    expert_names = set(EXPECTED_ORDER)
    assert set(config.SUBAGENT_TOOL_LIMITS) == expert_names, set(config.SUBAGENT_TOOL_LIMITS) ^ expert_names
    for name in ("KnowledgeAgent", "MediaAgent", "MapAgent", "BrowserAgent"):
        mws = by_name[name]["middleware"]
        assert [type(m) for m in mws] == [ToolCallLimitMiddleware, ModelCallLimitMiddleware], name
        limit = config.SUBAGENT_TOOL_LIMITS[name]
        assert mws[0].run_limit == limit and mws[1].run_limit == limit + 2, name


def _dummy_tool(name):
    @tool(name)
    def _t(x: str = "") -> str:
        """dummy"""
        return "x"
    return _t


def _expect_runtime_error(fn, *args):
    try:
        fn(*args)
    except RuntimeError:
        return
    raise AssertionError(f"期望 RuntimeError: {fn.__name__}")


def test_e_owner_and_member_graphs():
    """E owner 图仍是 7 个专家 + MAIN_AGENT_PROMPT；member 图恰为 Knowledge / Media / Map，主管工具与预算齐全。"""
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
    from app import config

    # owner 图：构造路径与单用户时代等价
    assert _captured["system_prompt"] is MAIN_AGENT_PROMPT
    assert len(_captured["subagents"]) == 7 and _agents.owner_experts == tuple(EXPECTED_ORDER), _agents.owner_experts

    # member 图
    subs = _captured_member["subagents"]
    assert [_name(s) for s in subs] == MEMBER_ORDER and _agents.member_experts == tuple(MEMBER_ORDER), [_name(s) for s in subs]
    assert all("runnable" not in s for s in subs), "member 图不应有预编译子专家（CodeAgent / FileAgent / DesktopAgent 都是）"
    by_name = {_name(s): s for s in subs}
    for name in ("KnowledgeAgent", "MediaAgent"):
        assert [t.name for t in by_name[name]["tools"]] == EXPECTED_TOOLS[name], name
    assert len(by_name["MapAgent"]["tools"]) == 12
    assert [t.name for t in _captured_member["tools"]] == [
        "get_current_time", "get_session_events", "search_session_history"
    ], _captured_member.get("tools")
    for name in MEMBER_ORDER:
        mws = by_name[name]["middleware"]
        assert [type(m) for m in mws] == [ToolCallLimitMiddleware, ModelCallLimitMiddleware], name
        assert mws[0].run_limit == config.SUBAGENT_TOOL_LIMITS[name], name

    # 两张不同的图，共用同一个 checkpointer；主管仍有“不写文件”的第二道保险
    assert _agents.owner is not _agents.member
    assert _captured["checkpointer"] is _captured_member["checkpointer"] and _captured["checkpointer"] is not None
    assert _captured_member["permissions"] == _captured["permissions"]

    # 实际传入的所有工具名都不在黑名单里
    names = {t.name for t in _captured_member["tools"]}
    for s in subs:
        names |= {t.name for t in s["tools"]}
    assert not (names & FORBIDDEN_TOOLS), names & FORBIDDEN_TOOLS


def test_f_startup_assertion():
    """F 启动期断言（失败即关闭）：黑名单工具逐个注入主管 / 每个声明式子专家、非法专家名、预编译子专家都会 RuntimeError。"""
    check = orchestrator._assert_member_graph_safe
    sup = [orchestrator.knowledge_tools.get_current_time, *orchestrator.session_tools.TOOLS]
    check(sup, orchestrator._member_subagents())                      # 基线通过

    assert set(orchestrator._MEMBER_FORBIDDEN_TOOL_NAMES) == FORBIDDEN_TOOLS, "黑名单必须恰为文档规定的 13 个"
    assert orchestrator._MEMBER_EXPERT_NAMES == tuple(MEMBER_ORDER)

    for name in sorted(FORBIDDEN_TOOLS):
        bad = _dummy_tool(name)
        _expect_runtime_error(check, [*sup, bad], orchestrator._member_subagents())
        for i in range(len(MEMBER_ORDER)):
            subs = orchestrator._member_subagents()
            subs[i]["tools"] = [*subs[i]["tools"], bad]
            _expect_runtime_error(check, sup, subs)

    subs = orchestrator._member_subagents()
    subs.append({"name": "BrowserAgent", "description": "x", "system_prompt": "x", "tools": []})
    _expect_runtime_error(check, sup, subs)
    subs = orchestrator._member_subagents()
    subs.append({"name": "KnowledgeAgent", "description": "x", "runnable": object()})   # 名字合法也不行：预编译的查不到内部工具
    _expect_runtime_error(check, sup, subs)

    # 端到端：断言失败时 _build_member_agent 不会走到 create_deep_agent
    reached = []
    real_subs, real_create = orchestrator._member_subagents, orchestrator.create_deep_agent

    def poisoned():
        subs = real_subs()
        subs[0]["tools"] = [*subs[0]["tools"], _dummy_tool("run_host_command")]
        return subs

    orchestrator._member_subagents = poisoned
    orchestrator.create_deep_agent = lambda *a, **k: reached.append(1)
    try:
        _expect_runtime_error(orchestrator._build_member_agent, MemorySaver())
    finally:
        orchestrator._member_subagents, orchestrator.create_deep_agent = real_subs, real_create
    assert not reached, "断言失败后不应继续构建主图"


def test_g_member_build_never_touches_host_paths():
    """G member 构建从不读取 desktop / markitdown 的 MCP 工具，也不构造 CodeAgent / FileAgent / DesktopAgent。"""
    assert {"desktop", "markitdown"} <= set(_mcp_requests), f"owner 构建应当读取这两类 MCP 工具（否则这条测试是空转的）: {_mcp_requests}"
    _mcp_requests.clear()

    def boom(*args, **kwargs):
        raise AssertionError("member 构建不应走到这里")

    patched = [
        (orchestrator.code_agent, "build_code_agent_runnable"),
        (orchestrator.file_agent, "build_file_agent_runnable"),
        (orchestrator.desktop_agent, "build_desktop_agent_subgraph"),
        (orchestrator.markitdown_tool, "build_tools"),
    ]
    originals = [(mod, attr, getattr(mod, attr)) for mod, attr in patched]
    try:
        for mod, attr in patched:
            setattr(mod, attr, boom)
        _agent, experts = orchestrator._build_member_agent(MemorySaver())
    finally:
        for mod, attr, original in originals:
            setattr(mod, attr, original)
    assert experts == tuple(MEMBER_ORDER) and hasattr(_agent, "ainvoke")
    assert _mcp_requests == [], f"member 构建不得读取任何 MCP 工具: {_mcp_requests}"


def test_h_member_prompt():
    """H MEMBER_MAIN_AGENT_PROMPT：含识别短语与能力边界，不含任何宿主机相关词，且不是 MAIN_AGENT_PROMPT 的别名。"""
    p = MEMBER_MAIN_AGENT_PROMPT
    assert p is not MAIN_AGENT_PROMPT and p != MAIN_AGENT_PROMPT and len(p) < len(MAIN_AGENT_PROMPT)
    assert "你是一个多智能体系统的主 Agent" in p, "cli_logger._identify_agent 靠这句话识别主管"
    assert "【能力边界】" in p and "不要尝试委派" in p
    for word in ("FileAgent", "CodeAgent", "DesktopAgent", "BrowserAgent", "workspace", "宿主机", "Docker", "沙箱"):
        assert word not in p, f"member 提示词不应出现 {word!r}"
    for word in ("KnowledgeAgent", "MediaAgent", "MapAgent", "get_current_time",
                 "search_session_history", "get_session_events", "<long_term_memory>", "【严格数据边界】"):
        assert word in p, f"member 提示词应保留 {word!r}"


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
                 ("D 主管时间工具与专家预算装配", test_d_time_tool_and_budgets),
                 ("E owner 图与 member 图", test_e_owner_and_member_graphs),
                 ("F member 启动期断言", test_f_startup_assertion),
                 ("G member 构建不碰宿主机路径", test_g_member_build_never_touches_host_paths),
                 ("H member 主管提示词", test_h_member_prompt)]
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
