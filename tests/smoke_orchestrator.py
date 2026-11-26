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
  J / K 定时任务开关打开时：owner / member 的主管追加调度说明与三个 schedule 工具，原常量不变，member 的能力边界不被破坏
  L 无人值守的第三张图：三个声明式专家、监督者工具只有 get_current_time、不带 CodeAgent、启动期断言逐项拒绝违规
不需要真实 API Key / 网络:模型客户端用占位 key 构造,构造时不会发请求。
"""
import asyncio
import contextlib
import importlib.metadata
import importlib.util
import os
import sys
import traceback

for _k in ("GEMINI_API_KEY", "SILICON_FLOW_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "AMAP_API_KEY", "MAGPIE_API_KEY"):
    os.environ[_k] = "dummy"   # 显式覆盖:本测试绝不应使用真实 key
# 本机 .env 里可能有真实的 Modal 令牌：load_dotenv 不覆盖已有的环境变量，所以这一行能压过它。
# 否则 member 图会多出 CodeAgent，A–H 全红。启用 CodeAgent 的场景（I / F2 / G2）自己用 monkeypatch + FakeModal。
os.environ["MODAL_DISABLED"] = "true"
# 定时任务开关同理：本机 .env 里可能是 true，那样主管会多三个 schedule 工具、提示词也不是原常量，A–I 全红。
# 开关打开时的断言放在 J / K 里（自己用上下文管理器临时打开）。
os.environ["SCHEDULER_ENABLED"] = "false"
for _role in ("WORKER_LLM", "ORCHESTRATOR_LLM", "KNOWLEDGE_LLM", "INGESTION_VLM", "BROWSER_VLM", "DESKTOP_VLM"):
    os.environ[f"{_role}_PROVIDER"] = "magpie"
    os.environ.pop(f"{_role}_MODEL", None)

from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app import config  # noqa: E402
from app.infra import mcp_service, models, modal_sandbox  # noqa: E402
from app.infra import sandbox as docker_sandbox  # noqa: E402
from app.agents import orchestrator  # noqa: E402
from app.agents.experts import code_agent  # noqa: E402
from app.agents.skills import SKILLS_DIR, SKILLS_MEMBER_DIR, ReadOnlyFilesystemBackend  # noqa: E402
from _fake_modal import FakeModal  # noqa: E402
from app.agents.prompts import (  # noqa: E402
    MAIN_AGENT_PROMPT,
    MAIN_AGENT_PROMPT_SCHED,
    MEMBER_MAIN_AGENT_PROMPT,
    MEMBER_MAIN_AGENT_PROMPT_CODE,
    MEMBER_MAIN_AGENT_PROMPT_CODE_SCHED,
    MEMBER_MAIN_AGENT_PROMPT_SCHED,
    SCHEDULED_MAIN_AGENT_PROMPT,
)

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
_captured_sched = {}        # 开关打开时各张图的 create_deep_agent 参数（键：owner / member / member_code / scheduled）
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
        elif prompt is MEMBER_MAIN_AGENT_PROMPT or prompt is MEMBER_MAIN_AGENT_PROMPT_CODE:
            _captured_member.update(kwargs)
        elif prompt is MAIN_AGENT_PROMPT_SCHED:
            _captured_sched["owner"] = dict(kwargs)
        elif prompt is MEMBER_MAIN_AGENT_PROMPT_SCHED:
            _captured_sched["member"] = dict(kwargs)
        elif prompt is MEMBER_MAIN_AGENT_PROMPT_CODE_SCHED:
            _captured_sched["member_code"] = dict(kwargs)
        elif prompt is SCHEDULED_MAIN_AGENT_PROMPT:
            _captured_sched["scheduled"] = dict(kwargs)
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

    def poisoned(code=None):
        subs = real_subs(code)
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
        (orchestrator.code_agent, "build_member_code_agent_runnable"),     # 禁用状态下它也不应被调用
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


# ══════════════════════════════════════════════════════════════════════════════
# 阶段 2：member 图带云端 CodeAgent（Modal 用 FakeModal 注入；A–H 针对 MODAL_DISABLED=true 的状态）
# ══════════════════════════════════════════════════════════════════════════════

# 阶段 1 的基线哈希（M0 在改动前算出）：MAIN_AGENT_PROMPT 一个字节都不能变；
# 阶段 1 的 MEMBER_MAIN_AGENT_PROMPT 把占位句换回台北例句之后必须与它逐字相同（证明只改了那一句）
_MAIN_PROMPT_SHA256 = "f9882a12a8c96ea5e98bef04a9cb678ac26664d3f7150aef4c2f013930f42816"
_STAGE1_MEMBER_PROMPT_SHA256 = "153f528c668e82c74f5c4fbdde905b3ba571a7f9ebf60b73fde5b0fde514076a"
_NEUTRAL_ADDRESS = "用户当前位置在（此处写用户给出的具体地址），请搜索附近的菜市场"
_TAIPEI_ADDRESS = "用户当前位置在台北市信义区市府路1号，请搜索附近的菜市场"
_CODE_TOOL_NAMES = {"execute_python_code", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}
_CODE_FORBIDDEN_WORDS = ("FileAgent", "DesktopAgent", "BrowserAgent", "workspace", "宿主机", "Docker")


@contextlib.contextmanager
def _code_enabled(**cfg):
    """让 member_code_status() 为真：不禁用、有令牌、modal 已装（没装 / 版本不对就补丁），Modal 用 FakeModal 注入。"""
    saved = {k: getattr(config, k) for k in ("MODAL_DISABLED", "MODAL_TOKENS_PRESENT", *cfg)}
    config.MODAL_DISABLED, config.MODAL_TOKENS_PRESENT = False, True
    for k, v in cfg.items():
        setattr(config, k, v)
    real_spec, real_version = importlib.util.find_spec, importlib.metadata.version
    if not modal_sandbox.member_code_status()[0]:
        importlib.util.find_spec = lambda name, *a, **k: object() if name == "modal" else real_spec(name, *a, **k)
        importlib.metadata.version = lambda name: "1.6.4" if name == "modal" else real_version(name)
    fake = FakeModal()
    fake.publish_default_image()
    modal_sandbox.set_modal_module(fake)
    try:
        assert modal_sandbox.member_code_status() == (True, "")
        yield fake
    finally:
        modal_sandbox.set_modal_module(None)
        importlib.util.find_spec, importlib.metadata.version = real_spec, real_version
        for k, v in saved.items():
            setattr(config, k, v)


def _build_member_capturing():
    """编译 member 图，返回 (图, 专家名元组, create_deep_agent 收到的参数, member CodeAgent 构建物或 None)。"""
    captured, built = {}, []
    real_create, real_build = orchestrator.create_deep_agent, code_agent.build_member_code_agent_runnable

    def spy_create(*args, **kwargs):
        captured.update(kwargs)
        return _real_create(*args, **kwargs)

    def spy_build(*args, **kwargs):
        built.append(real_build(*args, **kwargs))
        return built[-1]

    orchestrator.create_deep_agent, code_agent.build_member_code_agent_runnable = spy_create, spy_build
    try:
        agent, experts = orchestrator._build_member_agent(MemorySaver())
    finally:
        orchestrator.create_deep_agent, code_agent.build_member_code_agent_runnable = real_create, real_build
    return agent, experts, captured, (built[0] if built else None)


def test_h2_member_prompt_with_code():
    """H2 MEMBER_MAIN_AGENT_PROMPT_CODE：含识别短语与各段落，允许 CodeAgent / 沙箱，不含其余宿主机相关词；两个变体都没有真实地址。"""
    p, base = MEMBER_MAIN_AGENT_PROMPT_CODE, MEMBER_MAIN_AGENT_PROMPT
    assert p is not base and p != base and len(p) > len(base)
    assert "你是一个多智能体系统的主 Agent" in p
    for word in ("【能力边界】", "不要尝试委派", "CodeAgent", "沙箱", "【图文分离】", "【严格数据边界】", "<long_term_memory>",
                 "KnowledgeAgent", "MediaAgent", "MapAgent", "get_current_time", "search_session_history", "get_session_events"):
        assert word in p, f"member(CodeAgent) 提示词应含 {word!r}"
    for word in _CODE_FORBIDDEN_WORDS:
        assert word not in p, f"member(CodeAgent) 提示词不应出现 {word!r}"
    assert p.index("【过程不可见】") < p.index("【图文分离】") < p.index("【长期记忆】"), "图文分离在过程不可见之后、长期记忆之前"
    assert "MapAgent、CodeAgent 等" in p and "不要向用户提及沙箱内部的文件路径" in p
    for variant in (p, base):
        assert "台北市信义区市府路1号" not in variant and _NEUTRAL_ADDRESS in variant
    assert "台北市信义区市府路1号" in MAIN_AGENT_PROMPT, "owner 的提示词继续保留它"
    for word in ("CodeAgent", "沙箱"):
        assert word not in base, "禁用 CodeAgent 时的变体与阶段 1 一样不含这些词"


def test_h3_prompt_baselines_are_pinned():
    """H3 MAIN_AGENT_PROMPT 的 sha256 不变；member 变体把占位句换回台北例句之后与阶段 1 逐字相同。"""
    import hashlib
    sha = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert sha(MAIN_AGENT_PROMPT) == _MAIN_PROMPT_SHA256, "MAIN_AGENT_PROMPT 一个字节都不能变"
    assert sha(MEMBER_MAIN_AGENT_PROMPT.replace(_NEUTRAL_ADDRESS, _TAIPEI_ADDRESS)) == _STAGE1_MEMBER_PROMPT_SHA256, \
        "禁用 CodeAgent 的 member 提示词应与阶段 1 逐字相同（只有委派例句里的地址换成了中性占位）"
    assert orchestrator.MEMBER_MAIN_AGENT_PROMPT is MEMBER_MAIN_AGENT_PROMPT


def test_i_member_graph_with_code():
    """I 启用 CodeAgent 的 member 图；任何一种不可用都回到 3 个专家与阶段 1 的提示词，且不调用 CodeAgent 构建函数。"""
    with _code_enabled():
        agent, experts, cap, build = _build_member_capturing()
        assert hasattr(agent, "ainvoke") and build is not None
        assert experts == (*MEMBER_ORDER, "CodeAgent"), experts
        assert cap["system_prompt"] is MEMBER_MAIN_AGENT_PROMPT_CODE
        assert [t.name for t in cap["tools"]] == ["get_current_time", "get_session_events", "search_session_history"]
        subs = cap["subagents"]
        assert [_name(s) for s in subs] == [*MEMBER_ORDER, "CodeAgent"]
        pre = [s for s in subs if "runnable" in s]
        assert len(pre) == 1 and pre[0]["name"] == "CodeAgent" and pre[0]["runnable"] is build.runnable
        assert pre[0]["description"] == code_agent.MEMBER_DESCRIPTION
        assert build.tool_names == _CODE_TOOL_NAMES and "delete" not in build.tool_names
        skills_route = cap["backend"].routes["/skills/"]
        assert isinstance(skills_route, ReadOnlyFilesystemBackend) and os.path.realpath(str(skills_route.cwd)) == os.path.realpath(SKILLS_MEMBER_DIR)
        # 同一次测试里 owner 图与禁用状态下完全一样
        assert _captured["system_prompt"] is MAIN_AGENT_PROMPT and [_name(s) for s in _captured["subagents"]] == EXPECTED_ORDER

    boom_calls = []

    def boom(*args, **kwargs):
        boom_calls.append(1)
        raise AssertionError("不可用时不应调用 CodeAgent 构建函数")

    real_build = code_agent.build_member_code_agent_runnable
    code_agent.build_member_code_agent_runnable = boom
    try:
        real_spec, real_version = importlib.util.find_spec, importlib.metadata.version
        variants = (
            ("MODAL_DISABLED=true", dict(MODAL_DISABLED=True), None, None),
            ("令牌缺失", dict(MODAL_TOKENS_PRESENT=False), None, None),
            ("modal 没装", {}, lambda name, *a, **k: None if name == "modal" else real_spec(name, *a, **k), None),
            ("modal 版本过低", {}, None, lambda name: "1.5.0" if name == "modal" else real_version(name)),
        )
        for label, cfg, spec_patch, version_patch in variants:
            with _code_enabled():
                for k, v in cfg.items():
                    setattr(config, k, v)
                if spec_patch:
                    importlib.util.find_spec = spec_patch
                if version_patch:
                    importlib.metadata.version = version_patch
                assert modal_sandbox.member_code_status()[0] is False, label
                _agent, experts, cap, build = _build_member_capturing_plain()
                assert experts == tuple(MEMBER_ORDER) and build is None, (label, experts)
                assert cap["system_prompt"] is MEMBER_MAIN_AGENT_PROMPT and all("runnable" not in s for s in cap["subagents"]), label
        assert boom_calls == []
    finally:
        code_agent.build_member_code_agent_runnable = real_build


def _build_member_capturing_plain():
    """同 _build_member_capturing，但不包装 CodeAgent 构建函数（调用方已经换成了会炸的替身）。"""
    captured = {}
    real_create = orchestrator.create_deep_agent

    def spy_create(*args, **kwargs):
        captured.update(kwargs)
        return _real_create(*args, **kwargs)

    orchestrator.create_deep_agent = spy_create
    try:
        agent, experts = orchestrator._build_member_agent(MemorySaver())
    finally:
        orchestrator.create_deep_agent = real_create
    return agent, experts, captured, None


def test_f2_strict_exception_for_vetted_code():
    """F2 预编译子专家的严格例外：基线通过；逐个破坏构建物 / 子专家都 RuntimeError；13 个黑名单工具即使带着 vetted_code 也照样拒绝。"""
    import dataclasses

    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
    check = orchestrator._assert_member_graph_safe
    sup = [orchestrator.knowledge_tools.get_current_time, *orchestrator.session_tools.TOOLS]

    with _code_enabled():
        build = code_agent.build_member_code_agent_runnable(llm=models.worker_llm, system_prompt=code_agent.MEMBER_PROMPT)
        subs = orchestrator._member_subagents(build)
        check(sup, subs, vetted_code=build)                                       # 基线通过
        check(sup, orchestrator._member_subagents(), vetted_code=build)           # 带 vetted_code 但图里没有 CodeAgent：也可以
        # 没有 vetted_code 时，预编译子专家一律拒绝（报错文案不变）
        try:
            check(sup, subs)
            raise AssertionError("没有 vetted_code 时应拒绝预编译子专家")
        except RuntimeError as e:
            assert "member 图不允许预编译的子专家（无法检查其内部工具）" in str(e) or "member 图含有不允许的专家" in str(e), str(e)

        other = code_agent.build_member_code_agent_runnable(llm=models.worker_llm, system_prompt=code_agent.MEMBER_PROMPT)
        broken = {}
        # a runnable 身份对不上（相等不行，必须是同一个对象）
        subs_a = orchestrator._member_subagents(other)
        broken["runnable 身份对不上"] = (subs_a, build)
        # b 名字不是 CodeAgent
        subs_b = orchestrator._member_subagents(build)
        subs_b[-1] = {**subs_b[-1], "name": "FileAgent"}
        broken["名字不是 CodeAgent"] = (subs_b, build)
        # c 整张图里多一个预编译子专家
        subs_c = orchestrator._member_subagents(build)
        subs_c.append({"name": "CodeAgent", "description": "dup", "runnable": build.runnable})
        broken["多一个预编译子专家"] = (subs_c, build)
        # d 工具集多出 delete / 少一个
        broken["tool_names 多出 delete"] = (orchestrator._member_subagents(build), dataclasses.replace(build, tool_names=build.tool_names | {"delete"}))
        broken["tool_names 少一个"] = (orchestrator._member_subagents(build), dataclasses.replace(build, tool_names=build.tool_names - {"grep"}))
        # e backend.default 换成本机 Docker 沙箱
        routes = dict(build.backend.routes)
        docker_backend = CompositeBackend(default=docker_sandbox.DockerSandbox(), routes=routes, artifacts_root="/_artifacts")
        broken["默认后端是 Docker"] = (orchestrator._member_subagents(build), dataclasses.replace(build, backend=docker_backend))
        # f 路由多出 /workspace/
        extra_route = CompositeBackend(default=build.backend.default, routes={**routes, "/workspace/": StateBackend()}, artifacts_root="/_artifacts")
        broken["路由多出 /workspace/"] = (orchestrator._member_subagents(build), dataclasses.replace(build, backend=extra_route))
        # g /skills/ 换成可写的 FilesystemBackend / 指向 owner 的 SKILLS_DIR
        writable = CompositeBackend(default=build.backend.default, artifacts_root="/_artifacts",
                                    routes={**routes, "/skills/": FilesystemBackend(root_dir=SKILLS_MEMBER_DIR, virtual_mode=True)})
        broken["/skills/ 可写"] = (orchestrator._member_subagents(build), dataclasses.replace(build, backend=writable))
        owner_skills = CompositeBackend(default=build.backend.default, artifacts_root="/_artifacts",
                                        routes={**routes, "/skills/": ReadOnlyFilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)})
        broken["/skills/ 指向 owner 的 skills/"] = (orchestrator._member_subagents(build), dataclasses.replace(build, backend=owner_skills))
        # h /_artifacts/ 不是 StateBackend
        fs_artifacts = CompositeBackend(default=build.backend.default, artifacts_root="/_artifacts",
                                        routes={**routes, "/_artifacts/": FilesystemBackend(root_dir=SKILLS_MEMBER_DIR, virtual_mode=True)})
        broken["/_artifacts/ 不是 state"] = (orchestrator._member_subagents(build), dataclasses.replace(build, backend=fs_artifacts))
        # i 声明式的 CodeAgent（没有 runnable）
        subs_i = orchestrator._member_subagents(build)[:-1]
        subs_i.append({"name": "CodeAgent", "description": "x", "system_prompt": "x", "tools": []})
        broken["声明式的 CodeAgent"] = (subs_i, build)
        for label, (bad_subs, bad_build) in broken.items():
            _expect_runtime_error(lambda: check(sup, bad_subs, vetted_code=bad_build))
            assert label

        # 13 个黑名单工具注入主管 / 声明式子专家：即使带着 vetted_code 也照样 RuntimeError
        for name in sorted(FORBIDDEN_TOOLS):
            bad = _dummy_tool(name)
            _expect_runtime_error(lambda: check([*sup, bad], orchestrator._member_subagents(build), vetted_code=build))
            for i in range(len(MEMBER_ORDER)):
                subs_x = orchestrator._member_subagents(build)
                subs_x[i]["tools"] = [*subs_x[i]["tools"], bad]
                _expect_runtime_error(lambda: check(sup, subs_x, vetted_code=build))


def test_g2_code_enabled_build_never_touches_docker_or_host():
    """G2 启用 CodeAgent 时，member 构建也不得调用 owner 的 build_code_agent_runnable、sandbox.get_sandbox / set_docker_client、
    markitdown_tool.build_tools，也不读取任何 MCP 工具；member CodeAgent 的构建函数里没有这些标识符。"""
    import ast
    import inspect

    def boom(*args, **kwargs):
        raise AssertionError("member 构建不应走到这里")

    patched = [
        (orchestrator.code_agent, "build_code_agent_runnable"),
        (docker_sandbox, "get_sandbox"),
        (docker_sandbox, "set_docker_client"),
        (docker_sandbox, "reset_sandbox"),
        (orchestrator.markitdown_tool, "build_tools"),
        (orchestrator.file_agent, "build_file_agent_runnable"),
        (orchestrator.desktop_agent, "build_desktop_agent_subgraph"),
    ]
    originals = [(mod, attr, getattr(mod, attr)) for mod, attr in patched]
    with _code_enabled():
        _mcp_requests.clear()
        try:
            for mod, attr in patched:
                setattr(mod, attr, boom)
            _agent, experts, _cap, build = _build_member_capturing()
        finally:
            for mod, attr, original in originals:
                setattr(mod, attr, original)
        assert experts == (*MEMBER_ORDER, "CodeAgent") and build is not None
        assert _mcp_requests == [], f"member 构建不得读取任何 MCP 工具: {_mcp_requests}"

    tree = ast.parse(inspect.getsource(code_agent.build_member_code_agent_runnable).lstrip())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for name in ("build_code_agent_runnable", "get_sandbox", "DockerSandbox", "set_docker_client", "reset_sandbox",
                 "interrupt_sandbox", "shutdown_sandbox", "host_shell", "run_host_command", "SKILLS_DIR"):
        assert name not in used, f"member 的 CodeAgent 构建函数不得引用 {name}"


# ══════════════════════════════════════════════════════════════════════════════
# 定时任务：SCHEDULER_ENABLED 打开时 owner / member 的主管追加调度说明与三个 schedule 工具（A–I 针对关闭状态）
# ══════════════════════════════════════════════════════════════════════════════

_SCHEDULE_TOOL_NAMES = ["create_scheduled_task", "list_scheduled_tasks", "cancel_scheduled_task"]
_SUPERVISOR_BASE_TOOLS = ["get_current_time", "get_session_events", "search_session_history"]
# 追加的调度说明里不得出现的词：member 的能力边界不能被它破坏（能力描述词也一并禁用）
_SCHED_BANNED_WORDS = ("FileAgent", "CodeAgent", "DesktopAgent", "BrowserAgent", "SQLAgent", "workspace", "宿主机", "Docker", "沙箱",
                       "文件", "电脑", "代码")


@contextlib.contextmanager
def _scheduler_enabled():
    saved = config.SCHEDULER_ENABLED
    config.SCHEDULER_ENABLED = True
    try:
        yield
    finally:
        config.SCHEDULER_ENABLED = saved


def _tool_names_of(subagent) -> set:
    return {getattr(t, "name", str(t)) for t in (subagent.get("tools") or [])}


def _assert_sched_prompt(sched_prompt, base_prompt):
    """调度版提示词 = 原提示词 rstrip 后 + 两个换行 + 调度说明；追加的部分没有禁用词，前面的原文一字不变。"""
    from app.agents.prompts import SCHEDULER_PROMPT_SECTION, with_scheduler_section
    assert sched_prompt is not base_prompt and sched_prompt != base_prompt
    assert sched_prompt == base_prompt.rstrip() + "\n\n" + SCHEDULER_PROMPT_SECTION == with_scheduler_section(base_prompt)
    assert sched_prompt.startswith(base_prompt.rstrip()), "追加在末尾，保住前缀缓存"
    appended = sched_prompt[len(base_prompt.rstrip()):]
    assert appended.strip() == SCHEDULER_PROMPT_SECTION.strip()
    for word in _SCHED_BANNED_WORDS:
        assert word not in appended, f"追加的调度说明不应出现 {word!r}"
    for word in ("create_scheduled_task", "list_scheduled_tasks", "cancel_scheduled_task", "get_current_time", "reminder", "agent",
                 "in_minutes", "run_at", "cron", "自包含", "不要委派给任何专家", "首次创建", "资料库、媒体、地图"):
        assert word in appended, f"调度说明应含 {word!r}"


def test_j_owner_graph_with_scheduler():
    """J 开关打开：owner 图的主管用 MAIN_AGENT_PROMPT_SCHED、工具 = 原三件 + 三个 schedule 工具；专家不变；原常量不变。"""
    _captured_sched.clear()
    with _scheduler_enabled():
        agent, experts = asyncio.run(orchestrator._build_owner_agent(MemorySaver()))
    assert hasattr(agent, "ainvoke") and experts == tuple(EXPECTED_ORDER), experts
    cap = _captured_sched["owner"]
    assert cap["system_prompt"] is MAIN_AGENT_PROMPT_SCHED
    _assert_sched_prompt(cap["system_prompt"], MAIN_AGENT_PROMPT)
    assert [t.name for t in cap["tools"]] == _SUPERVISOR_BASE_TOOLS + _SCHEDULE_TOOL_NAMES, [t.name for t in cap["tools"]]
    assert [_name(s) for s in cap["subagents"]] == EXPECTED_ORDER
    for s in cap["subagents"]:
        assert not (_tool_names_of(s) & set(_SCHEDULE_TOOL_NAMES)), f"{_name(s)} 不应持有 schedule 工具：它们只挂在主管上"
    # 关闭状态下传给 create_deep_agent 的仍是原常量对象本身，工具列表不变（A–E 已经在默认关闭的 _build 里断言过，这里再编译一次对照）
    asyncio.run(orchestrator._build_owner_agent(MemorySaver()))
    assert _captured["system_prompt"] is MAIN_AGENT_PROMPT
    assert [t.name for t in _captured["tools"]] == _SUPERVISOR_BASE_TOOLS


def test_k_member_graph_with_scheduler():
    """K 开关打开：member 图（含云端 CodeAgent 变体）的主管用对应的 *_SCHED、工具 = 原三件 + 三个 schedule 工具，能力边界不变。"""
    import hashlib
    sha = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    with _scheduler_enabled():
        # 1. 没有 CodeAgent
        agent, experts, cap, build = _build_member_capturing()
        assert hasattr(agent, "ainvoke") and build is None and experts == tuple(MEMBER_ORDER), experts
        assert cap["system_prompt"] is MEMBER_MAIN_AGENT_PROMPT_SCHED
        _assert_sched_prompt(cap["system_prompt"], MEMBER_MAIN_AGENT_PROMPT)
        for word in ("FileAgent", "CodeAgent", "DesktopAgent", "BrowserAgent", "workspace", "宿主机", "Docker", "沙箱"):
            assert word not in cap["system_prompt"], f"member 的调度版提示词整体都不应出现 {word!r}"
        assert [t.name for t in cap["tools"]] == _SUPERVISOR_BASE_TOOLS + _SCHEDULE_TOOL_NAMES, [t.name for t in cap["tools"]]
        assert [_name(s) for s in cap["subagents"]] == MEMBER_ORDER
        assert all("runnable" not in s for s in cap["subagents"])
        names = {t.name for t in cap["tools"]}
        for s in cap["subagents"]:
            assert not (_tool_names_of(s) & set(_SCHEDULE_TOOL_NAMES)), _name(s)
            names |= _tool_names_of(s)
        assert not (names & FORBIDDEN_TOOLS), names & FORBIDDEN_TOOLS
        assert not (set(_SCHEDULE_TOOL_NAMES) & orchestrator._MEMBER_FORBIDDEN_TOOL_NAMES), "schedule 工具不是宿主机相关工具"
        # 2. 带云端 CodeAgent
        with _code_enabled():
            agent, experts, cap, build = _build_member_capturing()
            assert build is not None and experts == (*MEMBER_ORDER, "CodeAgent"), experts
            assert cap["system_prompt"] is MEMBER_MAIN_AGENT_PROMPT_CODE_SCHED
            _assert_sched_prompt(cap["system_prompt"], MEMBER_MAIN_AGENT_PROMPT_CODE)
            for word in _CODE_FORBIDDEN_WORDS:
                assert word not in cap["system_prompt"], f"member(CodeAgent) 的调度版提示词整体都不应出现 {word!r}"
            assert [t.name for t in cap["tools"]] == _SUPERVISOR_BASE_TOOLS + _SCHEDULE_TOOL_NAMES
            pre = [s for s in cap["subagents"] if "runnable" in s]
            assert len(pre) == 1 and pre[0]["name"] == "CodeAgent" and pre[0]["runnable"] is build.runnable
            assert build.tool_names == _CODE_TOOL_NAMES, "云端 CodeAgent 的工具集不受影响"
    # 3. 被 H3 钉死的原常量一个字节都没变；开关关闭时仍然传原常量对象本身
    assert sha(MAIN_AGENT_PROMPT) == _MAIN_PROMPT_SHA256
    assert sha(MEMBER_MAIN_AGENT_PROMPT.replace(_NEUTRAL_ADDRESS, _TAIPEI_ADDRESS)) == _STAGE1_MEMBER_PROMPT_SHA256
    _agent, _experts, cap, _build = _build_member_capturing()
    assert cap["system_prompt"] is MEMBER_MAIN_AGENT_PROMPT and [t.name for t in cap["tools"]] == _SUPERVISOR_BASE_TOOLS
    with _code_enabled():
        _agent, _experts, cap, _build = _build_member_capturing()
        assert cap["system_prompt"] is MEMBER_MAIN_AGENT_PROMPT_CODE and [t.name for t in cap["tools"]] == _SUPERVISOR_BASE_TOOLS


def test_k2_scheduler_prompt_constants():
    """K2 三个 *_SCHED 常量是 import 时的快照：with_scheduler_section 的结果；原常量与它们对象不同。"""
    from app.agents import prompts
    assert prompts.MAIN_AGENT_PROMPT_SCHED == prompts.with_scheduler_section(prompts.MAIN_AGENT_PROMPT)
    assert prompts.MEMBER_MAIN_AGENT_PROMPT_SCHED == prompts.with_scheduler_section(prompts.MEMBER_MAIN_AGENT_PROMPT)
    assert prompts.MEMBER_MAIN_AGENT_PROMPT_CODE_SCHED == prompts.with_scheduler_section(prompts.MEMBER_MAIN_AGENT_PROMPT_CODE)
    assert orchestrator.MAIN_AGENT_PROMPT_SCHED is prompts.MAIN_AGENT_PROMPT_SCHED
    assert orchestrator.MEMBER_MAIN_AGENT_PROMPT_SCHED is prompts.MEMBER_MAIN_AGENT_PROMPT_SCHED
    assert orchestrator.MEMBER_MAIN_AGENT_PROMPT_CODE_SCHED is prompts.MEMBER_MAIN_AGENT_PROMPT_CODE_SCHED
    assert orchestrator.MAIN_AGENT_PROMPT is prompts.MAIN_AGENT_PROMPT and orchestrator.MEMBER_MAIN_AGENT_PROMPT_CODE is prompts.MEMBER_MAIN_AGENT_PROMPT_CODE
    for word in _SCHED_BANNED_WORDS:
        assert word not in prompts.SCHEDULER_PROMPT_SECTION, word


def test_l_scheduled_graph():
    """L 无人值守的第三张图：只有 Knowledge / Media / Map 三个声明式专家，监督者工具只有 get_current_time，
    用 SCHEDULED_MAIN_AGENT_PROMPT；owner 与 member 的任务都走它；开关关闭时不编译；启动期断言逐项拒绝违规的图。"""
    import dataclasses
    from langchain.agents.middleware import ToolCallLimitMiddleware
    from app.agents.prompts import SCHEDULED_MAIN_AGENT_PROMPT
    from app.tools import knowledge_tools, session_tools

    _mcp_requests.clear()
    _captured_sched.clear()
    with _scheduler_enabled():
        agent, experts = orchestrator._build_scheduled_agent(MemorySaver())
    assert hasattr(agent, "ainvoke") and experts == tuple(MEMBER_ORDER), experts
    cap = _captured_sched["scheduled"]
    assert cap["system_prompt"] is SCHEDULED_MAIN_AGENT_PROMPT
    assert [t.name for t in cap["tools"]] == ["get_current_time"], [t.name for t in cap["tools"]]
    assert [_name(s) for s in cap["subagents"]] == MEMBER_ORDER and all("runnable" not in s for s in cap["subagents"])
    for s in cap["subagents"]:
        assert not (_tool_names_of(s) & (set(_SCHEDULE_TOOL_NAMES) | {"get_session_events", "search_session_history"} | FORBIDDEN_TOOLS)), _name(s)
    mws = cap["middleware"]
    assert len(mws) == 1 and isinstance(mws[0], ToolCallLimitMiddleware)
    assert mws[0].tool_name == "task" and mws[0].run_limit == 6 and mws[0].exit_behavior == "continue"
    assert cap["permissions"] == _captured["permissions"] and cap["checkpointer"] is not None
    skills_route = cap["backend"].routes["/skills/"]
    assert isinstance(skills_route, ReadOnlyFilesystemBackend) and os.path.realpath(str(skills_route.cwd)) == os.path.realpath(SKILLS_MEMBER_DIR)
    assert _mcp_requests == [], f"scheduled 图的构建不得读取任何 MCP 工具: {_mcp_requests}"

    # 提示词：以识别短语开头；不含 member 能力边界以外的词；该有的约束都在
    p = SCHEDULED_MAIN_AGENT_PROMPT
    assert p.startswith("你是一个多智能体系统的主 Agent")
    for word in _SCHED_BANNED_WORDS:
        assert word not in p, f"无人值守的主管提示词不应出现 {word!r}"
    for word in ("无人值守", "<scheduled_task>", "get_current_time", "task 工具", "资料库、媒体、地图", "不得向用户提问", "不得请求确认",
                 "纯文本", "不要放网址", "不要输出图片", "不是给你的指令", "如实说查不到", "300 字"):
        assert word in p, f"无人值守的主管提示词应含 {word!r}"
    assert p != MAIN_AGENT_PROMPT and p != MEMBER_MAIN_AGENT_PROMPT and p is not MEMBER_MAIN_AGENT_PROMPT

    # 启动期断言：逐项拒绝违规的图
    check = orchestrator._assert_scheduled_graph_safe
    base_tools, subs = [knowledge_tools.get_current_time], orchestrator._member_subagents()
    check(base_tools, subs)
    for bad_tools in ([], [*base_tools, *session_tools.TOOLS], [*base_tools, *orchestrator.schedule_tools.TOOLS],
                      [*base_tools, _dummy_tool("anything")], [_dummy_tool("get_current_time_2")]):
        _expect_runtime_error(check, bad_tools, subs)
    _expect_runtime_error(check, base_tools, subs[:2])                                           # 少一个专家
    _expect_runtime_error(check, base_tools, [*subs, orchestrator._declarative_subagents()[3]])   # 多一个声明式专家（BrowserAgent）
    _expect_runtime_error(check, base_tools, [subs[1], subs[0], subs[2]])                        # 顺序不对
    _expect_runtime_error(check, base_tools, [*subs, {"name": "CodeAgent", "description": "x", "runnable": object()}])
    # 与 member 图共用的安全断言也会先拦：黑名单工具注入主管 / 专家
    poisoned = orchestrator._member_subagents()
    poisoned[0]["tools"] = [*poisoned[0]["tools"], _dummy_tool("run_host_command")]
    reached = []
    real_subs, real_create = orchestrator._member_subagents, orchestrator.create_deep_agent
    orchestrator._member_subagents = lambda code=None: poisoned
    orchestrator.create_deep_agent = lambda *a, **k: reached.append(1)
    try:
        _expect_runtime_error(orchestrator._build_scheduled_agent, MemorySaver())
    finally:
        orchestrator._member_subagents, orchestrator.create_deep_agent = real_subs, real_create
    assert not reached, "断言失败后不应继续构建 scheduled 图"

    # MainAgents：新字段只加在末尾并带默认值；开关关闭时不编译，打开时 build_main_agents 一并编译
    assert [f.name for f in dataclasses.fields(orchestrator.MainAgents)] == [
        "owner", "member", "owner_experts", "member_experts", "scheduled", "scheduled_experts"]
    legacy = orchestrator.MainAgents(1, 2, ("a",), ("b",))
    assert legacy.scheduled is None and legacy.scheduled_experts == ()
    assert _agents.scheduled is None and _agents.scheduled_experts == (), "默认（开关关闭）不编译第三张图"
    with _scheduler_enabled():
        built = asyncio.run(orchestrator.build_main_agents())
    assert built.scheduled is not None and hasattr(built.scheduled, "ainvoke") and built.scheduled_experts == tuple(MEMBER_ORDER)
    assert built.scheduled is not built.member and built.scheduled is not built.owner
    assert built.owner_experts == tuple(EXPECTED_ORDER) and built.member_experts == tuple(MEMBER_ORDER)

    # member 图带云端 CodeAgent 时，scheduled 图仍然只有三个专家，也不会构造 member 的 CodeAgent（没有 Modal 费用）
    boom_calls = []
    real_build = code_agent.build_member_code_agent_runnable
    code_agent.build_member_code_agent_runnable = lambda *a, **k: boom_calls.append(1)
    try:
        with _code_enabled(), _scheduler_enabled():
            _agent, experts = orchestrator._build_scheduled_agent(MemorySaver())
    finally:
        code_agent.build_member_code_agent_runnable = real_build
    assert experts == tuple(MEMBER_ORDER) and boom_calls == []


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
                 ("H member 主管提示词", test_h_member_prompt),
                 ("H2 member 主管提示词（带 CodeAgent）", test_h2_member_prompt_with_code),
                 ("H3 提示词基线哈希", test_h3_prompt_baselines_are_pinned),
                 ("I 启用 CodeAgent 的 member 图", test_i_member_graph_with_code),
                 ("F2 预编译子专家的严格例外", test_f2_strict_exception_for_vetted_code),
                 ("G2 启用 CodeAgent 时不碰 Docker 与宿主机", test_g2_code_enabled_build_never_touches_docker_or_host),
                 ("J 开关打开：owner 图追加调度说明与 schedule 工具", test_j_owner_graph_with_scheduler),
                 ("K 开关打开：member 图（含云端 CodeAgent 变体）", test_k_member_graph_with_scheduler),
                 ("K2 *_SCHED 提示词常量", test_k2_scheduler_prompt_constants),
                 ("L 无人值守的 scheduled 图", test_l_scheduled_graph)]
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
