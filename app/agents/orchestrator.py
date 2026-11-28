"""主管智能体构建与多专家编排 (所在层: L4)

启动时编译两张主图，共用同一个 checkpointer：
- owner 图：7 个专家全挂（含 FileAgent / DesktopAgent / BrowserAgent / CodeAgent），构造路径与单用户时代完全等价；
- member 图：只有 KnowledgeAgent / MediaAgent / MapAgent，编译期就不含任何通往宿主机的专家与工具。
能力门控放在编译期而不是运行时：member 图里根本不存在通往宿主机资源的路径，不做按身份路由的设计。

cloud 形态（DEPLOY_MODE=cloud，config.is_cloud_mode()，调用时现读）：整台云主机不开放任何“碰这台机器本身”的能力，
所以只编译一张 member 式的主图，owner 与 member 共用（MainAgents.owner is MainAgents.member）：
- 专家是 KnowledgeAgent（用 knowledge_agent.CLOUD_SPEC，多一个读网页的 read_webpage_content）/ MediaAgent / MapAgent，
  Modal 令牌齐全时再加云端 CodeAgent（owner 的代码也走 Modal，沙箱键是 (owner, session_id)）；
- 不构建 FileAgent / DesktopAgent / BrowserAgent 与 owner 的 CodeAgent，不初始化 MCP（desktop、markitdown），
  不碰本机 Docker 沙箱与宿主机命令；
- 提示词用 prompts.CLOUD_MAIN_AGENT_PROMPT*（定时任务图用 CLOUD_SCHEDULED_MAIN_AGENT_PROMPT）；
- 现有的 member / scheduled 启动期断言一点不放松，另加 _assert_cloud_graph_safe 钉死 read_webpage_content 只在 KnowledgeAgent 手里。
desktop 形态（默认）的构建顺序、调用次数与传给 create_deep_agent 的提示词常量与引入 cloud 之前完全一致。
"""

from dataclasses import dataclass
from typing import Any, Optional
import os
from deepagents import (
    create_deep_agent,
    CompiledSubAgent,
    register_harness_profile,
    HarnessProfile,
    GeneralPurposeSubagentProfile,
    FilesystemPermission,
)
from deepagents.backends import CompositeBackend, StateBackend
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app import config
from app.infra import models, mcp_service, modal_sandbox
from app.tools import amap, knowledge_tools, markitdown_tool, schedule_tools, session_tools
from app.agents.prompts import (
    CLOUD_MAIN_AGENT_PROMPT,
    CLOUD_MAIN_AGENT_PROMPT_CODE,
    CLOUD_MAIN_AGENT_PROMPT_CODE_SCHED,
    CLOUD_MAIN_AGENT_PROMPT_SCHED,
    CLOUD_SCHEDULED_MAIN_AGENT_PROMPT,
    MAIN_AGENT_PROMPT,
    MAIN_AGENT_PROMPT_SCHED,
    MEMBER_MAIN_AGENT_PROMPT,
    MEMBER_MAIN_AGENT_PROMPT_CODE,
    MEMBER_MAIN_AGENT_PROMPT_CODE_SCHED,
    MEMBER_MAIN_AGENT_PROMPT_SCHED,
    SCHEDULED_MAIN_AGENT_PROMPT,
)
from app.agents.skills import (
    SKILLS_MEMBER_DIR,
    ReadOnlyFilesystemBackend,
    build_member_skills_backend,
    build_skills_backend,
)
from app.agents.experts import (
    knowledge_agent,
    media_agent,
    map_agent,
    code_agent,
    browser_agent,
    file_agent,
    desktop_agent,
)

# 与 provider 无关的第二道保险:主管永远不写文件。
# HarnessProfile.excluded_tools 按 provider 名字注册(google_genai/openai/anthropic),
# 换成没注册的 provider 时写工具会静默回来;permissions 是工具层的硬拒绝,与 provider 无关。
# 语义差异:excluded_tools 让工具"不可见",permissions 让工具"可见但调用报错"——两者并存。
_SUPERVISOR_PERMISSIONS = [
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
]


def _register_harness_profiles():
    for _provider in ("google_genai", "openai", "anthropic"):
        try:
            register_harness_profile(
                _provider,
                HarnessProfile(
                    # 禁用框架内置 general-purpose（它会继承主管的全部工具）；本项目不再提供兜底子专家。
                    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                    excluded_tools=frozenset({"write_file", "edit_file", "delete", "execute", "ls", "glob", "grep"}),
                ),
            )
        except Exception as e:
            print(f"[-] Warning: Failed to register harness profile for {_provider}: {e}")


_register_harness_profiles()

from langchain.agents.middleware import ToolCallLimitMiddleware


_DECLARATIVE_SPECS = [
    knowledge_agent.SPEC,
    media_agent.SPEC,
    map_agent.SPEC,
    browser_agent.SPEC,
]


def _declarative_subagents() -> list:
    return [spec.to_subagent(models) for spec in _DECLARATIVE_SPECS]


async def _local_subagents() -> tuple[CompiledSubAgent, CompiledSubAgent, CompiledSubAgent]:
    # CodeAgent / FileAgent 由 create_agent 独立编译；上方 HarnessProfile 只作用于主图与声明式专家，
    # 不会接管这两个 CompiledSubAgent 自己声明的工具白名单与 HITL。
    code_runnable = code_agent.build_code_agent_runnable(
        llm=models.worker_llm,
        system_prompt=code_agent.PROMPT,
    )
    code_subagent = CompiledSubAgent(
        name=code_agent.NAME,
        description=code_agent.DESCRIPTION,
        runnable=code_runnable,
    )

    desktop_tools = mcp_service.get_tools_by_server("desktop")

    markitdown_tools = mcp_service.get_tools_by_server("markitdown")
    file_extra = markitdown_tool.build_tools(markitdown_tools, config.WORKSPACE_DIR)
    if not file_extra:
        print("⚠️ 警告：FileAgent 未获取到 convert_to_markdown 工具，请检查 uvx 与 markitdown-mcp 环境！")

    file_agent_runnable = file_agent.build_file_agent_runnable(
        llm=models.worker_llm,
        system_prompt=file_agent.PROMPT + (file_agent.PROMPT_MARKITDOWN if file_extra else ""),
        workspace_dir=config.WORKSPACE_DIR,
        extra_tools=file_extra,
    )
    file_subagent = CompiledSubAgent(
        name=file_agent.NAME,
        description=file_agent.DESCRIPTION,
        runnable=file_agent_runnable,
    )
    desktop_agent_subgraph = desktop_agent.build_desktop_agent_subgraph(
        desktop_tools=desktop_tools,
        system_prompt=desktop_agent.PROMPT,
    )
    desktop_subagent = CompiledSubAgent(
        name=desktop_agent.NAME,
        description=desktop_agent.DESCRIPTION,
        runnable=desktop_agent_subgraph,
    )
    return code_subagent, file_subagent, desktop_subagent


# member 图允许的专家：按名字从 _DECLARATIVE_SPECS 里挑（_DECLARATIVE_SPECS 本身保持不变）
_MEMBER_EXPERT_NAMES = ("KnowledgeAgent", "MediaAgent", "MapAgent")

# member 图里显式传入的工具（主管 tools 与各声明式子专家 tools）不得出现的宿主机相关工具名：启动期断言
_MEMBER_FORBIDDEN_TOOL_NAMES = frozenset({
    "run_host_command", "move_file", "create_directory", "convert_to_markdown",
    "take_grid_screenshot", "click_grid_center", "safe_input_text", "press_hotkey",
    "execute_complex_browser_action", "execute_python_code",
    "execute", "write_file", "edit_file",
})


@dataclass(frozen=True)
class MainAgents:
    """两张编译好的主图，以及各自挂了哪些专家（启动时打印用）。"""
    owner: Any
    member: Any
    owner_experts: tuple
    member_experts: tuple
    # 无人值守的定时任务专用图（只有 SCHEDULER_ENABLED 打开时才编译）；新字段只能加在末尾并带默认值
    scheduled: Any = None
    scheduled_experts: tuple = ()


def _tool_names(tools) -> set:
    return {getattr(t, "name", str(t)) for t in (tools or [])}


# 经过验证的 member CodeAgent 的工具集合：恰为这 8 个，不含 delete
_MEMBER_CODE_TOOL_NAMES = frozenset({
    "execute_python_code", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute",
})


def _member_subagents(code=None, *, cloud=False) -> list:
    """member 图的子专家：KnowledgeAgent / MediaAgent / MapAgent（按名字从 _DECLARATIVE_SPECS 里挑），
    code 是通过构建的 MemberCodeAgentBuild 时再追加云端 CodeAgent（顺序：Knowledge、Media、Map、Code）。
    cloud=True 时 KnowledgeAgent 换成 knowledge_agent.CLOUD_SPEC（多一个读网页的工具；_DECLARATIVE_SPECS 本身不动），顺序不变。"""
    by_name = {spec.name: spec for spec in _DECLARATIVE_SPECS}
    if cloud:
        by_name[knowledge_agent.CLOUD_SPEC.name] = knowledge_agent.CLOUD_SPEC
    subagents = [by_name[name].to_subagent(models) for name in _MEMBER_EXPERT_NAMES]
    if code is not None:
        subagents.append(CompiledSubAgent(
            name=code_agent.NAME,
            description=code_agent.MEMBER_DESCRIPTION,
            runnable=code.runnable,
        ))
    return subagents


def _assert_vetted_code_build(vetted_code) -> None:
    """member 的云端 CodeAgent 构建物断言（失败即关闭）：工具集恰为 8 个、后端默认是 ModalRoutedSandbox（不是本机 Docker）、
    路由恰为 /skills/（只读、指向 skills_member/）与 /_artifacts/（内存 state）。"""
    if vetted_code.tool_names != _MEMBER_CODE_TOOL_NAMES:
        raise RuntimeError(f"member 的 CodeAgent 工具集不对: {sorted(vetted_code.tool_names)}")
    backend = vetted_code.backend
    if not isinstance(backend, CompositeBackend):
        raise RuntimeError("member 的 CodeAgent 后端必须是 CompositeBackend")
    if not isinstance(backend.default, modal_sandbox.ModalRoutedSandbox):
        raise RuntimeError(f"member 的 CodeAgent 默认后端必须是云端路由沙箱，实际是 {type(backend.default).__name__}")
    if set(backend.routes) != {"/skills/", "/_artifacts/"}:
        raise RuntimeError(f"member 的 CodeAgent 路由不对: {sorted(backend.routes)}")
    skills = backend.routes["/skills/"]
    if not isinstance(skills, ReadOnlyFilesystemBackend):
        raise RuntimeError("member 的 CodeAgent 的 /skills/ 必须是只读后端")
    if os.path.realpath(str(skills.cwd)) != os.path.realpath(SKILLS_MEMBER_DIR):
        raise RuntimeError("member 的 CodeAgent 的 /skills/ 必须指向 skills_member/")
    if not isinstance(backend.routes["/_artifacts/"], StateBackend):
        raise RuntimeError("member 的 CodeAgent 的 /_artifacts/ 必须是内存 state 后端")


def _assert_member_graph_safe(supervisor_tools: list, subagents: list, vetted_code=None) -> None:
    """启动期断言（失败即关闭，违反直接阻止启动）：
    - member 图的专家名必须是 {KnowledgeAgent, MediaAgent, MapAgent} 的子集（带 vetted_code 时再额外允许 CodeAgent）；
    - 预编译的子专家（CompiledSubAgent，如 FileAgent / DesktopAgent）一律不允许：它们内部的工具查不到。
      唯一的严格例外：vetted_code 通过构建物断言，且名字是 CodeAgent、runnable 就是它的 runnable（身份比较）、整张图里只有这一个预编译专家；
    - 主管 tools 与各声明式子专家 tools 不得包含宿主机相关的工具名（即使带着 vetted_code 也照样检查）。
    """
    names = [sa["name"] for sa in subagents]
    allowed = set(_MEMBER_EXPERT_NAMES) | ({code_agent.NAME} if vetted_code is not None else set())
    extra = sorted(set(names) - allowed)
    if extra:
        raise RuntimeError(f"member 图含有不允许的专家: {extra}")
    bad = sorted(_tool_names(supervisor_tools) & _MEMBER_FORBIDDEN_TOOL_NAMES)
    if bad:
        raise RuntimeError(f"member 图的主管持有宿主机相关工具: {bad}")
    precompiled = [sa for sa in subagents if "runnable" in sa]
    for sa in subagents:
        if "runnable" in sa:
            if (
                vetted_code is None
                or sa["name"] != code_agent.NAME
                or sa["runnable"] is not vetted_code.runnable
                or len(precompiled) != 1
            ):
                raise RuntimeError(f"member 图不允许预编译的子专家（无法检查其内部工具）: {sa['name']}")
            _assert_vetted_code_build(vetted_code)
            continue
        if sa["name"] == code_agent.NAME:
            raise RuntimeError(f"member 图里叫 {code_agent.NAME} 的子专家必须是经过验证的预编译 CodeAgent")
        bad = sorted(_tool_names(sa.get("tools")) & _MEMBER_FORBIDDEN_TOOL_NAMES)
        if bad:
            raise RuntimeError(f"member 图的子专家 '{sa['name']}' 持有宿主机相关工具: {bad}")


def _assert_scheduled_graph_safe(supervisor_tools: list, subagents: list) -> None:
    """scheduled 图的启动期断言（失败即关闭）：监督者工具名恰为 {get_current_time}（没有任何 schedule / 会话回溯工具，
    无人值守的任务既不能再创建任务，也不能读会话记录）；专家名恰为 Knowledge / Media / Map，没有任何预编译专家
    （CodeAgent 等会带来 HITL 中断与云端费用）。"""
    names = _tool_names(supervisor_tools)
    if names != {"get_current_time"}:
        raise RuntimeError(f"scheduled 图的主管工具必须恰为 get_current_time，实际是 {sorted(names)}")
    expert_names = [sa["name"] for sa in subagents]
    if expert_names != list(_MEMBER_EXPERT_NAMES):
        raise RuntimeError(f"scheduled 图的专家必须恰为 {list(_MEMBER_EXPERT_NAMES)}，实际是 {expert_names}")
    precompiled = [sa["name"] for sa in subagents if "runnable" in sa]
    if precompiled:
        raise RuntimeError(f"scheduled 图不允许预编译的子专家: {precompiled}")


# cloud 图里读网页的工具：只有 KnowledgeAgent 能拿着它（它不是宿主机相关工具，不在上面的黑名单里，别加进去）
_CLOUD_WEB_TOOL_NAME = "read_webpage_content"
_CLOUD_KNOWLEDGE_TOOL_NAMES = ["rag", "web_search", _CLOUD_WEB_TOOL_NAME]


def _assert_cloud_graph_safe(supervisor_tools: list, subagents: list, vetted_code=None) -> None:
    """cloud 图的启动期断言（失败即关闭，叠加在 _assert_member_graph_safe 之上，不替代它）：
    KnowledgeAgent 的工具名恰为 [rag, web_search, read_webpage_content]；主管与其余专家（含带 vetted_code 的 CodeAgent）
    都不得持有 read_webpage_content。"""
    if _CLOUD_WEB_TOOL_NAME in _tool_names(supervisor_tools):
        raise RuntimeError(f"cloud 图的主管不得持有 {_CLOUD_WEB_TOOL_NAME}")
    knowledge = [sa for sa in subagents if sa["name"] == knowledge_agent.CLOUD_SPEC.name]
    if len(knowledge) != 1:
        raise RuntimeError(f"cloud 图里必须恰有一个 {knowledge_agent.CLOUD_SPEC.name}，实际 {len(knowledge)} 个")
    got = [getattr(t, "name", str(t)) for t in (knowledge[0].get("tools") or [])]
    if got != _CLOUD_KNOWLEDGE_TOOL_NAMES:
        raise RuntimeError(f"cloud 图的 KnowledgeAgent 工具必须恰为 {_CLOUD_KNOWLEDGE_TOOL_NAMES}，实际是 {got}")
    for sa in subagents:
        if sa is not knowledge[0] and _CLOUD_WEB_TOOL_NAME in _tool_names(sa.get("tools")):
            raise RuntimeError(f"cloud 图的子专家 '{sa['name']}' 不得持有 {_CLOUD_WEB_TOOL_NAME}")
    if vetted_code is not None and _CLOUD_WEB_TOOL_NAME in vetted_code.tool_names:
        raise RuntimeError(f"cloud 图的 CodeAgent 不得持有 {_CLOUD_WEB_TOOL_NAME}")


def _build_scheduled_agent(checkpointer: BaseCheckpointSaver, *, cloud=False):
    """无人值守的第三张图：Knowledge / Media / Map 三个声明式专家，监督者工具只有 get_current_time。
    owner 的定时任务也走它：不复用 owner 图（有宿主机能力），也不复用 member 图（带会话回溯工具，可能带云端 CodeAgent）。
    图里不挂任何会产生 HITL 中断的工具，也没有会话回溯 / schedule 工具。只在 config.SCHEDULER_ENABLED 为真时构建（调用方判断）。
    cloud=True：KnowledgeAgent 用 CLOUD_SPEC（多读网页工具），提示词用 CLOUD_SCHEDULED_MAIN_AGENT_PROMPT，其余不变。
    desktop 下仍按原样调用 _member_subagents(None) 并传原常量对象本身。"""
    subagents = _member_subagents(None, cloud=True) if cloud else _member_subagents(None)
    supervisor_tools = [knowledge_tools.get_current_time]
    _assert_member_graph_safe(supervisor_tools, subagents)
    _assert_scheduled_graph_safe(supervisor_tools, subagents)
    if cloud:
        _assert_cloud_graph_safe(supervisor_tools, subagents)
    agent = create_deep_agent(
        model=models.orchestrator_llm,
        tools=supervisor_tools,
        system_prompt=CLOUD_SCHEDULED_MAIN_AGENT_PROMPT if cloud else SCHEDULED_MAIN_AGENT_PROMPT,
        subagents=subagents,
        middleware=[ToolCallLimitMiddleware(tool_name="task", run_limit=6, exit_behavior="continue")],
        backend=build_member_skills_backend(),
        permissions=_SUPERVISOR_PERMISSIONS,
        checkpointer=checkpointer,
    )
    return agent, tuple(sa["name"] for sa in subagents)


def _build_member_agent(checkpointer: BaseCheckpointSaver, *, cloud=False):
    """member 图：Knowledge / Media / Map 三个声明式专家 + 主管自带的时间与会话回溯工具；
    Modal 可用（member_code_status 为真）时再多一个云端 CodeAgent。不可用时与阶段 1 等价，不调用任何 CodeAgent 构建函数。
    绝不读取 desktop / markitdown 的 MCP 工具，也不构造 owner 的 CodeAgent / FileAgent / DesktopAgent / BrowserAgent，
    不触达本机 Docker 沙箱与宿主机命令。
    cloud=True（cloud 形态下 owner 与 member 共用这张图）：KnowledgeAgent 用 CLOUD_SPEC（多读网页工具），
    提示词按“有无 CodeAgent、SCHEDULER_ENABLED 是否打开”在 4 个 CLOUD_MAIN_AGENT_PROMPT* 里选一个，其余逻辑不变。
    desktop 下仍按原样调用 _member_subagents(code) 并传原常量对象本身。"""
    ok, _reason = modal_sandbox.member_code_status()
    code = None
    if ok:
        code = code_agent.build_member_code_agent_runnable(
            llm=models.worker_llm,
            system_prompt=code_agent.MEMBER_PROMPT,
        )
    subagents = _member_subagents(code, cloud=True) if cloud else _member_subagents(code)
    supervisor_tools = [knowledge_tools.get_current_time, *session_tools.TOOLS]
    if cloud:
        system_prompt = CLOUD_MAIN_AGENT_PROMPT_CODE if code is not None else CLOUD_MAIN_AGENT_PROMPT
    else:
        system_prompt = MEMBER_MAIN_AGENT_PROMPT_CODE if code is not None else MEMBER_MAIN_AGENT_PROMPT
    if config.SCHEDULER_ENABLED:            # 开关在调用时现读：打开才追加调度说明与三个 schedule 工具，关闭时与原来完全一致
        supervisor_tools = [*supervisor_tools, *schedule_tools.TOOLS]
        if cloud:
            system_prompt = CLOUD_MAIN_AGENT_PROMPT_CODE_SCHED if code is not None else CLOUD_MAIN_AGENT_PROMPT_SCHED
        else:
            system_prompt = MEMBER_MAIN_AGENT_PROMPT_CODE_SCHED if code is not None else MEMBER_MAIN_AGENT_PROMPT_SCHED
    _assert_member_graph_safe(supervisor_tools, subagents, vetted_code=code)
    if cloud:
        _assert_cloud_graph_safe(supervisor_tools, subagents, vetted_code=code)

    agent = create_deep_agent(
        model=models.orchestrator_llm,
        tools=supervisor_tools,
        system_prompt=system_prompt,
        subagents=subagents,
        middleware=[ToolCallLimitMiddleware(tool_name="task", run_limit=12, exit_behavior="continue")],
        backend=build_member_skills_backend(),
        permissions=_SUPERVISOR_PERMISSIONS,
        checkpointer=checkpointer,
    )
    return agent, tuple(sa["name"] for sa in subagents)


async def _build_owner_agent(checkpointer: BaseCheckpointSaver):
    """
    owner 图：构造路径与单用户时代完全等价（7 个专家、顺序、工具、MAIN_AGENT_PROMPT，CodeAgent 仍绑定本机 Docker 沙箱）。
    - 主模型: models.orchestrator_llm
    - 子专家: 声明式接入 (task 委派)，包含 FileAgent / DesktopAgent / BrowserAgent 等全部 7 个专家
    - 主管工具: get_current_time（时间类问题由主管就地回答，不再委派）、
      get_session_events / search_session_history（只读回溯会话记录，直接挂在主管上，不经 task 委派）
    - 中间件: ToolCallLimitMiddleware 防御 task 工具死循环
    """
    # 配置中间件
    middleware = [
        ToolCallLimitMiddleware(
            tool_name="task",
            run_limit=12,
            exit_behavior="continue",
        )
    ]

    decl_subagents = _declarative_subagents()  # [Knowledge, Media, Map, Browser]
    code_subagent, file_subagent, desktop_subagent = await _local_subagents()
    subagents = [
        decl_subagents[0],  # KnowledgeAgent
        decl_subagents[1],  # MediaAgent
        decl_subagents[2],  # MapAgent
        code_subagent,      # CodeAgent
        decl_subagents[3],  # BrowserAgent
        file_subagent,      # FileAgent
        desktop_subagent,   # DesktopAgent
    ]

    # 断言除 FileAgent 外，任何声明式子 Agent 的工具列表都不得包含 convert_to_markdown
    for sa in subagents:
        if isinstance(sa, dict) and sa.get("name") != "FileAgent":
            tool_names = [getattr(t, "name", str(t)) for t in sa.get("tools", [])]
            if "convert_to_markdown" in tool_names:
                raise RuntimeError(
                    f"子 Agent '{sa.get('name')}' 违规持有 convert_to_markdown 工具！"
                )

    supervisor_tools = [knowledge_tools.get_current_time, *session_tools.TOOLS]
    system_prompt = MAIN_AGENT_PROMPT
    if config.SCHEDULER_ENABLED:            # 开关在调用时现读：打开才追加调度说明与三个 schedule 工具，关闭时与原来完全一致
        supervisor_tools = [*supervisor_tools, *schedule_tools.TOOLS]
        system_prompt = MAIN_AGENT_PROMPT_SCHED

    agent = create_deep_agent(
        model=models.orchestrator_llm,
        tools=supervisor_tools,
        system_prompt=system_prompt,
        subagents=subagents,
        middleware=middleware,
        backend=build_skills_backend(),
        permissions=_SUPERVISOR_PERMISSIONS,
        checkpointer=checkpointer,
    )

    return agent, tuple(sa["name"] for sa in subagents)


async def build_main_agents(
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> MainAgents:
    """编译 owner 与 member 两张主图（共用同一个 checkpointer）。

    高德配置校验与 MCP 初始化只做一次；持久化由调用方传入 checkpointer（生产为 AsyncSqliteSaver），不传时回退 MemorySaver。
    cloud 形态（config.is_cloud_mode()，调用时现读）：跳过 MCP 初始化，不构建 owner 图与本机专家，只编译一张 member 式的图，
    owner 与 member 是同一个对象（owner_experts 也同样）；scheduled 图照旧由 SCHEDULER_ENABLED 决定。
    """
    cloud = config.is_cloud_mode()

    # 校验高德配置并获取本地工具
    amap.validate_config()
    amap_tools = amap.get_tools()
    if not amap_tools:
        print("⚠️ 警告：MapAgent 没有任何高德工具，地图/天气查询将不可用！")

    # 确保 MCP 服务已初始化（cloud 不加载任何本机 MCP：desktop、markitdown）
    if not cloud:
        await mcp_service.initialize_mcp()

    if checkpointer is None:
        checkpointer = MemorySaver()

    if cloud:
        member, member_experts = _build_member_agent(checkpointer, cloud=True)
        owner, owner_experts = member, member_experts
    else:
        owner, owner_experts = await _build_owner_agent(checkpointer)
        member, member_experts = _build_member_agent(checkpointer)
    scheduled, scheduled_experts = None, ()
    if config.SCHEDULER_ENABLED:            # 开关在调用时现读：关闭时不编译第三张图
        if cloud:
            scheduled, scheduled_experts = _build_scheduled_agent(checkpointer, cloud=True)
        else:
            scheduled, scheduled_experts = _build_scheduled_agent(checkpointer)
    return MainAgents(
        owner=owner, member=member, owner_experts=owner_experts, member_experts=member_experts,
        scheduled=scheduled, scheduled_experts=scheduled_experts,
    )
