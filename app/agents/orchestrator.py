"""主管智能体构建与多专家编排 (所在层: L4)

启动时编译两张主图，共用同一个 checkpointer：
- owner 图：7 个专家全挂（含 FileAgent / DesktopAgent / BrowserAgent / CodeAgent），构造路径与单用户时代完全等价；
- member 图：只有 KnowledgeAgent / MediaAgent / MapAgent，编译期就不含任何通往宿主机的专家与工具。
能力门控放在编译期而不是运行时：member 图里根本不存在通往宿主机资源的路径，不做按身份路由的设计。
"""

from dataclasses import dataclass
from typing import Any, Optional
from deepagents import (
    create_deep_agent,
    CompiledSubAgent,
    register_harness_profile,
    HarnessProfile,
    GeneralPurposeSubagentProfile,
    FilesystemPermission,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app import config
from app.infra import models, mcp_service
from app.tools import amap, knowledge_tools, markitdown_tool, session_tools
from app.agents.prompts import MAIN_AGENT_PROMPT, MEMBER_MAIN_AGENT_PROMPT
from app.agents.skills import build_skills_backend
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


def _tool_names(tools) -> set:
    return {getattr(t, "name", str(t)) for t in (tools or [])}


def _member_subagents() -> list:
    """member 图的子专家：只含 KnowledgeAgent / MediaAgent / MapAgent，按名字从 _DECLARATIVE_SPECS 里挑。"""
    by_name = {spec.name: spec for spec in _DECLARATIVE_SPECS}
    subagents = [by_name[name].to_subagent(models) for name in _MEMBER_EXPERT_NAMES]
    # STAGE2: 云端 CodeAgent
    return subagents


def _assert_member_graph_safe(supervisor_tools: list, subagents: list) -> None:
    """启动期断言（失败即关闭，违反直接阻止启动）：
    - member 图的专家名必须是 {KnowledgeAgent, MediaAgent, MapAgent} 的子集；
    - 预编译的子专家（CompiledSubAgent，如 CodeAgent / FileAgent / DesktopAgent）一律不允许：它们内部的工具查不到；
    - 主管 tools 与各声明式子专家 tools 不得包含宿主机相关的工具名。
    """
    names = [sa["name"] for sa in subagents]
    extra = sorted(set(names) - set(_MEMBER_EXPERT_NAMES))
    if extra:
        raise RuntimeError(f"member 图含有不允许的专家: {extra}")
    bad = sorted(_tool_names(supervisor_tools) & _MEMBER_FORBIDDEN_TOOL_NAMES)
    if bad:
        raise RuntimeError(f"member 图的主管持有宿主机相关工具: {bad}")
    for sa in subagents:
        if "runnable" in sa:
            raise RuntimeError(f"member 图不允许预编译的子专家（无法检查其内部工具）: {sa['name']}")
        bad = sorted(_tool_names(sa.get("tools")) & _MEMBER_FORBIDDEN_TOOL_NAMES)
        if bad:
            raise RuntimeError(f"member 图的子专家 '{sa['name']}' 持有宿主机相关工具: {bad}")


def _build_member_agent(checkpointer: BaseCheckpointSaver):
    """member 图：Knowledge / Media / Map 三个声明式专家 + 主管自带的时间与会话回溯工具。
    绝不读取 desktop / markitdown 的 MCP 工具，也不构造 CodeAgent / FileAgent / DesktopAgent / BrowserAgent。"""
    subagents = _member_subagents()
    supervisor_tools = [knowledge_tools.get_current_time, *session_tools.TOOLS]
    _assert_member_graph_safe(supervisor_tools, subagents)

    agent = create_deep_agent(
        model=models.orchestrator_llm,
        tools=supervisor_tools,
        system_prompt=MEMBER_MAIN_AGENT_PROMPT,
        subagents=subagents,
        middleware=[ToolCallLimitMiddleware(tool_name="task", run_limit=12, exit_behavior="continue")],
        backend=build_skills_backend(),
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

    agent = create_deep_agent(
        model=models.orchestrator_llm,
        tools=[knowledge_tools.get_current_time, *session_tools.TOOLS],
        system_prompt=MAIN_AGENT_PROMPT,
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
    """
    # 校验高德配置并获取本地工具
    amap.validate_config()
    amap_tools = amap.get_tools()
    if not amap_tools:
        print("⚠️ 警告：MapAgent 没有任何高德工具，地图/天气查询将不可用！")

    # 确保 MCP 服务已初始化
    await mcp_service.initialize_mcp()

    if checkpointer is None:
        checkpointer = MemorySaver()

    owner, owner_experts = await _build_owner_agent(checkpointer)
    member, member_experts = _build_member_agent(checkpointer)
    return MainAgents(owner=owner, member=member, owner_experts=owner_experts, member_experts=member_experts)
