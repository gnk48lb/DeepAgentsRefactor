"""主管智能体构建与多专家编排 (所在层: L4)"""

from typing import Optional
from deepagents import (
    create_deep_agent,
    CompiledSubAgent,
    register_harness_profile,
    HarnessProfile,
    GeneralPurposeSubagentProfile,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app import config
from app.infra import models, mcp_service
from app.tools import amap, markitdown_tool
from app.agents.prompts import MAIN_AGENT_PROMPT
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


def _register_harness_profiles():
    for _provider in ("google_genai", "openai"):
        try:
            register_harness_profile(
                _provider,
                HarnessProfile(
                    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                    excluded_tools=frozenset({"write_file", "edit_file", "delete", "execute", "ls", "glob", "grep"}),
                ),
            )
        except Exception as e:
            print(f"[-] Warning: Failed to register harness profile for {_provider}: {e}")


_register_harness_profiles()

try:
    from langchain.agents.middleware import ToolCallLimitMiddleware
except ImportError:
    ToolCallLimitMiddleware = None


_GENERAL_PURPOSE = {
    "name": "general-purpose",
    "description": "仅当用户请求明显不属于以上任何专家范围时才使用，绝大多数情况不应该被调用。",
    "system_prompt": (
        "如果你被调用，说明主 Agent 判断这个任务不属于任何专属领域专家的范围。"
        "请如实告知用户无法处理这类请求，不要尝试执行任何工具操作或猜测性回答。"
    ),
    "tools": [],
}


_DECLARATIVE_SPECS = [
    knowledge_agent.SPEC,
    media_agent.SPEC,
    map_agent.SPEC,
    code_agent.SPEC,
    browser_agent.SPEC,
]


def _declarative_subagents() -> list:
    return [spec.to_subagent(models) for spec in _DECLARATIVE_SPECS]


async def _local_subagents() -> tuple[CompiledSubAgent, CompiledSubAgent]:
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
    return file_subagent, desktop_subagent


async def build_main_agent(
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """
    构建基于 deepagents 的主 Agent 实例（单用户：本地操作类专家始终挂载）。
    - 主模型: models.orchestrator_llm
    - 子专家: 声明式接入 (task 委派)，包含 FileAgent / DesktopAgent / BrowserAgent 等全部 7 个专家
    - 中间件: ToolCallLimitMiddleware 防御 task 工具死循环
    - 持久化: 默认使用 MemorySaver
    """
    # 校验高德配置并获取本地工具
    amap.validate_config()
    amap_tools = amap.get_tools()
    if not amap_tools:
        print("⚠️ 警告：MapAgent 没有任何高德工具，地图/天气查询将不可用！")

    # 确保 MCP 服务已初始化
    await mcp_service.initialize_mcp()

    # 配置中间件
    middleware = []
    if ToolCallLimitMiddleware is not None:
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name="task",
                run_limit=12,
                exit_behavior="continue",
            )
        )
    else:
        print("[-] Warning: ToolCallLimitMiddleware not available, task tool limit disabled.")

    subagents = _declarative_subagents()
    file_subagent, desktop_subagent = await _local_subagents()
    subagents.append(file_subagent)
    subagents.append(desktop_subagent)

    # 显式覆盖框架内置的 general-purpose subagent
    gp_subagent = dict(_GENERAL_PURPOSE)
    gp_subagent["model"] = models.worker_llm
    subagents.append(gp_subagent)

    # 断言除 FileAgent 外，任何声明式子 Agent 的工具列表都不得包含 convert_to_markdown
    for sa in subagents:
        if isinstance(sa, dict) and sa.get("name") != "FileAgent":
            tool_names = [getattr(t, "name", str(t)) for t in sa.get("tools", [])]
            if "convert_to_markdown" in tool_names:
                raise RuntimeError(
                    f"子 Agent '{sa.get('name')}' 违规持有 convert_to_markdown 工具！"
                )

    if checkpointer is None:
        checkpointer = MemorySaver()

    agent = create_deep_agent(
        model=models.orchestrator_llm,
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=subagents,
        middleware=middleware,
        backend=build_skills_backend(),
        checkpointer=checkpointer,
    )

    return agent
