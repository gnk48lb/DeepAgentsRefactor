"""代码执行与数据分析专家智能体构建 (所在层: L4)

CodeAgent：基于 deepagents 的沙箱工作台，挂载 DockerSandbox、FilesystemMiddleware、
SkillsMiddleware 与 SandboxOutputsMiddleware（产物统一提取与多模态回传）。
提供工具：ls, read_file, write_file, edit_file, glob, grep, execute 以及 execute_python_code。
注意：故意不含 delete 工具；/workspace 只读挂载；/skills 走宿主虚拟路径。
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import ToolMessage

from app import config
from app.agents.budget import build_budget_middleware, tool_limit_for
from app.agents.prompts import WORKER_BASE_PROMPT
from app.agents.skills import SKILLS_DIR, SKILLS_MEMBER_DIR, SKILLS_ROUTE, ReadOnlyFilesystemBackend
from app.infra import modal_sandbox, sandbox
from app.tools import code_tools

NAME = "CodeAgent"

DESCRIPTION = (
    "代码与终端专家：在隔离 Docker 沙箱里写代码、跑 Python 与 Linux 命令（rg/jq/unzip…）、"
    "读写沙箱内文件（/work），只读访问用户工作区（/workspace）；用于计算、数据分析与绘图、批量处理文件。"
    "不能联网，不能修改用户工作区（新建/修改 workspace 文件一律找 FileAgent）。需要绘图要明确要求。"
    "需要联网、宿主机已安装的工具（git / pip / npm 等）、或要在用户电脑上真正执行的命令，一律找 FileAgent。"
)

PROMPT = (
    WORKER_BASE_PROMPT
    + "你是 CodeAgent，专职在隔离 Docker 沙箱里编写执行代码、运行 Linux 命令行工具、解决数学计算、数据分析与绘图任务。\n"
    + "【运行环境与文件权限 - 强制】\n"
    + "1. 全部使用绝对路径；主管或用户若写成 workspace/xxx，必须规整为 /workspace/xxx。\n"
    + "2. /workspace 是用户工作区的只读镜像（路径与其他专家一致），只能只读打开（如 pd.read_csv('/workspace/data.csv')），严禁尝试写入或修改，改写工作区文件一律找 FileAgent。\n"
    + "3. /work 是沙箱会话可读写工作目录（cwd，会话内保留，沙箱超时重建后清空）；多文件脚本写在 /work 下。\n"
    + "4. /work/outputs/ 是专用交付箱：保存到这里的图片会自动发送给用户，同一文件每个版本只发一次。\n"
    + "5. /skills/ 是只读技能目录，只能用 read_file(limit=1000) 读，在沙箱 shell 里看不到。\n"
    + "【工具决策表】\n"
    + "1. 一次性 Python（计算/分析/绘图）→ execute_python_code（无需写 python -c）。\n"
    + "2. Shell 命令（ls/rg/jq/unzip/zip/file/tree/wc/head…）→ execute：每次调用都是独立 subshell，cd 与 export 环境变量不跨调用保留，多条命令请用 '&&' 串联；默认 60s，可用 timeout 参数至多 300s。\n"
    + "3. 多文件/多步脚本 → 用 write_file 写到 /work/xxx.py → execute('python3 /work/xxx.py') 运行 → 局部微调优先用 edit_file 避免整文件重写 → 再次运行。\n"
    + "4. 读取与检索 → 用 ls / glob / grep / read_file；grep 是字面精确匹配，正则表达式搜索请用 execute('rg <regex> ...')；PDF/Word 等富文档无法在沙箱读取，如实汇报由 FileAgent 转换。\n"
    + "【行为红线与禁止事项】\n"
    + "1. 严禁尝试写入 /workspace；\n"
    + "2. 严禁 pip install（沙箱无网络连接，依赖均已预置）；\n"
    + "3. 严禁对根目录 '/' 或整个 '/workspace' 盲目递归搜索（必须先用 ls 缩小范围，限定子目录与文件类型）；\n"
    + "4. 纯计算任务无需读取技能；绘图任务必须先 read_file('/skills/matplotlib-chart/SKILL.md', limit=1000)；数据分析任务必须先 read_file('/skills/data-analysis/SKILL.md', limit=1000)。\n"
    + "【汇报格式】汇报遵循工作专家标准格式，以“【内部专家汇报 - 来源：CodeAgent】”开头；不向用户暴露沙箱内部临时路径。"
)

CODE_AGENT_PROMPT = PROMPT

# 沙箱工作台挂载的文件系统工具白名单（故意不含 delete）
SANDBOX_FS_TOOLS = ["ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"]


class SandboxOutputsMiddleware(AgentMiddleware):
    """拦截 execute 工具调用，统一扫描 /work/outputs 产物并推送到图片发件箱与模型多模态 blocks。"""

    def __init__(self, sandbox_instance: Any):
        super().__init__()
        self.sandbox = sandbox_instance

    def _process_outputs(self, result: Any) -> Any:
        try:
            if hasattr(self.sandbox, "collect_new_outputs"):
                outputs = self.sandbox.collect_new_outputs()
                if outputs.images or outputs.notes or outputs.other_files:
                    sent_names, model_blocks, fail_notes = code_tools.publish_images(
                        outputs.images, outputs.notes
                    )
                    if isinstance(result, ToolMessage) and (sent_names or model_blocks):
                        orig_text = result.content if isinstance(result.content, str) else ""
                        notice = (
                            f"\n[系统提示] 已向用户发送 {len(sent_names)} 张图片（{', '.join(sent_names)}）；"
                            f"附带的是缩略预览，仅供你检查"
                        )
                        new_content: list[dict[str, Any]] = [
                            {"type": "text", "text": orig_text + notice},
                            *model_blocks,
                        ]
                        return result.model_copy(update={"content": new_content})
        except Exception as e:
            print(f"[-] SandboxOutputsMiddleware warning: {e}")
        return result

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        result = handler(request)
        tool_name = getattr(request, "tool_call", {}).get("name")
        if tool_name == "execute":
            result = self._process_outputs(result)
        return result

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        result = await handler(request)
        tool_name = getattr(request, "tool_call", {}).get("name")
        if tool_name == "execute":
            # 异步执行产物收集，防止阻塞事件循环
            result = await asyncio.to_thread(self._process_outputs, result)
        return result


def build_code_agent_runnable(
    llm,
    system_prompt: str,
    checkpointer=None,
    sandbox_backend=None,
    max_tool_calls=None,
):
    """返回可直接放进 CompiledSubAgent(runnable=...) 的已编译 CodeAgent 图。"""
    if max_tool_calls is None:
        max_tool_calls = tool_limit_for(NAME)

    sb = sandbox_backend if sandbox_backend is not None else sandbox.get_sandbox()
    backend = CompositeBackend(
        default=sb,
        routes={
            SKILLS_ROUTE: FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True),
            "/_artifacts/": StateBackend(),
        },
        artifacts_root="/_artifacts",
    )

    return create_agent(
        model=llm,
        tools=list(code_tools.TOOLS),
        system_prompt=system_prompt,
        middleware=[
            SandboxOutputsMiddleware(sb),
            SkillsMiddleware(backend=backend, sources=[SKILLS_ROUTE]),
            FilesystemMiddleware(
                backend=backend,
                tools=SANDBOX_FS_TOOLS,
                max_execute_timeout=config.SANDBOX_MAX_TIMEOUT_SECONDS,
            ),
            *build_budget_middleware(max_tool_calls),
            # 与 deepagents 的尾部约定一致：缓存断点中间件放最内层，看到的是其他中间件处理完的最终请求；
            # 非 Anthropic 模型（如回退 Gemini）时静默 no-op。
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        ],
        checkpointer=checkpointer,
    )


# ══════════════════════════════════════════════════════════════════════════════
# member 的云端 CodeAgent（Modal 云端沙箱）
# 与上面 owner 的 Docker 版完全分开：不引用 build_code_agent_runnable，不引用本机沙箱的任何入口。
# 描述 / 提示词里随配置变化的数字与联网句子，由下面两个函数调用时才从 config 现算；
# MEMBER_DESCRIPTION / MEMBER_PROMPT 是它们在 import 时的快照（编译进图的用这两个常量）。
# ══════════════════════════════════════════════════════════════════════════════

def _idle_minutes() -> int:
    return max(1, (config.MODAL_IDLE_TIMEOUT_SECONDS + 30) // 60)


def build_member_description() -> str:
    """进主管 task 工具描述的那段：云端沙箱的事实（联网与否随 config.MODAL_BLOCK_NETWORK）。"""
    net = (
        "沙箱不能联网（不能下载数据，也不能 pip install）；"
        if config.MODAL_BLOCK_NETWORK
        else "沙箱可以联网（下载公开数据、pip 安装缺少的库）；"
    )
    return (
        "代码与计算专家：在云端隔离沙箱里写代码、跑 Python 与 Linux 命令，用于数学计算、数据处理与分析、绘图、批量处理文本与数据。"
        + net
        + "访问不到用户电脑上的任何文件，要处理的数据必须直接写进任务描述；"
        f"沙箱闲置约 {_idle_minutes()} 分钟会被回收，之前生成的文件随之丢失；需要绘图要明确要求。"
    )


def build_member_prompt() -> str:
    """member 的 CodeAgent 的系统提示词（以 WORKER_BASE_PROMPT 开头，必须含“你是 CodeAgent”：cli_logger 靠它识别角色）。"""
    if config.MODAL_BLOCK_NETWORK:
        net_rule = "沙箱不能联网，不能下载数据，也不能 pip install；只能使用预装的库，数据必须直接写进任务描述。"
        shell_tools = "ls/rg/jq/unzip/zip/file/tree/wc/head…"
        read_hint = "（用户的文件进不来，只有代码生成的文件可读）"
    else:
        net_rule = (
            "沙箱可以联网（有公网）：可以用 requests / httpx / curl 下载公开数据；"
            "缺少的库可以 pip install（先用 execute('pip list | rg 名称') 确认没有预装）。下载来的数据先看大小再读。"
        )
        shell_tools = "ls/rg/jq/unzip/zip/file/tree/wc/head/curl/pip…"
        read_hint = "（前提是文件已从公开网址下载到 /work）"
    mem_gib = f"{config.MODAL_MEMORY_MB_LIMIT / 1024:g}"
    return (
        WORKER_BASE_PROMPT
        + "你是 CodeAgent，专职在云端隔离沙箱里编写执行代码、运行 Linux 命令行工具、解决数学计算、数据处理与分析、绘图任务。\n"
        + "【运行环境 - 强制】\n"
        + f"1. 全部使用绝对路径。/work 是沙箱的可读写工作目录（cwd 与 HOME），同一会话内保留；沙箱闲置约 {_idle_minutes()} 分钟或被重置后会被回收，"
        + "/work 里的文件随之丢失，重要的中间结果要写进汇报。\n"
        + "2. /work/outputs/ 是专用交付箱：保存到这里的图片会自动发送给用户，同一文件每个版本只发一次。\n"
        + "3. 你看不到用户电脑上的任何文件，沙箱里也没有用户的文件；要处理的数据只会出现在任务描述里。数据缺失时如实汇报缺什么，不要编造数据。\n"
        + "4. /skills/ 是只读技能目录，只能用 read_file(limit=1000) 读，它不属于沙箱文件系统，在沙箱 shell 里看不到。\n"
        + "5. " + net_rule + "\n"
        + "6. 环境：Python 3.13；预装 pandas、numpy、matplotlib、scipy、seaborn、scikit-learn、statsmodels、sympy、openpyxl、networkx、Pillow、"
        + "requests、httpx、beautifulsoup4、lxml、pypdf、pdfplumber、python-docx、python-pptx 等，以及 rg、jq、unzip、zip、file、tree、git、curl、wget、pdftotext 等命令行工具。\n"
        + f"7. 资源限制：单次执行默认 {config.MODAL_EXEC_TIMEOUT_SECONDS} 秒、最长 {config.MODAL_EXEC_MAX_TIMEOUT_SECONDS} 秒（execute 可用 timeout 参数调整）；"
        + f"内存上限约 {mem_gib} GiB；单次最多导出 {config.SANDBOX_MAX_IMAGES} 张图片。超时或超内存的命令会被终止。\n"
        + "【工具决策表】\n"
        + "1. 一次性 Python（计算/分析/绘图）→ execute_python_code（无需写 python -c）。\n"
        + f"2. Shell 命令（{shell_tools}）→ execute：每次调用都是独立 subshell，cd 与 export 环境变量不跨调用保留，多条命令请用 '&&' 串联；"
        + f"默认 {config.MODAL_EXEC_TIMEOUT_SECONDS} 秒，可用 timeout 参数至多 {config.MODAL_EXEC_MAX_TIMEOUT_SECONDS} 秒。\n"
        + "3. 多文件/多步脚本 → 用 write_file 写到 /work/xxx.py → execute('python3 /work/xxx.py') 运行 → 局部微调优先用 edit_file 避免整文件重写 → 再次运行。\n"
        + "4. 读取与检索 → 用 ls / glob / grep / read_file；grep 是字面精确匹配，正则表达式搜索请用 execute('rg <regex> ...')；"
        + "PDF/Word 等富文档可用 pypdf / pdfplumber / python-docx 在沙箱里读" + read_hint + "。\n"
        + "【行为红线与禁止事项】\n"
        + "1. 严禁用沙箱做以下事：挖矿；对任何第三方的端口扫描、攻击、压力测试、暴力破解；批量注册、刷量、发送垃圾信息；"
        + "绕过他人的访问限制或抓取受保护的内容；编写或投放恶意软件。遇到这类要求直接拒绝，并如实汇报“该任务不在允许范围内”。\n"
        + "2. 严禁探测沙箱之外的内容，或打听云平台的账号、令牌、配置。\n"
        + "3. 严禁对根目录 '/' 盲目递归搜索（必须先用 ls 缩小范围，限定子目录与文件类型）。\n"
        + "4. 不要后台挂起长时间进程或开启服务端口（沙箱不对外开放端口）；命令必须在时限内结束。\n"
        + "5. 纯计算任务无需读取技能；绘图任务必须先 read_file('/skills/matplotlib-chart/SKILL.md', limit=1000)；"
        + "数据分析任务必须先 read_file('/skills/data-analysis/SKILL.md', limit=1000)。\n"
        + "【汇报格式】汇报遵循工作专家标准格式，以“【内部专家汇报 - 来源：CodeAgent】”开头；不向用户暴露沙箱内部临时路径。"
    )


MEMBER_DESCRIPTION = build_member_description()
MEMBER_PROMPT = build_member_prompt()


@dataclass(frozen=True)
class MemberCodeAgentBuild:
    """member 的云端 CodeAgent 的构建物：编排层的启动期断言会逐项检查它（工具集、后端、路由）。"""
    runnable: Any
    backend: Any                    # 构建时用的 CompositeBackend
    tool_names: frozenset           # {execute_python_code} ∪ SANDBOX_FS_TOOLS，恰为 8 个，不含 delete


def build_member_code_agent_runnable(llm, system_prompt: str, checkpointer=None, max_tool_calls=None) -> MemberCodeAgentBuild:
    """member 的云端 CodeAgent：结构与 build_code_agent_runnable 一致，只改四处——
    ① 沙箱用 ModalRoutedSandbox（按请求上下文路由到这个用户这个会话自己的云端沙箱），不接受外部传入的后端；
    ② /skills/ 路由到只读的 skills_member/；③ 工具是云端版 execute_python_code；④ execute 的超时上限用 MODAL_EXEC_MAX_TIMEOUT_SECONDS。
    绝不调用、也不引用 owner 那个构建函数与本机沙箱。"""
    if max_tool_calls is None:
        max_tool_calls = tool_limit_for(NAME)

    sb = modal_sandbox.get_routed_sandbox()
    backend = CompositeBackend(
        default=sb,
        routes={
            SKILLS_ROUTE: ReadOnlyFilesystemBackend(root_dir=SKILLS_MEMBER_DIR, virtual_mode=True),
            "/_artifacts/": StateBackend(),
        },
        artifacts_root="/_artifacts",
    )
    tools = code_tools.build_member_tools()
    runnable = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            SandboxOutputsMiddleware(sb),
            SkillsMiddleware(backend=backend, sources=[SKILLS_ROUTE]),
            FilesystemMiddleware(
                backend=backend,
                tools=SANDBOX_FS_TOOLS,
                max_execute_timeout=config.MODAL_EXEC_MAX_TIMEOUT_SECONDS,
            ),
            *build_budget_middleware(max_tool_calls),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        ],
        checkpointer=checkpointer,
    )
    return MemberCodeAgentBuild(
        runnable=runnable,
        backend=backend,
        tool_names=frozenset({t.name for t in tools} | set(SANDBOX_FS_TOOLS)),
    )
