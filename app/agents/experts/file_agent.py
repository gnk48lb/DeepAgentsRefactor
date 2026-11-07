"""本地工作区文件操作与宿主机命令专家 (所在层: L4)

FileAgent：deepagents 内置 FilesystemMiddleware + FilesystemBackend，加 langchain 原生
HumanInTheLoopMiddleware（approve / edit / reject 协议）。
另有自定义工具 run_host_command：在用户 Windows 宿主机上执行命令（app.infra.host_shell），
文件工具限于 workspace（virtual_mode），而该命令不受虚拟路径限制，仅以工作区根目录作为 cwd。

注意：这里故意用 create_agent 而不是 create_deep_agent 构建，这样 orchestrator 模块里全局
HarnessProfile.excluded_tools（隐藏 write_file/edit_file/delete/execute）不会作用到 FileAgent，
也不用改动主 Agent 与其他子 Agent 的工具可见性。
本模块不 import app.infra.models / app.agents.experts.desktop_agent，方便离线烟测（不需要任何 API Key）。
"""

import os
from pathlib import Path

from deepagents import FilesystemMiddleware
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from app import config
from app.agents.prompts import WORKER_BASE_PROMPT
from app.agents.budget import build_budget_middleware, tool_limit_for
from app.infra import host_shell

NAME = "FileAgent"

DESCRIPTION = (
    "本地文件与宿主机终端专家，是唯一能修改本地磁盘（workspace）、也是唯一能在用户 Windows 宿主机上执行命令的专家。"
    "文件方面：列目录、搜索、读取文本类文件，用 convert_to_markdown 读取 PDF、Word、PPT、Excel、HTML、EPub、CSV、ZIP 等文档，"
    "写入、编辑、新建目录、移动或重命名文件（“删除”一律改为移动到 archive_trash 归档）。"
    "终端方面：通过 run_host_command 在宿主机运行命令行（git、pip/npm 安装、联网下载、构建/测试脚本、运行 workspace 里的程序、"
    "本机进程/端口/环境检查），有网络、能用本机已安装的工具。"
    "写入/编辑/新建目录/移动，以及除少数只读状态查询（如 git status）外的所有命令，都会暂停并请求人工授权。"
    "纯计算、数据分析、绘图、不需要联网也不需要宿主机环境的代码，请派给 CodeAgent（隔离沙箱、无需授权），不要派给它；"
    "不要让它读取纯图片、音视频等媒体文件；点击/操作图形界面窗口请派给 DesktopAgent；"
    "网页上传文件的任务不要先派它读取，直接把路径交给 BrowserAgent。"
)

PROMPT = (
    WORKER_BASE_PROMPT
    + "你是 FileAgent，负责操作本地工作区（workspace）里的文件，并在需要时在用户的 Windows 宿主机上执行命令。\n"
    + "【工具】浏览/读取：ls、glob、grep、read_file；写入类（会暂停等待用户授权）：write_file、edit_file、create_directory、move_file；宿主机命令：run_host_command（只有少数只读状态查询免授权，其余都会暂停等待用户授权）。\n"
    + "【路径规范 - 强制】\n"
    + "  · 以下路径规则只针对文件工具；run_host_command 里的路径规则见下面专门一节。\n"
    + "  · 所有路径一律写成 /workspace/xxx（也接受 workspace/xxx）。用户或主管写的 \"workspace/xxx\" 指的就是工作区里的 xxx，原样传给工具，不要去掉前缀。\n"
    + "  · 只能在工作区内操作：不要使用 ../、Windows 盘符路径（如 C:\\...）或其他系统绝对路径。路径含反斜杠时先改成正斜杠。\n"
    + "  · ls / glob / grep 必须显式传 path，从 /workspace（或其子目录）开始；不要对 \"/\" 列目录或搜索。\n"
    + "【宿主机命令（run_host_command）- 强制】\n"
    + "  · 它直接运行在用户的 Windows 宿主机上（cmd.exe），没有任何沙箱隔离，权限等同于运行本系统的用户；只有少数只读状态查询（如 git status）免授权，其余命令都会暂停等待用户授权。\n"
    + "  · 工作目录固定为工作区根目录。命令里引用工作区文件时写相对路径（如 src\\a.py），不要写虚拟前缀 /workspace/（那只是文件工具的虚拟路径，Windows 磁盘上并不存在）。\n"
    + "  · 运行工作区里的脚本或程序要写成 .\\run.bat 这样带 .\\ 的形式（系统不会在当前目录里自动搜索可执行文件）；python x.py、git、npm 等已安装的命令照常写。\n"
    + "  · 每次调用是独立进程，cd / set 不会跨调用保留；多步用 && 串联（cmd.exe 里 ; 不是命令分隔符，也不能换行写多条命令）。\n"
    + "  · 标准输入是空的，命令不能交互：不要运行会等待输入的程序（pause、Python REPL、不带 -m 的 git commit、需要确认的安装器——加 -y / --yes / --quiet）；不要启动不会自行结束的前台程序（服务、ping -t、tail -f）。\n"
    + f"  · 默认超时 {config.HOST_SHELL_TIMEOUT_SECONDS} 秒；安装依赖、构建等确需更久时传 timeout 参数（最大 {config.HOST_SHELL_MAX_TIMEOUT_SECONDS}）。超时会终止整个进程树。\n"
    + "  · 输出过长只保留开头和结尾；需要完整内容时把输出重定向到工作区里的文件（> out.txt），再用 read_file / grep 分页读取。\n"
    + "  · 能用文件工具完成的（列目录、读写、搜索、移动、建目录）严禁改用本工具跑 dir / type / findstr / copy / move / mkdir。\n"
    + "  · 一次只发一条 run_host_command，不要在同一轮里并行发多条会改文件或互相依赖的命令。\n"
    + "  · 运行环境里没有项目的任何密钥和 API Key，不要尝试读取或索要；除非主管任务明确要求，不要读取、修改工作区以外的路径。\n"
    + "【效率原则】\n"
    + "  · 先用 ls 或 glob(pattern=\"**/*\", path=\"/workspace\") 了解目录全貌，再按需 read_file。\n"
    + "  · 大文件用 read_file 的 offset / limit 分页读取（每次不超过 200 行），不要一次读完整个大文件；找内容优先用 grep。\n"
    + "  · 纯文本与代码文件用 read_file（严禁对二进制或富文档调用 read_file）；遇到图片、音视频等媒体文件直接汇报无法读取。\n"
    + "【写入规范】\n"
    + "  · write_file 会整体覆盖已存在的文件，并自动创建父目录；覆盖前先 read_file 确认现有内容。局部修改优先 edit_file。\n"
    + "  · 没有删除工具。主管要求“删除”文件时，用 move_file 把它移动到 /workspace/archive_trash/ 下（目标同名已存在时，在文件名后加序号或时间戳）；除非主管明确写了“永久删除”，否则严禁用 run_host_command 执行 del / rd / rmdir。\n"
    + "  · 需要新建空文件夹时用 create_directory。\n"
    + "【授权被拒】写入类操作与 run_host_command 都会暂停等待用户授权。如果被用户拒绝（收到“用户明确拒绝/权限被拒绝”的提示），你必须立即停止对该操作及类似操作的尝试，严禁循环请求；礼貌地向主管汇报该步骤已被用户取消，并根据情况继续其他安全步骤或直接结束任务。\n"
    + "【汇报格式】最终回答以“【内部专家汇报 - 来源：FileAgent】”开头，然后给出工具实际返回的结果（路径、文件内容、执行结果等）；没有拿到工具结果时如实说明，禁止把计划当成结果。run_host_command 的汇报必须包含实际执行的命令原文、退出码，以及决定结论的关键输出行（出错时给出报错关键行）；命令被拒绝或超时要如实说明。\n"
)

PROMPT_MARKITDOWN = (
    "【文档转换】PDF、Word、PowerPoint、Excel、HTML、EPub、CSV、ZIP 等需要转成 Markdown 才能阅读的文件，"
    "调用 convert_to_markdown；纯文本、代码文件直接用 read_file 更省 token。图片、音视频仍不要读取。\n"
    "【转换路径】convert_to_markdown 的 uri 直接写成 /workspace/<相对路径>（或 file:///workspace/<相对路径>），"
    "例如 /workspace/报告/a.pdf。不要传 http(s) 链接，不要传工作区外的路径。\n"
    "【转换结果过大】结果超过阈值时，系统不会直接返回正文，而是把它存到 /_artifacts/large_tool_results/<id> 并告诉你这个路径。"
    "这个路径是合法的（前面“路径一律写成 /workspace/xxx”的规则对它不适用）：原样传给 read_file"
    "（用 offset / limit 分页，每次不超过 200 行）或 grep，不要改写成 /workspace/...，也不要去掉 /_artifacts 前缀。\n"
)

FILE_AGENT_PROMPT = PROMPT
FILE_AGENT_MARKITDOWN_PROMPT = PROMPT_MARKITDOWN

# 内置文件工具白名单。故意不含 delete / execute（删除 = 归档；宿主机命令走自定义工具 run_host_command）。
FS_TOOLS = ["ls", "read_file", "write_file", "edit_file", "glob", "grep"]

def _not_yet_answered(req) -> bool:
    """HITL 的 when 谓词：该调用若已经有 ToolMessage 应答，就不再弹授权。

    超预算的调用会先被限步中间件用错误 ToolMessage 拦下（after_model 钩子按列表倒序执行：
    限步在前、HITL 在后）。不跳过的话，用户会被问一个根本不会执行的授权；
    用户点“拒绝”时，同一个 tool_call_id 还会出现两条应答。
    """
    call_id = req.tool_call["id"]
    return not any(
        isinstance(m, ToolMessage) and m.tool_call_id == call_id
        for m in req.state["messages"]
    )


def _should_interrupt_command(req) -> bool:
    """run_host_command 的 HITL 谓词：已被预算中间件拦截的不再弹授权；整条命令精确匹配免授权名单的放行；其余全部要授权。"""
    if not _not_yet_answered(req):
        return False
    command = (req.tool_call.get("args") or {}).get("command")
    return not host_shell.is_auto_approved_command(command)


# 需要人工授权的工具，与旧版 DANGEROUS_FS_TOOLS 保持一致
FILE_AGENT_INTERRUPT_ON = {
    "write_file": {"allowed_decisions": ["approve", "edit", "reject"], "when": _not_yet_answered},
    "edit_file": {"allowed_decisions": ["approve", "edit", "reject"], "when": _not_yet_answered},
    "create_directory": {"allowed_decisions": ["approve", "reject"], "when": _not_yet_answered},
    "move_file": {"allowed_decisions": ["approve", "reject"], "when": _not_yet_answered},
    "run_host_command": {"allowed_decisions": ["approve", "reject"], "when": _should_interrupt_command},
}


def build_workspace_backend(workspace_dir: str) -> CompositeBackend:
    ws = FilesystemBackend(root_dir=workspace_dir, virtual_mode=True)
    return CompositeBackend(
        # default 也指向真实工作区：不带 /workspace/ 前缀的路径同样落到磁盘，
        # 避免“写进虚拟 state 却显示成功”的静默失败
        default=ws,
        routes={
            "/workspace/": ws,
            # 框架内部文件（large_tool_results 等）走内存 state，绝不落到真实工作区
            "/_artifacts/": StateBackend(),
        },
        artifacts_root="/_artifacts",
    )


def _resolve_in_workspace(root: Path, raw: str) -> Path:
    """把 /workspace/xxx、workspace/xxx、xxx 统一解析到工作区内的真实路径；越界抛 ValueError。"""
    s = (raw or "").replace("\\", "/").strip()
    low = s.lower()
    if low in ("workspace", "/workspace"):
        s = ""
    else:
        for prefix in ("/workspace/", "workspace/"):
            if low.startswith(prefix):
                s = s[len(prefix):]
                break
    target = (root / s.lstrip("/")).resolve()
    if target != root and not target.is_relative_to(root):
        raise ValueError(f"路径超出工作区范围: {raw}")
    return target


def make_workspace_tools(workspace_dir: str) -> list:
    """内置文件工具没有的能力：move_file（移动/重命名）、create_directory 与 run_host_command（宿主机命令）。"""
    root = Path(workspace_dir).resolve()

    @tool
    def move_file(source: str, destination: str) -> str:
        """移动或重命名工作区内的文件或目录。目标已存在时失败；会自动创建目标的父目录。路径写成 /workspace/xxx。"""
        try:
            src = _resolve_in_workspace(root, source)
            dst = _resolve_in_workspace(root, destination)
        except ValueError as e:
            return f"Error: {e}"
        if src == root:
            return "Error: 不能移动工作区根目录"
        if not src.exists():
            return f"Error: 源路径不存在: {source}"
        if dst.exists():
            return f"Error: 目标已存在: {destination}"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
        except OSError as e:
            return f"Error: 移动失败: {e}"
        return f"已移动 {source} -> {destination}"

    @tool
    def create_directory(path: str) -> str:
        """在工作区内创建目录（含父目录；已存在视为成功）。路径写成 /workspace/xxx。"""
        try:
            target = _resolve_in_workspace(root, path)
        except ValueError as e:
            return f"Error: {e}"
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return f"Error: 创建目录失败: {e}"
        return f"已创建目录 {path}"

    @tool
    def run_host_command(command: str, timeout: int = 0) -> str:
        r"""在用户的 Windows 宿主机上执行一条命令行（cmd.exe），返回合并后的输出和退出码。没有沙箱隔离，权限等同于运行本系统的用户。
        工作目录固定为工作区根目录：命令里用相对路径（如 src\a.py），不要写虚拟前缀 /workspace/；运行工作区里的程序写成 .\run.bat。
        每次调用是独立进程，cd / set 不跨调用保留，多步用 && 串联，不能换行写多条命令。标准输入为空，不能交互。
        timeout 单位秒，0 表示使用默认超时（较短），确需更久再传（有上限）；超时会终止整个进程树。输出过长只保留开头和结尾。
        列目录、读写、搜索、移动文件请用 ls / read_file / write_file / edit_file / grep / glob / move_file，不要用本工具跑 dir / type / del。
        除少数只读状态查询（如 git status）外，所有命令都会先请求用户授权。"""
        if timeout < 0 or timeout > config.HOST_SHELL_MAX_TIMEOUT_SECONDS:
            return (f"Error: timeout 必须在 0~{config.HOST_SHELL_MAX_TIMEOUT_SECONDS} 秒之间"
                    f"（0 表示默认 {config.HOST_SHELL_TIMEOUT_SECONDS} 秒），收到 {timeout}")
        return host_shell.format_result(host_shell.run_command(command, cwd=root, timeout=timeout or None))

    return [move_file, create_directory, run_host_command]


def build_file_agent_runnable(llm, system_prompt: str, workspace_dir: str, checkpointer=None, extra_tools=None, max_tool_calls=None):
    """返回可直接放进 CompiledSubAgent(runnable=...) 的已编译图。
    checkpointer 保持 None：由主图的 checkpointer 统一接管（与旧子图一致）。
    max_tool_calls：单次委派内的工具调用预算，默认取 config.SUBAGENT_TOOL_LIMITS["FileAgent"]。"""
    if max_tool_calls is None:
        max_tool_calls = tool_limit_for(NAME)
    tools = make_workspace_tools(workspace_dir) + list(extra_tools or [])
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            FilesystemMiddleware(backend=build_workspace_backend(workspace_dir), tools=FS_TOOLS),
            # 顺序有讲究：create_agent 的 after_model 钩子按列表倒序执行（排在后面的先执行），
            # 所以限步中间件放在 HITL 之后 → 先拦截超预算调用，再轮到 HITL（并通过 when 跳过已被拦截的调用）。
            HumanInTheLoopMiddleware(interrupt_on=FILE_AGENT_INTERRUPT_ON),
            *build_budget_middleware(max_tool_calls),
        ],
        checkpointer=checkpointer,
    )
