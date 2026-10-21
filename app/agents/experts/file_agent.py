"""本地工作区文件操作专家 (所在层: L4)"""

import os
from pathlib import Path

from deepagents import FilesystemMiddleware
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from app.agents.prompts import WORKER_BASE_PROMPT

NAME = "FileAgent"

DESCRIPTION = (
    "本地文件专家，是唯一能访问本地磁盘（workspace 工作区）的专家：列目录、搜索、读取文本类文件，"
    "支持用 convert_to_markdown 读取 PDF、Word、PPT、Excel、HTML、EPub、CSV、ZIP 等文档，"
    "写入、编辑、新建目录、移动或重命名文件（“删除”一律改为移动到 archive_trash 归档）。"
    "写入/编辑/新建目录/移动会暂停并请求人工授权。"
    "不要让它读取纯图片、音视频等媒体文件；"
    "网页上传文件的任务不要先派它读取，直接把路径交给 BrowserAgent。"
)

PROMPT = (
    WORKER_BASE_PROMPT
    + "你是 FileAgent，负责操作本地工作区（workspace）里的文件。\n"
    + "【工具】浏览/读取：ls、glob、grep、read_file；写入类（会暂停等待用户授权）：write_file、edit_file、create_directory、move_file。\n"
    + "【路径规范 - 强制】\n"
    + "  · 所有路径一律写成 /workspace/xxx（也接受 workspace/xxx）。用户或主管写的 \"workspace/xxx\" 指的就是工作区里的 xxx，原样传给工具，不要去掉前缀。\n"
    + "  · 只能在工作区内操作：不要使用 ../、Windows 盘符路径（如 C:\\...）或其他系统绝对路径。路径含反斜杠时先改成正斜杠。\n"
    + "  · ls / glob / grep 必须显式传 path，从 /workspace（或其子目录）开始；不要对 \"/\" 列目录或搜索。\n"
    + "【效率原则】\n"
    + "  · 先用 ls 或 glob(pattern=\"**/*\", path=\"/workspace\") 了解目录全貌，再按需 read_file。\n"
    + "  · 大文件用 read_file 的 offset / limit 分页读取（每次不超过 200 行），不要一次读完整个大文件；找内容优先用 grep。\n"
    + "  · 纯文本与代码文件用 read_file（严禁对二进制或富文档调用 read_file）；遇到图片、音视频等媒体文件直接汇报无法读取。\n"
    + "【写入规范】\n"
    + "  · write_file 会整体覆盖已存在的文件，并自动创建父目录；覆盖前先 read_file 确认现有内容。局部修改优先 edit_file。\n"
    + "  · 没有删除工具。主管要求“删除”文件时，用 move_file 把它移动到 /workspace/archive_trash/ 下（目标同名已存在时，在文件名后加序号或时间戳）。\n"
    + "  · 需要新建空文件夹时用 create_directory。\n"
    + "【授权被拒】写入类操作会暂停等待用户授权。如果被用户拒绝（收到“用户明确拒绝/权限被拒绝”的提示），你必须立即停止对该操作及类似操作的尝试，严禁循环请求；礼貌地向主管汇报该步骤已被用户取消，并根据情况继续其他安全步骤或直接结束任务。\n"
    + "【汇报格式】最终回答以“【内部专家汇报 - 来源：FileAgent】”开头，然后给出工具实际返回的结果（路径、文件内容、执行结果等）；没有拿到工具结果时如实说明，禁止把计划当成结果。\n"
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

# 内置文件工具白名单。故意不含 delete / execute（“删除 = 归档”：move_file 到 archive_trash）。
FS_TOOLS = ["ls", "read_file", "write_file", "edit_file", "glob", "grep"]

# 需要人工授权的工具，与旧版 DANGEROUS_FS_TOOLS 保持一致
FILE_AGENT_INTERRUPT_ON = {
    "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
    "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
    "create_directory": {"allowed_decisions": ["approve", "reject"]},
    "move_file": {"allowed_decisions": ["approve", "reject"]},
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
    """内置文件工具没有的两个能力：move_file（移动/重命名）与 create_directory。"""
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

    return [move_file, create_directory]


def build_file_agent_runnable(llm, system_prompt: str, workspace_dir: str, checkpointer=None, extra_tools=None):
    """返回可直接放进 CompiledSubAgent(runnable=...) 的已编译图。
    checkpointer 保持 None：由主图的 checkpointer 统一接管（与旧子图一致）。"""
    tools = make_workspace_tools(workspace_dir) + list(extra_tools or [])
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            FilesystemMiddleware(backend=build_workspace_backend(workspace_dir), tools=FS_TOOLS),
            HumanInTheLoopMiddleware(interrupt_on=FILE_AGENT_INTERRUPT_ON),
        ],
        checkpointer=checkpointer,
    )
