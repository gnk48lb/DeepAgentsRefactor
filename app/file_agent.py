"""
app/file_agent.py

FileAgent：deepagents 内置 FilesystemMiddleware + FilesystemBackend，加 langchain 原生
HumanInTheLoopMiddleware（approve / edit / reject 协议）。

注意：这里故意用 create_agent 而不是 create_deep_agent 构建，这样 deep_agent.py 里全局
HarnessProfile.excluded_tools（隐藏 write_file/edit_file/delete/execute）不会作用到 FileAgent，
也不用改动主 Agent 与其他子 Agent 的工具可见性。
本模块不 import app.models / app.graph，方便离线烟测（不需要任何 API Key）。
"""
import os
from pathlib import Path

from deepagents import FilesystemMiddleware
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool

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
