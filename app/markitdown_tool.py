"""
app/markitdown_tool.py

为 FileAgent 包装 markitdown-mcp 的 convert_to_markdown 工具：
完成 /workspace 虚拟路径 -> 宿主机真实路径 file:/// URI 转换，并实施严格的工作区边界与协议校验。
"""
import asyncio
import ipaddress
import os
import re
from pathlib import Path
from urllib.parse import urlparse, unquote

from langchain_core.tools import BaseTool, StructuredTool

from app import config


async def normalize_uri(
    raw_uri: str,
    workspace_dir: str | Path | None = None,
    allow_http: bool | None = None,
    max_mb: int | None = None,
) -> str:
    """校验输入路径或 URI，返回可安全传给宿主机 markitdown-mcp 的标准 URI；不合规抛 ValueError。"""
    root = Path(workspace_dir or config.WORKSPACE_DIR).resolve()
    if allow_http is None:
        allow_http = getattr(config, "MARKITDOWN_ALLOW_HTTP", False)
    if max_mb is None:
        max_mb = getattr(config, "MARKITDOWN_MAX_FILE_MB", 50)

    s = (raw_uri or "").strip()
    if not s:
        raise ValueError("参数 uri 不能为空")

    # 判断是否带有 URI scheme（注意排除 Windows 盘符路径如 C:\xxx 或 D:/xxx）
    has_scheme = "://" in s or (
        bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", s))
        and not bool(re.match(r"^[a-zA-Z]:[/\\]", s))
    )

    if has_scheme:
        u = urlparse(s)
        scheme = u.scheme.lower()
        if scheme in ("http", "https"):
            if not allow_http:
                raise ValueError(
                    "当前环境已禁用网络 URL 转换，只能转换工作区内的本地文件；网页内容请交给 BrowserAgent。"
                )
            if u.username or u.password:
                raise ValueError("不允许 URL 中包含账号密码")
            host = u.hostname
            if not host:
                raise ValueError("URL 缺少主机名")
            port = u.port or (443 if scheme == "https" else 80)
            loop = asyncio.get_running_loop()
            try:
                infos = await loop.getaddrinfo(host, port)
            except OSError as e:
                raise ValueError(f"无法解析主机名 {host}: {e}")
            for info in infos:
                ip = ipaddress.ip_address(info[4][0])
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                    or ip.is_multicast
                    or ip.is_unspecified
                ):
                    raise ValueError(f"不允许访问内网、本机或保留地址: {host}")
            return s

        if scheme != "file":
            raise ValueError(f"不允许的 scheme: {scheme!r}")
        if u.netloc not in ("", "localhost"):
            raise ValueError("不允许带远程主机的 file URI")

        p = unquote(u.path)
        # Windows 下 file:///D:/xxx 的 u.path 为 /D:/xxx，去掉前导斜杠
        if re.match(r"^/[A-Za-z]:[/\\]", p):
            p = p[1:]
    else:
        p = s

    norm = p.replace("\\", "/").strip()
    low = norm.lower()

    # Windows 上对 UNC（\\host\share）/ 设备路径（\\?\、\\.\）做 resolve() 会尝试连接对应主机（可能带上 NTLM 凭据），必须在 resolve 之前拒绝
    if norm.startswith("//"):
        raise ValueError("不允许 UNC / 设备路径")

    if low in ("workspace", "/workspace"):
        raise ValueError("不能转换工作区根目录，请指定具体文件路径")

    # 1. /workspace/xxx 或 workspace/xxx 虚拟路径 -> 映射到宿主机 root 下
    if low.startswith("/workspace/"):
        rel_str = norm[len("/workspace/"):].lstrip("/")
        candidate = root / rel_str
    elif low.startswith("workspace/"):
        rel_str = norm[len("workspace/"):].lstrip("/")
        candidate = root / rel_str
    # 2. Windows 盘符路径（如 C:/Windows/win.ini 或 D:/.../workspace/a.pdf）
    elif re.match(r"^[A-Za-z]:", norm):
        if os.name != "nt":
            raise ValueError("路径不在工作区内")
        candidate = Path(norm)
    # 3. 以 / 开头的其他绝对路径（如 /etc/passwd）
    elif norm.startswith("/"):
        candidate = Path(norm)
    # 4. 相对工作区路径（如 报告/a.pdf）
    else:
        candidate = root / norm

    try:
        real = candidate.resolve()
        in_workspace = (real != root) and real.is_relative_to(root)
    except ValueError:
        in_workspace = False

    if not in_workspace:
        raise ValueError("路径不在工作区内")
    if not real.is_file():
        raise ValueError(f"文件不存在: {raw_uri}")
    if max_mb and real.stat().st_size > max_mb * 1024 * 1024:
        raise ValueError(f"文件超过大小上限 ({max_mb} MB)")

    return real.as_uri()


def build_tools(mcp_tools: list[BaseTool], workspace_dir: str | Path) -> list[BaseTool]:
    """从 markitdown MCP 工具列表中提取并包装 convert_to_markdown。"""
    inner = {t.name: t for t in (mcp_tools or [])}.get("convert_to_markdown")
    if inner is None:
        return []

    async def _run(uri: str) -> str:
        try:
            mapped = await normalize_uri(uri, workspace_dir=workspace_dir)
        except ValueError as e:
            return f"convert_to_markdown 调用被拒绝：{e}"
        return await inner.ainvoke({"uri": mapped})

    return [
        StructuredTool(
            name=inner.name,
            description=inner.description,
            args_schema=inner.args_schema,  # 复用上游 schema，不手写
            coroutine=_run,
        )
    ]
