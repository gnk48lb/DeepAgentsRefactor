"""MCP 客户端与生命周期管理服务 (L1)




"""

import json
import os
import re
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import anyio
from mcp.shared.exceptions import McpError
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from app import config

# ── 常量 ────────────────────────────────────────────────────────────────────

# mcp_servers.json 所在路径
_CONFIG_PATH = Path(config.PROJECT_ROOT) / "mcp_servers.json"

# ── 全局单例与常驻 Runner ─────────────────────────────────────────────────────
_mcp_client: Optional[MultiServerMCPClient] = None
# 存储所有工具；每个工具对象均携带 __mcp_server__ 属性
_all_mcp_tools: Optional[List[BaseTool]] = None
# 服务名称 -> 专属后台 Runner
_runners: Dict[str, "_ServerRunner"] = {}


# ── 配置加载与占位符替换 ──────────────────────────────────────────────────────

def _resolve_placeholder(value: str) -> str:
    """把字符串中的 ${VAR_NAME} 占位符替换为真实值。

    查找顺序：config 模块属性 → os.environ → 原样保留（打印警告）。
    """
    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        cfg_val = getattr(config, var_name, None)
        if cfg_val is not None:
            return str(cfg_val)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        print(
            f"  ⚠️  警告：占位符 '${{{var_name}}}' 未在 config 或环境变量中找到，已保留原样。",
            file=sys.stderr,
        )
        return match.group(0)

    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def _resolve_config_node(node):
    """递归遍历 JSON 结构，对所有字符串值执行占位符替换。"""
    if isinstance(node, str):
        return _resolve_placeholder(node)
    if isinstance(node, list):
        return [_resolve_config_node(item) for item in node]
    if isinstance(node, dict):
        return {key: _resolve_config_node(val) for key, val in node.items()}
    return node  # int / bool / None 直接返回


def _load_mcp_config() -> dict:
    """读取 mcp_servers.json，完成变量替换，做基础校验后返回。"""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"找不到 MCP 配置文件：{_CONFIG_PATH}\n"
            f"请确保项目根目录中存在 mcp_servers.json。"
        )

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw: dict = json.load(f)

    resolved = _resolve_config_node(raw)
    return resolved




# ── 常驻会话 Runner 与调用路由 ──────────────────────────────────────────────────

def _format_exception(e: BaseException) -> str:
    """递归展开 ExceptionGroup 的子异常，生成直观清晰的错误提示。"""
    if isinstance(e, BaseExceptionGroup):
        sub_msgs = [_format_exception(sub) for sub in e.exceptions]
        return f"{type(e).__name__}: {e} [\n    " + "\n    ".join(sub_msgs) + "\n  ]"
    return f"{type(e).__name__}: {e}"


class _ServerRunner:
    """每个 MCP 服务一个专属后台 task：会话的进入与退出都在同一个 task 内完成。"""

    def __init__(self, name: str, client: MultiServerMCPClient):
        self.name = name
        self.client = client
        self.tools: Dict[str, BaseTool] = {}  # 工具名 -> 绑定在当前活会话上的原始工具
        self.ready = asyncio.Event()
        self.last_error: Optional[BaseException] = None
        self.generation = 0                   # 每次(重新)连接成功 +1
        self._cmd: asyncio.Queue[str] = asyncio.Queue()  # "restart" / "stop"
        self._restart_lock = asyncio.Lock()
        self.task: Optional[asyncio.Task] = None

    def start(self):
        self.task = asyncio.create_task(self._run(), name=f"mcp-{self.name}")

    async def _run(self):
        while True:
            self.ready.clear()
            cmd = None
            try:
                async with self.client.session(self.name) as session:
                    raw_tools = await load_mcp_tools(session)
                    self.tools = {t.name: t for t in raw_tools}
                    self.last_error = None
                    self.generation += 1
                    self.ready.set()
                    cmd = await self._cmd.get()      # 阻塞到收到 restart / stop
            except asyncio.CancelledError:
                raise
            except BaseException as e:               # 连接 / 初始化 / 运行时异常捕获
                self.tools = {}
                self.last_error = e
                self.ready.set()
                cmd = await self._cmd.get()

            if cmd == "stop":
                return

    async def restart(self, seen_generation: int, timeout: float = 60.0) -> bool:
        async with self._restart_lock:
            if self.generation != seen_generation and self.ready.is_set() and self.tools:
                return True                          # 别的调用方已抢先完成重连
            if self.task is None or self.task.done():
                return False
            self.ready.clear()
            await self._cmd.put("restart")
            await asyncio.wait_for(self.ready.wait(), timeout)
            return bool(self.tools)

    async def stop(self, timeout: float = 10.0):
        if self.task and not self.task.done():
            await self._cmd.put("stop")
            try:
                await asyncio.wait_for(self.task, timeout)
            except asyncio.TimeoutError:
                self.task.cancel()
                try:
                    await self.task
                except BaseException:
                    pass


async def _call_raw_tool(server_name: str, tool_name: str, tool_input: dict):
    """从活会话 Runner 中取当前代原始工具执行，并在卡死或连接断开时自动恢复。"""
    runner = _runners.get(server_name)
    if runner is None:
        raise RuntimeError(f"未找到 MCP 服务 '{server_name}' 的 Runner")

    timeout = getattr(config, "MCP_CALL_TIMEOUT_SECONDS_BY_SERVER", {}).get(
        server_name, config.MCP_CALL_TIMEOUT_SECONDS
    )
    for attempt in (1, 2):
        gen = runner.generation
        raw = runner.tools.get(tool_name)
        if raw is None:
            err_detail = _format_exception(runner.last_error) if runner.last_error else "服务未就绪"
            raise RuntimeError(f"MCP 服务 '{server_name}' 当前不可用: {err_detail}")
        try:
            return await asyncio.wait_for(raw.ainvoke(tool_input), timeout)
        except asyncio.TimeoutError:
            await runner.restart(gen)        # 回收卡死的进程；本次调用不重试（可能已有副作用）
            raise RuntimeError(
                f"MCP 工具 '{tool_name}' 超过 {timeout}s 未返回，已重启 {server_name} 服务"
            )
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            if attempt == 2:
                raise
            await runner.restart(gen)        # 请求根本没发出去：重连后重试一次
        except McpError as e:
            if "closed" in str(e).lower():
                await runner.restart(gen)    # 在途请求遇到连接断开：回收，但不重试
            raise


# ── 工具包装 ──────────────────────────────────────────────────────────────────

def _clean_schema(schema: dict) -> dict:
    """递归删除 Schema 中的 $schema 键，防止 Google Gemini 报错/报警。"""
    if not isinstance(schema, dict):
        return schema
    new_schema = {k: v for k, v in schema.items() if k != "$schema"}
    for k, v in new_schema.items():
        if isinstance(v, dict):
            new_schema[k] = _clean_schema(v)
        elif isinstance(v, list):
            new_schema[k] = [_clean_schema(i) if isinstance(i, dict) else i for i in v]
    return new_schema


def _wrap_tool(t: BaseTool, server_name: str) -> BaseTool:
    """包装单个 MCP 工具，注入来源标签和安全/限制逻辑。

    Args:
        t: 原始 MCP 工具对象。
        server_name: 该工具所属的 MCP 服务名称（如 'desktop'、'markitdown'）。

    Returns:
        包装后的 StructuredTool，携带 ``__mcp_server__`` 属性。
    """
    enhanced_description = t.description

    # ── 清洗 Schema ──────────────────────────────────────────────────────────
    # 官方 MCP 服务（如 filesystem）常在 schema 中带 $schema，Gemini 不支持此键
    cleaned_args_schema = None
    if hasattr(t, "args_schema") and t.args_schema:
        # 如果是 Pydantic 类，转为 dict 再清洗；如果是 dict 则直接清洗
        schema_dict = (
            t.args_schema.model_json_schema()
            if hasattr(t.args_schema, "model_json_schema")
            else t.args_schema
        )
        cleaned_args_schema = _clean_schema(schema_dict)

    async def _coro(*args, **kwargs):
        try:
            tool_input = args[0] if args else kwargs
            res = await _call_raw_tool(server_name, t.name, tool_input)

            # ── 列表结果展开为文本 ────────────────────────────────────────────
            if isinstance(res, list):
                res = "\n".join(
                    item.get("text", str(item)) if isinstance(item, dict)
                    else getattr(item, "text", str(item))
                    for item in res
                )
            else:
                res = str(res)

            return res

        except asyncio.CancelledError:
            raise
        except Exception as e:
            return f"工具 '{t.name}' 执行出错: {e}"

    # 构建包装后的工具
    wrapped = StructuredTool(
        name=t.name,
        description=enhanced_description,
        args_schema=cleaned_args_schema,
        coroutine=_coro,
        func=t.invoke if hasattr(t, "invoke") else None,
    )

    # ★ 核心：注入来源服务标签，供下游分流使用
    wrapped.__mcp_server__ = server_name

    return wrapped


# ── 初始化与销毁入口 ─────────────────────────────────────────────────────────

async def initialize_mcp() -> None:
    """初始化所有 MCP 服务并缓存已打标的工具列表。幂等：重复调用仅执行一次。"""
    global _mcp_client, _all_mcp_tools, _runners

    if _all_mcp_tools is not None:
        print("  MCP 已初始化，跳过。")
        return

    print(f"  - 加载 MCP 配置文件：{_CONFIG_PATH}")
    mcp_config = _load_mcp_config()

    server_names = list(mcp_config.keys())
    print(f"  - 发现 {len(server_names)} 个 MCP 服务：{', '.join(server_names)}")

    _mcp_client = MultiServerMCPClient(mcp_config)

    print("  - 正在连接 MCP 服务器并拉取工具列表...")

    # 为所有服务启动专属后台 runner
    _runners = {}
    for sname in server_names:
        runner = _ServerRunner(sname, _mcp_client)
        _runners[sname] = runner
        runner.start()

    # 并发等待就绪
    async def _wait_runner(r: _ServerRunner):
        try:
            await asyncio.wait_for(r.ready.wait(), timeout=config.MCP_STARTUP_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            r.last_error = TimeoutError(f"启动连接超时（超过 {config.MCP_STARTUP_TIMEOUT_SECONDS} 秒）")
            await r.stop()

    await asyncio.gather(*(_wait_runner(r) for r in _runners.values()))

    # 对成功的服务包装第一代工具
    all_wrapped: List[BaseTool] = []
    server_tool_counts: Dict[str, int] = {}

    for sname in server_names:
        runner = _runners[sname]
        if runner.tools:
            wrapped = [_wrap_tool(t, sname) for t in runner.tools.values()]
            all_wrapped.extend(wrapped)
            server_tool_counts[sname] = len(wrapped)
            print(f"    ✓ [{sname}] 获取到 {len(wrapped)} 个工具")
        else:
            err_msg = _format_exception(runner.last_error) if runner.last_error else "未知错误"
            print(f"    ✗ [{sname}] 工具拉取失败：{err_msg}", file=sys.stderr)
            server_tool_counts[sname] = 0

    if not all_wrapped:
        await shutdown_mcp()
        raise RuntimeError("未能从任何 MCP 服务器获取工具，请检查各服务是否正常启动。")

    _all_mcp_tools = all_wrapped

    # ── 汇总打印 ──────────────────────────────────────────────────────────────
    print(f"✅ MCP 工具初始化完成，共获取 {len(_all_mcp_tools)} 个工具：")
    for sname in server_names:
        tools_of_server = [t for t in _all_mcp_tools if getattr(t, "__mcp_server__", "") == sname]
        print(f"  [{sname}] ({len(tools_of_server)} 个)：{', '.join(t.name for t in tools_of_server)}")


async def shutdown_mcp() -> None:
    """幂等关闭所有 MCP 服务并清空模块级状态。"""
    global _mcp_client, _all_mcp_tools, _runners
    if not _runners and _all_mcp_tools is None:
        return

    print("🛑 正在关闭所有 MCP 服务 Runner...")
    runners_to_stop = list(_runners.values())
    if runners_to_stop:
        await asyncio.gather(*(r.stop() for r in runners_to_stop), return_exceptions=True)

    _runners = {}
    _all_mcp_tools = None
    _mcp_client = None
    print("✅ 所有 MCP 服务已安全关闭。")


# ── 公开 API ──────────────────────────────────────────────────────────────────

def get_all_mcp_tools() -> List[BaseTool]:
    """返回所有已加载且已打标的 MCP 工具。须在 initialize_mcp() 完成后调用。"""
    if _all_mcp_tools is None:
        return []
    return _all_mcp_tools


def get_tools_by_server(server_name: str) -> List[BaseTool]:
    """按服务名称过滤工具，利用 __mcp_server__ 属性精准分流。

    Args:
        server_name: mcp_servers.json 中的顶层 key，如 'desktop' 或 'markitdown'。

    Returns:
        属于该服务的工具列表；若未初始化则返回空列表。
    """
    if _all_mcp_tools is None:
        return []
    return [t for t in _all_mcp_tools if getattr(t, "__mcp_server__", "") == server_name]
