"""MCP 服务集成层 —— 基于 mcp_servers.json 动态加载多服务

设计要点
--------
* 逐服务拉取工具并打上 ``__mcp_server__`` 来源标签，供下游按标签路由。
* 对 filesystem 工具加两层防护：
  1. **路径安全检查**：禁止路径跳出 WORKSPACE_DIR（防止 ../../ 逃逸）。
  2. **大文件截断**：文本内容超过 FILE_CONTENT_LIMIT 字符时自动截断并提示。
* 高德坐标工具保留原有拦截逻辑（格式校验 + 描述增强）。
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

import config

# ── 常量 ────────────────────────────────────────────────────────────────────
# 高德坐标类工具关键词
_AMAP_COORD_KEYWORDS = ("around", "coordinates", "regeocode", "distance")

# 文件读取结果的最大允许字符数；超出则截断并附注
_FILE_CONTENT_LIMIT = 8000

# 文件读取类工具名称前缀/关键词集合
_FILE_READ_KEYWORDS = ("read_file", "read_text_file", "read_multiple_files")

# ★ 危险工具列表（写入/修改/移动类）——触发 Human-in-the-loop 拦截
DANGEROUS_FS_TOOLS: set[str] = {
    "write_file",
    "edit_file",
    "create_directory",
    "move_file",
}

# mcp_servers.json 所在路径
_CONFIG_PATH = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "mcp_servers.json"

# 允许 FileAgent 访问的根目录（规范化绝对路径，方便前缀比较）
_WORKSPACE_REAL = os.path.realpath(config.WORKSPACE_DIR)

# ── 全局单例 ─────────────────────────────────────────────────────────────────
_mcp_client: Optional[MultiServerMCPClient] = None
# 存储所有工具；每个工具对象均携带 __mcp_server__ 属性
_all_mcp_tools: Optional[List[BaseTool]] = None


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

    # 校验 amap 必须有有效的 API Key
    if "amap" in resolved:
        amap_key = resolved["amap"].get("env", {}).get("AMAP_MAPS_API_KEY", "")
        if not amap_key or amap_key.startswith("${"):
            raise ValueError(
                "AMAP_API_KEY 未配置或替换失败，请在 .env 中设置 AMAP_API_KEY。"
            )

    return resolved


# ── 安全与限制检查工具函数 ────────────────────────────────────────────────────

def _is_path_safe(path_str: str) -> bool:
    """判断路径是否在 WORKSPACE_DIR 范围内（防止 ../ 路径逃逸）。

    使用 os.path.realpath 解析符号链接后再做前缀比较。
    """
    try:
        real = os.path.realpath(os.path.join(_WORKSPACE_REAL, path_str))
    except Exception:
        return False
    return real.startswith(_WORKSPACE_REAL)


def _check_filesystem_args(tool_name: str, tool_input: dict) -> Optional[str]:
    """对 filesystem 工具的入参做路径安全检查。

    如果发现路径试图逃逸出 WORKSPACE_DIR，返回错误消息字符串；
    否则返回 None 表示检查通过。
    """
    # 可能包含路径的参数名
    path_fields = ("path", "paths", "source", "destination", "directory", "pattern")

    for field in path_fields:
        val = tool_input.get(field)
        if val is None:
            continue
        # 支持单个字符串或字符串列表
        candidates = val if isinstance(val, list) else [val]
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            if not _is_path_safe(candidate):
                return (
                    f"⛔ 工具 '{tool_name}' 调用被拒绝：路径 '{candidate}' "
                    f"超出了允许的工作区范围（{_WORKSPACE_REAL}）。"
                    f"禁止使用 '../' 或绝对路径跳出工作区！"
                )
    return None


def _truncate_if_needed(text: str, tool_name: str) -> str:
    """如果文本超过 _FILE_CONTENT_LIMIT，截断并追加提示。"""
    if len(text) > _FILE_CONTENT_LIMIT:
        truncated = text[:_FILE_CONTENT_LIMIT]
        return (
            f"{truncated}\n\n"
            f"... [⚠️ 文件内容过长，已截断至 {_FILE_CONTENT_LIMIT} 字符。"
            f"原始长度约 {len(text)} 字符。如需查看后续内容，请缩小范围后重新查询。]"
        )
    return text


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
        server_name: 该工具所属的 MCP 服务名称（如 'amap'、'filesystem'）。

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

    # ── Amap 坐标工具增强 ────────────────────────────────────────────────────

    # ── Amap 坐标工具增强 ────────────────────────────────────────────────────
    is_coord_tool = (server_name == "amap") and any(
        kw in t.name for kw in _AMAP_COORD_KEYWORDS
    )
    if is_coord_tool:
        enhanced_description += (
            "\n\n【重要说明】该工具要求传入精确的经纬度坐标（格式：'经度,纬度'）。"
            "如果你只有文字地址（如 '文化东路42号'），你必须先调用 maps_geo 获取坐标，"
            "严禁根据自身知识库猜测、编造或幻觉经纬度数字。"
        )

    # ── Filesystem 工具：说明可访问范围 ──────────────────────────────────────
    is_fs_tool = (server_name == "filesystem")
    is_read_tool = is_fs_tool and any(kw in t.name for kw in _FILE_READ_KEYWORDS)
    if is_fs_tool:
        enhanced_description += (
            f"\n\n【安全限制】此工具只能在项目工作区内操作：{_WORKSPACE_REAL}。"
            "严禁使用 '../' 或绝对路径尝试访问工作区以外的文件。"
        )

    async def _coro(*args, **kwargs):
        try:
            tool_input = args[0] if args else kwargs

            # ── Amap 坐标格式运行时校验 ──────────────────────────────────────
            if is_coord_tool:
                for field in ("location", "origin", "destination", "origins"):
                    if field in tool_input and isinstance(tool_input[field], str):
                        val = tool_input[field]
                        if not re.match(r"^[0-9.]+,[0-9.]+$", val.strip()):
                            return (
                                f"工具 '{t.name}' 调用失败：参数 '{field}' 格式错误。"
                                f"该字段必须是 '经度,纬度' 格式的数字字符串（如 '117.05,36.65'）。"
                                f"你直接传入了地址文本 '{val}'。"
                                f"请先调用 'maps_geo' 解析该地址获取坐标后再试。"
                            )

            # ── Filesystem 路径安全检查 ───────────────────────────────────────
            if is_fs_tool:
                err = _check_filesystem_args(t.name, tool_input)
                if err:
                    return err

            # ── 调用原始工具 ──────────────────────────────────────────────────
            res = await t.ainvoke(tool_input)

            # ── 列表结果展开为文本 ────────────────────────────────────────────
            if isinstance(res, list):
                res = "\n".join(
                    item.get("text", str(item)) if isinstance(item, dict)
                    else getattr(item, "text", str(item))
                    for item in res
                )
            else:
                res = str(res)

            # ── 大文件截断（仅对文件读取类工具生效） ─────────────────────────
            if is_read_tool:
                res = _truncate_if_needed(res, t.name)

            return res

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


# ── 初始化入口 ────────────────────────────────────────────────────────────────

async def initialize_mcp() -> None:
    """初始化所有 MCP 服务并缓存已打标的工具列表。幂等：重复调用仅执行一次。"""
    global _mcp_client, _all_mcp_tools

    if _mcp_client is not None:
        print("  MCP 已初始化，跳过。")
        return

    print(f"  - 加载 MCP 配置文件：{_CONFIG_PATH}")
    mcp_config = _load_mcp_config()

    server_names = list(mcp_config.keys())
    print(f"  - 发现 {len(server_names)} 个 MCP 服务：{', '.join(server_names)}")

    _mcp_client = MultiServerMCPClient(mcp_config)

    print("  - 正在连接 MCP 服务器并拉取工具列表...")

    # ── 逐服务拉取工具并打标 ──────────────────────────────────────────────────
    # MultiServerMCPClient 的 get_tools() 返回所有服务工具的混合列表，
    # 且通过 server_name= 参数可以只拉某个服务的工具，实现精准溯源。
    all_wrapped: List[BaseTool] = []
    server_tool_counts: Dict[str, int] = {}

    for sname in server_names:
        try:
            # 只拉取该服务的工具
            raw_tools: List[BaseTool] = await _mcp_client.get_tools(server_name=sname)
            wrapped = [_wrap_tool(t, sname) for t in raw_tools]
            all_wrapped.extend(wrapped)
            server_tool_counts[sname] = len(wrapped)
            print(f"    ✓ [{sname}] 获取到 {len(wrapped)} 个工具")
        except Exception as e:
            print(f"    ✗ [{sname}] 工具拉取失败：{e}", file=sys.stderr)
            server_tool_counts[sname] = 0

    if not all_wrapped:
        raise RuntimeError("未能从任何 MCP 服务器获取工具，请检查各服务是否正常启动。")

    _all_mcp_tools = all_wrapped

    # ── 汇总打印 ──────────────────────────────────────────────────────────────
    print(f"✅ MCP 工具初始化完成，共获取 {len(_all_mcp_tools)} 个工具：")
    for sname in server_names:
        tools_of_server = [t for t in _all_mcp_tools if getattr(t, "__mcp_server__", "") == sname]
        print(f"  [{sname}] ({len(tools_of_server)} 个)：{', '.join(t.name for t in tools_of_server)}")


# ── 公开 API ──────────────────────────────────────────────────────────────────

def get_all_mcp_tools() -> List[BaseTool]:
    """返回所有已加载且已打标的 MCP 工具。须在 initialize_mcp() 完成后调用。"""
    if _all_mcp_tools is None:
        return []
    return _all_mcp_tools


def get_tools_by_server(server_name: str) -> List[BaseTool]:
    """按服务名称过滤工具，利用 __mcp_server__ 属性精准分流。

    Args:
        server_name: mcp_servers.json 中的顶层 key，如 'amap' 或 'filesystem'。

    Returns:
        属于该服务的工具列表；若未初始化则返回空列表。
    """
    if _all_mcp_tools is None:
        return []
    return [t for t in _all_mcp_tools if getattr(t, "__mcp_server__", "") == server_name]


# 向后兼容别名
get_amap_tools = get_all_mcp_tools
