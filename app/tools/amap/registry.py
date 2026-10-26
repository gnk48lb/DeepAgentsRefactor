import asyncio
import json
from pathlib import Path
from typing import Callable, List
from langchain_core.tools import BaseTool, StructuredTool

from app.tools.amap import impl

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "tools_snapshot.json"

# 只有这些工具的这些参数必须是纯 '经度,纬度'；其余位置参数一律由 impl 自动识别（坐标 / 文字地址）
_COORD_ONLY_FIELDS: dict[str, tuple[str, ...]] = {
    "maps_regeocode": ("location",),
}


def _check_coords(name: str, kwargs: dict) -> str | None:
    for field in _COORD_ONLY_FIELDS.get(name, ()):
        val = kwargs.get(field)
        if isinstance(val, str) and not impl.is_coordinates(val):
            return (
                f"工具 '{name}' 调用失败：参数 '{field}' 格式错误。"
                f"该字段必须是 '经度,纬度' 格式的数字字符串（如 '117.05,36.65'）。"
                f"你直接传入了地址文本 '{val}'。"
                f"请先调用 'maps_geo' 解析该地址获取坐标后再试。"
            )
    return None


def _make_coroutine(tool_name: str, fn: Callable):
    async def _coro(*args, **kwargs):
        tool_input = dict(kwargs)
        if args and isinstance(args[0], dict):
            tool_input.update(args[0])
        coord_err = _check_coords(tool_name, tool_input)
        if coord_err:
            return coord_err
        try:
            res = await fn(**tool_input)
            return json.dumps(res, ensure_ascii=False, indent=2)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return f"工具 '{tool_name}' 执行出错: Error executing tool {tool_name}: {e}"

    return _coro


def get_tools() -> List[BaseTool]:
    if not _SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"找不到高德工具快照文件：{_SNAPSHOT_PATH}")

    with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        snapshots = json.load(f)

    tools = []
    for item in snapshots:
        name = item["name"]
        description = item["description"]
        input_schema = item["inputSchema"]

        fn = getattr(impl, name, None)
        if fn is None:
            raise AttributeError(f"未在 app.tools.amap.impl 中找到工具实现：{name}")

        tool = StructuredTool(
            name=name,
            description=description,
            args_schema=input_schema,
            coroutine=_make_coroutine(name, fn),
        )
        tools.append(tool)

    return tools
