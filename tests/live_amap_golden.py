"""
tests/live_amap_golden.py

在线（Live）Amap Golden 对比测试：
读取 tests/fixtures/amap/golden/*，对比真实 API 返回结果与 Golden 样本的“字段集合与类型”（而非文本全等）。
默认跳过（除非传入命令行参数 --live 或设置环境变量 AMAP_LIVE_TEST=1）。
支持 --update 开关（须与 --live 同用）用真实响应更新 Golden 样本。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import app.tools.amap as amap

GOLDEN_DIR = ROOT / "tests/fixtures/amap/golden"

CASES = [
    ("maps_geo", {"address": "北京市朝阳区阜通东大街6号"}, "maps_geo.default.json"),
    ("maps_regeocode", {"location": "116.481488,39.990464"}, "maps_regeocode.default.json"),
    ("maps_ip_location", {"ip": "114.114.114.114"}, "maps_ip_location.default.json"),
    ("maps_weather", {"city": "110101"}, "maps_weather.default.json"),
    ("maps_weather", {"city": "北京市东城区"}, "maps_weather.default.json"),
    (
        "maps_route_bicycling",
        {
            "origin": "116.481488,39.990464",
            "destination": "116.486488,39.995464",
        },
        "maps_route_bicycling.coord.json",
    ),
    (
        "maps_route_bicycling",
        {
            "origin": "北京市朝阳区阜通东大街6号",
            "destination": "北京市朝阳区望京SOHO",
        },
        "maps_route_bicycling.address.json",
    ),
    (
        "maps_route_walking",
        {
            "origin": "116.481488,39.990464",
            "destination": "116.486488,39.995464",
        },
        "maps_route_walking.coord.json",
    ),
    (
        "maps_route_walking",
        {
            "origin": "北京市朝阳区阜通东大街6号",
            "destination": "北京市朝阳区望京SOHO",
        },
        "maps_route_walking.address.json",
    ),
    (
        "maps_route_driving",
        {
            "origin": "116.481488,39.990464",
            "destination": "116.486488,39.995464",
        },
        "maps_route_driving.coord.json",
    ),
    (
        "maps_route_driving",
        {
            "origin": "北京市朝阳区阜通东大街6号",
            "destination": "北京市朝阳区望京SOHO",
        },
        "maps_route_driving.address.json",
    ),
    (
        "maps_route_transit",
        {
            "origin": "116.481488,39.990464",
            "destination": "116.486488,39.995464",
            "origin_city": "北京",
        },
        "maps_route_transit.coord.json",
    ),
    (
        "maps_route_transit",
        {
            "origin": "北京市朝阳区阜通东大街6号",
            "destination": "北京市朝阳区望京SOHO",
            "origin_city": "北京",
        },
        "maps_route_transit.address.json",
    ),
    (
        "maps_distance",
        {
            "origins": "北京市朝阳区阜通东大街6号|116.481488,39.990464",
            "destination": "北京市朝阳区望京SOHO",
            "city": "北京",
        },
        "maps_distance.default.json",
    ),
    (
        "maps_distance",
        {
            "origins": "116.481488,39.990464",
            "destination": "116.486488,39.995464",
        },
        "maps_distance.default.json",
    ),
    ("maps_text_search", {"keywords": "肯德基", "city": "北京"}, "maps_text_search.default.json"),
    (
        "maps_around_search",
        {"location": "116.481488,39.990464", "keywords": "餐饮"},
        "maps_around_search.default.json",
    ),
    (
        "maps_around_search",
        {"location": "北京市朝阳区阜通东大街6号", "keywords": "餐饮"},
        "maps_around_search.default.json",
    ),
    ("maps_search_detail", {"id": "B000A85652"}, "maps_search_detail.default.json"),
]


def compare_structure(actual, expected, path=""):
    """递归对比字段集合与基本类型。"""
    if type(actual) is not type(expected):
        raise AssertionError(
            f"类型不匹配 at {path}: expected {type(expected).__name__}, got {type(actual).__name__}"
        )

    if isinstance(expected, dict):
        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())
        if actual_keys != expected_keys:
            raise AssertionError(
                f"字典键不匹配 at {path}: missing={expected_keys - actual_keys}, extra={actual_keys - expected_keys}"
            )
        for k in expected_keys:
            compare_structure(actual[k], expected[k], f"{path}.{k}" if path else k)
    elif isinstance(expected, list):
        if expected and actual:
            # 抽样对比列表第 1 个元素结构
            compare_structure(actual[0], expected[0], f"{path}[0]")


async def run_live_test(update: bool = False):
    tools_map = {t.name: t for t in amap.get_tools()}
    for tool_name, args, golden_filename in CASES:
        golden_file = GOLDEN_DIR / golden_filename
        if not golden_file.exists() and not update:
            raise FileNotFoundError(f"未找到 Golden 文件: {golden_filename}")

        if golden_file.exists():
            golden_data = json.loads(golden_file.read_text(encoding="utf-8"))
            if tool_name == "maps_search_detail" and "id" in golden_data:
                args = {"id": golden_data["id"]}

        tool = tools_map[tool_name]
        res_str = await tool.ainvoke(args)
        actual_data = json.loads(res_str)

        if "error" in actual_data:
            raise RuntimeError(f"工具 {tool_name} 调用返回错误: {actual_data['error']}")

        if update:
            golden_file.write_text(json.dumps(actual_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  [UPDATED] {tool_name} -> {golden_filename}")
        else:
            compare_structure(actual_data, golden_data, path=f"{tool_name} ({golden_filename})")
            print(f"  [PASS] {tool_name} ({golden_filename}) 字段结构与类型校验一致")


def main():
    is_live = "--live" in sys.argv or os.getenv("AMAP_LIVE_TEST") == "1"
    update = "--update" in sys.argv
    if update and not is_live:
        print("❌ --update 必须与 --live 一同使用")
        sys.exit(1)

    if not is_live:
        print("ℹ️ 未指定 --live 参数或 AMAP_LIVE_TEST 环境变量，默认跳过在线测试。")
        print("运行方法: uv run python tests/live_amap_golden.py --live")
        return

    print("=" * 70)
    if update:
        print("开始运行高德本地 Tool 在线请求并更新 Golden 样本...")
    else:
        print("开始运行高德本地 Tool 与 Golden 样本在线结构对比测试...")
    print("=" * 70)
    try:
        asyncio.run(run_live_test(update=update))
        print("=" * 70)
        print("在线测试全部通过！" if not update else "Golden 样本全部更新完成！")
        print("=" * 70)
    finally:
        asyncio.run(amap.aclose())


if __name__ == "__main__":
    main()
