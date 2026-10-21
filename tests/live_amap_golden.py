"""
tests/live_amap_golden.py

在线（Live）Amap Golden 对比测试：
读取 tests/fixtures/amap/golden/*.default.json，对比真实 API 返回结果与 Golden 样本的“字段集合与类型”（而非文本全等）。
默认跳过（除非传入命令行参数 --live 或设置环境变量 AMAP_LIVE_TEST=1）。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


from app import config
import app.amap as amap

GOLDEN_DIR = ROOT / "tests/fixtures/amap/golden"

CASES = [
    ("maps_geo", {"address": "北京市朝阳区阜通东大街6号"}),
    ("maps_regeocode", {"location": "116.481488,39.990464"}),
    ("maps_ip_location", {"ip": "114.114.114.114"}),
    ("maps_weather", {"city": "110101"}),
    (
        "maps_bicycling_by_coordinates",
        {
            "origin_coordinates": "116.481488,39.990464",
            "destination_coordinates": "116.486488,39.995464",
        },
    ),
    (
        "maps_bicycling_by_address",
        {
            "origin_address": "北京市朝阳区阜通东大街6号",
            "destination_address": "北京市朝阳区望京SOHO",
        },
    ),
    (
        "maps_direction_walking_by_coordinates",
        {
            "origin": "116.481488,39.990464",
            "destination": "116.486488,39.995464",
        },
    ),
    (
        "maps_direction_walking_by_address",
        {
            "origin_address": "北京市朝阳区阜通东大街6号",
            "destination_address": "北京市朝阳区望京SOHO",
        },
    ),
    (
        "maps_direction_driving_by_coordinates",
        {
            "origin": "116.481488,39.990464",
            "destination": "116.486488,39.995464",
        },
    ),
    (
        "maps_direction_driving_by_address",
        {
            "origin_address": "北京市朝阳区阜通东大街6号",
            "destination_address": "北京市朝阳区望京SOHO",
        },
    ),
    (
        "maps_direction_transit_integrated_by_coordinates",
        {
            "origin": "116.481488,39.990464",
            "destination": "116.486488,39.995464",
            "city": "北京",
            "cityd": "北京",
        },
    ),
    (
        "maps_direction_transit_integrated_by_address",
        {
            "origin_address": "北京市朝阳区阜通东大街6号",
            "destination_address": "北京市朝阳区望京SOHO",
            "origin_city": "北京",
            "destination_city": "北京",
        },
    ),
    (
        "maps_distance",
        {
            "origins": "116.481488,39.990464",
            "destination": "116.486488,39.995464",
        },
    ),
    ("maps_text_search", {"keywords": "肯德基", "city": "北京"}),
    (
        "maps_around_search",
        {"location": "116.481488,39.990464", "keywords": "餐饮"},
    ),
    ("maps_search_detail", {"id": "B000A85652"}),
]


def compare_structure(actual, expected, path=""):
    """递归对比字段集合与基本类型。"""
    if type(actual) is not type(expected):
        raise AssertionError(
            f"类型不匹配 at {path}: expected {type(expected).__name__}, got {type(actual).__name__}"
        )

    if isinstance(expected, dict):
        # 允许某些动态扩展字段，但主要 key 集合应相同
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


async def run_live_test():
    tools_map = {t.name: t for t in amap.get_tools()}
    for tool_name, args in CASES:
        golden_file = GOLDEN_DIR / f"{tool_name}.default.json"
        if not golden_file.exists():
            print(f"  [SKIP] {tool_name} (未找到 Golden 文件: {golden_file.name})")
            continue

        golden_data = json.loads(golden_file.read_text(encoding="utf-8"))
        if tool_name == "maps_search_detail" and "id" in golden_data:
            args = {"id": golden_data["id"]}
        tool = tools_map[tool_name]
        res_str = await tool.ainvoke(args)
        actual_data = json.loads(res_str)

        try:
            compare_structure(actual_data, golden_data, path=tool_name)
            print(f"  [PASS] {tool_name} 字段结构与类型校验一致")
        except AssertionError as e:
            print(f"  [FAIL] {tool_name}: {e}")
            raise


def main():
    is_live = "--live" in sys.argv or os.getenv("AMAP_LIVE_TEST") == "1"
    if not is_live:
        print("ℹ️ 未指定 --live 参数或 AMAP_LIVE_TEST 环境变量，默认跳过在线测试。")
        print("运行方法: uv run python tests/live_amap_golden.py --live")
        return

    print("=" * 70)
    print("开始运行高德本地 Tool 与 Golden 样本在线结构对比测试...")
    print("=" * 70)
    try:
        asyncio.run(run_live_test())
        print("=" * 70)
        print("在线测试全部通过！")
        print("=" * 70)
    finally:
        asyncio.run(amap.aclose())


if __name__ == "__main__":
    main()
