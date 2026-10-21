"""
tests/smoke_amap_contract.py

离线契约测试：
1. 验证 get_tools() 返回 16 个工具，名称集合与 tests/fixtures/amap/tools_snapshot.json 完全一致。
2. 每个工具的 description 与快照逐字相等。
3. 每个工具的 args_schema 与快照的 inputSchema 完全相等。
4. app/amap/tools_snapshot.json 与 tests/fixtures/amap/tools_snapshot.json 内容完全一致。
5. 坐标校验规则：
   - 对 maps_around_search(location="文化东路42号") 返回固定错误文案；
   - 对 maps_bicycling_by_coordinates(origin_coordinates="文化东路42号", ...) 不触发校验（因参数名为 origin_coordinates）。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


import app.amap as amap

SNAPSHOT_FIXTURE_PATH = ROOT / "tests/fixtures/amap/tools_snapshot.json"
SNAPSHOT_RUNTIME_PATH = ROOT / "app/amap/tools_snapshot.json"


async def test_snapshot_file_consistency():
    fixture_text = SNAPSHOT_FIXTURE_PATH.read_text(encoding="utf-8")
    runtime_text = SNAPSHOT_RUNTIME_PATH.read_text(encoding="utf-8")
    assert fixture_text == runtime_text, "app/amap/tools_snapshot.json 与 tests/fixtures/amap/tools_snapshot.json 内容不一致！"
    fixture_json = json.loads(fixture_text)
    assert len(fixture_json) == 16, f"快照中工具数量应为 16，实际为 {len(fixture_json)}"


async def test_tools_contract():
    tools = amap.get_tools()
    assert len(tools) == 16, f"get_tools() 返回数量应为 16，实际为 {len(tools)}"

    fixture_json = json.loads(SNAPSHOT_FIXTURE_PATH.read_text(encoding="utf-8"))
    snap_map = {item["name"]: item for item in fixture_json}

    tool_names = {t.name for t in tools}
    snap_names = set(snap_map.keys())
    assert tool_names == snap_names, f"工具名称集合不一致: diff={tool_names ^ snap_names}"

    for t in tools:
        expected = snap_map[t.name]
        assert t.description == expected["description"], (
            f"工具 {t.name} 的 description 与快照不一致:\n"
            f"Expected: {expected['description']}\n"
            f"Actual: {t.description}"
        )
        assert t.args_schema == expected["inputSchema"], (
            f"工具 {t.name} 的 args_schema 与快照不一致:\n"
            f"Expected: {expected['inputSchema']}\n"
            f"Actual: {t.args_schema}"
        )


async def test_coord_validation():
    tools_map = {t.name: t for t in amap.get_tools()}

    # 1. maps_around_search 传入文字地址应拦截并返回固定文案
    around_tool = tools_map["maps_around_search"]
    res = await around_tool.ainvoke({"location": "文化东路42号"})
    expected_msg = (
        "工具 'maps_around_search' 调用失败：参数 'location' 格式错误。"
        "该字段必须是 '经度,纬度' 格式的数字字符串（如 '117.05,36.65'）。"
        "你直接传入了地址文本 '文化东路42号'。"
        "请先调用 'maps_geo' 解析该地址获取坐标后再试。"
    )
    assert res == expected_msg, f"坐标拦截文案不匹配:\nExpected: {expected_msg}\nActual: {res}"

    # 2. maps_bicycling_by_coordinates 的参数是 origin_coordinates/destination_coordinates，不被 _COORD_FIELDS 拦截
    # 在未配置网络或模拟时，其不会被 _check_coords 拦截为地址格式错误，而是继续向下调用 impl（触发 validate_config 或 network）
    bike_tool = tools_map["maps_bicycling_by_coordinates"]
    bike_res = await bike_tool.ainvoke({
        "origin_coordinates": "文化东路42号",
        "destination_coordinates": "116.486488,39.995464",
    })
    assert "调用失败：参数 'origin_coordinates' 格式错误" not in bike_res, (
        f"maps_bicycling_by_coordinates 不应触发坐标校验，实际输出: {bike_res}"
    )


SCENARIOS = [
    ("快照文件一致性检查", test_snapshot_file_consistency),
    ("16 个工具名称/描述/Schema 契约验证", test_tools_contract),
    ("坐标格式拦截规则验证", test_coord_validation),
]


def main():
    print("=" * 70)
    print("开始运行高德本地 Tool 契约离线冒烟测试...")
    print("=" * 70)
    passed = 0
    failed = 0

    for name, test_func in SCENARIOS:
        try:
            asyncio.run(test_func())
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
