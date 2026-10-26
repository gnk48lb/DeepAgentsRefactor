"""
tests/smoke_amap_contract.py

离线契约测试：
1. 验证 get_tools() 返回 12 个工具，名称集合与 tests/fixtures/amap/tools_snapshot.json 完全一致。
2. 每个工具的 description 与快照逐字相等，含中文且不含 Args:/Returns:，无旧后缀。
3. 工具参数与 impl 签名契约一致性。
4. 坐标校验规则验证（离线纯函数与 maps_regeocode 拦截）。
5. 坐标规整函数验证（去空白、全角转半角、小数位截断等）。
"""

import asyncio
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import app.tools.amap as amap

SNAPSHOT_FIXTURE_PATH = ROOT / "tests/fixtures/amap/tools_snapshot.json"
SNAPSHOT_RUNTIME_PATH = ROOT / "app/tools/amap/tools_snapshot.json"


async def test_snapshot_file_consistency():
    fixture_text = SNAPSHOT_FIXTURE_PATH.read_text(encoding="utf-8")
    runtime_text = SNAPSHOT_RUNTIME_PATH.read_text(encoding="utf-8")
    assert fixture_text == runtime_text, "app/tools/amap/tools_snapshot.json 与 tests/fixtures/amap/tools_snapshot.json 内容不一致！"
    fixture_json = json.loads(fixture_text)
    assert len(fixture_json) == 12, f"快照中工具数量应为 12，实际为 {len(fixture_json)}"


async def test_tools_contract():
    tools = amap.get_tools()
    assert len(tools) == 12, f"get_tools() 返回数量应为 12，实际为 {len(tools)}"

    fixture_json = json.loads(SNAPSHOT_FIXTURE_PATH.read_text(encoding="utf-8"))
    snap_map = {item["name"]: item for item in fixture_json}

    tool_names = {t.name for t in tools}
    snap_names = set(snap_map.keys())
    assert tool_names == snap_names, f"工具名称集合不一致: diff={tool_names ^ snap_names}"

    assert not any(t.name.endswith(("_by_address", "_by_coordinates")) for t in tools), "存在旧的 _by_address 或 _by_coordinates 工具"

    tools_by_name = {t.name: t for t in tools}
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
        assert any("\u4e00" <= c <= "\u9fff" for c in t.description), f"{t.name} description 应含中文"
        assert "Args:" not in t.description, f"{t.name} description 不应含 Args:"
        assert "Returns:" not in t.description, f"{t.name} description 不应含 Returns:"

    assert tools_by_name["maps_route_transit"].args_schema["required"] == ["origin", "destination", "origin_city"]
    assert tools_by_name["maps_around_search"].args_schema["properties"]["radius"]["type"] == "integer"
    assert tools_by_name["maps_distance"].args_schema["properties"]["type"]["enum"] == ["0", "1", "3"]


async def test_schema_matches_impl_signature():
    from app.tools.amap import impl
    for t in amap.get_tools():
        sig = inspect.signature(getattr(impl, t.name)).parameters
        props, req = t.args_schema["properties"], t.args_schema["required"]
        assert set(props) <= set(sig), f"{t.name}: schema 里有 impl 不接受的参数 {set(props) - set(sig)}"
        assert set(req) <= set(props), t.name
        need = {k for k, p in sig.items() if p.default is inspect.Parameter.empty}
        assert need <= set(req), f"{t.name}: impl 必填参数没在 schema.required 里 {need - set(req)}"
        for pname, p in props.items():
            assert p.get("description"), f"{t.name}.{pname} 缺少参数描述"


async def test_coord_validation():
    from app.tools.amap import registry
    tools_map = {t.name: t for t in amap.get_tools()}
    expected_msg = (
        "工具 'maps_regeocode' 调用失败：参数 'location' 格式错误。"
        "该字段必须是 '经度,纬度' 格式的数字字符串（如 '117.05,36.65'）。"
        "你直接传入了地址文本 '文化东路42号'。"
        "请先调用 'maps_geo' 解析该地址获取坐标后再试。"
    )
    assert await tools_map["maps_regeocode"].ainvoke({"location": "文化东路42号"}) == expected_msg
    probe = {"location": "文化东路42号", "origin": "文化东路42号", "destination": "望京SOHO", "origins": "文化东路42号"}
    for name in tools_map:   # 直接测纯函数，不走 ainvoke，避免本机有 key 时打真实请求
        assert (registry._check_coords(name, probe) is not None) == (name == "maps_regeocode"), name
    assert registry._check_coords("maps_regeocode", {"location": " 116.48, 39.99 "}) is None


async def test_coordinate_normalization():
    from app.tools.amap import impl
    assert impl.normalize_coordinates("116.481488,39.990464") == "116.481488,39.990464"
    assert impl.normalize_coordinates(" 116.48 , 39.99 ") == "116.48,39.99"
    assert impl.normalize_coordinates("116.48，39.99") == "116.48,39.99"
    assert impl.normalize_coordinates("-0.5,39.99") == "-0.5,39.99"
    assert impl.normalize_coordinates("116.48148812,39.99046499") == "116.481488,39.990465"
    for invalid in ["文化东路42号", "", "116.48", "116.48,39.99,10", "望京SOHO", None, 5]:
        assert impl.normalize_coordinates(invalid) is None, f"{invalid} 应返回 None"


SCENARIOS = [
    ("快照文件一致性检查", test_snapshot_file_consistency),
    ("12 个工具名称/描述/Schema 契约验证", test_tools_contract),
    ("工具参数签名契约验证", test_schema_matches_impl_signature),
    ("坐标格式拦截规则验证", test_coord_validation),
    ("坐标规整函数验证", test_coordinate_normalization),
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
