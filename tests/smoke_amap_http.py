"""
tests/smoke_amap_http.py

离线 HTTP 错误与边界冒烟测试（通过 httpx.MockTransport 模拟，不消耗真实网络与 API 配额）：
覆盖 13 个场景：
1. 成功请求响应及字段映射
2. 业务错误（status != "1" / errcode != 0）及错误前缀
3. HTTP 500 错误及 key=*** 脱敏
4. 超时异常捕获及错误格式
5. 非法 JSON 响应处理
6. status 字段缺失触发 KeyError 导致的外层双层执行出错包装
7. 路线双地址：geo 解析与参数传递
8. 坐标直通：坐标规整与跳过 geo
9. 歧义守卫：多城市候选拦截与城市限定放行
10. 公交城市：起终点城市联动与必填校验
11. 周边搜索：文字中心解析与字段映射
12. 多起点测距：坐标/地址混传、透传错误与超限拦截
13. 天气名称解析：中文转 adcode 与纯数字直传
"""

import asyncio
import json
import sys
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]

from app import config
import app.tools.amap as amap
import app.tools.amap.client as amap_client

if not config.AMAP_API_KEY:  # 离线测试不需要真实 key:仅为通过 validate_config() 而占位
    config.AMAP_API_KEY = "TEST_KEY_FOR_OFFLINE_SMOKE"


def setup_mock_client(handler):
    transport = httpx.MockTransport(handler)
    mock_c = httpx.AsyncClient(
        transport=transport,
        base_url="https://restapi.amap.com",
    )
    amap_client.set_client(mock_c)
    return mock_c


async def test_success_case():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "key=" in str(request.url)
        return httpx.Response(
            200,
            json={
                "status": "1",
                "info": "OK",
                "infocode": "10000",
                "province": "山东省",
                "city": "济南市",
                "adcode": "370100",
                "rectangle": "116.9,36.5;117.1,36.7",
            },
        )

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}
    res_str = await tools["maps_ip_location"].ainvoke({"ip": "1.1.1.1"})
    data = json.loads(res_str)
    assert data.get("province") == "山东省"
    assert data.get("city") == "济南市"
    assert "error" not in data


async def test_business_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "0",
                "info": "INVALID_USER_KEY",
                "infocode": "10001",
            },
        )

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}
    res_str = await tools["maps_geo"].ainvoke({"address": "测试地址"})
    data = json.loads(res_str)
    assert "error" in data
    assert data["error"] == "Geocoding failed: INVALID_USER_KEY"


async def test_http_500_and_key_redaction():
    secret_key = config.AMAP_API_KEY or "TEST_SECRET_KEY_123456"
    orig_key = config.AMAP_API_KEY
    config.AMAP_API_KEY = secret_key

    try:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        setup_mock_client(handler)
        tools = {t.name: t for t in amap.get_tools()}
        res_str = await tools["maps_regeocode"].ainvoke({"location": "116.48,39.99"})
        data = json.loads(res_str)
        assert "error" in data
        assert data["error"].startswith("Request failed: ")
        assert "500" in data["error"]
        assert secret_key not in data["error"], "真实 API Key 未被脱敏！"
        assert "key=***" in data["error"]
    finally:
        config.AMAP_API_KEY = orig_key


async def test_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("The read operation timed out")

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}
    res_str = await tools["maps_weather"].ainvoke({"city": "370100"})
    data = json.loads(res_str)
    assert "error" in data
    assert data["error"] == "Request failed: The read operation timed out"


async def test_invalid_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>502 Bad Gateway</html>")

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}
    res_str = await tools["maps_distance"].ainvoke({
        "origins": "116.48,39.99",
        "destination": "116.49,39.99",
    })
    data = json.loads(res_str)
    assert "error" in data
    assert data["error"].startswith("Request failed: ")


async def test_missing_status_key_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected_field": 123})

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}
    res_str = await tools["maps_regeocode"].ainvoke({"location": "116.48,39.99"})
    expected_prefix = "工具 'maps_regeocode' 执行出错: Error executing tool maps_regeocode: 'status'"
    assert res_str == expected_prefix, f"KeyError 双层包装文案不匹配:\nExpected: {expected_prefix}\nActual: {res_str}"


async def test_route_both_addresses():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v3/geocode/geo":
            addr = request.url.params.get("address")
            if addr == "起点文字":
                return httpx.Response(200, json={"status": "1", "geocodes": [{"province": "北京市", "city": "北京市", "location": "116.11,39.11", "level": "门址"}]})
            return httpx.Response(200, json={"status": "1", "geocodes": [{"province": "北京市", "city": "北京市", "location": "116.22,39.22", "level": "门址"}]})
        if request.url.path == "/v3/direction/walking":
            return httpx.Response(200, json={"status": "1", "route": {"origin": request.url.params["origin"], "destination": request.url.params["destination"], "paths": []}})
        return httpx.Response(404)

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}
    res_str = await tools["maps_route_walking"].ainvoke({"origin": "起点文字", "destination": "终点文字"})
    data = json.loads(res_str)
    assert len(calls) == 3
    assert calls[0].url.path == "/v3/geocode/geo"
    assert calls[1].url.path == "/v3/geocode/geo"
    assert calls[2].url.path == "/v3/direction/walking"
    assert calls[2].url.params["origin"] == "116.11,39.11"
    assert calls[2].url.params["destination"] == "116.22,39.22"
    assert "route" in data
    assert "addresses" in data
    assert data["addresses"]["origin"] == {"address": "起点文字", "coordinates": "116.11,39.11"}
    assert data["addresses"]["destination"] == {"address": "终点文字", "coordinates": "116.22,39.22"}


async def test_route_coordinates_passthrough():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v3/geocode/geo":
            return httpx.Response(200, json={"status": "1", "geocodes": [{"province": "北京市", "city": "北京市", "location": "116.22,39.22", "level": "门址"}]})
        if request.url.path == "/v3/direction/walking":
            return httpx.Response(200, json={"status": "1", "route": {"origin": request.url.params["origin"], "destination": request.url.params["destination"], "paths": []}})
        return httpx.Response(404)

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}

    # 纯坐标直通
    res1_str = await tools["maps_route_walking"].ainvoke({"origin": " 116.48, 39.99 ", "destination": "116.49,39.99"})
    data1 = json.loads(res1_str)
    assert len(calls) == 1
    assert calls[0].url.path == "/v3/direction/walking"
    assert calls[0].url.params["origin"] == "116.48,39.99"
    assert "route" in data1
    assert "addresses" not in data1

    # 一端坐标一端地址
    calls.clear()
    res2_str = await tools["maps_route_walking"].ainvoke({"origin": "116.48,39.99", "destination": "终点文字"})
    data2 = json.loads(res2_str)
    assert len(calls) == 2
    assert calls[0].url.path == "/v3/geocode/geo"
    assert calls[1].url.path == "/v3/direction/walking"
    assert "addresses" in data2
    assert "origin" not in data2["addresses"]
    assert data2["addresses"]["destination"] == {"address": "终点文字", "coordinates": "116.22,39.22"}


async def test_ambiguity_guard():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v3/geocode/geo":
            addr = request.url.params.get("address")
            city = request.url.params.get("city")
            if addr == "歧义地址" and not city:
                return httpx.Response(200, json={
                    "status": "1",
                    "geocodes": [
                        {"province": "北京市", "city": "北京市", "location": "116.1,39.1", "level": "门址"},
                        {"province": "山东省", "city": "济南市", "location": "117.1,36.1", "level": "门址"},
                    ],
                })
            if addr == "歧义地址" and city == "济南市":
                return httpx.Response(200, json={
                    "status": "1",
                    "geocodes": [
                        {"province": "山东省", "city": "济南市", "location": "117.1,36.1", "level": "门址"},
                    ],
                })
            if addr == "同城多候选":
                return httpx.Response(200, json={
                    "status": "1",
                    "geocodes": [
                        {"province": "北京市", "city": "北京市", "location": "116.1,39.1", "level": "门址"},
                        {"province": "北京市", "city": "北京市", "location": "116.2,39.2", "level": "门址"},
                    ],
                })
        if request.url.path == "/v3/direction/walking":
            return httpx.Response(200, json={"status": "1", "route": {"origin": request.url.params["origin"], "destination": request.url.params["destination"], "paths": []}})
        return httpx.Response(404)

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}

    # 1. 跨城歧义且未指定城市 -> 报错，不发 walking
    res1 = json.loads(await tools["maps_route_walking"].ainvoke({"origin": "歧义地址", "destination": "116.49,39.99"}))
    assert "error" in res1
    assert "有歧义" in res1["error"]
    assert "北京" in res1["error"] and "济南" in res1["error"]
    assert not any(r.url.path == "/v3/direction/walking" for r in calls)

    # 2. 传 origin_city="济南市" 放行
    calls.clear()
    res2 = json.loads(await tools["maps_route_walking"].ainvoke({"origin": "歧义地址", "destination": "116.49,39.99", "origin_city": "济南市"}))
    assert "error" not in res2
    assert any(r.url.path == "/v3/direction/walking" for r in calls)
    assert calls[0].url.params["city"] == "济南市"

    # 3. 两个候选同在一个城市放行
    calls.clear()
    res3 = json.loads(await tools["maps_route_walking"].ainvoke({"origin": "同城多候选", "destination": "116.49,39.99"}))
    assert "error" not in res3
    assert any(r.url.path == "/v3/direction/walking" for r in calls)


async def test_route_transit_city():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v3/direction/transit/integrated":
            return httpx.Response(200, json={"status": "1", "route": {"origin": request.url.params["origin"], "destination": request.url.params["destination"], "transits": []}})
        return httpx.Response(404)

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}

    # 只给 origin_city="北京"
    await tools["maps_route_transit"].ainvoke({"origin": "116.48,39.99", "destination": "116.49,39.99", "origin_city": "北京"})
    assert len(calls) == 1
    assert calls[0].url.params["city"] == "北京"
    assert calls[0].url.params["cityd"] == "北京"

    # 同时给 destination_city="天津"
    calls.clear()
    await tools["maps_route_transit"].ainvoke({"origin": "116.48,39.99", "destination": "117.1,39.1", "origin_city": "北京", "destination_city": "天津"})
    assert len(calls) == 1
    assert calls[0].url.params["city"] == "北京"
    assert calls[0].url.params["cityd"] == "天津"

    # 不给 origin_city -> 报错且不发请求
    calls.clear()
    res_err = json.loads(await tools["maps_route_transit"].ainvoke({"origin": "116.48,39.99", "destination": "116.49,39.99"}))
    assert "error" in res_err
    assert "origin_city" in res_err["error"]
    assert len(calls) == 0


async def test_around_search():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v3/geocode/geo":
            return httpx.Response(200, json={"status": "1", "geocodes": [{"province": "北京市", "city": "北京市", "location": "116.481488,39.990464", "level": "门址"}]})
        if request.url.path == "/v3/place/around":
            return httpx.Response(200, json={
                "status": "1",
                "pois": [
                    {
                        "id": "B001",
                        "name": "星巴克咖啡",
                        "address": "望京街10号",
                        "location": "116.48,39.99",
                        "distance": "120",
                        "type": "餐饮服务;咖啡厅;星巴克",
                        "typecode": "050400",
                    }
                ],
            })
        return httpx.Response(404)

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}
    res = json.loads(await tools["maps_around_search"].ainvoke({"location": "北京市朝阳区阜通东大街6号", "radius": 500.0, "keywords": "咖啡"}))

    assert len(calls) == 2
    assert calls[0].url.path == "/v3/geocode/geo"
    assert calls[1].url.path == "/v3/place/around"
    assert calls[1].url.params["radius"] == "500"
    assert calls[1].url.params["location"] == "116.481488,39.990464"
    assert calls[1].url.params["keywords"] == "咖啡"

    pois = res.get("pois", [])
    assert len(pois) == 1
    assert set(pois[0].keys()) == {"id", "name", "address", "location", "distance", "type"}
    assert pois[0]["distance"] == "120"
    assert pois[0]["type"] == "餐饮服务;咖啡厅;星巴克"


async def test_distance_multi_origin():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v3/geocode/geo":
            return httpx.Response(200, json={"status": "1", "geocodes": [{"province": "北京市", "city": "北京市", "location": "116.48,39.99", "level": "门址"}]})
        if request.url.path == "/v3/distance":
            return httpx.Response(200, json={
                "status": "1",
                "results": [
                    {"origin_id": "1", "distance": "5000", "duration": "1200"},
                    {"origin_id": "2", "distance": "0", "duration": "0", "info": "起点终点距离过近", "code": "10001"},
                ],
            })
        return httpx.Response(404)

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}

    # 1. 坐标与地址混传、type="3"、透传 info/code
    res = json.loads(await tools["maps_distance"].ainvoke({
        "origins": "116.1,39.1|望京SOHO",
        "destination": "116.2,39.2",
        "type": "3",
        "city": "北京",
    }))
    assert len(calls) == 2
    assert calls[0].url.path == "/v3/geocode/geo"
    assert calls[1].url.path == "/v3/distance"
    assert calls[1].url.params["origins"] == "116.1,39.1|116.48,39.99"
    assert calls[1].url.params["type"] == "3"
    results = res.get("results", [])
    assert len(results) == 2
    assert results[0]["origin"] == "116.1,39.1"
    assert "info" not in results[0]
    assert results[1]["origin"] == "望京SOHO"
    assert results[1]["info"] == "起点终点距离过近"
    assert results[1]["code"] == "10001"

    # 2. 文字地址超过 10 个拦截且不发请求
    calls.clear()
    over_limit = "|".join(f"地标{i}" for i in range(11))
    res_err = json.loads(await tools["maps_distance"].ainvoke({
        "origins": over_limit,
        "destination": "116.2,39.2",
    }))
    assert "error" in res_err
    assert "最多 10" in res_err["error"]
    assert len(calls) == 0


async def test_weather_name_resolution():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v3/geocode/geo":
            return httpx.Response(200, json={"status": "1", "geocodes": [{"province": "山东省", "city": "济南市", "adcode": "370100", "location": "117.0,36.6", "level": "市"}]})
        if request.url.path == "/v3/weather/weatherInfo":
            return httpx.Response(200, json={"status": "1", "forecasts": [{"city": "济南市", "adcode": "370100", "casts": []}]})
        return httpx.Response(404)

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}

    # 中文地名 -> 先 geo 获取 adcode，再请求 weatherInfo
    res1 = json.loads(await tools["maps_weather"].ainvoke({"city": "济南市"}))
    assert len(calls) == 2
    assert calls[0].url.path == "/v3/geocode/geo"
    assert calls[1].url.path == "/v3/weather/weatherInfo"
    assert calls[1].url.params["city"] == "370100"
    assert res1.get("city") == "济南市"

    # 6 位 adcode -> 不发 geo，直传 weatherInfo
    calls.clear()
    res2 = json.loads(await tools["maps_weather"].ainvoke({"city": "370100"}))
    assert len(calls) == 1
    assert calls[0].url.path == "/v3/weather/weatherInfo"
    assert calls[0].url.params["city"] == "370100"
    assert res2.get("city") == "济南市"


SCENARIOS = [
    ("成功请求与字段提取验证", test_success_case),
    ("业务错误响应与前缀验证", test_business_error),
    ("HTTP 500 错误与密钥脱敏验证", test_http_500_and_key_redaction),
    ("超时异常处理验证", test_timeout_error),
    ("非 JSON 异常响应验证", test_invalid_json),
    ("status 缺失触发 KeyError 双层前缀验证", test_missing_status_key_error),
    ("路线双地址：geo 解析与参数传递", test_route_both_addresses),
    ("坐标直通：坐标规整与跳过 geo", test_route_coordinates_passthrough),
    ("歧义守卫：多城市候选拦截与城市限定放行", test_ambiguity_guard),
    ("公交城市：起终点城市联动与必填校验", test_route_transit_city),
    ("周边搜索：文字中心解析与字段映射", test_around_search),
    ("多起点测距：坐标/地址混传、透传错误与超限拦截", test_distance_multi_origin),
    ("天气名称解析：中文转 adcode 与纯数字直传", test_weather_name_resolution),
]


def main():
    print("=" * 70)
    print("开始运行高德本地 Tool HTTP 异常与脱敏冒烟测试...")
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
        finally:
            amap_client.set_client(None)

    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
