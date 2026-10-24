"""
tests/smoke_amap_http.py

离线 HTTP 错误与边界冒烟测试（通过 httpx.MockTransport 模拟，不消耗真实网络与 API 配额）：
覆盖：
1. 成功请求响应及字段映射
2. 业务错误（status != "1" / errcode != 0）及错误前缀
3. HTTP 500 错误及 key=*** 脱敏
4. 超时异常捕获及错误格式
5. 非法 JSON 响应处理
6. status 字段缺失触发 KeyError 导致的外层双层执行出错包装
7. 敏感密钥绝对脱敏断言（无论出现在 URL、异常还是响应体）
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

if not config.AMAP_API_KEY:   # 离线测试不需要真实 key:仅为通过 validate_config() 而占位
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
    res_str = await tools["maps_weather"].ainvoke({"city": "北京"})
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
    # 模拟响应中缺少 status 字段，从而在 data["status"] 处抛出 KeyError
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected_field": 123})

    setup_mock_client(handler)
    tools = {t.name: t for t in amap.get_tools()}
    res_str = await tools["maps_regeocode"].ainvoke({"location": "116.48,39.99"})
    expected_prefix = "工具 'maps_regeocode' 执行出错: Error executing tool maps_regeocode: 'status'"
    assert res_str == expected_prefix, f"KeyError 双层包装文案不匹配:\nExpected: {expected_prefix}\nActual: {res_str}"


SCENARIOS = [
    ("成功请求与字段提取验证", test_success_case),
    ("业务错误响应与前缀验证", test_business_error),
    ("HTTP 500 错误与密钥脱敏验证", test_http_500_and_key_redaction),
    ("超时异常处理验证", test_timeout_error),
    ("非 JSON 异常响应验证", test_invalid_json),
    ("status 缺失触发 KeyError 双层前缀验证", test_missing_status_key_error),
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
