import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from langchain_mcp_adapters.client import MultiServerMCPClient
import app.mcp_service as mcp_service

GOLDEN_DIR = ROOT / "tests/fixtures/amap/golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

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
]


async def main():
    if not config.AMAP_API_KEY:
        print("AMAP_API_KEY is not set, skipping golden capture")
        return

    mcp_config = {
        "amap": {
            "command": "uvx",
            "args": [
                "--quiet",
                "--with",
                "amap-mcp-server==0.1.11",
                "--with",
                "mcp==1.8.1",
                "--with",
                "pydantic==2.10.6",
                "python",
                "app/amap_wrapper.py",
            ],
            "env": {"AMAP_MAPS_API_KEY": config.AMAP_API_KEY},
            "transport": "stdio",
        }
    }

    client = MultiServerMCPClient(mcp_config)
    runner = mcp_service._ServerRunner("amap", client)
    mcp_service._runners["amap"] = runner
    runner.start()
    try:
        await asyncio.wait_for(runner.ready.wait(), timeout=60.0)
        wrapped_map = {name: mcp_service._wrap_tool(tool, "amap") for name, tool in runner.tools.items()}

        poi_id = None
        for name, args in CASES:
            tool = wrapped_map[name]
            res = await tool.ainvoke(args)
            if config.AMAP_API_KEY in res:
                res = res.replace(config.AMAP_API_KEY, "***")
            file_path = GOLDEN_DIR / f"{name}.default.json"
            file_path.write_text(res, encoding="utf-8")
            print(f"Captured golden for {name} -> {file_path}")

            if name == "maps_text_search":
                try:
                    data = json.loads(res)
                    if data.get("pois"):
                        poi_id = data["pois"][0]["id"]
                except Exception:
                    pass

        if not poi_id:
            poi_id = "B000A85652"
        detail_tool = wrapped_map["maps_search_detail"]
        res = await detail_tool.ainvoke({"id": poi_id})
        if config.AMAP_API_KEY in res:
            res = res.replace(config.AMAP_API_KEY, "***")
        file_path = GOLDEN_DIR / "maps_search_detail.default.json"
        file_path.write_text(res, encoding="utf-8")
        print(f"Captured golden for maps_search_detail -> {file_path}")
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
