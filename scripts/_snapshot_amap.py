import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AMAP_MAPS_API_KEY", "snapshot-placeholder")
os.environ.setdefault("AMAP_API_KEY", "snapshot-placeholder")

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from app.mcp_service import _wrap_tool

CONFIG = {
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
        "env": {"AMAP_MAPS_API_KEY": "snapshot-placeholder"},
        "transport": "stdio",
    }
}


async def main():
    client = MultiServerMCPClient(CONFIG)
    async with client.session("amap") as session:
        raw = await load_mcp_tools(session)
        wrapped = [_wrap_tool(t, "amap") for t in raw]
        snap = [
            {
                "name": w.name,
                "description": w.description,
                "inputSchema": (
                    w.args_schema.model_json_schema()
                    if hasattr(w.args_schema, "model_json_schema")
                    else w.args_schema
                ),
            }
            for w in wrapped
        ]
    assert len(snap) == 16, [s["name"] for s in snap]
    out = ROOT / "tests/fixtures/amap/tools_snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(snap)} tools -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
