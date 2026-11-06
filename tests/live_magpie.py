"""Magpie Anthropic 协议实测。仅显式 RUN_MAGPIE_LIVE=1 时运行，不进入离线 smoke/CI。"""
import asyncio
import base64
import json
import os
import struct
import urllib.request
import zlib

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from app import config
from app.utils.text import extract_text


def _red_png_data_uri() -> str:
    def chunk(kind, data):
        raw = kind + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)
    width = height = 64
    pixels = b"".join(b"\x00" + bytes([220, 30, 30] * width) for _ in range(height))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _check_model_catalog():
    with urllib.request.urlopen(config.MAGPIE_BASE_URL.rstrip("/") + "/v1/models", timeout=10) as response:
        catalog = json.load(response)["data"]
    selected = next((m for m in catalog if m.get("id") == config.MAGPIE_MODEL), None)
    assert selected is not None, f"网关中不存在 {config.MAGPIE_MODEL}"
    modalities = (selected.get("modalities") or {}).get("input")
    if modalities is not None and "image" not in modalities:
        raise AssertionError(f"{config.MAGPIE_MODEL} 明确不支持图片，不能用于本次视觉探针")
    print(f"  模型: {config.MAGPIE_MODEL}; modalities={modalities or '未声明'}")


@tool
def get_status() -> str:
    """Get a harmless status value."""
    return "ready"


@tool
def make_chart() -> str:
    """Make a chart."""
    return "chart rendered"


class BrowserStep(BaseModel):
    thought: str
    action: str


async def _run():
    _check_model_catalog()
    llm = ChatAnthropic(
        model=config.MAGPIE_MODEL,
        base_url=config.MAGPIE_BASE_URL,
        api_key=config.MAGPIE_API_KEY,
        max_tokens=2048,
        timeout=120,
        max_retries=1,
    )

    plain = await llm.ainvoke("Reply with exactly: PONG")
    assert "PONG" in extract_text(plain.content), plain.content
    print("  [PASS] 普通文本（thinking 未混入正文）", plain.usage_metadata)

    bound = llm.bind_tools([get_status])
    first = await bound.ainvoke([HumanMessage(content="Use get_status once, then report its result.")])
    assert first.tool_calls, first.content
    second = await bound.ainvoke([
        HumanMessage(content="Use get_status once, then report its result."),
        first,
        ToolMessage(content="ready", tool_call_id=first.tool_calls[0]["id"], name="get_status"),
    ])
    assert "ready" in extract_text(second.content).lower(), second.content
    print("  [PASS] 工具调用与 tool result 第二轮")

    image = _red_png_data_uri()
    vision = await llm.ainvoke([HumanMessage(content=[
        {"type": "text", "text": "What is the dominant color? Answer with one English word."},
        {"type": "image_url", "image_url": {"url": image}},
    ])])
    assert "red" in extract_text(vision.content).lower(), vision.content
    print("  [PASS] image_url 视觉输入")

    chart_bound = llm.bind_tools([make_chart])
    chart_ai = AIMessage(content="", tool_calls=[{"name": "make_chart", "args": {}, "id": "chart_live"}])
    chart_result = await chart_bound.ainvoke([
        HumanMessage(content="Run make_chart, then tell me the dominant color in its result."),
        chart_ai,
        ToolMessage(content=[
            {"type": "text", "text": "chart rendered"},
            {"type": "image_url", "image_url": {"url": image}},
        ], tool_call_id="chart_live", name="make_chart"),
    ])
    assert "red" in extract_text(chart_result.content).lower(), chart_result.content
    print("  [PASS] tool result 内图片")

    from browser_use.llm.anthropic.chat import ChatAnthropic as BrowserAnthropic
    from browser_use.llm.messages import UserMessage
    browser_llm = BrowserAnthropic(
        model=config.MAGPIE_MODEL,
        base_url=config.MAGPIE_BASE_URL,
        api_key=config.MAGPIE_API_KEY,
        max_tokens=1024,
        timeout=120,
        max_retries=1,
    )
    structured = await browser_llm.ainvoke(
        [UserMessage(content="Decide one browser action: search for cats. Return thought and action.")],
        output_format=BrowserStep,
    )
    assert isinstance(structured.completion, BrowserStep), structured.completion
    assert structured.completion.action.strip()
    print("  [PASS] browser_use 强制 tool-use 结构化输出")


def main():
    if os.getenv("RUN_MAGPIE_LIVE") != "1":
        print("跳过：仅在显式设置 RUN_MAGPIE_LIVE=1 时访问 Magpie。")
        return
    asyncio.run(_run())
    print("Magpie 协议级 live 探针全部通过（未启动浏览器/Desktop，未执行真实工具）。")


if __name__ == "__main__":
    main()
