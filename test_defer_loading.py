"""test_defer_loading.py — 独立探测：当前上游模型 + 代理链路是否支持 Anthropic 工具延迟加载。

探测链路（默认值与 app/config.py 一致，但本脚本独立运行，不 import / 不修改 app 任何代码）：
    httpx -> Magpie 网关 (http://127.0.0.1:3425/v1/messages)
          -> minimax-code-global/MiniMax-M3.1-Flash（native_endpoints=["/v1/messages"]）

探测字段依据 Anthropic 官方 Tool Search Tool 文档（2026-11 核对，已 GA，无需 beta header）：
    - 普通工具定义上挂 "defer_loading": true
    - 服务端搜索工具 {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
    - 支持时响应会出现 tool_search_tool_result / tool_reference 块
    - 对照变体附带 2025-11 发布时的 legacy beta header: advanced-tool-use-2025-11-20

依赖：httpx（项目已有）。运行：
    .venv/Scripts/python.exe test_defer_loading.py
    .venv/Scripts/python.exe test_defer_loading.py --only tool_search_full --model <其他模型>

原始请求体 / 响应体 / 状态码全部写入 test_defer_loading_results.json（api key 脱敏）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

# ---- 配置：与 app/config.py 相同的默认值，可被环境变量 / 命令行覆盖 ----
DEFAULT_BASE_URL = os.getenv("MAGPIE_BASE_URL", "http://127.0.0.1:3425")
DEFAULT_API_KEY = os.getenv("MAGPIE_API_KEY", "local")
DEFAULT_MODEL = os.getenv("MAGPIE_MODEL", "minimax-code-global/MiniMax-M3.1-Flash")

LEGACY_BETA = "advanced-tool-use-2025-11-20"  # 发布时的 beta header，仅作对照

# ---- 工具定义：1 个普通工具 + 2 个 defer_loading 工具 ----
TOOL_TIME = {
    "name": "get_current_time",
    "description": "获取当前的日期和时间。询问当前时间时使用。",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}
TOOL_WEATHER_DEFERRED = {
    "name": "get_weather",
    "description": "查询指定城市的实时天气。询问天气时使用。",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "城市名，如 北京"}},
        "required": ["location"],
    },
    "defer_loading": True,
}
TOOL_FILES_DEFERRED = {
    "name": "search_files",
    "description": "在工作区文件中搜索内容。查找文件时使用。",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "搜索关键词"}},
        "required": ["query"],
    },
    "defer_loading": True,
}
TOOL_SEARCH_SERVER = {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}

DEFERRED_NAMES = {"get_weather", "search_files"}
PROMPT_TIME = "请查询当前时间。必须使用工具回答，不要凭记忆回答。"
PROMPT_WEATHER = "北京今天的天气怎么样？必须使用工具查询，不要凭记忆回答。"

CASES = [
    # 基线：确认链路本身可用（无 defer_loading）
    {"name": "baseline", "tools": [TOOL_TIME], "prompt": PROMPT_TIME, "beta": None},
    # 核心探测：defer_loading 字段是否被接受（当前 GA 文档不需要 beta header）
    {"name": "defer_field", "tools": [TOOL_TIME, TOOL_WEATHER_DEFERRED, TOOL_FILES_DEFERRED],
     "prompt": PROMPT_TIME, "beta": None},
    # 对照：附带 legacy beta header
    {"name": "defer_field_with_beta", "tools": [TOOL_TIME, TOOL_WEATHER_DEFERRED, TOOL_FILES_DEFERRED],
     "prompt": PROMPT_TIME, "beta": LEGACY_BETA},
    # 可见性探测：deferred 工具若被直接调用 => 字段被静默剥离/忽略
    {"name": "defer_visibility", "tools": [TOOL_TIME, TOOL_WEATHER_DEFERRED, TOOL_FILES_DEFERRED],
     "prompt": PROMPT_WEATHER, "beta": None},
    # 完整行为探测：search 工具 + deferred 工具，支持时应出现 tool_search_tool_result/tool_reference
    {"name": "tool_search_full", "tools": [TOOL_SEARCH_SERVER, TOOL_TIME, TOOL_WEATHER_DEFERRED, TOOL_FILES_DEFERRED],
     "prompt": PROMPT_WEATHER, "beta": None},
]


def mask_headers(headers: dict) -> dict:
    return {k: ("***" if k.lower() in ("x-api-key", "authorization") else v) for k, v in headers.items()}


def extract_signals(status: int, body: dict | None) -> dict:
    """从响应中提取判定所需的信号。"""
    sig = {
        "http_status": status,
        "error_message": None,
        "stop_reason": None,
        "content_block_types": [],
        "tool_use_names": [],
        "deferred_tool_called": False,
        "has_tool_search_result": False,
        "has_tool_reference": False,
        "usage": None,
        "text_preview": None,
    }
    if not isinstance(body, dict):
        return sig
    err = body.get("error")
    if isinstance(err, dict):
        sig["error_message"] = err.get("message") or json.dumps(err, ensure_ascii=False)
    elif body.get("type") == "error":
        sig["error_message"] = json.dumps(body, ensure_ascii=False)
    sig["stop_reason"] = body.get("stop_reason")
    sig["usage"] = body.get("usage")
    texts = []
    for block in body.get("content") or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        sig["content_block_types"].append(btype)
        if btype == "tool_use":
            sig["tool_use_names"].append(block.get("name"))
        if btype == "text":
            texts.append(block.get("text") or "")
    sig["deferred_tool_called"] = bool(set(sig["tool_use_names"]) & DEFERRED_NAMES)
    raw = json.dumps(body, ensure_ascii=False)
    sig["has_tool_search_result"] = "tool_search_tool_result" in raw or "tool_search_tool_search_result" in raw
    sig["has_tool_reference"] = '"tool_reference"' in raw
    sig["text_preview"] = (" ".join(texts))[:300] or None
    return sig


def run_case(client: httpx.Client, args, case: dict) -> dict:
    headers = {
        "x-api-key": args.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if case["beta"]:
        headers["anthropic-beta"] = case["beta"]
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,  # 思考类模型必须留够 output budget，否则只剩 thinking
        "messages": [{"role": "user", "content": case["prompt"]}],
        "tools": case["tools"],
    }
    url = args.base_url.rstrip("/") + "/v1/messages"
    t0 = time.time()
    record = {"case": case["name"], "request": {"url": url, "headers": mask_headers(headers), "body": payload}}
    try:
        resp = client.post(url, headers=headers, json=payload)
        elapsed = round(time.time() - t0, 2)
        try:
            resp_body = resp.json()
        except ValueError:
            resp_body = None
        record["response"] = {
            "http_status": resp.status_code,
            "headers": {k: v for k, v in resp.headers.items()
                        if k.lower() in ("content-type", "x-request-id", "request-id", "anthropic-organization-id")},
            "body": resp_body if resp_body is not None else resp.text,
            "elapsed_seconds": elapsed,
        }
        record["signals"] = extract_signals(resp.status_code, resp_body)
    except httpx.HTTPError as exc:
        record["response"] = None
        record["signals"] = {"http_status": None, "error_message": f"{type(exc).__name__}: {exc}"}
    return record


def print_case(rec: dict) -> None:
    s = rec["signals"]
    print(f"\n=== {rec['case']} ===")
    print(f"  HTTP {s['http_status']}  stop_reason={s.get('stop_reason')}")
    if s.get("error_message"):
        print(f"  ERROR: {s['error_message'][:500]}")
    if s.get("content_block_types"):
        print(f"  content blocks: {s['content_block_types']}")
    if s.get("tool_use_names"):
        print(f"  tool_use: {s['tool_use_names']}  (deferred 被直接调用: {s['deferred_tool_called']})")
    if s.get("has_tool_search_result") or s.get("has_tool_reference"):
        print(f"  tool_search_tool_result={s['has_tool_search_result']}  tool_reference={s['has_tool_reference']}")
    if s.get("usage"):
        print(f"  usage: {s['usage']}")
    if s.get("text_preview"):
        print(f"  text: {s['text_preview'][:200]}")


def summarize(records: list[dict]) -> str:
    by = {r["case"]: r["signals"] for r in records}
    lines = ["\n========== 结论 =========="]
    rejected = [n for n, s in by.items()
                if s.get("http_status") and s["http_status"] >= 400]
    field_rejected = [n for n in rejected if n.startswith("defer_field")]
    if field_rejected:
        lines.append(f"(a) 上游报错拒绝 defer_loading 字段：{field_rejected}")
        for n in field_rejected:
            lines.append(f"    {n}: {by[n].get('error_message', '')[:300]}")
    vis = by.get("defer_visibility", {})
    full = by.get("tool_search_full", {})
    if vis.get("deferred_tool_called") or (full.get("deferred_tool_called") and not full.get("has_tool_search_result")):
        lines.append("(b) 字段被静默剥离/忽略：deferred 工具对模型直接可见并被调用（无搜索过程）。")
    if full.get("has_tool_search_result") or full.get("has_tool_reference"):
        lines.append("(c) 上游支持延迟加载：响应出现 tool_search_tool_result / tool_reference 块。")
    if not field_rejected and not vis.get("deferred_tool_called") and not full.get("has_tool_search_result"):
        lines.append("字段未报错，但未观察到延迟加载行为证据（deferred 工具未被直接调用，也无搜索块）。")
        lines.append("可能：字段被接受但上游不支持 tool search 服务端工具；需看 tool_search_full 的具体响应。")
    if "tool_search_full" in rejected:
        lines.append(f"注意：tool_search_full 被拒绝（{by['tool_search_full'].get('error_message', '')[:200]}），"
                     "说明服务端搜索工具类型不被上游识别。")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--only", help="只运行指定 case（逗号分隔），默认全部")
    parser.add_argument("--output", default="test_defer_loading_results.json")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    cases = [c for c in CASES if only is None or c["name"] in only]
    print(f"探测目标: {args.base_url.rstrip('/')}/v1/messages  model={args.model}")
    print(f"cases: {[c['name'] for c in cases]}")

    records = []
    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        for case in cases:
            rec = run_case(client, args, case)
            records.append(rec)
            print_case(rec)

    summary = summarize(records)
    print(summary)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"target": {"base_url": args.base_url, "model": args.model},
                   "records": records, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(f"\n原始请求/响应已保存: {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())