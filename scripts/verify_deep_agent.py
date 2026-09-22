"""
scripts/verify_deep_agent.py

End-to-end verification script for Phase 1/2 of deepagents migration.
Validates:
1. app/deep_agent.py import and coexistence with app/graph.py
2. build_main_agent() compilation with all subagents & middleware
3. Hard schema validation: introspect task tool args_schema to confirm
   subagent_type only accepts the 9 declared expert names (no LLM call needed)
4. End-to-end ainvoke smoke test
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

async def main():
    report = {
        "coexistence_check": False,
        "build_agent_check": False,
        "schema_validation": False,
        "subagents_configured": [],
        "ainvoke_test": False,
        "response_sample": "",
        "errors": []
    }

    print("=" * 60, flush=True)
    print("[1/4] Testing coexistence of app.deep_agent and app.graph...", flush=True)
    try:
        from app import deep_agent
        from app import graph
        report["coexistence_check"] = True
        print("[+] SUCCESS: app.deep_agent and app.graph co-exist seamlessly!", flush=True)
    except Exception as e:
        report["errors"].append(f"Coexistence failed: {e}")
        print(f"[-] FAILED: Coexistence check failed: {e}", flush=True)
        return report

    print("=" * 60, flush=True)
    print("[2/4] Building main agent via deep_agent.build_main_agent()...", flush=True)
    try:
        agent = await deep_agent.build_main_agent()
        report["build_agent_check"] = True
        print(f"[+] SUCCESS: Main agent compiled successfully: {type(agent)}", flush=True)
        print(f"    Graph nodes: {list(agent.nodes.keys())}", flush=True)
    except Exception as e:
        report["errors"].append(f"Build agent failed: {e}")
        print(f"[-] FAILED: build_main_agent() failed: {e}", flush=True)
        return report

    print("=" * 60, flush=True)
    print("[3/4] Hard schema validation: inspecting task tool args_schema...", flush=True)
    try:
        # 从编译好的 agent 直接取 task 工具，不依赖模型文字生成
        task_tool = agent.get_tool("task")
        if task_tool is None:
            raise RuntimeError("agent.get_tool('task') returned None — task tool not found")

        schema = task_tool.args_schema
        if schema is None:
            raise RuntimeError("task tool has no args_schema")

        # 兼容 Pydantic v1 / v2：从 schema 元数据或 JSON schema 提取 subagent_type 的约束
        allowed_values: list | None = None

        # 方式 1：Pydantic v2 model_fields + annotation.__args__ (Literal)
        if hasattr(schema, "model_fields") and "subagent_type" in schema.model_fields:
            field_info = schema.model_fields["subagent_type"]
            ann = field_info.annotation
            if hasattr(ann, "__args__"):
                allowed_values = list(ann.__args__)

        # 方式 2：Pydantic v1 __fields__
        if allowed_values is None and hasattr(schema, "__fields__") and "subagent_type" in schema.__fields__:
            field = schema.__fields__["subagent_type"]
            outer = getattr(field, "outer_type_", None) or getattr(field, "annotation", None)
            if outer is not None and hasattr(outer, "__args__"):
                allowed_values = list(outer.__args__)

        # 方式 3：退化到 JSON schema enum（兜底，无论 Pydantic 版本）
        if allowed_values is None:
            try:
                if hasattr(schema, "model_json_schema"):
                    js = schema.model_json_schema()
                elif hasattr(schema, "schema"):
                    js = schema.schema()
                else:
                    js = {}
                props = js.get("properties", {})
                st = props.get("subagent_type", {})
                if "enum" in st:
                    allowed_values = list(st["enum"])
                elif "allOf" in st:
                    defs = js.get("$defs", {})
                    ref_name = st["allOf"][0].get("$ref", "").split("/")[-1]
                    allowed_values = list(defs.get(ref_name, {}).get("enum", []))
            except Exception as schema_err:
                raise RuntimeError(f"Failed to parse JSON schema: {schema_err}")

        if allowed_values is None:
            raise RuntimeError(
                "Could not extract allowed values for subagent_type from args_schema. "
                f"Schema type: {type(schema)}"
            )

        # 声明的 9 个专家名字（6 工具型 + 2 HITL 编译型 + 1 兜底覆盖）
        EXPECTED_SUBAGENTS = {
            "KnowledgeAgent",
            "MediaAgent",
            "MapAgent",
            "CodeAgent",
            "BrowserAgent",
            "SQLAgent",
            "FileAgent",
            "DesktopAgent",
            "general-purpose",
        }

        actual_set = set(allowed_values)
        print(f"    task.args_schema subagent_type allowed values ({len(actual_set)}):", flush=True)
        for name in sorted(actual_set):
            status = "✓" if name in EXPECTED_SUBAGENTS else "✗ UNEXPECTED"
            print(f"      {status}  {name}", flush=True)

        missing = EXPECTED_SUBAGENTS - actual_set
        extra   = actual_set - EXPECTED_SUBAGENTS

        if missing:
            raise AssertionError(f"Missing subagents in schema: {missing}")
        if extra:
            raise AssertionError(f"Unexpected subagents in schema: {extra}")

        report["schema_validation"] = True
        report["subagents_configured"] = sorted(actual_set)
        print(f"[+] SUCCESS: task tool schema contains exactly the expected {len(EXPECTED_SUBAGENTS)} subagents.", flush=True)

    except Exception as e:
        report["errors"].append(f"Schema validation failed: {e}")
        print(f"[-] FAILED: Schema validation failed: {e}", flush=True)
        return report

    print("=" * 60, flush=True)
    print("[4/4] Testing agent.ainvoke with thread_id checkpoint...", flush=True)
    try:
        test_query = "你好，简单介绍一下你自己。"
        messages = deep_agent.build_initial_messages(user_query=test_query)
        config_invoke = {"configurable": {"thread_id": "phase2_schema_verify_thread_1"}}
        result = await agent.ainvoke({"messages": messages}, config=config_invoke)

        last_message = result["messages"][-1]
        content = getattr(last_message, "content", str(last_message))
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and "text" in p]
            content = " ".join(text_parts) if text_parts else str(content)

        report["ainvoke_test"] = True
        report["response_sample"] = content[:300]
        print("[+] SUCCESS: ainvoke returned final response!", flush=True)
        print(f"    Response snippet: {content[:200]}...", flush=True)
    except Exception as e:
        report["errors"].append(f"ainvoke failed: {e}")
        print(f"[-] FAILED: agent.ainvoke failed: {e}", flush=True)

    # Save verification report
    scratch_dir = Path(__file__).resolve().parent.parent / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    report_file = scratch_dir / "verify_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[*] Verification report saved to {report_file}", flush=True)

    return report

if __name__ == "__main__":
    asyncio.run(main())
