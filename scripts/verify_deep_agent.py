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
    print("[3/4] Hard schema validation: inspecting task tool args_schema & subagents...", flush=True)
    try:
        # 从编译图的 tools 节点中提取 task 工具
        tools_node = agent.nodes.get("tools")
        if tools_node is None:
            raise RuntimeError("Graph has no 'tools' node")

        tools_by_name = getattr(tools_node, "tools_by_name", None)
        if tools_by_name is None and hasattr(tools_node, "bound"):
            tools_by_name = getattr(tools_node.bound, "tools_by_name", None)

        if not tools_by_name or "task" not in tools_by_name:
            raise RuntimeError(f"task tool not found in tools node. Available: {list(tools_by_name.keys()) if tools_by_name else None}")

        task_tool = tools_by_name["task"]
        print(f"[+] Found task tool: {task_tool.name}", flush=True)

        schema = task_tool.args_schema
        if schema is None:
            raise RuntimeError("task tool has no args_schema")

        # 打印 args_schema 的 JSON Schema 定义
        if hasattr(schema, "model_json_schema"):
            schema_json = schema.model_json_schema()
        elif hasattr(schema, "schema"):
            schema_json = schema.schema()
        else:
            schema_json = str(schema)
        print(f"    task.args_schema: {schema}", flush=True)
        print(f"    task.args_schema JSON: {schema_json}", flush=True)

        # 提取 subagent_type 实际允许的取值列表：
        # 1. 优先从 task 工具底层闭包的 subagent_graphs 获取（实际运行时路由表）
        allowed_values = None
        coro = getattr(task_tool, "coroutine", None) or getattr(task_tool, "func", None)
        if coro and hasattr(coro, "__code__") and hasattr(coro, "__closure__") and coro.__closure__:
            closure_dict = dict(zip(coro.__code__.co_freevars, [c.cell_contents for c in coro.__closure__]))
            if "subagent_graphs" in closure_dict:
                allowed_values = list(closure_dict["subagent_graphs"].keys())

        # 2. 如果闭包不可用，从 tool description 的 Available agent types 列表解析
        if not allowed_values and task_tool.description:
            import re
            matches = re.findall(r"^-\s*([A-Za-z0-9_-]+):", task_tool.description, flags=re.MULTILINE)
            if matches:
                allowed_values = matches

        # 3. 如果 schema 有 Literal / Enum 约束，也一并提取
        if hasattr(schema, "model_fields") and "subagent_type" in schema.model_fields:
            field_info = schema.model_fields["subagent_type"]
            ann = field_info.annotation
            if hasattr(ann, "__args__") and ann.__args__:
                allowed_values = list(ann.__args__)

        if allowed_values is None:
            raise RuntimeError("Could not extract allowed values for subagent_type from task tool")

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
        print(f"    task tool subagent_type allowed values ({len(actual_set)}):", flush=True)
        for name in sorted(actual_set):
            status = "✓" if name in EXPECTED_SUBAGENTS else "✗ UNEXPECTED"
            print(f"      {status}  {name}", flush=True)

        missing = EXPECTED_SUBAGENTS - actual_set
        extra   = actual_set - EXPECTED_SUBAGENTS

        if missing:
            raise AssertionError(f"Missing subagents in task tool: {missing}")
        if extra:
            raise AssertionError(f"Unexpected subagents in task tool: {extra}")

        report["schema_validation"] = True
        report["subagents_configured"] = sorted(actual_set)
        print(f"[+] SUCCESS: task tool contains exactly the expected {len(EXPECTED_SUBAGENTS)} subagents.", flush=True)

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
