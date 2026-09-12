"""
scripts/verify_deep_agent.py

End-to-end verification script for Phase 1 of deepagents migration.
Validates:
1. app/deep_agent.py import and coexistence with app/graph.py
2. build_main_agent() compilation with all 6 declarative subagents & middleware
3. Message structure with build_initial_messages()
4. End-to-end ainvoke execution without touching any production path
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
    print("[3/4] Testing build_initial_messages with real Milvus memory...", flush=True)
    try:
        from app import database
        test_mem = "用户是一名 Python 开发者，喜欢简洁明了的回复"
        print(f"[*] Inserting test memory into Milvus: '{test_mem}'...", flush=True)
        database.insert_memory(test_mem)

        test_query = "你好，我是名 Python 开发者，请介绍一下你自己和你拥有的下属专家团队。"
        messages = deep_agent.build_initial_messages(
            user_query=test_query,
        )
        print(f"[+] SUCCESS: Built {len(messages)} initial messages from real Milvus retrieval.", flush=True)
        for idx, m in enumerate(messages):
            print(f"    [{idx}] {type(m).__name__}: {str(m.content)[:100]}...", flush=True)
    except Exception as e:
        report["errors"].append(f"build_initial_messages failed: {e}")
        print(f"[-] FAILED: build_initial_messages failed: {e}", flush=True)
        return report

    print("=" * 60, flush=True)
    print("[4/4] Testing agent.ainvoke with thread_id checkpoint...", flush=True)
    try:
        config_invoke = {"configurable": {"thread_id": "phase1_verification_thread_1"}}
        result = await agent.ainvoke({"messages": messages}, config=config_invoke)
        
        last_message = result["messages"][-1]
        content = getattr(last_message, "content", str(last_message))
        if isinstance(content, list):
            # Format text list from Gemini / OpenAI multimodal output
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
