import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

async def main():
    results = {}
    
    # 1. Check deepagents
    try:
        import deepagents
        results["deepagents_version"] = getattr(deepagents, "__version__", "unknown")
        print(f"[*] deepagents version: {results['deepagents_version']}", flush=True)
    except Exception as e:
        results["deepagents_error"] = str(e)
        print(f"[!] Failed to import deepagents: {e}", flush=True)
        return

    # 2. Check ToolCallLimitMiddleware
    try:
        from langchain.agents.middleware import ToolCallLimitMiddleware
        results["has_tool_call_limit_middleware"] = True
        print("[*] ToolCallLimitMiddleware: available in langchain.agents.middleware", flush=True)
    except ImportError:
        results["has_tool_call_limit_middleware"] = False
        print("[-] ToolCallLimitMiddleware: NOT available in langchain.agents.middleware", flush=True)

    # 3. Check models & ChatGoogleGenerativeAI
    try:
        print("[*] Importing app.models...", flush=True)
        from app import models
        print(f"[*] models.worker_llm: {models.worker_llm}", flush=True)
        print(f"[*] models.vlm: {models.vlm}", flush=True)
    except Exception as e:
        results["models_error"] = str(e)
        print(f"[!] Failed to import app.models: {e}", flush=True)
        return

    # 4. Check create_deep_agent with worker_llm
    try:
        print("[*] Calling create_deep_agent...", flush=True)
        agent = deepagents.create_deep_agent(
            model=models.worker_llm,
            system_prompt="You are a test assistant. Reply strictly with 'SMOKE_TEST_OK'.",
        )
        print("[*] create_deep_agent successfully created agent instance", flush=True)
        
        # Test basic ainvoke
        print("[*] Invoking agent.ainvoke...", flush=True)
        resp = await agent.ainvoke({"messages": [{"role": "user", "content": "Please reply strictly with SMOKE_TEST_OK."}]})
        last_msg = resp["messages"][-1]
        content = getattr(last_msg, "content", str(last_msg))
        results["response"] = content
        print(f"[*] ainvoke test response: {content}", flush=True)
        results["status"] = "SUCCESS"
    except Exception as e:
        results["ainvoke_error"] = str(e)
        results["status"] = "FAILED"
        print(f"[!] ainvoke test failed: {e}", flush=True)

    # Save baseline results
    scratch_dir = Path(__file__).resolve().parent.parent / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = scratch_dir / "smoke_test_baseline.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[*] Smoke test baseline saved to {baseline_path}")

if __name__ == "__main__":
    asyncio.run(main())
