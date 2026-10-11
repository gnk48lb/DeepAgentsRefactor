import asyncio
import sys
import json
from pathlib import Path

# Unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_google_genai import ChatGoogleGenerativeAI
from app import models, deep_agent, database
from app.utils import extract_text

# Monkeypatch ChatGoogleGenerativeAI.bind_tools
orig_bind_tools = ChatGoogleGenerativeAI.bind_tools

def patched_bind_tools(self, tools, **kwargs):
    tool_names = []
    for t in tools:
        if isinstance(t, dict):
            name = t.get("name") or t.get("function", {}).get("name")
            tool_names.append(name)
        elif hasattr(t, "name"):
            tool_names.append(t.name)
        else:
            tool_names.append(str(t))
    
    role = "Main Agent (Supervisor)" if self is models.supervisor_llm else ("CodeAgent / Worker" if self is models.worker_llm else f"Other ({getattr(self, 'model', 'unknown')})")
    model_name = getattr(self, "model", "unknown")
    print(f"\n>>> [bind_tools 调用拦截] 角色: {role} | 模型: {model_name}", flush=True)
    print(f"    传入工具列表 ({len(tool_names)} 个): {tool_names}", flush=True)
    return orig_bind_tools(self, tools, **kwargs)

ChatGoogleGenerativeAI.bind_tools = patched_bind_tools


async def run_step(agent, query: str, thread_id: str):
    print("\n" + "=" * 70, flush=True)
    print(f"执行用户指令: \"{query}\" (thread_id={thread_id})", flush=True)
    print("=" * 70, flush=True)
    
    cfg = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": deep_agent.build_initial_messages(user_query=query, user_id="test_user")}
    
    subagent_tool_calls = []
    final_text = ""
    
    async for event in agent.astream(inputs, config=cfg):
        if not isinstance(event, dict):
            continue
        for node_name, state_update in event.items():
            if "Middleware" in node_name or node_name.startswith("__"):
                continue
            if not isinstance(state_update, dict):
                continue
            msgs = state_update.get("messages", [])
            if not isinstance(msgs, list):
                msgs = [msgs]
            for m in msgs:
                m_type = type(m).__name__
                if m_type == "AIMessage":
                    tc = getattr(m, "tool_calls", None) or []
                    for c in tc:
                        c_name = c.get("name") if isinstance(c, dict) else getattr(c, "name", "")
                        c_args = c.get("args") if isinstance(c, dict) else getattr(c, "args", {})
                        if c_name == "task":
                            sub_type = c_args.get("subagent_type")
                            desc = c_args.get("description")
                            print(f"  [主 Agent 调用 task] 委派给: {sub_type}, 描述: {desc}", flush=True)
                        else:
                            print(f"  [Agent 工具调用] 节点: {node_name}, 工具: {c_name}, 参数: {str(c_args)[:120]}", flush=True)
                    if not tc and getattr(m, "content", ""):
                        final_text = extract_text(m.content)
                elif m_type == "ToolMessage":
                    tool_name = getattr(m, "name", "unknown")
                    content = extract_text(getattr(m, "content", ""))
                    if tool_name == "task":
                        print(f"  [task 返回战报]: {content[:200]}...", flush=True)
                    else:
                        print(f"  [ToolMessage 来自 {tool_name}]: {content[:150]}", flush=True)
                        subagent_tool_calls.append(tool_name)

    print(f"\n[最终回答]: {final_text.strip()}", flush=True)
    return subagent_tool_calls, final_text


async def main():
    database.run_global_database_init()
    agent = await deep_agent.build_main_agent()
    
    print("\n" + "#" * 70, flush=True)
    print("【测试 (a)】主 Agent 普通回答: '1+1等于几'", flush=True)
    print("#" * 70, flush=True)
    await run_step(agent, "1+1等于几", "diag_step_a")
    
    print("\n" + "#" * 70, flush=True)
    print("【测试 (b)】委派 CodeAgent: '用 Python 计算 1+1 并告诉我结果'", flush=True)
    print("#" * 70, flush=True)
    calls, _ = await run_step(agent, "用 Python 计算 1+1 并告诉我结果", "diag_step_b")
    print(f"\n[CodeAgent 这次实际发起的工具调用名]: {calls}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
