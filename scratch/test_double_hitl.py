import asyncio
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent, CompiledSubAgent, register_harness_profile, HarnessProfile, GeneralPurposeSubagentProfile
from app import models, mcp_service, deep_agent, graph
import config as app_config

# Ensure general-purpose is disabled
register_harness_profile(
    "google_genai",
    HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)

class SubgraphCompiledWrapper(Runnable):
    def __init__(self, subgraph):
        self.subgraph = subgraph

    def invoke(self, state, config=None, **kwargs):
        raise NotImplementedError("Use async ainvoke")

    async def ainvoke(self, state, config=None, **kwargs):
        msgs = state.get("messages", [])
        description = ""
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                if isinstance(m.content, str):
                    description = m.content
                elif isinstance(m.content, list):
                    description = " ".join(p.get("text", "") for p in m.content if isinstance(p, dict) and "text" in p)
                break

        subgraph_input = {
            "messages": msgs,
            "current_tool_call_id": state.get("current_tool_call_id") or "call_subgraph",
            "instruction_to_worker": state.get("instruction_to_worker") or description,
            "user_query": state.get("user_query") or description,
        }

        result = await self.subgraph.ainvoke(subgraph_input, config)
        
        res_msgs = list(result.get("messages", []))
        if res_msgs and isinstance(res_msgs[-1], ToolMessage):
            final_content = res_msgs[-1].content
            res_msgs.append(AIMessage(content=final_content))
            result["messages"] = res_msgs

        return result

async def main():
    print("=" * 60)
    print("Starting Multi-Interrupt HITL Test Case (Full Loop)")
    print("=" * 60)

    # 1. Setup workspace test files
    workspace = Path(app_config.WORKSPACE_DIR)
    workspace.mkdir(parents=True, exist_ok=True)
    f1 = workspace / "test_hitl1.txt"
    f2 = workspace / "test_hitl2.txt"
    trash = workspace / "archive_trash"
    trash.mkdir(parents=True, exist_ok=True)
    dest_f2 = trash / "test_hitl2.txt"
    if dest_f2.exists():
        dest_f2.unlink()

    f1.write_text("original 1", encoding="utf-8")
    f2.write_text("original 2", encoding="utf-8")
    print(f"[*] Prepared files:\n    - {f1}\n    - {f2}\n    - {trash}")

    # 2. Init MCP & Build FileAgent Subgraph
    await mcp_service.initialize_mcp()
    filesystem_tools = mcp_service.get_tools_by_server("filesystem")

    file_prompt = (
        deep_agent.WORKER_BASE_PROMPT
        + "你是 FileAgent，负责操作本地文件系统（Filesystem MCP）。\n"
        + "你可以读取文件内容、查看目录结构、搜索文件、写入/编辑文件、移动文件等。\n"
        + "【安全红线 - 严重警告】\n"
        + f"  · 所有操作必须限定在项目工作区内：{app_config.WORKSPACE_DIR}\n"
        + "  · 如果你在执行危险操作时触发了 HITL 授权，且用户返回了 'DENIED' 或拒绝信息，"
        + "    你必须立即停止对该工具的尝试。严禁循环请求同一个工具！\n"
        + "【关于删除操作的特殊说明】\n"
        + "  · 当前工具集不直接提供 delete_file 工具。如果主管要求你“删除”文件，"
        + "    你必须使用 move_file 工具将目标文件移动到项目根目录下的 'archive_trash' 文件夹中。"
    )

    file_agent_subgraph = graph.build_file_agent_subgraph(
        filesystem_tools=filesystem_tools,
        system_prompt=file_prompt,
    )

    file_subagent = CompiledSubAgent(
        name="FileAgent",
        description="本地文件专家，读取/写入/编辑项目内文件，涉及危险操作（写入/编辑/移动）会暂停并请求人工授权。",
        runnable=SubgraphCompiledWrapper(file_agent_subgraph),
    )

    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=models.supervisor_llm,
        system_prompt=deep_agent.MAIN_AGENT_PROMPT,
        subagents=[file_subagent],
        checkpointer=checkpointer,
    )

    cfg = {"configurable": {"thread_id": "test_double_hitl_loop"}}

    test_prompt = (
        "请帮我完成以下两件事：\n"
        "1. 把 test_hitl1.txt 的内容写成 'hello'\n"
        "2. 把 test_hitl2.txt 移动到 archive_trash/test_hitl2.txt"
    )

    print(f"\n[Step 0] Sending initial instruction:\n{test_prompt}\n")
    await agent.ainvoke({"messages": [HumanMessage(content=test_prompt)]}, config=cfg)

    step = 0
    while step < 5:
        step += 1
        state = agent.get_state(cfg)
        print(f"\n--- Turn {step} Check ---")
        print(f"[*] state.next = {state.next}")

        interrupts = []
        for t in state.tasks:
            if hasattr(t, "interrupts") and t.interrupts:
                interrupts.extend(t.interrupts)

        print(f"[*] Interrupts pending: {len(interrupts)}")
        for idx, intr in enumerate(interrupts):
            val = intr.value if hasattr(intr, "value") else intr
            print(f"    [{idx}] {val}")

        if not state.next:
            print("[+] No further interrupts. Graph finished!")
            break

        print(f"[>] Resuming interrupt {step} with Command(resume='Y')...")
        await agent.ainvoke(Command(resume="Y"), config=cfg)

    # Verification
    state_final = agent.get_state(cfg)
    print("\n" + "=" * 60)
    print("Verification Results:")
    print("=" * 60)
    c1 = f1.read_text(encoding="utf-8")
    print(f"test_hitl1.txt content: '{c1}' (expected 'hello') -> {'PASS' if 'hello' in c1 else 'FAIL'}")
    f2_moved = dest_f2.exists()
    print(f"archive_trash/test_hitl2.txt exists: {f2_moved} -> {'PASS' if f2_moved else 'FAIL'}")

    last_msg = state_final.values.get("messages", [])[-1] if state_final.values else None
    print(f"\n[Final Agent Response]:")
    print(getattr(last_msg, "content", str(last_msg)))

if __name__ == "__main__":
    asyncio.run(main())
