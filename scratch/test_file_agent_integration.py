import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from deepagents import create_deep_agent, CompiledSubAgent, register_harness_profile, HarnessProfile, GeneralPurposeSubagentProfile
from langgraph.checkpoint.memory import MemorySaver
from app import models, mcp_service, deep_agent, graph

# Ensure general-purpose is disabled
register_harness_profile(
    "google_genai",
    HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)

class SubgraphCompiledWrapper(Runnable):
    """
    Adapter wrapper to bridge deepagents task tool invocation to legacy compiled subgraphs.
    Ensures state fields (current_tool_call_id, instruction_to_worker, user_query)
    are populated, and converts the final ToolMessage report to an AIMessage so that
    deepagents extracts the full report content.
    """
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
        
        # deepagents looks for the last AIMessage with text in result['messages']
        # If the subgraph ended with a ToolMessage containing the final report,
        # ensure there is an AIMessage with that content.
        res_msgs = list(result.get("messages", []))
        if res_msgs and isinstance(res_msgs[-1], ToolMessage):
            final_content = res_msgs[-1].content
            res_msgs.append(AIMessage(content=final_content))
            result["messages"] = res_msgs

        return result

async def test_file_agent_simple_read():
    await mcp_service.initialize_mcp()
    filesystem_tools = mcp_service.get_tools_by_server("filesystem")
    
    file_prompt = (
        deep_agent.WORKER_BASE_PROMPT
        + "你是 FileAgent，负责操作本地文件系统（Filesystem MCP）。\n"
        + "你可以读取文件内容、查看目录结构、搜索文件、写入/编辑文件、移动文件等。"
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
    
    print("[*] Testing simple read: '读一下 workspace 下 1.txt 的内容'...")
    config_invoke = {"configurable": {"thread_id": "test_thread_read"}}
    messages = [HumanMessage(content="读一下 workspace 下 1.txt 的内容")]
    res = await agent.ainvoke({"messages": messages}, config=config_invoke)
    print("[+] Simple read result:")
    last_msg = res["messages"][-1]
    print(last_msg.content)

if __name__ == "__main__":
    asyncio.run(test_file_agent_simple_read())
