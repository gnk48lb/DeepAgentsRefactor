# Windows os.rename compatibility monkey-patch for milvus-lite FileExistsError (WinError 183)
import sys
import os
import logging

# Suppress all third-party verbose INFO logs immediately
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("milvus_lite").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("pymilvus").setLevel(logging.WARNING)

if sys.platform == "win32":
    _orig_rename = os.rename
    def _safe_rename(src, dst):
        try:
            _orig_rename(src, dst)
        except (FileExistsError, OSError) as e:
            try:
                os.replace(src, dst)
            except Exception:
                raise e
    os.rename = _safe_rename

from app import deep_agent
from app import database
import config
import asyncio
import traceback
import json
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from app.utils import extract_text as _extract_text



def _truncate(text: str, limit: int = 800) -> str:
    if len(text) > limit:
        return text[:limit] + "\n... [内容已截断]"
    return text


# ============================================================================
# 基于真实 astream 事件结构的控制台展示逻辑
# ============================================================================
#
# 事件形态（通过实际 astream dump 确认）：
#
# 1. Node: PatchToolCallsMiddleware.before_agent  → raw: None → 跳过
# 2. Node: model  → AIMessage
#    - 有 tool_calls（如 task）  → 主 Agent 委派给子专家
#    - 有实际 content、无 tool_calls → 主 Agent 最终回复
# 3. Node: ToolCallLimitMiddleware[task].after_model → 跳过
# 4. Node: tools  → ToolMessage(name="task") → 子专家战报返回
# ============================================================================

async def run_agent(query: str, agent, config_dict: dict) -> str:
    """运行 DeepAgent 并在控制台打印彩色流式输出。
    返回最终回复文本（用于记忆提取）。"""
    messages = deep_agent.build_initial_messages(
        user_query=query,
        user_id="console_user",
    )
    inputs = {"messages": messages}

    print(f"\n--- 🚀 Running DeepAgent for: '{query}' ---")

    final_text = ""
    try:
        async for event in agent.astream(inputs, config=config_dict):
            if not isinstance(event, dict):
                continue

            for node_name, state_update in event.items():
                # ---- 跳过中间件节点 ----
                if "Middleware" in node_name or node_name.startswith("__"):
                    continue

                if not isinstance(state_update, dict):
                    continue

                msgs = state_update.get("messages", [])
                if not isinstance(msgs, list):
                    msgs = [msgs]

                for m in msgs:
                    m_type = type(m).__name__
                    tool_calls = getattr(m, "tool_calls", None) or []
                    content = getattr(m, "content", "")
                    text = _extract_text(content)

                    # ---- Node: model ----
                    if node_name == "model" and m_type == "AIMessage":
                        if tool_calls:
                            # 主 Agent 正在委派任务
                            for tc in tool_calls:
                                tc_name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                                tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})

                                if tc_name == "task":
                                    subagent = tc_args.get("subagent_type", "?")
                                    desc = tc_args.get("description", "")
                                    print(f"\n🔵 \033[94m[派发任务给 {subagent}]\033[0m: {desc}")
                                else:
                                    # 其他工具调用（deepagents 内置的 ls/grep/execute 等）
                                    args_str = json.dumps(tc_args, ensure_ascii=False)[:150]
                                    print(f"\n🔧 \033[95m[调用工具 {tc_name}]\033[0m: {args_str}")
                        elif text.strip():
                            # 主 Agent 最终回复
                            final_text = text
                            print(f"\n🌟 \033[96m[最终回答]\033[0m: {_truncate(text)}\n")

                    # ---- Node: tools ----
                    elif node_name == "tools" and m_type == "ToolMessage":
                        tool_name = getattr(m, "name", "unknown")
                        display = _truncate(text)

                        if tool_name == "task":
                            # 子专家战报
                            print(f"\n🟡 \033[93m[专家战报]\033[0m:\n{display}")
                        else:
                            # 其他工具返回
                            print(f"\n🟢 \033[92m[{tool_name} 返回]\033[0m:\n{display}")

    except Exception as e:
        print(f"\n❌ \033[91m[Agent 执行异常/中断]\033[0m: {str(e)}")
        traceback.print_exc()

    # ---- 中断处理：FileAgent HITL ----
    # astream 结束后检查 state.next，非空说明子图触发了 interrupt()
    try:
        state = await agent.aget_state(config_dict)
        while state.next:
            # 从 state.tasks[].interrupts 提取中断 payload
            interrupts = []
            for t in state.tasks:
                if hasattr(t, "interrupts") and t.interrupts:
                    interrupts.extend(t.interrupts)

            for intr in interrupts:
                val = intr.value if hasattr(intr, "value") else intr
                # val 是 FileAgent 自定义的授权请求消息（字符串或 dict）
                if isinstance(val, dict):
                    tool_name_h = val.get("tool_name", "未知操作")
                    tool_args_h = val.get("tool_args", {})
                    print(f"\n⚠️  \033[91m[HITL 授权请求]\033[0m: {tool_name_h}")
                    print(f"    参数: {json.dumps(tool_args_h, ensure_ascii=False)[:300]}")
                else:
                    print(f"\n⚠️  \033[91m[HITL 授权请求]\033[0m:\n{val}")

            answer = await asyncio.to_thread(input, "\n✋ 是否授权此操作？[Y/N]: ")
            answer = answer.strip().upper() or "N"
            resume_val = "Y" if answer == "Y" else "N"

            print(f"\n▶️  \033[90m[Resume → {resume_val}]\033[0m")
            # 继续 astream 处理恢复后的事件流
            async for event in agent.astream(Command(resume=resume_val), config=config_dict):
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
                        tool_calls = getattr(m, "tool_calls", None) or []
                        content = getattr(m, "content", "")
                        text = _extract_text(content)
                        if node_name == "model" and m_type == "AIMessage":
                            if tool_calls:
                                for tc in tool_calls:
                                    tc_name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                                    tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                    if tc_name == "task":
                                        subagent = tc_args.get("subagent_type", "?")
                                        desc = tc_args.get("description", "")
                                        print(f"\n🔵 \033[94m[派发任务给 {subagent}]\033[0m: {desc}")
                                    else:
                                        args_str = json.dumps(tc_args, ensure_ascii=False)[:150]
                                        print(f"\n🔧 \033[95m[调用工具 {tc_name}]\033[0m: {args_str}")
                            elif text.strip():
                                final_text = text
                                print(f"\n🌟 \033[96m[最终回答]\033[0m: {_truncate(text)}\n")
                        elif node_name == "tools" and m_type == "ToolMessage":
                            tool_name = getattr(m, "name", "unknown")
                            display = _truncate(text)
                            if tool_name == "task":
                                print(f"\n🟡 \033[93m[专家战报]\033[0m:\n{display}")
                            else:
                                print(f"\n🟢 \033[92m[{tool_name} 返回]\033[0m:\n{display}")

            state = await agent.aget_state(config_dict)
    except Exception as hitl_err:
        print(f"⚠️ \033[93m[HITL 检查异常]\033[0m: {hitl_err}")

    return final_text


async def main_loop():
    try:
        print("启动 DeepAgents Multi-Agent 架构...")

        # 启动前检测命令行参数
        if "--rebuild" in sys.argv:
            config.REFRESH_COLLECTION = True
            print("🚀 检测到 --rebuild 参数，将强制重建知识库索引！")

        # 数据库全家桶一键初始化（build_main_agent 不包含这一步）
        database.run_global_database_init()

        # 构建 deepagents 主 Agent（内部会初始化 MCP）
        agent = await deep_agent.build_main_agent()

        config_dict = {
            "recursion_limit": 30,
            "configurable": {"thread_id": "console_deepagent_user"},
        }

        print("\n" + "="*50)
        print("欢迎使用 GNK48-Agent ！(DeepAgents Version)")
        print("输入 'exit' 退出程序。")
        print("="*50 + "\n")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n您的问题: ")
                user_input = user_input.strip()

                # 兼容不同系统终端编码
                try:
                    user_input = user_input.encode("utf-8", "ignore").decode("utf-8")
                except Exception:
                    pass

                if not user_input:
                    continue

                if user_input.lower() in ['exit']:
                    print("👋 感谢使用，再见！")
                    break

                final_text = await run_agent(user_input, agent, config_dict)

                # ---- 记忆提取：只取本轮的用户消息 + 最终回复 ----
                # 从 state 取消息列表，找本轮最后一条 HumanMessage 和最后一条 AIMessage
                # 不取 ToolMessage / SystemMessage，避免混入工具调用和注入的记忆
                try:
                    final_state = await agent.aget_state(config_dict)
                    all_msgs = final_state.values.get("messages", [])

                    # 本轮 HumanMessage（最后一条）
                    last_human = ""
                    for m in reversed(all_msgs):
                        if isinstance(m, HumanMessage):
                            last_human = _extract_text(getattr(m, "content", ""))
                            break

                    # 本轮最终回复用 run_agent 返回的 final_text
                    last_ai = final_text

                    if last_human and last_ai:
                        recent_chat = f"human: {last_human}\nai: {last_ai}"
                        asyncio.create_task(deep_agent._background_extract_memory(recent_chat))
                except Exception as mem_err:
                    print(f"⚠️ \033[93m[记忆提取跳过]\033[0m: {mem_err}")

            except KeyboardInterrupt:
                print("\n👋 感谢使用，再见！")
                break
            except Exception as e:
                traceback.print_exc()
                print(f"处理问题时出错: {e}")

    except Exception as e:
        traceback.print_exc()
        print(f"系统启动失败: {e}")

if __name__ == "__main__":
    asyncio.run(main_loop())
