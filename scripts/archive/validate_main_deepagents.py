"""
scripts/validate_main_deepagents.py

验收脚本：
  1. KnowledgeAgent 知识型问题（游戏攻略）
  2. 天气问题重跑（贴原始输出，确认"夜间多云/多复"来源）
  3. 同 thread_id 两轮连续问（上下文接力）
  4. FileAgent HITL 全流程（自动 Y）
  5a. KnowledgeAgent 含图片 RAG chunk（是否报 400）
  5b. CodeAgent 画图（图片是否出现在 result messages）
"""
import sys
import os
import logging
import asyncio
import json
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from app import deep_agent, database
import config as app_config


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text", "")
                if isinstance(t, list):
                    text_parts.append(" ".join(
                        str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in t
                    ))
                else:
                    text_parts.append(str(t))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)
    return str(content)


async def run_and_print(query: str, agent, config_dict: dict, label: str) -> str:
    """非交互模式运行 astream，打印完整原始输出，返回最终文本。"""
    print(f"\n{'='*70}")
    print(f"[{label}] query: {query}")
    print(f"{'='*70}")

    messages = deep_agent.build_initial_messages(user_query=query, user_id="test")
    inputs = {"messages": messages}

    final_text = ""
    async for event in agent.astream(inputs, config=config_dict):
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
                                print(f"\n🔵 [派发任务给 {subagent}]: {desc}")
                            else:
                                print(f"\n🔧 [调用工具 {tc_name}]: {json.dumps(tc_args, ensure_ascii=False)[:150]}")
                    elif text.strip():
                        final_text = text
                        print(f"\n🌟 [最终回答]: {text}\n")
                elif node_name == "tools" and m_type == "ToolMessage":
                    tool_name = getattr(m, "name", "unknown")
                    if tool_name == "task":
                        print(f"\n🟡 [专家战报]:\n{text}")
                    else:
                        print(f"\n🟢 [{tool_name} 返回]:\n{text[:500]}")

    return final_text


async def run_hitl_auto(query: str, agent, config_dict: dict, label: str):
    """HITL 测试：自动回答 Y，打印完整流程。"""
    print(f"\n{'='*70}")
    print(f"[{label}] query: {query}")
    print(f"{'='*70}")

    messages = deep_agent.build_initial_messages(user_query=query, user_id="test")
    inputs = {"messages": messages}

    async def _stream_events(stream):
        async for event in stream:
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
                                    print(f"\n🔵 [派发任务给 {tc_args.get('subagent_type','?')}]: {tc_args.get('description','')}")
                                else:
                                    print(f"\n🔧 [调用工具 {tc_name}]: {json.dumps(tc_args, ensure_ascii=False)[:150]}")
                        elif text.strip():
                            print(f"\n🌟 [最终回答]: {text}\n")
                    elif node_name == "tools" and m_type == "ToolMessage":
                        tool_name = getattr(m, "name", "unknown")
                        if tool_name == "task":
                            print(f"\n🟡 [专家战报]:\n{text}")
                        else:
                            print(f"\n🟢 [{tool_name} 返回]:\n{text[:500]}")

    await _stream_events(agent.astream(inputs, config=config_dict))

    # HITL 循环
    step = 0
    while step < 10:
        step += 1
        state = await agent.aget_state(config_dict)
        if not state.next:
            print(f"\n[HITL] state.next 为空，图执行完毕。")
            break

        interrupts = []
        for t in state.tasks:
            if hasattr(t, "interrupts") and t.interrupts:
                interrupts.extend(t.interrupts)

        print(f"\n[HITL Turn {step}] state.next={state.next}, interrupts={len(interrupts)}")
        for idx, intr in enumerate(interrupts):
            val = intr.value if hasattr(intr, "value") else intr
            print(f"  [{idx}] interrupt.value = {val}")

        print(f"[HITL Turn {step}] → 自动回答 Y")
        await _stream_events(agent.astream(Command(resume="Y"), config=config_dict))


async def run_code_agent_image(agent, config_dict: dict):
    """调查 5b：CodeAgent 画图，检查图片是否出现在 result messages。"""
    print(f"\n{'='*70}")
    print("[调查 5b] CodeAgent 画图：检查图片是否在最终 result messages")
    print(f"{'='*70}")

    query = "用 Python 画一个简单的折线图，x 是 [1,2,3,4,5]，y 是 [2,4,1,5,3]，保存到 /workspace/outputs/ 目录下。"
    messages = deep_agent.build_initial_messages(user_query=query, user_id="test")
    inputs = {"messages": messages}

    result = await agent.ainvoke(inputs, config=config_dict)
    all_msgs = result.get("messages", [])
    print(f"\n全部 messages 数量: {len(all_msgs)}")
    for i, m in enumerate(all_msgs):
        mt = type(m).__name__
        content = getattr(m, "content", "")
        if isinstance(content, list):
            has_image = any(isinstance(c, dict) and c.get("type") == "image_url" for c in content)
            text_only = all(isinstance(c, dict) and c.get("type") == "text" for c in content if isinstance(c, dict))
            print(f"  [{i}] {mt}: list({len(content)} parts, has_image={has_image})")
        elif isinstance(content, str):
            print(f"  [{i}] {mt}: str({len(content)} chars) = {content[:150]!r}")
        else:
            print(f"  [{i}] {mt}: {type(content).__name__}")

    # 检查是否有 image_url 类型的内容
    found_images = []
    for m in all_msgs:
        content = getattr(m, "content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {})
                    found_images.append(str(url)[:100])

    if found_images:
        print(f"\n✅ 发现图片 ({len(found_images)} 张): {found_images}")
    else:
        print(f"\n❌ result['messages'] 里没有 image_url 类型内容（图片不在主 Agent 消息流中）")

    # 也打印最终回复
    for m in reversed(all_msgs):
        if isinstance(m, AIMessage):
            print(f"\n最终 AIMessage: {_extract_text(getattr(m,'content',''))[:300]}")
            break


async def main():
    database.run_global_database_init()
    agent = await deep_agent.build_main_agent()

    # =========================================================
    # 1. KnowledgeAgent 知识型问题（英魂之刃英雄攻略）
    # =========================================================
    cfg1 = {"recursion_limit": 30, "configurable": {"thread_id": "test_knowledge"}}
    try:
        await run_and_print("火神战姬的技能有哪些？", agent, cfg1, "验收1 KnowledgeAgent")
    except Exception as e:
        print(f"\n❌ [验收1 异常]: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 2. 天气重跑（直接 print，不做任何整理）
    # =========================================================
    cfg2 = {"recursion_limit": 30, "configurable": {"thread_id": "test_weather_rerun"}}
    try:
        await run_and_print("济南今天天气怎么样", agent, cfg2, "验收2 天气重跑")
    except Exception as e:
        print(f"\n❌ [验收2 异常]: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 3. 同 thread_id 连续两轮（上下文接力）
    # =========================================================
    cfg3 = {"recursion_limit": 30, "configurable": {"thread_id": "test_multiturn"}}
    try:
        await run_and_print("济南今天天气怎么样", agent, cfg3, "验收3a 第一轮")
        await run_and_print("那明天呢", agent, cfg3, "验收3b 第二轮（接上下文）")
    except Exception as e:
        print(f"\n❌ [验收3 异常]: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 4. FileAgent HITL（自动 Y，写 test_hitl1.txt）
    # =========================================================
    workspace = Path(app_config.WORKSPACE_DIR)
    workspace.mkdir(parents=True, exist_ok=True)
    test_file = workspace / "test_hitl1.txt"
    test_file.write_text("original content", encoding="utf-8")
    print(f"\n[HITL 准备] 已写入初始内容: {test_file}")

    cfg4 = {"recursion_limit": 30, "configurable": {"thread_id": "test_hitl_deepagent"}}
    try:
        await run_hitl_auto(
            f"请把 workspace/test_hitl1.txt 的内容改写成 'hello from deepagents'",
            agent, cfg4, "验收4 FileAgent HITL"
        )
        final_content = test_file.read_text(encoding="utf-8")
        print(f"\n[HITL 结果] test_hitl1.txt 最终内容: '{final_content}'")
        print(f"  → {'PASS ✅' if 'hello from deepagents' in final_content else 'FAIL ❌'}")
    except Exception as e:
        print(f"\n❌ [验收4 异常]: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 5a. KnowledgeAgent 含图片 RAG（查图片相关问题）
    # =========================================================
    cfg5a = {"recursion_limit": 30, "configurable": {"thread_id": "test_rag_image"}}
    print(f"\n{'='*70}")
    print("[调查 5a] KnowledgeAgent 含图片 RAG chunk 测试")
    print(f"{'='*70}")
    try:
        await run_and_print("火神战姬长什么样", agent, cfg5a, "调查5a 图片RAG")
        print("[调查 5a] 未报 400 错误")
    except Exception as e:
        print(f"[调查 5a] 异常: {type(e).__name__}: {e}")

    # =========================================================
    # 5b. CodeAgent 画图
    # =========================================================
    cfg5b = {"recursion_limit": 30, "configurable": {"thread_id": "test_code_image"}}
    try:
        await run_code_agent_image(agent, cfg5b)
    except Exception as e:
        print(f"\n❌ [调查5b 异常]: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
