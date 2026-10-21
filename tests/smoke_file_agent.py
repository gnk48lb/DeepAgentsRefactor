"""
tests/smoke_file_agent.py

FileAgent 离线冒烟测试套件（无外部依赖、无需 API Key、无需 Docker/网络）。
覆盖 Group A (A1-A4) 与 Group B (B1-B9)。
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any

# 保证在项目环境下统一 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 确保项目根目录在 sys.path
# _PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(_PROJECT_ROOT) not in sys.path:


import asyncio

# 仅导入 app.file_agent, app.hitl 与 app.markitdown_tool，严禁导入 app.models / app.deep_agent / app.graph
from app import file_agent, hitl, markitdown_tool
from langchain_core.tools import StructuredTool

from deepagents import (
    create_deep_agent,
    CompiledSubAgent,
    register_harness_profile,
    HarnessProfile,
    GeneralPurposeSubagentProfile,
    FilesystemMiddleware,
)
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

# 注册生产同款 harness profile
register_harness_profile(
    "scriptedmodel",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset({"write_file", "edit_file", "delete", "execute"}),
    ),
)


class ScriptedModel(BaseChatModel):
    """按脚本依次返回 AIMessage；bind_tools 返回自身并记录工具名。"""
    script: List[AIMessage]
    idx: int = 0
    seen_tools: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        self.seen_tools.append(sorted(t["name"] if isinstance(t, dict) else getattr(t, "name", str(t)) for t in tools))
        return self

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kw) -> ChatResult:
        msg = self.script[min(self.idx, len(self.script) - 1)]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


# DesktopAgent 替身（手写 interrupt 字符串，用来测并行中断）
import operator
from typing import Annotated
from typing_extensions import TypedDict

class _S(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def desktop_like():
    def node(state):
        d = interrupt("桌面高危操作，请回复 Y/N")
        return {"messages": [AIMessage(content=f"desktop 决定={d}")]}
    g = StateGraph(_S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Group A: 直接 invoke FileAgent runnable (读工具，不触发 HITL)
# ══════════════════════════════════════════════════════════════════════════════

def test_a1_tool_whitelist():
    """A1 工具白名单：fam.seen_tools[0] 恰好等于白名单，不含 delete/execute。"""
    with tempfile.TemporaryDirectory() as td:
        fam = ScriptedModel(script=[AIMessage(content="done")])
        runnable = file_agent.build_file_agent_runnable(fam, "prompt", td)
        runnable.invoke({"messages": [HumanMessage(content="hi")]})
        assert len(fam.seen_tools) > 0, "bind_tools 未被调用"
        actual = set(fam.seen_tools[0])
        expected = {"ls", "read_file", "write_file", "edit_file", "glob", "grep", "move_file", "create_directory"}
        assert actual == expected, f"工具集不匹配: actual={actual}, expected={expected}"
        assert "delete" not in actual, "delete 不应存在"
        assert "execute" not in actual, "execute 不应存在"


def test_a2_read_operations():
    """A2 读：中文目录/笔记.txt、a.py、5000 行 big.txt；3 种路径写法；offset/limit 分页。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "中文目录").mkdir()
        (root / "中文目录" / "笔记.txt").write_text("中文笔记内容", encoding="utf-8")
        (root / "a.py").write_text("print('hello')", encoding="utf-8")
        (root / "big.txt").write_text("\n".join(f"line {i}" for i in range(5000)), encoding="utf-8")

        script = [
            AIMessage(content="", tool_calls=[
                {"name": "ls", "args": {"path": "workspace"}, "id": "c1"},
                {"name": "ls", "args": {"path": "/workspace/中文目录"}, "id": "c2"},
                {"name": "glob", "args": {"pattern": "**/*", "path": "workspace"}, "id": "c3"},
                {"name": "read_file", "args": {"file_path": "workspace/中文目录/笔记.txt"}, "id": "c4"},
                {"name": "read_file", "args": {"file_path": "/workspace/中文目录/笔记.txt"}, "id": "c5"},
                {"name": "read_file", "args": {"file_path": "workspace\\中文目录\\笔记.txt"}, "id": "c6"},
                {"name": "read_file", "args": {"file_path": "workspace/big.txt", "offset": 4990, "limit": 5}, "id": "c7"},
            ]),
            AIMessage(content="finished reading"),
        ]
        fam = ScriptedModel(script=script)
        runnable = file_agent.build_file_agent_runnable(fam, "prompt", td)
        res = runnable.invoke({"messages": [HumanMessage(content="read files")]})
        tool_msgs = {m.tool_call_id: m for m in res["messages"] if isinstance(m, ToolMessage)}

        assert tool_msgs["c1"].status != "error", f"ls workspace 失败: {tool_msgs['c1'].content}"
        assert tool_msgs["c2"].status != "error", f"ls 中文目录 失败: {tool_msgs['c2'].content}"
        assert tool_msgs["c3"].status != "error", f"glob 失败: {tool_msgs['c3'].content}"

        assert "中文笔记内容" in str(tool_msgs["c4"].content), "read_file workspace/xxx 失败"
        assert "中文笔记内容" in str(tool_msgs["c5"].content), "read_file /workspace/xxx 失败"
        assert "中文笔记内容" in str(tool_msgs["c6"].content), "read_file workspace\\xxx 失败"

        big_content = str(tool_msgs["c7"].content)
        assert "[Read 5 lines" in big_content, f"分页提示缺失: {big_content}"


def test_a3_path_traversal():
    """A3 越界：read_file('workspace/../../etc/passwd') status == 'error'，含 'traversal'。"""
    with tempfile.TemporaryDirectory() as td:
        script = [
            AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"file_path": "workspace/../../etc/passwd"}, "id": "c1"}
            ]),
            AIMessage(content="done"),
        ]
        fam = ScriptedModel(script=script)
        runnable = file_agent.build_file_agent_runnable(fam, "prompt", td)
        res = runnable.invoke({"messages": [HumanMessage(content="read out of bounds")]})
        tool_msgs = {m.tool_call_id: m for m in res["messages"] if isinstance(m, ToolMessage)}
        tm = tool_msgs["c1"]
        assert tm.status == "error", f"期望 status == 'error'，实际: {tm.status}"
        assert "traversal" in str(tm.content).lower(), f"期望包含 'traversal'，实际: {tm.content}"


def test_a4_large_tool_results_isolation():
    """A4 不污染：工具结果巨大时落盘到虚拟 state，磁盘无 large_tool_results 文件。"""
    @tool
    def huge() -> str:
        """返回超大输出。"""
        return "x" * 400000

    with tempfile.TemporaryDirectory() as td:
        script = [
            AIMessage(content="", tool_calls=[{"name": "huge", "args": {}, "id": "h1"}]),
            AIMessage(content="huge done"),
        ]
        m = ScriptedModel(script=script)
        agent = create_agent(
            model=m,
            tools=[huge],
            middleware=[
                FilesystemMiddleware(backend=file_agent.build_workspace_backend(td), tools=file_agent.FS_TOOLS),
            ],
        )
        out = agent.invoke({"messages": [HumanMessage(content="run huge")]})
        # 验证真实磁盘没有框架文件
        disk_files = list(Path(td).rglob("*large_tool_results*"))
        assert len(disk_files) == 0, f"真实磁盘上发现了 large_tool_results 文件: {disk_files}"
        # 验证内存 state 中包含 large_tool_results
        files = out.get("files", {})
        assert any("large_tool_results" in str(k) for k in files.keys()), f"state['files'] 中缺少 large_tool_results: {files.keys()}"


# ══════════════════════════════════════════════════════════════════════════════
# Group B: 挂在 create_deep_agent 主图里 (带 MemorySaver 与 HITL 中断恢复)
# ══════════════════════════════════════════════════════════════════════════════

def _build_main_deep_agent(main_model, file_model, workspace_dir, checkpointer=None, extra_subagents=None):
    fa_runnable = file_agent.build_file_agent_runnable(file_model, "FileAgent prompt", workspace_dir)
    subagents = [
        CompiledSubAgent(name="FileAgent", description="本地文件专家", runnable=fa_runnable)
    ]
    if extra_subagents:
        subagents.extend(extra_subagents)
    return create_deep_agent(
        model=main_model,
        subagents=subagents,
        checkpointer=checkpointer or MemorySaver(),
    )


def test_b1_approve():
    """B1 approve：write_file('workspace/a/b/新建.txt') 批准后落地，ScriptedModel.idx == 2，分别测两种恢复值。"""
    for mode in ("plain_dict", "id_mapped"):
        with tempfile.TemporaryDirectory() as td:
            main_script = [
                AIMessage(content="", tool_calls=[
                    {"name": "task", "args": {"subagent_type": "FileAgent", "description": "写文件"}, "id": "t1"}
                ]),
                AIMessage(content="主任务已完成"),
            ]
            fa_script = [
                AIMessage(content="", tool_calls=[
                    {"name": "write_file", "args": {"file_path": "workspace/a/b/新建.txt", "content": "测试内容"}, "id": "w1"}
                ]),
                AIMessage(content="【内部专家汇报 - 来源：FileAgent】\n写入成功"),
            ]
            main_m = ScriptedModel(script=main_script)
            fa_m = ScriptedModel(script=fa_script)
            cp = MemorySaver()
            agent = _build_main_deep_agent(main_m, fa_m, td, checkpointer=cp)
            cfg = {"configurable": {"thread_id": f"b1_{mode}"}}

            # 首次 invoke 触发中断
            agent.invoke({"messages": [HumanMessage(content="请写文件")]}, config=cfg)
            state = agent.get_state(cfg)
            assert hitl.has_pending_interrupts(state), "应该触发 HITL 中断"
            intrs = hitl.pending_interrupts(state)
            assert len(intrs) == 1

            # 恢复值
            if mode == "plain_dict":
                resume_payload = {"decisions": [{"type": "approve"}]}
            else:
                resume_payload = {intrs[0].id: {"decisions": [{"type": "approve"}]}}

            res = agent.invoke(Command(resume=resume_payload), config=cfg)

            target = Path(td) / "a" / "b" / "新建.txt"
            assert target.exists(), f"文件未落盘: {target}"
            assert target.read_text(encoding="utf-8") == "测试内容"
            assert fa_m.idx == 2, f"FileAgent ScriptedModel.idx 应为 2，实际为 {fa_m.idx}（模型重复调用）"

            # 验证 task 结果包含第二条脚本文本
            msgs = res["messages"]
            tool_msgs = [m for m in msgs if isinstance(m, ToolMessage) and getattr(m, "name", "") == "task"]
            assert any("【内部专家汇报 - 来源：FileAgent】" in str(m.content) for m in tool_msgs), "task 工具返回中缺失专家汇报"


def test_b2_reject():
    """B2 reject：用 hitl.build_resume(state, 'N') → 没有文件产生，idx 仍为 2。"""
    with tempfile.TemporaryDirectory() as td:
        main_script = [
            AIMessage(content="", tool_calls=[
                {"name": "task", "args": {"subagent_type": "FileAgent", "description": "写文件"}, "id": "t1"}
            ]),
            AIMessage(content="主管处理拒绝"),
        ]
        fa_script = [
            AIMessage(content="", tool_calls=[
                {"name": "write_file", "args": {"file_path": "workspace/拒绝.txt", "content": "不要写入"}, "id": "w1"}
            ]),
            AIMessage(content="【内部专家汇报 - 来源：FileAgent】\n被拒绝退出"),
        ]
        main_m = ScriptedModel(script=main_script)
        fa_m = ScriptedModel(script=fa_script)
        agent = _build_main_deep_agent(main_m, fa_m, td)
        cfg = {"configurable": {"thread_id": "b2_reject"}}

        agent.invoke({"messages": [HumanMessage(content="写文件")]}, config=cfg)
        state = agent.get_state(cfg)
        assert hitl.has_pending_interrupts(state)

        resume_payload = hitl.build_resume(state, "N")
        agent.invoke(Command(resume=resume_payload), config=cfg)

        target = Path(td) / "拒绝.txt"
        assert not target.exists(), f"被拒绝的文件不应该存在: {target}"
        assert fa_m.idx == 2, f"FileAgent idx 应该为 2，实际为 {fa_m.idx}"


def test_b3_edit():
    """B3 edit：edited_action 把路径改成 'workspace/改过.txt' → 文件落在 <ws>/改过.txt。"""
    with tempfile.TemporaryDirectory() as td:
        main_script = [
            AIMessage(content="", tool_calls=[
                {"name": "task", "args": {"subagent_type": "FileAgent", "description": "写文件"}, "id": "t1"}
            ]),
            AIMessage(content="完成"),
        ]
        fa_script = [
            AIMessage(content="", tool_calls=[
                {"name": "write_file", "args": {"file_path": "workspace/原路径.txt", "content": "内容"}, "id": "w1"}
            ]),
            AIMessage(content="【内部专家汇报 - 来源：FileAgent】\n完成"),
        ]
        main_m = ScriptedModel(script=main_script)
        fa_m = ScriptedModel(script=fa_script)
        agent = _build_main_deep_agent(main_m, fa_m, td)
        cfg = {"configurable": {"thread_id": "b3_edit"}}

        agent.invoke({"messages": [HumanMessage(content="写文件")]}, config=cfg)
        state = agent.get_state(cfg)
        intrs = hitl.pending_interrupts(state)
        assert len(intrs) == 1

        resume_payload = {
            intrs[0].id: {
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "write_file",
                            "args": {"file_path": "workspace/改过.txt", "content": "改后内容"},
                        },
                    }
                ]
            }
        }
        agent.invoke(Command(resume=resume_payload), config=cfg)

        assert not (Path(td) / "原路径.txt").exists(), "原路径文件不应存在"
        edited_target = Path(td) / "改过.txt"
        assert edited_target.exists(), f"改后文件不存在: {edited_target}"
        assert edited_target.read_text(encoding="utf-8") == "改后内容"


def test_b4_multiple_write_files_in_one_message():
    """B4 一条 AIMessage 含两个 write_file：1 个 interrupt、2 个 action_requests；build_resume 产生 2 个 decisions；两个都写出。"""
    with tempfile.TemporaryDirectory() as td:
        main_script = [
            AIMessage(content="", tool_calls=[
                {"name": "task", "args": {"subagent_type": "FileAgent", "description": "批量写"}, "id": "t1"}
            ]),
            AIMessage(content="批量完成"),
        ]
        fa_script = [
            AIMessage(content="", tool_calls=[
                {"name": "write_file", "args": {"file_path": "workspace/f1.txt", "content": "aaa"}, "id": "w1"},
                {"name": "write_file", "args": {"file_path": "workspace/f2.txt", "content": "bbb"}, "id": "w2"},
            ]),
            AIMessage(content="【内部专家汇报 - 来源：FileAgent】\n两文件均完成"),
        ]
        main_m = ScriptedModel(script=main_script)
        fa_m = ScriptedModel(script=fa_script)
        agent = _build_main_deep_agent(main_m, fa_m, td)
        cfg = {"configurable": {"thread_id": "b4_multi"}}

        agent.invoke({"messages": [HumanMessage(content="批量写")]}, config=cfg)
        state = agent.get_state(cfg)
        intrs = hitl.pending_interrupts(state)
        assert len(intrs) == 1, f"应该只有 1 个中断，实际有 {len(intrs)} 个"
        actions = intrs[0].value.get("action_requests", [])
        assert len(actions) == 2, f"应该有 2 个 action_requests，实际为 {len(actions)}"

        resume_payload = hitl.build_resume(state, "Y")
        decisions = resume_payload[intrs[0].id]["decisions"]
        assert len(decisions) == 2, f"build_resume 应该生成 2 个 decisions，实际为 {len(decisions)}"

        agent.invoke(Command(resume=resume_payload), config=cfg)

        f1 = Path(td) / "f1.txt"
        f2 = Path(td) / "f2.txt"
        assert f1.exists() and f1.read_text(encoding="utf-8") == "aaa", f"f1 错误: {f1}"
        assert f2.exists() and f2.read_text(encoding="utf-8") == "bbb", f"f2 错误: {f2}"


def test_b5_unprefixed_path():
    """B5 不带前缀的路径：write_file('notes/无前缀.txt') 批准后落在 <ws>/notes/无前缀.txt。"""
    with tempfile.TemporaryDirectory() as td:
        main_script = [
            AIMessage(content="", tool_calls=[
                {"name": "task", "args": {"subagent_type": "FileAgent", "description": "写无前缀"}, "id": "t1"}
            ]),
            AIMessage(content="完成"),
        ]
        fa_script = [
            AIMessage(content="", tool_calls=[
                {"name": "write_file", "args": {"file_path": "notes/无前缀.txt", "content": "无前缀内容"}, "id": "w1"}
            ]),
            AIMessage(content="【内部专家汇报 - 来源：FileAgent】\n完成"),
        ]
        main_m = ScriptedModel(script=main_script)
        fa_m = ScriptedModel(script=fa_script)
        agent = _build_main_deep_agent(main_m, fa_m, td)
        cfg = {"configurable": {"thread_id": "b5_unprefix"}}

        agent.invoke({"messages": [HumanMessage(content="写")]}, config=cfg)
        state = agent.get_state(cfg)
        resume_payload = hitl.build_resume(state, "Y")
        agent.invoke(Command(resume=resume_payload), config=cfg)

        target = Path(td) / "notes" / "无前缀.txt"
        assert target.exists(), f"未落盘到真实工作区: {target}"
        assert target.read_text(encoding="utf-8") == "无前缀内容"


def test_b6_move_file_and_create_directory():
    """B6 move_file 归档、目标已存在报错、越界报错；create_directory 成功。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "source.txt"
        src.write_text("待移动内容", encoding="utf-8")

        # 1. 正常移动到 archive_trash
        main_script = [
            AIMessage(content="", tool_calls=[
                {"name": "task", "args": {"subagent_type": "FileAgent", "description": "移动"}, "id": "t1"}
            ]),
            AIMessage(content="已移动"),
        ]
        fa_script = [
            AIMessage(content="", tool_calls=[
                {"name": "move_file", "args": {"source": "/workspace/source.txt", "destination": "/workspace/archive_trash/source.txt"}, "id": "m1"}
            ]),
            AIMessage(content="【内部专家汇报 - 来源：FileAgent】\n移动成功"),
        ]
        agent = _build_main_deep_agent(ScriptedModel(script=main_script), ScriptedModel(script=fa_script), td)
        cfg = {"configurable": {"thread_id": "b6_move"}}
        agent.invoke({"messages": [HumanMessage(content="移动")]}, config=cfg)
        state = agent.get_state(cfg)
        assert hitl.has_pending_interrupts(state)
        agent.invoke(Command(resume=hitl.build_resume(state, "Y")), config=cfg)

        assert not src.exists(), "源文件应被移走"
        dst = root / "archive_trash" / "source.txt"
        assert dst.exists() and dst.read_text(encoding="utf-8") == "待移动内容"

        # 2. create_directory 批准后目录存在
        main_s2 = [
            AIMessage(content="", tool_calls=[
                {"name": "task", "args": {"subagent_type": "FileAgent", "description": "建目录"}, "id": "t1"}
            ]),
            AIMessage(content="已建"),
        ]
        fa_s2 = [
            AIMessage(content="", tool_calls=[
                {"name": "create_directory", "args": {"path": "/workspace/sub/dir"}, "id": "cd1"}
            ]),
            AIMessage(content="【内部专家汇报 - 来源：FileAgent】\n目录已创建"),
        ]
        agent2 = _build_main_deep_agent(ScriptedModel(script=main_s2), ScriptedModel(script=fa_s2), td)
        cfg2 = {"configurable": {"thread_id": "b6_mkdir"}}
        agent2.invoke({"messages": [HumanMessage(content="建目录")]}, config=cfg2)
        state2 = agent2.get_state(cfg2)
        assert hitl.has_pending_interrupts(state2)
        agent2.invoke(Command(resume=hitl.build_resume(state2, "Y")), config=cfg2)
        assert (root / "sub" / "dir").is_dir(), "目录未被创建"

        # 3. move_file 目标已存在报错与越界报错（直接调用 workspace tools 验证逻辑）
        tools_list = file_agent.make_workspace_tools(td)
        move_tool = {t.name: t for t in tools_list}["move_file"]
        # 先造一个 existing.txt
        (root / "existing.txt").write_text("x", encoding="utf-8")
        err1 = move_tool.invoke({"source": "archive_trash/source.txt", "destination": "existing.txt"})
        assert "目标已存在" in err1, f"期望'目标已存在'，实际: {err1}"

        err2 = move_tool.invoke({"source": "workspace/../outside.txt", "destination": "workspace/out.txt"})
        assert "超出工作区" in err2, f"期望'超出工作区'，实际: {err2}"


def test_b7_hitl_warning_formatting():
    """B7 hitl.format_hitl_warning(state) 输出包含工具名、目标路径与 '回复 Y 允许'；移动类显示 'source -> destination'。"""
    with tempfile.TemporaryDirectory() as td:
        main_script = [
            AIMessage(content="", tool_calls=[
                {"name": "task", "args": {"subagent_type": "FileAgent", "description": "操作"}, "id": "t1"}
            ]),
            AIMessage(content="完成"),
        ]
        fa_script = [
            AIMessage(content="", tool_calls=[
                {"name": "move_file", "args": {"source": "/workspace/a.txt", "destination": "/workspace/b.txt"}, "id": "m1"}
            ]),
            AIMessage(content="完成"),
        ]
        agent = _build_main_deep_agent(ScriptedModel(script=main_script), ScriptedModel(script=fa_script), td)
        cfg = {"configurable": {"thread_id": "b7_fmt"}}
        agent.invoke({"messages": [HumanMessage(content="移动")]}, config=cfg)
        state = agent.get_state(cfg)

        warning = hitl.format_hitl_warning(state)
        assert "move_file" in warning, f"未包含工具名: {warning}"
        assert "/workspace/a.txt -> /workspace/b.txt" in warning, f"未正确展示 source -> destination: {warning}"
        assert "回复 Y 允许" in warning, f"未包含 '回复 Y 允许': {warning}"


def test_b8_parallel_interrupts():
    """B8 并行中断：同时派 FileAgent 和 DesktopAgent，2 个中断；直接 resume='Y' 抛 RuntimeError；用 build_resume 正常恢复。"""
    with tempfile.TemporaryDirectory() as td:
        main_script = [
            AIMessage(content="", tool_calls=[
                {"name": "task", "args": {"subagent_type": "FileAgent", "description": "写文件"}, "id": "t1"},
                {"name": "task", "args": {"subagent_type": "DesktopAgent", "description": "桌面操作"}, "id": "t2"},
            ]),
            AIMessage(content="双任务完成"),
        ]
        fa_script = [
            AIMessage(content="", tool_calls=[
                {"name": "write_file", "args": {"file_path": "workspace/parallel.txt", "content": "parallel_ok"}, "id": "w1"}
            ]),
            AIMessage(content="【内部专家汇报 - 来源：FileAgent】\n并行写入成功"),
        ]
        desktop_sub = CompiledSubAgent(name="DesktopAgent", description="桌面专家", runnable=desktop_like())
        agent = _build_main_deep_agent(
            ScriptedModel(script=main_script),
            ScriptedModel(script=fa_script),
            td,
            extra_subagents=[desktop_sub],
        )
        cfg = {"configurable": {"thread_id": "b8_parallel"}}

        agent.invoke({"messages": [HumanMessage(content="同时执行两个任务")]}, config=cfg)
        state = agent.get_state(cfg)
        intrs = hitl.pending_interrupts(state)
        assert len(intrs) == 2, f"期望 2 个 pending interrupts，实际: {len(intrs)}"

        # 断言直接 Command(resume="Y") 抛出 RuntimeError
        threw = False
        try:
            agent.invoke(Command(resume="Y"), config=cfg)
        except RuntimeError as e:
            threw = True
            assert "When there are multiple pending interrupts" in str(e), f"异常信息不符合预期: {e}"
        assert threw, "多个中断时直接 Command(resume='Y') 未抛出 RuntimeError"

        # 使用 hitl.build_resume(state, 'Y') 恢复
        resume_payload = hitl.build_resume(state, "Y")
        res = agent.invoke(Command(resume=resume_payload), config=cfg)

        # 验证两个子 Agent 都完成
        assert (Path(td) / "parallel.txt").exists(), "FileAgent 写入文件未生成"
        msgs = res["messages"]
        tool_contents = [str(m.content) for m in msgs if isinstance(m, ToolMessage)]
        assert any("并行写入成功" in c for c in tool_contents), "未发现 FileAgent 成功回报"
        assert any("desktop 决定=Y" in c for c in tool_contents), "未发现 DesktopAgent 成功回报"


def test_b9_global_profile_isolation():
    """B9 全局 profile 不受影响：声明式子 Agent seen_tools 无写工具，FileAgent 有写工具且触发 interrupt。"""
    with tempfile.TemporaryDirectory() as td:
        main_script = [
            AIMessage(content="", tool_calls=[
                {"name": "task", "args": {"subagent_type": "NormalAgent", "description": "查东西"}, "id": "t1"}
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "task", "args": {"subagent_type": "FileAgent", "description": "写文件"}, "id": "t2"}
            ]),
            AIMessage(content="全流程完成"),
        ]
        normal_m = ScriptedModel(script=[AIMessage(content="normal subagent done")])
        fa_script = [
            AIMessage(content="", tool_calls=[
                {"name": "write_file", "args": {"file_path": "workspace/profile_test.txt", "content": "isolated"}, "id": "w1"}
            ]),
            AIMessage(content="【内部专家汇报 - 来源：FileAgent】\n完成"),
        ]
        fa_m = ScriptedModel(script=fa_script)
        main_m = ScriptedModel(script=main_script)

        declarative_sub = {
            "name": "NormalAgent",
            "description": "普通声明式子专家",
            "system_prompt": "prompt",
            "tools": [],
            "model": normal_m,
        }

        agent = _build_main_deep_agent(
            main_m,
            fa_m,
            td,
            extra_subagents=[declarative_sub],
        )
        cfg = {"configurable": {"thread_id": "b9_profile"}}

        # 执行第一步（派 NormalAgent）
        agent.invoke({"messages": [HumanMessage(content="先查再写")]}, config=cfg)
        state = agent.get_state(cfg)
        # NormalAgent 执行完毕，下一步派 FileAgent 会触发 interrupt
        assert hitl.has_pending_interrupts(state), "FileAgent 写操作应该触发中断"

        # 恢复并完成
        agent.invoke(Command(resume=hitl.build_resume(state, "Y")), config=cfg)
        assert (Path(td) / "profile_test.txt").exists()

        # 断言主 Agent 与声明式子 Agent 不暴露写/删工具
        assert len(main_m.seen_tools) > 0
        main_tools = set(main_m.seen_tools[0])
        assert not (main_tools & {"write_file", "edit_file", "delete", "execute"}), f"主 Agent 包含了被排除工具: {main_tools}"

        assert len(normal_m.seen_tools) > 0
        normal_tools = set(normal_m.seen_tools[0])
        assert not (normal_tools & {"write_file", "edit_file", "delete", "execute"}), f"声明式子 Agent 包含了被排除工具: {normal_tools}"

        # 断言 FileAgent 含有 write_file / edit_file
        assert len(fa_m.seen_tools) > 0
        fa_tools = set(fa_m.seen_tools[0])
        assert "write_file" in fa_tools and "edit_file" in fa_tools, f"FileAgent 缺失写工具: {fa_tools}"


# ══════════════════════════════════════════════════════════════════════════════
# Group C: MarkItDown 工具包装与路径/安全校验
# ══════════════════════════════════════════════════════════════════════════════

def test_c1_markitdown_normalize_uri():
    """C1: markitdown normalize_uri 合法映射与安全越界拦截。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        sub = root / "报告"
        sub.mkdir(parents=True, exist_ok=True)
        sample_file = sub / "a.pdf"
        sample_file.write_bytes(b"%PDF-1.4 dummy content")
        expected_uri = sample_file.as_uri()

        # 合法路径映射测试
        valid_cases = [
            "/workspace/报告/a.pdf",
            "workspace/报告/a.pdf",
            "报告/a.pdf",
            "file:///workspace/报告/a.pdf",
            str(sample_file),
            expected_uri,
        ]
        for v in valid_cases:
            res = asyncio.run(markitdown_tool.normalize_uri(v, workspace_dir=root))
            assert res == expected_uri, f"合法路径映射不符: {v} -> {res} (expected: {expected_uri})"

        # 拒绝输入测试
        rejected_cases = [
            "file:///etc/passwd",
            "file:///C:/Windows/win.ini",
            "C:\\Windows\\win.ini",
            "file:///workspace/../../etc/passwd",
            "/workspace/../../etc/passwd",
            "workspace/../../etc/passwd",
            "http://127.0.0.1/",
            "http://169.254.169.254/",
            "data:text/plain;base64,SGVsbG8=",
            "file://evil.com/a.pdf",
            "",
            "   ",
            "/workspace",
            "workspace",
            "/workspace/不存在的文件.pdf",
        ]
        for r in rejected_cases:
            try:
                asyncio.run(markitdown_tool.normalize_uri(r, workspace_dir=root))
                assert False, f"未被拦截的非法输入: {r}"
            except ValueError as e:
                assert str(e).strip(), f"拒绝原因不能为空: {r}"


def test_c2_markitdown_build_tools_and_mounting():
    """C2: markitdown build_tools 工具挂载、执行包装与 FileAgent 隔离性验证。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        test_file = root / "doc.pdf"
        test_file.write_bytes(b"%PDF-1.4 dummy")

        called_with = []
        async def mock_convert(uri: str) -> str:
            called_with.append(uri)
            return "# Mocked Markdown Content"

        mock_tool = StructuredTool(
            name="convert_to_markdown",
            description="Converts document to markdown",
            args_schema={"type": "object", "properties": {"uri": {"type": "string"}}, "required": ["uri"]},
            coroutine=mock_convert,
        )

        wrapped_tools = markitdown_tool.build_tools([mock_tool], workspace_dir=root)
        assert len(wrapped_tools) == 1
        wrapped = wrapped_tools[0]
        assert wrapped.name == "convert_to_markdown"
        assert wrapped.description == "Converts document to markdown"

        # 成功调用
        res = asyncio.run(wrapped.ainvoke({"uri": "/workspace/doc.pdf"}))
        assert res == "# Mocked Markdown Content"
        assert called_with == [test_file.as_uri()]

        # 被拒绝的调用返回错误字符串而非未捕获异常
        rej_res = asyncio.run(wrapped.ainvoke({"uri": "file:///etc/passwd"}))
        assert rej_res.startswith("convert_to_markdown 调用被拒绝："), f"拒绝文案前缀不符: {rej_res}"

        # 挂载到 FileAgent
        fam = ScriptedModel(script=[AIMessage(content="ok")])
        runnable = file_agent.build_file_agent_runnable(
            llm=fam,
            system_prompt="prompt",
            workspace_dir=str(root),
            extra_tools=wrapped_tools,
        )
        runnable.invoke({"messages": [HumanMessage(content="hi")]})
        assert fam.seen_tools, "未绑定工具"
        fa_tools = set(fam.seen_tools[0])
        assert "convert_to_markdown" in fa_tools, f"FileAgent 应持有 convert_to_markdown: {fa_tools}"

        # 默认不传 extra_tools 时不包含 convert_to_markdown
        fam_default = ScriptedModel(script=[AIMessage(content="ok")])
        runnable_default = file_agent.build_file_agent_runnable(
            llm=fam_default,
            system_prompt="prompt",
            workspace_dir=str(root),
        )
        runnable_default.invoke({"messages": [HumanMessage(content="hi")]})
        assert "convert_to_markdown" not in fam_default.seen_tools[0]


def test_c3_markitdown_edge_and_ssrf():
    """C3: normalize_uri 的额外边界输入 + allow_http=True 时的内网/元数据地址拦截。
    只用字面量 IP / localhost / .invalid 域名，不依赖外网。"""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        (root / "报告").mkdir()
        (root / "报告" / "a.pdf").write_bytes(b"x")
        (root / "dir_only").mkdir()
        expected_uri = (root / "报告" / "a.pdf").as_uri()

        async def run(raw, **kw):
            try:
                return await markitdown_tool.normalize_uri(raw, workspace_dir=root, **kw)
            except ValueError as e:
                return e

        allowed = [
            "file://localhost/workspace/报告/a.pdf", "file:/workspace/报告/a.pdf", "FILE:///workspace/报告/a.pdf",
            "/WORKSPACE/报告/a.pdf", "workspace\\报告\\a.pdf", "  /workspace/报告/a.pdf  ",
            "file:///workspace/%E6%8A%A5%E5%91%8A/a.pdf",
        ]
        for raw in allowed:
            res = asyncio.run(run(raw))
            assert res == expected_uri, f"应放行: {raw!r} -> {res!r}"

        denied = [
            "file:///workspace/%2e%2e/%2e%2e/etc/passwd", "workspace\\..\\..\\etc\\passwd",
            "C:win.ini", "javascript:alert(1)", "ftp://x/a.pdf",
            "/workspace/dir_only", "workspace/不存在的文件.pdf",
        ]
        for raw in denied:
            res = asyncio.run(run(raw))
            assert isinstance(res, ValueError), f"应拒绝: {raw!r} -> {res!r}"

        # UNC / 设备路径必须在 resolve() 之前就被拒绝（错误信息里要有 UNC 字样，证明走的是早期拦截）
        for raw in ["//evil.com/share/a.pdf", "\\\\evil.com\\share\\a.pdf", "file:////evil.com/share/a.pdf",
                    "\\\\?\\C:\\Windows\\win.ini", "//workspace/a.pdf"]:
            res = asyncio.run(run(raw))
            assert isinstance(res, ValueError) and "UNC" in str(res), f"UNC/设备路径应被早期拦截: {raw!r} -> {res!r}"

        # 符号链接逃逸（Windows 上创建符号链接可能需要开发者模式/管理员权限，失败则跳过此小项）
        with tempfile.TemporaryDirectory() as out_td:
            outside = Path(out_td).resolve()
            (outside / "secret.txt").write_text("s", encoding="utf-8")
            try:
                os.symlink(outside, root / "link_out", target_is_directory=True)
            except (OSError, NotImplementedError):
                print("    (跳过符号链接小项：当前环境无法创建符号链接)")
            else:
                res = asyncio.run(run("/workspace/link_out/secret.txt"))
                assert isinstance(res, ValueError), f"符号链接逃逸未被拦截: {res!r}"

        # allow_http=True：内网 / 本机 / 元数据 / 混淆写法 / 带账号密码 / 无法解析的域名，全部应以 ValueError 拒绝
        for raw in ["http://127.0.0.1/", "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/", "http://[::1]/",
                    "http://localhost:8000/x", "http://user:pw@example.com/", "http://[::ffff:127.0.0.1]/",
                    "http://0.0.0.0/", "http://2130706433/", "http://0x7f.0.0.1/",
                    "http://nonexistent-host.invalid/a.pdf"]:
            res = asyncio.run(run(raw, allow_http=True))
            assert isinstance(res, ValueError), f"allow_http=True 时应以 ValueError 拒绝: {raw!r} -> {res!r}"


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    ("A1 工具白名单", test_a1_tool_whitelist),
    ("A2 读操作与分页", test_a2_read_operations),
    ("A3 越界路径拦截", test_a3_path_traversal),
    ("A4 大结果落盘隔离不污染真实工作区", test_a4_large_tool_results_isolation),
    ("B1 approve 流程与恢复值映射", test_b1_approve),
    ("B2 reject 流程与拒绝消息", test_b2_reject),
    ("B3 edit 流程与参数修改", test_b3_edit),
    ("B4 单消息双 write_file 批量授权", test_b4_multiple_write_files_in_one_message),
    ("B5 无前缀路径正确落盘", test_b5_unprefixed_path),
    ("B6 move_file 归档与边界检查", test_b6_move_file_and_create_directory),
    ("B7 hitl.format_hitl_warning 文案格式", test_b7_hitl_warning_formatting),
    ("B8 并行中断与多 id 恢复", test_b8_parallel_interrupts),
    ("B9 全局 profile 隔离性验证", test_b9_global_profile_isolation),
    ("C1 markitdown 路径映射与安全越界拦截", test_c1_markitdown_normalize_uri),
    ("C2 markitdown build_tools 工具挂载与隔离", test_c2_markitdown_build_tools_and_mounting),
    ("C3 markitdown 边界输入与 SSRF 分支", test_c3_markitdown_edge_and_ssrf),
]


def main():
    print("=" * 70)
    print("开始运行 FileAgent 离线冒烟测试...")
    print("=" * 70)
    passed = 0
    failed = 0
    for name, func in SCENARIOS:
        try:
            func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
