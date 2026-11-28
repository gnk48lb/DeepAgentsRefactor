"""
tests/smoke_cloud_mode.py

DEPLOY_MODE=cloud（部署形态）离线冒烟测试：不需要 API Key / Docker / MCP / 网络 / 真实 Modal。
  C1 配置：DEPLOY_MODE 的解析（cloud / 带空格的 Cloud / 空串 / 写错）与 is_cloud_mode() 现读；云端分支不在 import 时缓存
  C2 desktop 回归：KnowledgeAgent 工具、_DECLARATIVE_SPECS、被钉死的提示词哈希、新常量都不是旧常量的别名；
     desktop 下 build_main_agents 的构建顺序与调用方式不变
  C3 云端 KnowledgeAgent：CLOUD_SPEC / CLOUD_PROMPT 与 read_webpage_content（短内容原样、超长截断、错误串透传）
  C4 路径归一：Milvus 元数据里带反斜杠的 source_path / image_path 能读到完整小作文与图片；绝对路径与越界路径被拒绝
  C5 云端提示词：5 个新常量的识别短语与禁用词；build_member_main_prompt 在 web_read=False 时与改动前逐字节相同
  C6 云端构建：不碰 MCP / owner 图 / 本机专家；owner 与 member 是同一张图；Modal 可用时多云端 CodeAgent；定时任务图
  C7 断言仍然有效：13 个宿主机工具名、非法专家名、预编译子专家都 RuntimeError；cloud 专用断言逐项拒绝
  C8 Modal 门控：cloud 下 owner 按 (owner, session_id) 取沙箱、可以上传文件；desktop 下仍拒绝
  C9 惰性导入：import app.tools.knowledge_tools 不会把 browser_use 带进进程
"""
import ast
import asyncio
import contextlib
import dataclasses
import hashlib
import importlib.util
import inspect
import io
import os
import posixpath
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

for _k in ("GEMINI_API_KEY", "SILICON_FLOW_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "AMAP_API_KEY", "MAGPIE_API_KEY", "TAVILY_API_KEY"):
    os.environ[_k] = "dummy"   # 占位值:本测试绝不应使用真实 key
# 本机 .env 里可能有真实的 Modal 令牌 / 定时任务开关 / DEPLOY_MODE=cloud：load_dotenv 不覆盖已有的环境变量，所以下面几行能压过它。
# 云端场景自己用上下文管理器临时 monkeypatch config.DEPLOY_MODE；启用 CodeAgent 的场景用 FakeModal。
os.environ["MODAL_DISABLED"] = "true"
os.environ["SCHEDULER_ENABLED"] = "false"
os.environ["DEPLOY_MODE"] = "desktop"
for _role in ("WORKER_LLM", "ORCHESTRATOR_LLM", "KNOWLEDGE_LLM", "INGESTION_VLM", "BROWSER_VLM", "DESKTOP_VLM"):
    os.environ[f"{_role}_PROVIDER"] = "magpie"
    os.environ.pop(f"{_role}_MODEL", None)

from langchain_core.messages import SystemMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from PIL import Image  # noqa: E402

from app import config  # noqa: E402
from app.infra import mcp_service, models, modal_sandbox  # noqa: E402
from app.infra import sandbox as docker_sandbox  # noqa: E402
from app.infra.sandbox import SandboxUnavailable  # noqa: E402
from app.agents import orchestrator, prompts  # noqa: E402
from app.agents.experts import code_agent, knowledge_agent  # noqa: E402
from app.runtime.cli_logger import _identify_agent  # noqa: E402
from app.tools import knowledge_tools, session_tools  # noqa: E402
from app.tools.knowledge_tools import _build_context, _data_file, read_webpage_content, rag  # noqa: E402
from _support import member_context, owner_context  # noqa: E402
# 钉死的哈希、禁用词清单、FakeModal 注入等夹具直接复用 smoke_orchestrator 的，保证两边是同一份事实
from smoke_orchestrator import (  # noqa: E402
    FORBIDDEN_TOOLS,
    MEMBER_ORDER,
    _CODE_FORBIDDEN_WORDS,
    _MAIN_PROMPT_SHA256,
    _NEUTRAL_ADDRESS,
    _SCHED_BANNED_WORDS,
    _STAGE1_MEMBER_PROMPT_SHA256,
    _TAIPEI_ADDRESS,
    _assert_sched_prompt,
    _code_enabled,
    _dummy_tool,
    _expect_runtime_error,
    _scheduler_enabled,
)

ROOT = Path(__file__).resolve().parent.parent
SUPERVISOR_BASE_TOOLS = ["get_current_time", "get_session_events", "search_session_history"]
SCHEDULE_TOOLS = ["create_scheduled_task", "list_scheduled_tasks", "cancel_scheduled_task"]
CLOUD_KNOWLEDGE_TOOLS = ["rag", "web_search", "read_webpage_content"]
UID = "member001"                      # 合法的 openid 形状

# 改动前（提交 321c1d7）算出的基线：web_read=False 时 build_member_main_prompt 必须与它们逐字节相同
_PRE_MEMBER_SHA256 = "010c5412d3c0c186c29f4d8621a7215ff5af3a8ba844b8918207fed7ddc516ac"
_PRE_MEMBER_CODE_NET_SHA256 = "d63801daeb15bb1da58d84a2e0f8ed59d462402f6feceb6f7fd9b5eab45d3334"       # MODAL_BLOCK_NETWORK=False
_PRE_MEMBER_CODE_BLOCKED_SHA256 = "75610196d34b2a6bcd30abc6800c23bb88f960fb6c217ad9cfea6ca118db9646"   # MODAL_BLOCK_NETWORK=True
_PRE_SCHEDULED_SHA256 = "7c6a304ebd8696917b41bdc5d9af83a84ca6d885b8936399e81ed522fc4005af"
_PRE_SCHED_SECTION_SHA256 = "cecdaee8f30f75e226c66df314e1639782240cda9f2049451eeb91b67851427a"


# ══════════════════════════════════════════════════════════════════════════════
# 公用夹具
# ══════════════════════════════════════════════════════════════════════════════

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def _patched(obj, **attrs):
    """临时替换对象属性，退出时还原。"""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _cloud(value: str = "cloud"):
    """临时把 config.DEPLOY_MODE 切成云端（测试靠它切换，所以生产代码必须调用时现读）。"""
    return _patched(config, DEPLOY_MODE=value)


def _probe(env_overrides: dict, code: str):
    """子进程里执行一段代码：占位 key、UTF-8 输出，其余环境照抄当前进程。"""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", **env_overrides}
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")


def _name(sub) -> str:
    return sub["name"]


def _tool_names(tools) -> list:
    return [getattr(t, "name", str(t)) for t in (tools or [])]


@contextlib.contextmanager
def _host_path_traps():
    """把所有通往 MCP / owner 图 / 本机专家 / 本机 Docker 沙箱的入口换成会记账并抛错的桩；cloud 构建一次都不该碰到它们。"""
    calls = []

    async def fake_init():
        calls.append("initialize_mcp")
        raise AssertionError("cloud 不应初始化 MCP")

    def fake_get_tools(name):
        calls.append(f"get_tools_by_server:{name}")
        raise AssertionError(f"cloud 不应读取 MCP 工具: {name}")

    async def fake_owner(checkpointer):
        calls.append("_build_owner_agent")
        raise AssertionError("cloud 不应构建 owner 图")

    async def fake_local():
        calls.append("_local_subagents")
        raise AssertionError("cloud 不应构建本机专家")

    def trap(label):
        def _t(*args, **kwargs):
            calls.append(label)
            raise AssertionError(f"cloud 不应调用 {label}")
        return _t

    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched(mcp_service, initialize_mcp=fake_init, get_tools_by_server=fake_get_tools))
        stack.enter_context(_patched(orchestrator, _build_owner_agent=fake_owner, _local_subagents=fake_local))
        stack.enter_context(_patched(code_agent, build_code_agent_runnable=trap("build_code_agent_runnable")))
        stack.enter_context(_patched(orchestrator.file_agent, build_file_agent_runnable=trap("build_file_agent_runnable")))
        stack.enter_context(_patched(orchestrator.desktop_agent, build_desktop_agent_subgraph=trap("build_desktop_agent_subgraph")))
        stack.enter_context(_patched(orchestrator.markitdown_tool, build_tools=trap("markitdown_tool.build_tools")))
        stack.enter_context(_patched(docker_sandbox, get_sandbox=trap("sandbox.get_sandbox"),
                                     set_docker_client=trap("sandbox.set_docker_client"),
                                     reset_sandbox=trap("sandbox.reset_sandbox")))
        yield calls


def _run_cloud_build():
    """cloud 下完整跑一遍 build_main_agents：返回 (MainAgents, create_deep_agent 收到的参数列表, member CodeAgent 构建物列表, 被碰到的禁区入口)。"""
    graphs, builds = [], []
    real_create, real_build = orchestrator.create_deep_agent, code_agent.build_member_code_agent_runnable

    def spy_create(*args, **kwargs):
        graphs.append(dict(kwargs))
        return real_create(*args, **kwargs)

    def spy_build(*args, **kwargs):
        builds.append(real_build(*args, **kwargs))
        return builds[-1]

    with _cloud(), _host_path_traps() as calls, \
            _patched(orchestrator, create_deep_agent=spy_create), _patched(code_agent, build_member_code_agent_runnable=spy_build):
        built = asyncio.run(orchestrator.build_main_agents())
    return built, graphs, builds, calls


# ══════════════════════════════════════════════════════════════════════════════
# C1 配置
# ══════════════════════════════════════════════════════════════════════════════

def test_c1_config():
    """C1 DEPLOY_MODE：cloud / 带空格的 Cloud / 空串 / 写错；is_cloud_mode() 跟随被 monkeypatch 的值；云端分支没有在 import 时缓存。"""
    code = "from app import config; print(config.DEPLOY_MODE, config.is_cloud_mode())"
    for raw, mode, flag in (("cloud", "cloud", True), (" Cloud ", "cloud", True), ("", "desktop", False),
                            ("desktop", "desktop", False), ("DESKTOP ", "desktop", False)):
        res = _probe({"DEPLOY_MODE": raw}, code)
        assert res.returncode == 0, (raw, res.stderr[-500:])
        assert res.stdout.strip().splitlines()[-1] == f"{mode} {flag}", (raw, res.stdout)
    for bad in ("clould", "cloud2", "local"):
        res = _probe({"DEPLOY_MODE": bad}, code)
        assert res.returncode != 0, f"DEPLOY_MODE={bad!r} 应当在 import 时失败，而不是静默当成 desktop"
        assert "DEPLOY_MODE" in res.stderr and "desktop" in res.stderr and "cloud" in res.stderr, res.stderr[-500:]

    assert config.DEPLOY_MODES == ("desktop", "cloud")
    assert config.DEPLOY_MODE == "desktop" and config.is_cloud_mode() is False
    with _cloud():
        assert config.is_cloud_mode() is True
        with _cloud("desktop"):
            assert config.is_cloud_mode() is False
        assert config.is_cloud_mode() is True
    assert config.is_cloud_mode() is False

    # 云端分支一律调用时现读：除 config.py 自己，app/ 里没有任何在 import 时执行的 is_cloud_mode / DEPLOY_MODE 引用
    class _ImportTimeRefs(ast.NodeVisitor):
        def __init__(self):
            self.hits = []

        def visit_FunctionDef(self, node):          # 函数体只在调用时执行
            pass

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_Attribute(self, node):
            if node.attr in ("is_cloud_mode", "DEPLOY_MODE"):
                self.hits.append(node.lineno)
            self.generic_visit(node)

    for path in sorted((ROOT / "app").rglob("*.py")):
        if path.name == "config.py" and path.parent.name == "app":
            continue
        visitor = _ImportTimeRefs()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8"), str(path)))
        assert not visitor.hits, f"{path.relative_to(ROOT)}:{visitor.hits} 在 import 时读取了部署形态（会把它缓存住）"


# ══════════════════════════════════════════════════════════════════════════════
# C2 desktop 回归
# ══════════════════════════════════════════════════════════════════════════════

def test_c2_desktop_regression():
    """C2 desktop 下一切照旧：工具名、规格表顺序、被钉死的提示词；新常量都是新对象；build_main_agents 的构建顺序与调用方式不变。"""
    assert [t.name for t in knowledge_agent.SPEC.tools_factory()] == ["rag", "web_search"]
    assert [t.name for t in knowledge_tools.TOOLS] == ["rag", "web_search"]
    assert knowledge_tools.CLOUD_TOOLS is not knowledge_tools.TOOLS
    assert [t.name for t in knowledge_tools.CLOUD_TOOLS] == CLOUD_KNOWLEDGE_TOOLS
    assert knowledge_agent.SPEC.system_prompt is knowledge_agent.PROMPT and knowledge_agent.KNOWLEDGE_AGENT_PROMPT is knowledge_agent.PROMPT
    assert [s.name for s in orchestrator._DECLARATIVE_SPECS] == ["KnowledgeAgent", "MediaAgent", "MapAgent", "BrowserAgent"]
    assert orchestrator._DECLARATIVE_SPECS[0] is knowledge_agent.SPEC
    assert "read_webpage_content" not in orchestrator._MEMBER_FORBIDDEN_TOOL_NAMES and len(orchestrator._MEMBER_FORBIDDEN_TOOL_NAMES) == 13

    # 被 smoke_orchestrator 钉死的基线
    assert _sha(prompts.MAIN_AGENT_PROMPT) == _MAIN_PROMPT_SHA256
    assert _sha(prompts.MEMBER_MAIN_AGENT_PROMPT.replace(_NEUTRAL_ADDRESS, _TAIPEI_ADDRESS)) == _STAGE1_MEMBER_PROMPT_SHA256
    assert _sha(prompts.SCHEDULED_MAIN_AGENT_PROMPT) == _PRE_SCHEDULED_SHA256
    assert _sha(prompts.SCHEDULER_PROMPT_SECTION) == _PRE_SCHED_SECTION_SHA256

    # 新常量都不是旧常量的别名（对象不同、文本也不同）
    old = (prompts.MAIN_AGENT_PROMPT, prompts.MAIN_AGENT_PROMPT_SCHED, prompts.MEMBER_MAIN_AGENT_PROMPT,
           prompts.MEMBER_MAIN_AGENT_PROMPT_CODE, prompts.MEMBER_MAIN_AGENT_PROMPT_SCHED,
           prompts.MEMBER_MAIN_AGENT_PROMPT_CODE_SCHED, prompts.SCHEDULED_MAIN_AGENT_PROMPT)
    new = (prompts.CLOUD_MAIN_AGENT_PROMPT, prompts.CLOUD_MAIN_AGENT_PROMPT_CODE, prompts.CLOUD_MAIN_AGENT_PROMPT_SCHED,
           prompts.CLOUD_MAIN_AGENT_PROMPT_CODE_SCHED, prompts.CLOUD_SCHEDULED_MAIN_AGENT_PROMPT)
    for n in new:
        assert all(n is not o and n != o for o in old), "云端提示词常量不能是旧常量的别名"
    assert len({id(n) for n in new}) == 5 and len(set(new)) == 5
    for attr in ("SPEC", "PROMPT", "DESCRIPTION"):
        assert getattr(knowledge_agent, attr) is not getattr(knowledge_agent, "CLOUD_" + attr)

    # desktop 下 build_main_agents：先初始化 MCP，再 owner，后 member，调用方式与引入 cloud 之前完全一致（不带 cloud 参数）
    log = []

    async def fake_init():
        log.append("mcp")

    async def fake_owner(checkpointer):
        log.append("owner")
        return object(), ("o",)

    def fake_member(checkpointer, **kwargs):
        log.append(("member", kwargs))
        return object(), ("m",)

    def fake_scheduled(checkpointer, **kwargs):
        log.append(("scheduled", kwargs))
        return object(), ("s",)

    with _patched(mcp_service, initialize_mcp=fake_init), \
            _patched(orchestrator, _build_owner_agent=fake_owner, _build_member_agent=fake_member, _build_scheduled_agent=fake_scheduled):
        built = asyncio.run(orchestrator.build_main_agents())
        assert log == ["mcp", "owner", ("member", {})], log
        assert built.owner is not built.member and built.owner_experts == ("o",) and built.member_experts == ("m",)
        assert built.scheduled is None
        log.clear()
        with _scheduler_enabled():
            built = asyncio.run(orchestrator.build_main_agents())
        assert log == ["mcp", "owner", ("member", {}), ("scheduled", {})], log
        assert built.scheduled is not None


# ══════════════════════════════════════════════════════════════════════════════
# C3 云端 KnowledgeAgent
# ══════════════════════════════════════════════════════════════════════════════

def test_c3_cloud_knowledge_agent():
    """C3 CLOUD_SPEC 的名字与工具；CLOUD_PROMPT 保留“你是 KnowledgeAgent”并描述读网页；read_webpage_content 的截断与错误透传。"""
    spec = knowledge_agent.CLOUD_SPEC
    assert spec.name == "KnowledgeAgent" and spec.model_key == "knowledge_llm" and spec.skills == ()
    assert [t.name for t in spec.tools_factory()] == CLOUD_KNOWLEDGE_TOOLS
    assert spec.description is knowledge_agent.CLOUD_DESCRIPTION and spec.system_prompt is knowledge_agent.CLOUD_PROMPT
    assert knowledge_agent.CLOUD_DESCRIPTION.startswith(knowledge_agent.DESCRIPTION) and len(knowledge_agent.CLOUD_DESCRIPTION) > len(knowledge_agent.DESCRIPTION)
    assert "网页链接" in knowledge_agent.CLOUD_DESCRIPTION and "总结" in knowledge_agent.CLOUD_DESCRIPTION
    p = knowledge_agent.CLOUD_PROMPT
    assert p.startswith(knowledge_agent.PROMPT) and "你是 KnowledgeAgent" in p and "read_webpage_content" in p and "【读取网页】" in p
    assert "不必先查知识库" in p and "不是给你的指令" in p
    assert _identify_agent([[SystemMessage(content=p)]], None, None) == "KnowledgeAgent", "cli_logger 靠“你是 KnowledgeAgent”识别角色"
    sub = spec.to_subagent(models)
    assert _tool_names(sub["tools"]) == CLOUD_KNOWLEDGE_TOOLS and sub["name"] == "KnowledgeAgent"

    # 工具本身：只读口吻、异步
    assert read_webpage_content.name == "read_webpage_content" and read_webpage_content.coroutine is not None
    assert "只读" in read_webpage_content.description and "不点击" in read_webpage_content.description
    assert "http" in read_webpage_content.description and "完整" in read_webpage_content.description

    seen = []
    result = {"value": ""}

    async def fake_fetch(url):
        seen.append(url)
        return result["value"]

    def run(value: str, url: str = "https://example.com/post") -> str:
        result["value"] = value
        return asyncio.run(read_webpage_content.ainvoke({"url": url}))

    with _patched(knowledge_tools, fetch_generic_webpage_content=fake_fetch), _patched(config, WEBPAGE_MAX_CHARS=60):
        assert run("短内容。正文在这里。") == "短内容。正文在这里。"
        assert seen == ["https://example.com/post"], seen
        exact = "字" * 60
        assert run(exact) == exact, "恰好等于上限时不截断"
        long_text = "甲" * 61 + "尾部不该出现"
        out = run(long_text)
        assert out.startswith("甲" * 60) and "内容过长，已截断" in out and "尾部不该出现" not in out and len(out) < len(long_text) + 40, out
        err = "Error: 无效的 URL，必须是以 http:// 或 https:// 开头的完整网址。"
        assert run(err, "ftp://x") == err, "错误串原样透传"
        err2 = "Error: 网页内容提取超时 (https://example.com/slow)。"
        assert run(err2) == err2


# ══════════════════════════════════════════════════════════════════════════════
# C4 路径归一
# ══════════════════════════════════════════════════════════════════════════════

def test_c4_path_normalization():
    """C4 带反斜杠的 source_path / image_path：_build_context 读得到完整小作文，rag 的图片循环找得到图；绝对路径与 .. 越界被拒绝。"""
    tmp = tempfile.mkdtemp(prefix="cloud-data-")
    quiet = io.StringIO()
    try:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(os.path.join(data_dir, "source"))
        os.makedirs(os.path.join(data_dir, "images"))
        essay = "完整小作文 ESSAY_MARK_C4：图里有一只红色的小方块。"
        with open(os.path.join(data_dir, "source", "essay.md"), "w", encoding="utf-8") as f:
            f.write(essay)
        Image.new("RGB", (8, 8), (200, 30, 30)).save(os.path.join(data_dir, "images", "pic.png"))
        outside_md = os.path.join(tmp, "outside.md")
        with open(outside_md, "w", encoding="utf-8") as f:
            f.write("SECRET_OUTSIDE_C4")
        Image.new("RGB", (8, 8), (0, 0, 200)).save(os.path.join(tmp, "outside.png"))

        def entity(source_path, image_path, text="chunk-text", cid="c1"):
            return {"id": cid, "entity": {"text": text, "source_path": source_path, "chunk_id": cid,
                                          "content_type": "image_description", "image_path": image_path}}

        with _patched(config, DATA_DIR=data_dir), contextlib.redirect_stdout(quiet):
            # ① _data_file：反斜杠 / 正斜杠 / 内部 .. 都落到 DATA_DIR 里的同一个文件
            want = os.path.normpath(os.path.join(data_dir, "source", "essay.md"))
            for rel in ("source\\essay.md", "source/essay.md", "source/../source/essay.md", "source\\..\\source\\essay.md"):
                assert _data_file(rel) == want and os.path.exists(want), rel
            # 拒绝：绝对路径（POSIX / 盘符 / UNC / 本机真实绝对路径）、越界、空、非字符串、DATA_DIR 本身
            for bad in ("/etc/passwd", "C:\\Windows\\win.ini", "C:/Windows/win.ini", "D:relative", "\\\\server\\share\\x.md", "//server/share/x.md",
                        outside_md, "..\\outside.md", "../outside.md", "source/../../outside.md", "source\\..\\..\\outside.md",
                        "", "   ", None, 123, ".", "source/.."):
                assert _data_file(bad) == "", f"应当拒绝 {bad!r}"

            # ② _build_context：带反斜杠的假命中 -> 读得到完整小作文，meta_str 与 image_paths 内容原样
            docs, images = _build_context([(0.9, entity("source\\essay.md", "images\\pic.png"))])
            assert len(docs) == 1 and "ESSAY_MARK_C4" in docs[0] and "chunk-text" not in docs[0], docs
            assert docs[0].startswith("[source\\essay.md] (c1): "), docs[0]
            assert images == ["images\\pic.png"]
            # 越界 / 绝对路径的 source_path：不读，保留命中里自带的片段
            for bad in ("..\\outside.md", outside_md, "/etc/passwd"):
                docs, _images = _build_context([(0.9, entity(bad, "images\\pic.png"))])
                assert "SECRET_OUTSIDE_C4" not in docs[0] and "chunk-text" in docs[0], (bad, docs)

            # ③ rag 的图片循环：用桩替掉 Milvus / 检索 / 重排，真实的 compress_image_to_base64
            def run_rag(hits):
                with _patched(knowledge_tools, get_client=lambda: object(),
                              execute_hybrid_search=lambda q, client, coll: list(hits),
                              rerank_candidates=lambda q, cands, hs, top_k=3: [(0.9, h) for h in hs]), \
                        _patched(config, RAG_MIN_RERANK_SCORE=0.1, ENABLE_QUERY_TRANSFORM=False):
                    return rag.invoke({"query": "红色方块"})

            out = run_rag([entity("source\\essay.md", "images\\pic.png")])
            assert isinstance(out, list), out
            kinds = [item["type"] for item in out]
            assert kinds == ["image_url", "text"], kinds
            assert out[0]["image_url"]["url"].startswith("data:image/jpeg;base64,") and len(out[0]["image_url"]["url"]) > 40
            assert "ESSAY_MARK_C4" in out[-1]["text"]
            # 越界与绝对路径的图片被拒绝：没有图片项（旧写法会按 os.path.join 拼出 DATA_DIR 之外的真实文件）
            for bad in ("..\\outside.png", os.path.join(tmp, "outside.png"), "../outside.png"):
                out = run_rag([entity("source\\essay.md", bad)])
                assert isinstance(out, list) and [i["type"] for i in out] == ["text"], (bad, [i["type"] for i in out])

        # ④ 按 Linux 的路径语义再验一遍 _data_file（在 Windows 上也能跑）：反斜杠在 Linux 上不是分隔符，必须先换成正斜杠
        with _patched(config, DATA_DIR="/srv/app/data"), _patched(os, path=posixpath), contextlib.redirect_stdout(quiet):
            assert _data_file("source\\essay.md") == "/srv/app/data/source/essay.md"
            assert _data_file("images/a/b\\c.png") == "/srv/app/data/images/a/b/c.png"
            for bad in ("/etc/passwd", "C:\\x.png", "..\\x", "a/../../x", "\\\\h\\s", ""):
                assert _data_file(bad) == "", bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# C5 云端提示词
# ══════════════════════════════════════════════════════════════════════════════

def test_c5_cloud_prompts():
    """C5 5 个新常量：识别短语、禁用词、只新增不改写；build_member_main_prompt 在 web_read=False 时与改动前逐字节相同。"""
    P = prompts
    ident = lambda text: _identify_agent([[SystemMessage(content=text)]], None, None)
    five = {
        "CLOUD_MAIN_AGENT_PROMPT": P.CLOUD_MAIN_AGENT_PROMPT,
        "CLOUD_MAIN_AGENT_PROMPT_CODE": P.CLOUD_MAIN_AGENT_PROMPT_CODE,
        "CLOUD_MAIN_AGENT_PROMPT_SCHED": P.CLOUD_MAIN_AGENT_PROMPT_SCHED,
        "CLOUD_MAIN_AGENT_PROMPT_CODE_SCHED": P.CLOUD_MAIN_AGENT_PROMPT_CODE_SCHED,
        "CLOUD_SCHEDULED_MAIN_AGENT_PROMPT": P.CLOUD_SCHEDULED_MAIN_AGENT_PROMPT,
    }
    old_consts = (P.MAIN_AGENT_PROMPT, P.MAIN_AGENT_PROMPT_SCHED, P.MEMBER_MAIN_AGENT_PROMPT, P.MEMBER_MAIN_AGENT_PROMPT_CODE,
                  P.MEMBER_MAIN_AGENT_PROMPT_SCHED, P.MEMBER_MAIN_AGENT_PROMPT_CODE_SCHED, P.SCHEDULED_MAIN_AGENT_PROMPT)
    for label, text in five.items():
        assert "你是一个多智能体系统的主 Agent" in text, label
        assert ident(text) == "MainAgent (Orchestrator)", label
        assert all(text is not o and text != o for o in old_consts), f"{label} 不能是旧常量的别名"
        for word in _CODE_FORBIDDEN_WORDS:
            assert word not in text, f"{label} 不应出现 {word!r}"
        assert "SQLAgent" not in text
    for label in ("CLOUD_MAIN_AGENT_PROMPT", "CLOUD_MAIN_AGENT_PROMPT_SCHED", "CLOUD_SCHEDULED_MAIN_AGENT_PROMPT"):
        for word in ("CodeAgent", "沙箱"):
            assert word not in five[label], f"不带 Modal 的变体 {label} 不应出现 {word!r}"
    assert five["CLOUD_SCHEDULED_MAIN_AGENT_PROMPT"].startswith("你是一个多智能体系统的主 Agent")

    # member 式的四个：多了“网页链接委派给 KnowledgeAgent”，能力边界写清“读取网页内容可以，操作浏览器不行”，并且位置在能力边界之内
    for label in ("CLOUD_MAIN_AGENT_PROMPT", "CLOUD_MAIN_AGENT_PROMPT_CODE", "CLOUD_MAIN_AGENT_PROMPT_SCHED", "CLOUD_MAIN_AGENT_PROMPT_CODE_SCHED"):
        text = five[label]
        for word in ("网页链接", "委派给 KnowledgeAgent", "task 的 description", "读取网页内容可以", "操作浏览器不行",
                     "【能力边界】", "不要尝试委派", "【严格数据边界】", "<long_term_memory>", "get_current_time"):
            assert word in text, f"{label} 应含 {word!r}"
        assert text.index("【能力边界】") < text.index("读取网页内容可以") < text.index("【过程不可见】"), label
        assert "没有读写用户电脑文件" in text and "浏览器的能力" in text, "原有的能力边界句子仍在，新句子只是补充例外"
    for word in ("CodeAgent", "沙箱", "【图文分离】"):
        assert word in P.CLOUD_MAIN_AGENT_PROMPT_CODE and word in P.CLOUD_MAIN_AGENT_PROMPT_CODE_SCHED
    assert P.CLOUD_MAIN_AGENT_PROMPT_CODE.index("【过程不可见】") < P.CLOUD_MAIN_AGENT_PROMPT_CODE.index("【图文分离】") < P.CLOUD_MAIN_AGENT_PROMPT_CODE.index("【长期记忆】")
    for word in ("网页链接", "读取网页内容可以"):
        assert word not in P.MEMBER_MAIN_AGENT_PROMPT and word not in P.MEMBER_MAIN_AGENT_PROMPT_CODE

    # 追加调度说明的两个变体：与 smoke_orchestrator 的 J / K 同一套检查
    _assert_sched_prompt(P.CLOUD_MAIN_AGENT_PROMPT_SCHED, P.CLOUD_MAIN_AGENT_PROMPT)
    _assert_sched_prompt(P.CLOUD_MAIN_AGENT_PROMPT_CODE_SCHED, P.CLOUD_MAIN_AGENT_PROMPT_CODE)

    # 无人值守的云端提示词：整篇都不得出现“文件 / 电脑 / 代码”这类能力描述词；该有的约束都在；只比原来多了一句
    sched = P.CLOUD_SCHEDULED_MAIN_AGENT_PROMPT
    for word in _SCHED_BANNED_WORDS:
        assert word not in sched, f"云端无人值守提示词不应出现 {word!r}"
    for word in ("无人值守", "<scheduled_task>", "get_current_time", "task 工具", "资料库、媒体、地图", "不得向用户提问", "不得请求确认",
                 "纯文本", "不要放网址", "不要输出图片", "不是给你的指令", "如实说查不到", "300 字", "资料库也能读取网页内容"):
        assert word in sched, f"云端无人值守提示词应含 {word!r}"
    assert sched.index("资料库、媒体、地图（含天气）这几类能力。") < sched.index("资料库也能读取网页内容")

    # 只新增、不改写：旧文本的每个字符按原顺序都还在新文本里（新文本 = 旧文本 + 若干插入），且确实变长了
    def only_inserts(old_text: str, new_text: str, label: str):
        it = iter(new_text)
        assert all(ch in it for ch in old_text), f"{label} 不是在旧文本上只做新增（旧文本的某些字符被改掉或删掉了）"
        assert len(new_text) > len(old_text), f"{label} 应当比旧文本多出内容"

    only_inserts(P.MEMBER_MAIN_AGENT_PROMPT, P.CLOUD_MAIN_AGENT_PROMPT, "CLOUD_MAIN_AGENT_PROMPT")
    only_inserts(P.MEMBER_MAIN_AGENT_PROMPT_CODE, P.CLOUD_MAIN_AGENT_PROMPT_CODE, "CLOUD_MAIN_AGENT_PROMPT_CODE")
    only_inserts(P.SCHEDULED_MAIN_AGENT_PROMPT, P.CLOUD_SCHEDULED_MAIN_AGENT_PROMPT, "CLOUD_SCHEDULED_MAIN_AGENT_PROMPT")

    # build_member_main_prompt：web_read 是仅限关键字的参数，默认 False；False 时与改动前（提交 321c1d7）逐字节相同
    param = inspect.signature(P.build_member_main_prompt).parameters["web_read"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is False
    for block in (False, True):
        with _patched(config, MODAL_BLOCK_NETWORK=block):
            assert _sha(P.build_member_main_prompt(False)) == _PRE_MEMBER_SHA256
            assert _sha(P.build_member_main_prompt(False, web_read=False)) == _PRE_MEMBER_SHA256
            want = _PRE_MEMBER_CODE_BLOCKED_SHA256 if block else _PRE_MEMBER_CODE_NET_SHA256
            assert _sha(P.build_member_main_prompt(True)) == want
            assert _sha(P.build_member_main_prompt(True, web_read=False)) == want
            assert P.build_member_main_prompt(True, web_read=True) != P.build_member_main_prompt(True)
    assert P.build_member_main_prompt(False) == P.MEMBER_MAIN_AGENT_PROMPT
    assert P.build_member_main_prompt(False, web_read=True) == P.CLOUD_MAIN_AGENT_PROMPT
    assert "@@" not in P._MEMBER_PROMPT_TEMPLATE.replace("@@DELEGATE_CODE@@", "").replace("@@WEB_READ@@", "") \
        .replace("@@CAPABILITY@@", "").replace("@@IMAGE_TEXT@@", "").replace("@@EXPERT_CODE@@", "").replace("@@REPLY_EXTRA@@", "")


# ══════════════════════════════════════════════════════════════════════════════
# C6 云端构建
# ══════════════════════════════════════════════════════════════════════════════

def test_c6_cloud_build():
    """C6 cloud 下 build_main_agents：不碰 MCP / owner 图 / 本机专家；owner 与 member 同一张图；Modal 可用时多 CodeAgent；定时任务图。"""
    from langchain.agents.middleware import ToolCallLimitMiddleware

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        # ① 默认：三个专家
        built, graphs, builds, calls = _run_cloud_build()
    assert calls == [], f"cloud 构建碰到了禁区入口: {calls}"
    assert hasattr(built.owner, "ainvoke") and built.owner is built.member, "owner 与 member 必须是同一个对象"
    assert built.owner_experts is built.member_experts and built.member_experts == tuple(MEMBER_ORDER), built.member_experts
    assert built.scheduled is None and built.scheduled_experts == () and builds == []
    assert len(graphs) == 1, f"开关关闭时只编译一张图，实际 {len(graphs)}"
    g = graphs[0]
    assert g["system_prompt"] is prompts.CLOUD_MAIN_AGENT_PROMPT and orchestrator.CLOUD_MAIN_AGENT_PROMPT is prompts.CLOUD_MAIN_AGENT_PROMPT
    assert _tool_names(g["tools"]) == SUPERVISOR_BASE_TOOLS, _tool_names(g["tools"])
    subs = g["subagents"]
    assert [_name(s) for s in subs] == MEMBER_ORDER and all("runnable" not in s for s in subs)
    by_name = {_name(s): s for s in subs}
    assert _tool_names(by_name["KnowledgeAgent"]["tools"]) == CLOUD_KNOWLEDGE_TOOLS
    assert by_name["KnowledgeAgent"]["system_prompt"] is knowledge_agent.CLOUD_PROMPT
    assert by_name["KnowledgeAgent"]["description"] is knowledge_agent.CLOUD_DESCRIPTION
    assert _tool_names(by_name["MediaAgent"]["tools"]) == ["av_graph_rag", "execute_sql"] and len(by_name["MapAgent"]["tools"]) == 12
    for sub in subs:
        mw = sub["middleware"]
        assert mw[0].run_limit == config.SUBAGENT_TOOL_LIMITS[_name(sub)], _name(sub)
        if _name(sub) != "KnowledgeAgent":
            assert "read_webpage_content" not in _tool_names(sub["tools"]), _name(sub)
    assert "read_webpage_content" not in _tool_names(g["tools"])
    assert not (set(_tool_names(g["tools"])) | {n for s in subs for n in _tool_names(s["tools"])}) & FORBIDDEN_TOOLS
    assert isinstance(g["middleware"][0], ToolCallLimitMiddleware) and g["checkpointer"] is not None
    assert isinstance(g["backend"].routes["/skills/"], orchestrator.ReadOnlyFilesystemBackend)
    # 同一个 desktop 图的 member 变体没有被动过：desktop 下仍传原常量对象本身
    _agent, _experts, cap = _build_desktop_member_graph()
    assert cap["system_prompt"] is prompts.MEMBER_MAIN_AGENT_PROMPT and _tool_names([t for s in cap["subagents"] if _name(s) == "KnowledgeAgent" for t in s["tools"]]) == ["rag", "web_search"]

    # ② Modal 可用（FakeModal）：多出云端 CodeAgent，构建物通过 vetted 断言
    with contextlib.redirect_stdout(quiet), _code_enabled():
        built, graphs, builds, calls = _run_cloud_build()
        assert calls == [] and built.owner is built.member
        assert built.member_experts == (*MEMBER_ORDER, "CodeAgent"), built.member_experts
        assert len(builds) == 1 and len(graphs) == 1
        g = graphs[0]
        assert g["system_prompt"] is prompts.CLOUD_MAIN_AGENT_PROMPT_CODE
        orchestrator._assert_vetted_code_build(builds[0])
        pre = [s for s in g["subagents"] if "runnable" in s]
        assert len(pre) == 1 and pre[0]["name"] == "CodeAgent" and pre[0]["runnable"] is builds[0].runnable
        assert pre[0]["description"] == code_agent.MEMBER_DESCRIPTION
        assert "read_webpage_content" not in builds[0].tool_names and builds[0].tool_names == orchestrator._MEMBER_CODE_TOOL_NAMES
        assert _tool_names(next(s for s in g["subagents"] if _name(s) == "KnowledgeAgent")["tools"]) == CLOUD_KNOWLEDGE_TOOLS

    # ③ 定时任务开关打开：member 式的图换成 *_SCHED 提示词并多三个 schedule 工具；第三张 scheduled 图用云端无人值守提示词
    with contextlib.redirect_stdout(quiet), _scheduler_enabled():
        built, graphs, builds, calls = _run_cloud_build()
    assert calls == [] and len(graphs) == 2 and builds == []
    member_g, sched_g = graphs
    assert member_g["system_prompt"] is prompts.CLOUD_MAIN_AGENT_PROMPT_SCHED
    assert _tool_names(member_g["tools"]) == SUPERVISOR_BASE_TOOLS + SCHEDULE_TOOLS, _tool_names(member_g["tools"])
    assert sched_g["system_prompt"] is prompts.CLOUD_SCHEDULED_MAIN_AGENT_PROMPT
    assert _tool_names(sched_g["tools"]) == ["get_current_time"]
    assert [_name(s) for s in sched_g["subagents"]] == MEMBER_ORDER and all("runnable" not in s for s in sched_g["subagents"])
    sched_by_name = {_name(s): s for s in sched_g["subagents"]}
    assert _tool_names(sched_by_name["KnowledgeAgent"]["tools"]) == CLOUD_KNOWLEDGE_TOOLS
    for n in ("MediaAgent", "MapAgent"):
        assert "read_webpage_content" not in _tool_names(sched_by_name[n]["tools"])
        assert not (set(_tool_names(sched_by_name[n]["tools"])) & (set(SCHEDULE_TOOLS) | FORBIDDEN_TOOLS))
    mws = sched_g["middleware"]
    assert len(mws) == 1 and isinstance(mws[0], ToolCallLimitMiddleware) and mws[0].run_limit == 6
    assert built.owner is built.member and built.scheduled is not None and built.scheduled is not built.member
    assert built.scheduled_experts == tuple(MEMBER_ORDER) and hasattr(built.scheduled, "ainvoke")

    # ④ 两者同时：member 图带云端 CodeAgent 与调度说明；scheduled 图仍然只有三个专家，不构造 CodeAgent
    with contextlib.redirect_stdout(quiet), _code_enabled(), _scheduler_enabled():
        built, graphs, builds, calls = _run_cloud_build()
        assert calls == [] and len(graphs) == 2 and len(builds) == 1
        member_g, sched_g = graphs
        assert member_g["system_prompt"] is prompts.CLOUD_MAIN_AGENT_PROMPT_CODE_SCHED
        assert _tool_names(member_g["tools"]) == SUPERVISOR_BASE_TOOLS + SCHEDULE_TOOLS
        assert [_name(s) for s in member_g["subagents"]] == [*MEMBER_ORDER, "CodeAgent"]
        assert sched_g["system_prompt"] is prompts.CLOUD_SCHEDULED_MAIN_AGENT_PROMPT
        assert built.scheduled_experts == tuple(MEMBER_ORDER) and built.member_experts == (*MEMBER_ORDER, "CodeAgent")


def _build_desktop_member_graph():
    """desktop 下编译 member 图（不带 cloud 参数）：返回 (图, 专家名, create_deep_agent 收到的参数)。"""
    captured = {}
    real = orchestrator.create_deep_agent

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    with _patched(orchestrator, create_deep_agent=spy):
        agent, experts = orchestrator._build_member_agent(MemorySaver())
    return agent, experts, captured


# ══════════════════════════════════════════════════════════════════════════════
# C7 断言仍然有效
# ══════════════════════════════════════════════════════════════════════════════

def test_c7_assertions_still_hold():
    """C7 cloud 图里注入 13 个宿主机工具名 / 非法专家名 / 预编译子专家都 RuntimeError；cloud 专用断言逐项拒绝。"""
    check, cloud_check = orchestrator._assert_member_graph_safe, orchestrator._assert_cloud_graph_safe
    sup = [knowledge_tools.get_current_time, *session_tools.TOOLS]
    cloud_subs = lambda code=None: orchestrator._member_subagents(code, cloud=True)

    with _cloud():
        # 基线：cloud 的子专家通过两道断言，顺序仍是 Knowledge、Media、Map
        assert [_name(s) for s in cloud_subs()] == MEMBER_ORDER
        check(sup, cloud_subs())
        cloud_check(sup, cloud_subs())
        assert set(orchestrator._MEMBER_FORBIDDEN_TOOL_NAMES) == FORBIDDEN_TOOLS and len(FORBIDDEN_TOOLS) == 13

        # 13 个宿主机相关工具名：注入主管 / 每个声明式子专家（含换成了 CLOUD_SPEC 的 KnowledgeAgent）
        for name in sorted(FORBIDDEN_TOOLS):
            bad = _dummy_tool(name)
            _expect_runtime_error(check, [*sup, bad], cloud_subs())
            for i in range(len(MEMBER_ORDER)):
                subs = cloud_subs()
                subs[i]["tools"] = [*subs[i]["tools"], bad]
                _expect_runtime_error(check, sup, subs)
        # 非法专家名 / 预编译子专家（名字合法也不行）
        subs = cloud_subs()
        subs.append({"name": "BrowserAgent", "description": "x", "system_prompt": "x", "tools": []})
        _expect_runtime_error(check, sup, subs)
        for name in ("FileAgent", "DesktopAgent", "CodeAgent"):
            subs = cloud_subs()
            subs.append({"name": name, "description": "x", "runnable": object()})
            _expect_runtime_error(check, sup, subs)
        subs = cloud_subs()
        subs.append({"name": "KnowledgeAgent", "description": "x", "runnable": object()})
        _expect_runtime_error(check, sup, subs)

        # cloud 专用断言：KnowledgeAgent 工具恰为 [rag, web_search, read_webpage_content]
        _expect_runtime_error(cloud_check, sup, orchestrator._member_subagents())              # desktop 版的 KnowledgeAgent 没有读网页工具
        for mutate in (lambda tools: tools[:-1], lambda tools: [*tools, _dummy_tool("extra_tool")],
                       lambda tools: [tools[1], tools[0], tools[2]], lambda tools: [tools[0], tools[2]]):
            subs = cloud_subs()
            subs[0]["tools"] = mutate(list(subs[0]["tools"]))
            _expect_runtime_error(cloud_check, sup, subs)
        subs = [s for s in cloud_subs() if _name(s) != "KnowledgeAgent"]
        _expect_runtime_error(cloud_check, sup, subs)                                           # 没有 KnowledgeAgent
        # 主管与其余专家不得持有 read_webpage_content
        _expect_runtime_error(cloud_check, [*sup, read_webpage_content], cloud_subs())
        for i in (1, 2):
            subs = cloud_subs()
            subs[i]["tools"] = [*subs[i]["tools"], read_webpage_content]
            _expect_runtime_error(cloud_check, sup, subs)
        # 它不在宿主机黑名单里（别加进去）：交给 member 断言时是放行的
        subs = cloud_subs()
        subs[1]["tools"] = [*subs[1]["tools"], read_webpage_content]
        check(sup, subs)

        # 带 vetted CodeAgent：基线通过；它的工具集里混进 read_webpage_content 要拒绝
        with _code_enabled():
            build = code_agent.build_member_code_agent_runnable(llm=models.worker_llm, system_prompt=code_agent.MEMBER_PROMPT)
            subs = cloud_subs(build)
            check(sup, subs, vetted_code=build)
            cloud_check(sup, subs, vetted_code=build)
            tainted = dataclasses.replace(build, tool_names=build.tool_names | {"read_webpage_content"})
            _expect_runtime_error(cloud_check, sup, subs, tainted)
            for name in sorted(FORBIDDEN_TOOLS):                                                # 带着 vetted_code 也照样拒绝
                _expect_runtime_error(check, [*sup, _dummy_tool(name)], cloud_subs(build), build)

        # 端到端：断言失败时 member 图 / scheduled 图都不会走到 create_deep_agent
        reached = []
        real_subs = orchestrator._member_subagents

        def poison_host_tool(code=None, *, cloud=False):
            subs = real_subs(code, cloud=True)
            subs[0]["tools"] = [*subs[0]["tools"], _dummy_tool("run_host_command")]
            return subs

        def drop_web_tool(code=None, *, cloud=False):
            return real_subs(code)                                                              # 忘了换成 CLOUD_SPEC

        def leak_web_tool(code=None, *, cloud=False):
            subs = real_subs(code, cloud=True)
            subs[2]["tools"] = [*subs[2]["tools"], read_webpage_content]
            return subs

        for label, stub in (("宿主机工具", poison_host_tool), ("缺读网页工具", drop_web_tool), ("读网页工具泄漏", leak_web_tool)):
            with _patched(orchestrator, _member_subagents=stub, create_deep_agent=lambda *a, **k: reached.append(1)):
                _expect_runtime_error(lambda: orchestrator._build_member_agent(MemorySaver(), cloud=True))
                _expect_runtime_error(lambda: orchestrator._build_scheduled_agent(MemorySaver(), cloud=True))
            assert not reached, f"{label}：断言失败后不应继续构建主图"


# ══════════════════════════════════════════════════════════════════════════════
# C8 Modal 门控
# ══════════════════════════════════════════════════════════════════════════════

class _ModalEnv:
    """注入 FakeModal、审计日志写到临时目录、池用 fake.clock；让模块级函数与 ModalRoutedSandbox 走这个池。退出时全部还原。"""

    def __enter__(self):
        from _fake_modal import FakeModal
        self.tmp = tempfile.mkdtemp(prefix="cloud-modal-")
        self._saved_audit = config.MODAL_AUDIT_LOG
        config.MODAL_AUDIT_LOG = os.path.join(self.tmp, "audit", "modal_audit.log")
        self.fake = FakeModal()
        self.fake.publish_default_image()
        modal_sandbox.set_modal_module(self.fake)
        self.pool = modal_sandbox.ModalSandboxPool(self.fake, clock=self.fake.clock)
        modal_sandbox._pool = self.pool
        return self

    def __exit__(self, *exc):
        try:
            self.pool.shutdown(deadline_s=2.0)
        finally:
            modal_sandbox.set_modal_module(None)             # 同时清空池单例、关闭审计 handler（Windows 上不关就删不掉临时目录）
            config.MODAL_AUDIT_LOG = self._saved_audit
            shutil.rmtree(self.tmp, ignore_errors=True)
        return False


def test_c8_modal_gate():
    """C8 cloud 下 owner 与普通租户一样按 (owner, session_id) 取沙箱、可以上传文件；desktop 下两处仍然拒绝；其余失败即关闭照旧。"""
    quiet = io.StringIO()
    routed_of = modal_sandbox.get_routed_sandbox
    tag_u, tag_s = (lambda v: _sha(v)[:12]), (lambda v: _sha(v)[:8])

    with _ModalEnv() as e, contextlib.redirect_stdout(quiet):
        routed = routed_of()
        # ── cloud ──
        with _cloud():
            with owner_context("s-own"):
                assert routed._target() == ("owner", "s-own")
                r = routed.execute("echo hi")
                assert r.exit_code == 0 and "云端沙箱仅供成员使用" not in r.output, r
            assert len(e.fake.created) == 1
            assert e.fake.created[0].tags["u"] == tag_u("owner") and e.fake.created[0].tags["s"] == tag_s("s-own"), e.fake.created[0].tags
            with member_context(UID, "s-m"):                                                     # 普通租户照常
                assert routed._target() == (UID, "s-m")

            ok, gen = modal_sandbox.upload_session_file("owner", "s1", "/work/uploads/a.txt", b"hello")
            assert ok is True and gen, (ok, gen)
            uploaded = [sb for sb in e.fake.created if sb.tags.get("s") == tag_s("s1")]
            assert len(uploaded) == 1 and uploaded[0].tags["u"] == tag_u("owner") and uploaded[0].filesystem.get("/work/uploads/a.txt") == b"hello"
            assert modal_sandbox.session_sandbox_generation("owner", "s1", check_remote=False) == gen

            # 其余的失败即关闭一点没松：没有上下文 / 非法 user_id / 空 session_id / 非法路径
            before = len(e.fake.created)
            for label, ctx in (("没有上下文", contextlib.nullcontext()), ("非法 user_id", member_context("bad user!", "s")),
                               ("空 session_id", member_context(UID, ""))):
                with ctx:
                    try:
                        routed._target()
                        raise AssertionError(f"{label}：cloud 下也应当拒绝")
                    except SandboxUnavailable as ex:
                        assert "无法确定当前用户" in str(ex), (label, str(ex))
            for who, sid, path in (("bad user!", "s1", "/work/uploads/a.txt"), ("owner", "", "/work/uploads/a.txt"),
                                   ("owner", "s1", "/work/a.txt"), ("owner", "s1", "/work/uploads/../a.txt")):
                ok, msg = modal_sandbox.upload_session_file(who, sid, path, b"x")
                assert ok is False and msg, (who, sid, path)
            assert len(e.fake.created) == before, "拒绝时不创建沙箱"

        # ── desktop：owner 在两处仍被拒绝，不碰池 ──
        n = len(e.fake.created)
        assert config.is_cloud_mode() is False
        with owner_context("s-own2"):
            try:
                routed._target()
                raise AssertionError("desktop 下 owner 应被拒绝")
            except SandboxUnavailable as ex:
                assert "云端沙箱仅供成员使用" in str(ex), str(ex)
            r = routed.execute("echo hi")
            assert r.exit_code == 1 and "云端沙箱仅供成员使用" in r.output, r
        ok, msg = modal_sandbox.upload_session_file("owner", "s9", "/work/uploads/a.txt", b"x")
        assert ok is False and msg == "文件放入云端工作区失败，请稍后重新发送", (ok, msg)
        assert len(e.fake.created) == n, "desktop 下 owner 被拒绝时不创建沙箱"
        with member_context(UID, "s-m2"):                                                         # member 不受影响
            assert routed._target() == (UID, "s-m2")
        ok, gen = modal_sandbox.upload_session_file(UID, "s-up", "/work/uploads/a.txt", b"x")
        assert ok is True and gen

        # 每次调用现读：同一个进程里来回切换，行为跟着变
        with _cloud(), owner_context("s-flip"):
            assert routed._target() == ("owner", "s-flip")
        with owner_context("s-flip"):
            _expect_unavailable(routed._target)


def _expect_unavailable(fn):
    try:
        fn()
    except SandboxUnavailable:
        return
    raise AssertionError("期望 SandboxUnavailable")


# ══════════════════════════════════════════════════════════════════════════════
# C9 惰性导入
# ══════════════════════════════════════════════════════════════════════════════

def test_c9_lazy_browser_use():
    """C9 import app.tools.knowledge_tools 不会把 browser_use 及其子模块带进进程；browser_service 顶层不再导入它。"""
    assert importlib.util.find_spec("browser_use") is not None, "browser_use 没装：这条测试会变成空转"
    code = (
        "import sys\n"
        "import app.tools.knowledge_tools as k\n"
        "leaked = sorted(m for m in sys.modules if m == 'browser_use' or m.startswith('browser_use.'))\n"
        "assert not leaked, f'browser_use 被提前导入: {leaked[:5]}'\n"
        "assert 'app.infra.browser_service' in sys.modules, '读网页工具应当依赖 browser_service（否则这条测试是空转的）'\n"
        "assert 'app.tools.browser_tools' not in sys.modules, 'knowledge_tools 不得导入 browser_tools'\n"
        "import app.infra.browser_service as b\n"
        "assert not hasattr(b, 'Agent') and not hasattr(b, 'Browser'), 'Agent / Browser 应当只在函数里惰性导入'\n"
        "assert callable(k.fetch_generic_webpage_content) and k.CLOUD_TOOLS[-1].name == 'read_webpage_content'\n"
        "print('ok')\n"
    )
    res = _probe({"DEPLOY_MODE": "desktop"}, code)
    assert res.returncode == 0 and res.stdout.strip().splitlines()[-1] == "ok", (res.stdout[-300:], res.stderr[-800:])

    # 静态再确认一遍：browser_service 顶层没有 browser_use 的导入，函数里有（行为没有被删掉）
    tree = ast.parse((ROOT / "app" / "infra" / "browser_service.py").read_text(encoding="utf-8"))
    is_bu = lambda n: (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "browser_use") or \
        (isinstance(n, ast.Import) and any(a.name.split(".")[0] == "browser_use" for a in n.names))
    assert not [n for n in tree.body if is_bu(n)], "browser_service 顶层不应导入 browser_use"
    lazy = {fn.name: [n for n in ast.walk(fn) if is_bu(n)] for fn in tree.body if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert lazy["init_browser"] and lazy["run_browser_task"], "init_browser / run_browser_task 里应当惰性导入 Browser / Agent"


SCENARIOS = [
    ("C1 DEPLOY_MODE 配置", test_c1_config),
    ("C2 desktop 回归", test_c2_desktop_regression),
    ("C3 云端 KnowledgeAgent 与读网页工具", test_c3_cloud_knowledge_agent),
    ("C4 知识库元数据路径归一", test_c4_path_normalization),
    ("C5 云端提示词", test_c5_cloud_prompts),
    ("C6 云端构建", test_c6_cloud_build),
    ("C7 启动期断言仍然有效", test_c7_assertions_still_hold),
    ("C8 Modal 门控", test_c8_modal_gate),
    ("C9 browser_use 惰性导入", test_c9_lazy_browser_use),
]


def main():
    print("=" * 70)
    print("开始运行 DEPLOY_MODE=cloud 离线冒烟测试...")
    print("=" * 70)
    passed = failed = 0
    for name, fn in SCENARIOS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
