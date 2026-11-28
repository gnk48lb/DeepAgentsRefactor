"""
tests/smoke_cloud_service.py

服务层适配 DEPLOY_MODE=cloud 的离线冒烟测试：不需要 API Key / Docker / MCP / 网络 / 真实 Modal。
cloud 下 owner 与 member 走同一条云端沙箱路径：文件直传 Modal、/reset /stop 失败回合只动云端沙箱、不碰本机 Docker / 宿主机命令 / 浏览器 / 宿主机 inbox。
  S1 门控：uses_host_inbox 的真值表；gate 在各种开关组合、四种身份与形态下的结果（desktop 与改动前逐项一致）
  S2 落地：cloud 下 owner 的文件进 ('owner', sid) 沙箱、宿主机没有 inbox、单文件上限仍按身份；desktop 下 owner 仍落 inbox
  S3 回执与清单：cloud 下 owner 是 member 版措辞与规则；Windows 时代的 /workspace/inbox/... 事件判为失效；desktop 输出与改动前逐字一致
  S4 chat_service 的 cloud 行为：/reset /whoami /stop、失败回合、owner 不占 member 信号量、init_service、shutdown_service
  S5 desktop 回归：同样的操作在 desktop 下与改动前一致（owner 走 Docker 与 inbox，member 走 Modal，启动日志两行专家清单）
  S6 静态检查：没有模块级的 is_cloud_mode()、没有把它的结果存进模块变量、cloud 分支不 import 本机沙箱 / 宿主机命令 / 浏览器
自成一体：夹具照抄自 smoke_qq_files.py 与 smoke_chat_service.py，不 import 它们（它们在 import 时会改 sys.modules）。
"""
import ast
import asyncio
import contextlib
import hashlib
import inspect
import io
import os
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

for _k in ("GEMINI_API_KEY", "SILICON_FLOW_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "AMAP_API_KEY", "MAGPIE_API_KEY", "TAVILY_API_KEY"):
    os.environ[_k] = "dummy"   # 占位值:本测试绝不应使用真实 key
# 本机 .env 里可能有真实的 Modal 令牌 / 定时任务开关 / DEPLOY_MODE=cloud：load_dotenv 不覆盖已有的环境变量，所以下面几行能压过它。
# 云端场景用上下文管理器临时 monkeypatch config.DEPLOY_MODE；Modal 用 FakeModal 或钩子替身，Docker / host_shell / browser_service 的入口换成绊线。
os.environ["MODAL_DISABLED"] = "true"
os.environ["SCHEDULER_ENABLED"] = "false"
os.environ["DEPLOY_MODE"] = "desktop"

import httpx  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

# ── 桩模块：必须在 import app.services.chat_service 之前安装，避免 import 时需要 API Key / Milvus（同 smoke_chat_service）──
import app  # noqa: E402
import app.agents  # noqa: E402
import app.memory  # noqa: E402
import app.services  # noqa: E402


def _stub(pkg, name, **attrs):
    m = types.ModuleType(f"{pkg.__name__}.{name}")
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[m.__name__] = m
    setattr(pkg, name, m)
    return m


async def _fake_extract_memory(user_text, *, user_id):
    return None


def _fake_build_initial_messages(user_query, history_messages=None, *, user_id):
    return [HumanMessage(content=user_query, name="user")]


_stub(app.agents, "orchestrator", build_main_agents=None)
_stub(app.memory, "injector", build_initial_messages=_fake_build_initial_messages)
_stub(app.memory, "extractor", extract_and_store_memory=_fake_extract_memory)
_stub(app.services, "startup", run_global_database_init=lambda: None)

from app import config  # noqa: E402
from app.infra import browser_service, checkpointer, host_shell, mcp_service, milvus_client, modal_sandbox, session_store  # noqa: E402
from app.infra import sandbox as docker_sandbox  # noqa: E402
from app.infra.session_store import SessionStore  # noqa: E402
from app.services import chat_service as cs  # noqa: E402
from app.services import file_staging  # noqa: E402
from app.tools import amap  # noqa: E402
from app.utils import files as F  # noqa: E402
from app.utils.files import IncomingFile  # noqa: E402
from app.utils.identity import Identity, owner_identity, resolve_platform_user  # noqa: E402
from _fake_modal import FakeModal  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MB = 1024 * 1024
UID = "member001"                      # 合法的 openid 形状
OWNER = owner_identity()
MEMBER = Identity(user_id=UID, is_owner=False, raw_id=UID)
MEMBER_B = resolve_platform_user("MEMBERBBB0002")
URL = "https://files.example.com/download?id=1"
PDF = b"%PDF-1.7\n" + b"x" * 2000
CLEAN_RESET_TEXT = "对话上下文已清空，会话记录仍保留。已开始新的对话。"


def _run(coro):
    return asyncio.run(coro)


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


def _incoming(name="report.pdf", size=0, url=URL):
    return IncomingFile(url=url, filename=name, size=size, content_type="file")


# ══════════════════════════════════════════════════════════════════════════════
# 夹具：记账的替身（绊线）、Modal 钩子替身、FakeModal 池、假 agent、临时环境
# ══════════════════════════════════════════════════════════════════════════════

# (记账名, 模块, 属性, 是否 async, desktop 下的返回值, desktop 下是否交还给真实实现, cloud 下是否绊线)
_TRIP_SPECS = (
    ("docker.reset_sandbox", docker_sandbox, "reset_sandbox", False, None, False, True),
    ("docker.shutdown_sandbox", docker_sandbox, "shutdown_sandbox", False, None, False, True),
    ("docker.interrupt_sandbox", docker_sandbox, "interrupt_sandbox", False, False, False, True),
    ("docker.get_sandbox", docker_sandbox, "get_sandbox", False, None, False, True),
    ("host.kill_all_running", host_shell, "kill_all_running", False, 2, False, True),
    ("browser.close_browser", browser_service, "close_browser", True, None, False, True),
    ("file.clear_session_inbox", file_staging, "clear_session_inbox", False, None, True, True),
    ("file.cleanup_expired_inbox", file_staging, "cleanup_expired_inbox", False, 0, False, True),
    ("file._stage_owner", file_staging, "_stage_owner", True, None, True, True),
    ("mcp.shutdown_mcp", mcp_service, "shutdown_mcp", True, None, False, False),
    ("milvus.close_client", milvus_client, "close_client", False, None, False, False),
    ("amap.aclose", amap, "aclose", True, None, False, False),
)


class _Trips:
    """把本机 Docker 沙箱 / 宿主机命令 / 浏览器 / 宿主机 inbox 的入口换成会记账的替身，记进共享的 log。
    cloud（strict=True）下这些入口一碰就抛 AssertionError（绊线）；但调用方可能 except Exception 把它吞掉，所以断言一律以 log 为准。
    desktop 下返回无害的值（kill_all_running 返回 2），其中 clear_session_inbox 与 _stage_owner 交还给真实实现，好让 desktop 的落盘照常发生。
    mcp / milvus / amap 的关闭只记账不绊线（cloud 里也会被调用，用来核对关闭顺序）。"""

    def __init__(self, log: list, *, strict: bool):
        self.log, self.strict = log, strict

    def __enter__(self):
        self._orig = []
        for name, module, attr, is_async, ret, passthrough, strict_in_cloud in _TRIP_SPECS:
            original = getattr(module, attr)
            self._orig.append((module, attr, original))
            setattr(module, attr, self._make(name, original, is_async, ret, passthrough, self.strict and strict_in_cloud))
        return self

    def __exit__(self, *exc):
        for module, attr, original in reversed(self._orig):
            setattr(module, attr, original)
        return False

    def _make(self, name, original, is_async, ret, passthrough, strict):
        log = self.log

        if is_async:
            async def fn(*args, **kwargs):
                log.append((name, args))
                if strict:
                    raise AssertionError(f"cloud 不应调用 {name}")
                if passthrough:
                    return await original(*args, **kwargs)
                return ret
        else:
            def fn(*args, **kwargs):
                log.append((name, args))
                if strict:
                    raise AssertionError(f"cloud 不应调用 {name}")
                if passthrough:
                    return original(*args, **kwargs)
                return ret
        return fn


class _ModalSpy:
    """替换 modal_sandbox 上的钩子并记进共享的 log（同 smoke_chat_service 的 _ModalSpy）。"""
    NAMES = ("reset_session", "reset_user", "interrupt_session_nowait", "shutdown_pool", "member_code_status")

    def __init__(self, log: list, status=(False, "测试里没有配置 Modal")):
        self.log, self.status = log, status

    def __enter__(self):
        self._orig = {n: getattr(modal_sandbox, n) for n in self.NAMES}

        def make(name, result=None):
            def hook(*args, **kwargs):
                self.log.append((f"modal.{name}", args))
                return result() if callable(result) else result
            return hook

        modal_sandbox.reset_session = make("reset_session")
        modal_sandbox.reset_user = make("reset_user", lambda: 0)
        modal_sandbox.interrupt_session_nowait = make("interrupt_session_nowait")
        modal_sandbox.shutdown_pool = make("shutdown_pool")
        modal_sandbox.member_code_status = make("member_code_status", lambda: self.status)
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(modal_sandbox, name, fn)
        return False


class _ModalEnv:
    """注入 FakeModal、审计日志写到临时目录、池用 fake.clock；让模块级函数与 ModalRoutedSandbox 走这个池。退出时全部还原。"""

    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="cloudsvc-modal-")
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


class _FakeState:
    def __init__(self, messages, interrupts=()):
        self.values = {"messages": list(messages)}
        self.interrupts = tuple(interrupts)


class _FakeAgent:
    """最小的主图替身：记录收到的消息；delay 模拟慢模型（可被 /stop 取消）；raises 让这一轮抛异常。"""

    def __init__(self, reply="好的", delay=0.0):
        self.reply, self.delay, self.raises = reply, delay, None
        self.invocations: list[list] = []
        self.threads: dict[str, list] = {}

    async def aget_state(self, cfg):
        return _FakeState(self.threads.get(cfg["configurable"]["thread_id"], []))

    async def ainvoke(self, inp, cfg):
        thread = self.threads.setdefault(cfg["configurable"]["thread_id"], [])
        self.invocations.append(list(inp["messages"]))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        thread.extend(inp["messages"])
        thread.append(AIMessage(content=self.reply))


class _Env:
    """临时 WORKSPACE_DIR + 临时 SessionStore + MockTransport 下载客户端 + 假主图装进 chat_service；退出时全部还原。
    cloud=True 时 owner 与 member 是同一张图（同 orchestrator 的 cloud 形态）；member_code 对应 _member_code_enabled。
    env.log 是 _Trips / _ModalSpy 共用的调用记录：[(记账名, 参数元组), ...]。"""

    def __init__(self, *, cloud: bool, member_code: bool = True, reply: str = "好的", **cfg):
        self.cloud, self.member_code, self.reply, self.cfg = cloud, member_code, reply, cfg
        self.handler = lambda request: httpx.Response(200, content=PDF)
        self.log: list = []

    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="cloudsvc-")
        self.ws = os.path.join(self.tmp, "ws")
        os.makedirs(self.ws)
        overrides = {
            "WORKSPACE_DIR": self.ws, "QQ_FILE_ENABLED": True, "QQ_FILE_MEMBER_ENABLED": True, "QQ_FILE_SAFE_NAMES": True,
            "DEPLOY_MODE": "cloud" if self.cloud else "desktop", **self.cfg,
        }
        self._saved = {k: getattr(config, k) for k in overrides}
        for k, v in overrides.items():
            setattr(config, k, v)
        self.store = SessionStore(os.path.join(self.tmp, "sessions.db"))
        session_store.set_store(self.store)
        self.requests = []
        self._orig_client = file_staging._make_client
        file_staging._make_client = self._make_client
        file_staging.reset_runtime_state()
        self.owner_agent = _FakeAgent(self.reply)
        self.member_agent = self.owner_agent if self.cloud else _FakeAgent(self.reply)
        self._saved_cs = (cs._initialized, cs._agents, cs._checkpointer)
        cs._reset_runtime_state()
        cs._agents = {"owner": self.owner_agent, "member": self.member_agent}
        cs._initialized, cs._checkpointer = True, None
        cs._member_code_enabled = self.member_code
        return self

    def __exit__(self, *exc):
        cs._reset_runtime_state()
        cs._initialized, cs._agents, cs._checkpointer = self._saved_cs
        file_staging._make_client = self._orig_client
        file_staging.reset_runtime_state()
        session_store.close_store()
        self.store.close()
        for k, v in self._saved.items():
            setattr(config, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)
        return False

    def _make_client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self._dispatch), follow_redirects=False)

    async def _dispatch(self, request):
        self.requests.append(str(request.url))
        result = self.handler(request)
        if inspect.isawaitable(result):
            result = await result
        return result

    def inbox(self, sid=None) -> Path:
        base = Path(self.ws) / "inbox"
        return base / sid if sid else base

    def events(self, sid, types=(session_store.EVENT_FILE_UPLOADED,)):
        return self.store.get_last_events(sid, limit=100, types=types)

    def calls(self, name):
        return [args for n, args in self.log if n == name]

    def touched_forbidden(self):
        """被碰过的本机资源入口（本机 Docker / 宿主机命令 / 浏览器 / 宿主机 inbox）。"""
        return [n for n, _ in self.log if n.startswith(("docker.", "host.", "browser.", "file."))]


async def _say(text, identity=OWNER, *, files=None):
    kwargs = {"files": files} if files else {}
    return await cs.process_message(text, None, identity=identity, **kwargs)


def _start(text, identity=OWNER):
    return asyncio.create_task(cs.process_message(text, None, identity=identity))


def _sid(uid="owner") -> str:
    return cs.get_active_session_id(user_id=uid)


@contextlib.contextmanager
def _quiet():
    """chat_service 的日志与异常回溯（traceback.print_exc 写 stderr）都不打到测试输出里。"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ══════════════════════════════════════════════════════════════════════════════
# 改动前的实现（照抄自提交 61a7c66），用来证明 desktop 下输出逐字不变
# ══════════════════════════════════════════════════════════════════════════════

def _old_gate(identity, *, member_code_enabled):
    if not config.QQ_FILE_ENABLED:
        return False, "当前未开启文件接收功能。"
    if not identity.is_owner:
        if not config.QQ_FILE_MEMBER_ENABLED:
            return False, "文件接收功能目前仅对管理员开放。"
        if not member_code_enabled:
            return False, "当前未启用云端代码环境，暂不支持处理文件附件（仅支持文本和图片）。"
    return True, ""


def _old_format_receipts(outcome, *, is_owner, approval_pending=False):
    safe = config.QQ_FILE_SAFE_NAMES
    lines = []
    staged = outcome.staged
    if staged:
        if is_owner:
            tail = "直接告诉我怎么处理（仅当前会话临时使用，不会存入知识库）。"
        else:
            tail = (
                f"直接告诉我怎么处理（仅当前会话临时使用；闲置约 {file_staging._idle_minutes()} 分钟或发送 /reset 后会清空，需要时请重新发送）。"
            )
        if len(staged) == 1:
            item = staged[0]
            name, size = F.receipt_name(item.display_name, safe=safe), F.format_file_size(item.size)
            if is_owner:
                lines.append(f"已暂存：{name}（{size}）。{tail}")
            else:
                lines.append(f"已收到：{name}（{size}），已放入你的云端工作区。{tail}")
        else:
            lines.append(f"已暂存 {len(staged)} 个文件：" if is_owner else f"已收到 {len(staged)} 个文件：")
            lines.extend(
                f"- {F.receipt_name(it.display_name, safe=safe)}（{F.format_file_size(it.size)}）" for it in staged
            )
            lines.append(tail if is_owner else f"已放入你的云端工作区。{tail}")
    lines.extend(outcome.errors)
    if staged and approval_pending:
        lines.append("提醒：当前还有一项待授权的操作，请先回复 Y 或 N。")
    return "\n".join(lines) if lines else "没有收到可处理的文件。"


def _old_format_block(notices, *, is_owner):
    if not notices:
        return ""
    ok = [n for n in notices if n.available]
    stale = [n for n in notices if not n.available]
    gone = "已被清理" if is_owner else "云端工作区已被回收"
    if not ok:
        lines = [
            "<uploaded_files>",
            f"用户在发这条消息之前上传过下面这些文件，但它们已经失效（{gone}），无法再读取：",
            *(file_staging._notice_line(n, with_path=False) for n in stale),
            "如果用户的问题需要这些文件：不要委派，直接告诉用户文件已失效、请重新发送；与文件无关的问题照常回答。",
            "</uploaded_files>",
        ]
        return "\n".join(lines)
    if is_owner:
        head = "用户在发这条消息之前，通过 QQ 上传了下面这些文件（仅本次会话临时使用，不会进入知识库；用户的原话在本块之后）："
    else:
        head = "用户在发这条消息之前，通过 QQ 上传了下面这些文件（仅本次会话临时使用，已放进用户的云端工作区；用户的原话在本块之后）："
    lines = ["<uploaded_files>", head, *(file_staging._notice_line(n, with_path=True) for n in ok)]
    if stale:
        lines.append(f"以下文件已失效（{gone}），不要委派，请告诉用户重新发送：")
        lines.extend(file_staging._notice_line(n, with_path=False) for n in stale)
    lines.append("")
    lines.append(file_staging._OWNER_RULES if is_owner else file_staging._MEMBER_RULES)
    lines.append("</uploaded_files>")
    return "\n".join(lines)


def _old_whoami(identity, member_code_enabled):
    return (
        f"平台 ID：{identity.raw_id}\n"
        f"内部 ID：{identity.user_id}\n"
        f"身份：{'管理员' if identity.is_owner else '成员'}\n"
        f"CodeAgent 后端：{'本机 Docker' if identity.is_owner else ('云端沙箱（Modal）' if member_code_enabled else '未启用')}"
    )


def _cloud(value: str = "cloud"):
    return _patched(config, DEPLOY_MODE=value)


# ══════════════════════════════════════════════════════════════════════════════
# S1 门控
# ══════════════════════════════════════════════════════════════════════════════

def test_s1_gate():
    """S1 uses_host_inbox 真值表（只有 desktop 的 owner 为真）；gate 的各种组合；desktop 与改动前逐项一致。"""
    for mode, is_owner, want in (("desktop", True, True), ("desktop", False, False), ("cloud", True, False), ("cloud", False, False)):
        with _cloud(mode):
            assert file_staging.uses_host_inbox(is_owner) is want, (mode, is_owner)
    with _cloud("cloud"):
        assert config.is_cloud_mode() is True and file_staging.uses_host_inbox(True) is False
    assert config.is_cloud_mode() is False and file_staging.uses_host_inbox(True) is True, "离开 cloud 之后每次现读，立刻恢复"

    msg_disabled = "当前未开启文件接收功能。"
    msg_admin_only = "文件接收功能目前仅对管理员开放。"
    msg_no_code = "当前未启用云端代码环境，暂不支持处理文件附件（仅支持文本和图片）。"
    for mode in ("desktop", "cloud"):
        for identity in (OWNER, MEMBER):
            for file_on in (True, False):
                for member_on in (True, False):
                    for code_on in (True, False):
                        with _cloud(mode), _patched(config, QQ_FILE_ENABLED=file_on, QQ_FILE_MEMBER_ENABLED=member_on):
                            got = file_staging.gate(identity, member_code_enabled=code_on)
                            label = (mode, identity.user_id, file_on, member_on, code_on)
                            if not file_on:
                                want = (False, msg_disabled)
                            elif mode == "desktop" and identity.is_owner:
                                want = (True, "")                       # 宿主机 inbox：不看 QQ_FILE_MEMBER_ENABLED，也不要求云端 CodeAgent
                            elif not identity.is_owner and not member_on:
                                want = (False, msg_admin_only)
                            elif not code_on:
                                want = (False, msg_no_code)             # cloud 下 owner 也要求云端 CodeAgent 可用
                            else:
                                want = (True, "")
                            assert got == want, (label, got, want)
                            if mode == "desktop":                        # desktop：与改动前逐项一致
                                assert got == _old_gate(identity, member_code_enabled=code_on), label
    # cloud 下 owner 不看 QQ_FILE_MEMBER_ENABLED，但和 member 一样要 member_code_enabled
    with _cloud("cloud"), _patched(config, QQ_FILE_ENABLED=True, QQ_FILE_MEMBER_ENABLED=False):
        assert file_staging.gate(OWNER, member_code_enabled=True) == (True, "")
        assert file_staging.gate(OWNER, member_code_enabled=False) == (False, msg_no_code)
        assert file_staging.gate(MEMBER, member_code_enabled=True) == (False, msg_admin_only)
    # 签名没变
    assert list(inspect.signature(file_staging.gate).parameters) == ["identity", "member_code_enabled"]
    assert list(inspect.signature(file_staging.format_receipts).parameters) == ["outcome", "is_owner", "approval_pending"]
    assert list(inspect.signature(file_staging.format_uploaded_files_block).parameters) == ["notices", "is_owner"]
    assert list(inspect.signature(file_staging.stage_incoming).parameters) == ["files", "identity", "session_id"]
    assert list(inspect.signature(file_staging.collect_pending_notices).parameters) == ["session_id", "user_id", "upto_seq"]


# ══════════════════════════════════════════════════════════════════════════════
# S2 落地
# ══════════════════════════════════════════════════════════════════════════════

def test_s2_staging():
    """S2 cloud 下 owner 发文件 -> ('owner', sid) 沙箱 /work/uploads/ 里有它；事件 role=owner、path 以 /work/uploads/ 开头、gen 非空；
    宿主机没有 inbox；QQ_FILE_MAX_MB_OWNER 仍生效。desktop 下 owner 仍落在 inbox，FakeModal 里没有文件。"""
    async def cloud_owner(env, menv):
        sid = _sid()
        idle = max(1, (config.MODAL_IDLE_TIMEOUT_SECONDS + 30) // 60)
        reply = await _say("", files=[_incoming("报告.pdf", size=len(PDF))])
        want = (f"已收到：报告[.]pdf（{F.format_file_size(len(PDF))}），已放入你的云端工作区。"
                f"直接告诉我怎么处理（仅当前会话临时使用；闲置约 {idle} 分钟或发送 /reset 后会清空，需要时请重新发送）。")
        assert reply.text == want, reply.text
        assert reply.images == [] and not reply.awaiting_approval
        assert len(menv.fake.created) == 1
        fsb = menv.fake.created[0]
        assert fsb.tags["u"] == _sha("owner")[:12] and fsb.tags["s"] == _sha(sid)[:8], fsb.tags
        evs = env.events(sid)
        assert len(evs) == 1, evs
        payload = evs[0].payload
        assert payload["role"] == "owner" and payload["name"] == "报告.pdf" and payload["size"] == len(PDF), payload
        assert payload["path"].startswith("/work/uploads/报告_") and payload["path"].endswith(".pdf"), payload
        assert payload["gen"] and payload["gen"] == fsb.object_id, payload
        assert fsb.filesystem.get(payload["path"]) == PDF
        assert not env.inbox().exists() and os.listdir(env.ws) == [], "宿主机 WORKSPACE_DIR 里不出现 inbox"
        assert env.calls("file._stage_owner") == [] and env.touched_forbidden() == [], env.log
        assert "http" not in str(payload) and "files.example.com" not in str(payload), "事件里没有下载地址"

        # 文字消息：清单块用 member 版规则、路径是沙箱里的
        r = await _say("总结一下这个文件")
        assert r.text == "好的"
        block_msg = env.owner_agent.invocations[-1][-1].content
        assert block_msg.startswith("<uploaded_files>\n") and block_msg.endswith("\n\n总结一下这个文件"), block_msg[:80]
        assert f"→ {payload['path']}" in block_msg and "已放进用户的云端工作区" in block_msg
        for banned in ("FileAgent", "/workspace/inbox/", "convert_to_markdown", "workspace"):
            assert banned not in block_msg, banned
        assert file_staging._MEMBER_RULES in block_msg and file_staging._OWNER_RULES not in block_msg

    with _quiet(), _ModalEnv() as menv, _Env(cloud=True, QQ_FILE_MEMBER_ENABLED=False) as env, _Trips(env.log, strict=True):
        _run(cloud_owner(env, menv))          # QQ_FILE_MEMBER_ENABLED=False 也不影响 cloud 下的 owner

    async def cloud_limits(env, menv):
        big = b"x" * int(1.5 * MB)
        env.handler = lambda r: httpx.Response(200, content=big)
        sid = _sid()
        # owner 的上限 2 MB：1.5 MB 的文件收得进来；member 的上限 1 MB：同样的文件被拒绝
        ok = await _say("", files=[_incoming("big.bin")])
        assert ok.text.startswith("已收到：big[.]bin（"), ok.text
        member_reply = await _say("", identity=MEMBER, files=[_incoming("big.bin")])
        assert "超过 1 MB 的上限，没有接收" in member_reply.text, member_reply.text
        # owner 把上限调到 1 MB：声明的大小超限的文件在下载前就被拒绝；实际体积超限的在下载中途被拒绝，沙箱里都没有这个文件
        with _patched(config, QQ_FILE_MAX_MB_OWNER=1):
            before = [sorted(sb.filesystem.paths()) for sb in menv.fake.created]
            declared = await _say("", files=[_incoming("declared.bin", size=2 * MB)])
            assert "文件「declared[.]bin」超过 1 MB 的上限，没有接收。" in declared.text, declared.text
            actual = await _say("", files=[_incoming("actual.bin")])
            assert "文件「actual[.]bin」超过 1 MB 的上限，没有接收。" in actual.text, actual.text
            assert [sorted(sb.filesystem.paths()) for sb in menv.fake.created] == before
        assert len(env.events(sid)) == 1, "被拒绝的文件不写事件"
        assert env.touched_forbidden() == [], env.log

    with _quiet(), _ModalEnv() as menv, _Env(cloud=True, QQ_FILE_MAX_MB_OWNER=2, QQ_FILE_MAX_MB_MEMBER=1) as env, \
            _Trips(env.log, strict=True):
        _run(cloud_limits(env, menv))

    # cloud 下云端 CodeAgent 没启用：owner 的文件也不收，固定文案，没有发起下载
    async def cloud_disabled(env, menv):
        r = await _say("", files=[_incoming("d.csv")])
        assert r.text == "当前未启用云端代码环境，暂不支持处理文件附件（仅支持文本和图片）。", r.text
        assert env.requests == [] and menv.fake.created == [] and not env.inbox().exists()

    with _quiet(), _ModalEnv() as menv, _Env(cloud=True, member_code=False) as env, _Trips(env.log, strict=True):
        _run(cloud_disabled(env, menv))

    # desktop：owner 仍落在宿主机 inbox，FakeModal 里什么都没有；member 仍直传 Modal
    async def desktop(env, menv):
        sid = _sid()
        reply = await _say("", files=[_incoming("报告.pdf", size=len(PDF))])
        assert reply.text == f"已暂存：报告[.]pdf（{F.format_file_size(len(PDF))}）。直接告诉我怎么处理（仅当前会话临时使用，不会存入知识库）。", reply.text
        evs = env.events(sid)
        assert len(evs) == 1 and evs[0].payload["role"] == "owner" and evs[0].payload["gen"] == "", evs
        assert evs[0].payload["path"].startswith(f"/workspace/inbox/{sid}/报告_"), evs[0].payload
        assert [p.read_bytes() for p in env.inbox(sid).iterdir()] == [PDF]
        assert menv.fake.created == [] and len(env.calls("file._stage_owner")) == 1
        m = await _say("", identity=MEMBER, files=[_incoming("m.pdf", size=len(PDF))])
        assert "已放入你的云端工作区" in m.text and len(menv.fake.created) == 1
        assert menv.fake.created[0].tags["u"] == _sha(UID)[:12] and len(env.calls("file._stage_owner")) == 1

    with _quiet(), _ModalEnv() as menv, _Env(cloud=False, member_code=True) as env, _Trips(env.log, strict=False):
        _run(desktop(env, menv))


# ══════════════════════════════════════════════════════════════════════════════
# S3 回执与清单
# ══════════════════════════════════════════════════════════════════════════════

def test_s3_receipts_and_notices():
    """S3 cloud 下 owner 的回执 / 清单是 member 版；Windows 时代的 /workspace/inbox/... 事件判为失效；desktop 输出与改动前逐字一致。"""
    item = file_staging.StagedFile(display_name="报告.pdf", disk_name="报告_ab12cd34.pdf", size=36045, path="/work/uploads/报告_ab12cd34.pdf", gen="sb-1")
    item2 = file_staging.StagedFile(display_name="b.txt", disk_name="b_22222222.txt", size=10, path="/work/uploads/b_22222222.txt", gen="sb-1")
    outcomes = (
        file_staging.StageOutcome(staged=(item,), errors=()),
        file_staging.StageOutcome(staged=(item, item2), errors=("文件「c[.]bin」处理失败，请稍后重新发送。",)),
        file_staging.StageOutcome(staged=(), errors=("下载超时，请重新发送",)),
        file_staging.StageOutcome(staged=(), errors=()),
    )
    notices = [
        file_staging.FileNotice(name="a.txt", size=1234, path="/work/uploads/a_11111111.txt", ts=1700000000.0, available=True),
        file_staging.FileNotice(name="b.txt", size=10, path="/work/uploads/b_22222222.txt", ts=1700000100.0, available=False),
    ]
    only_stale = [notices[1]]

    for pending in (False, True):
        for outcome in outcomes:
            # cloud：owner 的回执与 member 的一模一样（member 版措辞）
            with _cloud("cloud"):
                owner_text = file_staging.format_receipts(outcome, is_owner=True, approval_pending=pending)
                assert owner_text == file_staging.format_receipts(outcome, is_owner=False, approval_pending=pending) == \
                    _old_format_receipts(outcome, is_owner=False, approval_pending=pending)
                if outcome.staged:
                    assert owner_text.startswith("已收到") and "已放入你的云端工作区" in owner_text and "已暂存" not in owner_text
            # desktop：owner / member 都与改动前逐字一致
            for is_owner in (True, False):
                assert file_staging.format_receipts(outcome, is_owner=is_owner, approval_pending=pending) == \
                    _old_format_receipts(outcome, is_owner=is_owner, approval_pending=pending), (is_owner, pending)

    for group in (notices, only_stale, []):
        with _cloud("cloud"):
            owner_block = file_staging.format_uploaded_files_block(group, is_owner=True)
            member_block = file_staging.format_uploaded_files_block(group, is_owner=False)
            assert owner_block == member_block == _old_format_block(group, is_owner=False)
            if group:
                assert "FileAgent" not in owner_block and "/workspace/inbox/" not in owner_block and "已被清理" not in owner_block
                assert file_staging._OWNER_RULES not in owner_block
        for is_owner in (True, False):
            assert file_staging.format_uploaded_files_block(group, is_owner=is_owner) == _old_format_block(group, is_owner=is_owner), (is_owner, len(group))
    with _cloud("cloud"):
        block = file_staging.format_uploaded_files_block(notices, is_owner=True)
        assert file_staging._MEMBER_RULES in block and "CodeAgent" in block and "云端工作区已被回收" in block
    assert file_staging._OWNER_RULES in file_staging.format_uploaded_files_block(notices, is_owner=True)

    async def collect_scenario(env, menv, cloud):
        sid = _sid()
        win_path = f"/workspace/inbox/{sid}/a_11111111.txt"
        host_file = env.inbox(sid) / "a_11111111.txt"
        host_file.parent.mkdir(parents=True, exist_ok=True)
        host_file.write_bytes(b"x")                          # Windows 时代的文件还躺在宿主机磁盘上
        env.store.emit_event(sid, session_store.EVENT_FILE_UPLOADED,
                             {"name": "a.txt", "path": win_path, "size": 1, "role": "owner", "gen": ""})
        ok, gen = modal_sandbox.upload_session_file("owner", sid, "/work/uploads/b_22222222.txt", b"hello") if cloud else (False, "")
        if cloud:
            assert ok and gen, (ok, gen)
            env.store.emit_event(sid, session_store.EVENT_FILE_UPLOADED,
                                 {"name": "b.txt", "path": "/work/uploads/b_22222222.txt", "size": 5, "role": "owner", "gen": gen})
        upto = env.store.emit_event(sid, "user_message", {"text": "q"})
        got = {n.path: n.available for n in await file_staging.collect_pending_notices(sid, "owner", upto_seq=upto)}
        if cloud:
            assert got == {win_path: False, "/work/uploads/b_22222222.txt": True}, got       # 宿主机文件还在，也判为失效
            menv.fake.advance(10 ** 5)                                                         # 沙箱被回收：新的文件也失效
            upto2 = env.store.emit_event(sid, "user_message", {"text": "q2"})
            env.store.emit_event(sid, session_store.EVENT_FILE_UPLOADED,
                                 {"name": "b.txt", "path": "/work/uploads/b_22222222.txt", "size": 5, "role": "owner", "gen": gen})
            upto3 = env.store.emit_event(sid, "user_message", {"text": "q3"})
            assert upto2 < upto3
            got3 = {n.path: n.available for n in await file_staging.collect_pending_notices(sid, "owner", upto_seq=upto3)}
            assert got3 == {"/work/uploads/b_22222222.txt": False}, got3
        else:
            assert got == {win_path: True}, got                                                # desktop：按宿主机 inbox 判断，文件在就有效

    with _quiet(), _ModalEnv() as menv, _Env(cloud=True) as env, _Trips(env.log, strict=True):
        _run(collect_scenario(env, menv, True))
        assert env.touched_forbidden() == [], env.log
    with _quiet(), _ModalEnv() as menv, _Env(cloud=False) as env, _Trips(env.log, strict=False):
        _run(collect_scenario(env, menv, False))
        assert menv.fake.created == [], "desktop 下 owner 的清单不会去查沙箱"


# ══════════════════════════════════════════════════════════════════════════════
# S4 chat_service 的 cloud 行为
# ══════════════════════════════════════════════════════════════════════════════

EXPERTS_CODE = ("KnowledgeAgent", "MediaAgent", "MapAgent", "CodeAgent")
EXPERTS_PLAIN = ("KnowledgeAgent", "MediaAgent", "MapAgent")


async def _init_service(env, *, owner_experts, member_experts, shared: bool):
    """照 smoke_chat_service 的做法：用假的 build_main_agents / open_checkpointer 跑真实的 init_service，返回打印出来的文本。"""
    real_open, real_validate, real_build = checkpointer.open_checkpointer, cs.models.validate_required_keys, cs.orchestrator.build_main_agents

    async def open_tmp(*args, **kwargs):
        return await real_open(os.path.join(env.tmp, "checkpoints.db"))

    async def fake_build(checkpointer):
        a = _FakeAgent("o")
        b = a if shared else _FakeAgent("m")
        return types.SimpleNamespace(owner=a, member=b, owner_experts=owner_experts, member_experts=member_experts)

    checkpointer.open_checkpointer, cs.models.validate_required_keys, cs.orchestrator.build_main_agents = open_tmp, lambda: None, fake_build
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            await cs.init_service()
    finally:
        checkpointer.open_checkpointer, cs.models.validate_required_keys, cs.orchestrator.build_main_agents = real_open, real_validate, real_build
    return out.getvalue()


def test_s4_cloud_service():
    """S4 cloud 下 owner 与 member 走同一条云端沙箱路径（假 agent 驱动真实的 process_message 与 wake）。"""

    async def reset_and_whoami(env):
        await _say("owner 先出现")
        sid = _sid()
        reply = await _say("/reset")
        assert reply.text == CLEAN_RESET_TEXT, reply.text
        assert env.calls("modal.reset_session") == [("owner", sid)], env.log
        assert env.touched_forbidden() == [], env.log                  # sandbox.reset_sandbox 与 clear_session_inbox 一次都没被调用
        # member 同样只终止自己当前会话的云端沙箱
        await _say("A 先出现", MEMBER)
        sid_m = _sid(UID)
        assert (await _say("/reset", MEMBER)).text.endswith(CLEAN_RESET_TEXT)
        assert env.calls("modal.reset_session") == [("owner", sid), (UID, sid_m)], env.log
        assert env.touched_forbidden() == []

        # /whoami：owner 与 member 都是“云端沙箱（Modal）”或“未启用”，含部署形态，不含“本机 Docker”
        owner_who = (await _say("/whoami")).text
        assert owner_who == f"平台 ID：{OWNER.raw_id}\n内部 ID：owner\n身份：管理员\nCodeAgent 后端：云端沙箱（Modal）\n部署形态：cloud", owner_who
        member_who = (await _say("/whoami", MEMBER)).text
        assert member_who == f"平台 ID：{UID}\n内部 ID：{UID}\n身份：成员\nCodeAgent 后端：云端沙箱（Modal）\n部署形态：cloud", member_who
        cs._member_code_enabled = False
        for who in ((await _say("/whoami")).text, (await _say("/whoami", MEMBER)).text):
            assert "CodeAgent 后端：未启用" in who and "部署形态：cloud" in who and "本机 Docker" not in who and "云端沙箱" not in who, who
        cs._member_code_enabled = True
        # 管理类命令仍然只有 owner
        assert (await _say("/users", MEMBER)).text == "该命令仅管理员可用"
        assert "用户列表" in (await _say("/users")).text

    with _quiet(), _Env(cloud=True) as env, _Trips(env.log, strict=True), _ModalSpy(env.log):
        _run(reset_and_whoami(env))

    async def stop_and_failures(env):
        env.owner_agent.delay = 2.0
        # /stop：只中断自己会话的云端命令，不碰宿主机命令与 Docker 沙箱的进程级终止原语
        turn = _start("owner 的慢问题")
        await asyncio.sleep(0.4)
        sid = cs._current_turns["owner"].session_id
        stop = await _say("/stop")
        assert "已中止" in stop.text and "宿主机命令" not in stop.text, stop.text
        assert await turn is None
        assert env.calls("modal.interrupt_session_nowait") == [("owner", sid)], env.log
        assert env.touched_forbidden() == [], env.log
        # 空闲时 /stop 不触发任何东西
        assert "没有正在运行" in (await _say("/stop")).text
        assert len(env.calls("modal.interrupt_session_nowait")) == 1 and env.touched_forbidden() == []

        # 失败回合：超时 -> 中断一次；异常 -> 再中断一次；因 /stop 取消的回合不重复调用
        env.log.clear()
        with _patched(config, AGENT_TURN_TIMEOUT_SECONDS=0.3):
            timeout_reply = await _say("会超时的问题")
        assert "还没完成" in timeout_reply.text, timeout_reply.text
        sid = _sid()
        assert env.calls("modal.interrupt_session_nowait") == [("owner", sid)], env.log
        env.owner_agent.delay, env.owner_agent.raises = 0.0, RuntimeError("boom")
        err_reply = await _say("会出错的问题")
        assert err_reply.text == "处理时出了点问题，请稍后再试", err_reply.text
        assert env.calls("modal.interrupt_session_nowait") == [("owner", sid), ("owner", sid)], env.log
        env.owner_agent.delay, env.owner_agent.raises = 2.0, None
        env.log.clear()
        turn = _start("又一个慢问题")
        await asyncio.sleep(0.4)
        await _say("/stop")
        assert await turn is None
        assert len(env.calls("modal.interrupt_session_nowait")) == 1, "/stop 引起的失败只调用一次（不重复）"
        assert env.touched_forbidden() == [], env.log
        # 正常结束的回合不触发任何钩子
        env.owner_agent.delay, env.owner_agent.raises = 0.0, None
        env.log.clear()
        assert (await _say("正常问题")).text == "好的"
        assert env.log == [], env.log

    with _quiet(), _Env(cloud=True) as env, _Trips(env.log, strict=True), _ModalSpy(env.log):
        _run(stop_and_failures(env))

    async def semaphore(env):
        calls = []
        real = cs._member_semaphore

        def spy():
            calls.append(1)
            return real()

        with _patched(cs, _member_semaphore=spy):
            assert (await _say("owner 的问题")).text == "好的"
            assert calls == [], "owner 不占 member 的并发信号量"
            assert (await _say("member 的问题", MEMBER)).text.endswith("好的")
            assert calls == [1], "member 仍然占用"
            # 无人值守的任务同样：owner 不占
            assert cs._member_semaphore is spy

    with _quiet(), _Env(cloud=True) as env, _Trips(env.log, strict=True), _ModalSpy(env.log):
        _run(semaphore(env))

    async def block(env):
        # /block 回收的是 member 的云端沙箱（owner 不可被封禁）；owner 在 cloud 下的行为不变
        await _say("owner 先出现")                        # 用户列表按首次出现的先后排：owner 是 1，member 是 2
        await _say("A 先出现", MEMBER_B)
        reply = await _say("/block 2")
        assert reply.text.startswith("已封禁"), reply.text
        assert len(env.calls("modal.reset_user")) == 1 and env.touched_forbidden() == [], env.log
        assert (await _say("/block owner")).text == "管理员不可被封禁。"

    with _quiet(), _Env(cloud=True) as env, _Trips(env.log, strict=True), _ModalSpy(env.log):
        _run(block(env))

    # init_service：cloud 下不清理宿主机 inbox，只打印一行专家清单，有部署形态行与 CodeAgent 横幅
    async def init_cloud(env, with_code):
        experts = EXPERTS_CODE if with_code else EXPERTS_PLAIN
        banner = await _init_service(env, owner_experts=experts, member_experts=experts, shared=True)
        try:
            assert env.calls("file.cleanup_expired_inbox") == [] and env.touched_forbidden() == [], env.log
            lines = banner.splitlines()
            assert "🌐 [chat_service] 部署形态: cloud" in lines, banner
            expert_lines = [ln for ln in lines if ln.startswith("🧩")]
            assert expert_lines == [f"🧩 [chat_service] cloud 主图专家（owner 与 member 共用）: {', '.join(experts)}"], expert_lines
            assert "owner 图专家" not in banner and "member 图专家" not in banner
            assert cs._agents["owner"] is cs._agents["member"], "cloud 下 owner 与 member 是同一张图"
            assert cs._member_code_enabled is with_code
            if with_code:
                assert "☁️ [chat_service] 云端 CodeAgent（owner 与 member 共用）：已启用（Modal）" in lines, banner
            else:
                assert "☁️ [chat_service] 云端 CodeAgent（owner 与 member 共用）：未启用（MODAL_DISABLED=true）" in lines, banner
            assert "member 云端 CodeAgent" not in banner and "🧹" not in banner
        finally:
            await checkpointer.close_checkpointer(cs._checkpointer)
            cs._checkpointer = None

    for with_code in (True, False):
        with _quiet(), _Env(cloud=True) as env, _Trips(env.log, strict=True), \
                _ModalSpy(env.log, status=(False, "MODAL_DISABLED=true")):
            cs._reset_runtime_state()
            _run(init_cloud(env, with_code))

    # shutdown_service：cloud 下不关浏览器、不销毁本机 Docker 沙箱；shutdown_pool 与 shutdown_mcp 各一次，pool 在 mcp 之前
    async def shutdown_cloud(env):
        await cs.shutdown_service()
        order = [n for n, _ in env.log]
        assert order == ["modal.shutdown_pool", "mcp.shutdown_mcp", "milvus.close_client", "amap.aclose"], order
        assert env.touched_forbidden() == [], env.log
        assert cs._agents == {} and not cs.is_initialized() and cs._checkpointer is None

    with _quiet(), _Env(cloud=True) as env, _Trips(env.log, strict=True), _ModalSpy(env.log):
        _run(shutdown_cloud(env))


# ══════════════════════════════════════════════════════════════════════════════
# S5 desktop 回归
# ══════════════════════════════════════════════════════════════════════════════

def test_s5_desktop_regression():
    """S5 desktop 下：owner 走 Docker 与 inbox，member 走 Modal；回复文案、启动与关闭日志与改动前一致。"""

    async def commands(env):
        await _say("owner 先出现")
        sid = _sid()
        # 先放一个文件进 owner 的 inbox，/reset 之后目录被删
        await _say("", files=[_incoming("a.pdf", size=len(PDF))])
        assert env.inbox(sid).exists()
        env.log.clear()
        reply = await _say("/reset")
        assert reply.text == CLEAN_RESET_TEXT
        assert [n for n, _ in env.log] == ["docker.reset_sandbox", "file.clear_session_inbox"], env.log
        assert env.calls("file.clear_session_inbox") == [(sid,)] and not env.inbox(sid).exists()
        assert env.calls("modal.reset_session") == [], "desktop 下 owner 的 /reset 不得调用任何 Modal 钩子"
        # member 的 /reset：只终止自己当前会话的云端沙箱
        await _say("A 先出现", MEMBER)
        sid_m = _sid(UID)
        env.log.clear()
        assert (await _say("/reset", MEMBER)).text.endswith(CLEAN_RESET_TEXT)
        assert env.log == [("modal.reset_session", (UID, sid_m))], env.log

        # /whoami：与改动前逐字一致，没有“部署形态”
        for identity, code in ((OWNER, True), (OWNER, False), (MEMBER, True), (MEMBER, False)):
            cs._member_code_enabled = code
            got = (await _say("/whoami", identity)).text
            assert got == _old_whoami(identity, code), (identity.user_id, code, got)
            assert "部署形态" not in got
        cs._member_code_enabled = True

        # /stop：owner 终止宿主机命令与 Docker 沙箱里的进程，不碰 Modal；member 只中断自己会话的云端命令
        env.owner_agent.delay = env.member_agent.delay = 2.0
        env.log.clear()
        turn = _start("owner 的慢问题")
        await asyncio.sleep(0.4)
        stop = await _say("/stop")
        assert "已中止" in stop.text and "已终止 2 个正在运行的宿主机命令。" in stop.text, stop.text
        assert await turn is None
        assert [n for n, _ in env.log] == ["host.kill_all_running", "docker.interrupt_sandbox"], env.log
        env.log.clear()
        turn = _start("A 的慢问题", MEMBER)
        await asyncio.sleep(0.4)
        sid_m = cs._current_turns[UID].session_id
        stop = await _say("/stop", MEMBER)
        assert "已中止" in stop.text and "宿主机命令" not in stop.text
        assert await turn is None
        assert env.log == [("modal.interrupt_session_nowait", (UID, sid_m))], env.log

        # 失败回合：owner 不调用任何 Modal 钩子；member 调用一次
        env.owner_agent.delay = env.member_agent.delay = 1.5
        env.log.clear()
        with _patched(config, AGENT_TURN_TIMEOUT_SECONDS=0.3):
            assert "还没完成" in (await _say("owner 会超时")).text
            assert env.log == [], "desktop 下 owner 的失败回合不得触发任何 Modal 钩子"
            assert "还没完成" in (await _say("A 会超时", MEMBER)).text
        assert [n for n, _ in env.log] == ["modal.interrupt_session_nowait"], env.log

    with _quiet(), _Env(cloud=False) as env, _Trips(env.log, strict=False), _ModalSpy(env.log):
        _run(commands(env))

    # owner 不占 member 信号量、member 占用（desktop 同样）
    async def semaphore(env):
        calls = []
        real = cs._member_semaphore

        def spy():
            calls.append(1)
            return real()

        with _patched(cs, _member_semaphore=spy):
            await _say("owner 的问题")
            assert calls == []
            await _say("member 的问题", MEMBER)
            assert calls == [1]

    with _quiet(), _Env(cloud=False) as env, _Trips(env.log, strict=False), _ModalSpy(env.log):
        _run(semaphore(env))

    # init_service：desktop 下清理 inbox 一次，两行专家清单，member 云端 CodeAgent 横幅，没有部署形态行
    async def init_desktop(env, with_code):
        member_experts = EXPERTS_CODE if with_code else EXPERTS_PLAIN
        owner_experts = ("KnowledgeAgent", "MediaAgent", "MapAgent", "CodeAgent", "BrowserAgent", "FileAgent", "DesktopAgent")
        banner = await _init_service(env, owner_experts=owner_experts, member_experts=member_experts, shared=False)
        try:
            assert len(env.calls("file.cleanup_expired_inbox")) == 1, env.log
            lines = banner.splitlines()
            assert f"🧩 [chat_service] owner 图专家: {', '.join(owner_experts)}" in lines, banner
            assert f"🧩 [chat_service] member 图专家: {', '.join(member_experts)}" in lines, banner
            assert len([ln for ln in lines if ln.startswith("🧩")]) == 2
            assert "部署形态" not in banner and "cloud" not in banner and "共用" not in banner, banner
            if with_code:
                assert "☁️ [chat_service] member 云端 CodeAgent：已启用（Modal）" in lines, banner
            else:
                assert "☁️ [chat_service] member 云端 CodeAgent：未启用（MODAL_DISABLED=true）" in lines, banner
            assert cs._agents["owner"] is not cs._agents["member"]
        finally:
            await checkpointer.close_checkpointer(cs._checkpointer)
            cs._checkpointer = None

    for with_code in (True, False):
        with _quiet(), _Env(cloud=False) as env, _Trips(env.log, strict=False), \
                _ModalSpy(env.log, status=(False, "MODAL_DISABLED=true")):
            cs._reset_runtime_state()
            _run(init_desktop(env, with_code))
    # QQ_FILE_ENABLED 关闭时 desktop 也不清理
    with _quiet(), _Env(cloud=False, QQ_FILE_ENABLED=False) as env, _Trips(env.log, strict=False), _ModalSpy(env.log):
        cs._reset_runtime_state()
        _run(_init_service(env, owner_experts=EXPERTS_PLAIN, member_experts=EXPERTS_PLAIN, shared=False))
        assert env.calls("file.cleanup_expired_inbox") == []
        _run(checkpointer.close_checkpointer(cs._checkpointer))
        cs._checkpointer = None

    # shutdown_service：desktop 下的关闭顺序与改动前一致：pool -> mcp -> 浏览器 -> milvus -> amap -> 本机 Docker 沙箱
    async def shutdown_desktop(env):
        await cs.shutdown_service()
        order = [n for n, _ in env.log]
        assert order == ["modal.shutdown_pool", "mcp.shutdown_mcp", "browser.close_browser", "milvus.close_client",
                         "amap.aclose", "docker.shutdown_sandbox"], order
        assert cs._agents == {} and not cs.is_initialized()

    with _quiet(), _Env(cloud=False) as env, _Trips(env.log, strict=False), _ModalSpy(env.log):
        _run(shutdown_desktop(env))


# ══════════════════════════════════════════════════════════════════════════════
# S6 静态检查
# ══════════════════════════════════════════════════════════════════════════════

_LOCAL_MODULES = {"sandbox", "host_shell", "browser_service"}
_LOCAL_CALLS = {"reset_sandbox", "shutdown_sandbox", "interrupt_sandbox", "kill_all_running", "close_browser",
                "clear_session_inbox", "cleanup_expired_inbox", "_kill_side_effects"}


def _parents_of(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _mentions_cloud(expr) -> bool:
    for n in ast.walk(expr):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "is_cloud_mode":
            return True
        if isinstance(n, ast.Name) and n.id == "cloud":
            return True
    return False


def _implies_not_cloud(test) -> bool:
    """test 为真时一定不是 cloud：`not cloud` / `not config.is_cloud_mode()`，或者它们的 and 组合。"""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _mentions_cloud(test.operand)
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return any(_implies_not_cloud(v) for v in test.values)
    return False


def _implies_cloud(test) -> bool:
    """test 为假时一定不是 cloud 的反面：只认最简单的 `cloud` / `config.is_cloud_mode()`（复合条件的 else 分支不算）。"""
    if isinstance(test, ast.Name):
        return test.id == "cloud"
    return isinstance(test, ast.Call) and getattr(test.func, "attr", getattr(test.func, "id", "")) == "is_cloud_mode"


def _guarded_not_cloud(node, parents) -> bool:
    """node 是否只会在非 cloud 时执行：祖先里有 if 的 body 满足 _implies_not_cloud，或 if 的 else 满足 _implies_cloud。"""
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If):
            if child in parent.body and _implies_not_cloud(parent.test):
                return True
            if child in parent.orelse and _implies_cloud(parent.test):
                return True
        child = parent
    return False


def _enclosing_function(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


class _ImportTimeCloudRefs(ast.NodeVisitor):
    """只看 import 时会执行的代码（函数体不算）里有没有读取部署形态。"""

    def __init__(self):
        self.hits = []

    def visit_FunctionDef(self, node):
        pass

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_Attribute(self, node):
        if node.attr in ("is_cloud_mode", "DEPLOY_MODE"):
            self.hits.append(node.lineno)
        self.generic_visit(node)


def _imports_local(node) -> list:
    """这条 import 语句引入了哪些本机资源模块（sandbox / host_shell / browser_service）。"""
    hits = []
    if isinstance(node, ast.ImportFrom) and node.module == "app.infra":
        hits += [a.name for a in node.names if a.name in _LOCAL_MODULES]
    if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] in _LOCAL_MODULES and node.module.startswith("app.infra"):
        hits.append(node.module)
    if isinstance(node, ast.Import):
        hits += [a.name for a in node.names if a.name.startswith("app.infra.") and a.name.split(".")[-1] in _LOCAL_MODULES]
    return hits


def test_s6_static():
    """S6 用 ast 确认：没有模块级的 is_cloud_mode() 调用、没有把它的结果存进模块变量；cloud 分支不 import 本机沙箱 / 宿主机命令 / 浏览器。"""
    trees = {}
    for name in ("chat_service", "file_staging"):
        path = ROOT / "app" / "services" / f"{name}.py"
        trees[name] = (path, ast.parse(path.read_text(encoding="utf-8"), str(path)))

    for name, (path, tree) in trees.items():
        # ① import 时不读部署形态
        visitor = _ImportTimeCloudRefs()
        visitor.visit(tree)
        assert not visitor.hits, f"{name}.py:{visitor.hits} 在 import 时读取了部署形态（会把它缓存住）"
        # ② 没有把 is_cloud_mode() 的结果存进任何模块变量：global 声明过的名字不得被含 is_cloud_mode 的值赋值
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            declared = {nm for n in ast.walk(fn) if isinstance(n, ast.Global) for nm in n.names}
            for n in ast.walk(fn):
                if not isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) or n.value is None:
                    continue
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if _mentions_cloud(n.value) and any(isinstance(t, ast.Name) and t.id in declared for t in targets):
                    raise AssertionError(f"{name}.py:{n.lineno} 把 is_cloud_mode() 的结果存进了模块变量")
        # ③ 模块级没有任何本机资源模块的 import
        parents = _parents_of(tree)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)) and _imports_local(n):
                assert parents[n] is not tree, f"{name}.py:{n.lineno} 模块级 import 了本机资源模块 {_imports_local(n)}"

    # chat_service：每一处 import 本机资源模块、每一处对它们的调用，都只会在非 cloud 时执行
    path, tree = trees["chat_service"]
    parents = _parents_of(tree)
    seen_imports, seen_calls = [], []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)) and _imports_local(n):
            seen_imports.append(n.lineno)
            fn = _enclosing_function(n, parents)
            assert fn == "_kill_side_effects" or _guarded_not_cloud(n, parents), f"chat_service.py:{n.lineno} cloud 分支里 import 了 {_imports_local(n)}"
        if isinstance(n, ast.Call):
            callee = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            # asyncio.to_thread(sandbox.reset_sandbox) 这类把函数当参数传的写法也算调用点
            refs = [callee] + [getattr(a, "attr", getattr(a, "id", "")) for a in n.args]
            for ref in refs:
                if ref in _LOCAL_CALLS or (ref == "close_browser"):
                    seen_calls.append((ref, n.lineno))
                    fn = _enclosing_function(n, parents)
                    assert fn == "_kill_side_effects" or _guarded_not_cloud(n, parents), f"chat_service.py:{n.lineno} cloud 分支里调用了 {ref}"
    assert len(seen_imports) >= 3, f"静态检查没有找到应有的 import（检查本身是空转的）: {seen_imports}"
    assert {c for c, _ in seen_calls} >= {"reset_sandbox", "clear_session_inbox", "cleanup_expired_inbox", "shutdown_sandbox", "close_browser", "_kill_side_effects"}, seen_calls

    # file_staging：不 import 本机资源模块；调用 _stage_owner 只在 uses_host_inbox(...) 为真的分支里
    path, tree = trees["file_staging"]
    parents = _parents_of(tree)
    assert not [n.lineno for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) and _imports_local(n)]
    owner_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_stage_owner"]
    assert len(owner_calls) == 1
    node, guarded = owner_calls[0], False
    while node in parents:
        child, node = node, parents[node]
        if isinstance(node, ast.If) and child in node.body and isinstance(node.test, ast.Call) and getattr(node.test.func, "id", "") == "uses_host_inbox":
            guarded = True
    assert guarded, "_stage_owner 只能在 uses_host_inbox(...) 为真时调用"
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "uses_host_inbox"]
    assert len(uses) == 1 and "is_cloud_mode" in ast.dump(uses[0]), "uses_host_inbox 里现读 is_cloud_mode()"

    # 现有入口的签名没动：只允许新增函数
    assert list(inspect.signature(cs.process_message).parameters) == ["text", "image_url", "identity", "files"]


SCENARIOS = [
    ("S1 门控", test_s1_gate),
    ("S2 落地", test_s2_staging),
    ("S3 回执与清单", test_s3_receipts_and_notices),
    ("S4 chat_service 的 cloud 行为", test_s4_cloud_service),
    ("S5 desktop 回归", test_s5_desktop_regression),
    ("S6 静态检查", test_s6_static),
]


def main():
    print("=" * 70)
    print("开始运行服务层 DEPLOY_MODE=cloud 离线冒烟测试...")
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
