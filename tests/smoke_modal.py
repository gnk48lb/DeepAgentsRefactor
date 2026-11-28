"""
tests/smoke_modal.py

多用户改造（阶段 2）离线冒烟测试：Modal 云端沙箱与 member 的 CodeAgent。
全部离线、确定：不联网、不调用真实 Modal、不读 .env 的内容（用 tests/_fake_modal.py 的 FakeModal 注入）。
  B 配置夹紧与 member_code_status 门控、不联网、惰性导入（modal 不在 sys.modules）
"""
import ast
import asyncio
import builtins
import concurrent.futures
import contextlib
import contextvars
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import logging
import logging.handlers
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 本机 .env 里可能是 DEPLOY_MODE=cloud：load_dotenv 不覆盖已有的环境变量，所以这一行能压过它（本测试针对 desktop 形态）。
os.environ["DEPLOY_MODE"] = "desktop"

from PIL import Image

from app import config
from app.infra import modal_sandbox
from app.infra.sandbox import OutputsResult, RunResult, SandboxUnavailable
from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from _fake_modal import Block, Call, FakeModal, Fail, Hang, Ok, Out, Raise, Sleep
from _support import member_context, owner_context

ROOT = Path(__file__).resolve().parents[1]

# config 里 MODAL_* 对应的环境变量名（测试里逐个显式设值，防止用户机器上 .env 的真实配置混进来）
_MODAL_ENV = (
    "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_DISABLED", "MODAL_APP_NAME", "MODAL_IMAGE_NAME",
    "MODAL_CPU_REQUEST", "MODAL_CPU_LIMIT", "MODAL_MEMORY_MB_REQUEST", "MODAL_MEMORY_MB_LIMIT",
    "MODAL_IDLE_TIMEOUT_SECONDS", "MODAL_MAX_LIFETIME_SECONDS", "MODAL_EXEC_TIMEOUT_SECONDS",
    "MODAL_EXEC_MAX_TIMEOUT_SECONDS", "MODAL_STARTUP_TIMEOUT_SECONDS", "MODAL_MAX_SANDBOXES_PER_USER",
    "MODAL_MAX_SANDBOXES_TOTAL", "MODAL_BLOCK_NETWORK", "MODAL_AUDIT_LOG", "MODAL_INTERRUPT_MODE",
)
_FAKE_ID = "ak-FAKEFAKEFAKEFAKE1234"
_FAKE_SECRET = "as-FAKEFAKEFAKEFAKE5678"


@contextlib.contextmanager
def _patched(obj, **attrs):
    """临时替换对象属性，退出时还原（测试直接改 config.MODAL_* 来改变行为）。"""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


# ══════════════════════════════════════════════════════════════════════════════
# B 配置与门控
# ══════════════════════════════════════════════════════════════════════════════

_PROBE_CODE = """
import json
from app import config
names = [n for n in dir(config) if n.startswith("MODAL_")]
print(json.dumps({n: getattr(config, n) for n in names}, ensure_ascii=False))
"""


def _config_probe(env_extra: dict) -> dict:
    """子进程里导入 config 并导出全部 MODAL_*：先把所有相关环境变量清成空串（load_dotenv 不覆盖已存在的变量，.env 不会漏进来）。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for name in _MODAL_ENV:
        env[name] = ""
    env.update(env_extra)
    res = subprocess.run(
        [sys.executable, "-c", _PROBE_CODE], env=env, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8"
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_b1_config_defaults():
    """B1 全部留空时是文档规定的默认值。"""
    c = _config_probe({})
    audit = c.pop("MODAL_AUDIT_LOG")
    assert audit.replace("\\", "/").endswith("/storage/modal_audit.log"), audit
    assert c == {
        "MODAL_DISABLED": False, "MODAL_TOKENS_PRESENT": False,
        "MODAL_APP_NAME": "gnk48-code-sandbox", "MODAL_IMAGE_NAME": "gnk48-python-ds:v1",
        "MODAL_CPU_REQUEST": 0.25, "MODAL_CPU_LIMIT": 2.0,
        "MODAL_MEMORY_MB_REQUEST": 512, "MODAL_MEMORY_MB_LIMIT": 2048,
        "MODAL_IDLE_TIMEOUT_SECONDS": 600, "MODAL_MAX_LIFETIME_SECONDS": 3600,
        "MODAL_EXEC_TIMEOUT_SECONDS": 120, "MODAL_EXEC_MAX_TIMEOUT_SECONDS": 300,
        "MODAL_STARTUP_TIMEOUT_SECONDS": 120,
        "MODAL_MAX_SANDBOXES_PER_USER": 2, "MODAL_MAX_SANDBOXES_TOTAL": 8,
        "MODAL_BLOCK_NETWORK": False, "MODAL_INTERRUPT_MODE": "soft",
    }, c


def test_b2_config_clamping():
    """B2 非法值回默认、request 夹到 limit、各项下限与相互约束、字符串 / 布尔解析。"""
    c = _config_probe({"MODAL_CPU_REQUEST": "4", "MODAL_CPU_LIMIT": "1.5"})
    assert c["MODAL_CPU_LIMIT"] == 1.5 and c["MODAL_CPU_REQUEST"] == 1.5, c           # request 大于 limit → 夹到 limit
    for bad in ("abc", "nan", "inf", "-1", "0"):
        c = _config_probe({"MODAL_CPU_REQUEST": bad, "MODAL_CPU_LIMIT": bad})
        assert (c["MODAL_CPU_REQUEST"], c["MODAL_CPU_LIMIT"]) == (0.25, 2.0), (bad, c)

    c = _config_probe({"MODAL_MEMORY_MB_LIMIT": "100", "MODAL_MEMORY_MB_REQUEST": "9999"})
    assert c["MODAL_MEMORY_MB_LIMIT"] == 256 and c["MODAL_MEMORY_MB_REQUEST"] == 256, c   # limit 至少 256，request 再夹到它
    c = _config_probe({"MODAL_MEMORY_MB_LIMIT": "x", "MODAL_MEMORY_MB_REQUEST": "-5"})
    assert (c["MODAL_MEMORY_MB_REQUEST"], c["MODAL_MEMORY_MB_LIMIT"]) == (512, 2048), c

    assert _config_probe({"MODAL_IDLE_TIMEOUT_SECONDS": "5"})["MODAL_IDLE_TIMEOUT_SECONDS"] == 60
    assert _config_probe({"MODAL_IDLE_TIMEOUT_SECONDS": "abc"})["MODAL_IDLE_TIMEOUT_SECONDS"] == 600

    c = _config_probe({"MODAL_EXEC_MAX_TIMEOUT_SECONDS": "400", "MODAL_MAX_LIFETIME_SECONDS": "100", "MODAL_EXEC_TIMEOUT_SECONDS": "9999"})
    assert c["MODAL_MAX_LIFETIME_SECONDS"] == 460, c                                # 至少 exec 上限 + 60
    assert c["MODAL_EXEC_TIMEOUT_SECONDS"] == 400, c                                # 默认超时不超过上限
    assert _config_probe({"MODAL_EXEC_TIMEOUT_SECONDS": "0"})["MODAL_EXEC_TIMEOUT_SECONDS"] == 1
    assert _config_probe({"MODAL_STARTUP_TIMEOUT_SECONDS": "0"})["MODAL_STARTUP_TIMEOUT_SECONDS"] == 1
    assert _config_probe({"MODAL_STARTUP_TIMEOUT_SECONDS": "abc"})["MODAL_STARTUP_TIMEOUT_SECONDS"] == 120

    c = _config_probe({"MODAL_MAX_SANDBOXES_PER_USER": "0", "MODAL_MAX_SANDBOXES_TOTAL": "-3"})
    assert (c["MODAL_MAX_SANDBOXES_PER_USER"], c["MODAL_MAX_SANDBOXES_TOTAL"]) == (1, 1), c
    c = _config_probe({"MODAL_MAX_SANDBOXES_PER_USER": "3", "MODAL_MAX_SANDBOXES_TOTAL": "1"})
    assert (c["MODAL_MAX_SANDBOXES_PER_USER"], c["MODAL_MAX_SANDBOXES_TOTAL"]) == (3, 3), c   # total 至少等于每用户上限

    assert _config_probe({"MODAL_INTERRUPT_MODE": " TERMINATE "})["MODAL_INTERRUPT_MODE"] == "terminate"
    assert _config_probe({"MODAL_INTERRUPT_MODE": "Soft"})["MODAL_INTERRUPT_MODE"] == "soft"
    assert _config_probe({"MODAL_INTERRUPT_MODE": "bogus"})["MODAL_INTERRUPT_MODE"] == "soft"

    c = _config_probe({"MODAL_APP_NAME": "   ", "MODAL_IMAGE_NAME": "   "})
    assert (c["MODAL_APP_NAME"], c["MODAL_IMAGE_NAME"]) == ("gnk48-code-sandbox", "gnk48-python-ds:v1"), c
    c = _config_probe({"MODAL_APP_NAME": " my-app ", "MODAL_IMAGE_NAME": " x:v2 "})
    assert (c["MODAL_APP_NAME"], c["MODAL_IMAGE_NAME"]) == ("my-app", "x:v2"), c

    c = _config_probe({"MODAL_DISABLED": "TRUE", "MODAL_BLOCK_NETWORK": "yes"})
    assert c["MODAL_DISABLED"] is True and c["MODAL_BLOCK_NETWORK"] is True, c
    c = _config_probe({"MODAL_DISABLED": "no", "MODAL_BLOCK_NETWORK": "0"})
    assert c["MODAL_DISABLED"] is False and c["MODAL_BLOCK_NETWORK"] is False, c


def test_b3_tokens_are_only_a_boolean():
    """B3 MODAL_TOKENS_PRESENT 只在两个令牌都非空时为真；config 里没有任何值等于或包含令牌。"""
    assert _config_probe({"MODAL_TOKEN_ID": _FAKE_ID})["MODAL_TOKENS_PRESENT"] is False
    assert _config_probe({"MODAL_TOKEN_SECRET": _FAKE_SECRET})["MODAL_TOKENS_PRESENT"] is False
    c = _config_probe({"MODAL_TOKEN_ID": _FAKE_ID, "MODAL_TOKEN_SECRET": _FAKE_SECRET})
    assert c["MODAL_TOKENS_PRESENT"] is True
    leaked = [k for k, v in c.items() if isinstance(v, str) and (_FAKE_ID in v or _FAKE_SECRET in v)]
    assert not leaked, leaked

    # 在这个进程里再查一次 vars(config)：整个模块里的任何值都不是（或含有）令牌
    old = {n: os.environ.get(n) for n in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")}
    code = (
        "import os\n"
        "from app import config\n"
        f"ids = ({_FAKE_ID!r}, {_FAKE_SECRET!r})\n"
        "bad = [k for k, v in vars(config).items() if isinstance(v, str) and any(t in v for t in ids)]\n"
        "assert not bad, bad\n"
    )
    env = dict(os.environ)
    env.update({"MODAL_TOKEN_ID": _FAKE_ID, "MODAL_TOKEN_SECRET": _FAKE_SECRET})
    res = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr
    assert {n: os.environ.get(n) for n in old} == old


def test_b4_member_code_status_branches():
    """B4 member_code_status 的 5 个分支各自的返回值与原因；失败即关闭；原因里没有令牌值。"""
    real_spec, real_version = importlib.util.find_spec, importlib.metadata.version
    fake_spec = object()
    saved_env = {n: os.environ.get(n) for n in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")}
    os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"] = _FAKE_ID, _FAKE_SECRET
    try:
        importlib.util.find_spec = lambda name, *a, **k: fake_spec if name == "modal" else real_spec(name, *a, **k)
        importlib.metadata.version = lambda name: "1.6.4" if name == "modal" else real_version(name)
        ok_cfg = dict(MODAL_DISABLED=False, MODAL_TOKENS_PRESENT=True)

        with _patched(config, **ok_cfg):
            assert modal_sandbox.member_code_status() == (True, "")

        with _patched(config, MODAL_DISABLED=True, MODAL_TOKENS_PRESENT=True):       # 1 总开关优先于一切
            assert modal_sandbox.member_code_status() == (False, "MODAL_DISABLED=true")
        with _patched(config, MODAL_DISABLED=True, MODAL_TOKENS_PRESENT=False):
            assert modal_sandbox.member_code_status() == (False, "MODAL_DISABLED=true")

        with _patched(config, MODAL_DISABLED=False, MODAL_TOKENS_PRESENT=False):     # 2 没有令牌
            assert modal_sandbox.member_code_status() == (False, "未配置 MODAL_TOKEN_ID / MODAL_TOKEN_SECRET")

        with _patched(config, **ok_cfg):
            importlib.util.find_spec = lambda name, *a, **k: None if name == "modal" else real_spec(name, *a, **k)
            assert modal_sandbox.member_code_status() == (False, "未安装 modal 包")   # 3 没装
            importlib.util.find_spec = lambda name, *a, **k: (_ for _ in ()).throw(ValueError("modal.__spec__ is None"))
            assert modal_sandbox.member_code_status() == (False, "未安装 modal 包")   # find_spec 自己抛错也按不可用处理
            importlib.util.find_spec = lambda name, *a, **k: fake_spec if name == "modal" else real_spec(name, *a, **k)

            # 4 版本：按数字比较前两段，不是字符串比较
            for ver, expect in (("1.5.9", False), ("0.99", False), ("1.6.0", True), ("1.6", True), ("1.10.2", True),
                                ("2.0", True), ("1.6.4.dev1", True), ("abc", False), ("", False), ("v1.7", False)):
                importlib.metadata.version = lambda name, _v=ver: _v if name == "modal" else real_version(name)
                ok, reason = modal_sandbox.member_code_status()
                assert ok is expect, (ver, ok, reason)
                assert (reason == "") == expect, (ver, reason)
                if not expect:
                    assert "modal" in reason and "1.6" in reason, reason
            importlib.metadata.version = lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name))
            ok, reason = modal_sandbox.member_code_status()
            assert ok is False and reason, (ok, reason)                                 # 查不到版本：失败即关闭
    finally:
        importlib.util.find_spec, importlib.metadata.version = real_spec, real_version
        for n, v in saved_env.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v


def test_b5_member_code_status_is_offline_and_leak_free():
    """B5 判定过程不联网（连 socket.connect 都不会被调用）；每个分支的原因里没有令牌值。"""
    real_connect = socket.socket.connect
    real_spec, real_version = importlib.util.find_spec, importlib.metadata.version

    def no_network(self, *args, **kwargs):
        raise AssertionError("member_code_status 不应联网")

    reasons = []
    os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"] = _FAKE_ID, _FAKE_SECRET
    socket.socket.connect = no_network
    try:
        for cfg in (dict(MODAL_DISABLED=True, MODAL_TOKENS_PRESENT=True),
                    dict(MODAL_DISABLED=False, MODAL_TOKENS_PRESENT=False),
                    dict(MODAL_DISABLED=False, MODAL_TOKENS_PRESENT=True)):
            with _patched(config, **cfg):
                for spec_result, version in ((None, "1.6.4"), (object(), "1.5.0"), (object(), "abc"), (object(), "1.6.4")):
                    importlib.util.find_spec = lambda name, *a, _s=spec_result, **k: _s if name == "modal" else real_spec(name, *a, **k)
                    importlib.metadata.version = lambda name, _v=version: _v if name == "modal" else real_version(name)
                    reasons.append(modal_sandbox.member_code_status()[1])
    finally:
        socket.socket.connect = real_connect
        importlib.util.find_spec, importlib.metadata.version = real_spec, real_version
        os.environ.pop("MODAL_TOKEN_ID", None)
        os.environ.pop("MODAL_TOKEN_SECRET", None)
    assert reasons and not any(_FAKE_ID in r or _FAKE_SECRET in r for r in reasons), reasons


# 没有 modal、没有令牌时也必须能 import 的模块（随各里程碑追加）
_LAZY_MODULES = (
    "app.infra.modal_sandbox",
    "app.infra.modal_image",
)


def test_b6_lazy_import_of_modal():
    """B6 导入这些模块、调用 member_code_status 之后，modal 仍不在 sys.modules（惰性导入）。"""
    code = (
        "import sys\n"
        f"for name in {list(_LAZY_MODULES)!r}:\n"
        "    __import__(name)\n"
        "from app.infra import modal_sandbox\n"
        "modal_sandbox.member_code_status()\n"
        "leaked = [m for m in sys.modules if m == 'modal' or m.startswith('modal.')]\n"
        "assert not leaked, f'modal 被提前导入: {leaked[:5]}'\n"
    )
    env = dict(os.environ)
    env["MODAL_TOKEN_ID"], env["MODAL_TOKEN_SECRET"] = _FAKE_ID, _FAKE_SECRET     # 就算令牌齐全，门控判断也不能 import modal
    res = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr


# ══════════════════════════════════════════════════════════════════════════════
# 公用夹具
# ══════════════════════════════════════════════════════════════════════════════

UID = "member001"                      # 合法的 openid 形状
UID2 = "member002"
_MSG_UNAVAILABLE = "云端代码沙箱暂时不可用，请稍后再试"
_MSG_TOTAL_BUSY = "云端代码沙箱现在比较繁忙，请稍后再试"
_MSG_USER_CAP = "你同时运行的代码沙箱已达上限（{n} 个），请等当前任务结束后再试"


class _Env:
    """每个场景一个：注入 FakeModal、审计日志写到临时目录、按需覆盖 config.*，退出时全部还原。
    pool 的时钟是 fake.clock（池 / 熔断 / 驱逐 / 耗时都用它，不用真等）。"""

    def __init__(self, **cfg):
        self.cfg = cfg

    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="modal-smoke-")
        self.audit_path = os.path.join(self.tmp, "audit", "modal_audit.log")
        self._saved = {}
        for key, value in {"MODAL_AUDIT_LOG": self.audit_path, **self.cfg}.items():
            self._saved[key] = getattr(config, key)
            setattr(config, key, value)
        self.fake = FakeModal()
        self.fake.publish_default_image()
        modal_sandbox.set_modal_module(self.fake)
        self.pool = modal_sandbox.ModalSandboxPool(self.fake, clock=self.fake.clock)
        return self

    def __exit__(self, *exc):
        try:
            self.pool.shutdown(deadline_s=2.0)
        finally:
            modal_sandbox.set_modal_module(None)             # 同时关闭审计 handler（Windows 上不关就删不掉临时目录）
            for key, value in self._saved.items():
                setattr(config, key, value)
            shutil.rmtree(self.tmp, ignore_errors=True)
        return False

    def use_pool(self) -> None:
        """让模块级函数与 ModalRoutedSandbox 走这个（带 FakeClock 的）池。"""
        modal_sandbox._pool = self.pool

    def audit(self) -> list:
        if not os.path.exists(self.audit_path):
            return []
        with open(self.audit_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def audit_text(self) -> str:
        return open(self.audit_path, encoding="utf-8").read() if os.path.exists(self.audit_path) else ""


@contextlib.contextmanager
def _environ(mapping: dict):
    saved = {k: os.environ.get(k) for k in mapping}
    os.environ.update(mapping)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _setattrs(obj, **attrs):
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


class _Job:
    """后台线程跑一次调用；join 带总时限，挂死变成断言失败而不是卡住 CI。"""

    def __init__(self, fn, *args, **kwargs):
        self.result = None
        self.error = None
        self._thread = threading.Thread(target=self._run, args=(fn, args, kwargs), daemon=True)
        self._thread.start()

    def _run(self, fn, args, kwargs):
        try:
            self.result = fn(*args, **kwargs)
        except BaseException as e:
            self.error = e

    def done(self) -> bool:
        return not self._thread.is_alive()

    def join(self, timeout: float = 15.0):
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "后台调用没有在时限内返回（挂死）"
        return self

    def value(self, timeout: float = 15.0):
        self.join(timeout)
        if self.error is not None:
            raise self.error
        return self.result


def _png(w: int = 8, h: int = 8, color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def _bash_calls(fsb):
    return fsb.calls(lambda inner: tuple(inner[:2]) == ("bash", "-c"))


def _py_calls(fsb):
    return fsb.calls(lambda inner: tuple(inner[:2]) == ("python", "-u"))


def _sha(value: str, n: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


# ══════════════════════════════════════════════════════════════════════════════
# S FakeModal 自检（FakeModal 自己错了，后面所有场景都没意义）
# ══════════════════════════════════════════════════════════════════════════════

def _fresh_fake_sandbox(fake, **kwargs):
    app = fake.App.lookup("selfcheck", create_if_missing=True)
    return fake.Sandbox.create(
        "sleep", "infinity", app=app, image=fake.Image.from_name(config.MODAL_IMAGE_NAME),
        timeout=kwargs.pop("timeout", 3600), idle_timeout=kwargs.pop("idle_timeout", 600), **kwargs,
    )


def test_s_fake_modal_self_check():
    """S FakeModal 自检：returncode 提前访问抛错、兜底到点返回 -1 且静默、没有 kill、未知参数 TypeError、
    同名异常不继承内建、publish 要先 build、write_bytes 参数顺序、terminate 两种流行为、空闲 / 存活期死亡。"""
    fake = FakeModal()
    fake.publish_default_image()
    E = fake.exception
    sb = _fresh_fake_sandbox(fake, tags={"svc": "gnk48"})

    proc = sb.exec("echo", "x", text=False)
    try:
        proc.returncode
        raise AssertionError("wait() 之前访问 returncode 必须抛 InvalidError")
    except E.InvalidError:
        pass
    assert proc.wait() == 0 and proc.returncode == 0
    assert not hasattr(proc, "kill"), "ContainerProcess 没有 kill()"

    sb.on_exec("sleep 100", Hang())
    p2 = sb.exec("timeout", "-k", "3", "5", "bash", "-c", "sleep 100", text=False)
    assert p2.poll() is None
    assert fake.fire_exec_deadline(p2) is True
    assert p2.wait() == -1 and p2.poll() == -1
    assert list(p2.stdout) == [] and list(p2.stderr) == [], "兜底到点后输出流静默结束，不抛异常"

    for fn in (
        lambda: fake.Sandbox.create("sleep", bogus=1),
        lambda: sb.exec("true", bogus=1),
        lambda: fake.App.lookup("x", bogus=1),
        lambda: fake.Image.from_name("x", bogus=1),
        lambda: fake.Image.debian_slim(bogus=1),
        lambda: sb.terminate(bogus=1),
        lambda: list(fake.Sandbox.list(bogus=1)),
    ):
        try:
            fn()
            raise AssertionError("未知关键字参数必须 TypeError")
        except TypeError:
            pass

    assert not issubclass(E.ConnectionError, builtins.ConnectionError)
    assert not issubclass(E.TimeoutError, builtins.TimeoutError)
    assert issubclass(E.ExecTimeoutError, E.TimeoutError) and issubclass(E.OutputExpiredError, E.TimeoutError)
    assert issubclass(E.SandboxFilesystemFileTooLargeError, E.SandboxFilesystemError)
    assert all(issubclass(getattr(E, n), E.Error) for n in ("AuthError", "NotFoundError", "ConnectionError", "SandboxTerminatedError"))

    img = fake.Image.debian_slim(python_version="3.13").apt_install("jq")
    try:
        img.publish("x:v1")
        raise AssertionError("没有 build 过的镜像不能 publish")
    except E.InvalidError:
        pass
    with fake.enable_output():
        assert img.build(None) is img
    assert img.built and img.built_in_output
    img.publish("x:v1")
    assert fake.published["x:v1"] is img
    base = fake.Image.debian_slim()
    chained = base.pip_install("a")
    assert base.ops != chained.ops and len(chained.ops) == len(base.ops) + 1, "Image 是不可变的链式记录"
    try:
        fake.Sandbox.create("sleep", app=None, image=fake.Image.from_name("nope:v9"))
        raise AssertionError("镜像没发布，create 必须抛 NotFoundError")
    except E.NotFoundError:
        pass

    fs = sb.filesystem
    fs.write_bytes(b"hello", "/work/a.txt")
    assert fs.read_bytes("/work/a.txt") == b"hello"
    try:
        fs.write_bytes("/work/b.txt", b"hello")
        raise AssertionError("write_bytes 的参数顺序是 (data, path)：写反了必须报错")
    except (TypeError, E.SandboxFilesystemError):
        pass
    assert not fs.has("/work/b.txt")
    try:
        fs.read_bytes("work/a.txt")
        raise AssertionError("相对路径必须报错")
    except E.SandboxFilesystemError:
        pass
    for fn, exc in (
        (lambda: fs.read_bytes("/work/none"), E.SandboxFilesystemNotFoundError),
        (lambda: fs.read_bytes("/work"), E.SandboxFilesystemIsADirectoryError),
    ):
        try:
            fn()
            raise AssertionError("应抛 " + exc.__name__)
        except exc:
            pass
    fs.fs_max_bytes = 3
    try:
        fs.read_bytes("/work/a.txt")
        raise AssertionError("超大文件必须抛 FileTooLarge")
    except E.SandboxFilesystemFileTooLargeError:
        pass

    # terminate：幂等；两种模式下正在跑的进程怎么收尾
    for mode in ("end", "raise"):
        fake.terminate_mode = mode
        s = _fresh_fake_sandbox(fake)
        s.on_exec("sleep 100", Hang())
        p = s.exec("bash", "-c", "sleep 100", text=False)
        s.terminate(wait=False)
        s.terminate(wait=False)
        assert s.poll() is not None and s.terminate_calls == 2
        if mode == "end":
            assert p.wait() == -1 and list(p.stdout) == []
        else:
            try:
                p.wait()
                raise AssertionError("raise 模式下 wait() 应抛 SandboxTerminatedError")
            except E.SandboxTerminatedError:
                pass
            try:
                next(iter(p.stdout))
                raise AssertionError("raise 模式下读流应抛 SandboxTerminatedError")
            except E.SandboxTerminatedError:
                pass
        try:
            s.exec("true")
            raise AssertionError("死后 exec 必须抛 SandboxTerminatedError")
        except E.SandboxTerminatedError:
            pass

    fake.terminate_mode = "end"
    idle = _fresh_fake_sandbox(fake, idle_timeout=600)
    idle.exec("true", text=False)
    fake.advance(599)
    assert idle.poll() is None
    fake.advance(2)
    assert isinstance(idle.poll(), int), "空闲到点后沙箱死亡"
    lifetime = _fresh_fake_sandbox(fake, timeout=100, idle_timeout=None)
    fake.advance(101)
    assert isinstance(lifetime.poll(), int), "存活期到点后沙箱死亡"

    live = _fresh_fake_sandbox(fake, tags={"svc": "gnk48", "u": "a"})
    other = _fresh_fake_sandbox(fake, tags={"svc": "other"})
    listed = list(fake.Sandbox.list(tags={"svc": "gnk48"}))
    assert live in listed and other not in listed and all(s.alive for s in listed)


# ══════════════════════════════════════════════════════════════════════════════
# A 真实 SDK 的表面（没装 modal 就 SKIP；只做签名与属性存在性检查，不调用任何会联网的函数）
# ══════════════════════════════════════════════════════════════════════════════

def test_a_real_sdk_surface():
    """A 我们实际调用的每个接口真实 SDK 都接受或存在；反方向：FakeModal 的同名接口不得接受真实 SDK 不认识的参数。"""
    if importlib.util.find_spec("modal") is None:
        print("    [SKIP] 没有安装 modal 包，跳过真实 SDK 表面核对")
        return
    import modal                                     # 只在测试里导入真实 SDK；导入不联网
    import modal.exception as mex

    targets = {
        "Sandbox.create": modal.Sandbox.create, "Sandbox.exec": modal.Sandbox.exec, "Sandbox.list": modal.Sandbox.list,
        "Sandbox.terminate": modal.Sandbox.terminate, "App.lookup": modal.App.lookup,
        "Image.from_name": modal.Image.from_name, "Image.debian_slim": modal.Image.debian_slim,
        "Image.apt_install": modal.Image.apt_install, "Image.pip_install": modal.Image.pip_install,
        "Image.run_commands": modal.Image.run_commands, "Image.env": modal.Image.env,
        "Image.workdir": modal.Image.workdir, "Image.add_local_file": modal.Image.add_local_file,
        "Image.publish": modal.Image.publish,
    }
    assert set(targets) == set(FakeModal.ALLOWED), set(targets) ^ set(FakeModal.ALLOWED)
    for where, allowed in FakeModal.ALLOWED.items():
        real_params = set(inspect.signature(targets[where]).parameters)
        unknown = allowed - real_params
        assert not unknown, f"FakeModal 的 {where} 接受真实 SDK 不认识的参数: {sorted(unknown)}"

    # 我们真的会传的关键字（跑一遍 FakeModal 记录下来）都在真实签名里
    with _Env() as e:
        with e.pool.lease(UID, "sess-a") as sb:
            sb.run_python("print(1)")
            sb.execute("echo hi")
            sb.upload_files([("/work/x.txt", b"x")])
            sb.download_files(["/work/x.txt"])
        fsb = e.fake.created[0]
        create_params = set(inspect.signature(modal.Sandbox.create).parameters)
        exec_params = set(inspect.signature(modal.Sandbox.exec).parameters)
        assert set(fsb.kwargs) <= create_params, set(fsb.kwargs) - create_params
        for call in fsb.exec_calls:
            assert set(call.kwargs) <= exec_params, set(call.kwargs) - exec_params
        assert set(e.fake.lookup_calls[0]) - {"name"} <= set(inspect.signature(modal.App.lookup).parameters) | {"create_if_missing"}

    for attr in ("poll", "terminate", "exec", "set_tags", "get_tags", "object_id", "filesystem"):
        assert hasattr(modal.Sandbox, attr), f"modal.Sandbox 缺少 {attr}"
    assert hasattr(modal, "enable_output") and hasattr(modal.Image, "build")
    for name in ("write_bytes", "read_bytes", "make_directory", "remove", "list_files", "stat"):
        from modal.sandbox._filesystem import _SandboxFilesystem
        assert hasattr(_SandboxFilesystem, name), f"文件系统接口缺少 {name}"

    fake_exc = FakeModal.exception
    for name in dir(fake_exc):
        cls = getattr(fake_exc, name)
        if isinstance(cls, type) and issubclass(cls, Exception) and name[0].isupper():
            assert hasattr(mex, name), f"modal.exception 里没有 {name}"
    for name in ("SandboxTimeoutError", "ExecTimeoutError", "OutputExpiredError"):
        assert issubclass(getattr(mex, name), mex.TimeoutError) == issubclass(getattr(fake_exc, name), fake_exc.TimeoutError)
    for name in ("ConnectionError", "TimeoutError"):
        assert not issubclass(getattr(mex, name), getattr(builtins, name)), f"真实 SDK 的 {name} 现在继承内建了：实现里的 except 需要复核"

    from modal.container_process import ContainerProcess
    print(f"    [INFO] ContainerProcess 有 kill: {hasattr(ContainerProcess, 'kill')}（FakeModal 按“没有”模拟）")


# ══════════════════════════════════════════════════════════════════════════════
# C 创建参数与环境
# ══════════════════════════════════════════════════════════════════════════════

def test_c_creation_parameters_and_env():
    """C Sandbox.create 收到的恰是文档那一组参数；环境变量只有四项白名单；没有 secrets / volumes / 端口 / pty / name；
    tags 只含哈希；创建后建过目录；App.lookup 只调用一次；镜像不存在时给模型通用短句、给运营者指向构建脚本。"""
    sensitive = {
        "MODAL_TOKEN_ID": "ak-FAKEFAKEFAKE1234", "MODAL_TOKEN_SECRET": "as-FAKEFAKEFAKE5678",
        "GEMINI_API_KEY": "gem-secret-1", "MYSQL_PASSWORD": "pw-secret-1", "TAVILY_API_KEY": "tv-secret-1",
    }
    with _environ(sensitive), _Env() as e:
        with e.pool.lease(UID, "sess-1") as sb:
            sb.execute("echo hi")
        with e.pool.lease(UID, "sess-2") as sb:
            sb.execute("echo hi")
        first, second = e.fake.created
        kw = first.kwargs
        assert first.args == ("sleep", "infinity")
        assert set(kw) == {"app", "image", "env", "timeout", "idle_timeout", "workdir", "cpu", "memory", "block_network", "tags"}, set(kw)
        assert kw["app"] is e.fake.apps[config.MODAL_APP_NAME]
        assert kw["image"].ops == (("from_name", (config.MODAL_IMAGE_NAME,), {}),), kw["image"].ops
        assert kw["env"] == {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8", "HOME": "/work"}
        assert kw["timeout"] == config.MODAL_MAX_LIFETIME_SECONDS and kw["idle_timeout"] == config.MODAL_IDLE_TIMEOUT_SECONDS
        assert kw["workdir"] == "/work" and kw["block_network"] is config.MODAL_BLOCK_NETWORK
        assert kw["cpu"] == (config.MODAL_CPU_REQUEST, config.MODAL_CPU_LIMIT)
        assert kw["memory"] == (config.MODAL_MEMORY_MB_REQUEST, config.MODAL_MEMORY_MB_LIMIT)
        assert kw["tags"] == {"svc": "gnk48", "role": "member-code", "u": _sha(UID, 12), "s": _sha("sess-1", 8)}, kw["tags"]
        assert second.kwargs["tags"]["s"] == _sha("sess-2", 8) and second.kwargs["tags"]["u"] == _sha(UID, 12)
        text = repr((first.args, kw))
        assert UID not in text and "sess-1" not in text, "tags / 参数里不得有明文用户或会话 ID"
        assert not any(v in text for v in sensitive.values()), "沙箱的创建参数里不得出现任何密钥"
        assert first.exec_calls[0].argv == ("mkdir", "-p", "/work/outputs", "/work/.sandbox")
        for call in first.exec_calls:
            assert "env" not in call.kwargs and call.kwargs.get("workdir") == "/work" and call.kwargs.get("text") is False, call.kwargs
        assert len(e.fake.lookup_calls) == 1 and e.fake.lookup_calls[0] == {"name": config.MODAL_APP_NAME, "create_if_missing": True}

    with _Env(MODAL_BLOCK_NETWORK=True, MODAL_CPU_REQUEST=0.5, MODAL_CPU_LIMIT=1.0, MODAL_MEMORY_MB_LIMIT=768) as e:
        with e.pool.lease(UID, "s") as sb:
            sb.execute("true")
        kw = e.fake.created[0].kwargs
        assert kw["block_network"] is True and kw["cpu"] == (0.5, 1.0) and kw["memory"][1] == 768, kw

    with _Env() as e:                                    # 镜像没有发布
        e.fake.published.clear()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with e.pool.lease(UID, "s") as sb:
                try:
                    sb.run_python("print(1)")
                    raise AssertionError("镜像不存在时 run_python 应抛 SandboxUnavailable")
                except SandboxUnavailable as ex:
                    message = str(ex)
                resp = sb.execute("echo")
        assert message == _MSG_UNAVAILABLE, message
        assert "build_modal_image.py" in out.getvalue(), "给运营者的提示要指向构建脚本"
        for leaked in ("镜像", "NotFound", "modal", "Modal"):
            assert leaked not in message, f"给模型的文案不得泄露基础设施细节: {leaked}"
        assert resp.exit_code == 1 and resp.output == "[云端沙箱不可用] " + _MSG_UNAVAILABLE, resp


def _wait_until(predicate, timeout: float = 5.0, what: str = "条件") -> None:
    """轮询到条件成立；超时变成断言失败（测试里的等待，不是被测代码里的）。"""
    ev = threading.Event()
    waited = 0.0
    while not predicate():
        assert waited < timeout, f"等待{what}超时"
        ev.wait(0.01)
        waited += 0.01


def _touch(pool, user, session, command="true"):
    """租用并执行一条命令（触发沙箱创建）。"""
    with pool.lease(user, session) as sb:
        return sb.execute(command)


# ══════════════════════════════════════════════════════════════════════════════
# D 池、租约与驱逐
# ══════════════════════════════════════════════════════════════════════════════

def test_d_pool_lease_and_eviction():
    """D 同一 (用户, 会话) 复用、不同会话各一个；每用户 / 全局上限按 LRU 驱逐空闲条目；全忙才拒绝（两种文案、两种审计）；
    忙的条目绝不被驱逐；lease(create=False)；reset_*；shutdown 并行且限时；池锁不跨网络调用。"""
    # 1 复用与独立
    with _Env() as e:
        with e.pool.lease(UID, "s1") as a:
            a.execute("true")
        with e.pool.lease(UID, "s1") as a2:
            a2.execute("true")
        assert a is a2 and len(e.fake.created) == 1
        _touch(e.pool, UID, "s2")
        _touch(e.pool, UID2, "s1")
        assert len(e.fake.created) == 3, "不同会话、不同用户的同名会话各一个沙箱"

    # 2 每用户上限（2）：第 3 个会话驱逐该用户最久空闲的那个（先 s1 再 s2，再用一次 s1 → s2 最久没用）
    with _Env() as e:
        for s in ("s1", "s2", "s1"):
            _touch(e.pool, UID, s)
            e.fake.clock.advance(1)
        _touch(e.pool, UID, "s3")
        s1, s2, s3 = e.fake.created
        assert s1.alive and s3.alive and not s2.alive and s2.dead_reason == "terminate", (s1.alive, s2.alive, s3.alive)
        assert any(r["event"] == "terminate" and r.get("reason") == "evict" for r in e.audit())
        assert len(e.pool.snapshot()) == 2

    # 3 全局上限（3）：跨用户按 LRU 驱逐
    with _Env(MODAL_MAX_SANDBOXES_PER_USER=2, MODAL_MAX_SANDBOXES_TOTAL=3) as e:
        for user, s in ((UID, "s1"), (UID2, "s1"), (UID, "s2"), ("member003", "s1")):
            _touch(e.pool, user, s)
            e.fake.clock.advance(1)
        created = e.fake.created
        assert not created[0].alive and all(sb.alive for sb in created[1:]), [sb.alive for sb in created]

    # 4 全部都在忙：拒绝，两种文案、两种审计
    with _Env() as e:
        with e.pool.lease(UID, "s1"), e.pool.lease(UID, "s2"):
            try:
                with e.pool.lease(UID, "s3"):
                    raise AssertionError("每用户上限且全忙时应拒绝")
            except SandboxUnavailable as ex:
                assert str(ex) == _MSG_USER_CAP.format(n=2), str(ex)
        assert any(r["event"] == "reject" and r.get("reason") == "cap_user" for r in e.audit())
    with _Env(MODAL_MAX_SANDBOXES_PER_USER=3, MODAL_MAX_SANDBOXES_TOTAL=3) as e:
        with e.pool.lease(UID, "s1"), e.pool.lease(UID2, "s1"), e.pool.lease("member003", "s1"):
            try:
                with e.pool.lease("member004", "s1"):
                    raise AssertionError("全局上限且全忙时应拒绝")
            except SandboxUnavailable as ex:
                assert str(ex) == _MSG_TOTAL_BUSY, str(ex)
        assert any(r["event"] == "reject" and r.get("reason") == "cap_total" for r in e.audit())

    # 5 忙的条目绝不被驱逐（线程 A 一直占着 s1 的命令，B 触发驱逐）
    with _Env() as e:
        e.fake.on_exec_all("hold-s1", Hang())
        holder = _Job(lambda: _touch(e.pool, UID, "s1", "hold-s1"))
        _wait_until(lambda: e.fake.created and e.fake.created[0].wait_for_proc("hold-s1", timeout=0.01), what="s1 的命令开始运行")
        _touch(e.pool, UID, "s2")
        e.fake.clock.advance(1)
        _touch(e.pool, UID, "s3")                       # 每用户上限 2：s1 在忙，只能驱逐空闲的 s2
        s1, s2, s3 = e.fake.created
        assert s1.alive and not s2.alive and s3.alive, (s1.alive, s2.alive, s3.alive)
        assert s1.terminate_calls == 0, "忙的沙箱不能被 terminate"
        _wait_until(lambda: s1.hanging(), what="s1 的命令进入挂起")
        s1.hanging()[0].fire_inner_timeout()
        holder.value()

    # 6 create=False：没有条目就交出 None，不创建任何东西
    with _Env() as e:
        with e.pool.lease(UID, "nope", create=False) as sb:
            assert sb is None
        assert e.fake.created == [] and e.pool.snapshot() == []

    # 7 reset_session / reset_user 的返回值与被 terminate 的次数
    with _Env(MODAL_MAX_SANDBOXES_PER_USER=3) as e:
        for s in ("s1", "s2", "s3"):
            _touch(e.pool, UID, s)
        e.pool.reset_session(UID, "s2")
        e.pool.reset_session(UID, "missing")
        assert e.fake.terminate_count == 1 and not e.fake.created[1].alive
        assert e.pool.reset_user(UID2) == 0
        assert e.pool.reset_user(UID) == 2 and e.fake.terminate_count == 3
        assert e.pool.reset_user(UID) == 0 and e.pool.snapshot() == []
        reasons = {r.get("reason") for r in e.audit() if r["event"] == "terminate"}
        assert {"reset", "reset_user"} <= reasons, reasons

    # 8 shutdown：之后 lease 被拒绝；并行关闭，一个 terminate 卡住也不会让整体超过 deadline
    with _Env(MODAL_MAX_SANDBOXES_PER_USER=3) as e:
        for s in ("s1", "s2", "s3"):
            _touch(e.pool, UID, s)
        gate = threading.Event()
        e.fake.created[1].terminate_block = gate
        started = time.perf_counter()
        e.pool.shutdown(deadline_s=0.5)
        elapsed = time.perf_counter() - started
        assert elapsed < 4.0, f"shutdown 不应被卡住的 terminate 拖住: {elapsed:.2f}s"
        assert e.fake.created[0].terminated_by_terminate and e.fake.created[2].terminated_by_terminate
        gate.set()
        try:
            with e.pool.lease(UID, "s9"):
                raise AssertionError("shutdown 之后 lease 必须被拒绝")
        except SandboxUnavailable:
            pass
        assert any(r["event"] == "terminate" and r.get("reason") == "shutdown" for r in e.audit())

    # 9 池锁绝不跨网络调用持有：一次创建被卡住时，其他用户的 reset / lease / 创建都能立刻完成
    with _Env() as e:
        gate = threading.Event()
        e.fake.on_create(Block(gate))
        slow = _Job(lambda: _touch(e.pool, UID, "s1"))
        _wait_until(lambda: len(e.fake.create_attempts) == 1, what="创建开始")
        quick = _Job(lambda: (e.pool.reset_user(UID2), e.pool.reset_session(UID2, "x"), e.pool.snapshot(),
                              _touch(e.pool, UID2, "s1").exit_code))
        assert quick.value(timeout=5)[3] == 0, "创建被卡住时，别的用户的调用不应被池锁拖住"
        assert not slow.done()
        gate.set()
        assert slow.value().exit_code == 0


# ══════════════════════════════════════════════════════════════════════════════
# E 执行语义（对照退出码与超时判定表逐行）
# ══════════════════════════════════════════════════════════════════════════════

def test_e_execution_semantics():
    """E 命令包装、退出码 / 超时判定、不重试、重建 note、终止时不卡死、timeout 夹紧、截断、UTF-8、流硬上限、串行化。"""
    E_ = FakeModal.exception

    with _Env() as e:
        with e.pool.lease(UID, "s1") as sb:
            sb.execute("echo hi")
            fsb = e.fake.created[0]
            call = _bash_calls(fsb)[0]
            assert call.argv == ("timeout", "-k", "3", "120", "bash", "-c", "echo hi"), call.argv
            assert call.kwargs == {"workdir": "/work", "timeout": 132, "text": False}, call.kwargs
            fsb.filesystem.put("/work/data.csv", b"1")
            res = sb.run_python("print(1)")
            pcall = _py_calls(fsb)[0]
            assert pcall.argv == ("timeout", "-k", "3", "120", "python", "-u", "/work/.sandbox/snippet.py"), pcall.argv
            assert pcall.kwargs == {"workdir": "/work", "timeout": 132, "text": False}
            assert fsb.filesystem.get("/work/.sandbox/snippet.py") == b"print(1)"
            assert isinstance(res, RunResult) and res.exit_code == 0 and not res.timed_out and "data.csv" in res.work_listing, res

            # 退出码 0 / 非零
            fsb.on_exec("echo ok", Out(stdout=b"ok\n"))
            r = sb.execute("echo ok")
            assert (r.output, r.exit_code, r.truncated) == ("ok\n", 0, False), r
            fsb.on_exec("boom", Out(stderr=b"bad\n", exit=3))
            r = sb.execute("boom")
            assert r.exit_code == 3 and r.output == "bad\n", r

            # 124：容器里的 timeout 触发 → timed_out + note + 残留进程清理，沙箱保留
            fsb.on_exec("slow", Out(exit=124))
            before = len(fsb.softkill_scripts)
            r = sb.execute("slow")
            assert r.exit_code == 124 and "命令执行超过 120秒被终止；沙箱状态保留" in r.output, r
            assert len(fsb.softkill_scripts) == before + 1 and fsb.alive
            fsb.on_exec("slowpy", Out(exit=124))
            fsb.on_exec(("python", "-u"), Out(exit=124))
            res = sb.run_python("while True: pass")
            assert res.timed_out and res.exit_code == 124 and any("命令执行超过 120秒被终止" in n for n in res.notes), res

            # 137 且耗时 < N：进程被杀（很可能内存超限，且沙箱可能被回收）；137 且耗时 ≥ N：超时
            fsb.on_exec("oom", Out(exit=137))
            r = sb.execute("oom")
            assert r.exit_code == 137 and (
                f"进程被杀（可能超出内存上限 {config.MODAL_MEMORY_MB_LIMIT} MiB；沙箱可能已被回收，/work 里的文件已丢失）"
                in r.output
            ), r
            assert "超过" not in r.output
            fsb.on_exec("late", Sleep(121, Out(exit=137)))
            r = sb.execute("late")
            assert r.exit_code == 137 and "命令执行超过 120秒被终止" in r.output and "进程被杀" not in r.output, r

            # 客户端兜底 deadline 到点：-1、无异常；健康检查通过 → 沙箱保留
            fsb.on_exec("wedge", Hang())
            job = _Job(sb.execute, "wedge forever")
            proc = fsb.wait_for_proc("wedge forever")
            assert proc is not None and e.fake.fire_exec_deadline(proc)
            r = job.value()
            assert r.exit_code is None and "执行超时，云端看门狗终止了该命令" in r.output and "沙箱已重置" not in r.output, r
            assert fsb.alive and len(e.fake.created) == 1
            fsb.on_exec(("python", "-u"), Hang())                  # 先登记行为，再起线程（否则命令可能先跑到上一条规则）
            job = _Job(sb.run_python, "import time; time.sleep(999)")
            proc = fsb.wait_for_proc(("python", "-u"), nth=2)
            assert proc is not None
            e.fake.fire_exec_deadline(proc)
            res = job.value()
            assert res.timed_out and res.exit_code is None, res

            # 兜底到点且健康检查失败 → terminate 并重置；下一次调用惰性重建并带一次性 note
            fsb.on_exec(("true",), Out(exit=1))
            job = _Job(sb.execute, "wedge forever 2")
            proc = fsb.wait_for_proc("wedge forever 2")
            e.fake.fire_exec_deadline(proc)
            r = job.value()
            assert "执行超时，云端看门狗终止了该命令" in r.output and "沙箱已重置，/work 已清空" in r.output, r
            assert fsb.terminated_by_terminate and not fsb.alive
            r = sb.execute("echo again")
            assert r.output.count("云端沙箱已重建，/work 里之前的文件已不存在") == 1 and len(e.fake.created) == 2, r
            assert "已重建" not in sb.execute("echo third").output, "重建 note 只出现一次"

    # exec 本身抛异常：不重试（同一条命令只发出一次），标记已死，下一次调用惰性重建
    with _Env() as e:
        with e.pool.lease(UID, "s1") as sb:
            sb.execute("true")
            fsb = e.fake.created[0]
            fsb.on_exec("die", Raise(E_.SandboxTerminatedError("gone")))
            r = sb.execute("die now")
            assert r.exit_code == 1 and "云端沙箱已被回收（空闲超时或被终止），本次执行未完成；/work 里的文件已丢失，请重试" in r.output, r
            assert len(fsb.calls("die now")) == 1, "同一条命令只发出一次"
            r = sb.execute("echo next")
            assert len(e.fake.created) == 2 and "已重建" in r.output
            assert len(e.fake.created[1].calls("echo next")) == 1 and e.pool._fail_count == 0, "已有沙箱上的命令失败不计入熔断"

    # terminate_mode 的两种流行为都不能让运行中的命令卡死：/reset（close）后立刻返回带 note 的结果
    for mode in ("end", "raise"):
        with _Env() as e:
            e.fake.terminate_mode = mode
            e.fake.on_exec_all("hang-here", Hang())
            with e.pool.lease(UID, "s1") as sb:
                job = _Job(sb.execute, "hang-here")
                proc = None
                _wait_until(lambda: e.fake.created and e.fake.created[0].wait_for_proc("hang-here", timeout=0.01), what="命令开始运行")
                e.pool.reset_session(UID, "s1")
                r = job.value(timeout=10)
                assert "云端沙箱已被终止，本次执行未完成" in r.output, (mode, r)

    # timeout 参数夹紧：None / 0 / -5 → 默认，9999 → 上限，中间值原样
    with _Env() as e:
        with e.pool.lease(UID, "s1") as sb:
            for given, expect in ((None, "120"), (0, "120"), (-5, "120"), (7, "7"), (9999, "300")):
                sb.execute("t", timeout=given)
                assert _bash_calls(e.fake.created[0])[-1].argv[3] == expect, (given, expect)
            sb.run_python("print(1)", timeout=9999)
            assert _py_calls(e.fake.created[0])[-1].argv[3] == "300"

    # 输出：超过上限截断；被切在块边界的多字节字符与非法 UTF-8 不抛、不丢
    with _Env() as e:
        with e.pool.lease(UID, "s1") as sb:
            sb.execute("true")
            fsb = e.fake.created[0]
            fsb.on_exec("big", Out(stdout=b"a" * 1000))
            with _setattrs(config, SANDBOX_EXECUTE_MAX_OUTPUT_CHARS=100):
                r = sb.execute("big")
            assert r.truncated is True and "已省略中间" in r.output, r
            fsb.on_exec("zh", Out(stdout="你好，世界".encode("utf-8"), chunks=1))
            assert sb.execute("zh").output == "你好，世界"
            fsb.on_exec("bad", Out(stdout=b"ok\xff\xfeend"))
            r = sb.execute("bad")
            assert r.exit_code == 0 and "ok" in r.output and "end" in r.output and "�" in r.output, r
            fsb.on_exec("both", Out(stdout=b"out\n", stderr=b"err\n"))
            out = sb.execute("both").output
            assert "out" in out and "err" in out, "execute 的 stdout 与 stderr 共用一个缓冲"

    # 流的原始字节硬上限：触发恰好一次中断，结果里带「输出量过大」
    with _Env() as e:
        with e.pool.lease(UID, "s1") as sb:
            sb.execute("true")
            fsb = e.fake.created[0]
            fsb.on_exec("flood", Out(stdout=b"x" * 500, chunks=100))
            with _setattrs(modal_sandbox, _STREAM_HARD_CAP_BYTES=50):
                r = sb.execute("flood")
            assert "输出量过大，命令已终止" in r.output, r
            assert len(fsb.softkill_scripts) == 1, "硬上限只触发一次中断"

    # run_python 的 256 KiB 代码上限：直接拒绝，也不去创建沙箱
    with _Env() as e:
        with e.pool.lease(UID, "s1") as sb:
            res = sb.run_python("a" * (256 * 1024 + 1))
            assert res.exit_code == 1 and "256KB" in res.stderr and e.fake.created == [], res

    # 同一沙箱里并行的两次 execute 被 RLock 串行化
    with _Env() as e:
        e.fake.on_exec_all("serial-a", Hang())
        with e.pool.lease(UID, "s1") as sb:
            job_a = _Job(sb.execute, "serial-a")
            _wait_until(lambda: e.fake.created and e.fake.created[0].wait_for_proc("serial-a", timeout=0.01), what="a 开始运行")
            job_b = _Job(sb.execute, "serial-b")
            threading.Event().wait(0.3)
            assert e.fake.created[0].calls("serial-b") == [], "第一条命令没结束前，第二条不应开始"
            e.fake.created[0].fire_inner_timeout()
            assert job_a.value().exit_code == 124
            assert job_b.value().exit_code == 0 and len(e.fake.created[0].calls("serial-b")) == 1


# ══════════════════════════════════════════════════════════════════════════════
# F 永不抛异常
# ══════════════════════════════════════════════════════════════════════════════

_LEAKY = "ak-ABCDEFGHIJKLMN1234"          # 令牌形状：不能出现在任何返回值里

_FAULT_POINTS = ("create", "app.lookup", "exec", "stream", "wait", "poll_sandbox", "terminate",
                 "fs.write_bytes", "fs.read_bytes", "fs.make_directory")

_FAULT_EXCEPTIONS = (
    ("Error", lambda E: E.Error(f"generic {_LEAKY}")),
    ("modal.ConnectionError", lambda E: E.ConnectionError("connection reset")),
    ("OutputExpiredError", lambda E: E.OutputExpiredError("output expired")),
    ("SandboxTerminatedError", lambda E: E.SandboxTerminatedError("sandbox terminated")),
    ("RuntimeError", lambda E: RuntimeError(f"boom {_LEAKY}")),
    ("MemoryError", lambda E: MemoryError()),
)


def test_f_never_raises():
    """F 对 create / exec / 读流 / wait / poll / terminate / filesystem.* 逐个注入各类异常：
    execute / upload_files / download_files / collect_new_outputs / interrupt / close 都不抛、返回值形状正确；
    读流线程里的异常不会让主线程等不到结果；run_python 只会抛 SandboxUnavailable（通用短句，不含类名 / 令牌）。"""
    for point in _FAULT_POINTS:
        for exc_name, make in _FAULT_EXCEPTIONS:
            label = f"{point} × {exc_name}"
            with _Env() as e:
                with e.pool.lease(UID, "s1") as sb:
                    if point not in ("create", "app.lookup"):
                        sb.execute("true")                          # 先正常建好沙箱并放一张图，再注入故障
                        e.fake.created[0].filesystem.put("/work/outputs/a.png", _png())
                    e.fake.inject(point, make(e.fake.exception), times=80)
                    seen = []

                    def run(name, fn, check):
                        job = _Job(fn).join(30)
                        if job.error is not None:
                            assert name == "run_python" and isinstance(job.error, SandboxUnavailable), (label, name, repr(job.error))
                            assert str(job.error) == _MSG_UNAVAILABLE, (label, str(job.error))
                            seen.append(str(job.error))
                            return
                        assert check(job.result), (label, name, job.result)
                        seen.append(repr(job.result))

                    run("execute", lambda: sb.execute("echo hi"), lambda r: isinstance(r, ExecuteResponse) and isinstance(r.output, str))
                    run("run_python", lambda: sb.run_python("print(1)"), lambda r: isinstance(r, RunResult))
                    run("upload_files", lambda: sb.upload_files([("/work/a.txt", b"x"), ("/tmp/b.txt", b"y")]),
                        lambda r: len(r) == 2 and all(isinstance(x, FileUploadResponse) for x in r))
                    run("download_files", lambda: sb.download_files(["/work/a.txt", "/work/none"]),
                        lambda r: len(r) == 2 and all(isinstance(x, FileDownloadResponse) for x in r))
                    run("collect_new_outputs", lambda: sb.collect_new_outputs(), lambda r: isinstance(r, OutputsResult))
                    run("interrupt", lambda: sb.interrupt(), lambda r: isinstance(r, bool))
                    run("close", lambda: sb.close(), lambda r: r is None)
                    assert not any(_LEAKY in s for s in seen), (label, "令牌形状泄露到了返回值里")
                    assert "Error" not in "".join(s for s in seen if s == _MSG_UNAVAILABLE)


# ══════════════════════════════════════════════════════════════════════════════
# G 路径规则
# ══════════════════════════════════════════════════════════════════════════════

def test_g_path_rules():
    """G 上传只允许 /work 与 /tmp（精确前缀），其余各有错误码；5 MiB 上限；写入前建父目录；全部非法时不创建沙箱；下载的错误码。"""
    with _Env() as e:                                                      # 全部非法：不该为此创建云端沙箱
        with e.pool.lease(UID, "s1") as sb:
            out = sb.upload_files([("/etc/x", b"x"), ("rel/path", b"x"), ("", b"x")])
            assert len(out) == 2 + 1 and all(r.error and r.error.startswith("invalid_path") for r in out), out
            assert e.fake.created == []

    with _Env() as e:
        with e.pool.lease(UID, "s1") as sb:
            def up(path, data=b"x"):
                return sb.upload_files([(path, data)])[0]

            ok = up("/work/a.txt")
            assert ok.error is None and ok.path == "/work/a.txt"
            fsb = e.fake.created[0]
            assert fsb.filesystem.get("/work/a.txt") == b"x"
            assert up("/tmp/x").error is None and fsb.filesystem.get("/tmp/x") == b"x"

            for bad in ("/work/../etc/x", "/workfoo/x", "/tmpx/x", "a/b", "", "/work/a\x00b", "/work/./a/../../etc/passwd", "/"):
                err = up(bad).error
                assert err and err.startswith("invalid_path"), (bad, err)
            for bad in ("/workspace/x", "/workspace", "/workspace/../workspace/y"):
                err = up(bad).error
                assert err and err.startswith("permission_denied") and "/work" in err and "没有 /workspace" in err, (bad, err)
            assert up("/work").error == "is_directory", "/work 本身是目录，不能当文件写"
            assert up("/work/big.bin", b"a" * (5 * 1024 * 1024)).error is None, "刚好 5 MiB 可以"
            assert up("/work/huge.bin", b"a" * (5 * 1024 * 1024 + 1)).error == "file_too_large"
            assert not fsb.filesystem.has("/work/huge.bin")

            calls = []
            real_mkdir = fsb.filesystem.make_directory
            fsb.filesystem.make_directory = lambda p, create_parents=True: (calls.append((p, create_parents)), real_mkdir(p, create_parents=create_parents))[1]
            assert up("/work/deep/dir/f.txt").error is None
            assert calls == [("/work/deep/dir", True)], "写入前要对父目录做 make_directory(create_parents=True)"

            # 下载
            got = sb.download_files(["/work/a.txt", "/work/none", "/work", "rel", "/work/a.txt"])
            assert got[0].content == b"x" and got[0].error is None
            assert (got[1].content, got[1].error) == (None, "file_not_found")
            assert got[2].error == "is_directory"
            assert got[3].error == "invalid_path"
            assert got[4].content == b"x"
            fsb.filesystem.fs_max_bytes = 0
            assert sb.download_files(["/work/a.txt"])[0].error == "file_too_large"
            fsb.filesystem.fs_max_bytes = None
            e.fake.inject("fs.read_bytes", e.fake.exception.SandboxFilesystemPermissionError("denied"))
            assert sb.download_files(["/work/a.txt"])[0].error == "permission_denied"
            e.fake.inject("fs.read_bytes", RuntimeError("weird " + _LEAKY + " x" * 300))
            err = sb.download_files(["/work/a.txt"])[0].error
            assert err.startswith("RuntimeError") and _LEAKY not in err and len(err) <= 200, err


# ══════════════════════════════════════════════════════════════════════════════
# H 产物收集
# ══════════════════════════════════════════════════════════════════════════════

def test_h_outputs_collection():
    """H 指纹增量、图片扩展名白名单、非图片只报告不导出、大小 / 张数 / 像素上限与 notes、条数上限、
    find 输出里的 .. / 绝对路径被拒绝、沙箱未创建时不创建、图片字节来自 read_bytes。"""
    with _Env() as e:
        with e.pool.lease(UID, "s1") as sb:
            assert sb.collect_new_outputs().images == [] and e.fake.created == [], "沙箱未创建时不得为了扫描而创建"
            sb.execute("true")
            fsb = e.fake.created[0]
            fs = fsb.filesystem
            png_a, png_b = _png(8, 8), _png(4, 4, (0, 0, 255))
            fs.put("/work/outputs/a.png", png_a)
            fs.put("/work/outputs/sub/b.PNG", png_b)
            fs.put("/work/outputs/data.csv", b"1,2,3")
            fs.put("/work/outputs/x.svg", b"<svg/>")
            res = sb.run_python("print(1)")
            assert {i.name: i.data for i in res.images} == {"a.png": png_a, "b.PNG": png_b}, [i.name for i in res.images]
            assert sorted(res.other_files) == [("data.csv", 5), ("x.svg", 6)], res.other_files
            assert any("非图片产物未导出：data.csv (5 字节)" in n for n in res.notes), res.notes
            res2 = sb.run_python("print(2)")
            assert res2.images == [] and res2.other_files == [], "没变的文件不重发"
            e.fake.clock.advance(5)
            png_c = _png(9, 9, (0, 255, 0))
            fs.put("/work/outputs/a.png", png_c)
            res3 = sb.run_python("print(3)")
            assert [(i.name, i.data) for i in res3.images] == [("a.png", png_c)], "改动后再发"
            direct = sb.collect_new_outputs()
            assert direct.images == [] and direct.notes == []

    def fresh_with(files: dict, **cfg):
        env = _Env(**cfg)
        env.__enter__()
        try:
            with env.pool.lease(UID, "s1") as sb:
                sb.execute("true")
                for path, data in files.items():
                    env.fake.created[0].filesystem.put(path, data)
                return sb.collect_new_outputs()
        finally:
            env.__exit__(None, None, None)

    three = {f"/work/outputs/{n}.png": _png() for n in ("a", "b", "c")}
    res = fresh_with(three, SANDBOX_MAX_IMAGES=2)
    assert len(res.images) == 2 and any("单次生成图片超过上限 (2 张)" in n for n in res.notes), res.notes
    res = fresh_with({"/work/outputs/a.png": _png()}, SANDBOX_MAX_IMAGE_MB=0)
    assert res.images == [] and any("超过 0MB 上限" in n for n in res.notes), res.notes
    with _setattrs(modal_sandbox, _MAX_IMAGE_PIXELS=100):
        res = fresh_with({"/work/outputs/a.png": _png(20, 20)})
    assert res.images == [] and any("像素" in n and "40M 上限" in n for n in res.notes), res.notes
    res = fresh_with({"/work/outputs/bad.png": b"not really a png"})
    assert res.images == [] and any("解析校验失败" in n for n in res.notes), res.notes
    with _setattrs(modal_sandbox, _MAX_OUTPUT_ENTRIES=3):
        res = fresh_with({f"/work/outputs/f{i}.txt": b"x" for i in range(5)})
    assert len(res.other_files) == 3 and any("产物文件数量超过上限 (3 个)" in n for n in res.notes), (res.other_files, res.notes)

    with _Env() as e:                                  # find 输出里的 .. / 绝对路径一律拒绝
        with e.pool.lease(UID, "s1") as sb:
            sb.execute("true")
            fsb = e.fake.created[0]
            fsb.filesystem.put("/work/outputs/ok.png", _png())
            reads = []
            real_read = fsb.filesystem.read_bytes
            fsb.filesystem.read_bytes = lambda p: (reads.append(p), real_read(p))[1]
            crafted = b"../evil.png\t5\t1.0\0/abs.png\t5\t1.0\0a/../../x.png\t5\t1.0\0ok.png\t8\t2.0\0junk-without-tabs\0"
            fsb.on_exec(("find", "/work/outputs"), Out(stdout=crafted))
            res = sb.collect_new_outputs()
            assert [i.name for i in res.images] == ["ok.png"] and reads == ["/work/outputs/ok.png"], (res.images, reads)

    with _Env() as e:                                  # 读图失败 → note，不抛
        with e.pool.lease(UID, "s1") as sb:
            sb.execute("true")
            e.fake.created[0].filesystem.put("/work/outputs/a.png", _png())
            e.fake.inject("fs.read_bytes", RuntimeError("disk gone"))
            res = sb.collect_new_outputs()
            assert res.images == [] and any("下载图片 a.png 失败" in n for n in res.notes), res.notes

    with _Env() as e:                                  # 经路由层：用 create=False，没有条目 / 没有沙箱都不创建
        e.use_pool()
        with member_context(UID, "s-route"):
            assert modal_sandbox.get_routed_sandbox().collect_new_outputs().images == []
        assert e.fake.created == [] and e.pool.snapshot() == []


# ══════════════════════════════════════════════════════════════════════════════
# I 中止
# ══════════════════════════════════════════════════════════════════════════════

def _write_proc_tree(root: Path, procs: dict) -> None:
    """在 root 下伪造一棵 /proc：pid -> (comm, ppid, argv)。"""
    root.mkdir(parents=True, exist_ok=True)
    for pid, (comm, ppid, argv) in procs.items():
        d = root / str(pid)
        d.mkdir()
        (d / "stat").write_bytes(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1 4194304 100 0 0 0 0 0 0 0 20 0 1 0 100 1000000 100\n".encode())
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (root / "self").mkdir()                            # 非数字目录与普通文件都要被忽略
    (root / "cpuinfo").write_text("x")


_FAKE_PROCS = {
    1: ("sleep", 0, ["sleep", "infinity"]),
    100: ("timeout", 1, ["timeout", "-k", "3", "120", "bash", "-c", "sleep 99"]),
    101: ("bash", 100, ["bash", "-c", "sleep 99"]),
    102: ("(my) proc (x)", 101, ["sleep", "99"]),            # comm 含空格与括号
    103: ("sleep", 102, ["sleep", "1"]),
    200: ("python3", 1, ["python3", "-m", "daemon"]),        # 不相干的守护进程
    201: ("sh", 200, ["sh"]),
    300: ("python3", 1, ["python3", "-c", "script"]),         # 脚本自己
    301: ("sleep", 300, ["sleep", "5"]),
    400: ("timeout", 1, ["timeout", "5", "sleep", "9"]),      # 没有 -k 3：不是我们的包装
    401: ("timeout", 1, ["timeout", "-k", "10", "5", "sleep"]),
    500: ("timeout", 1, ["/usr/bin/timeout", "-k", "3", "30", "python", "-u", "x.py"]),
    501: ("python", 500, ["python", "-u", "x.py"]),
}


def test_i_interrupt_and_close():
    """I 软杀脚本的逻辑（伪造 /proc）；池层面 soft / terminate / 非法模式；软杀失败 / 抛异常 / 健康检查失败 / 超时都回退 terminate；
    interrupt() 不取锁；interrupt_session_nowait 立即返回且用守护线程；close() 幂等、不取锁。"""
    # —— 软杀脚本逻辑 ——
    killed_spy = []
    root = Path(tempfile.mkdtemp(prefix="modal-proc-"))
    try:
        _write_proc_tree(root / "a", _FAKE_PROCS)
        got = modal_sandbox._kill_wrapper_trees(str(root / "a"), self_pid=300, kill=killed_spy.append)
        assert got == [100, 101, 102, 103, 500, 501] and sorted(killed_spy) == got, (got, killed_spy)
        for pid in (1, 200, 201, 300, 301, 400, 401):
            assert pid not in killed_spy, f"不该杀 {pid}"

        # 脚本自己在包装树里（它是某个 timeout 包装的后代）：也绝不能杀它；PID 1 同理
        tree = {1: ("sleep", 0, ["sleep", "infinity"]),
                700: ("timeout", 1, ["timeout", "-k", "3", "9", "bash", "-c", "x"]),
                800: ("python3", 700, ["python3", "-c", "script"]),
                801: ("sleep", 800, ["sleep", "1"]),
                802: ("weird", 1, ["timeout", "-k", "3"])}
        _write_proc_tree(root / "b", tree)
        killed_spy.clear()
        got = modal_sandbox._kill_wrapper_trees(str(root / "b"), self_pid=800, kill=killed_spy.append)
        assert 800 not in got and 1 not in got and 700 in got and 801 in got, got
        assert modal_sandbox._kill_wrapper_trees(str(root / "missing"), self_pid=1, kill=killed_spy.append) == []

        # 自包含：只靠函数源码本身（不引用任何模块级名字）就能在干净的命名空间里跑出同样的结果；真实 kill 默认走 os.kill
        ns: dict = {}
        exec(compile(inspect.getsource(modal_sandbox._kill_wrapper_trees), "<kill>", "exec"), ns)
        killed_spy.clear()
        assert ns["_kill_wrapper_trees"](str(root / "a"), self_pid=300, kill=killed_spy.append) == [100, 101, 102, 103, 500, 501]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    script = modal_sandbox._kill_script()
    compile(script, "<kill-script>", "exec")           # 只编译，绝不真的跑（CI 上的父进程可能就是 timeout -k 3）
    assert script.lstrip().startswith("def _kill_wrapper_trees") and script.rstrip().endswith("print(_kill_wrapper_trees())")

    def start_hang(e, sb, cmd="long-run"):
        e.fake.on_exec_all(cmd, Hang())
        sb.execute("true")
        fsb = e.fake.created[-1]
        fsb.filesystem.put("/work/keep.txt", b"keep")
        job = _Job(sb.execute, cmd)
        proc = fsb.wait_for_proc(cmd)
        assert proc is not None
        _wait_until(lambda: proc.hanging, what="命令进入挂起")
        return fsb, job

    # —— 软杀成功：沙箱与 /work 保留；interrupt() 在命令持锁期间也能返回 ——
    with _Env() as e:
        with e.pool.lease(UID, "s1") as sb:
            fsb, job = start_hang(e, sb)
            assert _Job(sb.interrupt).value(10) is True, "interrupt() 不取锁，命令持锁时也必须在时限内返回"
            r = job.value(10)
            assert r.exit_code == 137 and "命令已被中止" in r.output and "沙箱已重置" not in r.output and "进程被杀" not in r.output, r
            assert fsb.alive and fsb.filesystem.get("/work/keep.txt") == b"keep"
            assert len(fsb.softkill_scripts) == 1 and len(fsb.calls(("true",))) == 1, "软杀一次 + 健康检查一次"
            assert "已重建" not in sb.execute("echo alive").output and len(e.fake.created) == 1
        assert any(r["event"] == "interrupt" and r.get("mode") == "soft" and r.get("result") == "soft" for r in e.audit())

    # —— 回退 terminate：软杀失败 / 抛异常 / 超时（挂住后兜底到点）/ 健康检查失败 ——
    def fallback_case(label, setup, after_start=None, killed_first=False):
        with _Env() as e:
            with e.pool.lease(UID, "s1") as sb:
                fsb, job = start_hang(e, sb)
                setup(fsb)
                if after_start is None:
                    assert _Job(sb.interrupt).value(10) is True, label
                else:
                    ijob = _Job(sb.interrupt)
                    after_start(e, fsb)
                    assert ijob.value(10) is True, label
                r = job.value(10)
                if killed_first:
                    # 软杀已经把命令杀掉了，之后健康检查才失败：命令看到的是“被中止”，随后沙箱被终止（命令自己的结果里没有重置 note）
                    assert r.exit_code == 137 and "命令已被中止" in r.output, (label, r)
                else:
                    assert "命令已被中止；沙箱已重置，/work 已清空" in r.output, (label, r)
                assert fsb.terminated_by_terminate and not fsb.alive, label
                nxt = sb.execute("echo again")
                assert len(e.fake.created) == 2 and "已重建" in nxt.output, label
            assert any(r["event"] == "interrupt" and r.get("result") == "terminate_fallback" for r in e.audit()), label

    fallback_case("softkill 失败", lambda f: setattr(f, "softkill_effect", "fail"))
    fallback_case("softkill 抛异常", lambda f: setattr(f, "softkill_effect", "raise"))
    fallback_case("健康检查失败", lambda f: f.on_exec(("true",), Out(exit=1)), killed_first=True)
    fallback_case(
        "softkill 挂住", lambda f: setattr(f, "softkill_effect", "hang"),
        after_start=lambda e, f: e.fake.fire_exec_deadline(
            f.wait_for_proc(lambda inner: tuple(inner[:2]) == ("python3", "-c"), timeout=10)),
    )

    # —— MODAL_INTERRUPT_MODE=terminate：不跑软杀脚本，直接终止；两种 terminate 后流行为都不卡死 ——
    for stream_mode in ("end", "raise"):
        with _Env(MODAL_INTERRUPT_MODE="terminate") as e:
            e.fake.terminate_mode = stream_mode
            with e.pool.lease(UID, "s1") as sb:
                fsb, job = start_hang(e, sb)
                assert sb.interrupt() is True
                r = job.value(10)
                assert "命令已被中止；沙箱已重置，/work 已清空" in r.output, (stream_mode, r)
                assert fsb.softkill_scripts == [] and fsb.terminate_calls == 1 and not fsb.alive
                assert "已重建" in sb.execute("echo again").output
            assert any(r["event"] == "interrupt" and r.get("mode") == "terminate" for r in e.audit())

    # —— 非法模式在运行时也按 soft 处理 ——
    with _Env(MODAL_INTERRUPT_MODE="bogus") as e:
        with e.pool.lease(UID, "s1") as sb:
            fsb, job = start_hang(e, sb)
            assert sb.interrupt() is True and len(fsb.softkill_scripts) == 1 and fsb.alive
            job.value(10)

    # —— 沙箱还没创建：返回 False 且不创建 ——
    with _Env() as e:
        with e.pool.lease(UID, "s1") as sb:
            assert sb.interrupt() is False and e.fake.created == []
        assert e.pool.interrupt_session(UID, "nope") is False and e.fake.created == []

    # —— interrupt_session_nowait：立即返回，用守护线程；池还没创建时什么都不做 ——
    with _Env() as e:
        release, entered = threading.Event(), threading.Event()

        def blocking(user_id, session_id):
            entered.set()
            release.wait(20)

        with _setattrs(modal_sandbox, interrupt_session=blocking, _pool=None):
            before = {t.ident for t in threading.enumerate()}
            modal_sandbox.interrupt_session_nowait(UID, "s1")
            assert {t.ident for t in threading.enumerate()} == before and not entered.wait(0.2), "池还没创建：不起线程"
        e.use_pool()
        with _setattrs(modal_sandbox, interrupt_session=blocking):
            started = time.perf_counter()
            modal_sandbox.interrupt_session_nowait(UID, "s1")
            assert time.perf_counter() - started < 1.0, "interrupt_session_nowait 必须立即返回"
            assert entered.wait(5)
            workers = [t for t in threading.enumerate() if t.name == "modal-interrupt"]
            assert workers and all(t.daemon for t in workers), "用守护线程"
            release.set()

    # —— close()：幂等、不取锁；运行中的命令返回带 note 的结果 ——
    for stream_mode in ("end", "raise"):
        with _Env() as e:
            e.fake.terminate_mode = stream_mode
            with e.pool.lease(UID, "s1") as sb:
                fsb, job = start_hang(e, sb)
                assert _Job(sb.close).value(5) is None and _Job(sb.close).value(5) is None
                assert "云端沙箱已被终止，本次执行未完成" in job.value(10).output
                assert fsb.terminate_calls == 1, "close() 幂等：只 terminate 一次"
                assert sb.alive is False


# ══════════════════════════════════════════════════════════════════════════════
# J 熔断、创建超时与脱敏
# ══════════════════════════════════════════════════════════════════════════════

def test_j_breaker_startup_timeout_and_redaction():
    """J 连续 3 次创建失败开路（不再调用 create）；60 秒后试探，成功恢复 / 失败再断开；已有沙箱上的命令失败不计数；
    创建超时计入熔断且迟到的沙箱被 terminate；运营者文案分类；脱敏；给模型的文案精确等于约定的短句。"""
    quiet = io.StringIO()
    unavailable = "[云端沙箱不可用] " + _MSG_UNAVAILABLE

    with _Env() as e, contextlib.redirect_stdout(quiet):
        E_ = e.fake.exception
        e.fake.create_default = Fail(E_.NotFoundError("Image not found"))
        with e.pool.lease(UID, "s1") as sb:
            for _ in range(3):
                assert sb.execute("true").output == unavailable
            assert len(e.fake.create_attempts) == 3
            assert sb.execute("true").output == unavailable and len(e.fake.create_attempts) == 3, "开路后不再调用 Sandbox.create"
            e.fake.clock.advance(59)
            assert sb.execute("true").output == unavailable and len(e.fake.create_attempts) == 3, "60 秒内仍开路"
            e.fake.clock.advance(2)                                      # 试探失败 → 再断开 60 秒
            assert sb.execute("true").output == unavailable and len(e.fake.create_attempts) == 4
            assert sb.execute("true").output == unavailable and len(e.fake.create_attempts) == 4
            e.fake.clock.advance(61)                                     # 试探成功 → 恢复
            e.fake.create_default = Ok()
            assert sb.execute("echo ok").exit_code == 0 and len(e.fake.create_attempts) == 5
            assert e.pool._fail_count == 0 and e.pool._open_until is None
        events = [(r["event"], r.get("reason")) for r in e.audit()]
        assert ("breaker_open", None) in events and ("breaker_close", None) in events and ("reject", "breaker") in events, events
        assert sum(1 for ev, _ in events if ev == "breaker_open") == 2

    # 已有沙箱上的命令失败不计入熔断
    with _Env() as e, contextlib.redirect_stdout(quiet):
        with e.pool.lease(UID, "s1") as sb:
            sb.execute("true")
            e.fake.created[0].on_exec("die", Raise(e.fake.exception.SandboxTerminatedError("gone")))
            for _ in range(5):
                sb.execute("die")                                        # 每次失败都会重建沙箱（重建成功会清零）
                e.fake.created[-1].on_exec("die", Raise(e.fake.exception.SandboxTerminatedError("gone")))
            assert e.pool._fail_count == 0 and e.pool._open_until is None

    # 运营者文案分类
    with _Env() as e:
        E_ = e.fake.exception
        cases = (
            (E_.NotFoundError("x"), "image_missing", "build_modal_image.py"),
            (E_.AuthError("x"), "auth", "凭证无效或已过期"),
            (E_.ResourceExhaustedError("x"), "quota", "额度不足"),
            (E_.ConnectionError("x"), "network", "连不上 Modal"),
            (E_.ServiceError("x"), "network", "连不上 Modal"),
            (E_.TimeoutError("x"), "network", "连不上 Modal"),
            (E_.ExecTimeoutError("x"), "network", "连不上 Modal"),
            (modal_sandbox._CreateTimeout("x"), "network", "连不上 Modal"),
        )
        for exc, category, needle in cases:
            got_category, text = modal_sandbox._describe_failure(exc, e.fake)
            assert got_category == category and needle in text, (type(exc).__name__, got_category, text)
        category, text = modal_sandbox._describe_failure(builtins.ConnectionError("os level"), e.fake)
        assert category == "other" and text.startswith("ConnectionError"), "内建的 ConnectionError 不是 modal 的，不能混为一谈"
        category, text = modal_sandbox._describe_failure(RuntimeError("y" * 1000), e.fake)
        assert category == "other" and text.startswith("RuntimeError") and len(text) <= 200

        # 系统代理已启用但缺 python-socks：SDK 把 ImportError 包成 ConnectionError，真因只在 __cause__ 链里
        def _proxy_error(inner: str) -> BaseException:
            exc = e.fake.exception.ConnectionError("Could not connect to the Modal server.")
            exc.__cause__ = ImportError(inner)
            return exc

        for inner in (
            "A proxy is configured (http://127.0.0.1:7897) but the 'python-socks' package is not installed. "
            "Install it with: pip install 'modal[api-proxy-support]'",
            "No module named 'python_socks'",
        ):
            proxy_exc = _proxy_error(inner)
            category, text = modal_sandbox._describe_failure(proxy_exc, e.fake)
            assert category == "proxy", (category, text)
            assert "python-socks" in text and "MODAL_DISABLE_API_PROXY" in text, text
            assert modal_sandbox.root_cause(proxy_exc).startswith("ImportError:"), modal_sandbox.root_cause(proxy_exc)
            explained = modal_sandbox.explain(proxy_exc)
            assert "真原因" in explained and "python" in explained, explained
        # 没有下层原因时 explain 只给一行，不画蛇添足
        assert modal_sandbox.explain(RuntimeError("boom")) == "RuntimeError: boom"
        assert modal_sandbox.root_cause(RuntimeError("boom")) == "RuntimeError: boom"

        # 启动时的代理自检：会用代理 + 缺 python-socks = 一定连不上
        with _setattrs(modal_sandbox, _modal_uses_proxy=lambda: True):
            assert modal_sandbox._proxy_misconfigured(has_python_socks=False) is True
            assert modal_sandbox._proxy_misconfigured(has_python_socks=True) is False
        with _setattrs(modal_sandbox, _modal_uses_proxy=lambda: False):
            assert modal_sandbox._proxy_misconfigured(has_python_socks=False) is False

    # 创建超时：计入熔断；迟到的沙箱必须被 terminate
    with _Env(MODAL_STARTUP_TIMEOUT_SECONDS=0.3) as e, contextlib.redirect_stdout(quiet):
        gate = threading.Event()
        e.fake.on_create(Block(gate))
        with e.pool.lease(UID, "s1") as sb:
            assert sb.execute("true").output == unavailable
        assert e.pool._fail_count == 1 and e.fake.created == []
        gate.set()
        _wait_until(lambda: len(e.fake.created) == 1 and not e.fake.created[0].alive, what="迟到的沙箱被终止")
        assert e.fake.live() == [] and e.fake.created[0].terminate_calls >= 1
        _wait_until(lambda: any(r["event"] == "terminate" and r.get("reason") == "late_create" for r in e.audit()), what="迟到沙箱的审计")

    # 脱敏：控制台输出、审计日志、SandboxUnavailable 消息、ExecuteResponse.output 里都不能有令牌形状
    tokens = ("ak-ABCDEFGHIJKLMN1234", "as-0123456789ABCDEF", "supersecretvalue99", "ak-ZZZZZZZZZZZZ1234")
    with _Env() as e:
        E_ = e.fake.exception
        e.fake.create_default = Fail(E_.Error(f"boom {tokens[0]} and {tokens[1]} with MODAL_TOKEN_SECRET={tokens[2]} end"))
        console = io.StringIO()
        collected = []
        with contextlib.redirect_stdout(console), e.pool.lease(UID, "s1") as sb:
            collected.append(sb.execute("true").output)
            try:
                sb.run_python("print(1)")
            except SandboxUnavailable as ex:
                collected.append(str(ex))
            assert collected == [unavailable, _MSG_UNAVAILABLE], collected
            e.fake.create_default = Ok()
            e.fake.clock.advance(120)
            sb.execute(f"curl -H 'x: {tokens[3]}' http://example.invalid")
        blob = console.getvalue() + e.audit_text() + "".join(collected)
        for token in tokens:
            assert token not in blob, f"令牌形状泄露了: {token}"
        assert "***" in console.getvalue(), "运营者控制台里应该看到脱敏后的原因"
        assert any(r["event"] == "exec" and "***" in r["command"] for r in e.audit())

    # 给模型的文案精确等于约定的三句之一（另外两句在 D 里断言过）
    assert modal_sandbox._MSG_UNAVAILABLE == _MSG_UNAVAILABLE and modal_sandbox._MSG_TOTAL_BUSY == _MSG_TOTAL_BUSY
    assert modal_sandbox._msg_user_cap(2) == _MSG_USER_CAP.format(n=2)


# ══════════════════════════════════════════════════════════════════════════════
# L 审计日志
# ══════════════════════════════════════════════════════════════════════════════

def _expect_reject(pool, user, session):
    try:
        with pool.lease(user, session):
            raise AssertionError("应被拒绝")
    except SandboxUnavailable:
        pass


def test_l_audit_log():
    """L 惰性创建；一行一个 JSON、字段齐全；覆盖全部事件与 reason；没有明文 ID / 输出内容 / 环境变量 / 令牌形状；
    命令只记前 500 字符；5 MiB × 3 的轮转 handler、propagate=False；日志写不了时自动关闭、只提示一次、执行不受影响。"""
    quiet = io.StringIO()
    with _environ({"GEMINI_API_KEY": "gem-env-secret-zzz"}), \
            _Env(MODAL_MAX_SANDBOXES_PER_USER=2, MODAL_MAX_SANDBOXES_TOTAL=3) as e, contextlib.redirect_stdout(quiet):
        E_ = e.fake.exception
        audit_logger = logging.getLogger("gnk48.modal_audit")
        assert not os.path.exists(os.path.dirname(e.audit_path)) and not audit_logger.handlers, "第一次写之前不做任何 IO"

        e.fake.create_default = Fail(E_.NotFoundError("no image"))
        with e.pool.lease(UID, "sess-audit-1") as sb:
            for _ in range(4):
                sb.execute("true")                                     # 3 次 create_fail + 开路 + 第 4 次 reject:breaker
        assert os.path.exists(e.audit_path), "第一次写审计才创建目录与文件"
        e.fake.clock.advance(61)
        e.fake.create_default = Ok()
        e.fake.on_exec_all("print-secret", Out(stdout=b"TOP-SECRET-OUTPUT-xyz"))
        e.fake.on_exec_all("die-now", Raise(E_.SandboxTerminatedError("gone")))
        long_cmd = "echo " + "A" * 2000
        with e.pool.lease(UID, "sess-audit-1") as sb:
            sb.execute("print-secret")                                 # create_ok + breaker_close + exec
            sb.execute(long_cmd)
            sb.run_python("print('hello')")
            sb.upload_files([("/work/u.txt", b"hello")])
            sb.download_files(["/work/u.txt"])
        _touch(e.pool, UID, "sess-audit-2")
        _touch(e.pool, UID2, "sess-audit-1")
        with e.pool.lease(UID, "sess-audit-1"), e.pool.lease(UID, "sess-audit-2"):
            _expect_reject(e.pool, UID, "sess-audit-3")                                  # reject:cap_user
        with e.pool.lease(UID, "sess-audit-1"), e.pool.lease(UID, "sess-audit-2"), e.pool.lease(UID2, "sess-audit-1"):
            _expect_reject(e.pool, "member003", "sess-audit-x")                          # reject:cap_total
        _touch(e.pool, UID, "sess-audit-3")                                              # 每用户上限：驱逐最久空闲的 → terminate:evict
        with e.pool.lease(UID, "sess-audit-3") as sb:
            assert sb.interrupt() is True                                                # interrupt（soft）
            with _setattrs(config, MODAL_INTERRUPT_MODE="terminate"):
                assert sb.interrupt() is True                                            # interrupt（terminate）→ terminate:stop
        with e.pool.lease(UID2, "sess-audit-1") as sb:
            sb.execute("die-now")                                                        # exec 抛异常 → terminate:dead
        e.use_pool()
        with owner_context():
            modal_sandbox.get_routed_sandbox().execute("echo")                           # reject:context
        e.pool.reset_session(UID, "sess-audit-2")                                        # terminate:reset
        e.pool.reset_user(UID)                                                           # terminate:reset_user
        _touch(e.pool, UID2, "sess-audit-9")
        e.pool.shutdown(deadline_s=5)                                                    # terminate:shutdown

        records = e.audit()
        assert all({"ts", "event", "u", "s"} <= set(r) for r in records), records[:3]
        names = {r["event"] for r in records}
        need_events = {"create_ok", "create_fail", "exec", "run_python", "upload", "download", "interrupt", "terminate",
                       "reject", "breaker_open", "breaker_close"}
        assert need_events <= names, need_events - names
        pairs = {(r["event"], r.get("reason")) for r in records}
        for need in (("terminate", "reset"), ("terminate", "reset_user"), ("terminate", "evict"), ("terminate", "stop"),
                     ("terminate", "shutdown"), ("terminate", "dead"), ("reject", "cap_user"), ("reject", "cap_total"),
                     ("reject", "breaker"), ("reject", "context")):
            assert need in pairs, (need, sorted(p for p in pairs if p[1]))
        failed = [r for r in records if r["event"] == "create_fail"]
        assert failed and all(r.get("category") == "image_missing" and r.get("error") == "NotFoundError" for r in failed), failed
        assert any(r["event"] == "interrupt" and r.get("mode") == "terminate" for r in records)
        assert any(r["u"] == _sha(UID, 12) and r["s"] == _sha("sess-audit-1", 8) for r in records), "u / s 是哈希"

        text = e.audit_text()
        for leaked in (UID, UID2, "sess-audit", "TOP-SECRET-OUTPUT-xyz", "gem-env-secret-zzz"):
            assert leaked not in text, f"审计日志里不得出现 {leaked!r}"
        long_records = [r for r in records if r["event"] == "exec" and r["command"].startswith("echo AAAA")]
        assert long_records and all(len(r["command"]) <= 500 for r in long_records), "命令只记前 500 字符"

        assert audit_logger.propagate is False and len(audit_logger.handlers) == 1
        handler = audit_logger.handlers[0]
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == 5 * 1024 * 1024 and handler.backupCount == 3, (handler.maxBytes, handler.backupCount)

    # 日志写不了（把一个普通文件当目录用）：自动关闭、控制台只提示一次、代码执行不受影响
    with _Env() as e:
        blocker = os.path.join(e.tmp, "a-plain-file")
        with open(blocker, "w") as f:
            f.write("x")
        config.MODAL_AUDIT_LOG = os.path.join(blocker, "audit.log")
        modal_sandbox._reset_audit()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r1 = _touch(e.pool, UID, "s1", "echo one")
            r2 = _touch(e.pool, UID, "s2", "echo two")
            r3 = _touch(e.pool, UID, "s1", "echo three")
        assert r1.exit_code == r2.exit_code == r3.exit_code == 0
        assert out.getvalue().count("审计日志不可用") == 1, out.getvalue()
        # 重新注入之后按当时的路径重建
        config.MODAL_AUDIT_LOG = e.audit_path
        modal_sandbox.set_modal_module(e.fake)
        _touch(e.pool, UID, "s4", "echo four")
        assert any(r["event"] == "exec" for r in e.audit())


# ══════════════════════════════════════════════════════════════════════════════
# N 静态扫描（AST）
# ══════════════════════════════════════════════════════════════════════════════

_DOCKER_NAMES = ("DockerSandbox", "get_sandbox", "reset_sandbox", "interrupt_sandbox", "shutdown_sandbox", "set_docker_client")


def _imports_modal(node) -> bool:
    if isinstance(node, ast.Import):
        return any(a.name == "modal" or a.name.startswith("modal.") for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.level == 0 and (node.module == "modal" or (node.module or "").startswith("modal."))
    return False


def test_n_static_scan():
    """N modal_sandbox.py / modal_image.py 不得碰 Docker 沙箱；运行时不得出现 .build( / .publish( / debian_slim(；
    import modal 只允许在 load_modal() 里；不得有裸名 ConnectionError / TimeoutError 的 except；
    系统时钟只允许出现在 clock= 的默认值与审计的 ts；全仓 app/ 里读取 MODAL_TOKEN_* 的只有 config.py 且只转成布尔值。"""
    targets = [ROOT / "app" / "infra" / "modal_sandbox.py"]
    image_file = ROOT / "app" / "infra" / "modal_image.py"
    if image_file.exists():
        targets.append(image_file)

    for path in targets:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, str(path))
        for word in _DOCKER_NAMES:
            assert word not in text, f"{path.name} 不得出现 {word}"

        if path.name == "modal_sandbox.py":
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
                    assert name not in ("build", "publish", "debian_slim"), f"modal_sandbox.py 运行时不得调用 {name}()（镜像只在构建脚本里构建）"

            load_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_modal")
            allowed = {id(n) for n in ast.walk(load_def) if _imports_modal(n)}
            all_modal_imports = [n for n in ast.walk(tree) if _imports_modal(n)]
            assert len(all_modal_imports) == 1 and id(all_modal_imports[0]) in allowed, "import modal 只允许在 load_modal() 里出现一次"

            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is not None:
                    types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
                    for t in types:
                        assert not (isinstance(t, ast.Name) and t.id in ("ConnectionError", "TimeoutError")), (
                            f"modal_sandbox.py:{node.lineno} 不得对裸名 {t.id} 写 except（要么用 modal.exception 里的，要么用 Exception）")

            allowed_clock = set()
            for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ModalSandboxPool"):
                init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
                for default in [*init.args.defaults, *init.args.kw_defaults]:
                    if default is not None:
                        allowed_clock |= {id(n) for n in ast.walk(default)}
            audit_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_audit")
            allowed_clock |= {id(n) for n in ast.walk(audit_def)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "time" \
                        and node.attr in ("monotonic", "time", "sleep", "perf_counter"):
                    assert id(node) in allowed_clock, f"modal_sandbox.py:{node.lineno} 不得直接读系统时钟 / sleep: time.{node.attr}"
        else:
            assert not any(_imports_modal(n) for n in ast.walk(tree)), "modal_image.py 不得导入 modal（只是纯定义）"

    # 全仓 app/ 里读取令牌环境变量的只有 config.py，并且只转成布尔值
    names = {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}
    for path in (ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            const = None
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.func, ast.Attribute) and node.func.attr in ("getenv", "get", "pop", "setdefault"):
                const = node.args[0].value
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                const = node.slice.value
            if const not in names:
                continue
            assert path.name == "config.py", f"{path.relative_to(ROOT)}:{node.lineno} 只有 config.py 能读取 {const}"
            outer = parents.get(id(node))
            while outer is not None and isinstance(outer, ast.Call) is False:
                outer = parents.get(id(outer))
            assert isinstance(outer, ast.Call) and getattr(outer.func, "id", "") == "bool", f"config.py:{node.lineno} 令牌只能转成布尔值"


# ══════════════════════════════════════════════════════════════════════════════
# K 路由与 BaseSandbox 契约
# ══════════════════════════════════════════════════════════════════════════════

def test_k_routed_sandbox():
    """K 没有上下文 / owner / 非法 user_id / 空 session_id 一律拒绝（审计 reject:context、不碰池）；member 落到对应的 (user, session)；
    两个 member 并发互不串；异步路径与线程池路径都拿得到上下文，没带上下文的线程 fail-closed；BaseSandbox 契约。"""
    from deepagents.backends.protocol import SandboxBackendProtocol

    quiet = io.StringIO()
    with _Env() as e, contextlib.redirect_stdout(quiet):
        e.use_pool()
        routed = modal_sandbox.get_routed_sandbox()
        assert routed is modal_sandbox.get_routed_sandbox() and routed.id == "modal-routed"
        assert isinstance(routed, SandboxBackendProtocol)
        assert not vars(routed) and UID not in repr(vars(routed)), "实例上不缓存任何用户 / 会话状态"

        contexts = (
            ("没有上下文", contextlib.nullcontext(), "无法确定当前用户"),
            ("owner", owner_context(), "云端沙箱仅供成员使用"),
            ("非法 user_id", member_context("bad user!", "s"), "无法确定当前用户"),
            ("空 session_id", member_context(UID, ""), "无法确定当前用户"),
        )
        for label, ctx, expected in contexts:
            with ctx:
                r = routed.execute("echo hi")
                assert r.exit_code == 1 and "[云端沙箱不可用]" in r.output and expected in r.output, (label, r)
                try:
                    routed.run_python("print(1)")
                    raise AssertionError(f"{label}：run_python 应抛 SandboxUnavailable")
                except SandboxUnavailable as ex:
                    assert expected in str(ex), (label, str(ex))
                up = routed.upload_files([("/work/a.txt", b"x")])
                assert up[0].error and expected in up[0].error, (label, up)
                down = routed.download_files(["/work/a.txt"])
                assert down[0].error and expected in down[0].error and down[0].content is None, (label, down)
                assert routed.collect_new_outputs().images == [] and routed.interrupt() is False
        assert e.fake.created == [] and e.pool.snapshot() == [], "拒绝时不碰池"
        assert sum(1 for r in e.audit() if r["event"] == "reject" and r.get("reason") == "context") >= 4 * 5

        # member：落到各自的 (user, session)，两个 member 并发互不串
        def work(uid, sid, name):
            with member_context(uid, sid):
                routed.upload_files([(f"/work/{name}.txt", name.encode())])
                return routed.download_files([f"/work/{name}.txt"])[0].content

        jobs = [_Job(work, UID, "s1", "alice"), _Job(work, UID2, "s1", "bob")]
        assert [j.value() for j in jobs] == [b"alice", b"bob"]
        by_user = {sb.tags["u"]: sb for sb in e.fake.created}
        assert set(by_user) == {_sha(UID, 12), _sha(UID2, 12)}
        assert by_user[_sha(UID, 12)].filesystem.has("/work/alice.txt") and not by_user[_sha(UID, 12)].filesystem.has("/work/bob.txt")
        assert by_user[_sha(UID2, 12)].filesystem.has("/work/bob.txt") and not by_user[_sha(UID2, 12)].filesystem.has("/work/alice.txt")

        # ① 异步路径：BaseSandbox.aexecute 走 asyncio.to_thread
        async def via_async():
            with member_context(UID, "s-async"):
                return await routed.aexecute("echo from-async")

        assert asyncio.run(via_async()).exit_code == 0
        assert any(sb.tags["s"] == _sha("s-async", 8) for sb in e.fake.created)
        # ② 同步路径：LangChain 线程池里 copy_context().run
        with member_context(UID, "s-sync"), concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            assert ex.submit(contextvars.copy_context().run, routed.execute, "echo from-sync").result(15).exit_code == 0
        assert any(sb.tags["s"] == _sha("s-sync", 8) for sb in e.fake.created)
        # ③ 反例：线程里没有带上下文 → fail-closed 拒绝
        before = len(e.fake.created)
        with member_context(UID, "s-bare"):
            bare = _Job(routed.execute, "echo nope").value()
        assert bare.exit_code == 1 and "无法确定当前用户" in bare.output and len(e.fake.created) == before

        # BaseSandbox 契约：write() 先预检再上传；上传下载经 FakeModal 文件系统往返一致
        with member_context(UID, "s-contract"):
            res = routed.write("/work/c.txt", "你好")
            assert res.error is None and res.path == "/work/c.txt", res
            target = next(sb for sb in e.fake.created if sb.tags["s"] == _sha("s-contract", 8))
            assert target.filesystem.get("/work/c.txt") == "你好".encode("utf-8")
            assert target.calls(lambda inner: tuple(inner[:2]) == ("bash", "-c") and "python3" in inner[2]), "write() 之前应先做 python3 预检"
            payload = bytes(range(256))
            assert routed.upload_files([("/work/bin.dat", payload)])[0].error is None
            assert routed.download_files(["/work/bin.dat"])[0].content == payload

        # 沙箱创建失败：execute 返回通用短句，run_python 抛 SandboxUnavailable，都不抛别的异常
        e.fake.create_default = Fail(e.fake.exception.NotFoundError("no image"))
        with member_context(UID2, "s-fail"):
            assert routed.execute("echo").output == "[云端沙箱不可用] " + _MSG_UNAVAILABLE
            try:
                routed.run_python("print(1)")
                raise AssertionError("创建失败时 run_python 应抛 SandboxUnavailable")
            except SandboxUnavailable as ex:
                assert str(ex) == _MSG_UNAVAILABLE


# ══════════════════════════════════════════════════════════════════════════════
# M 镜像定义与脚本
# ══════════════════════════════════════════════════════════════════════════════

def _load_script(name: str):
    """把 scripts/<name>.py 当模块加载（不执行 __main__），好直接调用它的 main(argv)。"""
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_main(mod, argv, inputs=None):
    """调用脚本的 main(argv)：捕获输出；input() 依次返回 inputs，用完（或没给）就抛 EOFError。"""
    out = io.StringIO()
    answers = iter(inputs or [])

    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError

    real_input = builtins.input
    builtins.input = fake_input
    try:
        with contextlib.redirect_stdout(out):
            code = mod.main(argv)
    finally:
        builtins.input = real_input
    return code, out.getvalue()


def _raiser(exc: BaseException):
    def fn(sandbox, inner):
        raise exc
    return fn


def _install_good_image_rules(fake) -> None:
    """让验收沙箱里的每条命令都给出“镜像正常”的输出。"""
    from app.infra import modal_image
    freeze = "\n".join([*modal_image.read_pinned_requirements(), "httpx==0.28.1", "Pillow==11.0.0"]) + "\n"

    def selftest(sandbox, inner):
        sandbox.filesystem.put("/work/outputs/selftest.png", _png(16, 16))
        return Out()

    for matcher, behavior in (
        (("python", "--version"), Out(stdout=b"Python 3.13.1\n")),
        (("python3", "--version"), Out(stdout=b"Python 3.13.1\n")),
        (("id", "-u"), Out(stdout=b"0\n")),
        (("pwd",), Out(stdout=b"/work\n")),
        ("echo $HOME", Out(stdout=b"/work\n")),
        ("echo $MPLBACKEND", Out(stdout=b"Agg /opt/mplrc /opt/mplcache\n")),
        (("pip", "freeze"), Out(stdout=freeze.encode())),
        ("urlopen", Out(stdout=b"200\n")),
        ("selftest.png", Call(selftest)),
    ):
        fake.on_exec_all(matcher, behavior)


def test_m_image_definition_and_scripts():
    """M modal_image 的常量与操作链；build_modal_image / manage_modal_sandboxes 的 main(argv)（FakeModal 注入）；--help 离线。"""
    import base64
    import re as _re
    from app.infra import modal_image

    # —— 常量 ——
    assert modal_image.IMAGE_PYTHON_VERSION == "3.13"
    assert modal_image.APT_PACKAGES == ("fonts-noto-cjk", "ripgrep", "jq", "unzip", "zip", "file", "tree", "procps", "git", "curl",
                                        "wget", "ca-certificates", "build-essential", "poppler-utils", "sqlite3", "graphviz")
    assert modal_image.APT_OPTIONAL == ("ffmpeg", "nodejs", "npm")
    assert modal_image.PIP_EXTRA == ("httpx", "requests", "beautifulsoup4", "lxml", "pillow", "pypdf", "pdfplumber",
                                     "python-docx", "python-pptx", "xlsxwriter", "networkx", "statsmodels")
    assert modal_image.IMAGE_ENV == {
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8",
        "MPLBACKEND": "Agg", "MATPLOTLIBRC": "/opt/mplrc", "MPLCONFIGDIR": "/opt/mplcache", "HOME": "/work",
    }
    assert "Glyph" in modal_image.FONT_SELFCHECK_PY and "findfont" in modal_image.FONT_SELFCHECK_PY
    assert 'savefig("/tmp/selftest.png")' in modal_image.FONT_SELFCHECK_PY
    assert 'savefig("/work/outputs/selftest.png")' in modal_image.font_selfcheck_py("/work/outputs/selftest.png")

    # —— read_pinned_requirements：忽略空行与注释 ——
    tmp = Path(tempfile.mkdtemp(prefix="modal-req-"))
    try:
        (tmp / "deploy" / "sandbox").mkdir(parents=True)
        (tmp / "deploy" / "sandbox" / "requirements.txt").write_text("# 注释\n\npandas==2.3.3\n   \n  numpy==2.2.6  \n# end\n", encoding="utf-8")
        with _setattrs(config, PROJECT_ROOT=str(tmp)):
            assert modal_image.read_pinned_requirements() == ["pandas==2.3.3", "numpy==2.2.6"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    real_lines = [ln.strip() for ln in (ROOT / "deploy" / "sandbox" / "requirements.txt").read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    assert modal_image.read_pinned_requirements() == real_lines and real_lines, "Docker 与 Modal 两个镜像共用同一份固定版本"

    # —— build_image 的操作链：顺序与内容 ——
    with _Env() as e:
        img = modal_image.build_image(e.fake)
        assert img.op_names() == ["debian_slim", "apt_install", "pip_install", "add_local_file", "env", "run_commands", "workdir"], img.op_names()
        ops = {op: (args, kwargs) for op, args, kwargs in img.ops}
        assert ops["debian_slim"] == ((), {"python_version": "3.13", "force_build": False})
        assert ops["apt_install"][0] == modal_image.APT_PACKAGES
        assert ops["pip_install"][0] == (*real_lines, *modal_image.PIP_EXTRA), "固定版本与额外库在同一层，让 pip 一次解析"
        local, remote = ops["add_local_file"][0]
        assert local.replace("\\", "/").endswith("deploy/sandbox/matplotlibrc") and remote == "/opt/mplrc/matplotlibrc"
        assert ops["add_local_file"][1] == {"copy": True}, "必须 copy=True，否则文件不会被烘进发布的镜像"
        assert ops["env"][0] == (modal_image.IMAGE_ENV,) and ops["workdir"][0] == ("/work",)
        commands = ops["run_commands"][0]
        assert commands[0].startswith("mkdir -p /work/outputs /opt/mplrc /opt/mplcache")
        assert any("ln -s" in c and "python3" in c for c in commands), "防御性保证 python3 存在"
        assert any("MPLCONFIGDIR=/opt/mplcache python -c 'import matplotlib.pyplot'" in c for c in commands)
        piped = next(c for c in commands if "base64 -d" in c)
        assert base64.b64decode(_re.search(r"echo (\S+) \|", piped).group(1)).decode("utf-8") == modal_image.FONT_SELFCHECK_PY
        assert not any(op in ("user", "run_as") for op in img.op_names()) and not any("useradd" in c or "USER" in c for c in commands)
        assert modal_image.build_image(e.fake, with_optional=True).ops[1][1] == (*modal_image.APT_PACKAGES, *modal_image.APT_OPTIONAL)
        assert modal_image.build_image(e.fake, force=True).ops[0][2]["force_build"] is True

    # —— scripts/build_modal_image.py ——
    build = _load_script("build_modal_image")
    offline = _offline_scripts_run()                                              # --help 完全离线：modal 不会被导入
    for script in ("build_modal_image", "manage_modal_sandboxes"):
        r = offline[f"scripts/{script}.py --help"]
        assert r["code"] == 0, (script, r)
        assert "usage" in r["stdout"].lower() and ("--check-only" in r["stdout"] or "kill" in r["stdout"]), r["stdout"]
    assert offline["modal_loaded"] is False, "这几个脚本的 --help / 不带 --live 不应导入 modal"

    with _Env(MODAL_TOKENS_PRESENT=False) as e:                                   # 没有令牌 → 退出码 2
        code, out = _run_main(build, ["--yes"])
        assert code == 2 and "MODAL_TOKEN_ID" in out and e.fake.build_calls == [] and e.fake.created == []
    with _Env(MODAL_TOKENS_PRESENT=True) as e:                                    # modal 导入失败 → 退出码 2
        modal_sandbox.set_modal_module(None)
        with _setattrs(modal_sandbox, load_modal=_raiser(ImportError("No module named modal"))):
            code, out = _run_main(build, ["--yes"])
        assert code == 2 and "modal" in out

    with _Env(MODAL_TOKENS_PRESENT=True) as e:                                    # 正常流程
        e.fake.published.clear()
        _install_good_image_rules(e.fake)
        code, out = _run_main(build, ["--yes"])
        assert code == 0, out
        assert e.fake.lookup_calls[0] == {"name": config.MODAL_APP_NAME, "create_if_missing": True}
        assert len(e.fake.build_calls) == 1 and e.fake.build_calls[0].built_in_output, "build 要在 enable_output() 里"
        assert e.fake.publish_calls == [config.MODAL_IMAGE_NAME] and config.MODAL_IMAGE_NAME in e.fake.published
        assert "镜像已发布并通过验收" in out and "MODAL_IMAGE_NAME" in out and "✅" in out and "❌" not in out, out
        assert e.fake.live() == [], "验收用的临时沙箱必须被终止"
        sandbox = e.fake.created[0]
        assert sandbox.kwargs["timeout"] == 600 and sandbox.kwargs["idle_timeout"] == 120 and sandbox.kwargs["tags"]["role"] == "image-check"
        assert set(sandbox.kwargs) == {"app", "image", "env", "timeout", "idle_timeout", "workdir", "cpu", "memory", "block_network", "tags"}
        assert sandbox.kwargs["env"] == {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8", "HOME": "/work"}

    with _Env(MODAL_TOKENS_PRESENT=True) as e:                                    # 没有 --yes 要交互确认；--name、--with-optional、--force 生效
        e.fake.published.clear()
        _install_good_image_rules(e.fake)
        code, out = _run_main(build, [], ["no"])
        assert code == 1 and "已取消" in out and e.fake.build_calls == [] and e.fake.lookup_calls == []
        code, out = _run_main(build, [])                                          # EOF 也按取消处理
        assert code == 1 and e.fake.build_calls == []
        code, out = _run_main(build, ["--name", "custom:v7", "--with-optional", "--force"], ["yes"])
        assert code == 0 and e.fake.publish_calls == ["custom:v7"], (code, out)
        built = e.fake.build_calls[0]
        assert built.ops[0][2]["force_build"] is True and built.ops[1][1][-3:] == modal_image.APT_OPTIONAL

    with _Env(MODAL_TOKENS_PRESENT=True) as e:                                    # 构建失败 → 退出码 1，消息已脱敏，不 publish
        _install_good_image_rules(e.fake)

        def broken_build(modal, **kwargs):
            image = e.fake.Image.debian_slim()
            image.build = _raiser(RuntimeError(f"build exploded {_LEAKY} MODAL_TOKEN_SECRET=hunter2hunter2"))
            return image

        with _setattrs(modal_image, build_image=broken_build):
            code, out = _run_main(build, ["--yes"])
        assert code == 1 and "构建失败" in out and _LEAKY not in out and "hunter2" not in out and e.fake.publish_calls == [], out

    with _Env(MODAL_TOKENS_PRESENT=True) as e:                                    # --check-only：不 build 也不 publish
        _install_good_image_rules(e.fake)
        code, out = _run_main(build, ["--check-only"])
        assert code == 0 and e.fake.build_calls == [] and e.fake.publish_calls == [] and e.fake.live() == [], out
        e.fake.published.clear()                                                  # 镜像不存在 → 明确文案 + 退出码 1
        code, out = _run_main(build, ["--check-only"])
        assert code == 1 and "镜像不存在" in out and "去掉 --check-only" in out and e.fake.build_calls == [], out

    failing = (                                                                    # 验收每一项失败都 → 退出码 1
        (("python", "--version"), Out(stdout=b"Python 3.12.0\n")),
        (("id", "-u"), Out(stdout=b"1000\n")),
        (("pwd",), Out(stdout=b"/\n")),
        ("echo $HOME", Out(stdout=b"/root\n")),
        ("touch /work/outputs", Out(exit=1)),
        ("import pandas", Out(exit=1, stderr=b"ModuleNotFoundError: pandas")),
        (("pip", "freeze"), Out(stdout=b"pandas==0.0.1\n")),
        ("command -v pdftotext", Out(exit=1)),
        (("find", "/work", "-maxdepth"), Out(exit=1)),
        ("echo $MPLBACKEND", Out(stdout=b"TkAgg\n")),
        ("selftest.png", Out(exit=1, stderr=b"Glyph missing")),
        ("selftest.png", Call(lambda sb, inner: (sb.filesystem.put("/work/outputs/selftest.png", b"not a png"), Out())[1])),
    )
    for matcher, behavior in failing:
        with _Env(MODAL_TOKENS_PRESENT=True) as e:
            _install_good_image_rules(e.fake)
            e.fake.on_exec_all(matcher, behavior)
            code, out = _run_main(build, ["--check-only"])
            assert code == 1 and "❌" in out, (matcher, out)
            assert e.fake.live() == [], "验收失败时临时沙箱也要终止"
    with _Env(MODAL_TOKENS_PRESENT=True) as e:                                    # 公网检查失败只算警告
        _install_good_image_rules(e.fake)
        e.fake.on_exec_all("urlopen", Out(exit=1, stderr=b"URLError"))
        code, out = _run_main(build, ["--check-only"])
        assert code == 0 and "⚠️" in out, out
    with _Env(MODAL_TOKENS_PRESENT=True, MODAL_BLOCK_NETWORK=True) as e:
        _install_good_image_rules(e.fake)
        code, out = _run_main(build, ["--check-only"])
        assert code == 0 and "跳过公网检查" in out and not any("urlopen" in " ".join(c.inner) for c in e.fake.created[0].exec_calls)
    with _Env(MODAL_TOKENS_PRESENT=True) as e:                                    # 验收中途抛异常 / Ctrl-C：临时沙箱都被终止
        _install_good_image_rules(e.fake)
        e.fake.on_exec_all(("id", "-u"), Call(_raiser(RuntimeError(f"boom {_LEAKY}"))))
        code, out = _run_main(build, ["--check-only"])
        assert code == 1 and _LEAKY not in out and e.fake.live() == [], out
    with _Env(MODAL_TOKENS_PRESENT=True) as e:
        _install_good_image_rules(e.fake)
        e.fake.on_exec_all(("id", "-u"), Call(_raiser(KeyboardInterrupt())))
        code, out = _run_main(build, ["--check-only"])
        assert code == 130 and e.fake.live() == [] and e.fake.created[0].terminate_calls >= 1, out

    # —— scripts/manage_modal_sandboxes.py ——
    manage = _load_script("manage_modal_sandboxes")
    with _Env(MODAL_TOKENS_PRESENT=False):
        code, out = _run_main(manage, ["list"])
        assert code == 2 and "MODAL_TOKEN_ID" in out
    with _environ({"MODAL_TOKEN_ID": _LEAKY, "MODAL_TOKEN_SECRET": "as-0123456789ABCDEF"}), _Env(MODAL_TOKENS_PRESENT=True) as e:
        code, out = _run_main(manage, ["list"])
        assert code == 0 and "没有" in out, "App 还不存在：当作没有沙箱"

        app = e.fake.App.lookup(config.MODAL_APP_NAME, create_if_missing=True)

        def make(tags):
            return e.fake.Sandbox.create("sleep", "infinity", app=app, image=e.fake.Image.from_name(config.MODAL_IMAGE_NAME), tags=tags)

        mine_a1 = make({"svc": "gnk48", "role": "member-code", "u": "aaaa11112222", "s": "sess0001"})
        mine_a2 = make({"svc": "gnk48", "role": "member-code", "u": "aaaa11112222", "s": "sess0002"})
        mine_b = make({"svc": "gnk48", "role": "member-code", "u": "bbbb33334444", "s": "sess0003"})
        other = make({"svc": "someone-else"})
        code, out = _run_main(manage, ["list"])
        assert code == 0 and "共 3 个沙箱" in out, out
        for sb in (mine_a1, mine_a2, mine_b):
            assert sb.object_id in out
        assert other.object_id not in out and "aaaa11112222" in out and _LEAKY not in out and "as-0123456789ABCDEF" not in out

        code, out = _run_main(manage, ["kill", "--user", "zzzz"], ["yes"])             # 前缀不匹配：不杀任何东西
        assert code == 0 and "没有匹配" in out and len(e.fake.live()) == 4
        code, out = _run_main(manage, ["kill", "--all"], ["no"])                       # 没有 --yes：要确认
        assert code == 1 and "已取消" in out and len(e.fake.live()) == 4
        code, out = _run_main(manage, ["kill", "--all"])
        assert code == 1 and len(e.fake.live()) == 4, "EOF 按取消处理"
        code, out = _run_main(manage, ["kill", "--user", "aaaa"], ["yes"])
        assert code == 0 and not mine_a1.alive and not mine_a2.alive and mine_b.alive and other.alive, out
        code, out = _run_main(manage, ["kill", "--all", "--yes"])
        assert code == 0 and not mine_b.alive and other.alive, "只碰带 svc=gnk48 标签的"
        code, out = _run_main(manage, ["kill", "--user", "  "], ["yes"])
        assert code == 1, "空前缀不能当作“全部”"
        assert _LEAKY not in out


# ══════════════════════════════════════════════════════════════════════════════
# O member 的 execute_python_code 工具
# ══════════════════════════════════════════════════════════════════════════════

_DESC_FORBIDDEN_WORDS = ("Docker", "/workspace", "workspace", "宿主机", "FileAgent", "DesktopAgent", "BrowserAgent")


def test_o_member_execute_python_tool():
    """O 在 member 上下文里返回状态行 + Stdout；有图时返回多模态内容并进 outbox、同一文件不重发；owner / 无上下文拒绝且不建沙箱；
    创建失败返回通用短句；描述是 member 版；owner 的工具对象与描述不受影响。"""
    from app.runtime import outbox
    from app.tools import code_tools

    quiet = io.StringIO()
    with _Env() as e, contextlib.redirect_stdout(quiet):
        e.use_pool()
        tools = code_tools.build_member_tools()
        assert [t.name for t in tools] == ["execute_python_code"] and tools[0] is not code_tools.execute_python_code
        tool = tools[0]
        assert tool.args_schema.model_json_schema()["properties"].keys() == {"code"}
        for word in _DESC_FORBIDDEN_WORDS:
            assert word not in tool.description, f"member 版描述不应出现 {word!r}"
        assert str(config.MODAL_EXEC_MAX_TIMEOUT_SECONDS) in tool.description and "云端沙箱" in tool.description

        e.fake.on_exec_all(("python", "-u"), Out(stdout=b"hello from sandbox\n"))
        with member_context(UID, "s-tool"):
            outbox.begin()
            text = tool.invoke({"code": "print('x')"})
            assert isinstance(text, str) and "执行完成 | 退出码: 0" in text and "Stdout:\nhello from sandbox" in text, text
            e.fake.created[0].filesystem.put("/work/outputs/chart.png", _png(32, 32))
            multi = tool.invoke({"code": "print(2)"})
            assert isinstance(multi, list) and multi[0]["type"] == "text" and "已向用户发送 1 张图片（chart.png）" in multi[0]["text"], multi
            assert multi[1]["type"] == "image_url" and multi[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            assert len(outbox.current()) == 1 and outbox.current()[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            again = tool.invoke({"code": "print(3)"})
            assert isinstance(again, str) and len(outbox.current()) == 1, "同一文件同一版本只发一次"
            outbox.begin()

        for label, ctx in (("owner", owner_context()), ("无上下文", contextlib.nullcontext())):
            before = (len(e.fake.created), len(e.pool.snapshot()))
            with ctx:
                denied = tool.invoke({"code": "print(1)"})
            assert isinstance(denied, str) and "代码执行沙箱不可用" in denied, (label, denied)
            assert (len(e.fake.created), len(e.pool.snapshot())) == before, f"{label}：不得创建任何沙箱"

        e.fake.create_default = Fail(e.fake.exception.NotFoundError("no image"))
        with member_context(UID2, "s-fail"):
            failed = tool.invoke({"code": "print(1)"})
        assert isinstance(failed, str) and _MSG_UNAVAILABLE in failed and "NotFound" not in failed and "镜像" not in failed, failed

    assert [t.name for t in code_tools.TOOLS] == ["execute_python_code"] and "Docker" in code_tools.execute_python_code.description, "owner 的工具不受影响"


# ══════════════════════════════════════════════════════════════════════════════
# P 提示词与描述的事实核对
# ══════════════════════════════════════════════════════════════════════════════

_POSITIVE_NET_FRAGMENTS = ("沙箱可以联网", "有公网", "缺少的库可以", "pip 安装缺少的库", "下载公开数据", "从公开网址下载",
                           "requests / httpx / curl", "curl/pip")


def test_p_prompt_and_description_facts():
    """P 描述 / 提示词里的数字与 config 一致、联网句子随 MODAL_BLOCK_NETWORK 切换（固定短语）、红线都在、禁用词一个都没有。"""
    from app.agents import prompts
    from app.agents.experts import code_agent
    from app.agents.prompts import WORKER_BASE_PROMPT
    from app.tools import code_tools

    builders = {
        "工具描述": code_tools.member_execute_python_description,
        "专家描述": code_agent.build_member_description,
        "专家提示词": code_agent.build_member_prompt,
        "主管提示词": lambda: prompts.build_member_main_prompt(True),
    }

    for blocked in (False, True):
        with _setattrs(config, MODAL_BLOCK_NETWORK=blocked):
            for label, build in builders.items():
                text = build()
                if not blocked:
                    assert text.count("沙箱可以联网") == 1 and "沙箱不能联网" not in text, (label, "联网版")
                else:
                    assert text.count("沙箱不能联网") == 1, (label, "断网版")
                    for frag in _POSITIVE_NET_FRAGMENTS:
                        assert frag not in text, (label, "断网版不得含肯定句片段", frag)
                if label != "主管提示词":
                    for word in _DESC_FORBIDDEN_WORDS:
                        assert word not in text, (label, word)
            plain = prompts.build_member_main_prompt(False)
            assert "沙箱可以联网" not in plain and "沙箱不能联网" not in plain, "禁用 CodeAgent 的变体与联网配置无关"
    assert code_agent.MEMBER_PROMPT == code_agent.build_member_prompt() and code_agent.MEMBER_DESCRIPTION == code_agent.build_member_description()
    assert prompts.MEMBER_MAIN_AGENT_PROMPT_CODE == prompts.build_member_main_prompt(True)

    # 数字随 config 现算
    with _setattrs(config, MODAL_EXEC_TIMEOUT_SECONDS=45, MODAL_EXEC_MAX_TIMEOUT_SECONDS=200, MODAL_IDLE_TIMEOUT_SECONDS=1500,
                   MODAL_MEMORY_MB_LIMIT=768, SANDBOX_MAX_IMAGES=3, SANDBOX_MAX_IMAGE_MB=7):
        tool_desc = code_tools.member_execute_python_description()
        for needle in ("单次执行默认 45 秒、上限 200 秒", "内存上限约 0.75 GiB", "单次最多导出 3 张图片（单张 ≤ 7MB", "闲置约 25 分钟"):
            assert needle in tool_desc, (needle, tool_desc)
        prompt = code_agent.build_member_prompt()
        for needle in ("单次执行默认 45 秒、最长 200 秒", "内存上限约 0.75 GiB", "单次最多导出 3 张图片", "闲置约 25 分钟",
                       "默认 45 秒，可用 timeout 参数至多 200 秒"):
            assert needle in prompt, (needle, prompt)
        assert "闲置约 25 分钟" in code_agent.build_member_description()
    defaults = code_tools.member_execute_python_description()
    assert "默认 120 秒、上限 300 秒" in defaults and "约 2 GiB" in defaults and "闲置约 10 分钟" in defaults, defaults

    prompt = code_agent.build_member_prompt()
    assert prompt.startswith(WORKER_BASE_PROMPT) and prompt.count("你是 CodeAgent") == 1
    for needle in ("挖矿", "端口扫描", "不在允许范围内", "严禁探测沙箱之外", "盲目递归搜索", "后台挂起长时间进程",
                   "/skills/matplotlib-chart/SKILL.md", "/skills/data-analysis/SKILL.md", "【内部专家汇报 - 来源：CodeAgent】",
                   "/skills/ 是只读技能目录", "你看不到用户电脑上的任何文件"):
        assert needle in prompt, f"提示词缺少: {needle}"
    assert "访问不到用户电脑上的任何文件" in code_agent.build_member_description()


# ══════════════════════════════════════════════════════════════════════════════
# K2 驱动真实编译的 member CodeAgent 做越权尝试
# ══════════════════════════════════════════════════════════════════════════════

def _dir_hashes(*dirs) -> dict:
    result = {}
    for d in dirs:
        for path in sorted(Path(d).rglob("*")):
            if path.is_file():
                result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _scripted_model(calls: list):
    """按脚本依次发出一次工具调用，最后一条给出结论。bind_tools 返回自己。"""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Scripted(BaseChatModel):
        script: list
        idx: int = 0

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            msg = self.script[min(self.idx, len(self.script) - 1)]
            self.idx += 1
            return ChatResult(generations=[ChatGeneration(message=msg)])

    script = [AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{i}"}]) for i, (name, args) in enumerate(calls)]
    script.append(AIMessage(content="【内部专家汇报 - 来源：CodeAgent】完成"))
    return Scripted(script=script)


def test_k2_member_code_agent_cannot_escape():
    """K2 真实编译的 member CodeAgent（脚本化模型）在 member 上下文里依次尝试：写 / 改 /skills/ 下的文件、读 /skills/../.env、
    列 /skills/..、调用一个叫 delete 的工具、触发大结果转存。断言：写 / 改都返回只读错误；skills/ 与 skills_member/ 的文件哈希前后一致；
    越界读取拿不到内容；delete 工具不存在；转存没有写进宿主机磁盘或 FakeModal 的文件系统；工具名集合恰为 8 个。"""
    from deepagents.backends import StateBackend
    from langchain_core.messages import HumanMessage, ToolMessage
    from app.agents.experts import code_agent
    from app.agents.skills import SKILLS_DIR, SKILLS_MEMBER_DIR

    before = _dir_hashes(SKILLS_DIR, SKILLS_MEMBER_DIR)
    assert before, "skills/ 与 skills_member/ 都应该有文件"
    skill_path = "/skills/matplotlib-chart/SKILL.md"
    attack = [
        ("write_file", {"file_path": skill_path, "content": "pwned"}),
        ("edit_file", {"file_path": skill_path, "old_string": "matplotlib", "new_string": "pwned"}),
        ("write_file", {"file_path": "/skills/new-skill/SKILL.md", "content": "pwned"}),
        ("read_file", {"file_path": "/skills/../.env"}),
        ("read_file", {"file_path": "/skills/a/../../README.md"}),
        ("ls", {"path": "/skills/.."}),
        ("delete", {"file_path": skill_path}),
    ]
    quiet = io.StringIO()
    with _Env() as e, contextlib.redirect_stdout(quiet):
        e.use_pool()
        llm = _scripted_model(attack)
        build = code_agent.build_member_code_agent_runnable(llm=llm, system_prompt=code_agent.MEMBER_PROMPT)
        tools_node = build.runnable.nodes["tools"]
        tool_dict = getattr(tools_node.triggers, "__self__", None) or tools_node.bound
        names = set(tool_dict._tools_by_name.keys())
        assert names == build.tool_names == {"execute_python_code", "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}, names
        assert "delete" not in names

        with member_context(UID, "s-k2"):
            result = build.runnable.invoke({"messages": [HumanMessage(content="go")]}, {"recursion_limit": 60})
        replies = {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}
        text = {cid: (m.content if isinstance(m.content, str) else str(m.content)) for cid, m in replies.items()}
        assert len(text) == len(attack), (len(text), sorted(text))
        assert "只读" in text["call-0"], text["call-0"]                       # write_file → 只读错误
        assert "只读" in text["call-1"], text["call-1"]                       # edit_file → 只读错误
        assert "只读" in text["call-2"], text["call-2"]                       # 新建技能文件也不行
        for cid in ("call-3", "call-4", "call-5"):
            leaked = [w for w in ("MODAL_TOKEN", "API_KEY", "MYSQL_PASSWORD", "pyproject.toml", "main.py") if w in text[cid]]
            assert not leaked, (cid, leaked, text[cid][:200])               # 越界读 / 列拿不到 skills_member/ 之外的东西
        assert replies["call-6"].status == "error" or "delete" in text["call-6"], text["call-6"]       # delete 工具不存在
        assert "not a valid tool" in text["call-6"] or "不存在" in text["call-6"], text["call-6"]
        assert _dir_hashes(SKILLS_DIR, SKILLS_MEMBER_DIR) == before, "skills/ 与 skills_member/ 的文件被改动了"
        assert not (Path(SKILLS_MEMBER_DIR) / "new-skill").exists() and not (Path(SKILLS_DIR) / "new-skill").exists()
        assert all(not p.startswith("/skills") for p in e.fake.created[0].filesystem.paths()) if e.fake.created else True

        # 大结果转存：只允许写进 StateBackend，宿主机磁盘与 FakeModal 的文件系统里都不能有它
        e.fake.on_exec_all("flood-output", Out(stdout=b"x" * 400_000))
        flood = _scripted_model([("execute", {"command": "flood-output"})])
        build2 = code_agent.build_member_code_agent_runnable(llm=flood, system_prompt=code_agent.MEMBER_PROMPT)
        with member_context(UID, "s-k2-flood"):
            result2 = build2.runnable.invoke({"messages": [HumanMessage(content="go")]}, {"recursion_limit": 60})
        flood_msg = next(m for m in result2["messages"] if isinstance(m, ToolMessage))
        evicted = len(str(flood_msg.content)) < 300_000
        if not evicted:
            print("    [INFO] 400K 字符的输出没有触发大结果转存（阈值更大），只断言了路由配置")
        for sb in e.fake.created:
            leaked_paths = [p for p in sb.filesystem.paths() if "large_tool_results" in p or p.startswith("/_artifacts")]
            assert not leaked_paths, f"转存不得写进沙箱文件系统: {leaked_paths}"
        for name in ("large_tool_results", "_artifacts"):
            assert not (ROOT / name).exists() and not (ROOT / "skills_member" / name).exists(), name
        assert isinstance(build2.backend.routes["/_artifacts/"], StateBackend)
        assert build2.backend.artifacts_root == "/_artifacts"


# ══════════════════════════════════════════════════════════════════════════════
# Q 真机验收脚本：没有 --live 完全离线
# ══════════════════════════════════════════════════════════════════════════════

_OFFLINE_RUN = None

_OFFLINE_DRIVER = """
import contextlib, io, json, runpy, socket, sys

def no_network(*args, **kwargs):
    raise AssertionError('不带 --live / 只看 --help 不应联网')

socket.socket.connect = no_network
cases = json.loads(sys.argv[1])
out = {}
for key, path, argv in cases:
    sys.argv = [path] + argv
    buf = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf):
        try:
            runpy.run_path(path, run_name='__main__')
        except SystemExit as e:
            code = e.code or 0
    out[key] = {'code': code, 'stdout': buf.getvalue()}
out['modal_loaded'] = 'modal' in sys.modules
print('@@RESULT@@' + json.dumps(out, ensure_ascii=False))
"""


def _offline_scripts_run() -> dict:
    """在同一个子进程里依次跑这几个脚本的 --help / 不带 --live（导入 deepagents 与 docker 要好几秒，不要每个脚本各开一个进程）。
    断网（任何 socket.connect 都会抛）、带着假令牌；返回每个脚本的退出码与输出，以及最后 modal 是否被导入过。"""
    global _OFFLINE_RUN
    if _OFFLINE_RUN is None:
        cases = [
            ["scripts/build_modal_image.py --help", "scripts/build_modal_image.py", ["--help"]],
            ["scripts/manage_modal_sandboxes.py --help", "scripts/manage_modal_sandboxes.py", ["--help"]],
            ["live", "tests/live_modal.py", []],
            ["live --help", "tests/live_modal.py", ["--help"]],
            ["live --slow", "tests/live_modal.py", ["--slow"]],
        ]
        env = dict(os.environ)
        env["MODAL_TOKEN_ID"], env["MODAL_TOKEN_SECRET"] = _FAKE_ID, _FAKE_SECRET
        res = subprocess.run([sys.executable, "-c", _OFFLINE_DRIVER, json.dumps(cases)], env=env, cwd=str(ROOT),
                             capture_output=True, text=True, encoding="utf-8")
        assert res.returncode == 0, res.stderr[-800:]
        line = next(ln for ln in res.stdout.splitlines() if ln.startswith("@@RESULT@@"))
        assert _FAKE_ID not in res.stdout + res.stderr and _FAKE_SECRET not in res.stdout + res.stderr
        _OFFLINE_RUN = json.loads(line[len("@@RESULT@@"):])
    return _OFFLINE_RUN


def test_q_live_script_is_inert_without_live():
    """Q tests/live_modal.py：不带 --live 打印用法并以退出码 0 退出、不导入 modal、不联网；--help 同样离线（带着假令牌也一样）。"""
    offline = _offline_scripts_run()
    for key in ("live", "live --help", "live --slow"):
        r = offline[key]
        assert r["code"] == 0, (key, r)
        assert "--live" in r["stdout"], (key, r["stdout"])
    assert "开始 Modal 真机验收" not in offline["live"]["stdout"] + offline["live --slow"]["stdout"], "没有 --live 时不应开始任何验收"
    assert offline["modal_loaded"] is False


# @@CHUNK_END@@

SCENARIOS = [
    ("B1 配置默认值", test_b1_config_defaults),
    ("B2 配置夹紧与回退", test_b2_config_clamping),
    ("B3 令牌只是布尔值", test_b3_tokens_are_only_a_boolean),
    ("B4 member_code_status 各分支", test_b4_member_code_status_branches),
    ("B5 门控不联网且不泄露令牌", test_b5_member_code_status_is_offline_and_leak_free),
    ("B6 modal 惰性导入", test_b6_lazy_import_of_modal),
    ("S FakeModal 自检", test_s_fake_modal_self_check),
    ("A 真实 SDK 表面", test_a_real_sdk_surface),
    ("C 创建参数与环境", test_c_creation_parameters_and_env),
    ("D 池、租约与驱逐", test_d_pool_lease_and_eviction),
    ("E 执行语义", test_e_execution_semantics),
    ("F 永不抛异常", test_f_never_raises),
    ("G 路径规则", test_g_path_rules),
    ("H 产物收集", test_h_outputs_collection),
    ("I 中止与关闭", test_i_interrupt_and_close),
    ("J 熔断、创建超时与脱敏", test_j_breaker_startup_timeout_and_redaction),
    ("L 审计日志", test_l_audit_log),
    ("N 静态扫描", test_n_static_scan),
    ("K 路由与 BaseSandbox 契约", test_k_routed_sandbox),
    ("M 镜像定义与脚本", test_m_image_definition_and_scripts),
    ("O member 的 execute_python_code 工具", test_o_member_execute_python_tool),
    ("P 提示词与描述的事实核对", test_p_prompt_and_description_facts),
    ("K2 member CodeAgent 越权尝试", test_k2_member_code_agent_cannot_escape),
    ("Q 真机验收脚本不带 --live 离线", test_q_live_script_is_inert_without_live),
]


def main():
    print("=" * 70)
    print("开始运行多用户改造（阶段 2）Modal 云端沙箱离线冒烟测试...")
    print("=" * 70)
    passed = failed = 0
    for name, fn in SCENARIOS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e!r}")
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
