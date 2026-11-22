"""测试公用：一个忠实的 FakeModal（只给 tests/ 下的脚本用，不属于 app/）。

`FakeModal()` 的实例就当作 `modal` 模块，通过 `modal_sandbox.set_modal_module(fake)` 注入（传 None 恢复真实导入）。
它的价值取决于有多像真的 SDK，尤其要重现会让天真实现出错的行为（modal 1.6.4 源码核对过）：
  - `ContainerProcess.returncode` 在 `wait()` 之前访问抛 InvalidError；没有 `kill`
  - exec 的 timeout 是客户端兜底 deadline：到点后 wait() / poll() 返回 -1，输出流静默结束，**不抛异常**
  - modal.exception 里的 ConnectionError / TimeoutError 是与内建同名的新类，不继承内建
  - 未知关键字参数一律 TypeError；所有收到的参数原样记录
  - terminate() 之后正在读的流到底是“静默结束”还是“抛 SandboxTerminatedError”没有验证过，所以两种都支持（terminate_mode）
一切确定、线程安全、不联网。任何会阻塞的等待都带总时限：挂死会变成断言失败而不是卡住 CI。
"""
import contextlib
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

HANG_GUARD_SECONDS = 20.0         # 任何内部阻塞等待的总时限


# ══════════════════════════════════════════════════════════════════════════════
# 时钟
# ══════════════════════════════════════════════════════════════════════════════

class FakeClock:
    """可注入的时钟：now() / advance(s) / 直接调用。"""

    def __init__(self, start: float = 1000.0):
        self._now = start
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._now

    __call__ = now

    def advance(self, seconds: float) -> float:
        with self._lock:
            self._now += seconds
            return self._now


# ══════════════════════════════════════════════════════════════════════════════
# modal.exception
# ══════════════════════════════════════════════════════════════════════════════

class _Exceptions:
    """与真实 modal.exception 同名、同继承关系（简化掉 gRPC 包装层）。"""

    class Error(Exception):
        pass

    class AuthError(Error):
        pass

    class ResourceExhaustedError(Error):
        pass

    class NotFoundError(Error):
        pass

    class ConnectionError(Error):          # 与内建同名，但不继承内建
        pass

    class TimeoutError(Error):             # 与内建同名，但不继承内建
        pass

    class ServiceError(Error):
        pass

    class InvalidError(Error):
        pass

    class SandboxTerminatedError(Error):
        pass

    class SandboxTimeoutError(TimeoutError):
        pass

    class ExecTimeoutError(TimeoutError):
        pass

    class OutputExpiredError(TimeoutError):
        pass

    class SandboxFilesystemError(Error):
        pass

    class SandboxFilesystemNotFoundError(SandboxFilesystemError):
        pass

    class SandboxFilesystemIsADirectoryError(SandboxFilesystemError):
        pass

    class SandboxFilesystemFileTooLargeError(SandboxFilesystemError):
        pass

    class SandboxFilesystemPermissionError(SandboxFilesystemError):
        pass

    class SandboxFilesystemNotADirectoryError(SandboxFilesystemError):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 命令行为与创建行为
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Out:
    """立即产出：stdout / stderr 字节、退出码；chunks 指定每块字节数（用来把多字节字符切在块边界）。"""
    stdout: bytes = b""
    stderr: bytes = b""
    exit: int = 0
    chunks: Optional[int] = None


class Hang:
    """一直阻塞，直到被终止 / 被软杀脚本命中 / 兜底到点 / fire_inner_timeout()。"""


@dataclass
class Raise:
    """exec 本身抛异常（沙箱被回收 / 连接断开 …）。"""
    exc: BaseException


@dataclass
class Sleep:
    """推进注入的 FakeClock（用来造“耗时”），然后执行 then。"""
    seconds: float
    then: Any = field(default_factory=Out)


@dataclass
class Call:
    """执行时调用 fn(sandbox, inner) -> 行为（Out / Hang / Raise …）：用来在命令里改假文件系统，或抛任意异常（含 KeyboardInterrupt）。"""
    fn: Any


@dataclass
class Ok:
    """创建动作：成功。"""


@dataclass
class Fail:
    """创建动作：抛指定异常。"""
    exc: BaseException


@dataclass
class Block:
    """创建动作：被 Event 卡住（用来测创建超时与“迟到的沙箱必须被 terminate”），放开后按 then 继续。"""
    event: threading.Event
    then: Any = field(default_factory=Ok)


# ══════════════════════════════════════════════════════════════════════════════
# 进程与流
# ══════════════════════════════════════════════════════════════════════════════

class _Stream:
    """可迭代的字节块流：行为线程喂数据，读流线程阻塞读取；close() 后静默结束，fail(exc) 后抛异常。"""

    def __init__(self, fake: "FakeModal", text: bool):
        self._fake = fake
        self._text = text
        self._q: "queue.Queue" = queue.Queue()

    def put(self, chunk: bytes) -> None:
        self._q.put(("data", chunk))

    def close(self) -> None:
        self._q.put(("end", None))

    def fail(self, exc: BaseException) -> None:
        self._q.put(("error", exc))

    def __iter__(self):
        return self

    def __next__(self):
        self._fake._maybe_raise("stream")
        try:
            kind, value = self._q.get(timeout=HANG_GUARD_SECONDS)
        except queue.Empty:
            raise AssertionError("FakeModal 的流在 %.0f 秒内没有数据也没有结束（测试挂死）" % HANG_GUARD_SECONDS)
        if kind == "data":
            return value.decode("utf-8", errors="replace") if self._text else value
        self._q.put((kind, value))          # 再放回去：重复迭代也能看到结束 / 异常
        if kind == "end":
            raise StopIteration
        raise value


class FakeProcess:
    """modal.ContainerProcess 的替身：没有 kill，returncode 在 wait() 之前访问抛 InvalidError。"""

    def __init__(self, fake: "FakeModal", sandbox: "FakeSandbox", argv: tuple, text: bool):
        self._fake = fake
        self.sandbox = sandbox
        self.argv = argv
        self.stdout = _Stream(fake, text)
        self.stderr = _Stream(fake, text)
        self._done = threading.Event()
        self._code: Optional[int] = None
        self._exc: Optional[BaseException] = None
        self._waited = False
        self._lock = threading.Lock()
        self.deadline_fired = False
        self.hanging = False

    # ── 真实 SDK 的表面 ──────────────────────────────────────────────────────

    def wait(self) -> int:
        self._fake._maybe_raise("wait")
        if not self._done.wait(HANG_GUARD_SECONDS):
            raise AssertionError("FakeModal 的 wait() 在 %.0f 秒内没有返回（测试挂死）" % HANG_GUARD_SECONDS)
        self._waited = True
        if self._exc is not None:
            raise self._exc
        return self._code

    def poll(self) -> Optional[int]:
        self._fake._maybe_raise("poll")
        if not self._done.is_set():
            return None
        if self._exc is not None:
            raise self._exc
        return self._code

    @property
    def returncode(self) -> int:
        if not self._waited:
            raise _Exceptions.InvalidError(
                "You must call wait() before accessing the returncode. To poll for the status of a running process, use poll() instead."
            )
        if self._exc is not None:
            raise self._exc
        return self._code

    # ── 测试控制 ─────────────────────────────────────────────────────────────

    def _finish(self, code: int, *, stdout: bytes = b"", stderr: bytes = b"", chunks: Optional[int] = None) -> bool:
        with self._lock:
            if self._done.is_set():
                return False
            for stream, data in ((self.stdout, stdout), (self.stderr, stderr)):
                if data:
                    step = chunks if chunks and chunks > 0 else len(data)
                    for i in range(0, len(data), step):
                        stream.put(data[i:i + step])
                stream.close()
            self._code = code
            self._done.set()
            return True

    def _fail(self, exc: BaseException) -> bool:
        with self._lock:
            if self._done.is_set():
                return False
            self.stdout.fail(exc)
            self.stderr.fail(exc)
            self._exc = exc
            self._done.set()
            return True

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def fire_inner_timeout(self) -> bool:
        """模拟容器里的 `timeout -k 3 N` 触发：以 124 退出。"""
        return self._finish(124)

    def feed(self, stdout: bytes = b"", stderr: bytes = b"") -> None:
        """给还在跑的进程追加输出（不结束它）。"""
        if stdout:
            self.stdout.put(stdout)
        if stderr:
            self.stderr.put(stderr)


@dataclass
class ExecCall:
    argv: tuple
    kwargs: dict
    proc: FakeProcess
    inner: tuple          # 剥掉 `timeout -k 3 N` 包装之后的内层命令


# ══════════════════════════════════════════════════════════════════════════════
# 文件系统
# ══════════════════════════════════════════════════════════════════════════════

class FakeFilesystem:
    """基于 dict 的假文件系统：只接受绝对路径；write_bytes 的参数顺序是（data, path）。"""

    def __init__(self, sandbox: "FakeSandbox"):
        self._sb = sandbox
        self._files: dict[str, tuple[bytes, float]] = {}
        self._dirs: set[str] = {"/", "/work", "/tmp"}
        self._lock = threading.RLock()
        self.fs_max_bytes: Optional[int] = None          # 单文件大小上限（None = 不限）

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _pre(self, point: str, path: str) -> str:
        self._sb._require_alive()
        self._sb.fake._maybe_raise(point)
        if not isinstance(path, str) or not path.startswith("/"):
            raise _Exceptions.SandboxFilesystemError(f"remote_path must be an absolute path: {path!r}")
        return path

    def _mkparents(self, path: str) -> None:
        parent = path.rsplit("/", 1)[0] or "/"
        parts = [p for p in parent.split("/") if p]
        cur = ""
        self._dirs.add("/")
        for p in parts:
            cur += "/" + p
            self._dirs.add(cur)

    # ── 真实 SDK 的表面 ──────────────────────────────────────────────────────

    def write_bytes(self, data, remote_path):
        path = self._pre("fs.write_bytes", remote_path)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        with self._lock:
            if path in self._dirs:
                raise _Exceptions.SandboxFilesystemIsADirectoryError(path)
            if self.fs_max_bytes is not None and len(data) > self.fs_max_bytes:
                raise _Exceptions.SandboxFilesystemFileTooLargeError(path)
            self._mkparents(path)
            self._files[path] = (bytes(data), self._sb.fake.clock.now())

    def read_bytes(self, remote_path):
        path = self._pre("fs.read_bytes", remote_path)
        with self._lock:
            if path in self._dirs:
                raise _Exceptions.SandboxFilesystemIsADirectoryError(path)
            if path not in self._files:
                raise _Exceptions.SandboxFilesystemNotFoundError(path)
            data = self._files[path][0]
            if self.fs_max_bytes is not None and len(data) > self.fs_max_bytes:
                raise _Exceptions.SandboxFilesystemFileTooLargeError(path)
            return data

    def make_directory(self, remote_path, create_parents=True):
        path = self._pre("fs.make_directory", remote_path).rstrip("/") or "/"
        with self._lock:
            if path in self._files:
                raise _Exceptions.SandboxFilesystemNotADirectoryError(path)
            if not create_parents:
                parent = path.rsplit("/", 1)[0] or "/"
                if parent not in self._dirs:
                    raise _Exceptions.SandboxFilesystemNotFoundError(parent)
            self._mkparents(path + "/x")
            self._dirs.add(path)

    def remove(self, remote_path, recursive=False):
        path = self._pre("fs.remove", remote_path)
        with self._lock:
            if path in self._files:
                del self._files[path]
                return
            if path in self._dirs:
                children = [f for f in self._files if f.startswith(path + "/")]
                if children and not recursive:
                    raise _Exceptions.SandboxFilesystemError(f"directory not empty: {path}")
                for f in children:
                    del self._files[f]
                self._dirs = {d for d in self._dirs if d != path and not d.startswith(path + "/")}
                return
            raise _Exceptions.SandboxFilesystemNotFoundError(path)

    def list_files(self, remote_path):
        path = self._pre("fs.list_files", remote_path).rstrip("/") or "/"
        with self._lock:
            if path not in self._dirs:
                raise _Exceptions.SandboxFilesystemNotFoundError(path)
            prefix = path.rstrip("/") + "/"
            names = {f[len(prefix):].split("/")[0] for f in self._files if f.startswith(prefix)}
            names |= {d[len(prefix):].split("/")[0] for d in self._dirs if d.startswith(prefix) and d != path}
            return sorted(n for n in names if n)

    def stat(self, remote_path):
        path = self._pre("fs.stat", remote_path)
        with self._lock:
            if path in self._files:
                return {"path": path, "size": len(self._files[path][0]), "is_dir": False}
            if path in self._dirs:
                return {"path": path, "size": 0, "is_dir": True}
            raise _Exceptions.SandboxFilesystemNotFoundError(path)

    # ── 测试便利 ─────────────────────────────────────────────────────────────

    def put(self, path: str, data: bytes) -> None:
        """测试里直接放文件（不走活性检查与异常注入）。"""
        with self._lock:
            self._mkparents(path)
            self._files[path] = (bytes(data), self._sb.fake.clock.now())

    def has(self, path: str) -> bool:
        with self._lock:
            return path in self._files or path in self._dirs

    def get(self, path: str) -> Optional[bytes]:
        with self._lock:
            return self._files[path][0] if path in self._files else None

    def paths(self) -> list[str]:
        with self._lock:
            return sorted(self._files)

    def _render_outputs(self) -> bytes:
        """`find /work/outputs -mindepth 1 -type f -printf '%P\\t%s\\t%T@\\0'` 的输出。"""
        with self._lock:
            out = []
            for path in sorted(self._files):
                if path.startswith("/work/outputs/"):
                    data, mtime = self._files[path]
                    out.append(f"{path[len('/work/outputs/'):]}\t{len(data)}\t{mtime:.9f}\0")
            return "".join(out).encode("utf-8")

    def _ls_work(self) -> bytes:
        with self._lock:
            names = {f[len("/work/"):].split("/")[0] for f in self._files if f.startswith("/work/")}
            names |= {d[len("/work/"):].split("/")[0] for d in self._dirs if d.startswith("/work/")}
            return "".join(n + "\n" for n in sorted(names) if n and not n.startswith(".")).encode("utf-8")

    def _wipe(self) -> None:
        with self._lock:
            self._files.clear()
            self._dirs = {"/", "/work", "/tmp"}


# ══════════════════════════════════════════════════════════════════════════════
# 沙箱
# ══════════════════════════════════════════════════════════════════════════════

class FakeSandbox:
    """modal.Sandbox 的替身。"""

    _seq = 0
    _seq_lock = threading.Lock()

    def __init__(self, fake: "FakeModal", args: tuple, kwargs: dict):
        with FakeSandbox._seq_lock:
            FakeSandbox._seq += 1
            seq = FakeSandbox._seq
        self.fake = fake
        self.args = args
        self.kwargs = kwargs
        self.tags = dict(kwargs.get("tags") or {})
        self.app_id = getattr(kwargs.get("app"), "app_id", None)
        self._object_id = f"sb-FAKE{seq:08d}"
        self.filesystem = FakeFilesystem(self)
        self.exec_calls: list[ExecCall] = []
        self.softkill_scripts: list[str] = []
        self.softkill_effect = "ok"                     # ok / fail / hang / raise
        self.terminate_block: Optional[threading.Event] = None
        self.terminate_calls = 0
        self.terminated_by_terminate = False
        self._rules: list[tuple[Any, Any]] = list(fake.sandbox_rules)      # 之后创建的每个沙箱都带上 fake.on_exec_all 登记的行为
        self._dead_code: Optional[int] = None
        self._dead_reason: Optional[str] = None
        self._lock = threading.RLock()
        self._exec_cond = threading.Condition()
        self._procs: list[FakeProcess] = []
        self.created_at = fake.clock.now()
        self.last_active = self.created_at

    # ── SDK 表面 ─────────────────────────────────────────────────────────────

    @property
    def object_id(self) -> str:
        return self._object_id

    def poll(self) -> Optional[int]:
        self.fake._maybe_raise("poll_sandbox")
        return self._dead_code

    def get_tags(self) -> dict:
        return dict(self.tags)

    def set_tags(self, tags, *, client=None) -> None:
        self.tags = dict(tags)

    def terminate(self, *, wait: bool = False, **kwargs) -> Optional[int]:
        if kwargs:
            raise TypeError(f"terminate() got an unexpected keyword argument {sorted(kwargs)[0]!r}")
        self.fake._maybe_raise("terminate")
        with self._lock:
            self.terminate_calls += 1
        block = self.terminate_block
        if block is not None and not block.wait(HANG_GUARD_SECONDS):
            raise AssertionError("FakeModal 的 terminate() 被卡住超过总时限（测试挂死）")
        self._die(137, "terminate")
        return self._dead_code

    def exec(self, *args, **kwargs) -> FakeProcess:
        self.fake._check_kwargs("Sandbox.exec", kwargs)
        self.fake._maybe_raise("exec")
        self._require_alive()
        if not args or not all(isinstance(a, str) for a in args):
            raise TypeError("exec() arguments must be non-empty strings")
        text = kwargs.get("text", True)
        proc = FakeProcess(self.fake, self, tuple(args), bool(text))
        inner = self._strip_wrapper(tuple(args))
        call = ExecCall(argv=tuple(args), kwargs=dict(kwargs), proc=proc, inner=inner)
        with self._exec_cond:
            self.last_active = self.fake.clock.now()
            self.exec_calls.append(call)
            self._procs.append(proc)
            self._exec_cond.notify_all()
        self._run_behavior(call, self._find_behavior(inner, tuple(args)))
        return proc

    # ── 测试控制 ─────────────────────────────────────────────────────────────

    def on_exec(self, matcher, behavior) -> None:
        """注册命令行为；后注册的优先。matcher 看剥掉 `timeout -k 3 N` 包装之后的内层命令：
        可以是 callable(inner) / list|tuple（前缀匹配）/ str（内层命令拼成一行后的子串）。"""
        with self._lock:
            self._rules.insert(0, (matcher, behavior))

    def wait_for_proc(self, match=None, *, timeout: float = 5.0, nth: int = 0) -> Optional[FakeProcess]:
        """等到第 nth 个（从 0 起）匹配的 exec 出现，返回它的进程；超时返回 None。"""
        def find():
            hits = [c for c in self.exec_calls if match is None or self._matches(match, c.inner)]
            return hits[nth].proc if len(hits) > nth else None

        with self._exec_cond:
            self._exec_cond.wait_for(lambda: find() is not None, timeout)
            return find()

    def calls(self, match=None) -> list[ExecCall]:
        return [c for c in list(self.exec_calls) if match is None or self._matches(match, c.inner)]

    def hanging(self) -> list[FakeProcess]:
        return [p for p in list(self._procs) if p.hanging and not p.finished]

    def fire_inner_timeout(self) -> bool:
        """让最近一个还在 Hang 的命令以 124 退出（模拟容器里的 timeout 触发）。"""
        procs = self.hanging()
        return procs[-1].fire_inner_timeout() if procs else False

    @property
    def alive(self) -> bool:
        return self._dead_code is None

    @property
    def dead_reason(self) -> Optional[str]:
        return self._dead_reason

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _require_alive(self) -> None:
        if self._dead_code is not None:
            raise _Exceptions.SandboxTerminatedError("Sandbox has terminated")

    @staticmethod
    def _strip_wrapper(argv: tuple) -> tuple:
        if len(argv) >= 5 and argv[0] == "timeout" and argv[1] == "-k":
            return tuple(argv[4:])
        return argv

    @staticmethod
    def _matches(matcher, inner: tuple) -> bool:
        if callable(matcher):
            return bool(matcher(list(inner)))
        if isinstance(matcher, (list, tuple)):
            return tuple(inner[:len(matcher)]) == tuple(matcher)
        return str(matcher) in " ".join(inner)

    def _find_behavior(self, inner: tuple, argv: tuple):
        with self._lock:
            rules = list(self._rules)
        for matcher, behavior in rules:
            if self._matches(matcher, inner):
                return behavior
        return self._default_behavior(inner, argv)

    def _default_behavior(self, inner: tuple, argv: tuple):
        if inner[:2] == ("mkdir", "-p"):
            for d in inner[2:]:
                self.filesystem.make_directory(d, create_parents=True)
            return Out()
        if inner[:1] == ("true",):
            return Out()
        if inner[:3] == ("ls", "-1", "/work"):
            return Out(stdout=self.filesystem._ls_work())
        if inner[:2] == ("find", "/work/outputs"):
            return Out(stdout=self.filesystem._render_outputs())
        if inner[:2] == ("python3", "-c") and len(inner) > 2 and "_kill_wrapper_trees" in inner[2]:
            self.softkill_scripts.append(inner[2])
            effect = self.softkill_effect
            if effect == "ok":
                return _SoftKill(exit=0, kills=True)
            if effect == "fail":
                return _SoftKill(exit=1, kills=False)          # 脚本失败：什么都没杀掉
            if effect == "hang":
                return Hang()
            return Raise(self.fake.exception.Error("softkill exec failed"))
        return Out()                         # 未注册的命令：退出码 0 并被记录（deepagents 的文件工具靠 python3 -c 预检）

    def _run_behavior(self, call: ExecCall, behavior) -> None:
        proc = call.proc
        if isinstance(behavior, Sleep):
            self.fake.clock.advance(behavior.seconds)
            behavior = behavior.then
        if isinstance(behavior, Call):
            try:
                behavior = behavior.fn(self, call.inner)
            except BaseException:
                with self._exec_cond:
                    if proc in self._procs:
                        self._procs.remove(proc)
                raise
        if isinstance(behavior, Raise):
            with self._exec_cond:
                self._procs.remove(proc)
            raise behavior.exc
        if isinstance(behavior, _SoftKill):
            if behavior.kills:
                for p in self.hanging():
                    p._finish(137)
            proc._finish(behavior.exit)
        elif isinstance(behavior, Hang):
            proc.hanging = True
        elif isinstance(behavior, Out):
            proc._finish(behavior.exit, stdout=behavior.stdout, stderr=behavior.stderr, chunks=behavior.chunks)
        else:
            raise TypeError(f"未知的命令行为: {behavior!r}")

    def _die(self, code: int, reason: str) -> None:
        """沙箱死亡：所有还在跑的进程按 fake.terminate_mode 收尾。"""
        with self._lock:
            if self._dead_code is not None:
                return
            self._dead_code = code
            self._dead_reason = reason
            if reason == "terminate":
                self.terminated_by_terminate = True
            procs = list(self._procs)
        self.filesystem._wipe()
        for p in procs:
            if self.fake.terminate_mode == "raise":
                p._fail(_Exceptions.SandboxTerminatedError("Sandbox terminated"))
            else:
                p._finish(-1)
        self.fake._on_sandbox_death(self, reason)


@dataclass
class _SoftKill:
    exit: int = 0
    kills: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# modal 顶层命名空间
# ══════════════════════════════════════════════════════════════════════════════

class FakeApp:
    def __init__(self, name: str, app_id: str):
        self.name = name
        self.app_id = app_id


class FakeImage:
    """不可变的链式记录器：每个方法返回追加了操作记录的新对象。"""

    def __init__(self, fake: "FakeModal", ops: tuple = ()):
        self._fake = fake
        self.ops = ops
        self.built = False
        self.built_in_output = False
        self.built_app = None

    def _chain(self, where: str, op: str, args: tuple, kwargs: dict) -> "FakeImage":
        self._fake._check_kwargs(where, kwargs)
        return FakeImage(self._fake, self.ops + ((op, args, dict(kwargs)),))

    def apt_install(self, *packages, **kwargs):
        return self._chain("Image.apt_install", "apt_install", packages, kwargs)

    def pip_install(self, *packages, **kwargs):
        return self._chain("Image.pip_install", "pip_install", packages, kwargs)

    def run_commands(self, *commands, **kwargs):
        return self._chain("Image.run_commands", "run_commands", commands, kwargs)

    def env(self, vars):
        return self._chain("Image.env", "env", (dict(vars),), {})

    def workdir(self, path):
        return self._chain("Image.workdir", "workdir", (str(path),), {})

    def add_local_file(self, local_path, remote_path, *, copy=False):
        return self._chain("Image.add_local_file", "add_local_file", (str(local_path), remote_path), {"copy": copy})

    def build(self, app=None):
        self._fake.build_calls.append(self)
        self.built = True
        self.built_in_output = self._fake._output_depth > 0
        self.built_app = app
        return self

    def publish(self, name, **kwargs):
        self._fake._check_kwargs("Image.publish", kwargs)
        self._fake.publish_calls.append(name)
        if not self.built:
            raise _Exceptions.InvalidError("Image must be built before it can be published")
        self._fake.published[name] = self
        return None

    def op_names(self) -> list[str]:
        return [op[0] for op in self.ops]


class _SandboxNS:
    """modal.Sandbox：create / list。"""

    def __init__(self, fake: "FakeModal"):
        self._fake = fake

    def create(self, *args, **kwargs) -> FakeSandbox:
        fake = self._fake
        fake._check_kwargs("Sandbox.create", kwargs)
        with fake._lock:
            fake.create_attempts.append((args, dict(kwargs)))
            action = fake._create_plan.pop(0) if fake._create_plan else fake.create_default
        fake._run_create_action(action)
        fake._maybe_raise("create")
        image = kwargs.get("image")
        if not isinstance(image, FakeImage):
            raise TypeError("image must be a modal.Image")
        if image.ops and image.ops[0][0] == "from_name" and image.ops[0][1][0] not in fake.published:
            raise _Exceptions.NotFoundError(f"Image '{image.ops[0][1][0]}' not found")
        sb = FakeSandbox(fake, tuple(args), dict(kwargs))
        with fake._lock:
            fake.created.append(sb)
        return sb

    def list(self, *, app_id=None, tags=None, client=None):
        with self._fake._lock:
            sandboxes = list(self._fake.created)
        out = []
        for sb in sandboxes:
            if not sb.alive:
                continue
            if app_id is not None and sb.app_id != app_id:
                continue
            if tags and any(sb.tags.get(k) != v for k, v in tags.items()):
                continue
            out.append(sb)
        return iter(out)


class _AppNS:
    def __init__(self, fake: "FakeModal"):
        self._fake = fake

    def lookup(self, name, *, client=None, environment_name=None, create_if_missing=False) -> FakeApp:
        fake = self._fake
        fake._maybe_raise("app.lookup")
        with fake._lock:
            fake.lookup_calls.append({"name": name, "create_if_missing": create_if_missing})
            app = fake.apps.get(name)
            if app is None:
                if not create_if_missing:
                    raise _Exceptions.NotFoundError(f"App '{name}' not found")
                app = FakeApp(name, f"ap-FAKE{len(fake.apps) + 1:06d}")
                fake.apps[name] = app
            return app


class _ImageNS:
    def __init__(self, fake: "FakeModal"):
        self._fake = fake

    def from_name(self, name, *, environment_name=None, client=None) -> FakeImage:
        return FakeImage(self._fake, (("from_name", (name,), {}),))

    def debian_slim(self, python_version=None, force_build=False) -> FakeImage:
        return FakeImage(self._fake, (("debian_slim", (), {"python_version": python_version, "force_build": force_build}),))


class FakeModal:
    """当作 `modal` 模块使用。"""

    # 每个接口接受的关键字参数（精确复刻真实 SDK 的签名；smoke_modal 的场景 A 会拿它和真实 inspect.signature 对照）
    ALLOWED = {
        "Sandbox.create": {
            "app", "name", "tags", "image", "env", "secrets", "network_file_systems", "timeout", "idle_timeout",
            "workdir", "gpu", "cloud", "region", "cpu", "memory", "runtime", "block_network",
            "outbound_cidr_allowlist", "outbound_domain_allowlist", "inbound_cidr_allowlist", "volumes", "pty",
            "encrypted_ports", "h2_ports", "unencrypted_ports", "custom_domain", "proxy",
            "include_oidc_identity_token", "readiness_probe", "verbose", "experimental_options", "client",
            "environment_name", "cidr_allowlist",
        },
        "Sandbox.exec": {"stdout", "stderr", "timeout", "workdir", "env", "secrets", "text", "bufsize", "pty"},
        "Sandbox.list": {"app_id", "tags", "client"},
        "Sandbox.terminate": {"wait"},
        "App.lookup": {"client", "environment_name", "create_if_missing"},
        "Image.from_name": {"environment_name", "client"},
        "Image.debian_slim": {"python_version", "force_build"},
        "Image.apt_install": {"force_build", "env", "secrets", "gpu"},
        "Image.pip_install": {"find_links", "index_url", "extra_index_url", "pre", "extra_options", "force_build",
                              "env", "secrets", "gpu"},
        "Image.run_commands": {"env", "secrets", "volumes", "gpu", "force_build"},
        "Image.env": set(),
        "Image.workdir": set(),
        "Image.add_local_file": {"copy"},
        "Image.publish": {"environment_name", "experimental_options", "client"},
    }

    exception = _Exceptions

    def __init__(self):
        self.clock = FakeClock()
        self._lock = threading.RLock()
        self.terminate_mode = "end"              # end：流静默结束、wait() 返回 -1；raise：读流与 wait() 抛 SandboxTerminatedError
        self.created: list[FakeSandbox] = []
        self.create_attempts: list[tuple] = []
        self.create_default: Any = Ok()
        self._create_plan: list[Any] = []
        self.apps: dict[str, FakeApp] = {}
        self.lookup_calls: list[dict] = []
        self.published: dict[str, FakeImage] = {}
        self.build_calls: list[FakeImage] = []
        self.publish_calls: list[str] = []
        self._output_depth = 0
        self._inject: dict[str, list[BaseException]] = {}
        self.deaths: list[tuple[str, str]] = []
        self.sandbox_rules: list[tuple[Any, Any]] = []
        self.Sandbox = _SandboxNS(self)
        self.App = _AppNS(self)
        self.Image = _ImageNS(self)

    # ── 与 modal 同名的顶层函数 ──────────────────────────────────────────────

    @contextlib.contextmanager
    def enable_output(self):
        with self._lock:
            self._output_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._output_depth -= 1

    # ── 测试控制 ─────────────────────────────────────────────────────────────

    def on_create(self, *actions) -> None:
        """依次安排接下来几次 Sandbox.create 的行为（Ok / Fail(exc) / Block(event, then=...)）；用完后回到 create_default。"""
        with self._lock:
            self._create_plan.extend(actions)

    def on_exec_all(self, matcher, behavior) -> None:
        """给之后创建的每个沙箱登记命令行为（沙箱在被测代码里惰性创建，测试拿不到它时用这个）。"""
        with self._lock:
            self.sandbox_rules.insert(0, (matcher, behavior))

    def publish_default_image(self) -> None:
        from app import config
        img = FakeImage(self, (("debian_slim", (), {}),))
        img.built = True
        self.published[config.MODAL_IMAGE_NAME] = img

    def inject(self, point: str, exc: BaseException, times: int = 1) -> None:
        """让某个调用点接下来 times 次抛 exc。
        point：create / exec / stream / wait / poll / poll_sandbox / terminate / app.lookup / fs.write_bytes / fs.read_bytes /
        fs.make_directory / fs.remove / fs.list_files / fs.stat。"""
        with self._lock:
            self._inject.setdefault(point, []).extend([exc] * times)

    def clear_injections(self) -> None:
        with self._lock:
            self._inject.clear()

    def live(self) -> list[FakeSandbox]:
        with self._lock:
            return [s for s in self.created if s.alive]

    def advance(self, seconds: float) -> None:
        """推进虚拟时间：到点按 idle_timeout / timeout 让沙箱死亡（之后 exec 抛 SandboxTerminatedError、poll() 返回 int）。"""
        now = self.clock.advance(seconds)
        for sb in self.live():
            lifetime = sb.kwargs.get("timeout")
            idle = sb.kwargs.get("idle_timeout")
            if lifetime is not None and now - sb.created_at >= lifetime:
                sb._die(0, "timeout")
            elif idle is not None and now - sb.last_active >= idle:
                sb._die(0, "idle")

    def fire_exec_deadline(self, proc: FakeProcess) -> bool:
        """模拟客户端兜底 deadline 到点：wait() / poll() 返回 -1，流静默结束，不抛任何异常。"""
        proc.deadline_fired = True
        return proc._finish(-1)

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _check_kwargs(self, where: str, kwargs: dict) -> None:
        unknown = sorted(set(kwargs) - self.ALLOWED[where])
        if unknown:
            raise TypeError(f"{where}() got an unexpected keyword argument {unknown[0]!r}")

    def _maybe_raise(self, point: str) -> None:
        with self._lock:
            queue_ = self._inject.get(point)
            exc = queue_.pop(0) if queue_ else None
        if exc is not None:
            raise exc

    def _run_create_action(self, action) -> None:
        if isinstance(action, Block):
            if not action.event.wait(HANG_GUARD_SECONDS):
                raise AssertionError("FakeModal 的 create 被卡住超过总时限（测试挂死）")
            action = action.then
        if isinstance(action, Fail):
            raise action.exc
        if not isinstance(action, Ok):
            raise TypeError(f"未知的创建行为: {action!r}")

    def _on_sandbox_death(self, sandbox: FakeSandbox, reason: str) -> None:
        with self._lock:
            self.deaths.append((sandbox.object_id, reason))

    @property
    def terminate_count(self) -> int:
        """被 terminate() 终止的沙箱个数（不含空闲 / 超时回收）。"""
        with self._lock:
            return sum(1 for s in self.created if s.terminated_by_terminate)
