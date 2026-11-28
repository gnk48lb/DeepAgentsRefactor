"""Modal 云端代码沙箱（member 的 CodeAgent 用） (所在层: L1)

契约与分层规则：
  - desktop 形态：owner 的 CodeAgent 用本机 Docker 沙箱（app/infra/sandbox.py），member 的 CodeAgent 用本模块的 Modal 沙箱，
    两条路径完全分开：本模块不引用本机沙箱的任何入口，也不触碰宿主机命令；owner 在这里一律被拒绝。
    cloud 形态（config.is_cloud_mode()）：没有本机 Docker 路径，owner 与普通租户一样按 (owner, session_id) 取沙箱；
    其余的失败即关闭（没有上下文、非法 user_id、空 session_id）照旧。
  - 复用 app.infra.sandbox 里的纯数据 / 工具类（RunResult、OutputsResult、SandboxImage、StreamingBuffer、SandboxUnavailable）。
  - `modal` 只在 load_modal() 里惰性导入：没装 modal、没有令牌时，全部 app 模块照样能被导入。
  - 令牌只由 modal SDK 自己从环境变量读取；本模块只看 config.MODAL_TOKENS_PRESENT（布尔），
    绝不读取、保存、打印、记录令牌值；所有写进日志 / 控制台 / 返回值的异常文本都先过 _redact()。
  - 所有 Modal 调用都是阻塞的：调用方必须在线程里用（异步路径一律 asyncio.to_thread）。
  - 时间：整个文件只允许 clock= 参数的默认值与审计的 ts 字段直接读系统时钟；等待一律用 Event / join。

结构：ModalSandbox（一个 (用户, 会话) 的沙箱）< ModalSandboxPool（上限、驱逐、熔断）< ModalRoutedSandbox（按请求上下文路由）。
"""

import atexit
import codecs
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import logging
import logging.handlers
import os
import posixpath
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from PIL import Image
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from app import config
from app.infra.sandbox import (
    OutputsResult,
    RunResult,
    SandboxImage,
    SandboxUnavailable,
    StreamingBuffer,
)
from app.utils.context import get_request_context
from app.utils.identity import OWNER_USER_ID, validate_user_id


# ══════════════════════════════════════════════════════════════════════════════
# 常量与文案
# ══════════════════════════════════════════════════════════════════════════════

# 单个输出流的原始字节硬上限：超过即中断命令。读流代码在运行时读这个模块全局（测试里会调小它）
_STREAM_HARD_CAP_BYTES = 64 * 1024 * 1024

_MAX_CODE_BYTES = 256 * 1024                 # run_python 的代码大小上限
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024          # 单个上传文件上限
_MAX_OUTPUT_ENTRIES = 500                    # 单次扫描 /work/outputs 最多处理的文件条数
_MAX_IMAGE_PIXELS = 40_000_000
_EXEC_GRACE_SECONDS = 12                     # exec 的客户端兜底超时 = 命令超时 + 这个值
_HELPER_EXEC_TIMEOUT = 30                    # 内部辅助命令（扫描产物、ls、mkdir）的 exec 超时
_HEALTH_TIMEOUT = 15
_KILL_TIMEOUT = 20
_READER_JOIN_SECONDS = 5.0
_BREAKER_THRESHOLD = 3
_BREAKER_OPEN_SECONDS = 60
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

# 传给沙箱的环境变量只有这四项；绝不放任何凭证
_ENV_WHITELIST = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "LANG": "C.UTF-8",
    "HOME": "/work",
}

# 给模型 / member 看的文案：通用、不泄露基础设施细节
_MSG_UNAVAILABLE = "云端代码沙箱暂时不可用，请稍后再试"
_MSG_TOTAL_BUSY = "云端代码沙箱现在比较繁忙，请稍后再试"
_MSG_OWNER_DENIED = "云端沙箱仅供成员使用"
_MSG_NO_IDENTITY = "无法确定当前用户，已拒绝使用云端沙箱"
_MSG_REBUILT = "云端沙箱已重建，/work 里之前的文件已不存在"
_MSG_DEAD = "云端沙箱已被回收（空闲超时或被终止），本次执行未完成；/work 里的文件已丢失，请重试"
_MSG_CLOSED = "云端沙箱已被终止，本次执行未完成"
_MSG_INTERRUPTED = "命令已被中止"
_MSG_OVERFLOW = "输出量过大，命令已终止"


def _msg_user_cap(n: int) -> str:
    return f"你同时运行的代码沙箱已达上限（{n} 个），请等当前任务结束后再试"


# ══════════════════════════════════════════════════════════════════════════════
# 脱敏与哈希
# ══════════════════════════════════════════════════════════════════════════════

# 形如 ak-… / as-…（后面跟 10 个以上字母数字）的令牌形状，以及 MODAL_TOKEN_ID/SECRET = 值
_TOKEN_SHAPE_RE = re.compile(r"\b(?:ak|as)-[A-Za-z0-9]{10,}")
_TOKEN_ASSIGN_RE = re.compile(r"MODAL_TOKEN_(?:ID|SECRET)\s*[=:]\s*\S+")


def _redact(text: Any) -> str:
    """把令牌形状替换成 ***。它不需要读取令牌值本身；所有写进日志 / 控制台 / 返回值的异常文本都先过它。"""
    try:
        s = str(text)
    except Exception:
        s = f"<{type(text).__name__}>"
    s = _TOKEN_ASSIGN_RE.sub("MODAL_TOKEN_***=***", s)
    return _TOKEN_SHAPE_RE.sub("***", s)


def redact(text: Any) -> str:
    """公开版的脱敏（构建 / 管理脚本打印异常前用）。"""
    return _redact(text)


def sandbox_create_kwargs(modal_mod: Any, app: Any, *, tags: dict, image_name: Optional[str] = None,
                          timeout: Optional[int] = None, idle_timeout: Optional[int] = None) -> dict:
    """Sandbox.create 的关键字参数（除入口命令 'sleep infinity' 外）：运行时、镜像验收脚本、真机验收脚本共用这一份，
    保证“验收时用的创建参数”就是“运行时用的创建参数”。不挂 secrets / volumes / 端口 / pty，环境变量只有四项白名单。"""
    return dict(
        app=app,
        image=modal_mod.Image.from_name(image_name or config.MODAL_IMAGE_NAME),
        env=dict(_ENV_WHITELIST),
        timeout=timeout if timeout is not None else config.MODAL_MAX_LIFETIME_SECONDS,
        idle_timeout=idle_timeout if idle_timeout is not None else config.MODAL_IDLE_TIMEOUT_SECONDS,
        workdir="/work",
        cpu=(config.MODAL_CPU_REQUEST, config.MODAL_CPU_LIMIT),
        memory=(config.MODAL_MEMORY_MB_REQUEST, config.MODAL_MEMORY_MB_LIMIT),
        block_network=config.MODAL_BLOCK_NETWORK,
        tags=dict(tags),
    )


def _safe_text(exc: BaseException, limit: int = 200) -> str:
    return _redact(f"{type(exc).__name__}: {exc}")[:limit]


def _deepest_cause(exc: BaseException) -> BaseException:
    """沿 __cause__ / __context__ 链找出最底层的那个异常。"""
    best: BaseException = exc
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        nxt = cur.__cause__ or cur.__context__
        if nxt is None:
            break
        best = nxt
        cur = nxt
    return best


def root_cause(exc: BaseException, limit: int = 300) -> str:
    """最底层原因的一行文本（脱敏、截断）。

    Modal SDK 把建通道失败包装成 "Could not connect to the Modal server."，真正的原因（缺 python-socks、
    DNS 失败、TLS 失败）在 __cause__ 里。运营者看到的应该是真原因，不是包装层。
    """
    best = _deepest_cause(exc)
    return _redact(f"{type(best).__name__}: {best}")[:limit]


def explain(exc: BaseException, limit: int = 300) -> str:
    """脚本用的一行失败说明：包装层文案 + 真原因（真因藏在 __cause__ 链里时才附上）。已脱敏。"""
    top = _redact(f"{type(exc).__name__}: {exc}")[:limit]
    if _deepest_cause(exc) is exc:
        return top
    return f"{top}（真原因：{root_cause(exc, limit)}）"


def _proxy_support_missing(exc: BaseException) -> bool:
    """判断失败是不是“系统代理已启用但缺 gRPC 代理支持包”造成的。

    Modal 走系统代理时需要 python-socks（modal[api-proxy-support]）；缺了它时 SDK 会把 ImportError
    包成 ConnectionError("Could not connect to the Modal server.")，只有沿 __cause__ 链才看得到。
    """
    cur: Optional[BaseException] = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        try:
            text = str(cur)
        except Exception:
            text = ""
        if "python-socks" in text or "python_socks" in text:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _uhash(user_id: Any) -> str:
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:12]


def _shash(session_id: Any) -> str:
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:8]


def _console(message: str) -> None:
    """给运营者看的控制台提示（脱敏；不含明文用户 / 会话 ID）。"""
    try:
        print(_redact(message))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 审计日志
# ══════════════════════════════════════════════════════════════════════════════

_AUDIT_FIELD_CHARS = 500          # 命令 / 代码截断到 500 字符；其余字符串字段同样封顶
_audit_lock = threading.Lock()
_audit_logger: Optional[logging.Logger] = None
_audit_disabled = False
_audit_warned = False


def _get_audit_logger() -> Optional[logging.Logger]:
    """惰性创建审计 logger（第一次写才建目录与 handler）；创建失败就关掉审计并只提示一次。"""
    global _audit_logger, _audit_disabled, _audit_warned
    with _audit_lock:
        if _audit_disabled:
            return None
        if _audit_logger is not None:
            return _audit_logger
        try:
            path = config.MODAL_AUDIT_LOG
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler.handleError = lambda record: None           # 写入失败也不能吵到别人
            logger = logging.getLogger("gnk48.modal_audit")
            logger.propagate = False
            logger.setLevel(logging.INFO)
            for old in list(logger.handlers):
                logger.removeHandler(old)
            logger.addHandler(handler)
            _audit_logger = logger
            return logger
        except Exception as e:
            _audit_disabled = True
            if not _audit_warned:
                _audit_warned = True
                _console(f"⚠️ [modal] 审计日志不可用，已关闭（非致命）: {_safe_text(e)}")
            return None


def _reset_audit() -> None:
    """关闭并重置审计 handler：下一次写审计会按当时的 config.MODAL_AUDIT_LOG 重新创建。"""
    global _audit_logger, _audit_disabled, _audit_warned
    with _audit_lock:
        logger = logging.getLogger("gnk48.modal_audit")
        for handler in list(logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            logger.removeHandler(handler)
        _audit_logger = None
        _audit_disabled = False
        _audit_warned = False


def _audit(event: str, user_id: Any = None, session_id: Any = None, **fields: Any) -> None:
    """一行一个 JSON：{ts, event, u, s, ...}。u / s 是哈希；绝不记录输出内容、环境变量、令牌、明文 ID。绝不抛异常。"""
    try:
        logger = _get_audit_logger()
        if logger is None:
            return
        record: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "event": event,
            "u": _uhash(user_id) if user_id else "",
            "s": _shash(session_id) if session_id else "",
        }
        for key, value in fields.items():
            if isinstance(value, str):
                value = _redact(value)[:_AUDIT_FIELD_CHARS]
            record[key] = value
        logger.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 异常分类（给运营者看的文案）
# ══════════════════════════════════════════════════════════════════════════════

class _CreateTimeout(Exception):
    """创建沙箱超过 MODAL_STARTUP_TIMEOUT_SECONDS。"""


def _modal_exc(modal_mod: Any, name: str) -> Optional[type]:
    """取 modal.exception 里的异常类（modal 的 ConnectionError / TimeoutError 与内建同名，必须经这里取，不能裸写）。"""
    cls = getattr(getattr(modal_mod, "exception", None), name, None)
    return cls if isinstance(cls, type) else None


def _is_modal_exc(exc: BaseException, modal_mod: Any, *names: str) -> bool:
    for name in names:
        cls = _modal_exc(modal_mod, name)
        if cls is not None and isinstance(exc, cls):
            return True
    return False


def _describe_failure(exc: BaseException, modal_mod: Any) -> tuple[str, str]:
    """创建沙箱失败的原因分类：返回 (类别, 给运营者看的中文文案)。文案已脱敏。"""
    if _is_modal_exc(exc, modal_mod, "NotFoundError"):
        return "image_missing", "镜像可能没有发布，请运行 scripts/build_modal_image.py（可加 --check-only 检查）"
    if _is_modal_exc(exc, modal_mod, "AuthError"):
        return "auth", "凭证无效或已过期，请检查 .env 里的 Modal 令牌"
    if _is_modal_exc(exc, modal_mod, "ResourceExhaustedError"):
        return "quota", "Modal 资源或额度不足"
    if _proxy_support_missing(exc):
        return "proxy", (
            "系统代理已启用，但缺少 Modal 走代理所需的 python-socks："
            "请安装 modal[api-proxy-support]，或在 .env 里设 MODAL_DISABLE_API_PROXY=1 改为直连"
        )
    if isinstance(exc, _CreateTimeout) or _is_modal_exc(exc, modal_mod, "ConnectionError", "ServiceError", "TimeoutError"):
        return "network", "连不上 Modal（网络、服务异常或创建超时）"
    return "other", _safe_text(exc)


# ══════════════════════════════════════════════════════════════════════════════
# 软杀：只杀我们自己的命令包装进程及其后代
# ══════════════════════════════════════════════════════════════════════════════

def _kill_wrapper_trees(proc_root="/proc", *, self_pid=None, kill=None):
    """扫 proc_root，杀掉我们自己的命令包装进程（cmdline 以 `timeout -k 3` 开头的 timeout）及其全部后代。

    这个函数会被 inspect.getsource 拼成 `python3 -c` 脚本在沙箱里跑，所以必须自包含：只用标准库，
    函数体里自己 import，不引用任何模块级名字。名单里永远排除 PID 1 与 self_pid（脚本自己），
    不碰任何不属于这棵树的进程（它们可能是 Modal 自己的基础设施进程，或 sleep infinity）。
    返回被杀的 pid 列表。
    """
    import os
    import signal

    if self_pid is None:
        self_pid = os.getpid()
    if kill is None:
        def kill(pid):
            os.kill(pid, signal.SIGKILL)

    try:
        names = os.listdir(proc_root)
    except OSError:
        return []

    parent_of = {}
    wrappers = []
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(os.path.join(proc_root, name, "stat"), "rb") as f:
                stat = f.read().decode("utf-8", "replace")
            # comm 字段可能含空格与括号：按最后一个 ")" 切；其后依次是 状态、ppid、…
            parent_of[pid] = int(stat[stat.rindex(")") + 1:].split()[1])
        except Exception:
            continue
        try:
            with open(os.path.join(proc_root, name, "cmdline"), "rb") as f:
                argv = f.read().decode("utf-8", "replace").split("\0")
            if len(argv) >= 3 and os.path.basename(argv[0]) == "timeout" and argv[1:3] == ["-k", "3"]:
                wrappers.append(pid)
        except Exception:
            pass

    children = {}
    for pid, ppid in parent_of.items():
        children.setdefault(ppid, []).append(pid)

    doomed = set()
    stack = list(wrappers)
    while stack:
        pid = stack.pop()
        if pid in doomed:
            continue
        doomed.add(pid)
        stack.extend(children.get(pid, ()))
    doomed.discard(1)
    doomed.discard(self_pid)

    killed = []
    for pid in sorted(doomed, reverse=True):
        try:
            kill(pid)
            killed.append(pid)
        except Exception:
            pass
    return sorted(killed)


def _kill_script() -> Optional[str]:
    try:
        return inspect.getsource(_kill_wrapper_trees) + "\nprint(_kill_wrapper_trees())\n"
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 小工具
# ══════════════════════════════════════════════════════════════════════════════

class _Latch:
    """倒计数门闩：所有读流线程收尾后放行，用一次 wait(timeout) 给整体限时。"""

    def __init__(self, count: int):
        self._count = count
        self._lock = threading.Lock()
        self._event = threading.Event()
        if count <= 0:
            self._event.set()

    def count_down(self) -> None:
        with self._lock:
            self._count -= 1
            if self._count <= 0:
                self._event.set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


class _Run:
    """一次命令执行的结果（内部用）。"""

    __slots__ = ("exit_code", "timed_out", "duration", "notes", "alive")

    def __init__(self, exit_code: Optional[int], timed_out: bool, duration: float, notes: list[str], alive: bool):
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.duration = duration
        self.notes = notes
        self.alive = alive


def _decode_lines(raw: bytes) -> list[str]:
    return [ln.strip() for ln in raw.decode("utf-8", errors="replace").splitlines() if ln.strip()]


def _fs_error_code(exc: BaseException, modal_mod: Any) -> str:
    """文件系统异常 -> 与 Docker 版一致的错误码；其余用脱敏、截断的文本。"""
    if _is_modal_exc(exc, modal_mod, "SandboxFilesystemNotFoundError"):
        return "file_not_found"
    if _is_modal_exc(exc, modal_mod, "SandboxFilesystemIsADirectoryError"):
        return "is_directory"
    if _is_modal_exc(exc, modal_mod, "SandboxFilesystemFileTooLargeError"):
        return "file_too_large"
    if _is_modal_exc(exc, modal_mod, "SandboxFilesystemPermissionError"):
        return "permission_denied"
    return _safe_text(exc)


# ══════════════════════════════════════════════════════════════════════════════
# ModalSandbox：一个 (用户, 会话) 的沙箱
# ══════════════════════════════════════════════════════════════════════════════

class ModalSandbox:
    """包装一个 Modal 沙箱，接口与本机 Docker 沙箱对齐（惰性创建、会话级常驻、空闲由 Modal 回收）。

    并发规则：每个实例一把 RLock，run_python / execute / upload_files / download_files / collect_new_outputs
    持锁执行（同一沙箱里的并行工具调用被串行化）；interrupt() 与 close() 不取这把锁，
    否则中断要等被中断的命令跑完，等于没中断。锁里绝不保持池的锁。
    """

    def __init__(self, pool: "ModalSandboxPool", user_id: str, session_id: str):
        self._pool = pool
        self._modal = pool._modal
        self._clock = pool._clock
        self._user_id = user_id
        self._session_id = session_id
        self._uh = _uhash(user_id)
        self._sh = _shash(session_id)
        self._lock = threading.RLock()
        self._sb: Any = None                    # 当前的 modal 沙箱对象；None = 还没创建 / 已死 / 已关闭
        self._dead = False                      # 被回收 / 被我们终止后置位，下一次命令惰性重建
        self._closed = False                    # close() 之后永久置位（池里的条目已摘掉）
        self._ever_created = False
        self._interrupt_gen = 0                 # interrupt() 每次调用 +1，用来识别“这条命令是被我们中止的”
        self._run_count = 0
        self._seen_outputs: dict[str, tuple[int, float]] = {}

    # ── 状态 ─────────────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        sb = self._sb
        if sb is not None:
            try:
                return str(sb.object_id)
            except Exception:
                pass
        return "modal-lazy"

    @property
    def alive(self) -> bool:
        return self._sb is not None and not self._dead and not self._closed

    @property
    def dead(self) -> bool:
        return self._closed or self._dead

    def generation(self, *, check_remote: bool = True) -> Optional[str]:
        """当前这一代云端沙箱的标识（modal 沙箱的 object_id）：沙箱被回收 / 重建 / 重置后标识就变了，
        /work 里的文件也随之消失，所以它可以用来判断“之前上传的文件还在不在”。
        只读：不创建沙箱、不取 _lock（别被长命令堵住）、不调用 _mark_dead、永不抛异常。"""
        try:
            sb = self._sb
            if sb is None or self._dead or self._closed:
                return None
            if check_remote and not self._is_alive(sb):
                return None
            return str(sb.object_id)
        except Exception:
            return None

    def _effective_timeout(self, timeout: Any) -> int:
        """None / ≤0 / 非数字取默认，否则夹到上限。"""
        try:
            value = int(timeout) if timeout is not None else 0
        except (TypeError, ValueError, OverflowError):
            value = 0
        if value <= 0:
            return config.MODAL_EXEC_TIMEOUT_SECONDS
        return max(1, min(value, config.MODAL_EXEC_MAX_TIMEOUT_SECONDS))

    # ── 创建 ─────────────────────────────────────────────────────────────────

    def _tags(self) -> dict[str, str]:
        return {"svc": "gnk48", "role": "member-code", "u": self._uh, "s": self._sh}

    def _create_sandbox(self, app: Any) -> Any:
        """在辅助线程里创建（带整体超时）。晚于超时才创建成功的沙箱必须立刻终止，否则会变成孤儿一直计费。"""
        modal = self._modal
        state: dict[str, Any] = {"sb": None, "exc": None, "abandoned": False}
        state_lock = threading.Lock()
        done = threading.Event()

        def worker() -> None:
            try:
                sb = modal.Sandbox.create("sleep", "infinity", **sandbox_create_kwargs(modal, app, tags=self._tags()))
            except BaseException as e:      # 线程里的任何异常都转交给主线程
                with state_lock:
                    state["exc"] = e
                done.set()
                return
            late = False
            with state_lock:
                if state["abandoned"]:
                    late = True
                else:
                    state["sb"] = sb
            if late:
                _audit("terminate", self._user_id, self._session_id, reason="late_create")
                try:
                    sb.terminate(wait=False)
                except Exception:
                    pass
            done.set()

        threading.Thread(target=worker, name="modal-create", daemon=True).start()
        if not done.wait(config.MODAL_STARTUP_TIMEOUT_SECONDS):
            with state_lock:
                if state["sb"] is None and state["exc"] is None:
                    state["abandoned"] = True
                    raise _CreateTimeout(f"创建沙箱超过 {config.MODAL_STARTUP_TIMEOUT_SECONDS} 秒")
        with state_lock:
            if state["exc"] is not None:
                raise state["exc"]
            return state["sb"]

    def _create(self) -> Any:
        """创建沙箱：熔断检查 -> App.lookup -> Sandbox.create -> 建目录。失败统一转成 SandboxUnavailable（通用文案）。"""
        self._pool._breaker_check(self._user_id, self._session_id)       # 熔断中直接 SandboxUnavailable
        started = self._clock()
        sb = None
        try:
            app = self._pool._lookup_app()
            sb = self._create_sandbox(app)
            code = self._simple_exec(sb, ("mkdir", "-p", "/work/outputs", "/work/.sandbox"), _HELPER_EXEC_TIMEOUT)
            if code != 0:
                raise RuntimeError(f"初始化目录失败（退出码 {code}）")
        except Exception as e:
            if sb is not None:
                try:
                    sb.terminate(wait=False)
                except Exception:
                    pass
            category, text = _describe_failure(e, self._modal)
            self._pool._breaker_failure()
            _audit("create_fail", self._user_id, self._session_id, category=category, error=type(e).__name__, detail=text)
            _console(f"⚠️ [modal] 创建云端沙箱失败（{category}）: {text}")
            raise SandboxUnavailable(_MSG_UNAVAILABLE) from None
        self._pool._breaker_success()
        _audit("create_ok", self._user_id, self._session_id, ms=int((self._clock() - started) * 1000))
        return sb

    def _ensure(self) -> tuple[Any, list[str]]:
        """确保沙箱存在且活着（持 _lock 调用）。返回 (modal 沙箱, 一次性 notes)。"""
        if self._closed:
            raise SandboxUnavailable(_MSG_UNAVAILABLE)
        notes: list[str] = []
        sb = self._sb
        if sb is not None and not self._dead and not self._is_alive(sb):
            self._mark_dead()
            sb = None
        if sb is None or self._dead:
            was_created = self._ever_created
            sb = self._create()
            self._sb = sb
            self._dead = False
            self._ever_created = True
            self._seen_outputs.clear()
            if was_created:
                notes.append(_MSG_REBUILT)
        return sb, notes

    @staticmethod
    def _is_alive(sb: Any) -> bool:
        try:
            return sb.poll() is None
        except Exception:
            return False

    def _mark_dead(self) -> None:
        self._dead = True
        self._sb = None
        self._seen_outputs.clear()

    def _terminate_quietly(self, sb: Any, reason: str) -> None:
        try:
            sb.terminate(wait=False)
        except Exception:
            pass
        _audit("terminate", self._user_id, self._session_id, reason=reason)

    # ── 辅助命令 ─────────────────────────────────────────────────────────────

    def _simple_exec(self, sb: Any, argv: tuple, timeout: int) -> Optional[int]:
        """跑一条短命令并丢弃输出，返回退出码（-1 = 客户端兜底到点）。异常往上抛。"""
        proc = sb.exec(*argv, workdir="/work", timeout=timeout, text=False)
        return proc.wait()

    def _capture(self, sb: Any, argv: tuple, timeout: int, max_bytes: int = 1024 * 1024) -> tuple[Optional[int], bytes]:
        """跑一条短命令并取回 stdout（最多 max_bytes）。异常往上抛。"""
        proc = sb.exec(*argv, workdir="/work", timeout=timeout, text=False)
        chunks: list[bytes] = []
        size = 0
        for chunk in proc.stdout:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "replace")
            size += len(chunk)
            if size <= max_bytes:
                chunks.append(chunk)
        return proc.wait(), b"".join(chunks)

    # ── 命令执行 ─────────────────────────────────────────────────────────────

    def _pump(self, stream: Any, buf: StreamingBuffer, buf_lock: threading.Lock, capped: threading.Event,
              errors: list[BaseException], latch: _Latch) -> None:
        """读一路输出：增量 UTF-8 解码（多字节字符可能被切在块边界）、喂有界缓冲、超过原始字节硬上限就中断命令。
        线程里的任何异常（含 OutputExpiredError、沙箱被终止后流的各种怪行为）都转成记录，绝不带着异常退出。"""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        total = 0
        try:
            for chunk in stream:
                if not chunk:
                    continue
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", "replace")
                total += len(chunk)
                text = decoder.decode(chunk)
                if text:
                    with buf_lock:
                        buf.feed(text)
                if total > _STREAM_HARD_CAP_BYTES:       # 运行时读模块全局：不要绑成默认参数
                    if not capped.is_set():
                        capped.set()
                        self.interrupt()
                    break
            tail = decoder.decode(b"", final=True)
            if tail:
                with buf_lock:
                    buf.feed(tail)
        except BaseException as e:
            errors.append(e)
        finally:
            latch.count_down()

    def _health_ok(self, sb: Any) -> bool:
        try:
            return self._simple_exec(sb, ("true",), _HEALTH_TIMEOUT) == 0
        except Exception:
            return False

    def _run(self, sb: Any, argv: list[str], n: int, out_buf: StreamingBuffer, err_buf: StreamingBuffer) -> _Run:
        """起命令、并发读两路输出、判定退出原因（持 _lock 调用）。绝不抛异常。"""
        gen0 = self._interrupt_gen
        start = self._clock()
        buf_lock = threading.Lock()
        capped = threading.Event()
        reader_errors: list[BaseException] = []
        code: Optional[int] = None
        failure: Optional[BaseException] = None
        threads: list[threading.Thread] = []
        latch = _Latch(0)
        try:
            proc = sb.exec(*argv, workdir="/work", timeout=n + _EXEC_GRACE_SECONDS, text=False)
            latch = _Latch(2)
            for stream, buf in ((proc.stdout, out_buf), (proc.stderr, err_buf)):
                t = threading.Thread(
                    target=self._pump, args=(stream, buf, buf_lock, capped, reader_errors, latch),
                    name="modal-read", daemon=True,
                )
                threads.append(t)
                t.start()
            code = proc.wait()                      # 退出码只用 wait() 的返回值，不访问 returncode
        except Exception as e:
            failure = e
        finally:
            if threads:
                latch.wait(_READER_JOIN_SECONDS)    # 主线程 wait 之后给读流线程最多 5 秒收尾
        elapsed = self._clock() - start

        notes: list[str] = []
        timed_out = False
        interrupted = self._interrupt_gen != gen0
        alive = True

        if capped.is_set():
            notes.append(_MSG_OVERFLOW)
        if self._closed:
            notes.append(_MSG_CLOSED)
            alive = False
            code = None if (failure is not None or code == -1) else code
        elif failure is not None:
            # 沙箱被回收 / 终止 / 连接断开：命令没跑完。不自动重试（无法确定它有没有已经执行过）
            if interrupted and not capped.is_set():
                notes.append(f"{_MSG_INTERRUPTED}；沙箱已重置，/work 已清空")
            elif not capped.is_set():
                notes.append(_MSG_DEAD)
            if not self._dead:                      # interrupt() 的 terminate 回退已经终止过它了：不重复
                self._terminate_quietly(sb, "dead")
            self._mark_dead()
            alive = False
            code = 1
        elif code == -1:
            if interrupted and self._dead:
                if not capped.is_set():
                    notes.append(f"{_MSG_INTERRUPTED}；沙箱已重置，/work 已清空")
                alive = False
                code = None
            else:
                # 客户端兜底 deadline 到点：没有异常，流静默结束
                timed_out = True
                code = None
                notes.append("执行超时，云端看门狗终止了该命令")
                if not self._health_ok(sb):
                    self._terminate_quietly(sb, "dead")
                    self._mark_dead()
                    alive = False
                    notes.append("沙箱已重置，/work 已清空")
        elif capped.is_set():
            pass
        elif interrupted:
            notes.append(_MSG_INTERRUPTED)           # 软杀：沙箱与 /work 保留
        elif code == 124 or (code == 137 and elapsed >= n):
            timed_out = True
            notes.append(f"命令执行超过 {n}秒被终止；沙箱状态保留")
            self._cleanup_residue(sb)
        elif code == 137 and elapsed < n:
            # 实测（live_modal.py Case 7）：内存超限杀掉进程之后，Modal 很快就把整个沙箱回收了，
            # /work 里的文件随之丢失（紧接着扫描产物目录会拿到 NotFoundError）。下一次命令会重建并提示。
            notes.append(
                f"进程被杀（可能超出内存上限 {config.MODAL_MEMORY_MB_LIMIT} MiB；"
                "沙箱可能已被回收，/work 里的文件已丢失）"
            )

        if reader_errors and failure is None and not (self._closed or interrupted or capped.is_set()):
            notes.append("读取命令输出时出现异常，输出可能不完整")
        return _Run(code, timed_out, elapsed, notes, alive and not self._closed and not self._dead)

    def _cleanup_residue(self, sb: Any) -> None:
        """命令超时后清理残留进程：只跑软杀脚本一次，失败就算了（不终止沙箱，不做健康检查）。"""
        script = _kill_script()
        if script is None:
            return
        try:
            self._simple_exec(sb, ("python3", "-c", script), _KILL_TIMEOUT)
        except Exception:
            pass

    def run_python(self, code: str, *, timeout: Optional[int] = None) -> RunResult:
        """在沙箱里执行 Python 代码。创建失败时抛 SandboxUnavailable（由 execute_python_code 转成文本）。"""
        with self._lock:
            self._run_count += 1
            code_bytes = str(code).encode("utf-8", "replace")
            if len(code_bytes) > _MAX_CODE_BYTES:
                return RunResult(
                    stdout="",
                    stderr="代码大小超过 256KB 上限，已被直接拒绝执行。",
                    exit_code=1,
                    timed_out=False,
                    duration=0.0,
                    notes=["代码超过 256KB 阈值"],
                )
            n = self._effective_timeout(timeout)
            sb, notes = self._ensure()

            snippet_path = "/work/.sandbox/snippet.py"
            try:
                sb.filesystem.make_directory("/work/.sandbox", create_parents=True)
                sb.filesystem.write_bytes(code_bytes, snippet_path)
            except Exception as e:
                if not self._is_alive(sb):
                    self._terminate_quietly(sb, "dead")
                    self._mark_dead()
                    notes.append(_MSG_DEAD)
                return RunResult(
                    stdout="",
                    stderr=f"沙箱环境写入代码失败: {_safe_text(e)}",
                    exit_code=1,
                    timed_out=False,
                    duration=0.0,
                    notes=notes,
                )

            stdout_buf = StreamingBuffer(config.SANDBOX_MAX_STDOUT_CHARS)
            stderr_buf = StreamingBuffer(config.SANDBOX_MAX_STDERR_CHARS)
            run = self._run(sb, ["timeout", "-k", "3", str(n), "python", "-u", snippet_path], n, stdout_buf, stderr_buf)
            notes.extend(run.notes)

            stdout_text, stdout_truncated = stdout_buf.get_result()
            stderr_text, stderr_truncated = stderr_buf.get_result()
            if stdout_truncated:
                notes.append(f"stdout 输出已超过 {config.SANDBOX_MAX_STDOUT_CHARS} 字符并完成头尾截断")
            if stderr_truncated:
                notes.append(f"stderr 输出已超过 {config.SANDBOX_MAX_STDERR_CHARS} 字符并完成头尾截断")

            images: list[SandboxImage] = []
            other_files: list[tuple[str, int]] = []
            work_listing: list[str] = []
            if run.alive and self._sb is not None:
                try:
                    outputs = self.collect_new_outputs()
                    images.extend(outputs.images)
                    other_files.extend(outputs.other_files)
                    notes.extend(outputs.notes)
                except Exception as e:
                    notes.append(f"产物提取异常: {_safe_text(e)}")
                try:
                    exit_ls, raw_ls = self._capture(self._sb, ("ls", "-1", "/work"), _HELPER_EXEC_TIMEOUT)
                    if exit_ls == 0:
                        work_listing = _decode_lines(raw_ls)[:20]
                except Exception:
                    pass

            print(
                f"[Modal Run #{self._run_count}] exit={run.exit_code} duration={run.duration:.2f}s "
                f"stdout={len(stdout_text)}chars stderr={len(stderr_text)}chars images={len(images)}"
            )
            _audit(
                "run_python", self._user_id, self._session_id,
                code_len=len(code_bytes), code=str(code)[:_AUDIT_FIELD_CHARS],
                exit=run.exit_code, timed_out=run.timed_out, ms=int(run.duration * 1000),
            )
            return RunResult(
                stdout=stdout_text,
                stderr=stderr_text,
                exit_code=run.exit_code,
                timed_out=run.timed_out,
                duration=run.duration,
                images=images,
                other_files=other_files,
                work_listing=work_listing,
                notes=notes,
            )

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        """执行 Shell 命令，满足 BaseSandbox 契约且绝不向外抛异常（创建失败也返回带说明的结果）。"""
        try:
            with self._lock:
                n = self._effective_timeout(timeout)
                sb, notes = self._ensure()
                buf = StreamingBuffer(config.SANDBOX_EXECUTE_MAX_OUTPUT_CHARS)      # 两路输出共用一个缓冲（读流时加锁）
                run = self._run(sb, ["timeout", "-k", "3", str(n), "bash", "-c", str(command)], n, buf, buf)
                notes.extend(run.notes)
                text, truncated = buf.get_result()
                _audit(
                    "exec", self._user_id, self._session_id,
                    command=str(command)[:_AUDIT_FIELD_CHARS], timeout=n, exit=run.exit_code,
                    timed_out=run.timed_out, ms=int(run.duration * 1000), truncated=truncated,
                )
                if notes:
                    text += "".join(f"\n[{note}]" for note in notes)
                return ExecuteResponse(output=text, exit_code=run.exit_code, truncated=truncated)
        except SandboxUnavailable as e:
            return ExecuteResponse(output=f"[云端沙箱不可用] {e}", exit_code=1, truncated=False)
        except Exception as e:
            _console(f"⚠️ [modal] execute 遇到未预期异常（非致命）: {_safe_text(e)}")
            return ExecuteResponse(output=f"[云端沙箱不可用] {_MSG_UNAVAILABLE}", exit_code=1, truncated=False)

    # ── 文件 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_upload_path(path: Any) -> tuple[Optional[str], Optional[str]]:
        """返回 (规范化后的路径, 错误码)。只允许写 /work 与 /tmp（精确前缀判断，/workfoo 不放行）。"""
        if not isinstance(path, str) or not path or "\x00" in path or not path.startswith("/"):
            return None, "invalid_path：路径必须是绝对路径，且只能写入 /work/ 或 /tmp/"
        norm = posixpath.normpath(path)
        if norm == "/workspace" or norm.startswith("/workspace/"):
            return None, "permission_denied：云端沙箱没有 /workspace，请写到 /work/ 下"
        if norm == "/work" or norm.startswith("/work/") or norm == "/tmp" or norm.startswith("/tmp/"):
            return norm, None
        return None, "invalid_path：只能写入 /work/ 或 /tmp/"

    def upload_files(self, files: list[tuple[str, bytes]], *, max_bytes: Optional[int] = None) -> list[FileUploadResponse]:
        """上传文件到沙箱：逐个文件分别捕获异常，绝不向外抛异常。
        max_bytes 只给宿主机自己的上传入口（QQ 文件接收）用；agent 侧（ModalRoutedSandbox）不传，仍是 _MAX_UPLOAD_BYTES。"""
        limit = _MAX_UPLOAD_BYTES if max_bytes is None else max(1, int(max_bytes))
        responses: list[Optional[FileUploadResponse]] = [None] * len(files)
        todo: list[tuple[int, str, str, bytes]] = []
        for i, item in enumerate(files):
            try:
                path, data = item
            except Exception:
                responses[i] = FileUploadResponse(path=str(item)[:100], error="invalid_path：参数格式错误")
                continue
            norm, err = self._check_upload_path(path)
            if err is not None:
                responses[i] = FileUploadResponse(path=path, error=err)
            elif not isinstance(data, (bytes, bytearray, memoryview)):
                responses[i] = FileUploadResponse(path=path, error="invalid_path：文件内容必须是字节")
            elif len(data) > limit:
                responses[i] = FileUploadResponse(path=path, error="file_too_large")
            else:
                todo.append((i, path, norm, bytes(data)))
        if todo:
            try:
                with self._lock:
                    sb, _notes = self._ensure()
                    for i, path, norm, data in todo:
                        try:
                            sb.filesystem.make_directory(posixpath.dirname(norm) or "/", create_parents=True)
                            sb.filesystem.write_bytes(data, norm)
                            responses[i] = FileUploadResponse(path=path, error=None)
                            _audit("upload", self._user_id, self._session_id, path=norm, bytes=len(data))
                        except Exception as e:
                            responses[i] = FileUploadResponse(path=path, error=_fs_error_code(e, self._modal))
                            if not self._is_alive(sb):
                                self._terminate_quietly(sb, "dead")
                                self._mark_dead()
                                for j, other_path, _n, _d in todo:        # 沙箱没了：剩下的文件也一并报错
                                    if responses[j] is None:
                                        responses[j] = FileUploadResponse(path=other_path, error=_MSG_DEAD)
                                break
            except SandboxUnavailable as e:
                for i, path, _norm, _data in todo:
                    if responses[i] is None:
                        responses[i] = FileUploadResponse(path=path, error=str(e))
            except Exception as e:
                for i, path, _norm, _data in todo:
                    if responses[i] is None:
                        responses[i] = FileUploadResponse(path=path, error=_safe_text(e))
        return [r if r is not None else FileUploadResponse(path=str(files[i][0]) if files[i] else "", error="invalid_path")
                for i, r in enumerate(responses)]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """从沙箱下载文件：逐个文件分别捕获异常，绝不向外抛异常。"""
        responses: list[Optional[FileDownloadResponse]] = [None] * len(paths)
        todo: list[tuple[int, str, str]] = []
        for i, path in enumerate(paths):
            if not isinstance(path, str) or not path or "\x00" in path or not path.startswith("/"):
                responses[i] = FileDownloadResponse(path=str(path), content=None, error="invalid_path")
            else:
                todo.append((i, path, posixpath.normpath(path)))
        if todo:
            try:
                with self._lock:
                    sb, _notes = self._ensure()
                    for i, path, norm in todo:
                        try:
                            content = sb.filesystem.read_bytes(norm)
                            responses[i] = FileDownloadResponse(path=path, content=bytes(content), error=None)
                            _audit("download", self._user_id, self._session_id, path=norm, bytes=len(content))
                        except Exception as e:
                            responses[i] = FileDownloadResponse(path=path, content=None, error=_fs_error_code(e, self._modal))
                            if not self._is_alive(sb):
                                self._terminate_quietly(sb, "dead")
                                self._mark_dead()
                                for j, other_path, _n in todo:            # 沙箱没了：剩下的文件也一并报错
                                    if responses[j] is None:
                                        responses[j] = FileDownloadResponse(path=other_path, content=None, error=_MSG_DEAD)
                                break
            except SandboxUnavailable as e:
                for i, path, _norm in todo:
                    if responses[i] is None:
                        responses[i] = FileDownloadResponse(path=path, content=None, error=str(e))
            except Exception as e:
                for i, path, _norm in todo:
                    if responses[i] is None:
                        responses[i] = FileDownloadResponse(path=path, content=None, error=_safe_text(e))
        return [r if r is not None else FileDownloadResponse(path=str(paths[i]), content=None, error="file_not_found")
                for i, r in enumerate(responses)]

    def collect_new_outputs(self) -> OutputsResult:
        """扫描 /work/outputs，按指纹 (路径: (大小, mtime)) 提取新增或修改的产物。不会为了扫描而创建沙箱。"""
        with self._lock:
            sb = self._sb
            if sb is None or self._dead or self._closed:
                return OutputsResult()

            images: list[SandboxImage] = []
            other_files: list[tuple[str, int]] = []
            notes: list[str] = []
            try:
                code, raw = self._capture(
                    sb, ("find", "/work/outputs", "-mindepth", "1", "-type", "f", "-printf", "%P\\t%s\\t%T@\\0"),
                    _HELPER_EXEC_TIMEOUT,
                )
            except Exception as e:
                return OutputsResult(notes=[f"扫描产物目录失败: {_safe_text(e)}"])
            if code != 0 or not raw:
                return OutputsResult()

            entries = [e for e in raw.decode("utf-8", errors="replace").split("\0") if e.strip()]
            if len(entries) > _MAX_OUTPUT_ENTRIES:
                notes.append(f"产物文件数量超过上限 ({_MAX_OUTPUT_ENTRIES} 个)，多余文件已忽略")
                entries = entries[:_MAX_OUTPUT_ENTRIES]

            new_or_modified: list[tuple[str, int, float]] = []
            for entry in entries:
                parts = entry.split("\t")
                if len(parts) != 3:
                    continue
                try:
                    size = int(parts[1])
                    mtime = float(parts[2])
                except ValueError:
                    continue
                norm_rel = posixpath.normpath(parts[0])
                if norm_rel.startswith("..") or norm_rel.startswith("/") or "/../" in f"/{norm_rel}/":
                    continue
                if self._seen_outputs.get(norm_rel) != (size, mtime):
                    new_or_modified.append((norm_rel, size, mtime))

            for norm_rel, size, mtime in new_or_modified:
                self._seen_outputs[norm_rel] = (size, mtime)       # 因超限被跳过的文件也记为已见
                basename = posixpath.basename(norm_rel)
                ext = posixpath.splitext(basename)[1].lower()
                if ext not in _IMAGE_EXTS:
                    other_files.append((basename, size))
                    notes.append(f"非图片产物未导出：{basename} ({size} 字节)")
                    continue
                if size > config.SANDBOX_MAX_IMAGE_MB * 1024 * 1024:
                    notes.append(
                        f"图片 {basename} 大小 ({size} 字节) 超过 {config.SANDBOX_MAX_IMAGE_MB}MB 上限，已跳过：{basename}，需要时请重新生成"
                    )
                    continue
                if len(images) >= config.SANDBOX_MAX_IMAGES:
                    notes.append(
                        f"单次生成图片超过上限 ({config.SANDBOX_MAX_IMAGES} 张)，已跳过：{basename}，需要时请重新生成"
                    )
                    continue
                try:
                    data = bytes(sb.filesystem.read_bytes(f"/work/outputs/{norm_rel}"))
                except Exception as e:
                    notes.append(f"下载图片 {basename} 失败: {_safe_text(e)}")
                    continue
                try:
                    with Image.open(io.BytesIO(data)) as img:
                        w, h = img.size
                        if w * h > _MAX_IMAGE_PIXELS:
                            notes.append(
                                f"图片 {basename} 像素 ({w}x{h}={w * h}) 超过 40M 上限，已跳过：{basename}，需要时请重新生成"
                            )
                            continue
                except Exception as e:
                    notes.append(f"图片 {basename} 解析校验失败，已跳过: {_safe_text(e)}")
                    continue
                images.append(SandboxImage(name=basename, data=data))
            return OutputsResult(images=images, other_files=other_files, notes=notes)

    # ── 中止与关闭（不取 _lock）──────────────────────────────────────────────

    def _soft_interrupt(self, sb: Any) -> bool:
        """软杀：只杀命令包装进程树，保留 /work；随后做健康检查。任何一步失败 / 超时都返回 False（由调用方回退 terminate）。"""
        script = _kill_script()
        if script is None:
            return False
        try:
            if self._simple_exec(sb, ("python3", "-c", script), _KILL_TIMEOUT) != 0:
                return False
            return self._health_ok(sb)
        except Exception:
            return False

    def interrupt(self) -> bool:
        """中止沙箱里正在运行的命令。不取锁、永不抛异常；沙箱还没创建返回 False。

        MODAL_INTERRUPT_MODE=soft（默认）：只杀我们自己的命令进程，保留 /work，失败自动回退 terminate；
        terminate：直接终止沙箱（/work 丢失），下一次命令惰性重建。
        """
        try:
            sb = self._sb
            if sb is None or self._closed:
                return False
            self._interrupt_gen += 1
            mode = config.MODAL_INTERRUPT_MODE if config.MODAL_INTERRUPT_MODE in ("soft", "terminate") else "soft"
            result = "soft"
            if mode == "soft" and self._soft_interrupt(sb):
                pass
            else:
                result = "terminate" if mode == "terminate" else "terminate_fallback"
                self._dead = True
                self._terminate_quietly(sb, "stop")
            _audit("interrupt", self._user_id, self._session_id, mode=mode, result=result)
            return True
        except Exception:
            return False

    def close(self, reason: str = "close") -> None:
        """幂等关闭：不取锁；terminate(wait=False)，异常全吞；标记已关闭、清空指纹。
        被它杀掉的正在跑的命令会自己返回一个带「沙箱已被终止」note 的结果，不会卡死。"""
        try:
            sb = self._sb
            self._closed = True
            self._dead = True
            self._sb = None
            self._seen_outputs.clear()
            if sb is not None:
                self._terminate_quietly(sb, reason)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# ModalSandboxPool：上限、驱逐、熔断
# ══════════════════════════════════════════════════════════════════════════════

class _Entry:
    __slots__ = ("sandbox", "busy", "last_used")

    def __init__(self, sandbox: ModalSandbox, now: float):
        self.sandbox = sandbox
        self.busy = 0
        self.last_used = now


class ModalSandboxPool:
    """(user_id, session_id) -> ModalSandbox。池锁绝不跨网络调用持有（终止、创建、健康检查都在锁外）。"""

    def __init__(self, modal_mod: Any, *, clock: Callable[[], float] = time.monotonic):
        self._modal = modal_mod
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._closing = False
        self._app: Any = None
        self._app_lock = threading.Lock()
        self._breaker_lock = threading.Lock()
        self._fail_count = 0
        self._open_until: Optional[float] = None
        self._probing = False

    # ── 租用 ─────────────────────────────────────────────────────────────────

    @contextmanager
    def lease(self, user_id: str, session_id: str, *, create: bool = True) -> Iterator[Optional[ModalSandbox]]:
        """唯一的取用方式：租用期间条目的 busy 计数 +1，不会被驱逐。create=False 时条目不存在就交出 None。"""
        entry = self._acquire(user_id, session_id, create)
        try:
            yield entry.sandbox if entry is not None else None
        finally:
            if entry is not None:
                with self._lock:
                    entry.busy -= 1
                    entry.last_used = self._clock()

    def _acquire(self, user_id: str, session_id: str, create: bool) -> Optional[_Entry]:
        to_close: list[tuple[_Entry, str]] = []
        try:
            with self._lock:
                if self._closing:
                    raise SandboxUnavailable(_MSG_UNAVAILABLE)
                key = (user_id, session_id)
                entry = self._entries.get(key)
                if entry is None:
                    if not create:
                        return None
                    now = self._clock()
                    # 先剪掉已死且不在租用的条目，以及早该被 Modal 回收的空闲条目
                    stale_after = config.MODAL_IDLE_TIMEOUT_SECONDS + 60
                    for k in [k for k, e in self._entries.items()
                              if e.busy == 0 and (e.sandbox.dead or now - e.last_used > stale_after)]:
                        to_close.append((self._entries.pop(k), "evict"))
                    # 每用户上限：从这个用户自己的空闲条目里驱逐最久没用的
                    while sum(1 for k in self._entries if k[0] == user_id) >= config.MODAL_MAX_SANDBOXES_PER_USER:
                        victim = self._oldest_idle(lambda k: k[0] == user_id)
                        if victim is None:
                            _audit("reject", user_id, session_id, reason="cap_user")
                            raise SandboxUnavailable(_msg_user_cap(config.MODAL_MAX_SANDBOXES_PER_USER))
                        to_close.append((self._entries.pop(victim), "evict"))
                    # 全局上限：从所有用户的空闲条目里驱逐
                    while len(self._entries) >= config.MODAL_MAX_SANDBOXES_TOTAL:
                        victim = self._oldest_idle(lambda k: True)
                        if victim is None:
                            _audit("reject", user_id, session_id, reason="cap_total")
                            raise SandboxUnavailable(_MSG_TOTAL_BUSY)
                        to_close.append((self._entries.pop(victim), "evict"))
                    entry = _Entry(ModalSandbox(self, user_id, session_id), now)
                    self._entries[key] = entry
                entry.busy += 1
                entry.last_used = self._clock()
                return entry
        finally:
            for victim_entry, reason in to_close:        # 被摘掉的条目在池锁之外终止
                victim_entry.sandbox.close(reason)

    def _oldest_idle(self, match: Callable[[tuple[str, str]], bool]) -> Optional[tuple[str, str]]:
        candidates = [(e.last_used, k) for k, e in self._entries.items() if match(k) and e.busy == 0]
        return min(candidates)[1] if candidates else None

    # ── 重置、中止、关闭 ──────────────────────────────────────────────────────

    @staticmethod
    def _close_entry(entry: _Entry, reason: str) -> bool:
        had_live = entry.sandbox._sb is not None
        entry.sandbox.close(reason)
        return had_live

    def reset_session(self, user_id: str, session_id: str) -> None:
        with self._lock:
            entry = self._entries.pop((user_id, session_id), None)
        if entry is not None:
            self._close_entry(entry, "reset")

    def reset_user(self, user_id: str) -> int:
        with self._lock:
            keys = [k for k in self._entries if k[0] == user_id]
            entries = [self._entries.pop(k) for k in keys]
        return sum(1 for e in entries if self._close_entry(e, "reset_user"))

    def interrupt_session(self, user_id: str, session_id: str) -> bool:
        with self._lock:
            entry = self._entries.get((user_id, session_id))
        return entry.sandbox.interrupt() if entry is not None else False

    def shutdown(self, *, deadline_s: float = 15.0) -> None:
        """先设 closing（之后的 lease 直接拒绝），摘下全部条目并行终止，整体最多等 deadline_s 秒（超时就走）。"""
        with self._lock:
            self._closing = True
            entries = list(self._entries.values())
            self._entries.clear()
        if not entries:
            return
        latch = _Latch(len(entries))

        def closer(e: _Entry) -> None:
            try:
                self._close_entry(e, "shutdown")
            finally:
                latch.count_down()

        for e in entries:
            threading.Thread(target=closer, args=(e,), name="modal-shutdown", daemon=True).start()
        latch.wait(deadline_s)

    def snapshot(self) -> list[dict]:
        """诊断 / 测试用：哈希后的 id、状态、busy、last_used。"""
        with self._lock:
            return [
                {"u": _uhash(k[0]), "s": _shash(k[1]), "alive": e.sandbox.alive, "busy": e.busy, "last_used": e.last_used}
                for k, e in self._entries.items()
            ]

    # ── App 缓存 ─────────────────────────────────────────────────────────────

    def _lookup_app(self) -> Any:
        """App.lookup 只查一次并缓存（线程安全）；失败不缓存。"""
        with self._app_lock:
            if self._app is None:
                self._app = self._modal.App.lookup(config.MODAL_APP_NAME, create_if_missing=True)
            return self._app

    # ── 熔断 ─────────────────────────────────────────────────────────────────

    def _breaker_check(self, user_id: str, session_id: str) -> None:
        """创建前询问：开路期间立即拒绝；到期后只放行一次试探（试探期间其他创建仍被拒绝）。"""
        reject = False
        with self._breaker_lock:
            if self._open_until is not None:
                if self._clock() < self._open_until or self._probing:
                    reject = True
                else:
                    self._probing = True
        if reject:
            _audit("reject", user_id, session_id, reason="breaker")
            raise SandboxUnavailable(_MSG_UNAVAILABLE)

    def _breaker_success(self) -> None:
        with self._breaker_lock:
            was_open = self._open_until is not None
            self._fail_count = 0
            self._open_until = None
            self._probing = False
        if was_open:
            _audit("breaker_close")

    def _breaker_failure(self) -> None:
        opened = False
        with self._breaker_lock:
            self._fail_count += 1
            self._probing = False
            if self._fail_count >= _BREAKER_THRESHOLD:
                self._open_until = self._clock() + _BREAKER_OPEN_SECONDS
                opened = True
        if opened:
            _audit("breaker_open", seconds=_BREAKER_OPEN_SECONDS)


# ══════════════════════════════════════════════════════════════════════════════
# ModalRoutedSandbox：按请求上下文路由的无状态单例
# ══════════════════════════════════════════════════════════════════════════════

class ModalRoutedSandbox(BaseSandbox):
    """同时当 CompositeBackend 的 default 与 execute_python_code / SandboxOutputsMiddleware 的“沙箱”用。

    无状态：每次调用都现解析请求上下文，拿到 (user_id, session_id) 后用池的 lease() 取沙箱并转发。
    失败即关闭：没有上下文、非法 user_id、空 session_id 一律拒绝；desktop 下 owner 也拒绝（owner 永远不该走到这里，这是纵深防御），
    cloud 下 owner 是普通租户，不拒绝（config.is_cloud_mode() 在每次调用时现读）。
    deepagents 同步路径下工具跑在 LangChain 的线程池（copy_context），异步路径下 BaseSandbox.aexecute 走 asyncio.to_thread，
    两条路径都会继承 ContextVar。
    """

    @property
    def id(self) -> str:
        return "modal-routed"

    def _target(self) -> tuple[str, str]:
        ctx = get_request_context()
        reason = None
        if ctx is None:
            reason, text = "no_context", _MSG_NO_IDENTITY
        elif ctx.is_owner and not config.is_cloud_mode():       # desktop 下 owner 用本机 Docker，不该走到这里；cloud 下 owner 是普通租户
            reason, text = "owner", _MSG_OWNER_DENIED
        elif not ctx.session_id or not isinstance(ctx.session_id, str):
            reason, text = "no_session", _MSG_NO_IDENTITY
        else:
            try:
                validate_user_id(ctx.user_id)
            except ValueError:
                reason, text = "bad_user", _MSG_NO_IDENTITY
        if reason is not None:
            _audit(
                "reject",
                ctx.user_id if ctx is not None else None,
                ctx.session_id if ctx is not None else None,
                reason="context", detail=reason,
            )
            raise SandboxUnavailable(text)
        return ctx.user_id, ctx.session_id

    def execute(self, command: str, *, timeout: Optional[int] = None) -> ExecuteResponse:
        try:
            uid, sid = self._target()
            with get_pool().lease(uid, sid) as sb:
                return sb.execute(command, timeout=timeout)
        except SandboxUnavailable as e:
            return ExecuteResponse(output=f"[云端沙箱不可用] {e}", exit_code=1, truncated=False)
        except Exception as e:
            _console(f"⚠️ [modal] execute 路由异常（非致命）: {_safe_text(e)}")
            return ExecuteResponse(output=f"[云端沙箱不可用] {_MSG_UNAVAILABLE}", exit_code=1, truncated=False)

    def run_python(self, code: str, *, timeout: Optional[int] = None) -> RunResult:
        """拒绝或创建失败时抛 SandboxUnavailable（和 Docker 版一致，由 execute_python_code 转成文本）。"""
        uid, sid = self._target()
        with get_pool().lease(uid, sid) as sb:
            return sb.run_python(code, timeout=timeout)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        try:
            uid, sid = self._target()
            with get_pool().lease(uid, sid) as sb:
                return sb.upload_files(files)
        except Exception as e:
            msg = str(e) if isinstance(e, SandboxUnavailable) else _MSG_UNAVAILABLE
            return [FileUploadResponse(path=str(f[0]) if f else "", error=msg) for f in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        try:
            uid, sid = self._target()
            with get_pool().lease(uid, sid) as sb:
                return sb.download_files(paths)
        except Exception as e:
            msg = str(e) if isinstance(e, SandboxUnavailable) else _MSG_UNAVAILABLE
            return [FileDownloadResponse(path=str(p), content=None, error=msg) for p in paths]

    def collect_new_outputs(self) -> OutputsResult:
        """只扫描、不创建：拒绝、没有条目、出错一律返回空结果。"""
        try:
            uid, sid = self._target()
            with get_pool().lease(uid, sid, create=False) as sb:
                return sb.collect_new_outputs() if sb is not None else OutputsResult()
        except Exception:
            return OutputsResult()

    def interrupt(self) -> bool:
        try:
            uid, sid = self._target()
            return get_pool().interrupt_session(uid, sid)
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════════════════
# 模块级函数（chat_service 只通过它们访问，且必须用模块属性调用）
# ══════════════════════════════════════════════════════════════════════════════

_modal_override: Any = None
_pool: Optional[ModalSandboxPool] = None
_pool_lock = threading.Lock()
_routed: Optional[ModalRoutedSandbox] = None


def _modal_uses_proxy() -> bool:
    """Modal 的 SDK 会不会给 api.modal.com 走系统代理（读本机代理设置，不联网、不导入 modal）。

    Windows 上 urllib 读的是注册表里的系统代理，所以 `set HTTP_PROXY` 之类查不到，这里才看得到。
    """
    try:
        import urllib.request

        if not urllib.request.getproxies():
            return False
        return not urllib.request.proxy_bypass("api.modal.com")
    except Exception:
        return False


def _proxy_misconfigured(*, has_python_socks: Optional[bool] = None) -> bool:
    """系统代理会被用于 Modal，但缺少 Modal 走代理所需的 python-socks —— 这时 Modal 一定连不上。

    只读本地状态，不联网、不导入 modal。has_python_socks 只给测试用（默认现场判断）。
    """
    if has_python_socks is None:
        has_python_socks = importlib.util.find_spec("python_socks") is not None
    if has_python_socks:
        return False
    return _modal_uses_proxy()


def load_modal() -> Any:
    """返回 modal 模块（或 set_modal_module 注入的假模块）。整个文件里唯一导入 modal 的地方；导入失败就让 ImportError 往上抛。"""
    if _modal_override is not None:
        return _modal_override
    import modal
    if _proxy_misconfigured():
        _console(
            "⚠️ [modal] 系统代理已启用，但缺少 python-socks，Modal 连不上："
            "请安装 modal[api-proxy-support]，或在 .env 里设 MODAL_DISABLE_API_PROXY=1 改为直连"
        )
    return modal


def set_modal_module(mod: Any) -> None:
    """测试钩子：注入 FakeModal（传 None 恢复真实导入）。同时清空池单例、关闭并重置审计 handler。"""
    global _modal_override, _pool
    with _pool_lock:
        old, _pool = _pool, None
        _modal_override = mod
    if old is not None:
        try:
            old.shutdown(deadline_s=2.0)
        except Exception:
            pass
    _reset_audit()


def get_pool() -> ModalSandboxPool:
    """惰性单例：首次调用才经 load_modal() 导入 modal；导入失败抛 SandboxUnavailable。"""
    global _pool
    with _pool_lock:
        if _pool is None:
            try:
                modal_mod = load_modal()
            except Exception as e:
                _console(f"⚠️ [modal] 无法加载 modal 包: {_safe_text(e)}")
                raise SandboxUnavailable(_MSG_UNAVAILABLE) from None
            _pool = ModalSandboxPool(modal_mod)
        return _pool


def get_routed_sandbox() -> ModalRoutedSandbox:
    """ModalRoutedSandbox 的进程级单例；创建它既不导入 modal 也不联网。"""
    global _routed
    with _pool_lock:
        if _routed is None:
            _routed = ModalRoutedSandbox()
        return _routed


def reset_session(user_id: str, session_id: str) -> None:
    """终止这个会话的沙箱（池还没创建时是空操作，不 import modal，永不抛异常）。"""
    try:
        pool = _pool
        if pool is not None:
            pool.reset_session(user_id, session_id)
    except Exception:
        pass


def reset_user(user_id: str) -> int:
    """终止这个用户的全部沙箱，返回个数（池还没创建时返回 0，永不抛异常）。"""
    try:
        pool = _pool
        return pool.reset_user(user_id) if pool is not None else 0
    except Exception:
        return 0


_UPLOAD_ROOT = "/work/uploads/"
_MSG_UPLOAD_FAILED = "文件放入云端工作区失败，请稍后重新发送"
_MSG_UPLOAD_TOO_LARGE = "文件超过云端工作区允许的大小上限"


def upload_session_file(user_id: str, session_id: str, remote_path: str, data: bytes,
                        *, max_bytes: Optional[int] = None) -> tuple[bool, str]:
    """把一个用户上传的文件写进这个 (用户, 会话) 的沙箱（desktop 下 member 专用，owner 绝不能走到这里；cloud 下 owner 是普通租户）。
    成功 -> (True, 沙箱代标识)；失败 -> (False, 给用户看的中文短句)。永不抛异常，返回的文案不含内部细节。
    阻塞调用：调用方用 asyncio.to_thread。"""
    try:
        try:
            validate_user_id(user_id)
        except ValueError:
            return False, _MSG_UPLOAD_FAILED
        if (user_id == OWNER_USER_ID and not config.is_cloud_mode()) or not isinstance(session_id, str) or not session_id:
            return False, _MSG_UPLOAD_FAILED
        if (not isinstance(remote_path, str) or not remote_path.startswith(_UPLOAD_ROOT)
                or posixpath.normpath(remote_path) != remote_path):
            return False, _MSG_UPLOAD_FAILED
        pool = get_pool()
        with pool.lease(user_id, session_id, create=True) as sb:
            resp = sb.upload_files([(remote_path, data)], max_bytes=max_bytes)[0]
            if resp.error is None:
                gen = sb.generation(check_remote=False)
                return (True, gen) if gen is not None else (False, _MSG_UPLOAD_FAILED)
            if resp.error == "file_too_large":
                return False, _MSG_UPLOAD_TOO_LARGE
            return False, _MSG_UPLOAD_FAILED
    except SandboxUnavailable as e:
        return False, str(e)
    except Exception:
        return False, _MSG_UPLOAD_FAILED


def session_sandbox_generation(user_id: str, session_id: str, *, check_remote: bool = True) -> Optional[str]:
    """该会话当前那一代沙箱的标识；池没创建 / 没有这个会话的沙箱 / 已死 -> None。
    绝不创建沙箱、绝不 import modal（不调用 get_pool），永不抛异常。"""
    try:
        pool = _pool
        if pool is None:
            return None
        with pool.lease(user_id, session_id, create=False) as sb:
            return sb.generation(check_remote=check_remote) if sb is not None else None
    except Exception:
        return None


def interrupt_session(user_id: str, session_id: str) -> bool:
    try:
        pool = _pool
        return pool.interrupt_session(user_id, session_id) if pool is not None else False
    except Exception:
        return False


def interrupt_session_nowait(user_id: str, session_id: str) -> None:
    """/stop 用：起一个守护线程去中断，调用方立即返回（池还没创建时什么都不做）。"""
    try:
        if _pool is None:
            return
        threading.Thread(
            target=interrupt_session, args=(user_id, session_id), name="modal-interrupt", daemon=True
        ).start()
    except Exception:
        pass


def shutdown_pool() -> None:
    """终止全部沙箱（池还没创建时是空操作）。之后再用会建一个新池，所以关闭服务后重启不会被旧的 closing 状态卡住。"""
    global _pool
    try:
        with _pool_lock:
            pool, _pool = _pool, None
        if pool is not None:
            pool.shutdown()
    except Exception:
        pass


atexit.register(shutdown_pool)


# ══════════════════════════════════════════════════════════════════════════════
# 编译期门控
# ══════════════════════════════════════════════════════════════════════════════

def member_code_status() -> tuple[bool, str]:
    """member 图能不能编译进云端 CodeAgent（纯静态判断：不联网、不导入 modal，每次现读 config）。

    语义是“可以编译进图”，不是“现在一定能用”：令牌是否有效、镜像是否已发布、Modal 是否可达，
    要到第一次真的创建沙箱时才知道。返回的原因是给运营者看的中文短句，不带任何令牌值。
    """
    if config.MODAL_DISABLED:
        return False, "MODAL_DISABLED=true"
    if not config.MODAL_TOKENS_PRESENT:
        return False, "未配置 MODAL_TOKEN_ID / MODAL_TOKEN_SECRET"
    try:
        spec = importlib.util.find_spec("modal")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return False, "未安装 modal 包"
    try:
        version = importlib.metadata.version("modal")
    except Exception:
        return False, "无法识别 modal 版本（需要 >= 1.6）"
    m = re.match(r"^(\d+)\.(\d+)", str(version))
    if m is None:
        return False, "无法识别 modal 版本（需要 >= 1.6）"          # 失败即关闭
    if (int(m.group(1)), int(m.group(2))) < (1, 6):
        return False, "modal 版本过低（需要 >= 1.6）"
    return True, ""
