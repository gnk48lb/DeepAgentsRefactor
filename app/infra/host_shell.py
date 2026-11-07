"""宿主机命令执行：运行器 + 环境变量白名单 + 免授权判定 (所在层: L1)

FileAgent 的 run_host_command 工具通过本模块在用户的宿主机上执行命令（Windows 下走 cmd.exe）。
刻意不用 deepagents 的 LocalShellBackend：它用 subprocess.run(shell=True, text=True)，
Windows 上超时只杀 cmd.exe、孙进程占着管道会让调用一直挂住；标准输入继承自机器人进程；
输出按本地代码页解码；内存里的输出没有上限。
本模块不得 import app.runtime / app.tools / app.agents / app.knowledge / app.memory / app.infra.models。
"""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from app import config

_IS_WINDOWS = os.name == "nt"

# ── 环境变量：白名单（绝不整体继承 os.environ）──────────────────────────────────
# config.py 会把 .env 里的全部内容灌进 os.environ（含 QQ_BOTS 里的 token/secret、各种 API Key、数据库密码），
# 后缀黑名单（*_KEY / *_TOKEN ...）拦不住 QQ_BOTS 这类名字，所以只放行下面这些。
_ENV_ALLOW = frozenset({
    # Windows 运行时必需（缺 SYSTEMROOT 连 Python 都起不来）
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMFILES",
    "COMMONPROGRAMFILES(X86)", "COMMONPROGRAMW6432", "ALLUSERSPROFILE", "PUBLIC",
    "USERNAME", "USERDOMAIN", "COMPUTERNAME", "OS", "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL", "PROCESSOR_REVISION", "NUMBER_OF_PROCESSORS",
    "LOGONSERVER", "SESSIONNAME", "PSMODULEPATH", "DRIVERDATA",
    # POSIX（CI / 开发机）
    "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
    # 代理（git / pip / npm 常用）
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    # 常见工具链安装目录（非敏感）
    "JAVA_HOME", "GOROOT", "GOPATH", "CARGO_HOME", "RUSTUP_HOME", "DOTNET_ROOT",
    "NVM_HOME", "NVM_SYMLINK", "PNPM_HOME",
})

# 强制注入：让子进程输出稳定、不阻塞在交互提示上
_ENV_FORCED = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
    "NO_COLOR": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    # Windows：cmd.exe 不再先在当前目录里找可执行文件，防止工作区里的 git.bat / python.cmd 抢在真正的 git / python 之前运行
    "NoDefaultCurrentDirectoryInExePath": "1",
}


def build_safe_env(
    environ: Mapping[str, str] | None = None,
    extra_allow: Iterable[str] | None = None,
) -> dict[str, str]:
    """从 environ（默认 os.environ）里只挑白名单变量，再叠加强制注入项。名字比较不区分大小写。"""
    src = os.environ if environ is None else environ
    extra = config.HOST_SHELL_ENV_ALLOW if extra_allow is None else extra_allow
    allow = set(_ENV_ALLOW) | {str(n).strip().upper() for n in extra if str(n).strip()}
    env = {k: v for k, v in src.items() if k.upper() in allow}
    env.update(_ENV_FORCED)
    return env


# ── 输出解码与截断 ────────────────────────────────────────────────────────────

def _fallback_encoding() -> str:
    """非 UTF-8 输出的回退编码：Windows 用 OEM 代码页（控制台程序按它输出），其余平台用本地编码。"""
    if _IS_WINDOWS:
        try:
            import ctypes
            return f"cp{ctypes.windll.kernel32.GetOEMCP()}"
        except Exception:
            pass
    import locale
    return locale.getpreferredencoding(False) or "utf-8"


def decode_output(data: bytes, fallback: str | None = None) -> str:
    """先严格按 UTF-8 解码（子进程里的 Python 被强制 UTF-8，git 也输出 UTF-8），失败再回退到本地代码页（cmd 内建命令）。"""
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode(fallback or _fallback_encoding(), errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


def _head_tail(text: str, limit: int) -> tuple[str, bool]:
    """超长时保留开头 60% + 结尾 40%（报错信息通常在结尾）。"""
    if limit < 10 or len(text) <= limit:
        return text, False
    head_n = int(limit * 0.6)
    tail_n = limit - head_n
    omitted = len(text) - head_n - tail_n
    return f"{text[:head_n]}\n\n... [已省略中间 {omitted} 字符] ...\n\n{text[-tail_n:]}", True


# ── 免授权判定：整条命令精确匹配 ────────────────────────────────────────────────
# 不解析 cmd 语法（元字符黑名单永远补不全）。整条命令（空白折叠、忽略大小写）必须与
# config.HOST_SHELL_AUTO_APPROVE_COMMANDS 里的某一项完全一致才免授权，其余一律走人工授权。
_UNSAFE_ENTRY_CHARS = frozenset('&|<>^%!;`$()"\'')


def _normalize_command(command: Any) -> str | None:
    if not isinstance(command, str):
        return None
    if any(not (0x20 <= ord(c) <= 0x7E) for c in command):  # 换行 / 制表符 / 非 ASCII 一律不免授权
        return None
    return " ".join(command.split()).casefold()


def _auto_approve_set() -> set[str]:
    out: set[str] = set()
    for entry in config.HOST_SHELL_AUTO_APPROVE_COMMANDS:
        norm = _normalize_command(entry)
        if norm and not (set(norm) & _UNSAFE_ENTRY_CHARS):  # 配置项里带元字符的直接忽略，防手滑
            out.add(norm)
    return out


def is_auto_approved_command(command: Any) -> bool:
    norm = _normalize_command(command)
    return bool(norm) and norm in _auto_approve_set()


# ── 运行器 ───────────────────────────────────────────────────────────────────

@dataclass
class HostCommandResult:
    output: str
    exit_code: int | None
    duration: float = 0.0
    timed_out: bool = False
    truncated: bool = False
    notes: list[str] = field(default_factory=list)


def format_result(res: HostCommandResult) -> str:
    """给模型看的最终文本：输出 + 提示 + 退出码行。"""
    lines = [res.output if res.output else "(无输出)"]
    lines += [f"[提示] {n}" for n in res.notes]
    code = "?" if res.exit_code is None else res.exit_code
    lines.append(f"[exit_code={code} duration={res.duration:.1f}s]")
    return "\n".join(lines)


def _kill_tree(proc: subprocess.Popen) -> None:
    """终止 shell 及其全部后代（Windows: taskkill /T；POSIX: 杀整个进程组）。"""
    try:
        if _IS_WINDOWS:
            root = os.environ.get("SystemRoot") or r"C:\Windows"
            subprocess.run(
                [os.path.join(root, "System32", "taskkill.exe"), "/F", "/T", "/PID", str(proc.pid)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=10, check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _spawn(command: str, cwd: str, env: dict[str, str]) -> subprocess.Popen:
    kwargs: dict[str, Any] = dict(
        stdin=subprocess.DEVNULL,       # 不继承机器人的标准输入（CLI 模式下会抢走控制台输入）
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,       # 合并输出，保持先后顺序
        cwd=cwd,
        env=env,
        bufsize=0,
    )
    if _IS_WINDOWS:
        comspec = os.environ.get("COMSPEC") or os.path.join(
            os.environ.get("SystemRoot") or r"C:\Windows", "System32", "cmd.exe"
        )
        # 字符串形式 + shell=False：原样交给 CreateProcess；/s 让 cmd 只剥掉最外层一对引号，命令里的引号不被改写；/d 忽略 AutoRun
        cmdline = f'"{comspec}" /d /s /c "{command}"'
        return subprocess.Popen(
            cmdline,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            **kwargs,
        )
    return subprocess.Popen(["/bin/sh", "-c", command], start_new_session=True, **kwargs)


def _audit(command: str, exit_code: int | None, duration: float) -> None:
    """追加一行审计日志（storage/host_shell_audit.log）；任何失败都不影响命令结果。"""
    try:
        os.makedirs(config.STORAGE_DIR, exist_ok=True)
        path = os.path.join(config.STORAGE_DIR, "host_shell_audit.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\texit={exit_code}\t{duration:.1f}s\t{command[:500]!r}\n")
    except Exception:
        pass


def run_command(
    command: str,
    *,
    cwd: Any,
    timeout: int | None = None,
    env: Mapping[str, str] | None = None,
    max_output_chars: int | None = None,
    hard_cap_bytes: int | None = None,
) -> HostCommandResult:
    """在 cwd 下执行一条命令；永不抛异常（所有失败都变成 HostCommandResult）。"""
    if not isinstance(command, str) or not command.strip():
        return HostCommandResult("Error: 命令不能为空", 1)
    if len(command) > config.HOST_SHELL_MAX_COMMAND_CHARS:
        return HostCommandResult(
            f"Error: 命令长 {len(command)} 字符，超过上限 {config.HOST_SHELL_MAX_COMMAND_CHARS}；"
            "请先用 write_file 把脚本写成文件，再用短命令运行", 1)
    if "\x00" in command:
        return HostCommandResult("Error: 命令包含非法字符 NUL", 1)
    if _IS_WINDOWS and ("\n" in command or "\r" in command):
        return HostCommandResult("Error: cmd.exe 不支持多行命令，请用 && 串联，或写成 .bat / .py 文件后运行", 1)

    eff_timeout = (
        config.HOST_SHELL_TIMEOUT_SECONDS
        if not timeout or timeout <= 0
        else min(int(timeout), config.HOST_SHELL_MAX_TIMEOUT_SECONDS)
    )
    max_chars = config.HOST_SHELL_MAX_OUTPUT_CHARS if max_output_chars is None else max_output_chars
    cap = config.HOST_SHELL_HARD_CAP_BYTES if hard_cap_bytes is None else hard_cap_bytes

    start = time.monotonic()
    try:
        proc = _spawn(command, str(cwd), dict(env) if env is not None else build_safe_env())
    except Exception as e:
        return HostCommandResult(f"Error: 启动命令失败 ({type(e).__name__}): {e}", 1)

    buf = bytearray()
    overflow = threading.Event()

    def _pump() -> None:
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                room = cap - len(buf)
                if len(chunk) > room:       # 超过内存硬上限：丢弃多余部分并终止进程，避免输出洪泛撑爆内存
                    if room > 0:
                        buf.extend(chunk[:room])
                    overflow.set()
                    _kill_tree(proc)
                    break
                buf.extend(chunk)
        except Exception:
            pass

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()

    timed_out = False
    try:
        proc.wait(timeout=eff_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    # shell 已退出但后台孙进程仍占着输出管道时，不无限等：宽限一小会儿就用已收集到的内容
    reader.join(timeout=5.0 if (timed_out or overflow.is_set()) else 2.0)
    pipe_held = reader.is_alive()
    if not pipe_held:  # 读线程已结束才关管道；仍被后台孙进程占着时交给 GC，避免跨线程关闭阻塞中的读
        try:
            proc.stdout.close()
        except Exception:
            pass

    duration = time.monotonic() - start
    text, truncated = _head_tail(decode_output(bytes(buf)).replace("\r\n", "\n"), max_chars)
    exit_code = 124 if timed_out else proc.returncode

    notes: list[str] = []
    if timed_out:
        notes.append(f"命令超过 {eff_timeout} 秒未结束，已终止整个进程树")
    if overflow.is_set():
        notes.append(f"输出超过 {cap} 字节的内存上限，进程已被终止")
    if pipe_held and not timed_out and not overflow.is_set():
        notes.append("有后台子进程仍占用输出管道，已不再等待其后续输出")

    _audit(command, exit_code, duration)
    return HostCommandResult(text, exit_code, duration, timed_out, truncated or overflow.is_set(), notes)
