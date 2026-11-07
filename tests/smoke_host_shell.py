"""
tests/smoke_host_shell.py

宿主机命令执行（app/infra/host_shell.py）离线冒烟测试：无需 API Key / 网络 / Docker。
  H1 build_safe_env 白名单   H2 decode_output   H3 is_auto_approved_command   H4 基础运行
  H5 子进程环境与编码        H6 超时杀进程树    H7 脱离的孙进程不挂住
  H8 截断与洪泛              H9 审计日志        H10 Windows 专项（其他平台跳过）
"""
import os
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app import config
from app.infra import host_shell

PY = f'"{sys.executable}"'
IS_WIN = os.name == "nt"


def _script(td, name, body):
    path = Path(td) / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_h1_safe_env():
    fake = {
        "PATH": "/bin", "SYSTEMROOT": "C:\\Windows", "HOME": "/home/x", "http_proxy": "http://p", "JAVA_HOME": "/jdk",
        "GEMINI_API_KEY": "k", "MYSQL_PASSWORD": "p", "QQ_BOTS": "[{}]", "FAKE_API_KEY": "k",
        "HF_TOKEN": "t", "NEO4J_URI": "bolt://x", "MY_TOOL_HOME": "/t",
    }
    env = host_shell.build_safe_env(fake, extra_allow=())
    lower = {k.lower() for k in env}
    for k in ("PATH", "SYSTEMROOT", "HOME", "http_proxy", "JAVA_HOME"):
        assert k.lower() in lower, f"应保留 {k}"
    for k in ("GEMINI_API_KEY", "MYSQL_PASSWORD", "QQ_BOTS", "FAKE_API_KEY", "HF_TOKEN", "NEO4J_URI", "MY_TOOL_HOME"):
        assert k not in env, f"不应放行 {k}"
    assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["NoDefaultCurrentDirectoryInExePath"] == "1"
    env2 = host_shell.build_safe_env(fake, extra_allow=("my_tool_home",))
    assert env2.get("MY_TOOL_HOME") == "/t"


def test_h2_decode_output():
    assert host_shell.decode_output(b"") == ""
    s = "中文 emoji ✅ 😀"
    assert host_shell.decode_output(s.encode("utf-8")) == s
    assert host_shell.decode_output("中文".encode("gbk"), fallback="gbk") == "中文"
    assert isinstance(host_shell.decode_output(b"\xff\xfe\xfa bad"), str)
    assert isinstance(host_shell.decode_output(b"\xff\xfe\xfa", fallback="no-such-codec"), str)


def test_h3_auto_approve():
    ok = host_shell.is_auto_approved_command
    for c in ("git status", "  GIT   status ", "git  log --oneline -n 10", "whoami", "python --version"):
        assert ok(c), f"应免授权: {c!r}"
    bad = [
        "git status\necho x", "git status\r\necho x", "whoami & calc", "whoami && calc", "whoami | more",
        "whoami > x.txt", "whoami ^& calc", "echo %PATH%", 'git status "x"', "git branch -D x",
        "git log --output=x", "git diff --no-index \\a \\b", "git\tstatus", "git status；calc",
        "git status\x00", "", "   ", None, 5, ["git status"],
    ]
    for c in bad:
        assert not ok(c), f"不应免授权: {c!r}"
    original = config.HOST_SHELL_AUTO_APPROVE_COMMANDS
    try:
        config.HOST_SHELL_AUTO_APPROVE_COMMANDS = ("echo hi", "git status & calc", "echo $HOME")
        assert ok("echo hi")
        assert not ok("git status & calc") and not ok("echo $HOME")
        assert not ok("git status") and not ok("whoami")
    finally:
        config.HOST_SHELL_AUTO_APPROVE_COMMANDS = original


def test_h4_basic_run():
    with tempfile.TemporaryDirectory() as td:
        r = host_shell.run_command("echo hello", cwd=td)
        assert r.exit_code == 0 and "hello" in r.output, r
        assert "[exit_code=0" in host_shell.format_result(r)

        r = host_shell.run_command(f'{PY} -c "import sys; sys.exit(3)"', cwd=td)
        assert r.exit_code == 3, r

        s = _script(td, "cwd.py", "import os\nprint(os.getcwd())\n")
        r = host_shell.run_command(f'{PY} "{s}"', cwd=td)
        assert r.exit_code == 0 and os.path.samefile(r.output.strip(), td), r.output

        r = host_shell.run_command("definitely_not_a_real_command_xyz", cwd=td)
        assert r.exit_code != 0, r

        assert host_shell.run_command("", cwd=td).exit_code == 1
        long_cmd = "echo " + "a" * (config.HOST_SHELL_MAX_COMMAND_CHARS + 1)
        r = host_shell.run_command(long_cmd, cwd=td)
        assert "超过上限" in r.output, r.output

        r = host_shell.run_command("echo fallback_ok", cwd=td, timeout=-5)
        assert r.exit_code == 0 and "fallback_ok" in r.output, r


def test_h5_env_encoding_stdin():
    with tempfile.TemporaryDirectory() as td:
        s = _script(td, "env.py", """
            import os, sys
            print("KEY=" + repr(os.environ.get("SMOKE_FAKE_API_KEY")))
            print("QQ=" + repr(os.environ.get("QQ_BOTS")))
            print("UTF8=" + repr(os.environ.get("PYTHONUTF8")))
            print("中文 ✅ 😀")
            print("STDIN=" + repr(sys.stdin.read()))
        """)
        os.environ["SMOKE_FAKE_API_KEY"] = "secret-key"
        os.environ["QQ_BOTS"] = "secret-bots"
        try:
            t0 = time.monotonic()
            r = host_shell.run_command(f'{PY} "{s}"', cwd=td)
            elapsed = time.monotonic() - t0
        finally:
            os.environ.pop("SMOKE_FAKE_API_KEY", None)
            os.environ.pop("QQ_BOTS", None)
        assert r.exit_code == 0, r.output
        assert "KEY=None" in r.output and "QQ=None" in r.output, r.output
        assert "secret" not in r.output, r.output
        assert "UTF8='1'" in r.output, r.output
        assert "中文 ✅ 😀" in r.output, r.output
        assert "STDIN=''" in r.output, r.output
        assert elapsed < 20, f"stdin 未关闭导致挂住: {elapsed:.1f}s"


def test_h6_timeout_kills_tree():
    with tempfile.TemporaryDirectory() as td:
        s = _script(td, "parent.py", """
            import subprocess, sys, time
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            print("spawned", flush=True)
            time.sleep(60)
        """)
        t0 = time.monotonic()
        r = host_shell.run_command(f'{PY} "{s}"', cwd=td, timeout=2)
        elapsed = time.monotonic() - t0
        assert r.timed_out and r.exit_code == 124, r
        assert elapsed < 15, f"超时处理耗时 {elapsed:.1f}s"
        assert "spawned" in r.output, r.output
        assert any("进程树" in n for n in r.notes), r.notes


def test_h7_detached_grandchild_does_not_hang():
    with tempfile.TemporaryDirectory() as td:
        s = _script(td, "detach.py", f"""
            import subprocess, sys, tempfile
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(6)"], cwd=tempfile.gettempdir())
            print("done", flush=True)
        """)
        t0 = time.monotonic()
        r = host_shell.run_command(f'{PY} "{s}"', cwd=td)
        elapsed = time.monotonic() - t0
        assert r.exit_code == 0 and "done" in r.output, r
        assert elapsed < 5, f"被后台孙进程拖住 {elapsed:.1f}s"


def test_h8_truncate_and_flood():
    with tempfile.TemporaryDirectory() as td:
        s = _script(td, "big.py", """
            print("HEAD_MARK")
            for i in range(20000):
                print("line", i)
            print("TAIL_MARK")
        """)
        r = host_shell.run_command(f'{PY} "{s}"', cwd=td, max_output_chars=2000)
        assert r.truncated and "HEAD_MARK" in r.output and "TAIL_MARK" in r.output, r.output[:200]
        assert "已省略中间" in r.output and len(r.output) < 2200, len(r.output)

        f = _script(td, "flood.py", """
            import sys
            chunk = "x" * 1000 + "\\n"
            while True:
                sys.stdout.write(chunk)
        """)
        t0 = time.monotonic()
        r = host_shell.run_command(f'{PY} "{f}"', cwd=td, timeout=30, hard_cap_bytes=1_000_000)
        elapsed = time.monotonic() - t0
        assert r.truncated and any("内存上限" in n for n in r.notes), r.notes
        assert elapsed < 20, f"洪泛处理耗时 {elapsed:.1f}s"


def test_h9_audit_log():
    original = config.STORAGE_DIR
    with tempfile.TemporaryDirectory() as td:
        try:
            config.STORAGE_DIR = os.path.join(td, "storage")
            host_shell.run_command("echo audit_marker", cwd=td)
            log = Path(config.STORAGE_DIR, "host_shell_audit.log").read_text(encoding="utf-8")
        finally:
            config.STORAGE_DIR = original
        assert "echo audit_marker" in log and "exit=0" in log, log


def test_h10_windows_only():
    if not IS_WIN:
        print("    跳过: H10 仅在 Windows 上有意义")
        return
    with tempfile.TemporaryDirectory() as td:
        marker = Path(td) / "bat_ran.txt"
        (Path(td) / "whoami.bat").write_text(f"@echo off\r\necho x> \"{marker}\"\r\n", encoding="utf-8")
        host_shell.run_command("whoami", cwd=td)
        assert not marker.exists(), "NoDefaultCurrentDirectoryInExePath 未生效：工作区里的 whoami.bat 被执行了"

        r = host_shell.run_command("echo a\necho b", cwd=td)
        assert "不支持多行" in r.output, r.output

        r = host_shell.run_command(f'{PY} -c "print(\'a b\')"', cwd=td)
        assert "a b" in r.output, r.output

        r = host_shell.run_command("ver", cwd=td)
        assert r.output.strip(), r


SCENARIOS = [
    ("H1 build_safe_env 白名单", test_h1_safe_env),
    ("H2 decode_output", test_h2_decode_output),
    ("H3 is_auto_approved_command 精确匹配", test_h3_auto_approve),
    ("H4 基础运行", test_h4_basic_run),
    ("H5 子进程环境 / 编码 / stdin", test_h5_env_encoding_stdin),
    ("H6 超时杀进程树", test_h6_timeout_kills_tree),
    ("H7 脱离的孙进程不挂住", test_h7_detached_grandchild_does_not_hang),
    ("H8 截断与洪泛硬上限", test_h8_truncate_and_flood),
    ("H9 审计日志", test_h9_audit_log),
    ("H10 Windows 专项", test_h10_windows_only),
]


def main():
    print("=" * 70)
    print("开始运行宿主机命令执行冒烟测试...")
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
