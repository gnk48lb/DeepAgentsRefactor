"""
tests/live_modal.py

Modal 云端沙箱真机验收与实测（member 的 CodeAgent）：只由用户运行，Claude Code / CI 不得运行。
把阶段 2 里“读源码得到、没有在真实 Modal 上验证过”的几件事量出来，结果交给你决定 MODAL_INTERRUPT_MODE 等默认值：
  软杀（只杀命令进程、保留 /work）是否安全、terminate 之后读流到底静默结束还是抛异常、内存超限时只杀进程还是连沙箱一起杀、
  exec 超时到点后 wait() 的返回值、冷启动与重建耗时、网络、池护栏、审计日志、空闲回收（--slow）。

用法：
    uv run python tests/live_modal.py --live [--slow]

  没有 --live 只打印用法并以退出码 0 退出（此时不导入 modal、不联网）；--help 同样完全离线。
  --slow 开启需要等待空闲回收的那一步（要多等几分钟）。
前提：.env 里有 MODAL_TOKEN_ID / MODAL_TOKEN_SECRET，镜像已用 scripts/build_modal_image.py 发布。
会产生少量 Modal 费用；所有沙箱（含直接探 SDK 的“原始沙箱”，带 svc=gnk48 标签）都会在 finally 里终止，
漏了也可以用 scripts/manage_modal_sandboxes.py 清理。不打印令牌，也不打印环境变量的值（只打印名字）。
每个步骤打印 [Case N] 描述 与 PASS / FAIL，所有量出来的值用一行 `MEASURE key=value` 打印，结尾集中打印
`=== MEASUREMENTS ===` 块（整块复制给我）。任何步骤 FAIL 不中断其余步骤，最后汇总并以退出码 1 结束。
"""

from app import bootstrap
bootstrap.apply()

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 终止沙箱时 asyncio 会为每个被提前关掉的 TLS 连接打一条 warning（sslproto.py 的 "SSL connection is closed"），
# 一次验收要终止十几个沙箱，输出会被刷屏。这里只压这一条，其余 asyncio 警告照常显示。
logging.getLogger("asyncio").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent

from app import config
from app.infra import modal_sandbox
from app.utils.context import RequestContext, request_context
from app.utils.identity import cli_member_identity

_TOKEN_SHAPE = re.compile(r"\b(?:ak|as)-[A-Za-z0-9]{10,}")

USAGE = """\
Modal 云端沙箱真机验收（不带 --live 什么都不做）：
    uv run python tests/live_modal.py --live [--slow]
需要真实 Modal 令牌与已发布的镜像，会产生少量费用；--slow 会多等几分钟测空闲回收。
"""

measurements: dict[str, str] = {}
results: list[tuple[int, str, bool, str]] = []


def measure(key: str, value) -> None:
    measurements[key] = str(value)
    print(f"MEASURE {key}={value}")


class _Ctx:
    """真实的 member 上下文：用 cli_member_identity('livetest') 得到的 user_id。"""

    def __init__(self, session_id: str):
        self.user_id = cli_member_identity("livetest").user_id
        self.session_id = session_id

    def __enter__(self):
        self._cm = request_context(RequestContext(user_id=self.user_id, session_id=self.session_id, is_owner=False))
        self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def routed():
    return modal_sandbox.get_routed_sandbox()


def case(number: int, description: str):
    """装饰器：运行一个步骤，打印 PASS / FAIL，失败不中断其余步骤。"""
    def deco(fn):
        def run():
            print(f"\n[Case {number}] {description}")
            try:
                fn()
                print("  PASS")
                results.append((number, description, True, ""))
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                detail = modal_sandbox.explain(e, limit=300)
                print(f"  FAIL: {detail}")
                if os.environ.get("LIVE_MODAL_TRACE"):
                    traceback.print_exc()
                results.append((number, description, False, detail))
        run.number = number
        return run
    return deco


def audit_records() -> list:
    path = config.MODAL_AUDIT_LOG
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_sandbox(modal, tag_role: str):
    """直接探 SDK 用的“原始沙箱”：同样的镜像与资源护栏，timeout=300、idle_timeout=60，带 svc=gnk48 标签（漏了也能被管理脚本看到）。"""
    app = modal.App.lookup(config.MODAL_APP_NAME, create_if_missing=True)
    return modal.Sandbox.create(
        "sleep", "infinity",
        **modal_sandbox.sandbox_create_kwargs(modal, app, tags={"svc": "gnk48", "role": tag_role}, timeout=300, idle_timeout=60),
    ), app


def read_text(proc) -> str:
    return "".join(proc.stdout) + "".join(proc.stderr)


def run_steps(include_slow: bool) -> None:
    modal = modal_sandbox.load_modal()
    sessions = {"main": "live-main", "evict": ["live-e1", "live-e2", "live-e3"]}
    raw_boxes = []

    @case(1, "冷启动与热调用耗时")
    def c1():
        with _Ctx(sessions["main"]):
            t0 = time.perf_counter()
            r = routed().execute("echo hi")
            measure("cold_start_s", f"{time.perf_counter() - t0:.2f}")
            assert r.exit_code == 0 and "hi" in r.output, r
            t1 = time.perf_counter()
            routed().execute("echo hi")
            measure("warm_exec_s", f"{time.perf_counter() - t1:.2f}")

    @case(2, "退出码与输出（UTF-8、stderr）")
    def c2():
        with _Ctx(sessions["main"]):
            assert routed().execute("exit 3").exit_code == 3
            r = routed().execute("echo 你好世界; echo 错误信息 >&2")
            assert "你好世界" in r.output and "错误信息" in r.output, r.output

    @case(3, "环境核对（Python / root / 目录 / 没有任何凭证类环境变量）")
    def c3():
        with _Ctx(sessions["main"]):
            out = routed().execute("python --version; id -u; pwd; echo $HOME; touch /work/outputs/.w && rm /work/outputs/.w && echo writable").output
            assert "Python 3.13" in out and "\n0\n" in out and "/work" in out and "writable" in out, out
            names = routed().execute("env | cut -d= -f1").output.split()
            print(f"  环境变量名: {', '.join(sorted(names))}")
            # 真正要防的是凭证：名字里带这些词的变量必须一个都不存在。
            # 注意别把 MODAL_IMAGE_ID / MODAL_SANDBOX_ID / MODAL_TASK_ID / MODAL_REGION /
            # MODAL_CLOUD_PROVIDER / MODAL_CONTAINER_ARGUMENTS_PATH 这类 Modal 自己注入的运行时元数据算成凭证
            # （平台必然注入，控制不了），GPG_KEY 也是 Debian 基础镜像自带的公钥指纹。
            bad = [n for n in names if re.search(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE|API_?KEY", n, re.I)]
            assert not bad, f"沙箱里不应有凭证类环境变量: {bad}"
            assert "MODAL_TOKEN_ID" not in names and "MODAL_TOKEN_SECRET" not in names, names
            gpg = routed().execute("printenv GPG_KEY || true").output.strip()
            if gpg:
                assert re.fullmatch(r"[0-9A-Fa-f]{40}", gpg), f"GPG_KEY 应当是公钥指纹，实际: {gpg[:16]}…"

    @case(4, "文件往返与路径规则")
    def c4():
        with _Ctx(sessions["main"]):
            payload = "中文内容 hello\n".encode("utf-8")
            assert routed().upload_files([("/work/live.txt", payload)])[0].error is None
            got = routed().download_files(["/work/live.txt"])[0]
            assert got.content == payload and got.error is None, got
            denied = routed().upload_files([("/workspace/x", b"x")])[0].error
            assert denied and denied.startswith("permission_denied"), denied
            assert routed().upload_files([("/work/five.bin", b"a" * (5 * 1024 * 1024))])[0].error is None
            assert routed().upload_files([("/work/six.bin", b"a" * (5 * 1024 * 1024 + 1))])[0].error == "file_too_large"

    @case(5, "超时：封装层 124 + /work 保留；原始 exec 兜底到点的返回值")
    def c5():
        with _Ctx(sessions["main"]):
            routed().execute("echo keep > /work/keep.txt")
            t0 = time.perf_counter()
            r = routed().execute("sleep 30", timeout=5)
            measure("timeout_wall_s", f"{time.perf_counter() - t0:.1f}")
            assert r.exit_code == 124 and "命令执行超过" in r.output, r
            survives = "keep" in routed().execute("cat /work/keep.txt").output
            measure("work_survives_timeout", survives)
            assert survives
        sb, _app = raw_sandbox(modal, "live-raw-timeout")
        raw_boxes.append(sb)
        proc = sb.exec("sleep", "60", timeout=5, text=False)
        raised = False
        code = None
        try:
            for _ in proc.stdout:
                pass
            code = proc.wait()
        except Exception as e:
            raised = True
            code = f"{type(e).__name__}"
        measure("raw_exec_timeout_returncode", code)
        measure("raw_exec_timeout_raises", raised)

    @case(6, "产物：画中文与负号的图 → 1 张 PNG，第二次扫描不重发")
    def c6():
        code = (
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "ax.plot([1, 2, 3], [-1.5, 0.0, 2.5], label='示例曲线')\n"
            "ax.set_title('测试中文与负号 -1.5'); ax.legend()\n"
            "fig.savefig('/work/outputs/a.png'); plt.close(fig)\n"
        )
        with _Ctx(sessions["main"]):
            res = routed().run_python(code)
            assert res.exit_code == 0 and len(res.images) == 1 and res.images[0].name == "a.png", (res.exit_code, res.stderr[-200:], res.notes)
            assert res.images[0].data[:8] == b"\x89PNG\r\n\x1a\n"
            assert routed().collect_new_outputs().images == []

    @case(7, "内存上限：分配约 1.5 倍上限 → 看退出码、沙箱与 /work 是否还在")
    def c7():
        mb = int(config.MODAL_MEMORY_MB_LIMIT * 1.5)
        with _Ctx(sessions["main"]):
            routed().execute("echo marker > /work/oom_marker.txt")
            res = routed().run_python(f"buf = bytearray({mb} * 1024 * 1024)\nfor i in range(0, len(buf), 4096):\n    buf[i] = 1\nprint('allocated')")
            measure("oom_exit_code", res.exit_code)
            print(f"  notes: {res.notes}")
            after = routed().execute("echo alive; cat /work/oom_marker.txt")
            survives = "alive" in after.output
            measure("oom_sandbox_survives", survives)
            measure("oom_work_survives", "marker" in after.output)

    @case(8, "软杀实测：中断耗时、被中断命令的结果、之后 exec 通道是否正常（连续 3 轮）；terminate 模式与重建耗时；terminate 后读流的行为")
    def c8():
        uid = _Ctx(sessions["main"]).user_id
        results_soft = []
        with modal_sandbox.get_pool().lease(uid, "live-interrupt") as sb:
            sb.execute("echo seed > /work/seed.txt")
            for i in range(3):
                box = {}

                def worker():
                    box["r"] = sb.execute("sleep 300")

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()
                time.sleep(3)
                t0 = time.perf_counter()
                assert sb.interrupt() is True
                took = time.perf_counter() - t0
                thread.join(30)
                after = sb.execute("echo alive; cat /work/seed.txt")
                ok = "alive" in after.output and "seed" in after.output
                results_soft.append(ok)
                if i == 0:
                    measure("soft_interrupt_s", f"{took:.2f}")
                    print(f"  被中断的命令: exit={box['r'].exit_code} output={box['r'].output[-120:]!r}")
                    measure("soft_survives", ok)
            measure("soft_repeat_ok", all(results_soft))
            with_mode = config.MODAL_INTERRUPT_MODE
            config.MODAL_INTERRUPT_MODE = "terminate"
            try:
                box = {}
                thread = threading.Thread(target=lambda: box.setdefault("r", sb.execute("sleep 300")), daemon=True)
                thread.start()
                time.sleep(3)
                t0 = time.perf_counter()
                sb.interrupt()
                measure("terminate_interrupt_s", f"{time.perf_counter() - t0:.2f}")
                thread.join(30)
                t1 = time.perf_counter()
                again = sb.execute("echo rebuilt")
                measure("rebuild_s", f"{time.perf_counter() - t1:.2f}")
                print(f"  重建 note: {again.output[-160:]!r}")
            finally:
                config.MODAL_INTERRUPT_MODE = with_mode
        raw, _app = raw_sandbox(modal, "live-raw-terminate")
        raw_boxes.append(raw)
        proc = raw.exec("sleep", "60", text=False)
        seen = {}

        def reader():
            try:
                for _ in proc.stdout:
                    pass
                seen["stream"] = "silent_end"
            except Exception as e:
                seen["stream"] = f"raised:{type(e).__name__}"
            try:
                seen["wait"] = f"returned:{proc.wait()}"
            except Exception as e:
                seen["wait"] = f"raised:{type(e).__name__}"

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        time.sleep(2)
        raw.terminate()
        thread.join(30)
        measure("terminate_stream_behavior", f"stream={seen.get('stream', 'hung')};wait={seen.get('wait', 'hung')}")

    @case(9, "网络（与 MODAL_BLOCK_NETWORK 对照）与 pip")
    def c9():
        with _Ctx(sessions["main"]):
            r = routed().execute("python -c \"import urllib.request; print(urllib.request.urlopen('https://pypi.org', timeout=10).status)\"")
            reachable = r.exit_code == 0 and "200" in r.output
            measure("network", f"{'reachable' if reachable else 'blocked_or_failed'};MODAL_BLOCK_NETWORK={config.MODAL_BLOCK_NETWORK}")
            assert routed().execute("pip --version").exit_code == 0

    @case(10, "池护栏：同一用户开 3 个会话（上限 2）→ 驱逐最久空闲的")
    def c10():
        uid = _Ctx(sessions["main"]).user_id
        pool = modal_sandbox.get_pool()
        for sid in sessions["evict"]:
            with pool.lease(uid, sid) as sb:
                assert sb.execute("echo hi").exit_code == 0
            time.sleep(1)
        evicted = any(r["event"] == "terminate" and r.get("reason") == "evict" for r in audit_records())
        measure("evicted", evicted)
        assert evicted

    @case(11, "审计日志：存在、一行一 JSON、没有明文 ID、没有令牌形状")
    def c11():
        path = config.MODAL_AUDIT_LOG
        assert os.path.exists(path), path
        text = Path(path).read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        assert records and all({"ts", "event", "u", "s"} <= set(r) for r in records)
        user_id = _Ctx("x").user_id
        assert user_id not in text and "live-main" not in text and "live-e1" not in text, "审计日志里不得出现明文用户 / 会话 ID"
        assert not _TOKEN_SHAPE.search(text), "审计日志里不得出现令牌形状"
        print(f"  {len(records)} 条记录，事件: {sorted({r['event'] for r in records})}")

    if include_slow:
        @case(12, "（--slow）空闲回收：等 MODAL_IDLE_TIMEOUT_SECONDS + 30 秒后再执行")
        def c12():
            with _Ctx("live-idle"):
                routed().execute("echo gone > /work/idle.txt")
                wait = config.MODAL_IDLE_TIMEOUT_SECONDS + 30
                print(f"  等待 {wait} 秒……")
                time.sleep(wait)
                r = routed().execute("cat /work/idle.txt")
                rebuilt = "已重建" in r.output and "gone" not in r.output
                measure("idle_rebuild", rebuilt)
                assert rebuilt, r.output

    steps = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]
    if include_slow:
        steps.append(c12)
    created_before = len([r for r in audit_records() if r["event"] == "create_ok"])
    try:
        for step in steps:
            step()
    finally:
        print("\n[Case 13] 收尾：终止全部沙箱并检查没有残留")
        modal_sandbox.shutdown_pool()
        for sb in raw_boxes:
            try:
                sb.terminate()
            except Exception:
                pass
        try:
            app = modal.App.lookup(config.MODAL_APP_NAME, create_if_missing=False)
            left = [s.object_id for s in modal.Sandbox.list(app_id=app.app_id, tags={"svc": "gnk48"})]
        except Exception as e:
            left = [f"检查失败:{type(e).__name__}"]
        measure("leftover", len(left))
        if left:
            print(f"  ⚠️ 仍有残留沙箱: {left}（可用 scripts/manage_modal_sandboxes.py kill --all 清理）")
        records = audit_records()
        created = len([r for r in records if r["event"] == "create_ok"]) - created_before
        starts, minutes = {}, 0.0
        for r in records:
            key = (r["u"], r["s"])
            if r["event"] == "create_ok":
                starts[key] = r["ts"]
            elif r["event"] == "terminate" and key in starts:
                minutes += (r["ts"] - starts.pop(key)) / 60
        measure("sandboxes_created", created)
        measure("approx_sandbox_minutes", f"{minutes:.1f}")
        results.append((13, "收尾", not left, "" if not left else f"残留 {len(left)} 个"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Modal 云端沙箱真机验收与实测（只由用户运行）")
    parser.add_argument("--live", action="store_true", help="真的连接 Modal 运行（会产生少量费用）")
    parser.add_argument("--slow", action="store_true", help="多做空闲回收那一步（要多等几分钟）")
    args = parser.parse_args(argv)
    if not args.live:
        print(USAGE)
        return 0

    ok, reason = modal_sandbox.member_code_status()
    if not ok:
        print(f"❌ 现在不能做真机验收：{reason}")
        return 2
    print("开始 Modal 真机验收（会产生少量费用）……")
    try:
        run_steps(args.slow)
    except KeyboardInterrupt:
        print("\n已中断：沙箱会在收尾里终止。")
        modal_sandbox.shutdown_pool()
        return 130
    except Exception as e:
        print(f"\n❌ 真机验收中断: {modal_sandbox.explain(e, limit=500)}")
        return 1

    print("\n=== MEASUREMENTS ===")
    for key, value in measurements.items():
        print(f"{key}={value}")
    print("=== END MEASUREMENTS ===")
    failed = [(n, d, why) for n, d, ok_, why in results if not ok_]
    print(f"\n结果：{len(results) - len(failed)}/{len(results)} 步通过。")
    for n, d, why in failed:
        print(f"  FAIL [Case {n}] {d}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
