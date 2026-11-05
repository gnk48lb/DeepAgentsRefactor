"""
tests/live_sandbox.py

在线（Live）Docker 沙箱验收测试：
验证常驻容器生命周期、字体渲染、只读 workspace、tmpfs 限额、超时与安全隔离。
默认跳过（除非传入命令行参数 --live 或设置环境变量 SANDBOX_LIVE_TEST=1）。
需要本地 Docker 守护进程与已构建的 python-ds:latest 镜像。
"""

import os
import sys
import time
from pathlib import Path

# 保证标准输出统一 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

import docker
from app import config
from app.infra.sandbox import (
    get_sandbox,
    reset_sandbox,
    shutdown_sandbox,
    SandboxUnavailable,
    set_docker_client,
)


def run_live_tests():
    client = docker.from_env()

    # 1. 镜像构建与标签验证
    print("\n[Case 1] 验证镜像 python-ds:latest 标签与元数据...")
    img = client.images.get(config.DOCKER_IMAGE_NAME)
    assert img.labels.get("gnk48.sandbox.version") == config.SANDBOX_IMAGE_VERSION, (
        f"镜像版本标签应为 {config.SANDBOX_IMAGE_VERSION}"
    )
    print("  [PASS] 镜像存在且包含 gnk48.sandbox.version=2 标签")

    # 2. 跨调用复用与热启动耗时 (< 3s)
    print("\n[Case 2] 连续调用复用与 matplotlib 热启动耗时...")
    sb = get_sandbox()
    res1 = sb.run_python("a = 1; print('init')")
    assert res1.exit_code == 0
    cid1 = sb.id

    t0 = time.time()
    res2 = sb.run_python(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2], [3, 4])\n"
        "plt.close(fig)\n"
        "print('mpl ok')\n"
    )
    duration2 = time.time() - t0
    cid2 = sb.id

    assert cid1 == cid2, "两次调用容器 ID 应一致（跨调用复用）"
    assert res2.exit_code == 0
    assert duration2 < 3.0, f"预热后 matplotlib 执行耗时 ({duration2:.2f}s) 应小于 3.0s"
    print(f"  [PASS] 容器跨调用复用成功，第二次绘图耗时: {duration2:.2f}s (< 3.0s)")

    # 3. 中文标题与负号渲染无缺字警告
    print("\n[Case 3] 验证中文与负号图表渲染...")
    chart_code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.plot([1, 2, 3], [-10.5, 0.0, 15.2], label='季度利润曲线')\n"
        "ax.set_title('2026 年季度亏损与盈利 (单位: 万元)')\n"
        "ax.set_xlabel('季度')\n"
        "ax.set_ylabel('净利润')\n"
        "ax.legend()\n"
        "fig.savefig('/work/outputs/profit_chart.png')\n"
        "plt.close(fig)\n"
        "print('chart saved')\n"
    )
    res_chart = sb.run_python(chart_code)
    assert res_chart.exit_code == 0
    assert "Glyph" not in res_chart.stderr and "missing" not in res_chart.stderr.lower(), (
        f"stderr 出现字体缺字警告: {res_chart.stderr}"
    )
    assert "findfont" not in res_chart.stderr, f"stderr 出现 findfont 警告: {res_chart.stderr}"
    assert len(res_chart.images) == 1
    assert res_chart.images[0].name == "profit_chart.png"
    print("  [PASS] 中文与负号图表生成正常，无 Glyph missing 警告")

    # 4. 只读挂载 workspace 验证
    print("\n[Case 4] 验证用户工作区 (/workspace) 只读访问...")
    test_csv = Path(config.WORKSPACE_DIR) / "live_test_sample.csv"
    test_csv.write_text("id,val\n1,100\n2,200\n", encoding="utf-8")
    try:
        read_code = (
            "import pandas as pd\n"
            "df = pd.read_csv('/workspace/live_test_sample.csv')\n"
            "print('SUM:', df['val'].sum())\n"
        )
        res_read = sb.run_python(read_code)
        assert res_read.exit_code == 0
        assert "SUM: 300" in res_read.stdout

        # 尝试写入工作区
        write_code = "open('/workspace/forbidden.txt', 'w').write('bad')"
        res_write = sb.run_python(write_code)
        assert res_write.exit_code != 0
        assert (
            "Read-only file system" in res_write.stderr
            or "PermissionError" in res_write.stderr
        )
        print("  [PASS] 成功读取 /workspace 下 CSV 文件，向工作区写入被系统拦截 (只读生效)")
    finally:
        if test_csv.exists():
            test_csv.unlink()

    # 5. /work tmpfs 容量上限限制
    print("\n[Case 5] 验证 /work tmpfs 配额限制...")
    overflow_code = (
        "try:\n"
        "    chunk = b'0' * (1024 * 1024)\n"
        "    for i in range(4):\n"
        "        with open(f'/work/large_{i}.bin', 'wb') as f:\n"
        "            for _ in range(80):\n"
        "                f.write(chunk)\n"
        "    print('UNEXPECTED_SUCCESS')\n"
        "except OSError as e:\n"
        "    print('CAUGHT_OS_ERROR:', type(e).__name__, e.errno, str(e))\n"
        "finally:\n"
        "    import glob, os\n"
        "    for p in glob.glob('/work/large_*.bin'):\n"
        "        try: os.remove(p)\n"
        "        except Exception: pass\n"
    )
    res_overflow = sb.run_python(overflow_code)
    assert "No space left on device" in res_overflow.stdout or "OSError 28" in res_overflow.stdout
    print("  [PASS] 写入超过配额时触发 No space left on device，宿主机不受影响")

    # 6. 超时恢复与宿主有界
    print("\n[Case 6] 验证无限输出脚本超时强杀与恢复...")
    res_timeout = sb.run_python("while True: print('x')", timeout=3)
    assert res_timeout.timed_out is True
    assert any("超时" in n for n in res_timeout.notes)

    # 恢复后下一次调用正常
    res_recovered = sb.run_python("print('back to normal')")
    assert res_recovered.exit_code == 0
    assert "back to normal" in res_recovered.stdout
    print("  [PASS] while True: print() 超时后容器被强制销毁重建，后续调用恢复")

    # 7. 网络隔离与非 root 用户身份
    print("\n[Case 7] 验证无网络权限与非 root 用户身份...")
    net_code = (
        "import socket\n"
        "try:\n"
        "    s = socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
        "    print('NET_CONNECTED')\n"
        "except Exception as e:\n"
        "    print('NET_FAILED:', type(e).__name__)\n"
    )
    res_net = sb.run_python(net_code)
    assert "NET_FAILED" in res_net.stdout
    assert "NET_CONNECTED" not in res_net.stdout

    user_code = "import os; print(f'UID:{os.getuid()};GID:{os.getgid()}')"
    res_user = sb.run_python(user_code)
    assert "UID:10001;GID:10001" in res_user.stdout
    print("  [PASS] 容器断网生效 (无外网连接)，运行用户为非 root (10001:10001)")

    # 8. 重置清理与标签过滤器
    print("\n[Case 8] 验证重置后容器清理与标签过滤...")
    reset_sandbox()
    active = client.containers.list(all=True, filters={"label": "gnk48.sandbox=1"})
    assert len(active) == 0, f"reset 后遗留容器列表应为空，实际发现: {active}"
    print("  [PASS] reset_sandbox 后所有 gnk48.sandbox=1 容器已全部销毁清理")

    # 9. 旧镜像校验拦截
    print("\n[Case 9] 验证旧镜像检测与重新构建指引...")
    class FakeOldImage:
        labels = {"gnk48.sandbox.version": "1"}

    class MockDockerClient:
        def __init__(self, real):
            self._real = real
            from unittest.mock import MagicMock
            self.images = MagicMock()
            self.images.get.return_value = FakeOldImage()
        def __getattr__(self, name):
            return getattr(self._real, name)

    try:
        set_docker_client(MockDockerClient(client))
        try:
            get_sandbox().run_python("print(1)")
            assert False, "应当抛出 SandboxUnavailable"
        except SandboxUnavailable as e:
            assert "docker build -t python-ds:latest deploy/sandbox" in str(e)
            print("  [PASS] 旧镜像版本正确抛出 SandboxUnavailable 并给出修复命令")
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def main():
    is_live = "--live" in sys.argv or os.getenv("SANDBOX_LIVE_TEST") == "1"
    if not is_live:
        print("ℹ️ 未指定 --live 参数或 SANDBOX_LIVE_TEST 环境变量，默认跳过在线测试。")
        print("运行方法: uv run python tests/live_sandbox.py --live")
        return

    print("=" * 70)
    print("开始运行 Docker 沙箱真实环境 Live 验收测试...")
    print("=" * 70)
    try:
        run_live_tests()
        print("=" * 70)
        print("Docker 沙箱 Live 验收测试全部通过！")
        print("=" * 70)
    finally:
        shutdown_sandbox()


if __name__ == "__main__":
    main()
