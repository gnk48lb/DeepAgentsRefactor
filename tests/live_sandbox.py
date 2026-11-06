"""
tests/live_sandbox.py

在线（Live）Docker 沙箱验收测试：
验证常驻容器生命周期、字体渲染、只读 workspace、tmpfs 限额、超时与安全隔离、
命令行工具（rg/jq/zip/file/tree/procps）、BaseSandbox 核心命令（read/write/edit/ls/glob/grep）、
bash 复杂语法、execute 与中间件产物投递去重、以及端到端 CodeAgent 工作台执行链路。
默认跳过（除非传入命令行参数 --live 或设置环境变量 SANDBOX_LIVE_TEST=1）。
慢测试（如 95s sleep）需额外传入 --slow 参数。
需要本地 Docker 守护进程与已构建的 python-ds:latest 镜像。
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# 保证标准输出统一 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

import docker
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import config
from app.agents.experts import code_agent
from app.agents.experts.code_agent import SandboxOutputsMiddleware
from app.infra.sandbox import (
    SandboxUnavailable,
    get_sandbox,
    reset_sandbox,
    set_docker_client,
    shutdown_sandbox,
)
from app.runtime import outbox


def run_live_tests(include_slow: bool = False):
    client = docker.from_env()

    # 1. 镜像构建与标签验证
    print(f"\n[Case 1] 验证镜像 python-ds:latest 标签与元数据...")
    img = client.images.get(config.DOCKER_IMAGE_NAME)
    assert img.labels.get("gnk48.sandbox.version") == config.SANDBOX_IMAGE_VERSION, (
        f"镜像版本标签应为 {config.SANDBOX_IMAGE_VERSION}"
    )
    print(f"  [PASS] 镜像存在且包含 gnk48.sandbox.version={config.SANDBOX_IMAGE_VERSION} 标签")

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

    # 6. 超时细分与状态保留 (P0-4)
    print("\n[Case 6] 验证无限输出脚本超时强杀与状态保留...")
    res_timeout = sb.run_python("while True: print('x')", timeout=3)
    assert res_timeout.timed_out is True
    assert any("超过" in n or "超时" in n for n in res_timeout.notes)
    # 容器保留
    assert sb._container is not None

    # 恢复后下一次调用正常
    res_recovered = sb.run_python("print('back to normal')")
    assert res_recovered.exit_code == 0
    assert "back to normal" in res_recovered.stdout
    print("  [PASS] while True: print() 超时后进程终止但沙箱状态保留，后续调用恢复")

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

    # 8. 预置 Linux 命令行工具可用性验证 (A5)
    print("\n[Case 8] 验证预置命令行工具 (rg/jq/unzip/zip/file/tree/ps/pkill)...")
    tools_cmd = (
        "rg --version && jq --version && unzip -v && zip -v && "
        "file --version && tree --version && ps --version"
    )
    res_tools = sb.execute(tools_cmd)
    assert res_tools.exit_code == 0, f"工具命令执行失败: {res_tools.output}"
    print("  [PASS] rg/jq/unzip/zip/file/tree/procps 预置工具全部正常运行")

    # 9. BaseSandbox 核心文件操作链 (write -> read -> edit -> ls -> glob -> grep)
    print("\n[Case 9] 验证 BaseSandbox 完整文件工具链 (含单引号、双引号与中文)...")
    content_test = "首行：'single' \"double\"\n次行：中文数据统计分析\n末行：123456"
    sb.write("/work/test_doc.txt", content_test)

    r1 = sb.read("/work/test_doc.txt")
    assert r1.file_data is not None and "中文数据统计分析" in r1.file_data["content"]

    sb.edit("/work/test_doc.txt", old_string="123456", new_string="999888")
    r2 = sb.read("/work/test_doc.txt")
    assert "999888" in r2.file_data["content"]

    ls_r = sb.ls("/work")
    assert any("test_doc.txt" in e["path"] for e in ls_r.entries)

    glob_r = sb.glob("*.txt", "/work")
    assert any("test_doc.txt" in p["path"] for p in glob_r.matches)

    grep_r = sb.grep("中文数据", "/work")
    assert any("test_doc.txt" in m["path"] for m in grep_r.matches)
    print("  [PASS] write -> read -> edit -> ls -> glob -> grep 全流程通过，特殊引号与中文正常")

    # 10. execute bash 复杂语法与引号管道 (P0-1)
    print("\n[Case 10] 验证 execute 的 bash 高级语法与引号管道...")
    bash_script = 'if [[ 2 -gt 1 ]]; then echo {X,Y}_{1,2}; fi && echo "$((5 * 4))" | awk \'{print $1 / 2}\''
    res_bash = sb.execute(bash_script)
    assert res_bash.exit_code == 0
    assert "X_1 X_2 Y_1 Y_2" in res_bash.output
    assert "10" in res_bash.output
    print("  [PASS] bash [[ ]] 语法、花括号展开、算术展开与 awk 嵌套引号执行正常")

    # 11. execute 超时保留状态与容器保持 (P0-4)
    print("\n[Case 11] 验证 execute 超时保留状态与容器 ID 保持...")
    sb.write("/work/state_file.txt", "persisted_value")
    cid_before = sb.id
    res_sleep = sb.execute("sleep 20", timeout=3)
    assert res_sleep.exit_code == 124
    assert "沙箱状态保留" in res_sleep.output
    assert sb.id == cid_before, "超时后容器 ID 不应改变"
    assert sb.read("/work/state_file.txt").file_data["content"] == "persisted_value"
    print("  [PASS] execute 超时返回 124，容器未被重建，/work 状态完好保留")

    # 12. execute 绘图产物自动扫描、指纹去重与多模态改写 (A3)
    print("\n[Case 12] 验证 SandboxOutputsMiddleware 产物投递与指纹去重...")
    mw = SandboxOutputsMiddleware(sb)
    # 用 execute 运行一段绘图脚本
    plot_cmd = (
        'python3 -c "'
        'import matplotlib.pyplot as plt; '
        'fig, ax = plt.subplots(); '
        'ax.plot([1, 2], [10, 20]); '
        'fig.savefig(\'/work/outputs/middleware_chart.png\'); '
        'plt.close(fig)"'
    )
    sb.execute(plot_cmd)

    outbox.begin()
    req_exec = MagicMock()
    req_exec.tool_call = {"name": "execute", "args": {"command": plot_cmd}, "id": "tc_exec_1"}
    msg_raw = ToolMessage(content="script output", tool_call_id="tc_exec_1", artifact={"exit_code": 0})
    rewritten = mw.wrap_tool_call(req_exec, lambda r: msg_raw)

    assert len(outbox.current()) == 1, "应向用户推送 1 张图片"
    assert isinstance(rewritten.content, list)
    assert "已向用户发送 1 张图片" in rewritten.content[0]["text"]
    assert rewritten.content[1]["type"] == "image_url"
    assert rewritten.artifact == {"exit_code": 0}

    # 再次调用同指纹不重复投递
    outbox.begin()
    rewritten_dup = mw.wrap_tool_call(req_exec, lambda r: msg_raw)
    assert len(outbox.current()) == 0, "同一指纹不应重复推送"

    # 改写图片后再次触发应重新投递
    plot_cmd_v2 = (
        'python3 -c "'
        'import matplotlib.pyplot as plt; '
        'fig, ax = plt.subplots(); '
        'ax.plot([1, 2, 3], [5, 15, 25]); '
        'fig.savefig(\'/work/outputs/middleware_chart.png\'); '
        'plt.close(fig)"'
    )
    sb.execute(plot_cmd_v2)
    outbox.begin()
    rewritten_v2 = mw.wrap_tool_call(req_exec, lambda r: msg_raw)
    assert len(outbox.current()) == 1, "内容修改后应识别为新版本并重新投递"
    print("  [PASS] execute 产物由中间件自动捕获，指纹去重生效，改写后正确重新投递")

    # 13. write 到 /workspace 拦截与自解释错误 (A7)
    print("\n[Case 13] 验证向 /workspace 写入时返回自解释错误...")
    up_err = sb.upload_files([("/workspace/test.txt", b"data")])
    assert "permission_denied" in up_err[0].error
    assert "只读挂载" in up_err[0].error
    print(f"  [PASS] 工作区写入被拦截并返回自解释错误: {up_err[0].error}")

    # 14. 超大输出 (20万行) 洪泛不崩溃 (P0-3)
    print("\n[Case 14] 验证超大命令输出洪泛不崩溃...")
    t_start = time.time()
    res_flood = sb.execute("seq 1 100000")
    t_cost = time.time() - t_start
    assert res_flood.exit_code == 0
    assert len(res_flood.output) > 500000
    print(f"  [PASS] 10万行输出处理耗时: {t_cost:.2f}s，输出长度: {len(res_flood.output)} 字符，无内存崩溃")

    # 15. 端到端工作台链路 (ScriptedModel + CodeAgent Runnable)
    print("\n[Case 15] 验证 CodeAgent 端到端工作台链路 (write -> execute -> edit -> execute -> read)...")
    class WorkbenchModel:
        def __init__(self):
            self.step = 0
            self.profile = None
        def bind_tools(self, tools, **kwargs):
            return self
        def invoke(self, messages, **kwargs):
            self.step += 1
            if self.step == 1:
                return AIMessage(content="", tool_calls=[{"name": "write_file", "args": {"file_path": "/work/task.py", "content": "print('step1_ok')\n"}, "id": "c1"}])
            elif self.step == 2:
                return AIMessage(content="", tool_calls=[{"name": "execute", "args": {"command": "python3 /work/task.py"}, "id": "c2"}])
            elif self.step == 3:
                return AIMessage(content="", tool_calls=[{"name": "edit_file", "args": {"file_path": "/work/task.py", "old_string": "step1_ok", "new_string": "step2_all_passed"}, "id": "c3"}])
            elif self.step == 4:
                return AIMessage(content="", tool_calls=[{"name": "execute", "args": {"command": "python3 /work/task.py"}, "id": "c4"}])
            elif self.step == 5:
                return AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": "/work/task.py"}, "id": "c5"}])
            else:
                return AIMessage(content="【内部专家汇报 - 来源：CodeAgent】端到端工作台自动化链路全部顺利完成")

    agent_runnable = code_agent.build_code_agent_runnable(
        llm=WorkbenchModel(),
        system_prompt=code_agent.PROMPT,
        sandbox_backend=sb,
    )
    res_e2e = agent_runnable.invoke({"messages": [HumanMessage(content="执行工作台管道任务")]})
    assert "端到端工作台自动化链路全部顺利完成" in str(res_e2e["messages"][-1].content)
    print("  [PASS] CodeAgent 端到端工作台链路全流程执行通过")

    # 16. 重置清理与标签过滤器
    print("\n[Case 16] 验证重置后容器清理与标签过滤...")
    reset_sandbox()
    active = client.containers.list(all=True, filters={"label": "gnk48.sandbox=1"})
    assert len(active) == 0, f"reset 后遗留容器列表应为空，实际发现: {active}"
    print("  [PASS] reset_sandbox 后所有 gnk48.sandbox=1 容器已全部销毁清理")

    # 17. 旧镜像校验拦截与指引
    print("\n[Case 17] 验证旧镜像检测与重新构建指引...")
    class FakeOldImage:
        labels = {"gnk48.sandbox.version": "1"}

    class MockDockerClient:
        def __init__(self, real):
            self._real = real
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

    # 18. (可选慢测试) 超长执行读超时不误杀
    if include_slow:
        print("\n[Case 18 (慢测试)] 验证 95 秒长时间命令执行不被 HTTP 读超时误杀...")
        t_slow_start = time.time()
        res_slow = sb.execute("sleep 95 && echo 'slow_success'", timeout=120)
        t_slow_cost = time.time() - t_slow_start
        assert res_slow.exit_code == 0
        assert "slow_success" in res_slow.output
        print(f"  [PASS] 95s 命令顺利执行完成 (耗时: {t_slow_cost:.2f}s)，客户端 150s 动态超时未触发误杀")
    else:
        print("\n[Case 18 (慢测试)] 跳过（需传入 --slow 参数启用）")


def main():
    is_live = "--live" in sys.argv or os.getenv("SANDBOX_LIVE_TEST") == "1"
    include_slow = "--slow" in sys.argv
    if not is_live:
        print("ℹ️ 未指定 --live 参数或 SANDBOX_LIVE_TEST 环境变量，默认跳过在线测试。")
        print("运行方法: uv run python tests/live_sandbox.py --live [--slow]")
        return

    print("=" * 70)
    print("开始运行 Docker 沙箱真实环境 Live 验收测试...")
    print("=" * 70)
    try:
        run_live_tests(include_slow=include_slow)
        print("=" * 70)
        print("Docker 沙箱 Live 验收测试全部通过！")
        print("=" * 70)
    finally:
        shutdown_sandbox()


if __name__ == "__main__":
    main()
