"""
tests/smoke_sandbox.py

代码沙箱离线冒烟测试：
使用手写 FakeDocker 测试 DockerSandbox 的生命周期、安全配置、流式截断、
产物过滤、BaseSandbox 契约与 code_tools 集成。
不依赖真实的 Docker 守护进程、不需要网络、不需要任何 API Key。
"""

import concurrent.futures
import io
import sys
import tarfile
import time
from typing import Any
from unittest.mock import MagicMock
from PIL import Image

# 保证标准输出统一 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend

from app import config
from app.infra.sandbox import (
    SandboxUnavailable,
    get_sandbox,
    reset_sandbox,
    shutdown_sandbox,
    set_docker_client,
)
from app.tools import code_tools
from app.runtime import outbox


# -----------------------------------------------------------------------------
# 手写 FakeDocker 系列类
# -----------------------------------------------------------------------------

class FakeImage:
    def __init__(self, labels: dict[str, str] | None = None):
        self.labels = labels if labels is not None else {"gnk48.sandbox.version": "2"}


class FakeImages:
    def __init__(self, image: FakeImage | None = None, raise_missing: bool = False):
        self._image = image if image is not None else FakeImage()
        self._raise_missing = raise_missing

    def get(self, name: str) -> FakeImage:
        if self._raise_missing:
            import docker.errors
            raise docker.errors.ImageNotFound(f"Image {name} not found")
        return self._image


class FakeContainer:
    def __init__(self, client: Any, cid: str = "fake-c-001"):
        self.client = client
        self.id = cid
        self.status = "running"
        self.killed = False
        self.removed = False
        self.exec_commands: list[str] = []
        self.archives_put: list[tuple[str, bytes]] = []
        self.mock_outputs_tar: bytes | None = None

    def reload(self) -> None:
        if self.removed:
            import docker.errors
            raise docker.errors.NotFound("Container removed")

    def kill(self) -> None:
        self.killed = True
        self.status = "exited"

    def remove(self, force: bool = False) -> None:
        self.removed = True
        self.status = "removed"

    def exec_run(self, cmd: Any, user: str = "10001:10001") -> Any:
        self.exec_commands.append(str(cmd))
        mock_res = MagicMock()
        mock_res.exit_code = 0
        if "ls -1 /work" in str(cmd):
            mock_res.output = b"main.py\noutputs\ntemp_data.csv\n"
        elif "tar -cf - -C /work/outputs ." in str(cmd) and self.mock_outputs_tar:
            mock_res.output = self.mock_outputs_tar
        else:
            mock_res.output = b""
        return mock_res

    def put_archive(self, path: str, data: bytes) -> None:
        self.archives_put.append((path, data))

    def get_archive(self, path: str) -> tuple[list[bytes], dict]:
        if path == "/work/outputs" and self.mock_outputs_tar:
            return [self.mock_outputs_tar], {"name": "outputs"}
        import docker.errors
        raise docker.errors.NotFound(f"Path {path} not found")


class FakeContainers:
    def __init__(self, client: Any):
        self.client = client
        self.created_containers: list[FakeContainer] = []
        self.run_calls: list[dict[str, Any]] = []
        self.fail_mount_once = False
        self._mount_failed = False

    def list(self, all: bool = True, filters: dict | None = None) -> list[FakeContainer]:
        return [c for c in self.created_containers if not c.removed]

    def run(self, *args, **kwargs) -> FakeContainer:
        self.run_calls.append(kwargs)
        if self.fail_mount_once and not self._mount_failed and kwargs.get("volumes"):
            self._mount_failed = True
            import docker.errors
            raise docker.errors.DockerException("Simulated drive sharing mount failure")

        c = FakeContainer(self.client, cid=f"fake-c-{len(self.created_containers) + 1}")
        self.created_containers.append(c)
        return c


class FakeAPI:
    def __init__(self, client: Any):
        self.client = client
        self.exec_calls: list[dict[str, Any]] = []
        self.next_stream_chunks: list[tuple[bytes | None, bytes | None]] = [
            (b"Hello from sandbox stdout\n", None)
        ]
        self.next_exit_code = 0

    def exec_create(self, container_id: str, cmd: str, **kwargs) -> dict[str, str]:
        self.exec_calls.append({"cid": container_id, "cmd": cmd, **kwargs})
        return {"Id": "exec-999"}

    def exec_start(self, exec_id: str, stream: bool = True, demux: bool = True) -> Any:
        for chunk in self.next_stream_chunks:
            yield chunk

    def exec_inspect(self, exec_id: str) -> dict[str, Any]:
        return {"ExitCode": self.next_exit_code}


class FakeDockerClient:
    def __init__(self, image: FakeImage | None = None, raise_image_missing: bool = False):
        self.images = FakeImages(image=image, raise_missing=raise_image_missing)
        self.containers = FakeContainers(self)
        self.api = FakeAPI(self)


# -----------------------------------------------------------------------------
# 测试用例 A - K
# -----------------------------------------------------------------------------

def test_a_creation_parameters_and_isolation():
    """A 容器创建参数安全与挂载唯一性验证。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()
        res = sb.run_python("print('test')")
        assert res.exit_code == 0
        assert len(fake.containers.run_calls) == 1
        kwargs = fake.containers.run_calls[0]

        # 1. 安全基线参数校验
        assert kwargs.get("cap_drop") == ["ALL"]
        assert kwargs.get("security_opt") == ["no-new-privileges"]
        assert kwargs.get("network_disabled") is True
        assert kwargs.get("pids_limit") == config.DOCKER_PIDS_LIMIT
        assert kwargs.get("mem_limit") == config.DOCKER_MAX_MEM
        assert kwargs.get("memswap_limit") == config.DOCKER_MAX_MEM
        assert kwargs.get("nano_cpus") == config.DOCKER_NANO_CPUS
        assert kwargs.get("user") == "10001:10001"
        assert kwargs.get("init") is True
        assert kwargs.get("labels") == {"gnk48.sandbox": "1"}

        # 2. tmpfs 校验
        tmpfs = kwargs.get("tmpfs", {})
        assert "/work" in tmpfs
        assert "/tmp" in tmpfs
        assert f"size={config.SANDBOX_WORK_TMPFS_MB}m" in tmpfs["/work"]

        # 3. 宿主挂载校验：唯一的宿主挂载是只读 workspace，绝无任何可写挂载
        vols = kwargs.get("volumes")
        assert vols is not None, "应挂载 workspace"
        for host_path, spec in vols.items():
            assert spec["mode"] == "ro", f"宿主挂载 {host_path} 必须为只读 ro"
            assert spec["bind"] == "/workspace"
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_b_lifecycle_reuse_and_idle_ttl():
    """B 跨调用复用、空闲 TTL 超时与容器丢失自动重建。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()

        # 1. 连续两次执行只创建一次容器
        res1 = sb.run_python("a = 1")
        res2 = sb.run_python("b = 2")
        assert len(fake.containers.created_containers) == 1
        assert len(fake.containers.run_calls) == 1

        # 2. 模拟空闲 TTL 过期后重建
        sb._last_active_time = time.time() - (config.SANDBOX_IDLE_TTL_SECONDS + 10)
        res3 = sb.run_python("c = 3")
        assert len(fake.containers.created_containers) == 2
        assert any("空闲超时重建" in n for n in res3.notes)

        # 3. 模拟容器非预期停止丢失
        curr_c = fake.containers.created_containers[-1]
        curr_c.status = "exited"
        res4 = sb.run_python("d = 4")
        assert len(fake.containers.created_containers) == 3
        assert any("沙箱已重建" in n or "自动重建" in n for n in res4.notes)
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_c_timeout_handling():
    """C 脚本超时终止、timed_out 标志与容器重建。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()
        # 模拟容器内 timeout 命令以 124 退出码退出
        fake.api.next_exit_code = 124
        fake.api.next_stream_chunks = [(b"partial stdout\n", b"timed out")]

        res = sb.run_python("while True: pass", timeout=2)
        assert res.timed_out is True
        assert any("超时" in n for n in res.notes)
        # 超时后应重建容器以消除死循环残留进程
        assert sb._container is None
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_d_output_flooding_truncation():
    """D 大输出流式截断与宿主内存有界。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()
        # 制造超大输出流 (如 500,000 字符)
        large_chunk = b"X" * 10000
        fake.api.next_stream_chunks = [(large_chunk, None) for _ in range(50)]

        res = sb.run_python("print('flood')")
        assert len(res.stdout) < config.SANDBOX_MAX_STDOUT_CHARS + 200
        assert "已省略中间" in res.stdout
        assert any("截断" in n for n in res.notes)
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_e_artifacts_handling_and_guards():
    """E 产物提取、图片上限、像素检查及恶意 tar 成员防护。"""
    fake = FakeDockerClient()
    set_docker_client(fake)

    # 构造含合法图片、超限图片、大像素图片、非图片产物及恶意成员的 tar
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as t:
        # 1. 正常小图 4 张
        for i in range(4):
            buf = io.BytesIO()
            img = Image.new("RGB", (50, 50), color="blue")
            img.save(buf, format="PNG")
            data = buf.getvalue()
            ti = tarfile.TarInfo(name=f"plot_{i}.png")
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))

        # 2. 超额第 5 张图片
        ti_excess = tarfile.TarInfo(name="plot_excess.png")
        ti_excess.size = len(data)
        t.addfile(ti_excess, io.BytesIO(data))

        # 3. 非图片产物
        csv_data = b"col1,col2\n1,2\n"
        ti_csv = tarfile.TarInfo(name="result.csv")
        ti_csv.size = len(csv_data)
        t.addfile(ti_csv, io.BytesIO(csv_data))

        # 4. 恶意跨目录路径成员
        evil_data = b"evil script"
        ti_evil = tarfile.TarInfo(name="../etc/evil.sh")
        ti_evil.size = len(evil_data)
        t.addfile(ti_evil, io.BytesIO(evil_data))

    tar_bytes = tar_stream.getvalue()

    try:
        sb = get_sandbox()
        c, _ = sb._ensure_container()
        c.mock_outputs_tar = tar_bytes

        res = sb.run_python("import matplotlib")
        # 图片数量上限为 4
        assert len(res.images) == 4
        # 非图片产物只列出名字与大小
        assert len(res.other_files) == 1
        assert res.other_files[0][0] == "result.csv"
        # 恶意文件被忽略
        assert not any("evil.sh" in f[0] for f in res.other_files)
        # notes 中应包含忽略与上限提示
        assert any("超过上限" in n for n in res.notes)
        assert any("非图片产物未导出" in n for n in res.notes)
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_f_docker_unavailability_errors():
    """F Docker 不可达、镜像缺失及版本过旧时的明确引导报错。"""
    # 1. 镜像缺失
    fake_missing = FakeDockerClient(raise_image_missing=True)
    set_docker_client(fake_missing)
    try:
        sb = get_sandbox()
        try:
            sb.run_python("print(1)")
            assert False, "应当抛出 SandboxUnavailable"
        except SandboxUnavailable as e:
            assert "docker build -t python-ds:latest deploy/sandbox" in str(e)
    finally:
        shutdown_sandbox()

    # 2. 镜像版本过旧 (未标记或版本为 1)
    fake_old = FakeDockerClient(image=FakeImage(labels={"gnk48.sandbox.version": "1"}))
    set_docker_client(fake_old)
    try:
        sb = get_sandbox()
        try:
            sb.run_python("print(1)")
            assert False, "应当抛出 SandboxUnavailable"
        except SandboxUnavailable as e:
            assert "重新构建镜像" in str(e)
            assert "docker build -t python-ds:latest deploy/sandbox" in str(e)
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_g_workspace_mount_fallback():
    """G 用户工作区挂载失败时平滑降级。"""
    fake = FakeDockerClient()
    fake.containers.fail_mount_once = True
    set_docker_client(fake)
    try:
        sb = get_sandbox()
        res = sb.run_python("print(1)")
        assert res.exit_code == 0
        assert any("工作区挂载失败" in n for n in res.notes)
        # 第二次启动应不带 volumes
        assert len(fake.containers.run_calls) == 2
        assert fake.containers.run_calls[1].get("volumes") is None
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_h_base_sandbox_contract_and_composite():
    """H BaseSandbox 标准契约实现与 CompositeBackend 挂载。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()
        # 1. 协议类型检测
        assert isinstance(sb, SandboxBackendProtocol)

        # 2. execute 保留 timeout 关键字参数，且 0 回落默认
        resp = sb.execute("ls", timeout=10)
        assert resp.exit_code == 0
        assert isinstance(resp.output, str)

        resp_zero = sb.execute("ls", timeout=0)
        assert resp_zero.exit_code == 0

        # 3. upload_files 校验：写入 /workspace 返回 permission_denied
        up_res = sb.upload_files([
            ("/workspace/data.csv", b"a,b\n1,2\n"),
            ("/work/code.py", b"print('ok')\n"),
            ("/invalid_dir/bad.txt", b"bad"),
        ])
        assert len(up_res) == 3
        assert up_res[0].error == "permission_denied"
        assert up_res[1].error is None
        assert up_res[2].error == "invalid_path"

        # 4. download_files 校验：文件不存在返回 file_not_found
        dl_res = sb.download_files(["/work/non_existent.txt"])
        assert len(dl_res) == 1
        assert dl_res[0].error == "file_not_found"

        # 5. CompositeBackend 挂载兼容性
        skills_backend = FilesystemBackend(root_dir=config.PROJECT_ROOT)
        composite = CompositeBackend(default=sb, routes={"/skills/": skills_backend})
        comp_res = composite.execute("echo test")
        assert comp_res.exit_code == 0
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_i_reset_and_shutdown_idempotency():
    """I reset_sandbox 与 shutdown_sandbox 幂等性。"""
    # 无容器时调用，绝不抛出异常也不连接 Docker
    set_docker_client(None)
    reset_sandbox()
    shutdown_sandbox()
    reset_sandbox()
    shutdown_sandbox()


def test_j_concurrency_serialization():
    """J 多线程并发调用安全串行化。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(sb.run_python, f"print({i})") for i in range(5)]
            results = [f.result() for f in futures]
        assert len(results) == 5
        assert all(r.exit_code == 0 for r in results)
        assert len(fake.containers.created_containers) == 1
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_k_code_tools_integration_and_outbox():
    """K code_tools 封装、outbox 图片流转与异常转文本。"""
    fake = FakeDockerClient()
    set_docker_client(fake)

    # 制作一张测试输出图
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color="green")
    img.save(buf, format="PNG")
    png_data = buf.getvalue()

    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as t:
        ti = tarfile.TarInfo(name="chart.png")
        ti.size = len(png_data)
        t.addfile(ti, io.BytesIO(png_data))
    mock_tar = tar_stream.getvalue()

    try:
        sb = get_sandbox()
        c, _ = sb._ensure_container()
        c.mock_outputs_tar = mock_tar

        # 初始化发件箱
        outbox.begin()

        # 调用工具函数
        tool_res = code_tools.execute_python_code.invoke({"code": "import matplotlib.pyplot as plt"})

        # 1. 有图片时返回多模态 list
        assert isinstance(tool_res, list)
        assert tool_res[0]["type"] == "text"
        assert "Stdout:" in tool_res[0]["text"]
        assert "Stderr:" in tool_res[0]["text"]
        assert "已向用户发送 1 张图片" in tool_res[0]["text"]

        # 模型缩略图返回
        assert tool_res[1]["type"] == "image_url"
        assert "data:image/jpeg;base64," in tool_res[1]["image_url"]["url"]

        # 2. 发件箱推送给用户
        assert len(outbox.current()) == 1
        assert "data:image/jpeg;base64," in outbox.current()[0]["image_url"]["url"]

        # 3. 沙箱故障时转为友好文本绝不向外抛出
        set_docker_client(FakeDockerClient(raise_image_missing=True))
        err_res = code_tools.execute_python_code.invoke({"code": "print(1)"})
        assert isinstance(err_res, str)
        assert "代码执行沙箱不可用" in err_res
    finally:
        outbox.begin()
        shutdown_sandbox()
        set_docker_client(None)


SCENARIOS = [
    ("A 容器创建参数安全与挂载唯一性", test_a_creation_parameters_and_isolation),
    ("B 跨调用复用与空闲 TTL 重建", test_b_lifecycle_reuse_and_idle_ttl),
    ("C 超时看门狗与容器重建", test_c_timeout_handling),
    ("D 超大输出流式截断与有界内存", test_d_output_flooding_truncation),
    ("E 产物提取、图片上限与恶意路径防护", test_e_artifacts_handling_and_guards),
    ("F Docker 不可达与镜像版本校验引导", test_f_docker_unavailability_errors),
    ("G 工作区挂载失败自动降级", test_g_workspace_mount_fallback),
    ("H BaseSandbox 契约与 CompositeBackend 转发", test_h_base_sandbox_contract_and_composite),
    ("I reset 与 shutdown 幂等性", test_i_reset_and_shutdown_idempotency),
    ("J 多线程并发串行化", test_j_concurrency_serialization),
    ("K code_tools 多模态与错误兜底", test_k_code_tools_integration_and_outbox),
]


def main():
    print("=" * 70)
    print("开始运行 DockerSandbox 离线冒烟测试...")
    print("=" * 70)
    passed = failed = 0
    for name, fn in SCENARIOS:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e!r}")
            import traceback
            traceback.print_exc()
            failed += 1
    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(SCENARIOS)}, FAIL: {failed}/{len(SCENARIOS)}")
    print("=" * 70)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
