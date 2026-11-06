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
        self.labels = labels if labels is not None else {"gnk48.sandbox.version": config.SANDBOX_IMAGE_VERSION}


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
        assert any("超过" in n or "超时" in n for n in res.notes)
        # P0-4: 容器内 124 超时保留容器，工作台状态不丢失
        assert sb._container is not None
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
        assert up_res[0].error.startswith("permission_denied")
        assert up_res[1].error is None
        assert up_res[2].error.startswith("invalid_path")

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




def test_l_p0_1_command_wrapping_list():
    """L (P0-1) execute 命令包装为 list，不经 shlex，workdir=/work，user=10001:10001。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()
        test_cmd = "python3 -c \"import os; print('hello world')\" | grep -F 'hello'"
        resp = sb.execute(test_cmd, timeout=45)
        assert resp.exit_code == 0
        call = fake.api.exec_calls[-1]
        cmd = call["cmd"]
        assert isinstance(cmd, list), f"cmd 应为 list，实际为: {type(cmd)}"
        assert cmd[0] == "timeout"
        assert cmd[1] == "-k"
        assert cmd[2] == "3"
        assert cmd[3] == "45"
        assert cmd[4] == "bash"
        assert cmd[5] == "-c"
        assert cmd[6] == test_cmd
        assert call.get("workdir") == "/work"
        assert call.get("user") == "10001:10001"
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_m_p0_2_protocol_methods_fail_safe():
    """M (P0-2) 协议方法 (execute/upload/download) 遇故障转为友好结果，绝不向外抛异常。"""
    fake_err = FakeDockerClient(raise_image_missing=True)
    set_docker_client(fake_err)
    try:
        sb = get_sandbox()
        # 1. execute 返回 exit_code=1 且包含错误指引，不抛异常
        exec_res = sb.execute("ls")
        assert exec_res.exit_code == 1
        assert "沙箱不可用" in exec_res.output

        # 2. upload_files 逐文件返回 error 字符串，不抛异常
        up_res = sb.upload_files([("/work/a.py", b"print(1)"), ("/work/b.py", b"print(2)")])
        assert len(up_res) == 2
        assert all(r.error and "沙箱不可用" in r.error for r in up_res)

        # 3. download_files 逐文件返回 error 字符串，不抛异常
        dl_res = sb.download_files(["/work/a.py", "/work/b.py"])
        assert len(dl_res) == 2
        assert all(r.error and "沙箱不可用" in r.error for r in dl_res)

        # 4. run_python 仍抛 SandboxUnavailable 给 L3 工具层捕获
        try:
            sb.run_python("print(1)")
            assert False, "应当抛出 SandboxUnavailable"
        except SandboxUnavailable:
            pass
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_n_p0_3_execute_large_output_json_not_truncated():
    """N (P0-3) execute 使用 SANDBOX_EXECUTE_MAX_OUTPUT_CHARS，大段 JSON 不被截断。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()
        import json
        large_data = {"files": [f"/work/file_{i}.txt" for i in range(5000)]}
        large_json_bytes = json.dumps(large_data).encode("utf-8")
        assert len(large_json_bytes) > 50000  # 远大于 8000
        fake.api.next_stream_chunks = [(large_json_bytes, None)]

        resp = sb.execute("cat /work/large.json")
        assert resp.truncated is False
        assert len(resp.output) == len(large_json_bytes)
        parsed = json.loads(resp.output)
        assert len(parsed["files"]) == 5000
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_o_p0_4_timeout_semantics():
    """O (P0-4) 超时细分处理：124 保留容器；137 耗时 < T 提示超出内存；仅宿主看门狗才销毁容器。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()

        # 1. exit_code == 124：命令超时终止，但容器保留
        fake.api.next_exit_code = 124
        resp = sb.execute("sleep 100", timeout=5)
        assert "命令超过 5 秒被终止；沙箱状态保留" in resp.output
        assert sb._container is not None

        # 2. exit_code == 137 且耗时很短：可能 OOM
        fake.api.next_exit_code = 137
        resp_oom = sb.execute("python -c 'oom'", timeout=10)
        assert "超出内存上限" in resp_oom.output
        assert sb._container is not None
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_p_p0_5_client_caching_and_timeout_clamp():
    """P (P0-5) DockerClient 实例级缓存，且 timeout 参数正确 clamp。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()
        c1 = sb._get_client()
        c2 = sb._get_client()
        assert c1 is c2, "客户端应被缓存复用"

        # timeout clamp: 传入 500 -> clamp 到 300
        sb.execute("ls", timeout=500)
        call = fake.api.exec_calls[-1]
        assert call["cmd"][3] == str(config.SANDBOX_MAX_TIMEOUT_SECONDS)

        # timeout <= 0 或 None -> 回落默认 60
        sb.execute("ls", timeout=0)
        assert fake.api.exec_calls[-1]["cmd"][3] == str(config.SANDBOX_TIMEOUT_SECONDS)

        sb.execute("ls", timeout=None)
        assert fake.api.exec_calls[-1]["cmd"][3] == str(config.SANDBOX_TIMEOUT_SECONDS)
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_q_collect_new_outputs_fingerprint():
    """Q (A3) collect_new_outputs 仅投递新增/修改产物，跳过超限并记已见。"""
    fake = FakeDockerClient()
    set_docker_client(fake)
    try:
        sb = get_sandbox()
        c, _ = sb._ensure_container()

        # 模拟生成一张图片
        buf = io.BytesIO()
        Image.new("RGB", (30, 30), color="green").save(buf, format="PNG")
        img_bytes = buf.getvalue()
        c.mock_outputs_tar = None

        # 模拟 find 输出：chart1.png 大小 500, mtime 100.0
        find_raw = f"chart1.png\t{len(img_bytes)}\t100.0\0".encode("utf-8")
        orig_exec_run = c.exec_run
        def mock_exec_run(cmd, **kwargs):
            if "find /work/outputs" in str(cmd):
                mock = MagicMock()
                mock.exit_code = 0
                mock.output = find_raw
                return mock
            return orig_exec_run(cmd, **kwargs)
        c.exec_run = mock_exec_run

        orig_dl = sb._download_single_file
        sb._download_single_file = lambda cont, path: img_bytes

        # 第 1 次扫描：发现 chart1.png
        res1 = sb.collect_new_outputs()
        assert len(res1.images) == 1
        assert res1.images[0].name == "chart1.png"

        # 第 2 次扫描：同一指纹不再重复返回
        res2 = sb.collect_new_outputs()
        assert len(res2.images) == 0

        # 第 3 次扫描：文件 mtime 变更 -> 再次返回
        find_raw = f"chart1.png\t{len(img_bytes)}\t200.0\0".encode("utf-8")
        res3 = sb.collect_new_outputs()
        assert len(res3.images) == 1
    finally:
        shutdown_sandbox()
        set_docker_client(None)


def test_r_code_agent_assembly_and_routing():
    """R (A1-A2) CodeAgent 编译图装配：包含 7 个文件工具 + execute_python_code，无 delete，技能与工作区正确路由。"""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from app.agents.experts import code_agent
    from deepagents.backends.sandbox import BaseSandbox
    from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse

    calls_log = []
    class MockSandbox(BaseSandbox):
        @property
        def id(self): return "mock-sb"
        def execute(self, cmd, *, timeout=None):
            calls_log.append(("execute", cmd))
            return ExecuteResponse(output="ok\n", exit_code=0)
        def upload_files(self, files):
            calls_log.append(("upload", [f[0] for f in files]))
            return [FileUploadResponse(path=p, error=None) for p, _ in files]
        def download_files(self, paths):
            calls_log.append(("download", paths))
            return [FileDownloadResponse(path=p, content=b"content", error=None) for p in paths]

    msb = MockSandbox()
    llm = FakeListChatModel(responses=["done"])
    runnable = code_agent.build_code_agent_runnable(
        llm=llm,
        system_prompt=code_agent.PROMPT,
        sandbox_backend=msb,
    )
    tools_node = runnable.nodes["tools"]
    tool_dict = getattr(tools_node.triggers, "__self__", None) or tools_node.bound
    names = list(tool_dict._tools_by_name.keys())

    # 工具白名单：7 个 FS 工具 + execute_python_code，严禁包含 delete
    for expected in ["ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute", "execute_python_code"]:
        assert expected in names, f"缺少预期工具: {expected}"
    assert "delete" not in names, "严禁挂载 delete 工具"

    assert "matplotlib-chart" in code_agent.PROMPT
    assert "data-analysis" in code_agent.PROMPT
    assert "你是 CodeAgent" in code_agent.PROMPT


def test_s_sandbox_outputs_middleware():
    """S (A3 L4) SandboxOutputsMiddleware：拦截 execute 统一投递并改写 ToolMessage，非 execute 透明透传。"""
    from langchain_core.messages import ToolMessage
    from app.agents.experts.code_agent import SandboxOutputsMiddleware
    from app.infra.sandbox import SandboxImage, OutputsResult

    class FakeOutputsSandbox:
        def __init__(self):
            self.called = False
        def collect_new_outputs(self):
            self.called = True
            buf = io.BytesIO()
            Image.new("RGB", (20, 20), color="red").save(buf, format="PNG")
            return OutputsResult(images=[SandboxImage(name="chart.png", data=buf.getvalue())])

    fos = FakeOutputsSandbox()
    mw = SandboxOutputsMiddleware(fos)

    req_execute = MagicMock()
    req_execute.tool_call = {"name": "execute", "args": {"command": "python chart.py"}, "id": "call_1"}

    outbox.begin()
    tool_msg = ToolMessage(content="stdout output", tool_call_id="call_1", artifact={"exit_code": 0})
    res = mw.wrap_tool_call(req_execute, lambda r: tool_msg)

    assert fos.called is True
    assert len(outbox.current()) == 1
    assert isinstance(res.content, list)
    assert res.content[0]["type"] == "text"
    assert "已向用户发送 1 张图片" in res.content[0]["text"]
    assert res.content[1]["type"] == "image_url"
    assert res.artifact == {"exit_code": 0}

    # 非 execute 工具透明透传
    req_ls = MagicMock()
    req_ls.tool_call = {"name": "ls", "args": {"path": "/work"}, "id": "call_2"}
    fos.called = False
    tool_msg_ls = ToolMessage(content="file.txt", tool_call_id="call_2")
    res_ls = mw.wrap_tool_call(req_ls, lambda r: tool_msg_ls)
    assert fos.called is False
    assert res_ls.content == "file.txt"
    outbox.begin()

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
    ("L P0-1 命令包装为 list", test_l_p0_1_command_wrapping_list),
    ("M P0-2 协议方法不向外抛异常", test_m_p0_2_protocol_methods_fail_safe),
    ("N P0-3 execute 200万字符输出上限", test_n_p0_3_execute_large_output_json_not_truncated),
    ("O P0-4 超时细分与保留容器", test_o_p0_4_timeout_semantics),
    ("P P0-5 DockerClient 缓存与超时 clamp", test_p_p0_5_client_caching_and_timeout_clamp),
    ("Q A3 产物指纹去重与状态保留", test_q_collect_new_outputs_fingerprint),
    ("R A1-A2 CodeAgent 沙箱工作台装配", test_r_code_agent_assembly_and_routing),
    ("S A3-L4 SandboxOutputsMiddleware 多模态投递", test_s_sandbox_outputs_middleware),
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
