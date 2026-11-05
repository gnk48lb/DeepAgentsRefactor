"""Docker 隔离代码执行沙箱服务 (所在层: L1)

提供会话级常驻、只读挂载 workspace、资源限制与安全加固的 Docker 代码执行沙箱。
契约与分层规则：
  - 本模块仅返回基础数据类型、字节流与数据类 (RunResult, SandboxImage)。
  - 严禁导入 L3 (app.tools)、L4 (app.agents) 或同层平级子包 (app.runtime)。
  - 外部消费（如 outbox 推送多模态图片）统一由上层工具模块 (app.tools.code_tools) 负责。
"""

import atexit
import base64
import collections
import io
import os
import posixpath
import tarfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image
import docker
import docker.errors
import docker.types
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from app import config


class SandboxUnavailable(RuntimeError):
    """Docker 沙箱不可用异常（Docker 守护进程未启动、镜像缺失或版本过旧）。"""
    pass


@dataclass
class SandboxImage:
    """沙箱执行产生的图片产物。"""
    name: str
    data: bytes


@dataclass
class RunResult:
    """Python 代码执行结果。"""
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    duration: float
    images: list[SandboxImage] = field(default_factory=list)
    other_files: list[tuple[str, int]] = field(default_factory=list)
    work_listing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class StreamingBuffer:
    """有界流式字符缓冲，边读边做头尾截断，严格保证宿主内存 O(1) 有界。"""

    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self.head_max = int(max_chars * 0.6)
        self.tail_max = max_chars - self.head_max
        self.head = ""
        self.tail: collections.deque[str] = collections.deque(maxlen=self.tail_max)
        self.total_chars = 0

    def feed(self, text: str) -> None:
        if not text:
            return
        self.total_chars += len(text)
        if len(self.head) < self.head_max:
            needed = self.head_max - len(self.head)
            self.head += text[:needed]
            remaining = text[needed:]
        else:
            remaining = text
        for ch in remaining:
            self.tail.append(ch)

    def get_result(self) -> tuple[str, bool]:
        if self.total_chars <= self.max_chars:
            return self.head + "".join(self.tail), False
        tail_str = "".join(self.tail)
        omitted = self.total_chars - len(self.head) - len(tail_str)
        return (
            f"{self.head}\n\n... [已省略中间 {omitted} 字符] ...\n\n{tail_str}",
            True,
        )


class DockerSandbox(BaseSandbox):
    """基于 Docker 容器的会话级常驻 Python 代码沙箱。"""

    def __init__(self, client: Any | None = None):
        self._lock = threading.RLock()
        self._custom_client = client
        self._container: Any = None
        self._last_active_time: float = 0.0
        self._cleaned_old_containers = False
        self._workspace_mounted = False
        self._run_count = 0
        self._creation_notes: list[str] = []

    @property
    def id(self) -> str:
        with self._lock:
            if self._container:
                return str(self._container.id)
            return "docker-sandbox-lazy"

    def _get_client(self) -> docker.DockerClient:
        if self._custom_client is not None:
            return self._custom_client
        try:
            http_timeout = config.SANDBOX_TIMEOUT_SECONDS + 30
            return docker.from_env(timeout=http_timeout)
        except Exception as e:
            raise SandboxUnavailable(
                f"无法连接到 Docker 守护进程，请确认 Docker 是否已启动：{type(e).__name__}: {e}"
            ) from e

    def _cleanup_stale_containers(self, client: Any) -> None:
        if self._cleaned_old_containers:
            return
        try:
            containers = client.containers.list(all=True, filters={"label": "gnk48.sandbox=1"})
            for c in containers:
                try:
                    c.remove(force=True)
                except Exception:
                    pass
            self._cleaned_old_containers = True
        except Exception:
            pass

    def _validate_image(self, client: Any) -> None:
        try:
            image = client.images.get(config.DOCKER_IMAGE_NAME)
        except Exception as e:
            raise SandboxUnavailable(
                f"Docker 沙箱镜像 '{config.DOCKER_IMAGE_NAME}' 不存在，请执行构建：\n"
                f"docker build -t {config.DOCKER_IMAGE_NAME} deploy/sandbox"
            ) from e

        labels = image.labels or {}
        img_version = labels.get("gnk48.sandbox.version")
        if img_version != config.SANDBOX_IMAGE_VERSION:
            raise SandboxUnavailable(
                f"Docker 沙箱镜像 '{config.DOCKER_IMAGE_NAME}' 版本不匹配 "
                f"(当前: {img_version!r}, 期望: {config.SANDBOX_IMAGE_VERSION!r})，请重新构建镜像：\n"
                f"docker build -t {config.DOCKER_IMAGE_NAME} deploy/sandbox"
            )

    def _create_container(self, client: Any) -> Any:
        self._cleanup_stale_containers(client)
        self._validate_image(client)
        self._creation_notes.clear()

        # tmpfs 限制内存与工作空间大小
        tmpfs = {
            "/work": f"size={config.SANDBOX_WORK_TMPFS_MB}m,uid=10001,gid=10001,mode=1777",
            "/tmp": "size=64m,uid=10001,gid=10001,mode=1777",
        }

        ulimits = [
            docker.types.Ulimit(name="core", soft=0, hard=0),
            docker.types.Ulimit(name="fsize", soft=104857600, hard=104857600),
            docker.types.Ulimit(name="nofile", soft=1024, hard=2048),
        ]

        volumes: dict[str, dict[str, str]] = {}
        mount_workspace = getattr(config, "SANDBOX_MOUNT_WORKSPACE", True)
        if mount_workspace:
            ws_path = os.path.abspath(config.WORKSPACE_DIR)
            volumes[ws_path] = {"bind": "/workspace", "mode": "ro"}

        container = None
        # 尝试带 workspace 只读挂载启动
        try:
            container = client.containers.run(
                image=config.DOCKER_IMAGE_NAME,
                command=["sleep", "infinity"],
                working_dir="/work",
                user="10001:10001",
                volumes=volumes if volumes else None,
                tmpfs=tmpfs,
                mem_limit=config.DOCKER_MAX_MEM,
                memswap_limit=config.DOCKER_MAX_MEM,
                pids_limit=config.DOCKER_PIDS_LIMIT,
                nano_cpus=config.DOCKER_NANO_CPUS,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                network_disabled=True,
                init=True,
                labels={"gnk48.sandbox": "1"},
                ulimits=ulimits,
                detach=True,
            )
            self._workspace_mounted = bool(volumes)
        except Exception as e:
            if volumes:
                # 降级尝试：不带 workspace 挂载启动
                try:
                    container = client.containers.run(
                        image=config.DOCKER_IMAGE_NAME,
                        command=["sleep", "infinity"],
                        working_dir="/work",
                        user="10001:10001",
                        volumes=None,
                        tmpfs=tmpfs,
                        mem_limit=config.DOCKER_MAX_MEM,
                        memswap_limit=config.DOCKER_MAX_MEM,
                        pids_limit=config.DOCKER_PIDS_LIMIT,
                        nano_cpus=config.DOCKER_NANO_CPUS,
                        cap_drop=["ALL"],
                        security_opt=["no-new-privileges"],
                        network_disabled=True,
                        init=True,
                        labels={"gnk48.sandbox": "1"},
                        ulimits=ulimits,
                        detach=True,
                    )
                    self._workspace_mounted = False
                    self._creation_notes.append(
                        f"用户工作区挂载失败（降级为未挂载）：{e}"
                    )
                except Exception as inner_e:
                    raise SandboxUnavailable(
                        f"启动 Docker 沙箱容器失败: {inner_e}"
                    ) from inner_e
            else:
                raise SandboxUnavailable(
                    f"启动 Docker 沙箱容器失败: {e}"
                ) from e

        # 容器启动后将预热字体缓存拷贝到 /tmp/mpl
        try:
            container.exec_run("sh -c 'mkdir -p /tmp/mpl && cp -a /opt/mplcache/. /tmp/mpl/'", user="10001:10001")
        except Exception:
            pass

        self._last_active_time = time.time()
        return container

    def _ensure_container(self) -> tuple[Any, list[str]]:
        """确保沙箱容器处于运行状态，并在必要时重建（返回容器与附加 notes）。"""
        notes: list[str] = []
        client = self._get_client()

        # 空闲 TTL 检测
        now = time.time()
        if (
            self._container is not None
            and self._last_active_time > 0
            and (now - self._last_active_time) > config.SANDBOX_IDLE_TTL_SECONDS
        ):
            self.close()
            notes.append("沙箱已空闲超时重建，/work 内容已清空")

        # 检查现有容器存活
        if self._container is not None:
            try:
                self._container.reload()
                status = getattr(self._container, "status", None)
                if status != "running":
                    self.close()
                    notes.append("沙箱容器已停止，自动重建，/work 内容已清空")
            except Exception:
                self.close()
                notes.append("沙箱容器丢失，自动重建，/work 内容已清空")

        if self._container is None:
            self._container = self._create_container(client)
            if self._creation_notes:
                notes.extend(self._creation_notes)

        self._last_active_time = time.time()
        return self._container, notes

    def _upload_via_exec(self, container: Any, norm_path: str, data: bytes) -> None:
        b64_data = base64.b64encode(data).decode("ascii")
        py_code = (
            "import sys, base64; "
            "open(sys.argv[1], 'wb').write(base64.b64decode(sys.argv[2]))"
        )
        res = container.exec_run(
            ["python", "-c", py_code, norm_path, b64_data],
            user="10001:10001",
        )
        if res.exit_code != 0:
            out_msg = res.output.decode("utf-8", errors="replace") if isinstance(res.output, bytes) else str(res.output)
            raise RuntimeError(f"写入文件失败: {out_msg}")

    def _upload_single_file(self, container: Any, norm_path: str, data: bytes) -> None:
        parent = posixpath.dirname(norm_path)
        filename = posixpath.basename(norm_path)
        if parent:
            container.exec_run(f"mkdir -p '{parent}'", user="10001:10001")
        container.exec_run(f"rm -f '{norm_path}'", user="10001:10001")

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            ti = tarfile.TarInfo(name=filename)
            ti.size = len(data)
            ti.uid = 10001
            ti.gid = 10001
            ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(data))
        tar_bytes = tar_stream.getvalue()

        try:
            container.put_archive(parent or "/work", tar_bytes)
        except Exception:
            pass

        # 检查是否实际落盘（应对 containerd snapshotter 下 tmpfs 路径静默不写入的情况）
        try:
            chk = container.exec_run(f"test -f '{norm_path}'", user="10001:10001")
            if chk.exit_code != 0:
                self._upload_via_exec(container, norm_path, data)
        except Exception:
            self._upload_via_exec(container, norm_path, data)

    def _download_single_file(self, container: Any, path: str) -> bytes:
        norm_path = posixpath.normpath(path)
        # 优先使用 API get_archive
        try:
            bits, stat = container.get_archive(norm_path)
            tar_bytes = b"".join(bits)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as t:
                members = [m for m in t.getmembers() if m.isreg()]
                dir_members = [m for m in t.getmembers() if m.isdir()]
                if not members and dir_members:
                    raise IsADirectoryError(f"'{path}' 是目录")
                if not members:
                    raise FileNotFoundError(f"文件 '{path}' 不存在")
                f = t.extractfile(members[0])
                if f is None:
                    raise FileNotFoundError(f"文件 '{path}' 不存在")
                return f.read()
        except (docker.errors.NotFound, FileNotFoundError):
            pass
        except IsADirectoryError:
            raise

        # 降级：通过容器内 tar 获取（保证 containerd snapshotter 下对 tmpfs 可读）
        chk = container.exec_run(f"sh -c 'test -d \"{norm_path}\" && echo DIR || (test -f \"{norm_path}\" && echo FILE || echo NOTFOUND)'")
        chk_out = chk.output.strip() if isinstance(chk.output, bytes) else chk.output.strip().encode()
        if chk_out == b"DIR":
            raise IsADirectoryError(f"'{path}' 是目录")
        if chk_out != b"FILE":
            raise FileNotFoundError(f"文件 '{path}' 不存在")

        tar_res = container.exec_run(f"tar -cf - '{norm_path}'")
        if tar_res.exit_code != 0:
            raise FileNotFoundError(f"文件 '{path}' 读取失败")
        with tarfile.open(fileobj=io.BytesIO(tar_res.output)) as t:
            members = [m for m in t.getmembers() if m.isreg()]
            if not members:
                raise FileNotFoundError(f"文件 '{path}' 不存在")
            f = t.extractfile(members[0])
            if f is None:
                raise FileNotFoundError(f"文件 '{path}' 不存在")
            return f.read()

    def run_python(self, code: str, *, timeout: int | None = None) -> RunResult:
        """在常驻沙箱中执行 Python 代码。"""
        with self._lock:
            self._run_count += 1
            notes: list[str] = []

            # 代码大小防护 (256KB)
            code_bytes = code.encode("utf-8")
            if len(code_bytes) > 256 * 1024:
                return RunResult(
                    stdout="",
                    stderr="代码大小超过 256KB 上限，已被直接拒绝执行。",
                    exit_code=1,
                    timed_out=False,
                    duration=0.0,
                    notes=["代码超过 256KB 阈值"],
                )

            effective_timeout = (
                timeout
                if (timeout is not None and timeout > 0)
                else config.SANDBOX_TIMEOUT_SECONDS
            )

            container, init_notes = self._ensure_container()
            if init_notes:
                notes.extend(init_notes)

            # 将代码写入 /work/main.py
            try:
                self._upload_single_file(container, "/work/main.py", code_bytes)
            except Exception as e:
                return RunResult(
                    stdout="",
                    stderr=f"沙箱环境写入代码失败: {e}",
                    exit_code=1,
                    timed_out=False,
                    duration=0.0,
                    notes=notes,
                )

            # 每次执行前清理并重新创建 /work/outputs
            client = self._get_client()
            cmd = (
                f"sh -c 'rm -rf /work/outputs && mkdir -p /work/outputs && "
                f"exec timeout -k 3 {effective_timeout} python -u /work/main.py'"
            )

            start_time = time.time()
            timed_out = False
            exit_code: int | None = None

            stdout_buf = StreamingBuffer(config.SANDBOX_MAX_STDOUT_CHARS)
            stderr_buf = StreamingBuffer(config.SANDBOX_MAX_STDERR_CHARS)

            try:
                exec_instance = client.api.exec_create(
                    container.id,
                    cmd,
                    workdir="/work",
                    user="10001:10001",
                    stdout=True,
                    stderr=True,
                )
                exec_id = exec_instance["Id"]
                stream = client.api.exec_start(exec_id, stream=True, demux=True)

                # 设置宿主机看门狗定时器 (timeout + 5s)
                watchdog_fired = False

                def _watchdog_kill():
                    nonlocal watchdog_fired
                    watchdog_fired = True
                    try:
                        container.kill()
                    except Exception:
                        pass

                timer = threading.Timer(effective_timeout + 5.0, _watchdog_kill)
                timer.daemon = True
                timer.start()

                try:
                    for out_chunk, err_chunk in stream:
                        if out_chunk:
                            stdout_buf.feed(out_chunk.decode("utf-8", errors="replace"))
                        if err_chunk:
                            stderr_buf.feed(err_chunk.decode("utf-8", errors="replace"))
                finally:
                    timer.cancel()

                inspect_res = client.api.exec_inspect(exec_id)
                exit_code = inspect_res.get("ExitCode")

                if watchdog_fired or exit_code in (124, 137):
                    timed_out = True

            except Exception as e:
                timed_out = True
                notes.append(f"执行异常中断: {e}")

            duration = time.time() - start_time
            self._last_active_time = time.time()

            # 超时处理：强制重建容器以消除残留进程
            if timed_out:
                self.close()
                notes.append(f"执行超时 ({effective_timeout}s)，沙箱已重置清空 /work")

            stdout_text, stdout_truncated = stdout_buf.get_result()
            stderr_text, stderr_truncated = stderr_buf.get_result()

            if stdout_truncated:
                notes.append(f"stdout 输出已超过 {config.SANDBOX_MAX_STDOUT_CHARS} 字符并完成头尾截断")
            if stderr_truncated:
                notes.append(f"stderr 输出已超过 {config.SANDBOX_MAX_STDERR_CHARS} 字符并完成头尾截断")

            # 收集产物：/work/outputs
            images: list[SandboxImage] = []
            other_files: list[tuple[str, int]] = []

            if not timed_out:
                try:
                    outputs_tar_bytes = self._get_outputs_tar(container)
                    if outputs_tar_bytes:
                        collected_images, collected_others, output_notes = self._parse_outputs(outputs_tar_bytes)
                        images.extend(collected_images)
                        other_files.extend(collected_others)
                        notes.extend(output_notes)
                except Exception as e:
                    notes.append(f"产物提取异常: {e}")

            # 列出 /work 顶层现有文件 (<=20)
            work_listing: list[str] = []
            if not timed_out and self._container is not None:
                try:
                    res_ls = container.exec_run("ls -1 /work", user="10001:10001")
                    if res_ls.exit_code == 0:
                        lines = [
                            ln.strip()
                            for ln in res_ls.output.decode("utf-8", errors="replace").splitlines()
                            if ln.strip()
                        ]
                        work_listing = lines[:20]
                except Exception:
                    pass

            print(
                f"[Sandbox Run #{self._run_count}] exit={exit_code} "
                f"duration={duration:.2f}s stdout={len(stdout_text)}chars "
                f"stderr={len(stderr_text)}chars images={len(images)}"
            )

            return RunResult(
                stdout=stdout_text,
                stderr=stderr_text,
                exit_code=exit_code,
                timed_out=timed_out,
                duration=duration,
                images=images,
                other_files=other_files,
                work_listing=work_listing,
                notes=notes,
            )

    def _get_outputs_tar(self, container: Any) -> bytes | None:
        """获取 /work/outputs 的归档字节流。"""
        # 尝试标准 get_archive
        try:
            bits, _ = container.get_archive("/work/outputs")
            tar_bytes = b"".join(bits)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as t:
                reg_files = [m for m in t.getmembers() if m.isreg()]
                if reg_files:
                    return tar_bytes
        except Exception:
            pass

        # 降级尝试：exec tar 获取（保证 containerd snapshotter 下对 tmpfs 真实产物可读）
        try:
            res = container.exec_run("tar -cf - -C /work/outputs .", user="10001:10001")
            if res.exit_code == 0 and res.output:
                return res.output
        except Exception:
            pass
        return None

    def _parse_outputs(self, tar_bytes: bytes) -> tuple[list[SandboxImage], list[tuple[str, int]], list[str]]:
        images: list[SandboxImage] = []
        other_files: list[tuple[str, int]] = []
        notes: list[str] = []

        total_bytes = 0
        file_count = 0

        try:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as t:
                for member in t.getmembers():
                    if not member.isreg():
                        continue
                    # 安全防御：跳过含有 .. 或绝对路径的恶意文件
                    norm_name = posixpath.normpath(member.name)
                    if (
                        norm_name.startswith("..")
                        or norm_name.startswith("/")
                        or "/../" in f"/{norm_name}/"
                    ):
                        continue

                    file_count += 1
                    if file_count > 50:
                        notes.append("产物文件数量超过上限 (50 个)，多余文件已忽略")
                        break

                    total_bytes += member.size
                    if total_bytes > 50 * 1024 * 1024:
                        notes.append("产物文件总大小超过上限 (50MB)，已终止提取")
                        break

                    f = t.extractfile(member)
                    if f is None:
                        continue
                    data = f.read()
                    basename = posixpath.basename(member.name)
                    ext = posixpath.splitext(basename)[1].lower()

                    if ext in (".png", ".jpg", ".jpeg", ".webp"):
                        if len(data) > config.SANDBOX_MAX_IMAGE_MB * 1024 * 1024:
                            notes.append(
                                f"图片 {basename} 大小 ({len(data)}B) 超过 {config.SANDBOX_MAX_IMAGE_MB}MB 上限，已忽略"
                            )
                            continue

                        if len(images) >= config.SANDBOX_MAX_IMAGES:
                            notes.append(
                                f"单次执行生成图片超过上限 ({config.SANDBOX_MAX_IMAGES} 张)，已忽略超出部分的图片：{basename}"
                            )
                            continue

                        # 校验像素尺寸 (<= 40M)
                        try:
                            with Image.open(io.BytesIO(data)) as img:
                                w, h = img.size
                                if w * h > 40_000_000:
                                    notes.append(
                                        f"图片 {basename} 像素 ({w}x{h}={w*h}) 超过 40M 上限，已忽略"
                                    )
                                    continue
                        except Exception as e:
                            notes.append(f"图片 {basename} 解析校验失败，已跳过: {e}")
                            continue

                        images.append(SandboxImage(name=basename, data=data))
                    else:
                        other_files.append((basename, len(data)))
                        notes.append(f"非图片产物未导出：{basename} ({len(data)} 字节)")
        except Exception as e:
            notes.append(f"解析产物 tar 归档失败: {e}")

        return images, other_files, notes

    # -------------------------------------------------------------------------
    # BaseSandbox 抽象成员契约实现
    # -------------------------------------------------------------------------
    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        with self._lock:
            effective_timeout = (
                timeout
                if (timeout is not None and timeout > 0)
                else config.SANDBOX_TIMEOUT_SECONDS
            )

            container, _ = self._ensure_container()
            client = self._get_client()

            cmd = f"sh -c '{command}'"
            buf = StreamingBuffer(config.SANDBOX_MAX_STDOUT_CHARS)
            exit_code: int | None = None

            try:
                exec_instance = client.api.exec_create(
                    container.id,
                    cmd,
                    workdir="/work",
                    user="10001:10001",
                    stdout=True,
                    stderr=True,
                )
                exec_id = exec_instance["Id"]
                stream = client.api.exec_start(exec_id, stream=True, demux=True)

                watchdog_fired = False

                def _watchdog_kill():
                    nonlocal watchdog_fired
                    watchdog_fired = True
                    try:
                        container.kill()
                    except Exception:
                        pass

                timer = threading.Timer(effective_timeout + 5.0, _watchdog_kill)
                timer.daemon = True
                timer.start()

                try:
                    for out_chunk, err_chunk in stream:
                        if out_chunk:
                            buf.feed(out_chunk.decode("utf-8", errors="replace"))
                        if err_chunk:
                            buf.feed(err_chunk.decode("utf-8", errors="replace"))
                finally:
                    timer.cancel()

                inspect_res = client.api.exec_inspect(exec_id)
                exit_code = inspect_res.get("ExitCode")
                if watchdog_fired or exit_code in (124, 137):
                    self.close()

            except Exception as e:
                buf.feed(f"\n[执行异常: {e}]")
                exit_code = 1

            self._last_active_time = time.time()
            text, truncated = buf.get_result()
            return ExecuteResponse(output=text, exit_code=exit_code, truncated=truncated)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        with self._lock:
            container, _ = self._ensure_container()
            responses: list[FileUploadResponse] = []

            for path, data in files:
                norm = posixpath.normpath(path)
                if norm.startswith("/workspace") or norm == "/workspace":
                    responses.append(FileUploadResponse(path=path, error="permission_denied"))
                    continue

                if not (norm.startswith("/work") or norm.startswith("/tmp")):
                    responses.append(FileUploadResponse(path=path, error="invalid_path"))
                    continue

                try:
                    self._upload_single_file(container, norm, data)
                    responses.append(FileUploadResponse(path=path, error=None))
                except Exception as e:
                    responses.append(FileUploadResponse(path=path, error=str(e)))

            self._last_active_time = time.time()
            return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        with self._lock:
            container, _ = self._ensure_container()
            responses: list[FileDownloadResponse] = []

            for path in paths:
                try:
                    content = self._download_single_file(container, path)
                    responses.append(FileDownloadResponse(path=path, content=content, error=None))
                except FileNotFoundError:
                    responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                except IsADirectoryError:
                    responses.append(FileDownloadResponse(path=path, content=None, error="is_directory"))
                except Exception as e:
                    responses.append(FileDownloadResponse(path=path, content=None, error=str(e)))

            self._last_active_time = time.time()
            return responses

    def reset(self) -> None:
        """重置沙箱状态，安全销毁当前容器。"""
        with self._lock:
            self.close()

    def close(self) -> None:
        """幂等关闭沙箱，清理容器。"""
        with self._lock:
            if self._container is not None:
                try:
                    self._container.remove(force=True)
                except Exception:
                    pass
                self._container = None
            self._last_active_time = 0.0


# -----------------------------------------------------------------------------
# 模块级单例管理
# -----------------------------------------------------------------------------
_GLOBAL_SANDBOX: DockerSandbox | None = None
_GLOBAL_LOCK = threading.RLock()
_CUSTOM_DOCKER_CLIENT: Any | None = None


def set_docker_client(client: Any) -> None:
    """仅供测试注入 Docker 客户端。"""
    global _CUSTOM_DOCKER_CLIENT, _GLOBAL_SANDBOX
    with _GLOBAL_LOCK:
        _CUSTOM_DOCKER_CLIENT = client
        if _GLOBAL_SANDBOX is not None:
            _GLOBAL_SANDBOX.close()
            _GLOBAL_SANDBOX = None


def get_sandbox() -> DockerSandbox:
    """获取进程级惰性 DockerSandbox 单例。"""
    global _GLOBAL_SANDBOX
    with _GLOBAL_LOCK:
        if _GLOBAL_SANDBOX is None:
            _GLOBAL_SANDBOX = DockerSandbox(client=_CUSTOM_DOCKER_CLIENT)
        return _GLOBAL_SANDBOX


def reset_sandbox() -> None:
    """重置沙箱容器（无容器时为 no-op 且不连接 Docker，绝不抛异常）。"""
    global _GLOBAL_SANDBOX
    with _GLOBAL_LOCK:
        if _GLOBAL_SANDBOX is not None:
            try:
                _GLOBAL_SANDBOX.reset()
            except Exception:
                pass


def shutdown_sandbox() -> None:
    """幂等销毁沙箱容器。"""
    global _GLOBAL_SANDBOX
    with _GLOBAL_LOCK:
        if _GLOBAL_SANDBOX is not None:
            try:
                _GLOBAL_SANDBOX.close()
            except Exception:
                pass
            _GLOBAL_SANDBOX = None


# 注册进程退出时的自动兜底清理
atexit.register(shutdown_sandbox)
