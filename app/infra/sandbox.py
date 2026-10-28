"""Docker 隔离代码执行沙箱服务 (所在层: L1)

本地 Docker 代码执行沙箱。
功能：
  1. 将用户提交的 Python 代码写入临时目录中的 main.py。
  2. 启动 Docker 容器（无网络，内存受限，pids 与 CPU 配额限制，删除所有 capabilities，禁止提权）执行代码。
  3. 截断过长的 stdout/stderr，防止撑爆 LLM 上下文。
  4. 扫描 outputs/ 目录，将生成的图片压缩为 Base64 返回（单次最多收集 SANDBOX_MAX_IMAGES 张）。
  5. 清理临时目录。

返回格式：
  {
    "stdout": str,
    "stderr": str,
    "images": list[str]   # base64 编码的图片列表
  }
"""

import os
import tempfile
import shutil
import glob
from typing import TypedDict

import docker
from app import config
from app.utils.images import compress_image_to_base64

# stdout / stderr 最大保留字符数（保留头尾，防止 LLM 上下文溢出同时确保末尾 traceback 可见）
_MAX_OUTPUT_CHARS = 1000

# Docker 容器执行超时（秒）
_TIMEOUT_SECONDS = 30


class SandboxResult(TypedDict):
    stdout: str
    stderr: str
    images: list  # list[str], base64 编码


def _truncate(text: str, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """如果文本超过 max_chars（默认 1000），保留前 600 字符与后 400 字符，确保末尾 Traceback 不丢失。"""
    if len(text) > max_chars:
        head_len = int(max_chars * 0.6)
        tail_len = max_chars - head_len
        omitted = len(text) - head_len - tail_len
        return f"{text[:head_len]}\n\n... [已省略中间 {omitted} 字符] ...\n\n{text[-tail_len:]}"
    return text


def run_code_in_sandbox(code: str) -> SandboxResult:
    """
    在本地 Docker 沙箱中执行 Python 代码。

    参数
    ----
    code : str
        要执行的 Python 源代码字符串。

    返回
    ----
    SandboxResult
        包含 stdout、stderr 和 images（Base64 列表）的字典。
    """
    # 在 storage 目录下创建临时沙箱根目录，规避 Windows 跨盘符挂载 Docker 报错并避免污染 FileAgent 工作区
    sandbox_tmp_root = config.SANDBOX_TMP_DIR
    os.makedirs(sandbox_tmp_root, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="sandbox_", dir=sandbox_tmp_root)
    outputs_dir = os.path.join(tmp_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # 写入用户代码
    code_path = os.path.join(tmp_dir, "main.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    stdout_text = ""
    stderr_text = ""
    images: list = []

    try:
        client = docker.from_env()

        # 挂载宿主机临时目录到容器 /workspace，进行绝对路径标准化以防止 Windows Docker 识别错误
        abs_tmp_dir = os.path.abspath(tmp_dir)
        volumes = {
            abs_tmp_dir: {"bind": "/workspace", "mode": "rw"}
        }

        try:
            # 创建并启动容器 (detach=True 以便手动控制超时)
            container = client.containers.run(
                image=config.DOCKER_IMAGE_NAME,
                command="python main.py",
                volumes=volumes,
                working_dir="/workspace",
                mem_limit=config.DOCKER_MAX_MEM,
                memswap_limit=config.DOCKER_MAX_MEM,
                pids_limit=config.DOCKER_PIDS_LIMIT,
                nano_cpus=config.DOCKER_NANO_CPUS,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                network_disabled=True,
                detach=True,
                stdout=True,
                stderr=True,
            )

            try:
                # 等待容器结束，支持超时
                wait_res = container.wait(timeout=_TIMEOUT_SECONDS)
                
                # 分别获取 stdout 与 stderr 日志并各自截断
                raw_stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                stdout_text = _truncate(raw_stdout)

                raw_stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
                stderr_text = _truncate(raw_stderr)
                
                # 如果退出码不为 0，追加在 stderr 末尾而不是替换它
                exit_code = wait_res.get("StatusCode", 0)
                if exit_code != 0:
                    exit_msg = f"容器以错误码 {exit_code} 退出。"
                    if stderr_text:
                        stderr_text = f"{stderr_text}\n{exit_msg}"
                    else:
                        stderr_text = exit_msg
            except Exception as e:
                # 超时处理
                container.kill()
                stderr_text = _truncate(f"Docker 执行超时 ({_TIMEOUT_SECONDS}s) 或异常: {e}")
            finally:
                # 手动清理容器
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        except Exception as e:
            stderr_text = _truncate(f"Docker 启动失败: {type(e).__name__}: {e}")

        # 扫描 outputs/ 目录，收集图片（最多收集 config.SANDBOX_MAX_IMAGES 张）
        img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            img_paths.extend(glob.glob(os.path.join(outputs_dir, ext)))
        img_paths.sort()

        max_images = getattr(config, "SANDBOX_MAX_IMAGES", 4)
        if len(img_paths) > max_images:
            excess = len(img_paths) - max_images
            notice = f"\n[提示] 单次执行生成图片超过上限 ({max_images} 张)，已忽略超出部分的 {excess} 张图片。"
            if stdout_text:
                stdout_text = f"{stdout_text}{notice}"
            else:
                stdout_text = notice.strip()
            img_paths = img_paths[:max_images]

        for img_path in img_paths:
            b64 = compress_image_to_base64(img_path, max_size=(800, 800), quality=80)
            if b64:
                images.append(b64)

    finally:
        # 无论成功与否，清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return SandboxResult(stdout=stdout_text, stderr=stderr_text, images=images)
