"""
app/sandbox.py
==============
本地 Docker 代码执行沙箱。

功能：
  1. 将用户提交的 Python 代码写入临时目录中的 main.py。
  2. 启动 Docker 容器（无网络，内存受限）执行代码。
  3. 截断过长的 stdout/stderr，防止撑爆 LLM 上下文。
  4. 扫描 outputs/ 目录，将生成的图片压缩为 Base64 返回。
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
import config
from .loader import compress_image_to_base64

# stdout / stderr 最大保留字符数（防止 LLM 上下文溢出）
_MAX_OUTPUT_CHARS = 1000
_TRUNCATION_SUFFIX = "\n...(已截断)"

# Docker 容器执行超时（秒）
_TIMEOUT_SECONDS = 30


class SandboxResult(TypedDict):
    stdout: str
    stderr: str
    images: list  # list[str], base64 编码


def _truncate(text: str) -> str:
    """如果文本超过 _MAX_OUTPUT_CHARS，截取前段并附上截断提示。"""
    if len(text) > _MAX_OUTPUT_CHARS:
        return text[:_MAX_OUTPUT_CHARS] + _TRUNCATION_SUFFIX
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
    # 确保在项目根目录下创建一个隐藏的临时沙箱根目录，规避 Windows 跨盘符挂载 Docker 报错
    sandbox_tmp_root = os.path.join(config.WORKSPACE_DIR, ".sandbox_tmp")
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
                network_disabled=True,
                detach=True,
                stdout=True,
                stderr=True,
            )

            try:
                # 等待容器结束，支持超时
                wait_res = container.wait(timeout=_TIMEOUT_SECONDS)
                # 获取所有日志
                logs = container.logs().decode("utf-8", errors="replace")
                stdout_text = _truncate(logs)
                
                # 如果退出码不为 0，记录到 stderr
                exit_code = wait_res.get("StatusCode", 0)
                if exit_code != 0:
                    stderr_text = f"容器以错误码 {exit_code} 退出。"
            except Exception as e:
                # 超时处理
                container.kill()
                stderr_text = _truncate(f"Docker 执行超时 ({_TIMEOUT_SECONDS}s) 或异常: {e}")
            finally:
                # 手动清理容器
                try:
                    container.remove(force=True)
                except:
                    pass

        except Exception as e:
            stderr_text = _truncate(f"Docker 启动失败: {type(e).__name__}: {e}")

        # 扫描 outputs/ 目录，收集图片
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_path in glob.glob(os.path.join(outputs_dir, ext)):
                b64 = compress_image_to_base64(img_path, max_size=(800, 800), quality=80)
                if b64:
                    images.append(b64)

    finally:
        # 无论成功与否，清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return SandboxResult(stdout=stdout_text, stderr=stderr_text, images=images)
