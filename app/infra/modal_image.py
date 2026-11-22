"""Modal 云端沙箱的命名镜像定义 (所在层: L1，纯定义，不导入 modal)

镜像由用户手动运行 scripts/build_modal_image.py 构建并发布一次；运行时只用 Image.from_name(config.MODAL_IMAGE_NAME)，
运行时代码里绝不构建镜像。镜像内容变了就升 tag（如 :v2）并改 .env 里的 MODAL_IMAGE_NAME，不要原地覆盖旧 tag。

build_image() 接收**已经导入的 modal 模块**（构建脚本通过 modal_sandbox.load_modal() 取得），返回未构建的 modal.Image，
所以这个模块本身不依赖 modal 包，测试里传 FakeModal 就能离线检查整条操作链。
依赖版本的唯一来源是 deploy/sandbox/requirements.txt：Docker 与 Modal 两个镜像都从它来。
"""

import base64
import os
from typing import Any

from app import config

# 与现有 Docker 镜像一致，保证 deploy/sandbox/requirements.txt 的固定版本能装
IMAGE_PYTHON_VERSION = "3.13"

# 前 7 个与 Docker 镜像一致，其余是云端有网络之后常用的
APT_PACKAGES = (
    "fonts-noto-cjk", "ripgrep", "jq", "unzip", "zip", "file", "tree",
    "procps", "git", "curl", "wget", "ca-certificates", "build-essential", "poppler-utils", "sqlite3", "graphviz",
)
# 体积大、不一定用得到，默认不装，由 --with-optional 开启
APT_OPTIONAL = ("ffmpeg", "nodejs", "npm")

# 在固定版本之外额外装的库（版本不钉；想钉的话参考验收脚本打印的 pip freeze）
PIP_EXTRA = (
    "httpx", "requests", "beautifulsoup4", "lxml", "pillow", "pypdf", "pdfplumber",
    "python-docx", "python-pptx", "xlsxwriter", "networkx", "statsmodels",
)

# 与 Docker 镜像的差异：MPLCONFIGDIR 直接用构建时预热过字体缓存的 /opt/mplcache
# （云端沙箱以 root 运行，该目录可写，不需要 Docker 那样另开 /tmp/mpl）
IMAGE_ENV = {
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    "LANG": "C.UTF-8",
    "MPLBACKEND": "Agg",
    "MATPLOTLIBRC": "/opt/mplrc",
    "MPLCONFIGDIR": "/opt/mplcache",
    "HOME": "/work",
}

# 构建期 / 验收期自检：渲染含中文与负号的图，断言没有 Glyph missing 警告与 findfont 噪声（内容取自 deploy/sandbox/Dockerfile）
_FONT_SELFCHECK_TEMPLATE = '''import sys, warnings, io
import matplotlib
import matplotlib.pyplot as plt

stderr_capture = io.StringIO()
old_stderr = sys.stderr
sys.stderr = stderr_capture

with warnings.catch_warnings(record=True) as warns:
    warnings.simplefilter("always")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [-1.5, 0.0, 2.5], label="示例曲线")
    ax.set_title("测试中文与负号 -1.5")
    ax.set_xlabel("横轴 (X)")
    ax.set_ylabel("纵轴 (Y)")
    ax.legend()
    fig.savefig("@@OUTPUT@@")
    plt.close(fig)

sys.stderr = old_stderr
err_out = stderr_capture.getvalue()
assert "findfont" not in err_out, f"findfont noise in stderr: {err_out}"
for w in warns:
    msg = str(w.message)
    assert "Glyph" not in msg and "missing" not in msg.lower(), f"Glyph missing warning: {msg}"
print("Font self check passed successfully.")
'''


def font_selfcheck_py(output_path: str = "/tmp/selftest.png") -> str:
    """自检脚本文本；output_path 是 PNG 的保存位置（构建期用 /tmp，验收期写到 /work/outputs/ 再取回校验）。"""
    return _FONT_SELFCHECK_TEMPLATE.replace("@@OUTPUT@@", output_path)


FONT_SELFCHECK_PY = font_selfcheck_py()


def read_pinned_requirements() -> list[str]:
    """读 deploy/sandbox/requirements.txt（忽略空行与 # 注释）。这个文件是依赖版本的唯一来源。"""
    path = os.path.join(config.PROJECT_ROOT, "deploy", "sandbox", "requirements.txt")
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def matplotlibrc_path() -> str:
    return os.path.join(config.PROJECT_ROOT, "deploy", "sandbox", "matplotlibrc")


def build_commands() -> list[str]:
    """镜像构建期在 shell 里依次跑的命令（脚本用 base64 传，免得引号转义出错）。"""
    selfcheck_b64 = base64.b64encode(FONT_SELFCHECK_PY.encode("utf-8")).decode("ascii")
    return [
        "mkdir -p /work/outputs /opt/mplrc /opt/mplcache",
        # deepagents 的文件工具都靠 `python3 -c ...`，防御性地保证 python3 存在
        'command -v python3 || ln -s "$(command -v python)" /usr/local/bin/python3',
        # 预热 matplotlib 字体缓存到 /opt/mplcache
        "MPLCONFIGDIR=/opt/mplcache python -c 'import matplotlib.pyplot'",
        f"echo {selfcheck_b64} | base64 -d | MPLCONFIGDIR=/opt/mplcache python",
        "rm -f /tmp/selftest.png",
    ]


def build_image(modal: Any, *, with_optional: bool = False, force: bool = False) -> Any:
    """返回未构建的 modal.Image。不建非 root 用户，不设 USER（隔离靠 Modal 沙箱本身）。"""
    apt = [*APT_PACKAGES, *(APT_OPTIONAL if with_optional else ())]
    pinned = read_pinned_requirements()
    image = modal.Image.debian_slim(python_version=IMAGE_PYTHON_VERSION, force_build=force)
    image = image.apt_install(*apt)
    image = image.pip_install(*pinned, *PIP_EXTRA)                      # 同一层，让 pip 一次解析
    # 必须 copy=True，否则文件不会被烘进发布的镜像
    image = image.add_local_file(matplotlibrc_path(), "/opt/mplrc/matplotlibrc", copy=True)
    image = image.env(dict(IMAGE_ENV))
    image = image.run_commands(*build_commands())
    image = image.workdir("/work")
    return image
