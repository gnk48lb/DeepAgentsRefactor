"""构建并发布 Modal 云端沙箱镜像（member 的 CodeAgent 用），随后在一个临时沙箱里做验收

运行时代码只用 Image.from_name(MODAL_IMAGE_NAME)，绝不在运行时构建镜像；镜像由你手动运行本脚本构建并发布一次。
构建会在 Modal 上产生少量费用，可能需要几分钟。镜像内容变了就升 tag（如 :v2）并改 .env 里的 MODAL_IMAGE_NAME，不要原地覆盖。

用法：
    uv run python scripts/build_modal_image.py [--name NAME[:TAG]] [--with-optional] [--force] [--check-only] [--yes]

    --name           要发布的镜像名（默认取 config.MODAL_IMAGE_NAME）
    --with-optional  额外安装 ffmpeg / nodejs / npm（体积大，默认不装）
    --force          强制重建（默认命中 Modal 的缓存时不会重建）
    --check-only     只做验收，不构建也不发布。这也是判断“镜像到底发布了没有”的唯一方式
    --yes            跳过交互确认

退出码：0 成功；1 失败（构建失败、验收有 ❌、镜像不存在、已取消）；2 缺少令牌或没装 modal；130 被 Ctrl-C 中断。
令牌只在 .env 里，本脚本不读取、不打印令牌；任何异常文本都先脱敏。
"""

from app import bootstrap
bootstrap.apply()

import argparse
import io
import sys
from typing import Any, Optional

from PIL import Image

from app import config
from app.infra import modal_image, modal_sandbox

_IMPORT_CHECK = (
    "import pandas, numpy, matplotlib, scipy, seaborn, openpyxl, xlrd, tabulate, yaml, sympy, sklearn, "
    "httpx, requests, bs4, lxml, PIL, pypdf, pdfplumber, docx, pptx, xlsxwriter, networkx, statsmodels"
)
_CLI_TOOLS = ("rg", "jq", "unzip", "zip", "file", "tree", "git", "curl", "wget", "pdftotext", "sqlite3", "dot", "timeout", "bash")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="构建并发布 Modal 云端沙箱镜像，然后在临时沙箱里验收（会产生少量 Modal 费用）"
    )
    parser.add_argument("--name", default=None, help="要发布的镜像名 NAME[:TAG]（默认取 MODAL_IMAGE_NAME）")
    parser.add_argument("--with-optional", action="store_true", help="额外安装 ffmpeg / nodejs / npm（体积大，默认不装）")
    parser.add_argument("--force", action="store_true", help="强制重建镜像（默认命中缓存时不会重建）")
    parser.add_argument("--check-only", action="store_true", help="只做验收，不构建也不发布")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认")
    return parser


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


class _Check:
    """逐项打印 ✅ / ❌ / ⚠️ 并统计失败数。"""

    def __init__(self):
        self.failed = 0

    def ok(self, label: str, detail: str = "") -> None:
        print(f"  ✅ {label}" + (f"：{detail}" if detail else ""))

    def fail(self, label: str, detail: str = "") -> None:
        self.failed += 1
        print(f"  ❌ {label}" + (f"：{detail}" if detail else ""))

    def warn(self, label: str, detail: str = "") -> None:
        print(f"  ⚠️ {label}" + (f"：{detail}" if detail else ""))

    def expect(self, label: str, condition: bool, detail: str = "") -> None:
        (self.ok if condition else self.fail)(label, detail)


def _run(sb: Any, argv: tuple, timeout: int = 120) -> tuple[Optional[int], str, str]:
    """在验收沙箱里跑一条命令，返回 (退出码, stdout, stderr)。"""
    proc = sb.exec(*argv, workdir="/work", timeout=timeout, text=True)
    out = "".join(proc.stdout)
    err = "".join(proc.stderr)
    return proc.wait(), out, err


def _freeze_map(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        if "==" in line:
            name, _, version = line.strip().partition("==")
            result[_norm(name)] = version.strip()
    return result


def _acceptance(modal: Any, app: Any, image_name: str) -> int:
    """在一个临时沙箱里验收镜像（创建参数与运行时一致）。沙箱在 finally 里终止，包括 Ctrl-C。返回失败项数。"""
    print(f"\n== 验收镜像 {image_name} ==")
    sb = None
    try:
        try:
            sb = modal.Sandbox.create(
                "sleep", "infinity",
                **modal_sandbox.sandbox_create_kwargs(
                    modal, app, tags={"svc": "gnk48", "role": "image-check"},
                    image_name=image_name, timeout=600, idle_timeout=120,
                ),
            )
        except Exception as e:
            exc_cls = getattr(getattr(modal, "exception", None), "NotFoundError", None)
            if exc_cls is not None and isinstance(e, exc_cls):
                print("❌ 镜像不存在：请去掉 --check-only 重新运行本脚本来构建并发布它。")
            else:
                print(f"❌ 无法创建验收沙箱：{modal_sandbox.explain(e)}")
            return 1

        c = _Check()
        code, out, _ = _run(sb, ("python", "--version"))
        c.expect("python 版本", code == 0 and modal_image.IMAGE_PYTHON_VERSION in out, out.strip())
        code, out, _ = _run(sb, ("python3", "--version"))
        c.expect("python3 版本", code == 0 and modal_image.IMAGE_PYTHON_VERSION in out, out.strip())
        code, out, _ = _run(sb, ("id", "-u"))
        c.expect("以 root 运行（id -u = 0）", code == 0 and out.strip() == "0", out.strip())
        code, out, _ = _run(sb, ("pwd",))
        c.expect("工作目录 pwd = /work", out.strip() == "/work", out.strip())
        code, out, _ = _run(sb, ("bash", "-c", "echo $HOME"))
        c.expect("HOME = /work", out.strip() == "/work", out.strip())
        code, _, _ = _run(sb, ("bash", "-c", "touch /work/outputs/.w && rm /work/outputs/.w"))
        c.expect("/work/outputs 可写", code == 0)

        code, _, err = _run(sb, ("python", "-c", _IMPORT_CHECK))
        c.expect("一次导入全部库", code == 0, err.strip()[-200:] if code else "")

        code, out, _ = _run(sb, ("pip", "freeze"))
        frozen = _freeze_map(out)
        mismatched = []
        for line in modal_image.read_pinned_requirements():
            if "==" not in line:
                continue
            name, _, version = line.partition("==")
            if frozen.get(_norm(name)) != version.strip():
                mismatched.append(f"{name}: 期望 {version.strip()}，实际 {frozen.get(_norm(name))}")
        c.expect("pip freeze 与 deploy/sandbox/requirements.txt 的固定版本一致", code == 0 and not mismatched, "; ".join(mismatched))
        print("  —— pip freeze（想钉 PIP_EXTRA 的版本时参考）——")
        for line in out.splitlines():
            print(f"     {line}")

        missing = []
        for tool in _CLI_TOOLS:
            code, _, _ = _run(sb, ("bash", "-c", f"command -v {tool}"))
            if code != 0:
                missing.append(tool)
        c.expect("命令行工具齐全", not missing, "缺少: " + ", ".join(missing) if missing else "")
        code, _, _ = _run(sb, ("find", "/work", "-maxdepth", "0", "-printf", ""))
        c.expect("GNU find 支持 -printf（产物收集依赖它）", code == 0)

        code, out, _ = _run(sb, ("bash", "-c", "echo $MPLBACKEND $MATPLOTLIBRC $MPLCONFIGDIR"))
        c.expect("matplotlib 环境变量", out.strip() == "Agg /opt/mplrc /opt/mplcache", out.strip())

        selfcheck = modal_image.font_selfcheck_py("/work/outputs/selftest.png")
        code, out, err = _run(sb, ("python", "-c", selfcheck))
        png_ok = False
        if code == 0:
            try:
                data = sb.filesystem.read_bytes("/work/outputs/selftest.png")
                with Image.open(io.BytesIO(data)) as img:
                    img.verify()
                png_ok = len(data) > 0
            except Exception as e:
                err = f"取回或校验 PNG 失败: {type(e).__name__}"
        c.expect("中文与负号出图（无 Glyph missing / findfont 噪声）", code == 0 and png_ok, (err.strip()[-200:] if not png_ok else ""))
        _run(sb, ("rm", "-f", "/work/outputs/selftest.png"))

        if config.MODAL_BLOCK_NETWORK:
            print("  ℹ️ MODAL_BLOCK_NETWORK=true，跳过公网检查")
        else:
            code, out, err = _run(
                sb, ("python", "-c", "import urllib.request; print(urllib.request.urlopen('https://pypi.org', timeout=10).status)"),
            )
            if code == 0 and out.strip() == "200":
                c.ok("公网可达（pypi.org）")
            else:
                c.warn("公网检查没通过（不算失败）", err.strip()[-120:])
        return c.failed
    finally:
        if sb is not None:
            try:
                sb.terminate(wait=False)
            except Exception:
                pass


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    name = (args.name or config.MODAL_IMAGE_NAME).strip()

    if not config.MODAL_TOKENS_PRESENT:
        print("❌ 没有配置 MODAL_TOKEN_ID / MODAL_TOKEN_SECRET：请把令牌写进 .env（令牌只放在 .env 里）后重试。")
        return 2
    try:
        modal = modal_sandbox.load_modal()
    except Exception as e:
        print(f"❌ 无法导入 modal 包（请先 uv sync）: {modal_sandbox.redact(f'{type(e).__name__}: {e}')[:200]}")
        return 2

    try:
        if not args.check_only:
            pinned = modal_image.read_pinned_requirements()
            apt = [*modal_image.APT_PACKAGES, *(modal_image.APT_OPTIONAL if args.with_optional else ())]
            print("== 构建计划 ==")
            print(f"  镜像名: {name}")
            print(f"  基础: debian_slim, Python {modal_image.IMAGE_PYTHON_VERSION}")
            print(f"  apt: {' '.join(apt)}")
            print(f"  pip: {' '.join([*pinned, *modal_image.PIP_EXTRA])}")
            print(f"  强制重建: {'是' if args.force else '否（命中 Modal 缓存时不重建）'}")
            print("⚠️ 这会在 Modal 上构建镜像，产生少量费用，可能需要几分钟。")
            if not args.yes:
                try:
                    answer = input("确认继续？输入 yes 继续: ").strip().lower()
                except EOFError:
                    answer = ""
                if answer not in ("yes", "y"):
                    print("已取消。")
                    return 1
            try:
                app = modal.App.lookup(config.MODAL_APP_NAME, create_if_missing=True)
                image = modal_image.build_image(modal, with_optional=args.with_optional, force=args.force)
                with modal.enable_output():
                    image.build(app)
                image.publish(name)
            except Exception as e:
                print(f"❌ 构建失败: {modal_sandbox.explain(e, limit=500)}")
                return 1
            print(f"✅ 镜像已发布：{name}")
        else:
            try:
                app = modal.App.lookup(config.MODAL_APP_NAME, create_if_missing=True)
            except Exception as e:
                print(f"❌ 无法连接 Modal: {modal_sandbox.explain(e)}")
                return 1

        try:
            failed = _acceptance(modal, app, name)
        except Exception as e:
            print(f"❌ 验收过程出错: {modal_sandbox.explain(e)}")
            failed = 1
    except KeyboardInterrupt:
        print("\n已中断（临时沙箱已清理）。")
        return 130

    if failed:
        print(f"\n❌ 验收有 {failed} 项没通过，请按上面的提示处理。")
        return 1
    print(f"\n✅ 镜像已发布并通过验收：{name}。如果与 .env 里的 MODAL_IMAGE_NAME 不同，请改成它。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
