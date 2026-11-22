"""安全沙箱 Python 代码执行工具 (所在层: L3)

本模块将 L1 沙箱返回的底层数据类转为上层可消费格式：
  1. 将高清图（长边 ≤ 1600px）通过 outbox 推送给终端用户。
  2. 将自检缩略图（长边 ≤ 768px）通过多模态 content 列表回传给模型。
  3. 将 stdout、stderr、执行状态、超时告警、工作区只读提示及 /work 现有文件统一格式化为文本。
  4. 绝不向外抛出异常，所有错误均转为对模型和用户友好的说明文本。
"""

import traceback
import typing
from langchain_core.tools import StructuredTool, tool

from app import config
from app.infra import modal_sandbox, sandbox
from app.infra.sandbox import SandboxUnavailable
from app.runtime import outbox
from app.utils.images import encode_image_bytes


def publish_images(
    images: list[sandbox.SandboxImage],
    notes: list[str] | None = None,
) -> tuple[list[str], list[dict[str, typing.Any]], list[str]]:
    """将产物图片推送到 outbox（用户高清版），并生成回传给模型的缩略图多模态 blocks。

    返回: (已发送图片名列表, 模型缩略图 blocks, 失败 notes 列表)
    """
    sent_image_names: list[str] = []
    model_blocks: list[dict[str, typing.Any]] = []
    fail_notes: list[str] = []

    for img in images:
        user_b64 = encode_image_bytes(
            img.data,
            max_side=config.SANDBOX_USER_IMAGE_MAX_SIDE,
            quality=88,
        )
        model_b64 = encode_image_bytes(
            img.data,
            max_side=config.SANDBOX_MODEL_IMAGE_MAX_SIDE,
            quality=75,
        )

        if user_b64:
            outbox.push({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{user_b64}"},
            })
            sent_image_names.append(img.name)
        else:
            fail_notes.append(f"图片 {img.name} 编码压缩失败，未发送给用户")

        if model_b64:
            model_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{model_b64}"},
            })

    return sent_image_names, model_blocks, fail_notes


def _run_code(get_sandbox: typing.Callable[[], typing.Any], code: str, *, workspace_hint: bool) -> typing.Any:
    """执行代码并把结果格式化成文本 / 多模态列表（owner 的 Docker 版与 member 的云端版共用）。

    get_sandbox 在调用时才取沙箱（测试会在运行时替换它）；绝不向外抛异常。
    """
    print(f"\n[Tool Call: execute_python_code] 代码长度: {len(code)} 字符")
    try:
        res = get_sandbox().run_python(code)
    except SandboxUnavailable as e:
        return f"代码执行沙箱不可用：\n{e}"
    except Exception as e:
        traceback.print_exc()
        return f"沙箱执行遇到未预期异常：{type(e).__name__}: {e}"

    # 处理只读工作区提示（只有 owner 的 Docker 版有 /workspace）
    stderr_text = res.stderr
    if workspace_hint and "Read-only file system" in stderr_text and ("/workspace" in stderr_text or "/workspace" in code):
        stderr_text += (
            "\n[提示] /workspace 为只读用户工作区，无法直接写入或修改文件。"
            "若需保存产物，请改存到 /work/outputs/；中间数据请保存在 /work/ 目录下。"
        )

    # 状态行（超时重点标出）
    if res.timed_out:
        status_line = f"⚠️ [执行超时中断] 退出码: {res.exit_code} | 耗时: {res.duration:.2f}s"
    else:
        status_line = f"执行完成 | 退出码: {res.exit_code} | 耗时: {res.duration:.2f}s"

    sections = [status_line]

    # Stdout 与 Stderr
    sections.append(f"Stdout:\n{res.stdout if res.stdout else '(无输出)'}")
    sections.append(f"Stderr:\n{stderr_text if stderr_text else '(无输出)'}")

    # Notes 提示信息
    all_notes = list(res.notes)

    # 处理图片与多模态
    sent_image_names, model_image_blocks, fail_notes = publish_images(res.images, res.notes)
    all_notes.extend(fail_notes)

    if sent_image_names:
        sections.append(f"已向用户发送 {len(sent_image_names)} 张图片（{', '.join(sent_image_names)}）")

    # 未导出的非图片产物
    if res.other_files:
        items = [f"{name} ({size} 字节)" for name, size in res.other_files]
        sections.append(f"未导出的非图片产物: {', '.join(items)}")

    # /work 顶层现有文件
    if res.work_listing:
        sections.append(f"/work 当前文件: {', '.join(res.work_listing)}")
    else:
        sections.append("/work 当前文件: (空)")

    if all_notes:
        sections.append("提示信息:\n" + "\n".join(f"• {n}" for n in all_notes))

    summary_text = "\n\n".join(sections)

    # 记录终端日志
    print(f"  stdout: {res.stdout[:200]}")
    if stderr_text:
        print(f"  stderr: {stderr_text[:200]}")
    print(f"  images: {len(res.images)} 张 (已发送: {len(sent_image_names)} 张)")

    # 多模态组装：有图片时返回 list，模型可查看缩略图自检
    if model_image_blocks:
        content: list[dict[str, typing.Any]] = [{"type": "text", "text": summary_text}]
        content.extend(model_image_blocks)
        return content
    else:
        return summary_text



@tool("execute_python_code")
def execute_python_code(code: str) -> typing.Any:
    """在本地 Docker 沙箱中执行 Python 代码，用于数学计算、数据处理与图表生成。

环境规范与事实约束：
1. 运行环境：Python 3.13，预装 pandas, numpy, matplotlib, scipy, seaborn, openpyxl, xlrd, tabulate, pyyaml, sympy, scikit-learn。无网络访问权限。
2. 目录布局：
   - /workspace：用户工作区目录，只读挂载！严禁尝试向 /workspace 写入或修改文件；分析数据请直接只读打开（如 pd.read_csv('/workspace/data.csv')）。
   - /work：沙箱会话工作目录（HOME 及当前工作路径），可读可写，会话内跨多次工具调用保留。变量不跨调用保留，但保存在 /work 中的文件会保留供后续调用读取。
   - /work/outputs/：专用产物导出目录。图片保存到 /work/outputs/ 即自动发送给用户，同一文件每个版本只发一次。若需绘图，代码必须用 savefig 将图片保存到 '/work/outputs/<英文名>.png'，禁止调用 plt.show()。
3. 限制：单次执行超时上限为 60 秒，内存上限 1GB。单次最多导出 4 张图片（单张 ≤ 10MB，总像素 ≤ 40M）。输出字符过多会被截断，请勿直接 print 整个超大 DataFrame。
"""
    return _run_code(lambda: sandbox.get_sandbox(), code, workspace_hint=True)

TOOLS = [execute_python_code]


def build_execute_python_code(get_sandbox: typing.Callable[[], typing.Any], *, description: str, workspace_hint: bool = False):
    """用给定的沙箱取用函数与工具描述造一个 execute_python_code 工具（参数签名仍是 code: str）。"""

    def _execute(code: str) -> typing.Any:
        return _run_code(get_sandbox, code, workspace_hint=workspace_hint)

    return StructuredTool.from_function(func=_execute, name="execute_python_code", description=description)


def member_execute_python_description() -> str:
    """member 版 execute_python_code 的工具描述：结构与 owner 版一致，但全部是云端事实；数字与联网句子调用时才从 config 现算。"""
    idle_minutes = max(1, (config.MODAL_IDLE_TIMEOUT_SECONDS + 30) // 60)
    net = (
        "沙箱不能联网，不能下载数据，也不能 pip install，只能用预装的库。"
        if config.MODAL_BLOCK_NETWORK
        else "沙箱可以联网，缺少的库可以 pip install。"
    )
    return (
        "在云端沙箱中执行 Python 代码，用于数学计算、数据处理与图表生成。\n\n"
        "环境规范与事实约束：\n"
        "1. 运行环境：Python 3.13，预装 pandas, numpy, matplotlib, scipy, seaborn, scikit-learn, statsmodels, sympy, "
        "openpyxl, networkx, Pillow, requests, httpx 等。" + net + "\n"
        "2. 目录布局：\n"
        f"   - /work：沙箱会话工作目录（HOME 及当前工作路径），可读可写，同一会话内跨多次工具调用保留；沙箱闲置约 {idle_minutes} 分钟后会被回收，"
        "文件随之丢失。变量不跨调用保留，但保存在 /work 中的文件会保留供后续调用读取。\n"
        "   - /work/outputs/：专用产物导出目录。图片保存到 /work/outputs/ 即自动发送给用户，同一文件每个版本只发一次。"
        "若需绘图，代码必须用 savefig 将图片保存到 '/work/outputs/<英文名>.png'，禁止调用 plt.show()。\n"
        f"3. 限制：单次执行默认 {config.MODAL_EXEC_TIMEOUT_SECONDS} 秒、上限 {config.MODAL_EXEC_MAX_TIMEOUT_SECONDS} 秒，"
        f"内存上限约 {config.MODAL_MEMORY_MB_LIMIT / 1024:g} GiB。"
        f"单次最多导出 {config.SANDBOX_MAX_IMAGES} 张图片（单张 ≤ {config.SANDBOX_MAX_IMAGE_MB}MB，总像素 ≤ 40M）。"
        "输出字符过多会被截断，请勿直接 print 整个超大 DataFrame。\n"
    )


def build_member_tools() -> list:
    """member 的 CodeAgent 的工具：只有云端版 execute_python_code（沙箱由 ModalRoutedSandbox 按请求上下文路由，调用时才取）。"""
    return [
        build_execute_python_code(
            lambda: modal_sandbox.get_routed_sandbox(),
            description=member_execute_python_description(),
        )
    ]
