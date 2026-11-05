"""安全沙箱 Python 代码执行工具 (所在层: L3)

本模块将 L1 沙箱返回的底层数据类转为上层可消费格式：
  1. 将高清图（长边 ≤ 1600px）通过 outbox 推送给终端用户。
  2. 将自检缩略图（长边 ≤ 768px）通过多模态 content 列表回传给模型。
  3. 将 stdout、stderr、执行状态、超时告警、工作区只读提示及 /work 现有文件统一格式化为文本。
  4. 绝不向外抛出异常，所有错误均转为对模型和用户友好的说明文本。
"""

import typing
import traceback
from langchain_core.tools import tool

from app import config
from app.infra import sandbox
from app.infra.sandbox import SandboxUnavailable
from app.runtime import outbox
from app.utils.images import encode_image_bytes


@tool("execute_python_code")
def execute_python_code(code: str) -> typing.Any:
    """在本地 Docker 沙箱中执行 Python 代码，用于数学计算、数据处理与图表生成。

环境规范与事实约束：
1. 运行环境：Python 3.13，预装 pandas, numpy, matplotlib, scipy, seaborn, openpyxl, xlrd, tabulate, pyyaml, sympy, scikit-learn。无网络访问权限。
2. 目录布局：
   - /workspace：用户工作区目录，只读挂载！严禁尝试向 /workspace 写入或修改文件；分析数据请直接只读打开（如 pd.read_csv('/workspace/data.csv')）。
   - /work：沙箱会话工作目录（HOME 及当前工作路径），可读可写，会话内跨多次工具调用保留。变量不跨调用保留，但保存在 /work 中的文件会保留供后续调用读取。
   - /work/outputs/：专用产物导出目录。每次执行前会被清空重建。若需绘图，代码必须用 savefig 将图片保存到 '/work/outputs/<英文名>.png'，禁止调用 plt.show()。
3. 限制：单次执行超时上限为 60 秒，内存上限 1GB。单次最多导出 4 张图片（单张 ≤ 10MB，总像素 ≤ 40M）。输出字符过多会被截断，请勿直接 print 整个超大 DataFrame。
"""
    print(f"\n[Tool Call: execute_python_code] 代码长度: {len(code)} 字符")
    try:
        res = sandbox.get_sandbox().run_python(code)
    except SandboxUnavailable as e:
        return f"代码执行沙箱不可用：\n{e}"
    except Exception as e:
        traceback.print_exc()
        return f"沙箱执行遇到未预期异常：{type(e).__name__}: {e}"

    # 处理只读工作区提示
    stderr_text = res.stderr
    if "Read-only file system" in stderr_text and ("/workspace" in stderr_text or "/workspace" in code):
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

    # 处理图片
    sent_image_names: list[str] = []
    model_images_b64: list[str] = []

    for img in res.images:
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
            all_notes.append(f"图片 {img.name} 编码压缩失败，未发送给用户")

        if model_b64:
            model_images_b64.append(model_b64)

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
    if model_images_b64:
        content: list[dict[str, typing.Any]] = [{"type": "text", "text": summary_text}]
        for b64 in model_images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        return content
    else:
        return summary_text


TOOLS = [execute_python_code]
