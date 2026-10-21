"""安全沙箱 Python 代码执行工具 (所在层: L3)"""

import typing
import traceback
from langchain_core.tools import tool
from app.infra.sandbox import run_code_in_sandbox
from app.runtime import outbox


@tool("execute_python_code")
def execute_python_code(code: str) -> typing.Any:
    """
    在本地 Docker 沙箱中执行 Python 代码，用于数据分析、数学计算和图表生成。
    沙箱已预装 pandas、numpy、matplotlib。
    若需绘图，代码中必须将图片以 savefig 保存到 '/workspace/outputs/' 目录，且以 .png 结尾，禁止调用 plt.show()。
    返回执行结果（包含 stdout、stderr）及生成的图片（若有）。
    """
    print(f"\n[Tool Call: execute_python_code] 代码长度: {len(code)} 字符")
    try:
        res = run_code_in_sandbox(code)

        # 记录终端日志（仅 stdout 前 200 字符，避免刷屏）
        print(f"  stdout: {res['stdout'][:200]}")
        if res["stderr"]:
            print(f"  stderr: {res['stderr'][:200]}")
        print(f"  images: {len(res['images'])} 张")

        # 【多模态组装】：有图片时返回 list，与现有拦截器无缝对接
        if res["images"]:
            content = [
                {
                    "type": "text",
                    "text": f"Stdout:\n{res['stdout']}\nStderr:\n{res['stderr']}"
                }
            ]
            for img_b64 in res["images"]:
                img_item = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                }
                outbox.push(img_item)
                content.append(img_item)
            return content
        else:
            # 无图片时返回纯文本，保持简洁
            return f"Stdout:\n{res['stdout']}\nStderr:\n{res['stderr']}"

    except Exception as e:
        traceback.print_exc()
        return f"沙箱执行失败: {type(e).__name__}: {e}"


TOOLS = [execute_python_code]
