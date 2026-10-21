"""
tests/smoke_markitdown_e2e.py

端到端（E2E）验证：
通过 mcp_service.initialize_mcp() 真实拉起 markitdown MCP 服务进程，
调用包装后的 convert_to_markdown 转换工作区内的测试 HTML 文档，
验证成功返回 Markdown 文本。
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


from app import config
import app.infra.mcp_service as mcp_service
import app.tools.markitdown_tool as markitdown_tool


async def main():
    print("=" * 70)
    print("开始运行 MarkItDown MCP 端到端验证测试...")
    print("=" * 70)

    # 1. 在真实工作区创建测试文件
    ws_dir = Path(config.WORKSPACE_DIR).resolve()
    ws_dir.mkdir(parents=True, exist_ok=True)
    test_file = ws_dir / "test_sample.html"
    test_file.write_text(
        "<html><body><h1>DeepAgents Refactor</h1><p>Welcome to <b>MarkItDown MCP</b>!</p></body></html>",
        encoding="utf-8",
    )

    try:
        # 2. 初始化 MCP 服务
        await mcp_service.initialize_mcp()

        # 3. 提取 markitdown 工具并包装
        mcp_tools = mcp_service.get_tools_by_server("markitdown")
        print(f"获取到 markitdown 原始工具数量: {len(mcp_tools)}")
        assert any(t.name == "convert_to_markdown" for t in mcp_tools), "未在 markitdown 服务中找到 convert_to_markdown"

        wrapped = markitdown_tool.build_tools(mcp_tools, config.WORKSPACE_DIR)
        assert len(wrapped) == 1, f"包装后工具应为 1 个，实际为 {len(wrapped)}"
        tool = wrapped[0]
        assert tool.name == "convert_to_markdown"

        # 4. 执行虚拟路径转换调用
        print("正在调用包装后的 convert_to_markdown 转换 /workspace/test_sample.html ...")
        res = await tool.ainvoke({"uri": "/workspace/test_sample.html"})
        print("转换结果:")
        print("-" * 50)
        print(res.strip())
        print("-" * 50)

        assert "DeepAgents Refactor" in res, "转换结果缺失标题 'DeepAgents Refactor'"
        assert "MarkItDown MCP" in res, "转换结果缺失正文 'MarkItDown MCP'"
        print("✅ 端到端验证成功通过！")

    finally:
        # 清理测试文件和服务
        if test_file.exists():
            test_file.unlink()
        await mcp_service.shutdown_mcp()


if __name__ == "__main__":
    asyncio.run(main())
