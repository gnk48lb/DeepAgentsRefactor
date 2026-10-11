import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 设置标准输出编码为 utf-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

async def main():
    print("=== Step 1: initialize_mcp ===")
    from app import mcp_service
    await mcp_service.initialize_mcp()

    amap_tools = mcp_service.get_tools_by_server("amap")
    fs_tools = mcp_service.get_tools_by_server("filesystem")
    desktop_tools = mcp_service.get_tools_by_server("desktop")
    print(f"amap tools count: {len(amap_tools)}")
    print(f"filesystem tools count: {len(fs_tools)}")
    print(f"desktop tools count: {len(desktop_tools)}")

    # 检查 FileAgent 过滤后的工具列表
    filtered_fs = [t for t in fs_tools if t.name != "read_media_file"]
    print(f"filesystem tools after removing read_media_file: {len(filtered_fs)}")
    print(f"has read_media_file in filtered_fs: {'read_media_file' in [t.name for t in filtered_fs]}")

    print("\n=== Step 2: B1 (a) execute_python_code ===")
    from app import tools
    code = """import matplotlib.pyplot as plt
x = [1, 2, 3]
y = [1, 4, 9]
plt.plot(x, y)
plt.title("Test Plot")
plt.savefig("/workspace/outputs/a.png")
"""
    try:
        res = tools.execute_python_code.invoke({"code": code})
        print(f"execute_python_code return type: {type(res)}")
        if isinstance(res, list):
            img_count = sum(1 for item in res if isinstance(item, dict) and item.get("type") == "image_url")
            print(f"image_url count: {img_count}")
        elif isinstance(res, str):
            print(f"return string: {res[:200]}")
    except Exception as e:
        print(f"execute_python_code raised exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
