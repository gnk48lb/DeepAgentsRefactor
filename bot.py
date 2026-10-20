import sys
import os

# 确保在 Windows 终端下正确输出 UTF-8 编码字符与 ANSI 颜色
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import nonebot
from nonebot.adapters.qq import Adapter as QQAdapter

# 确保能正确导入 app 等模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# 压低 faiss 与 milvus 的日志噪音
logging.getLogger("milvus_lite").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("pymilvus").setLevel(logging.WARNING)

# 初始化 NoneBot
nonebot.init()

# 注册 QQ 适配器
driver = nonebot.get_driver()
driver.register_adapter(QQAdapter)

# 加载 plugins 文件夹中的插件
nonebot.load_plugins("plugins")

if __name__ == "__main__":
    nonebot.run()
