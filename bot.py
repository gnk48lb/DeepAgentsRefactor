import nonebot
from nonebot.adapters.qq import Adapter as QQAdapter
import sys
import os

# 确保能正确导入 app 等模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 初始化 NoneBot
nonebot.init()

# 注册 QQ 适配器
driver = nonebot.get_driver()
driver.register_adapter(QQAdapter)

# 加载 plugins 文件夹中的插件
nonebot.load_plugins("plugins")

if __name__ == "__main__":
    nonebot.run()
