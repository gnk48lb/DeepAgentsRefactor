from app import bootstrap
bootstrap.apply()

import nonebot
from nonebot.adapters.qq import Adapter as QQAdapter

# 初始化 NoneBot
nonebot.init()

# 注册 QQ 适配器
driver = nonebot.get_driver()
driver.register_adapter(QQAdapter)

# 加载 plugins 文件夹中的插件
nonebot.load_plugins("plugins")

if __name__ == "__main__":
    nonebot.run()
