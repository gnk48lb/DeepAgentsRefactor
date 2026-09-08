import asyncio
import os
from playwright.async_api import async_playwright

# 定义浏览器数据的持久化保存目录
# 这个目录必须和你的 app/browser_service.py 中配置的 user_data_dir 保持完全一致！
# 这里假设保存在项目根目录下的 data/browser_profile 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
USER_DATA_DIR = os.path.join(PROJECT_ROOT, "storage", "browser_profile")

async def main():
    print("初始化环境...")
    # 确保目录存在
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    print(f"浏览器数据将保存在: {USER_DATA_DIR}")

    async with async_playwright() as p:
        # 以有头模式（headless=False）启动浏览器，这样你才能看到界面去扫码
        # args 里面加了一些反反爬虫的常规参数
        print("启动浏览器中...")
        browser_context = await p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,
            viewport={"width": 1280, "height": 720},
            args=[
                "--disable-blink-features=AutomationControlled",
            ]
        )

        print("\n" + "="*50)
        print("🚀 浏览器已启动！请在弹出的窗口中进行手动登录。")
        print("="*50 + "\n")

        # 获取当前打开的页面
        pages = browser_context.pages
        page = pages[0] if pages else await browser_context.new_page()

        # 示例 1：打开 Bilibili
        print("👉 正在打开 Bilibili，请去网页上扫码或密码登录...")
        # await page.goto("https://www.bilibili.com")
        
        # # 示例 2：你可以新建标签页打开小红书（可选）
        # page2 = await browser_context.new_page()
        # await page2.goto("https://www.xiaohongshu.com/explore")
        # print("👉 也在新标签页打开了小红书，如果你需要的话也请登录...")
        await page.goto("https://ikuuu.win/auth/login")

        # --- 核心：程序在这里挂起，等待你的操作 ---
        input("\n⏳ 请在浏览器中完成所有的登录。确认页面上都已经显示你的头像后，在这里按下【回车键】结束并保存...")

        print("\n💾 正在保存登录状态并安全关闭浏览器...")
        await browser_context.close()
        print("✅ 登录状态已成功持久化保存！现在你可以去启动你的 NoneBot 主程序了。")

if __name__ == "__main__":
    asyncio.run(main())