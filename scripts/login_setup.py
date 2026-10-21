import argparse
import asyncio
import os
import sys
from playwright.async_api import async_playwright
from app import config

# 定义浏览器数据的持久化保存目录
USER_DATA_DIR = config.BROWSER_PROFILE_DIR


async def main():
    parser = argparse.ArgumentParser(description="Browser Login Setup Helper")
    parser.add_argument(
        "urls",
        nargs="*",
        default=["https://www.bilibili.com"],
        help="目标网站 URL（可指定多个，默认 https://www.bilibili.com）",
    )
    args = parser.parse_args()
    urls = args.urls

    print("初始化环境...")
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    print(f"浏览器数据将保存在: {USER_DATA_DIR}")

    async with async_playwright() as p:
        print("启动浏览器中...")
        browser_context = await p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=False,
            viewport={"width": 1280, "height": 720},
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )

        print("\n" + "=" * 50)
        print("🚀 浏览器已启动！请在弹出的窗口中进行手动登录。")
        print("=" * 50 + "\n")

        for idx, url in enumerate(urls):
            if idx == 0:
                pages = browser_context.pages
                page = pages[0] if pages else await browser_context.new_page()
            else:
                page = await browser_context.new_page()
            print(f"👉 正在打开第 {idx+1}/{len(urls)} 个目标页面: {url}")
            await page.goto(url)

        input("\n⏳ 请在浏览器中完成所有的登录操作。完成后请在此处按下【回车键】保存并退出...")

        print("\n💾 正在保存登录状态并安全关闭浏览器...")
        await browser_context.close()
        print("✅ 登录状态已成功持久化保存！BrowserAgent 将复用该登录配置。")


if __name__ == "__main__":
    asyncio.run(main())