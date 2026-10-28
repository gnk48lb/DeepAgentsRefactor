"""Playwright 浏览器自动化与网页抓取服务 (L1)"""

import os
from browser_use import Agent, Browser
from app import config

_browser = None

async def init_browser():
    """
    初始化浏览器实例并保留上下文
    """
    global _browser
    if _browser is None:
        user_data_dir = config.BROWSER_PROFILE_DIR
        
        print(f"\n[BrowserService] 🚀 正在启动可见模式浏览器...")
        print(f"[BrowserService] 📂 数据目录: {user_data_dir}")
        
        # 确保目录存在
        os.makedirs(user_data_dir, exist_ok=True)
        
        try:
            # 在 0.12.6 版本中，Browser (BrowserSession) 直接接受配置参数
            _browser = Browser(
                headless=False,  # 强制可见模式
                user_data_dir=user_data_dir,
                args=["--disable-blink-features=AutomationControlled"]
            )
            # 显式启动浏览器进程
            await _browser.start()
            print("[BrowserService] ✅ 浏览器进程启动成功，窗口应已弹出。")
        except Exception as e:
            print(f"[BrowserService] ❌ 浏览器启动失败: {str(e)}")
            _browser = None
            raise e
    return _browser


async def close_browser() -> None:
    """
    幂等关闭浏览器实例并释放进程资源。
    """
    global _browser
    if _browser is not None:
        try:
            await _browser.kill()
            print("[BrowserService] ✅ 浏览器进程已关闭。")
        except Exception as e:
            print(f"[BrowserService] ❌ 关闭浏览器失败: {e}")
        finally:
            _browser = None

# browser_service.py

BROWSER_ACTION_RULES = (
    "【⚠️ 动态网页操作红线准则】：\n"
    "1. 【严禁原地复读】：如果执行 click 或 input 工具后提示 Element index not available（索引不可用）或没有任何反应，绝对禁止在下一步重复尝试同一个 index！\n"
    "2. 【自救策略】：一旦遇到索引失效、网页无响应或被弹窗遮挡，你必须立即采取以下行动之一：\n"
    "   - 尝试使用 scroll_to_element 或主动滚动页面刷新视图。\n"
    "   - 放弃盲目点击 index，直接改用 evaluate 动作编写一小段强力的 JavaScript 代码进行文本精准匹配点击（例如：document.querySelectorAll('span').find(el => el.textContent.includes('原创')).click()）。\n"
    "3. 【处理弹窗】：上传完封面或勾选协议时，网页常会弹出“确认/确定”的蒙版遮罩弹窗。你必须优先寻找并点击该弹窗中的“确定”按钮（若点击失效，用 JS evaluate 强行清除或点击），直到弹窗彻底消失，才能去操作底部的后续按钮（如“提交稿件”）。"
)

async def run_browser_task(task_description: str) -> str:
    """
    运行 BrowserAgent 任务
    """
    try:
        if "gemini" in config.BROWSER_VLM_MODEL.lower():
            from browser_use.llm.google.chat import ChatGoogle
            browser_llm = ChatGoogle(
                model=config.BROWSER_VLM_MODEL,
                api_key=config.GEMINI_API_KEY,
            )
        else:
            raise ValueError("BROWSER_VLM_MODEL 必须是 Gemini 模型")
            
        browser = await init_browser()

        # ── 🛡️ 路径绝对化与安全白名单升级 ──
        import re
        workspace_raw = config.WORKSPACE_DIR
        workspace_abs = os.path.abspath(workspace_raw)
        
        # ⭐【核心修复】：仅将相对路径形式的 workspace 替换为绝对路径
        task_description_abs = re.sub(
            r"(?<![\w:/\\.])workspace([\\/])",
            lambda m: workspace_abs + m.group(1),
            task_description,
        )

        print(f"[BrowserService] 📝 任务描述中的路径已全自动绝对化:\n-> {task_description_abs}")

        # 生成完备的白名单
        available_paths = [workspace_raw, workspace_abs]
        if os.path.exists(workspace_abs):
            for root, _, files in os.walk(workspace_abs):
                for f in files:
                    fp = os.path.join(root, f)
                    available_paths.extend([os.path.abspath(fp), fp])
        
        cleaned_paths = []
        for p in set(available_paths):
            if isinstance(p, str):
                cleaned_paths.extend([p, p.replace('/', '\\'), p.replace('\\', '/')])
                
        final_whitelist = list(set(cleaned_paths))

        # 实例化 Agent，传入被绝对化后的任务文本与底层操作规则
        agent = Agent(
            task=task_description_abs,  # 👈 完美的绝对路径任务
            llm=browser_llm,
            browser=browser,
            available_file_paths=final_whitelist,
            extend_system_message=BROWSER_ACTION_RULES,
        )
        
        result = await agent.run()
        
        final_result = result.final_result()
        if final_result:
            return final_result
        return "Browser task completed, but no specific text was returned."
    except Exception as e:
        import traceback
        error_msg = f"BrowserAgent 执行失败: {str(e)}"
        print(f"\n[BrowserAgent Error]\n{traceback.format_exc()}")
        return error_msg
        
def _extract_text_from_html(html: str) -> str:
    """从 HTML 中剥离无用标签，提取干净的正文文本"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    # 移除噪音标签
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # 优先提取主内容区（role=main, main, article, 或 body）
    main_content = (
        soup.find(attrs={"role": "main"})
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=lambda c: c and any(k in str(c).lower() for k in ["content", "article", "post", "main", "body"]))
        or soup.body
        or soup
    )
    text = main_content.get_text(separator="\n", strip=True)
    import re
    return re.sub(r"\n{3,}", "\n\n", text)


async def fetch_generic_webpage_content(url: str) -> str:
    import httpx

    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
    }

    # 1. 优先尝试 Jina Reader（若配置了 JINA_API_KEY 或未被风控）
    jina_key = config.JINA_API_KEY
    jina_headers = dict(default_headers)
    if jina_key:
        jina_headers["Authorization"] = f"Bearer {jina_key}"

    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(jina_url, headers=jina_headers)
            if response.status_code == 200 and response.text.strip():
                return response.text
            # 若 Jina 出现 401（匿名 IP 风控）/ 402 / 403 / 429 等，输出提示并自动回退为本地直连抓取
            print(f"[BrowserService] ℹ️ Jina API 返回状态码 {response.status_code}，自动降级为本地直接抓取: {url}")
    except Exception as e:
        print(f"[BrowserService] ℹ️ Jina API 请求异常 ({e})，自动降级为本地直接抓取: {url}")

    # 2. 本地直接抓取降级方案（Direct HTTP Fetch + BeautifulSoup Text Extraction）
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=default_headers) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return _extract_text_from_html(resp.text)
            return f"Error: 抓取网页失败，目标网站响应状态码 HTTP {resp.status_code}。"
    except Exception as e:
        return f"Error fetching webpage: {str(e)}"

async def fetch_bilibili_profile_data(uid: str) -> str:
    from bilibili_api import user
    import json
    try:
        u = user.User(int(uid))
        info = await u.get_user_info()
        dynamics = await u.get_dynamics(offset=0)
        
        result = {
            "user_info": info,
            "recent_dynamics": dynamics.get("cards", [])[:10] if isinstance(dynamics, dict) else dynamics[:10]
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching Bilibili profile: {str(e)}"
