import os
from browser_use import Agent, Browser

_browser = None

async def init_browser():
    """
    初始化浏览器实例并保留上下文
    """
    global _browser
    if _browser is None:
        # 修正：获取项目根目录 (app 的上一级)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        user_data_dir = os.path.join(project_root, "storage", "browser_profile")
        
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

# browser_service.py

async def run_browser_task(task_description: str, llm) -> str:
    """
    运行 BrowserAgent 任务
    """
    try:
        import config
        
        if "gemini" in config.BROWSER_VLM_MODEL.lower():
            from browser_use.llm.google.chat import ChatGoogle
            browser_llm = ChatGoogle(
                model=config.BROWSER_VLM_MODEL,
                api_key=config.GEMINI_API_KEY,
                temperature=0
            )
        else:
            from browser_use.llm.openai.chat import ChatOpenAI
            browser_llm = ChatOpenAI(
                model=config.BROWSER_VLM_MODEL,
                api_key=config.GITHUB_TOKEN,
                base_url=config.GITHUB_BASE_URL,
                temperature=0
            )
            
        browser = await init_browser()

        # ── 🛡️ 路径绝对化与安全白名单升级 ──
        import re
        workspace_raw = config.WORKSPACE_DIR
        workspace_abs = os.path.abspath(workspace_raw)
        
        # ⭐【核心修复】：直接在任务文本中，把所有相对路径强行替换为绝对路径！
        # 这样 AI 读到的指令就是“上传 d:\code\...”，调用工具时也会用绝对路径，彻底根治浏览器文件读取失败
        task_description_abs = task_description
        if "workspace\\" in task_description_abs:
            task_description_abs = task_description_abs.replace("workspace\\", workspace_abs + "\\")
        if "workspace/" in task_description_abs:
            task_description_abs = task_description_abs.replace("workspace/", workspace_abs + "/")

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

        # 实例化 Agent，传入被绝对化后的任务文本
        agent = Agent(
            task=task_description_abs,  # 👈 完美的绝对路径任务
            llm=browser_llm,
            browser=browser,
            available_file_paths=final_whitelist
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
        
async def fetch_generic_webpage_content(url: str) -> str:
    import httpx
    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(jina_url)
            if response.status_code == 200:
                return response.text
            return f"Error: Received status code {response.status_code} from Jina API."
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
