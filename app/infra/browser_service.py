"""Playwright 浏览器自动化与网页抓取服务 (L1)"""

import json
import os
import urllib.parse
import httpx
from browser_use import Agent, Browser
from app import config

_browser = None
_http_client: httpx.AsyncClient | None = None


def _redact_key(text: str, key: str) -> str:
    """脱敏字符串中出现的敏感 API Key"""
    if key and key in text:
        return text.replace(key, "***")
    return text

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
        
async def fetch_generic_webpage_content(url: str) -> str:
    """使用 Tavily Extract API 云端提取网页 Markdown 内容 (L1)"""
    clean_url = (url or "").strip()
    try:
        parsed = urllib.parse.urlparse(clean_url)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            return "Error: 无效的 URL，必须是以 http:// 或 https:// 开头的完整网址。"
    except Exception:
        return "Error: 无效的 URL，必须是以 http:// 或 https:// 开头的完整网址。"

    api_key = (getattr(config, "TAVILY_API_KEY", None) or "").strip()
    if not api_key:
        return "Error: 未配置 TAVILY_API_KEY，无法调用云端网页解析服务。"

    endpoint = "https://api.tavily.com/extract"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "urls": [clean_url],
        "extract_depth": "basic",
        "format": "markdown",
    }

    try:
        if _http_client is not None:
            response = await _http_client.post(endpoint, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.post(endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            err_detail = ""
            try:
                err_json = response.json()
                if isinstance(err_json, dict):
                    err_detail = (
                        err_json.get("detail")
                        or err_json.get("error")
                        or err_json.get("message")
                        or response.text[:200]
                    )
                else:
                    err_detail = response.text[:200]
            except Exception:
                err_detail = response.text[:200]
            safe_err = _redact_key(str(err_detail or ""), api_key)
            return f"Error: Tavily Extract 接口请求失败 (HTTP {response.status_code}): {safe_err}"

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError):
            return "Error: Tavily Extract 返回了非法的 JSON 响应。"

        results = data.get("results") or []
        failed_results = data.get("failed_results") or []

        if results and isinstance(results[0], dict):
            raw_content = (results[0].get("raw_content") or "").strip()
            if raw_content:
                return raw_content

        if failed_results and isinstance(failed_results[0], dict):
            err_msg = failed_results[0].get("error") or "Unknown extraction error"
            safe_msg = _redact_key(str(err_msg), api_key)
            return f"Error: 网页内容提取失败 ({clean_url}): {safe_msg}"

        return f"Error: 网页内容提取为空 ({clean_url})，目标页面可能无有效正文或受反爬限制。"

    except httpx.TimeoutException:
        return f"Error: 网页内容提取超时 ({clean_url})。"
    except Exception as e:
        safe_err = _redact_key(str(e), api_key)
        return f"Error: 网页内容提取请求异常: {safe_err}"


