"""网页抓取与浏览器自动化工具 (所在层: L3)"""

from langchain_core.tools import tool
from app import config
from app.infra.browser_service import fetch_generic_webpage_content, fetch_bilibili_profile_data, run_browser_task


@tool
async def read_webpage_content(url: str) -> str:
    """用于分析普通网页、读取新闻/文章内容、总结网页。当用户扔来一个网页链接并要求“总结这个网页”、“这个网页说了什么”时，必须优先调用此工具。注意：输入必须是完整的 URL。"""
    content = await fetch_generic_webpage_content(url)
    max_chars = getattr(config, "WEBPAGE_MAX_CHARS", 30000)
    if len(content) > max_chars:
        return f"{content[:max_chars]}\n…（内容过长，已截断至前 {max_chars} 字符）"
    return content


@tool
async def get_bilibili_profile(uid: str) -> str:
    """专门用于获取B站(bilibili)用户的个人主页资料和近期动态。当用户要求“总结这个B站个人主页”、“分析这个B站UP主”时调用。输入必须是从链接中提取的纯数字 UID。"""
    content = await fetch_bilibili_profile_data(uid)
    max_chars = getattr(config, "BILI_PROFILE_MAX_CHARS", 8000)
    if len(content) > max_chars:
        return f"{content[:max_chars]}\n…（内容过长，已截断至前 {max_chars} 字符）"
    return content


@tool
async def execute_complex_browser_action(instruction: str) -> str:
    """用于需要模拟真人操作浏览器的复杂网页任务，例如：在网页上点击按钮、输入文字搜索、登录、翻页，或是处理极度复杂的动态交互网页。仅当普通的网页内容读取(read_webpage_content)无法满足需求，或用户明确要求“操作”、“点击”时才使用此工具。"""
    # 这里直接调用原有的服务函数，传入配置好的 VLM
    return await run_browser_task(instruction)


TOOLS = [read_webpage_content, get_bilibili_profile, execute_complex_browser_action]
