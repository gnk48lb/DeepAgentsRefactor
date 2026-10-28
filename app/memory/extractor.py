"""后台异步长期记忆提取与持久化 (所在层: L2)"""

import string
import time
import asyncio
from app.infra import models
from app.runtime import cli_logger
from app.utils.text import extract_text
from app.memory import store as memory_store
from app.memory.injector import strip_memory_block

MEMORY_EXTRACTOR_PROMPT = """你是一个极其冷酷、挑剔且吝啬的“个人隐私与偏好提取官”。
你的目标是：只有当用户透露了【足以跨越数月甚至数年都有参考价值】的个人核心信息时，才进行记录。
绝大多数对话都应该是 NONE。

【严禁记录 - 只要包含以下特征，立即回复 NONE】
1. 任何关于“明天”、“今天”、“周几”、“几点”的行程、天气或临时打算。
2. 任何从工具、知识库、地图中查出的客观数据（如英雄技能、经纬度、路线指引、百科知识）。
3. 任何临时的询问意图（如“我想吃...”、“帮我查...”）。
4. 任何不涉及用户“本人”特征的客观事实。

【仅允许记录 - 必须符合以下条件之一】
1. 永久性的个人基本信息：家/公司的精确地址（需具体到门牌号或小区名）、真实姓名、生日、手机号。
   ⚠️ 特别注意：用户哪怕只是陈述性地说"我家在XX"或"我住在XX"，也必须立即提取并记录，无需等待用户提出具体需求。
2. 核心且长期的身体/习惯特征：如“对海鲜严重过敏”、“平时只喝冰美式”、“不吃任何辣的东西”。
3. 强烈的、相对稳定的个人关系或身份：如“我老婆叫小红”、“我是一名程序员”。

【输出要求】
- 符合条件：总结成一句话（如“用户家住在济南市文化东路42号”）。
- 不符合条件：**必须回复且只能回复** NONE。禁止解释原因，禁止加标点。

对话内容：
{recent_chat}
"""


_PUNCTUATION = string.punctuation + "。，、！？；：“”‘’（）《》【】…— \t\r\n"


def _is_none_reply(text: str) -> bool:
    """纯函数：去掉首尾空白与标点后 upper() == 'NONE' 才算无记忆。"""
    if not text:
        return False
    return text.strip(_PUNCTUATION).upper() == "NONE"


async def extract_and_store_memory(recent_chat: str):
    """后台异步执行记忆提取与保存，不阻塞主流程"""
    try:
        clean_chat = strip_memory_block(recent_chat)
        final_prompt = MEMORY_EXTRACTOR_PROMPT.format(recent_chat=clean_chat)
        mem_start = time.perf_counter()
        res = await models.extractor_llm.ainvoke(
            final_prompt,
            config={"callbacks": [cli_logger.CLIDebugCallbackHandler()]},
        )
        mem_elapsed = time.perf_counter() - mem_start
        content = extract_text(res.content).strip()
        if content and not _is_none_reply(content):
            await asyncio.to_thread(memory_store.insert_memory, content)
            print(f"--- \033[92m[后台记忆保存] (耗时 {mem_elapsed:.2f}s)\033[0m: {content} ---", flush=True)
        else:
            print(f"--- \033[90m[后台记忆跳过] (耗时 {mem_elapsed:.2f}s): 未发现长期价值信息\033[0m ---", flush=True)
    except Exception as e:
        print(f"⚠️ \033[91m[后台记忆提取异常]\033[0m: {str(e)}", flush=True)
