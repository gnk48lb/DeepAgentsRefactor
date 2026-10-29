"""后台异步长期记忆两阶段提取、对账与版本化持久化 (所在层: L2)"""

import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from app import config
from app.infra import models
from app.runtime import cli_logger
from app.utils.text import extract_text
from app.memory import store as memory_store
from app.memory.injector import strip_memory_block
from app.memory.guards import (
    validate_statement,
    validate_reconciliation_ops,
)

T = TypeVar("T", bound=BaseModel)

MEMORY_EXTRACTOR_PROMPT = """你是一个极其冷酷、挑剔且吝啬的“个人隐私与偏好提取官”。
你的目标是：只有当用户透露了【足以跨越数月甚至数年都有参考价值】的个人核心信息时，才进行记录。
绝大多数对话都应该是空的（不产生任何陈述）。

【严禁记录 - 只要包含以下特征，绝不记录】
1. 任何关于“明天”、“今天”、“周几”、“几点”的行程、天气或临时打算。
2. 任何从工具、知识库、地图中查出的客观数据（如英雄技能、经纬度、路线指引、百科知识）。
3. 任何临时的询问意图（如“我想吃...”、“帮我查...”）。
4. 任何不涉及用户“本人”特征的客观事实。

【仅允许记录 - 必须符合以下条件之一】
1. 永久性的个人基本信息：家/公司的精确地址（需具体到门牌号或小区名）、真实姓名、生日、手机号。
   ⚠️ 特别注意：用户哪怕只是陈述性地说"我家在XX"或"我住在XX"，也必须立即提取并记录，无需等待用户提出具体需求。
2. 核心且长期的身体/习惯特征：如“对海鲜严重过敏”、“平时只喝冰美式”、“不吃任何辣的东西”。
3. 强烈的、相对稳定的个人关系或身份：如“我老婆叫小红”、“我是一名程序员”。

【输出要求与原则】
1. 只处理用户本人关于自己的陈述（粘贴/转发的第三方文本、引用、代码里的第一人称不算）。
2. 每条陈述必须原子化（一条陈述仅包含单一事实，多个事实必须拆开）、且以第三人称（"用户…"）陈述。
3. 区分断言事实（facts）与撤回/遗忘请求（retractions）：
   - 用户断言的新事实放入 facts；
   - 用户明确撤回、纠正或要求忘记的请求（如"我不再对海鲜过敏"、"忘掉我的住址"），只产出 retraction，绝不产出否定式事实；retractions 里的 text 必须写成被撤回事实本身的第三人称陈述（如"用户的家庭住址"、"用户对海鲜过敏"），用于检索召回旧记忆。
4. 每条陈述必须包含 evidence 字段，必须逐字摘自用户原话的片段（子串且长度>=2）；没有原话依据的绝不记录。
5. 若没有符合长期价值的信息，返回空列表。

{format_instructions}

用户本轮原话：
{user_text}
"""


MEMORY_RECONCILER_PROMPT = """你是一个极其严谨、客观的“个人隐私与偏好提取官”（阶段2：长期记忆对账仲裁员）。
你的职责是将当轮提取出的原子陈述（S1, S2...）与从记忆库中召回的旧记忆（M1, M2...）进行对账仲裁。

【候选旧记忆列表】
{candidates_block}

【当轮提取陈述列表】
{statements_block}

【操作判定规则】
1. 对每条陈述，判定操作为 ADD、UPDATE、DELETE、NONE 之一：
   - ADD：新发现的独立事实，与已有旧记忆无冲突、非取代关系。
   - UPDATE：同一事实属性的新值取代旧值（如搬家换地址、换手机号、改名、口味习惯改变）。必须提供 target 指向被取代的旧记忆（如 M1）。
   - DELETE：用户明确要求撤回、遗忘旧记忆。必须提供 target 指向被撤回的旧记忆（如 M1）。
   - NONE：拿不准、信息冗余、已有完全等价记忆，或者无法确定对应关系时，必须判定为 NONE（宁缺毋滥）。
2. 边界约束：
   - 标注为 [assert] 的陈述只能判定为 ADD、UPDATE 或 NONE，严禁判为 DELETE。
   - 标注为 [retract] 的陈述只能判定为 DELETE 或 NONE，严禁判为 ADD 或 UPDATE。
   - UPDATE 和 DELETE 必须指定具体的 target（旧记忆别名如 M1）；ADD 和 NONE 的 target 为 null。
   - 严禁合并多条事实进同一条；补充性的不同事实一律 ADD。
   - 严禁产出陈述之外的新文本。
   - 若候选列表中已有完全等价的记忆，请判定为 NONE，无需重复添加。

{format_instructions}
"""


class FactStatement(BaseModel):
    text: str = Field(description="原子事实的第三人称陈述，以'用户'开头")
    evidence: str = Field(description="逐字摘自用户原话的片段，必须是原话子串且长度>=2")


class RetractionStatement(BaseModel):
    text: str = Field(description="被撤回事实本身的第三人称陈述（用于语义召回旧记忆）")
    evidence: str = Field(description="逐字摘自用户原话的片段，必须是原话子串且长度>=2")


class ExtractionResult(BaseModel):
    facts: List[FactStatement] = Field(default_factory=list, description="符合长期记忆标准的断言事实列表")
    retractions: List[RetractionStatement] = Field(default_factory=list, description="用户明确要求撤回或遗忘的事实列表")


class ReconcileAction(BaseModel):
    statement: str = Field(description="陈述别名编号，如 S1")
    op: str = Field(description="操作类型：ADD, UPDATE, DELETE, NONE")
    target: Optional[str] = Field(None, description="旧记忆别名编号，如 M1。ADD/NONE 时为 null，UPDATE/DELETE 时必填")


class ReconcileResult(BaseModel):
    actions: List[ReconcileAction] = Field(default_factory=list, description="针对各陈述的对账操作列表")


async def _invoke_structured(prompt: str, schema: Type[T]) -> Optional[T]:
    """通过 PydanticOutputParser 调用 LLM 并解析为结构化输出。失败返回 None。"""
    try:
        parser = PydanticOutputParser(pydantic_object=schema)
        format_instructions = parser.get_format_instructions()
        if "{format_instructions}" in prompt:
            final_prompt = prompt.replace("{format_instructions}", format_instructions)
        else:
            final_prompt = f"{prompt}\n\n{format_instructions}"

        res = await models.extractor_llm.ainvoke(
            final_prompt,
            config={"callbacks": [cli_logger.CLIDebugCallbackHandler()]},
        )
        content = extract_text(res.content)
        return parser.parse(content)
    except Exception as e:
        print(f"⚠️ [结构化解析失败]: {e}")
        return None


async def extract_and_store_memory(user_text: str) -> None:
    """后台异步执行记忆提取、对账与保存，fail-closed 永不抛异常。"""
    try:
        # D1: 输入隔离与门控
        if not user_text or not isinstance(user_text, str):
            return
        clean_user_text = strip_memory_block(user_text).strip()
        if not clean_user_text:
            return

        if len(clean_user_text) > config.MEMORY_EXTRACT_MAX_USER_CHARS:
            print(f"--- 🧠 [记忆提取跳过]: 用户消息过长 ({len(clean_user_text)} > {config.MEMORY_EXTRACT_MAX_USER_CHARS})，疑似第三方文本 ---")
            return

        # 阶段1：原子事实抽取
        p1_prompt = MEMORY_EXTRACTOR_PROMPT.format(
            user_text=clean_user_text,
            format_instructions="{format_instructions}",
        )
        extract_result: Optional[ExtractionResult] = await _invoke_structured(p1_prompt, ExtractionResult)
        if extract_result is None:
            print("--- 🧠 [记忆提取失败]: 阶段1 LLM 解析失败，fail-closed 本轮不记录 ---")
            return

        # 护栏校验阶段1产出的陈述
        valid_statements: List[Dict[str, Any]] = []
        for f in extract_result.facts:
            if len(valid_statements) >= config.MEMORY_MAX_STATEMENTS_PER_TURN:
                break
            ok, reason = validate_statement(f.text, f.evidence, clean_user_text)
            if ok:
                valid_statements.append({"kind": "assert", "text": f.text, "evidence": f.evidence})
            else:
                print(f"🧠 [记忆护栏拦截] fact '{f.text}' 丢弃: {reason}")

        for r in extract_result.retractions:
            if len(valid_statements) >= config.MEMORY_MAX_STATEMENTS_PER_TURN:
                break
            ok, reason = validate_statement(r.text, r.evidence, clean_user_text)
            if ok:
                valid_statements.append({"kind": "retract", "text": r.text, "evidence": r.evidence})
            else:
                print(f"🧠 [记忆护栏拦截] retraction '{r.text}' 丢弃: {reason}")

        if not valid_statements:
            print("--- 🧠 [记忆提取完成]: 未发现符合条件的长期事实或撤回请求 ---")
            return

        # 召回旧记忆候选
        candidates_by_id: Dict[int, Dict[str, Any]] = {}
        for s in valid_statements:
            cands = await asyncio.to_thread(
                memory_store.recall_similar,
                s["text"],
                config.MEMORY_RECALL_TOP_K,
            )
            for c in cands:
                candidates_by_id[c["id"]] = c

        # 分支：无候选旧记忆
        if not candidates_by_id:
            for s in valid_statements:
                if s["kind"] == "assert":
                    await asyncio.to_thread(
                        memory_store.add_memory,
                        s["text"],
                        evidence=s["evidence"],
                    )
                else:
                    print(f"🧠 [记忆对账] 无候选旧记忆，撤回操作丢弃: {s['text']}")
            return

        # 分支：有候选旧记忆，进入阶段2对账
        statements_by_alias: Dict[str, Dict[str, Any]] = {
            f"S{i+1}": s for i, s in enumerate(valid_statements)
        }
        candidates_by_alias: Dict[str, Dict[str, Any]] = {
            f"M{j+1}": c for j, c in enumerate(candidates_by_id.values())
        }

        statements_block = "\n".join([
            f"- {alias} [{s['kind']}]: {s['text']} (原话依据: {s['evidence']})"
            for alias, s in statements_by_alias.items()
        ])
        candidates_block = "\n".join([
            f"- {alias}: {c['text']}"
            for alias, c in candidates_by_alias.items()
        ])

        p2_prompt = MEMORY_RECONCILER_PROMPT.format(
            statements_block=statements_block,
            candidates_block=candidates_block,
            format_instructions="{format_instructions}",
        )

        reconcile_result: Optional[ReconcileResult] = await _invoke_structured(p2_prompt, ReconcileResult)
        if reconcile_result is None:
            print("--- 🧠 [记忆对账失败]: 阶段2 LLM 解析失败，fail-closed 本轮不执行写操作 ---")
            return

        # 护栏校验阶段2操作
        valid_ops = validate_reconciliation_ops(
            reconcile_result.actions,
            statements_by_alias,
            candidates_by_alias,
        )

        # 逐条调用 store 写接口
        for op_info in valid_ops:
            op = op_info["op"]
            stmt = op_info["statement"]
            target_id = op_info["target_id"]

            if op == "ADD":
                await asyncio.to_thread(
                    memory_store.add_memory,
                    stmt["text"],
                    evidence=stmt["evidence"],
                )
            elif op == "UPDATE":
                if target_id is not None:
                    await asyncio.to_thread(
                        memory_store.supersede_memory,
                        target_id,
                        stmt["text"],
                        evidence=stmt["evidence"],
                    )
            elif op == "DELETE":
                if target_id is not None:
                    await asyncio.to_thread(
                        memory_store.retire_memory,
                        target_id,
                        evidence=stmt["evidence"],
                    )
            elif op == "NONE":
                print(f"🧠 [记忆对账] {op_info['statement_alias']} 对账判定为 NONE，跳过")

    except Exception as e:
        print(f"⚠️ \033[91m[后台记忆处理异常]\033[0m: {str(e)}")
