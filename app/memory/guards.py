"""长期记忆代码护栏与纯函数校验器 (所在层: L2)

提供文本规范化、证据溯源校验、文本合规性拦截以及对账操作的别名与约束校验。
纯函数设计，不执行任何 I/O 操作，不导入 models 或 store。
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from app import config


def normalize_text(text: str) -> str:
    """两边先做 NFKC + 空白折叠 + casefold / strip，用于证据匹配与幂等去重。"""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", str(text))
    folded = re.sub(r"\s+", " ", norm).strip().casefold()
    return folded


def evidence_in_text(evidence: str, user_text: str) -> bool:
    """
    D5 a: 每条陈述必须带 evidence：逐字摘自用户原话的片段。
    校验：两边先做 NFKC + 空白折叠 + casefold，evidence 必须是子串且长度 ≥ 2。
    """
    if not evidence or not user_text:
        return False
    norm_evidence = normalize_text(evidence)
    if len(norm_evidence) < 2:
        return False
    norm_user = normalize_text(user_text)
    return norm_evidence in norm_user


def check_memory_text(text: str) -> Tuple[bool, str]:
    """
    D5 b: 文本护栏：strip 后非空、≤ MEMORY_TEXT_MAX_CHARS、
    不含换行/制表符、不含 "<" 或 ">"、不含 "long_term_memory"。
    返回 (ok, reason)。
    """
    if not text or not isinstance(text, str):
        return False, "文本为空或类型不正确"
    s = text.strip()
    if not s:
        return False, "strip 后文本为空"
    if len(s) > config.MEMORY_TEXT_MAX_CHARS:
        return False, f"文本长度超过上限 ({len(s)} > {config.MEMORY_TEXT_MAX_CHARS})"
    if "\n" in s or "\r" in s or "\t" in s:
        return False, "文本包含换行符或制表符"
    if "<" in s or ">" in s:
        return False, "文本包含尖括号 '<' 或 '>'"
    if "long_term_memory" in s:
        return False, "文本包含保留关键字 'long_term_memory'"
    return True, ""


def validate_statement(text: str, evidence: str, user_text: str) -> Tuple[bool, str]:
    """校验单条陈述的文本合规性与证据有效性。"""
    ok, reason = check_memory_text(text)
    if not ok:
        return False, f"文本护栏未通过: {reason}"
    if not evidence_in_text(evidence, user_text):
        return False, f"证据护栏未通过: evidence '{evidence}' 不是用户原话的有效子串"
    return True, ""


def validate_reconciliation_ops(
    raw_actions: List[Any],
    statements_by_alias: Dict[str, Dict[str, Any]],
    candidates_by_alias: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    D5 c/d: 校验阶段2的对账操作。
    - LLM 看到的 id 一律是别名：陈述 S1..Sn、候选 M1..Mk。
    - 别名不存在 → 丢弃。
    - 同一批内一个 target 最多被用一次。
    - 每条陈述最多一个操作（重复以第一次为准）。
    - 阶段2 漏掉的陈述视为 NONE（不产生写操作）。
    - assert 只能 ADD/UPDATE/NONE，retract 只能 DELETE/NONE。
    - UPDATE/DELETE 必须提供有效 target 别名。
    - 阶段2 不得产出新文本：文本完全来自阶段1陈述。
    """
    validated_ops: List[Dict[str, Any]] = []
    seen_statements: set[str] = set()
    seen_targets: set[int] = set()

    for item in raw_actions:
        s_alias = getattr(item, "statement", None) or (item.get("statement") if isinstance(item, dict) else None)
        op = getattr(item, "op", None) or (item.get("op") if isinstance(item, dict) else None)
        target_alias = getattr(item, "target", None) or (item.get("target") if isinstance(item, dict) else None)

        if not s_alias or s_alias not in statements_by_alias:
            continue
        if s_alias in seen_statements:
            continue  # 每条陈述最多一个操作，多余的丢弃
        seen_statements.add(s_alias)

        stmt_info = statements_by_alias[s_alias]
        stmt_kind = stmt_info.get("kind", "assert")  # "assert" 或 "retract"

        op_str = str(op).strip().upper() if op else "NONE"
        if op_str not in ("ADD", "UPDATE", "DELETE", "NONE"):
            continue

        if stmt_kind == "assert":
            if op_str not in ("ADD", "UPDATE", "NONE"):
                # assert 只能 ADD/UPDATE/NONE
                continue
        elif stmt_kind == "retract":
            if op_str not in ("DELETE", "NONE"):
                # retract 只能 DELETE/NONE
                continue

        target_id: Optional[int] = None
        if op_str in ("UPDATE", "DELETE"):
            if not target_alias or target_alias not in candidates_by_alias:
                # UPDATE/DELETE 必须有在候选列表中的 target
                continue
            cand = candidates_by_alias[target_alias]
            target_id = cand["id"]
            if target_id in seen_targets:
                # 同一批内一个 target 最多被用一次
                continue
            seen_targets.add(target_id)
        else:
            # ADD / NONE 不需要 target
            target_alias = None
            target_id = None

        validated_ops.append({
            "statement_alias": s_alias,
            "statement": stmt_info,
            "op": op_str,
            "target_alias": target_alias,
            "target_id": target_id,
        })

    return validated_ops
