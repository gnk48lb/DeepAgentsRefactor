"""
tests/eval/run_routing_eval.py

路由评测（Routing Eval）—— 衡量 orchestrator 把 query 委派给哪个专家。

与 tests/smoke_*.py 的区别：
  - 冒烟测试验证「架构契约」（工具名、顺序、导入隔离），不涉及模型决策。
  - 本评测验证「行为」：给真实 query，看 orchestrator 实际派给谁。

两种模式：

  [mock] 默认模式，零成本、零网络、无副作用。
      真实读取 orchestrator 的 system_prompt / subagent 清单 / 工具集，
      但把 LLM 换成 KeywordRouterModel（按关键词打分模拟路由）。
      作用：验证评测集本身、跑通打分与报告管线，并给出一条「关键词基线」。
      ⚠️ 它衡量的不是真模型的准确率，不要拿它当生产指标。

  [live] --live 模式，真实调用 GEMINI（产生 API 费用）。
      用生产同款 model / prompt / subagent 描述构建 orchestrator，
      把每个 subagent 的 runnable 换成空壳（不执行真实工具），
      只记录 task 调用，因此不会写文件、不会动桌面、不会发网络请求。
      专家内部行为仍会消耗 token（它们会真的尝试调用工具）。

用法：
    uv run python tests/eval/run_routing_eval.py                 # mock 基线
    uv run python tests/eval/run_routing_eval.py --live          # 真实模型
    uv run python tests/eval/run_routing_eval.py --live -v       # 打印逐条委派详情
    uv run python tests/eval/run_routing_eval.py --live --only G,S   # 只跑指定前缀
    uv run python tests/eval/run_routing_eval.py --live --save baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = Path(__file__).resolve().parent / "routing_cases.json"

# 单条用例超时。生产是 600s，但评测里卡住一条就会拖垮整轮，所以收紧。
EVAL_CASE_TIMEOUT = int(os.getenv("EVAL_CASE_TIMEOUT", "120"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Mock 路由模型：按关键词打分，模拟一个「读了专家描述」的路由器
# ══════════════════════════════════════════════════════════════════════════════

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# 关键词 -> 专家。刻意写得「合理但不完美」，让 mock 基线暴露评测集的区分度。
_ROUTER_RULES: List[tuple[str, List[str]]] = [
    ("DesktopAgent", ["桌面", "双击", "图标", "屏幕", "点击", "窗口"]),
    ("FileAgent", ["workspace", "文件", "目录", "读取", "写入", "新建", "文件夹里的", "md 文件"]),
    ("BrowserAgent", ["http", "网页", "上传", "登录", "浏览器", "发帖"]),
    ("MapAgent", ["天气", "路线", "怎么走", "附近", "骑行", "步行", "导航", "带伞", "ip", "城市", "行程"]),
    ("SQLAgent", ["身高", "罩杯", "出生", "多少部", "统计", "数据库", "年发行", "有多少"]),
    ("MediaAgent", ["女优", "合作", "题材", "番号", "人妻", "熟女", "作品是什么"]),
    ("CodeAgent", ["python", "代码", "画成", "折线图", "计算", "排序", "算一下"]),
    ("KnowledgeAgent", ["几点", "星期", "多少天", "怎么做", "攻略", "讲讲", "原理", "是什么"]),
]


class KeywordRouterModel(BaseChatModel):
    """第一轮按关键词选出 subagent 并发出 task 调用；之后直接给最终答案。"""

    idx: int = 0
    seen_subagents: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "keyword-router"

    def bind_tools(self, tools, **kw):
        return self

    def _pick(self, text: str) -> str | None:
        scores: Dict[str, int] = defaultdict(int)
        for agent, kws in _ROUTER_RULES:
            for kw in kws:
                if kw.lower() in text.lower():
                    scores[agent] += 1
        if not scores:
            return None
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kw) -> ChatResult:
        # 只关心第一条 human 消息（subagent 内部也会用到本模型，直接给空答案）
        text = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                text = str(m.content)
                break

        already = any(isinstance(m, AIMessage) and getattr(m, "tool_calls", None) for m in messages)
        if self.idx == 0 and text and not already:
            self.idx += 1
            agent = self._pick(text)
            if agent:
                self.seen_subagents.append(agent)
                msg = AIMessage(
                    content="",
                    tool_calls=[{"name": "task", "args": {"subagent_type": agent, "description": text}, "id": "t1"}],
                )
            else:
                msg = AIMessage(content="（mock）无需委派，直接回答。")
        else:
            msg = AIMessage(content="（mock）专家执行完毕。")
        return ChatResult(generations=[ChatGeneration(message=msg)])


# ══════════════════════════════════════════════════════════════════════════════
# 评测执行
# ══════════════════════════════════════════════════════════════════════════════


def load_cases(only_prefixes: List[str] | None = None) -> List[dict]:
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    if only_prefixes:
        cases = [c for c in cases if any(c["id"].startswith(p) for p in only_prefixes)]
    return cases


def score_case(case: dict, delegated: List[str]) -> dict:
    """expected / forbidden 双向打分。"""
    got = set(delegated)
    expected = set(case.get("expected", []))
    forbidden = set(case.get("forbidden", []))

    missing = sorted(expected - got)
    extra = sorted(got - expected - forbidden)
    violated = sorted(got & forbidden)

    ok = not missing and not violated
    return {
        "ok": ok,
        "delegated": sorted(got),
        "missing": missing,
        "forbidden_hit": violated,
        "unexpected": extra,
    }


async def run_mock(cases: List[dict], verbose: bool) -> List[dict]:
    """用 KeywordRouterModel 驱动真实的 orchestrator 图结构。"""
    from deepagents import create_deep_agent, CompiledSubAgent
    from langchain_core.runnables import RunnableLambda
    from langgraph.checkpoint.memory import MemorySaver

    from app.agents import orchestrator as orch
    from app.agents.prompts import MAIN_AGENT_PROMPT
    from app.agents.experts import (
        knowledge_agent, media_agent, map_agent, code_agent, sql_agent, browser_agent,
    )

    # 真实 subagent 清单（名字 + 描述），runnable 换成空壳，避免执行真实工具
    def _shell(name: str) -> CompiledSubAgent:
        return CompiledSubAgent(
            name=name,
            description="",
            runnable=RunnableLambda(lambda state: {"messages": [AIMessage(content=f"（mock）{name} 汇报")]}),
        )

    subagents = []
    for spec in orch._DECLARATIVE_SPECS:
        subagents.append({
            "name": spec.name,
            "description": spec.description,
            "system_prompt": "mock",
            "tools": [],
            "model": KeywordRouterModel(),
        })

    model = KeywordRouterModel()
    agent = create_deep_agent(
        model=model,
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=subagents,
        checkpointer=MemorySaver(),
    )

    results = []
    for case in cases:
        model.idx = 0
        model.seen_subagents = []
        cfg = {"configurable": {"thread_id": case["id"]}, "recursion_limit": 40}
        try:
            await agent.ainvoke({"messages": [HumanMessage(content=case["query"])]}, cfg)
            delegated = list(model.seen_subagents)
        except Exception as e:  # noqa: BLE001
            delegated = []
            if verbose:
                print(f"    [ERROR] {case['id']}: {e}")
        results.append({"id": case["id"], "difficulty": case.get("difficulty", "?"),
                        "query": case["query"], "delegated": delegated,
                        **score_case(case, delegated)})
    return results


async def run_live(cases: List[dict], verbose: bool) -> List[dict]:
    """真实模型 + 真实 prompt；subagent runnable 换成空壳以隔离副作用。"""
    from deepagents import create_deep_agent, CompiledSubAgent, register_harness_profile, HarnessProfile, GeneralPurposeSubagentProfile
    from langchain_core.runnables import RunnableLambda
    from langgraph.checkpoint.memory import MemorySaver

    from app import config
    from app.infra import models
    from app.agents import orchestrator as orch
    from app.agents.prompts import MAIN_AGENT_PROMPT

    if not config.GEMINI_API_KEY:
        raise SystemExit("未配置 GEMINI_API_KEY / GOOGLE_API_KEY，无法运行 --live。")

    # 与生产一致的 harness profile（排除危险工具、禁用内置 general-purpose）
    for provider in ("google_genai", "openai"):
        try:
            register_harness_profile(
                provider,
                HarnessProfile(
                    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
                    excluded_tools=frozenset({"write_file", "edit_file", "delete", "execute"}),
                ),
            )
        except Exception:  # noqa: BLE001
            pass

    # 记录器：空壳 runnable 把「自己被调用」写进这个列表
    called: List[str] = []

    def _shell(name: str):
        def _run(_state):
            called.append(name)
            return {"messages": [AIMessage(content=f"【内部专家汇报 - 来源：{name}】\n（评测桩：未执行真实工具）")]}
        return RunnableLambda(_run)

    subagents = []
    for spec in orch._DECLARATIVE_SPECS:
        subagents.append(CompiledSubAgent(name=spec.name, description=spec.description, runnable=_shell(spec.name)))
    # FileAgent / DesktopAgent 同样只登记名字，不构建真实子图（避免 MCP 依赖）
    from app.agents.experts import file_agent, desktop_agent
    subagents.append(CompiledSubAgent(name=file_agent.NAME, description=file_agent.DESCRIPTION, runnable=_shell(file_agent.NAME)))
    subagents.append(CompiledSubAgent(name=desktop_agent.NAME, description=desktop_agent.DESCRIPTION, runnable=_shell(desktop_agent.NAME)))

    agent = create_deep_agent(
        model=models.orchestrator_llm,
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=subagents,
        checkpointer=MemorySaver(),
    )

    results = []
    for i, case in enumerate(cases, 1):
        called.clear()
        cfg = {"configurable": {"thread_id": case["id"]}, "recursion_limit": 40}
        print(f"  [{i}/{len(cases)}] {case['id']} {case['query'][:30]} ...", end="", flush=True)
        try:
            await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=case["query"])]}, cfg),
                timeout=EVAL_CASE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            print(f" 超时(>{EVAL_CASE_TIMEOUT}s)", flush=True)
            if verbose:
                traceback.print_exc()
        except Exception as e:  # noqa: BLE001
            print(f" 错误: {type(e).__name__}: {e}", flush=True)
            if verbose:
                traceback.print_exc()
        else:
            print(f" -> {called or '未委派'}", flush=True)
        delegated = list(called)
        results.append({"id": case["id"], "difficulty": case.get("difficulty", "?"),
                        "query": case["query"], "delegated": delegated,
                        **score_case(case, delegated)})
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════════════════════════


def report(results: List[dict], mode: str, verbose: bool) -> int:
    print("\n" + "=" * 78)
    print(f"路由评测报告（mode={mode}，共 {len(results)} 条）")
    print("=" * 78)

    if verbose:
        for r in results:
            flag = "PASS" if r["ok"] else "FAIL"
            print(f"\n  [{flag}] {r['id']}  {r['query']}")
            print(f"         实际委派: {r['delegated']}")
            if r["missing"]:
                print(f"         漏派:     {r['missing']}")
            if r["forbidden_hit"]:
                print(f"         ⚠ 误派(禁区): {r['forbidden_hit']}")
            if r["unexpected"]:
                print(f"         多派(可接受): {r['unexpected']}")

    total = len(results)
    passed = sum(r["ok"] for r in results)
    print(f"\n总准确率: {passed}/{total} = {passed / total:.1%}" if total else "\n无用例")

    # 按难度
    by_diff: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)
    print("\n按难度:")
    for d in ("easy", "medium", "hard", "?"):
        if d in by_diff:
            rs = by_diff[d]
            p = sum(r["ok"] for r in rs)
            print(f"  {d:7} {p}/{len(rs)} = {p / len(rs):.1%}")

    # 按前缀（专家域）
    by_prefix: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_prefix[re.match(r"[A-Z]+", r["id"]).group(0)].append(r)
    print("\n按用例组:")
    for pfx in sorted(by_prefix):
        rs = by_prefix[pfx]
        p = sum(r["ok"] for r in rs)
        mark = "" if p == len(rs) else "  ←"
        print(f"  {pfx:3} {p}/{len(rs)} = {p / len(rs):.1%}{mark}")

    # 失败模式
    miss_counter: Counter = Counter()
    forbid_counter: Counter = Counter()
    for r in results:
        for m in r["missing"]:
            miss_counter[m] += 1
        for f in r["forbidden_hit"]:
            forbid_counter[f] += 1

    if miss_counter or forbid_counter:
        print("\n失败模式:")
        for agent, n in miss_counter.most_common():
            print(f"  漏派 {agent:16} {n} 次")
        for agent, n in forbid_counter.most_common():
            print(f"  误派 {agent:16} {n} 次  ← 禁区违规，优先级最高")

    # 混淆矩阵：期望 -> 实际
    print("\n委派分布:")
    for agent, n in Counter(d for r in results for d in r["delegated"]).most_common():
        print(f"  {agent:16} {n} 次")

    print("\n" + "=" * 78)
    if mode == "mock":
        print("⚠️ mock 模式衡量的是「关键词基线」，不是真实模型能力。")
        print("   要得到可用于决策的指标，请运行：--live")
    print("=" * 78)
    return 0 if passed == total else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Orchestrator 路由评测")
    ap.add_argument("--live", action="store_true", help="调用真实模型（产生 API 费用）")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印逐条结果")
    ap.add_argument("--only", default=None, help="只跑指定前缀，逗号分隔，如 K,M,S")
    ap.add_argument("--save", default=None, help="把结果保存为 JSON 基线文件")
    args = ap.parse_args()

    prefixes = args.only.split(",") if args.only else None
    cases = load_cases(prefixes)
    if not cases:
        print("没有匹配的用例。")
        return 2

    mode = "live" if args.live else "mock"
    print(f"载入 {len(cases)} 条用例，mode={mode}")
    results = asyncio.run(run_live(cases, args.verbose) if args.live else run_mock(cases, args.verbose))

    if args.save:
        out = Path(args.save)
        out.write_text(json.dumps({"mode": mode, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存基线: {out}")

    return report(results, mode, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
