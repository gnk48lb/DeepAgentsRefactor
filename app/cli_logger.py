"""
app/cli_logger.py

Terminal debug logger and LangChain callback handler for GNK48-Agent.
Provides colored, framed, and hierarchical visual traces of LLM invocations,
tool executions, subagent delegations, thinking processes, and HITL interrupts.
"""

import os
import sys
import time
import json
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.utils import extract_text

# 确保 Windows 控制台启用 ANSI 与 UTF-8 编码
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── ANSI 颜色与修饰常量 ──
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

FG_RED = "\033[91m"
FG_GREEN = "\033[92m"
FG_YELLOW = "\033[93m"
FG_BLUE = "\033[94m"
FG_MAGENTA = "\033[95m"
FG_CYAN = "\033[96m"
FG_WHITE = "\033[97m"
FG_GRAY = "\033[90m"

# 正则表达式用于 Base64 图片脱敏
DATA_URI_REGEX = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+")
BASE64_LONG_REGEX = re.compile(r"^[A-Za-z0-9+/=\r\n\s]{200,}$")


def is_debug_enabled() -> bool:
    """检查是否开启命令行调试输出（通过环境变量 CLI_DEBUG 控制，默认开启）"""
    val = os.getenv("CLI_DEBUG", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def get_max_chars() -> int:
    """获取单条输出截断字符数上限（默认 600）"""
    try:
        return int(os.getenv("CLI_DEBUG_MAX_CHARS", "600"))
    except Exception:
        return 600


def _desensitize_str(s: str) -> str:
    """脱敏字符串中的 Base64 内容"""
    try:
        s = DATA_URI_REGEX.sub("<Base64 Image>", s)
        # 若整串长度 > 200 且无空字符间隔，匹配纯 Base64 数据串
        if len(s) > 200 and not any(c.isspace() for c in s) and BASE64_LONG_REGEX.match(s):
            return f"<Base64 Data, len={len(s)}>"
    except Exception:
        pass
    return s


def _desensitize_obj(obj: Any) -> Any:
    """递归脱敏对象中可能内嵌的 Base64 字典与列表"""
    try:
        if isinstance(obj, str):
            return _desensitize_str(obj)
        elif isinstance(obj, dict):
            if obj.get("type") == "image_url" and "image_url" in obj:
                return {"type": "image_url", "image_url": "<Base64 Image>"}
            return {k: _desensitize_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_desensitize_obj(x) for x in obj]
    except Exception:
        pass
    return obj


def _format_preview(obj: Any, max_chars: Optional[int] = None) -> str:
    """
    Base64 与超长内容脱敏截断器：
    - 将 Base64 图片替换为 <Base64 Image> 或 <Base64 Data, len=...>
    - 针对多模态列表提取文本块与图片计数 🖼️ xN
    - 超过 max_chars 时截断并提示字符数
    """
    limit = max_chars if max_chars is not None else get_max_chars()
    try:
        if obj is None:
            return "(空)"

        # 如果是带有 content 属性的消息对象
        if hasattr(obj, "content"):
            obj = getattr(obj, "content", "")

        # 多模态列表检测
        if isinstance(obj, list):
            has_image = any(
                isinstance(x, dict) and (x.get("type") == "image_url" or "image_url" in x)
                for x in obj
            )
            if has_image:
                img_count = sum(
                    1 for x in obj
                    if isinstance(x, dict) and (x.get("type") == "image_url" or "image_url" in x)
                )
                text_parts = [
                    str(x.get("text", "")) for x in obj
                    if isinstance(x, dict) and x.get("type") == "text" and x.get("text")
                ]
                text_summary = " ".join(text_parts).strip()
                res = f"🖼️ 图片 x{img_count}" + (f" | 文本: {text_summary}" if text_summary else "")
                if len(res) > limit:
                    return res[:limit] + f"\n... (共 {len(res)} 字符，已截断)"
                return res

        # 结构化数据或字符串处理
        if isinstance(obj, (dict, list)):
            cleaned = _desensitize_obj(obj)
            res = json.dumps(cleaned, ensure_ascii=False, indent=2)
        else:
            res = _desensitize_str(str(obj))

        if len(res) > limit:
            return res[:limit] + f"\n... (共 {len(res)} 字符，已截断)"
        return res
    except Exception:
        try:
            raw = str(obj)
            return raw[:limit] if len(raw) > limit else raw
        except Exception:
            return "(预览失败)"


def print_box(title: str, content: str, color: str = FG_CYAN, border_width: int = 70) -> None:
    """
    使用 Unicode 边框输出单色多行内容块：
    ╭─ [Title] ──────────────────────────────────────────
    │  Line 1
    │  Line 2
    ╰────────────────────────────────────────────────────
    """
    if not is_debug_enabled():
        return
    try:
        title_stripped = title.strip()
        title_bar = f"╭─ {title_stripped} "
        remaining_dashes = max(4, border_width - len(title_stripped) - 6)
        top = f"{color}{title_bar}{'─' * remaining_dashes}{RESET}"

        lines = content.splitlines() if content else ["(无内容)"]
        body_lines = [f"{color}│{RESET}  {line}" for line in lines]
        body = "\n".join(body_lines)

        bottom_width = max(20, len(title_bar) + remaining_dashes - 2)
        bottom = f"{color}╰{'─' * bottom_width}{RESET}"

        print(f"{top}\n{body}\n{bottom}", flush=True)
    except Exception:
        pass


def print_banner(lines: List[str], color: str = FG_CYAN, border_width: int = 72) -> None:
    """
    使用双线 Unicode 边框输出重要阶段横幅：
    ╔════════════════════════════════════════════════════
    ║ Line 1
    ║ Line 2
    ╚════════════════════════════════════════════════════
    """
    if not is_debug_enabled():
        return
    try:
        top = f"{color}╔{'═' * (border_width - 1)}{RESET}"
        bottom = f"{color}╚{'═' * (border_width - 1)}{RESET}"
        body = "\n".join(f"{color}║{RESET}  {line}" for line in lines)
        print(f"\n{top}\n{body}\n{bottom}", flush=True)
    except Exception:
        pass


def _identify_agent(messages: Any, metadata: Optional[Dict[str, Any]], serialized: Optional[Dict[str, Any]]) -> str:
    """从上下文消息特征、metadata 与 serialized 中识别调用 LLM 的角色名"""
    try:
        # 1. 尝试从 messages 中提取系统提示词与首条消息特征
        sys_texts = []
        all_texts = []
        if isinstance(messages, list):
            for item in messages:
                if isinstance(item, list):
                    for m in item:
                        c = getattr(m, "content", "")
                        if isinstance(m, SystemMessage):
                            sys_texts.append(extract_text(c))
                        all_texts.append(extract_text(c))
                elif isinstance(item, BaseMessage):
                    c = getattr(item, "content", "")
                    if isinstance(item, SystemMessage):
                        sys_texts.append(extract_text(c))
                    all_texts.append(extract_text(c))

        corpus = "\n".join(sys_texts) if sys_texts else "\n".join(all_texts[:3])

        if "你是一个多智能体系统的主 Agent" in corpus:
            return "MainAgent (Supervisor)"
        if "个人隐私与偏好提取官" in corpus:
            return "MemoryExtractor"
        if "意图翻译官" in corpus:
            return "VLM (意图翻译)"

        # 精准匹配专家 Agent（使用否定后行断言 (?<!如果) 排除 WORKER_BASE_PROMPT 中的举例 "如果你是 MapAgent"）
        agent_match = re.search(r"(?<!如果)你是\s*([A-Za-z]+Agent)", corpus)
        if agent_match:
            candidate = agent_match.group(1)
            if candidate in (
                "KnowledgeAgent", "FileAgent", "MapAgent", "CodeAgent",
                "SQLAgent", "MediaAgent", "BrowserAgent", "DesktopAgent"
            ):
                return candidate
        if "KnowledgeAgent" in corpus and "知识百科专家" in corpus:
            return "KnowledgeAgent"

        # 2. 尝试从 metadata 中识别（langgraph / deepagents 命名空间）
        if metadata and isinstance(metadata, dict):
            checkpoint_ns = str(metadata.get("checkpoint_ns", ""))
            node = str(metadata.get("langgraph_node", ""))
            for role in (
                "FileAgent", "DesktopAgent", "BrowserAgent",
                "KnowledgeAgent", "MapAgent", "CodeAgent",
                "SQLAgent", "MediaAgent"
            ):
                if role in checkpoint_ns or role in node:
                    return role

            # 若 checkpoint_ns 为空，说明是顶层主图
            if not checkpoint_ns:
                return "MainAgent (Supervisor)"

            if checkpoint_ns and not checkpoint_ns.startswith(("model:", "tools:")):
                return checkpoint_ns

        # 3. 尝试从 serialized 获取名称
        if serialized and isinstance(serialized, dict):
            s_name = serialized.get("name")
            if s_name:
                return str(s_name)

        return "MainAgent (Supervisor)"
    except Exception:
        return "Agent"


class CLIDebugCallbackHandler(BaseCallbackHandler):
    """
    命令行美化调试回调处理器：
    挂载到 LangChain RunnableConfig 的 callbacks 中，实时捕获 LLM 与工具执行生命周期。
    所有回调方法均包含防御性异常处理，保证主流程绝对安全。
    """

    def __init__(self):
        super().__init__()
        self._llm_runs: Dict[UUID, Dict[str, Any]] = {}
        self._tool_runs: Dict[UUID, Dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            if not is_debug_enabled():
                return
            agent_name = _identify_agent(messages, metadata, serialized)

            # 获取模型名称
            model_name = "unknown-model"
            if isinstance(serialized, dict):
                model_name = (
                    serialized.get("kwargs", {}).get("model")
                    or serialized.get("name")
                    or (serialized.get("id", [])[-1] if serialized.get("id") else None)
                )
            if not model_name or model_name == "unknown-model":
                inv_params = kwargs.get("invocation_params", {})
                if isinstance(inv_params, dict) and inv_params.get("model"):
                    model_name = inv_params.get("model")

            self._llm_runs[run_id] = {
                "agent": agent_name,
                "model": model_name,
                "start": time.perf_counter(),
            }

            msg_count = sum(len(sub) if isinstance(sub, list) else 1 for sub in messages) if messages else 0
            print(
                f"{FG_MAGENTA}🤖 [LLM 调用开始] {BOLD}{agent_name}{RESET}"
                f"{FG_MAGENTA} (model={model_name}, 消息数={msg_count}){RESET}",
                flush=True,
            )
        except Exception:
            pass

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            if not is_debug_enabled():
                return
            prompt_corpus = "\n".join(prompts) if prompts else ""
            if "个人隐私与偏好提取官" in prompt_corpus:
                agent_name = "MemoryExtractor"
            elif "意图翻译官" in prompt_corpus:
                agent_name = "VLM (意图翻译)"
            else:
                agent_name = _identify_agent(None, metadata, serialized)

            model_name = "unknown-model"
            if isinstance(serialized, dict):
                model_name = (
                    serialized.get("kwargs", {}).get("model")
                    or serialized.get("name")
                    or (serialized.get("id", [])[-1] if serialized.get("id") else None)
                )

            self._llm_runs[run_id] = {
                "agent": agent_name,
                "model": model_name,
                "start": time.perf_counter(),
            }

            print(
                f"{FG_MAGENTA}🤖 [LLM 调用开始] {BOLD}{agent_name}{RESET}"
                f"{FG_MAGENTA} (model={model_name}, Prompt 数={len(prompts)}){RESET}",
                flush=True,
            )
        except Exception:
            pass

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            if not is_debug_enabled():
                return
            info = self._llm_runs.pop(run_id, {})
            agent_name = info.get("agent", "Agent")
            start = info.get("start", time.perf_counter())
            elapsed = time.perf_counter() - start

            # 获取 Generation 与 AIMessage
            gen = None
            message = None
            if response and hasattr(response, "generations") and response.generations:
                for sublist in response.generations:
                    if sublist:
                        gen = sublist[0]
                        message = getattr(gen, "message", None)
                        break

            # 1. 提取 Token 统计与缓存命中
            usage = getattr(message, "usage_metadata", None)
            if not usage and hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
                usage = response.llm_output.get("token_usage") or response.llm_output.get("usage_metadata")

            tok_str = ""
            if usage and isinstance(usage, dict):
                in_tok = usage.get("input_tokens")
                out_tok = usage.get("output_tokens")
                tot_tok = usage.get("total_tokens")
                parts = []
                if tot_tok is not None:
                    parts.append(f"in={in_tok}, out={out_tok}, total={tot_tok}")
                details = usage.get("input_token_details", {})
                if isinstance(details, dict):
                    cache_read = details.get("cache_read")
                    if cache_read:
                        parts.append(f"cache_read={cache_read}")
                if parts:
                    tok_str = f" | tokens: {', '.join(parts)}"

            print(
                f"{FG_MAGENTA}🤖 [LLM 调用结束] {BOLD}{agent_name}{RESET}"
                f"{FG_MAGENTA} (耗时 {elapsed:.2f}s{tok_str}){RESET}",
                flush=True,
            )

            # 2. 提取模型思考与推理过程
            thinking_texts = []
            if message:
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            b_type = block.get("type", "")
                            if b_type in ("thinking", "reasoning", "thought"):
                                t = block.get("thinking") or block.get("thought") or block.get("text")
                                if t and str(t).strip():
                                    thinking_texts.append(str(t).strip())
                            elif "thinking" in block and block["thinking"]:
                                thinking_texts.append(str(block["thinking"]).strip())
                            elif "thought" in block and block["thought"]:
                                thinking_texts.append(str(block["thought"]).strip())
                            elif "reasoning_content" in block and block["reasoning_content"]:
                                thinking_texts.append(str(block["reasoning_content"]).strip())

                add_kwargs = getattr(message, "additional_kwargs", {})
                if isinstance(add_kwargs, dict):
                    for k in ("thinking", "reasoning_content", "thoughts", "thought"):
                        val = add_kwargs.get(k)
                        if val and isinstance(val, str) and val.strip():
                            if val.strip() not in thinking_texts:
                                thinking_texts.append(val.strip())

                # 伴随 tool_calls 输出的推理/说明文本
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    text_part = extract_text(content).strip()
                    if text_part and text_part not in thinking_texts:
                        thinking_texts.append(text_part)

            if thinking_texts:
                full_thinking = "\n".join(thinking_texts)
                print_box(f"💭 [{agent_name} 思考 / 推理]", _format_preview(full_thinking), color=FG_GRAY)

            # 3. 提取工具调用与委派决策
            tool_calls = getattr(message, "tool_calls", None) if message else None
            if tool_calls:
                for tc in tool_calls:
                    tc_name = tc.get("name", "tool")
                    tc_args = tc.get("args", {})
                    if tc_name == "task":
                        subagent = tc_args.get("subagent_type", "未知专家")
                        desc = tc_args.get("description", "")
                        content = f"目标专家: {subagent}\n任务指令: {desc}"
                        print_box(f"📡 [MainAgent 委派子专家] ➔ {subagent}", content, color=FG_CYAN)
                    else:
                        content = f"工具名称: {tc_name}\n调用参数: {_format_preview(tc_args)}"
                        print_box(f"🎯 [{agent_name} 决定调用工具] {tc_name}", content, color=FG_YELLOW)
            else:
                # 无工具调用时的正文回复或子专家阶段汇报
                reply_text = ""
                if message:
                    reply_text = extract_text(getattr(message, "content", "")).strip()
                if not reply_text and gen and hasattr(gen, "text"):
                    reply_text = str(gen.text).strip()
                if reply_text:
                    title = f"💬 [{agent_name} 输出回复] (耗时 {elapsed:.2f}s{tok_str})"
                    print_box(title, _format_preview(reply_text), color=FG_GREEN)
                else:
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    resp_meta = getattr(message, "response_metadata", {}) or {}
                    finish_reason = gen_info.get("finish_reason") or resp_meta.get("finish_reason")
                    if finish_reason:
                        print(f"{FG_YELLOW}⚠️  [{agent_name} 未产生文本回复 | finish_reason={finish_reason}]{RESET}", flush=True)
        except Exception:
            pass

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            if not is_debug_enabled():
                return
            info = self._llm_runs.pop(run_id, {})
            agent_name = info.get("agent", "Agent")
            start = info.get("start", time.perf_counter())
            elapsed = time.perf_counter() - start

            err_type = type(error).__name__
            if err_type in ("GraphInterrupt", "NodeInterrupt") or "Interrupt" in err_type:
                print(f"{FG_YELLOW}⏸️  [LLM 执行挂起] {agent_name}: 触发状态中断{RESET}", flush=True)
                return

            title = f"❌ [LLM 调用异常] {agent_name} (耗时 {elapsed:.2f}s)"
            content = f"错误类型: {err_type}\n错误详情: {str(error)}"
            print_box(title, content, color=FG_RED)
        except Exception:
            pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            if not is_debug_enabled():
                return
            tool_name = (
                (serialized.get("name") if isinstance(serialized, dict) else None)
                or kwargs.get("name")
                or "Tool"
            )
            params = inputs if inputs is not None else input_str
            subagent = None
            if tool_name == "task" and isinstance(params, dict):
                subagent = params.get("subagent_type")

            self._tool_runs[run_id] = {
                "name": tool_name,
                "subagent": subagent,
                "start": time.perf_counter(),
            }

            if tool_name == "task":
                target = f" -> {subagent}" if subagent else ""
                print(f"\n{FG_BLUE}┌── 🚀 [进入子专家执行{target}] ...{RESET}", flush=True)
            else:
                title = f"🔧 [工具执行] {tool_name}"
                content = f"参数: {_format_preview(params)}"
                print_box(title, content, color=FG_YELLOW)
        except Exception:
            pass

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            if not is_debug_enabled():
                return
            info = self._tool_runs.pop(run_id, {})
            name = info.get("name", "Tool")
            subagent = info.get("subagent")
            start = info.get("start", time.perf_counter())
            elapsed = time.perf_counter() - start

            if name == "task":
                target = f" ({subagent})" if subagent else ""
                print(f"{FG_BLUE}└── 🏁 [子专家执行完毕{target}] (耗时 {elapsed:.2f}s){RESET}", flush=True)
                preview = _format_preview(output)
                print_box(f"📋 [子专家战报摘要{target}]", preview, color=FG_BLUE)
            else:
                title = f"🟢 [工具返回] {name} (耗时 {elapsed:.2f}s)"
                preview = _format_preview(output)
                print_box(title, preview, color=FG_GREEN)
        except Exception:
            pass

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            if not is_debug_enabled():
                return
            info = self._tool_runs.pop(run_id, {})
            name = info.get("name", "Tool")
            start = info.get("start", time.perf_counter())
            elapsed = time.perf_counter() - start

            err_type = type(error).__name__
            if err_type in ("GraphInterrupt", "NodeInterrupt") or "Interrupt" in err_type:
                print(f"{FG_YELLOW}⏸️  [工具执行挂起] {name}: 触发敏感操作人工授权拦截 (HITL){RESET}", flush=True)
                return

            title = f"❌ [工具异常] {name} (耗时 {elapsed:.2f}s)"
            content = f"错误类型: {err_type}\n错误详情: {str(error)}"
            print_box(title, content, color=FG_RED)
        except Exception:
            pass


def log_turn_start(text: str, image_url: Optional[str] = None) -> None:
    """打印新一轮请求开始横幅（含时间戳、用户原话、是否附带图片）"""
    try:
        if not is_debug_enabled():
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        has_img = "是 (已附带图片)" if image_url else "否"
        clean_text = _format_preview(text, max_chars=300)
        lines = [
            f"📥 [新请求] {timestamp} | 附带图片: {has_img}",
            f"用户输入: {clean_text}",
        ]
        print_banner(lines, color=FG_CYAN)
    except Exception:
        pass


def log_vlm_translation(translated_text: str, elapsed: float) -> None:
    """当用户发图触发 VLM 意图翻译时，打印耗时与转换后的纯文本指令"""
    try:
        if not is_debug_enabled():
            return
        title = f"👁️ [VLM 意图翻译] (耗时 {elapsed:.2f}s)"
        content = f"指令转换: {_format_preview(translated_text)}"
        print_box(title, content, color=FG_CYAN)
    except Exception:
        pass


def log_memory_status(hit: bool, details: str = "") -> None:
    """打印长期记忆检索结果（是否命中并注入当轮上下文）"""
    try:
        if not is_debug_enabled():
            return
        if hit:
            title = "🧠 [长期记忆检索] 状态: 命中并注入当轮上下文"
            content = _format_preview(details)
            print_box(title, content, color=FG_CYAN)
        else:
            print(f"{FG_GRAY}🧠 [长期记忆检索] 状态: 未命中相关历史记忆{RESET}", flush=True)
    except Exception:
        pass


def log_hitl_status(event_type: str, detail: str = "") -> None:
    """打印 HITL 挂起等待授权 / 用户输入 Y 或 N 恢复执行的提示"""
    try:
        if not is_debug_enabled():
            return
        if event_type in ("SUSPEND", "AWAITING_APPROVAL"):
            lines = [
                "🛡️ [HITL 敏感操作拦截] 等待人工授权 (回复 Y 允许 / N 拒绝)",
                _format_preview(detail, max_chars=500),
            ]
            print_banner(lines, color=FG_RED)
        elif event_type == "RESUMED":
            content = f"授权指令: {detail.strip().upper()} (恢复 Agent 执行流程)"
            print_box("🔓 [HITL 恢复执行]", content, color=FG_CYAN)
        elif event_type == "REMIND":
            content = f"收到输入: {detail}\n当前仍处于敏感操作授权挂起状态，提示用户回复 Y/N。"
            print_box("⚠️ [HITL 授权提醒]", content, color=FG_YELLOW)
    except Exception:
        pass


def log_turn_end(reply_text: str, image_count: int, awaiting_approval: bool, elapsed: float) -> None:
    """打印本轮处理完毕的总耗时、发件箱图片数与最终回复预览横幅"""
    try:
        if not is_debug_enabled():
            return
        status_str = "等待人工授权" if awaiting_approval else "正常结束"
        lines = [
            f"📤 [回合收尾] 总耗时: {elapsed:.2f}s | 发件箱图片: {image_count} 张 | 状态: {status_str}",
            f"回复内容: {_format_preview(reply_text, max_chars=400)}",
        ]
        color = FG_YELLOW if awaiting_approval else FG_GREEN
        print_banner(lines, color=color)
    except Exception:
        pass
