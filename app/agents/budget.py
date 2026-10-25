"""专家独立停止条件：单次委派内的工具调用预算 (所在层: L4)

每个专家在一次 task 委派内最多调用 N 次工具（N 见 config.SUBAGENT_TOOL_LIMITS，统计包含框架内置工具），两级止损：
1. 软停止  ToolCallLimitMiddleware(run_limit=N, exit_behavior="continue")
   第 N+1 次起的工具调用不会执行，而是以错误 ToolMessage（"Tool call limit exceeded..."）回给模型；
   模型仍可基于已有结果写汇报（WORKER_BASE_PROMPT 的【步骤预算】一节教它这样做）。
2. 硬停止  ModelCallLimitMiddleware(run_limit=N+2, exit_behavior="end")
   模型无视软停止继续要工具时，在第 N+3 次模型调用之前直接结束，task 返回 "Model call limits exceeded: ..."。
   N+2 = N 次正常调用 + 1 次触发拦截的调用 + 1 次收尾汇报。
计数以“一次子 Agent 运行”为单位，不跨委派累计。

故意直接 import、不做 try/except ImportError 降级：限步保护悄悄失效，比启动失败更危险。
本模块只依赖 langchain 与 app.config，不得 import app.infra.models / app.knowledge / app.memory
（tests/smoke_architecture.py 的轻量导入测试会检查）。
"""

from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

from app import config


def tool_limit_for(name: str) -> int:
    """按专家名取工具调用预算；未登记的专家使用默认值。"""
    return config.SUBAGENT_TOOL_LIMITS.get(name, config.SUBAGENT_TOOL_LIMIT_DEFAULT)


def build_budget_middleware(max_tool_calls: int) -> list:
    """返回 [软停止, 硬停止] 两个中间件（每次调用都返回新实例）。"""
    if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 1:
        raise ValueError(f"max_tool_calls 必须是 >= 1 的整数，收到: {max_tool_calls!r}")
    return [
        ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="continue"),
        ModelCallLimitMiddleware(run_limit=max_tool_calls + 2, exit_behavior="end"),
    ]
