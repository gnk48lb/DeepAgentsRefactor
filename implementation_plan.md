# DeepAgents 迁移 Phase 1 实施计划

将现有 Supervisor+Router 手动路由架构改造成 deepagents 的主 Agent ReAct 循环 + task 工具架构。
本阶段只做骨架搭建和工具型专家迁移，不动 FileAgent/DesktopAgent 内部逻辑，不接入生产路径。

---

## 关键发现与风险说明

> [!IMPORTANT]
> **`ToolCallLimitMiddleware` 不存在于 deepagents 的中间件体系中**
>
> 经过实际查阅 [deepagents middleware API](https://reference.langchain.com/python/deepagents/middleware)，
> deepagents 0.7.x 的内置中间件只有：`FilesystemMiddleware`、`SubAgentMiddleware`、
> `MemoryMiddleware`、`SkillsMiddleware`、`SummarizationMiddleware`。
>
> 用户需求里的 `from langchain.agents.middleware import ToolCallLimitMiddleware` 是否存在于
> langchain 1.x 尚未 100% 确认（langchain 1.x 是 deepagents 0.7 的实际依赖版本，
> 与当前项目的 langchain 0.3.x 版本相差较大）。
>
> **处理方案**：将 `ToolCallLimitMiddleware` 的导入放进 `try/except ImportError` 保护块，
> 若 langchain 1.x 中该类确实存在则正常用，若不存在则在 `deep_agent.py` 内用注释占位，
> 并在验证脚本中以烟雾测试确认实际行为。

> [!WARNING]
> **依赖版本跳跃极大**
>
> 当前 `pyproject.toml` 锁定 `langchain>=0.3.0`、`langchain-core>=0.3.15`、
> `langchain-google-genai>=2.0.0`。
> deepagents 0.7.13 需要 `langchain>=1.3.18`、`langchain-core>=1.6.1`、
> `langchain-google-genai>=4.3.7`。
>
> 这是 **major version 跳跃**（0.x → 1.x），`uv add` 会联动升级上述包。
> 现有 `app/graph.py` 使用的 `langgraph.prebuilt.create_react_agent` 签名
> 在新版本可能有变化，但因为本次不改 `graph.py`，影响可控。
> 主要风险是 **其他依赖包是否与新版 langchain-core 兼容**（特别是 langchain-mcp-adapters、
> langchain-openai、langchain-huggingface）。

> [!NOTE]
> **第 0 步烟雾测试的目的**
>
> 在写任何新代码之前，先验证 ChatGoogleGenerativeAI 实例可以直接传给 `create_deep_agent(model=...)`。
> 若出现 `model=` 参数必须传字符串（如 `"google:gemini-..."`）而不接受 LangChain ChatModel 实例的错误，
> 则需要调整第 6 步的组装方式。

---

## 开放性问题

1. **`ToolCallLimitMiddleware` 在 langchain 1.x 中的实际位置和签名是否与用户规格完全一致？**
   → 装完 deepagents 后立即用 `python -c "from langchain.agents.middleware import ToolCallLimitMiddleware; help(ToolCallLimitMiddleware)"` 核实，若不存在则用注释占位标记"待验证后接入"。

2. **`create_deep_agent` 的 `model=` 参数是否接受 LangChain ChatModel 实例？**
   → 由第 0 步烟雾测试确认。

3. **`deepagents` 的 `subagents` 参数格式**：是否接受含 `model=` 字段（ChatModel 实例）的 dict？
   → 同上，烟雾测试+看实际 TypeError 报错来确认。

---

## 拟变更文件列表

### Step 0 — 环境准备

#### [MODIFY] [pyproject.toml](file:///d:/code/VisualStudioCode/AI/project/GNK48-Agent/DeepAgentsRefactor/pyproject.toml)
- 添加 `deepagents>=0.7,<0.8` 到 `dependencies`
- 放开 `langchain`/`langchain-core`/`langchain-google-genai`/`langgraph` 的下界，
  允许 uv 把它们一起升级到 deepagents 0.7 需要的版本
  （具体：`langchain>=1.3.18`、`langchain-core>=1.6.1`、`langchain-google-genai>=4.3.7`）
- **不**手动锁旧版本，不加 upper bound 阻止升级

#### [NEW] 烟雾测试脚本（临时，不提交 / 跑完即删）
- `scripts/smoke_test_deepagents.py`
- 验证：`from deepagents import create_deep_agent` OK
- 验证：`create_deep_agent(model=models.supervisor_llm, system_prompt="test")` 不报错
- 验证：`await agent.ainvoke({"messages": [HumanMessage(content="你好")]})` 能跑通并返回
- 记录请求/响应原始内容到 `scratch/smoke_test_baseline.json`

---

### Step 1–6 — 新建 `app/deep_agent.py`

#### [NEW] [deep_agent.py](file:///d:/code/VisualStudioCode/AI/project/GNK48-Agent/DeepAgentsRefactor/app/deep_agent.py)

文件结构（按实现顺序）：

**Step 2 — 主 Agent system_prompt**
- 定义 `MAIN_AGENT_PROMPT`（直接从用户规格抄入，无改动）

**Step 3 — 六个工具型专家声明式定义**
- 定义 `WORKER_BASE_PROMPT`（从 `graph.py` 的 `build_graph()` 内部原样搬出，不改一字）
- 定义 `SUBAGENT_DESCRIPTIONS` dict（6 个专家的描述字符串）
- 定义 `_build_subagent_specs()` 函数，返回 6 个专家的完整 dict 列表：
  ```
  KnowledgeAgent  → tools=[tools.rag, tools.web_search], model=models.vlm
  MediaAgent      → tools=[tools.av_graph_rag], model=models.worker_llm
  MapAgent        → tools=amap_tools (mcp_service.get_tools_by_server("amap")), model=models.worker_llm
  SQLAgent        → tools=[tools.execute_sql], model=models.worker_llm
  BrowserAgent    → tools=tools.browser_tools, model=models.worker_llm
  CodeAgent       → tools=code_tools, model=models.worker_llm
  ```
  每个 dict 包含：`name`, `description`, `system_prompt`, `tools`, `model`
  末尾注释占位：FileAgent / DesktopAgent 下一阶段以 CompiledSubAgent 接入

**Step 4 — 死循环防护**
- 用 `try/except ImportError` 保护 `ToolCallLimitMiddleware` 的导入
- 若导入成功，实例化 `task_limit = ToolCallLimitMiddleware(tool_name="task", run_limit=12, exit_behavior="continue")`
- 若导入失败，`task_limit = None`，注释说明"待 langchain 1.x 确认导入路径后接入"

**Step 5 — 长期记忆应用层函数**
- `async def build_initial_messages(user_text: str, user_id: str) -> list`
  - 调用 `database.search_memory(user_text, top_k=1)`
  - 若有结果，prepend `SystemMessage(content="【用户长期记忆】\n...")`
  - 末尾 append `HumanMessage(content=user_text, name="user")`
- `async def _background_extract_memory(recent_chat: str)` —— 从 `graph.py` 直接复用，不改逻辑

**Step 6 — 组装 main_agent**
- `async def build_main_agent() -> CompiledGraph`（异步工厂，需要先 await mcp 初始化）
  - 调用 `_build_subagent_specs()` 获取 subagents 列表
  - 构建 `middleware` 列表（若 `task_limit` 不为 None 则带入，否则为空）
  - 调用 `create_deep_agent(model=models.supervisor_llm, system_prompt=MAIN_AGENT_PROMPT, subagents=subagents, middleware=middleware, checkpointer=MemorySaver())`
  - 返回 agent 实例

**辅助：验证入口**
- 模块末尾提供 `if __name__ == "__main__": asyncio.run(_smoke_verify())` 供单独验证（不接入 chat.py）

---

### Step 7 — 验证脚本

#### [NEW] `scripts/verify_deep_agent.py`
- 独立可运行的验证脚本（`python scripts/verify_deep_agent.py`）
- 流程：`await mcp_service.initialize_mcp()` → `agent = await build_main_agent()` → `build_initial_messages(...)` → `await agent.ainvoke(...)` → 打印 result
- **不接入 `plugins/chat.py`**，不影响生产路径

---

## 明确不做的事

| 项目 | 理由 |
|------|------|
| 不动 `app/graph.py` | 新旧并存验证阶段 |
| 不动 `main.py` / `plugins/chat.py` | 不做流量切换 |
| 不接入 FileAgent / DesktopAgent | 下一阶段事项 |
| 不改 `mcp_servers.json` / `mcp_service.py` | filesystem MCP 保持现状 |
| 不加 TodoListMiddleware | 用户明确禁止 |
| 不实现精细的"同一子 Agent 连续 N 次"自定义中间件 | 用户明确推迟 |

---

## 验证计划

### 自动化测试
- `python scripts/smoke_test_deepagents.py` — 验证包安装和 API 连通性
- `python scripts/verify_deep_agent.py` — 端到端验证新骨架

### 手动验证检查点
1. `uv add "deepagents>=0.7,<0.8"` 完成后，确认 `langchain`/`langchain-core`/`langchain-google-genai` 版本被升级到 deepagents 要求的范围
2. 烟雾测试脚本能成功调用并拿到 LLM 响应（记录基线）
3. `verify_deep_agent.py` 中，6 个专家的工具能被正确注入（MapAgent 的 amap_tools 不为空）
4. `ToolCallLimitMiddleware` 的实际可用性（能导入就用，不能则注释占位，不阻塞其余工作）
5. `app/graph.py` 的 `build_graph()` 仍然可以正常 import（新旧两套架构可以同时 import）
