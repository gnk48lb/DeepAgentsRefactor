# Phase 1 deepagents 架构迁移验收报告

## 迁移概览

本阶段（Phase 1）完成了从原有的 Supervisor+Router 手动路由架构到 deepagents `create_deep_agent`（主 Agent ReAct 循环 + `task` 工具委派）的骨架搭建与 6 个工具型专家的声明式接入。迁移全程严格遵守隔离边界，未触碰任何生产入口与现有图定义文件。

---

## 变更明细

### 1. 依赖升级与环境配置
- **Git 分支**：在独立分支 `deepagents-phase1` 上开发并合并提交。
- **[pyproject.toml](file:///d:/code/VisualStudioCode/AI/project/GNK48-Agent/DeepAgentsRefactor/pyproject.toml)**：
  - 添加 `deepagents>=0.7,<0.8`（实际解析为 `0.7.13`）。
  - 升级 `langchain>=1.3.18`（实际为 `1.4.0`）、`langchain-core>=1.6.1`（实际为 `1.6.2`）、`langchain-google-genai>=4.3.7`。
  - 通过 `[tool.uv.override-dependencies]` 声明 `anthropic>=1.4.0` 解决 `browser-use` 历史固钉冲突。
  - `uv.lock` 重新解析锁定，并在 `.venv` 完成全部 210 个依赖包的同步。

### 2. 核心模块 [app/deep_agent.py](file:///d:/code/VisualStudioCode/AI/project/GNK48-Agent/DeepAgentsRefactor/app/deep_agent.py)
- **主 Agent**：基于 `create_deep_agent` 构建，配置规范化 `MAIN_AGENT_PROMPT`（去代词化、过程不可见、图文分离、信息搬运工原则，去除冗余专家名录），主模型使用 `models.supervisor_llm` (`gemini-3.1-flash-lite`)。
- **6 个声明式工具型专家**：
  1. `KnowledgeAgent`: 模型使用 `models.vlm`，负责 RAG 本地知识库检索 (`rag`) 与时效网络检索 (`web_search`)。明确排除天气/出行问题。
  2. `MediaAgent`: 负责音视频专有知识图谱检索 (`av_graph_rag`)。
  3. `MapAgent`: 负责高德地图服务（动态挂载 `amap` MCP 工具），明确包含地理编码、天气查询与路线规划。
  4. `CodeAgent`: 负责 Python 沙箱执行与数据可视化 (`code_tools`)。
  5. `BrowserAgent`: 负责网页读取、B站主页分析及复杂动态网页交互 (`browser_tools`)。
  6. `SQLAgent`: 负责 MySQL 关系数据库只读查询与自动纠错 (`execute_sql`)。
  *(注：FileAgent 与 DesktopAgent 保留注释占位，留待后续阶段处理)*。
- **防跑飞兜底守护**：
  - 接入 `ToolCallLimitMiddleware(tool_name="task", run_limit=12, exit_behavior="continue")`，拦截异常死循环委派。
- **应用层长期记忆适配**：
  - `build_initial_messages()` 真实查询 Milvus 向量库 `database.search_memory(user_query, top_k=1)`，动态注入 `【用户长期记忆】` 到 `SystemMessage`。
  - 保留并复用 `_background_extract_memory()` 后台异步提取保存机制。
- **异步构建入口**：
  - 提供 `await build_main_agent(checkpointer=None)`，内部自愈保证 MCP 服务就绪，默认配置 `MemorySaver`。

### 3. 系统兼容与配置完善
- **[config.py](file:///d:/code/VisualStudioCode/AI/project/GNK48-Agent/DeepAgentsRefactor/config.py)**：支持 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY` 双环境变量回退。
- **[app/database.py](file:///d:/code/VisualStudioCode/AI/project/GNK48-Agent/DeepAgentsRefactor/app/database.py)**：解决 Windows 平台 Milvus Lite 重新连接后需要显式 `load_collection` 以及 `flush` 临时文件替换锁的问题。

---

## 验证与实测结果

### 1. 基础环境冒烟测试 ([scripts/smoke_test_deepagents.py](file:///d:/code/VisualStudioCode/AI/project/GNK48-Agent/DeepAgentsRefactor/scripts/smoke_test_deepagents.py))
- **deepagents 版本**：0.7.13
- **ToolCallLimitMiddleware 导入**：`langchain.agents.middleware.ToolCallLimitMiddleware` 可用
- **ChatGoogleGenerativeAI 兼容性**：`gemini-3.1-flash-lite` 正常驱动 deepagents 单轮对话，产出基准文件 `scratch/smoke_test_baseline.json`。

### 2. 双架构并存与真实记忆注入验证 ([scripts/verify_deep_agent.py](file:///d:/code/VisualStudioCode/AI/project/GNK48-Agent/DeepAgentsRefactor/scripts/verify_deep_agent.py))
- **并存验证**：`app.deep_agent` 与旧 `app.graph` 同时 import 无任何符号冲突或覆盖。
- **编译检查**：主 Agent 图节点包含 `['__start__', 'model', 'tools', 'PatchToolCallsMiddleware.before_agent', 'ToolCallLimitMiddleware[task].after_model']`。
- **真实 Milvus 长期记忆插入与检索**：
  - 插入记忆：“用户是一名 Python 开发者，喜欢简洁明了的回复”
  - `build_initial_messages` 自动检索 top_1 并在 `SystemMessage` 中生成 `【用户长期记忆】`。
- **端到端对话**：成功响应自我介绍并详述了 6 大专家职责分工，报告存入 `scratch/verify_results.json`。

### 3. 子专家 Task 委派端到端实测 ([scripts/verify_subagent_delegation.py](file:///d:/code/VisualStudioCode/AI/project/GNK48-Agent/DeepAgentsRefactor/scripts/verify_subagent_delegation.py))
- **测试查询**：“请帮我查询济南市今天的天气情况。”
- **实际执行轨迹 (Trajectory)**：
  ```text
  Step 0 [SystemMessage]: 【用户长期记忆】 用户是一名 Python 开发者，喜欢简洁明了的回复...
  Step 1 [HumanMessage]: 请帮我查询济南市今天的天气情况。...
  Step 2 [AIMessage]: Tool Call -> task(args={"subagent_type": "MapAgent", "description": "查询济南市今天的天气情况。"}...)
  Step 3 [ToolMessage]: 济南市今日（2026-09-15）天气情况：晴，气温 21℃ - 33℃，西南风 1-3 级。...
  Step 4 [AIMessage]: 济南市今日（2026-09-15）天气情况：晴，气温 21℃ - 33℃，西南风 1-3 级。
  ```
- **委派与工具路由成功率**：100% 精准识别天气属于地图出行领域，成功委派至 `MapAgent`，并调用高德 MCP 工具返回实时天气。

---

## 生产安全防线核对

| 文件/模块 | 状态 | 说明 |
| :--- | :--- | :--- |
| `app/graph.py` | 未修改 | 旧架构完整保留 |
| `main.py` | 未修改 | 生产入口未触碰 |
| `plugins/chat.py` | 未修改 | 生产插件未触碰 |
| `mcp_servers.json` | 未修改 | MCP 配置未触碰 |
| `app/mcp_service.py` | 未修改 | MCP 服务层未触碰 |

Phase 1 所有目标均已高质量达成，所有修改已提交至 Git。
