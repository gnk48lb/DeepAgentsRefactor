# GNK48-Agent (DeepAgents Refactor)

基于 **deepagents** 与 **LangGraph** 构建的多智能体 (Multi-Agent) 协作系统与 QQ 机器人助理。

---

## 1. 项目简介

GNK48-Agent 采用声明式主智能体配合垂直领域子专家架构。主 Agent 基于 ReAct 循环进行意图分析与全局任务规划，通过 deepagents 内置的 `task` 工具将子任务委派给各个领域专家；针对涉及敏感操作的专家，通过编译式 LangGraph 子图实现单步 Human-in-the-loop (HITL) 动态授权与状态隔离。

---

## 2. 系统架构与分层规则

### 2.1 分层架构与依赖约束（单向，只能向下 import）
系统严格遵循 6 层单向依赖架构：
```text
L5 Services  (app/services/startup.py, app/services/chat_service.py)
   ↓
L4 Agents    (app/agents/spec.py, prompts.py, orchestrator.py, experts/*)
   ↓
L3 Tools     (app/tools/*_tools.py, app/tools/amap)
   ↓
L2 Knowledge & Memory (app/knowledge/*, app/memory/*)
   ↓
L1 Runtime & Infra   (app/runtime/*, app/infra/*)
   ↓
L0 Config & Utils    (app/config.py, app/utils/*)
```

- `app.config` 允许被系统任意层直接 import。
- 同层不同子包互不 import（`runtime` ↔ `infra`、`knowledge` ↔ `memory` 相互隔离）。
- `app/tools/*_tools.py` 之间互不 import。
- `app/agents/experts/*` 各专家之间互不 import。
- `app.bootstrap` 仅在应用最顶层入口由 `main.py` / `bot.py` 调用，`app/` 内部禁止 import 它。
- `mcp_servers/` 独立于应用层，禁止 import `app.*`。

### 2.2 主 Agent 与 Task 委派
- **Main Agent (Orchestrator)**: 运行 ReAct 决策循环，负责分析用户意图、协调任务依赖、汇总专家汇报并最终回复用户。
- **任务委派**: 通过 deepagents 的 `task` 工具按需调遣特定的子专家，执行完成后仅将纯净战报回传主 Agent。
- **图片发件箱 (Outbox)**: 子 Agent 产出的图片（如代码绘图）通过旁路发件箱系统收集并递交给接入层，避免将海量 Base64 塞入主 Agent 上下文。

### 2.3 子专家清单
1. **KnowledgeAgent**: 知识检索专家，优先检索本地 Milvus 知识库 (`rag`)，无果且具时效性时通过 Tavily 全网搜索 (`web_search`)；另提供 `get_current_time`，回答当前日期 / 时间 / 星期类问题。
2. **MediaAgent**: 专有图谱检索专家，基于 Neo4j 知识图谱 (`av_graph_rag`) 进行实体与关系网络查询。
3. **MapAgent**: 地理与出行专家，挂载高德地图本地工具（共 16 个高德本地 Tool），支持天气查询、周边检索、路径规划与地理编码。
4. **CodeAgent**: 代码执行专家，通过 Docker 隔离沙箱安全运行 Python 脚本，支持数据分析与 Matplotlib 绘图（绘图规范由 Skill `matplotlib-chart` 提供）。
5. **SQLAgent**: 关系数据库专家，负责将自然语言转为 SQL 并查询本地 MySQL (`execute_sql`)。
6. **BrowserAgent**: 浏览器自动化专家，基于 Playwright/browser-use 操控真实或无头浏览器，处理网页端交互。
7. **FileAgent (HITL)**: 本地文件专家，内置 FilesystemMiddleware + FilesystemBackend + 原生 HITL。提供工具：`ls`/`glob`/`grep`/`read_file` 免授权，`convert_to_markdown`（由 markitdown-mcp 提供，支持富文档转 Markdown），`write_file`/`edit_file`/`create_directory`/`move_file` 需授权，删除 = 移动到 archive_trash 归档。只能在 workspace 工作区内操作。转换结果过大时会暂存到 `/_artifacts/large_tool_results/<id>`，FileAgent 用 `read_file` 分页读回。
8. **DesktopAgent (HITL)**: 桌面操作专家，基于网格坐标截屏、模拟点击与键鼠交互。

### 2.4 单用户模式
系统为单用户专属模式，仅供本人使用：
- 所有专家（含具有本地文件读写权限的 `FileAgent`、物理桌面控制的 `DesktopAgent` 以及自动化操作网页的 `BrowserAgent` 等全量 8 个专家）始终挂载。
- 系统内部没有任何用户身份或权限校验逻辑。
- 对话会话使用固定的 `thread_id`（`"main"`），私聊与群聊共享同一份对话上下文。
- 发送 `/reset`（或 `/new`、`/新对话`）可清空固定会话，含挂起中的授权。

> **警告**：机器人必须保持私有；若将来对外开放，需要重新引入用户隔离与权限层。

### 2.5 MCP 常驻会话与容错机制
系统通过 Model Context Protocol (MCP) 集成外部能力（如文档转换 `markitdown`、桌面控制 `desktop`）。
各 MCP 服务进程在机器人启动时由专属后台 Runner 统一拉起并常驻运行，退出时自动回收，大幅降低工具调用开销。
运行时支持并发调用与调用超时熔断（默认 60s）；若检测到服务异常断开或进程挂死，将自动触发重连与会话重建。

### 2.6 长期记忆
系统基于 Milvus 向量库维护跨会话长期记忆：
- **检索过滤**：检索时通过 `config.MEMORY_MIN_SCORE` 进行相似度阈值过滤，低于阈值的记忆视为与当前问题无关，不注入对话上下文。
- **写入去重**：写入时通过 `config.MEMORY_DEDUP_SCORE` 进行语义去重，与已有记忆相似度高于该阈值时自动覆盖旧记忆。
- **日志审计**：记忆检索阶段输出 `[记忆检索]` 标签日志，明确标注每条命中条目的分数与采用/丢弃状态。

### 2.7 Skills 扩展机制
系统采用 deepagents 官方 Skills 机制扩展专家的专业技能：
- **组织形式**：技能存放于 `skills/<name>/SKILL.md`，文件采用 YAML frontmatter 定义元数据，其中 `name` 必须与所在目录名完全一致，`description` 明确说明何时使用。
- **专家装配**：自定义子专家在声明式 spec 中显式配置 `skills=[SKILLS_ROUTE]`。
- **虚拟路由**：由 `app/agents/skills.py` 构建 `CompositeBackend`，将虚拟路径 `/skills/` 路由到本地真实的 `skills/` 磁盘目录。

---

## 3. 目录结构

```text
DeepAgentsRefactor/
├── app/                      # 核心业务逻辑
│   ├── config.py             # L0: 全局配置中心 (路径与环境变量规范)
│   ├── bootstrap.py          # 启动环境补丁 (Windows 编码 / 事件循环)
│   ├── utils/                # L0: 通用基础工具
│   │   ├── text.py           # 纯文本抽取与规范化
│   │   └── images.py         # 图像压缩、MIME 类型推导与 Base64 编码
│   ├── runtime/              # L1: 运行时公共辅助
│   │   ├── outbox.py         # 多模态消息发件箱旁路
│   │   ├── hitl.py           # 人工授权 (HITL) 辅助与中断恢复
│   │   └── cli_logger.py     # 终端 ANSI 彩色日志与角色识别审计
│   ├── infra/                # L1: 基础设施与外设服务
│   │   ├── models.py         # LLM / VLM / 向量模型 / 重排器惰性单例
│   │   ├── milvus_client.py  # Milvus 客户端单例工厂
│   │   ├── sandbox.py        # Docker Python 沙箱代码执行环境
│   │   ├── browser_service.py# 浏览器自动化控制封装 (Playwright)
│   │   └── mcp_service.py    # MCP 协议客户端管理
│   ├── knowledge/            # L2: 知识库检索与构建
│   │   ├── knowledge_base.py # Milvus 知识库初始化与增量入库
│   │   ├── loader.py         # 多模态文档解析、切片与图片编码
│   │   ├── search.py         # 混合检索执行与 Reranker 重排
│   │   ├── query_transform.py# 查询转换策略 (HyDE, Step-Back, Multi-Query)
│   │   └── media/            # 媒体专有图谱构建与检索
│   │       ├── builder.py    # 媒体库初始化与索引构建
│   │       ├── retriever.py  # 媒体混合检索器 (Neo4j + Milvus)
│   │       └── intent.py     # 媒体意图识别
│   ├── memory/               # L2: 跨会话长期记忆
│   │   ├── store.py          # Milvus 记忆集合初始化、检索与去重写入
│   │   ├── injector.py       # 对话初次消息记忆注入与过滤
│   │   └── extractor.py      # 后台记忆抽取与异步落盘
│   ├── tools/                # L3: 工具集合与适配器
│   │   ├── knowledge_tools.py# RAG / Tavily / 时间查询工具
│   │   ├── media_tools.py    # 媒体图谱检索工具
│   │   ├── sql_tools.py      # MySQL 查询工具
│   │   ├── code_tools.py     # Python 代码执行沙箱工具
│   │   ├── browser_tools.py  # 网页自动化浏览工具
│   │   ├── markitdown_tool.py# FileAgent 专属 MarkItDown 工具包装
│   │   └── amap/             # 高德地图本地 Tool 实现与契约快照
│   ├── agents/               # L4: 智能体编排与专家定义
│   │   ├── spec.py           # 声明式 ExpertSpec 数据结构
│   │   ├── prompts.py        # 主 Agent 提示词与唯一的 WORKER_BASE_PROMPT
│   │   ├── skills.py         # Skills 虚拟文件系统 Backend 路由
│   │   ├── orchestrator.py   # 主管智能体构建与多专家编排
│   │   └── experts/          # 8 个垂直领域子专家实现
│   │       ├── knowledge_agent.py
│   │       ├── media_agent.py
│   │       ├── map_agent.py
│   │       ├── code_agent.py
│   │       ├── sql_agent.py
│   │       ├── browser_agent.py
│   │       ├── file_agent.py
│   │       └── desktop_agent.py
│   └── services/             # L5: 应用服务与业务流程调度
│       ├── startup.py        # 全局数据库与服务初始化入口
│       └── chat_service.py   # 单用户多模态会话状态机与消息调度服务
├── deploy/                   # 部署配置
│   └── docker-compose.yml    # Neo4j 等后台服务容器编排
├── mcp_servers/              # 独立 MCP 服务实现
│   ├── desktop_agent_server.py
│   └── windows_desktop.py
├── plugins/                  # NoneBot2 插件层
│   └── chat.py               # QQ 机器人消息接收与 ChatService 调度
├── scripts/                  # 运维与工具脚本
│   ├── login_setup.py        # 浏览器自动化持久化登录引导脚本
│   ├── manage_memory.py      # Milvus 长期记忆查看与清理工具
│   └── migrate_to_mysql.py   # 数据迁移至 MySQL
├── skills/                   # deepagents 官方 Skills 目录
│   └── matplotlib-chart/     # Matplotlib 绘图规范技能
├── tests/                    # 测试用例
│   ├── fixtures/amap/        # 高德工具契约与 Golden 样本数据
│   ├── run_smoke.py          # 一键运行全部离线冒烟测试
│   ├── smoke_architecture.py # 架构分层、规范与契约冒烟测试
│   ├── smoke_file_agent.py   # FileAgent 离线冒烟测试
│   ├── smoke_skills.py       # Skills 规范与 Backend 路由冒烟测试
│   ├── smoke_amap_contract.py# 高德工具快照与契约冒烟测试
│   ├── smoke_amap_http.py    # 高德工具 HTTP 异常与密钥脱敏测试
│   ├── smoke_chat_service.py # chat_service 服务层离线冒烟测试
│   ├── live_amap_golden.py   # 高德工具在线 Golden 结构对比（--live）
│   └── eval/                 # 路由行为评测（见 8.1）
├── storage/                  # 运行时持久化目录（Milvus DB、浏览器配置等，已忽略）
├── workspace/                # FileAgent 操作工作区（已忽略）
├── bot.py                    # NoneBot2 QQ 机器人启动入口
├── main.py                   # 本地控制台多轮对话入口 (合并了原 CLI 功能)
├── mcp_servers.json          # MCP 服务配置文件
└── pyproject.toml            # 依赖与项目元数据 (uv)
```

---

## 4. 环境准备与服务组件说明

### 4.1 安装依赖
本项目推荐使用 `uv` 进行环境同步与包管理：
```bash
uv sync
```

### 4.2 配置环境变量
参考 `.env.example` 在根目录下创建 `.env` 文件并填入相应 API 密钥：
- `GEMINI_API_KEY`（或 `GOOGLE_API_KEY`）：必填，Gemini 模型 API 密钥。
- `AMAP_API_KEY`：高德地图 Web 服务 API Key。
- `TAVILY_API_KEY`：Tavily 搜索 API Key。
- `SILICON_FLOW_API_KEY`：硅基流动 Embedding / Reranker API Key。
- `HUGGINGFACEHUB_API_TOKEN`：Hugging Face 访问令牌。

### 4.3 基础服务组件与说明
- **Milvus 向量数据库说明**:
  - 系统默认使用嵌入式 **Milvus Lite**（本地文件模式，存储于 `storage/milvus_data.db`），无需启动 Docker 即可运行知识库检索与长期记忆存储。
  - `deploy/docker-compose.yml` 里的 Milvus/etcd/MinIO 服务仅在将 `MILVUS_DB_PATH` 配置为远程 HTTP URI 时才需要启用。
  - `--rebuild` 参数触发的物理索引清理仅对本地 Milvus Lite 生效。
- **Neo4j**:
  - 提供知识图谱存储与 Cypher 关系查询，供 `MediaAgent` 使用。可通过 `docker compose -f deploy/docker-compose.yml up -d neo4j` 启动。
- **MySQL**:
  - 存储女优与作品结构化关系数据，供 `SQLAgent` 精确筛选查询。MySQL 需用户自备（`deploy/docker-compose.yml` 中不包含）。
- **Docker Python 沙箱**:
  - 供 `CodeAgent` 启动轻量容器 (`python-ds:latest`) 安全运行数据分析与绘图代码。

---

## 5. 运行方式

### 5.1 启动控制台版本（多轮对话与 CLI 合并入口）
```bash
# 启动普通交互式控制台
uv run python main.py

# 首条消息附带图片
uv run python main.py --image path/to/picture.jpg

# 强制重建本地知识库索引
uv run python main.py --rebuild
```

### 5.2 启动 QQ 机器人 (生产入口)
```bash
uv run python bot.py
```

### 5.3 构建媒体库索引
```bash
uv run python -m app.knowledge.media.builder
```

---

## 6. 新增专家步骤

若需向系统扩展新的垂直领域专家，按以下 6 步标准流程实施：
1. **定义工具集**：在 `app/tools/<x>_tools.py` 中编写 `@tool` 函数并导出 `TOOLS = [...]`。
2. **定义专家规格**：在 `app/agents/experts/<x>_agent.py` 中定义 `NAME`、`DESCRIPTION`、`PROMPT`（基于统一的 `WORKER_BASE_PROMPT` 拼接），并导出 `SPEC = ExpertSpec(...)`。
3. **注册至主管**：在 `app/agents/orchestrator.py` 的 `_DECLARATIVE_SPECS` 列表中按期望的展示顺序追加该专家的 `SPEC`。
4. **补充委派规则**：在 `app/agents/prompts.py` 的 `MAIN_AGENT_PROMPT` 任务委派决策流程中补充何时委派给该专家的指示。
5. **补充审计名单**：在 `app/runtime/cli_logger.py` 的 `_identify_agent` 专家名称白名单中添加该专家的名称。
6. **更新架构验证**：在 `tests/smoke_architecture.py` 的专家规格契约断言表（`test_e_expert_spec_table`）中新增一行断言。

---

## 7. HITL 人工授权说明

当 `FileAgent` 或 `DesktopAgent` 尝试执行高危操作（如写文件、编辑文件、移动文件、执行高危系统按键）时，系统会自动挂起并弹出授权请求：
```text
⚠️ 敏感操作授权拦截
──────────────────
操作: [write_file]
目标: workspace/test.txt
──────────────────
回复 Y 允许，回复 N 拒绝。
```
- **回复 `Y` / `y`**：系统恢复执行并完成敏感工具调用。
- **回复 `N` / `n`**：阻断该操作，向专家注入明确拒绝信息并由其礼貌告知用户。
- **发送 `/reset`**：可随时丢弃未决的授权与历史上下文，重新开始新对话。

---

## 8. 测试体系

本项目包含完整的离线冒烟测试套件，在无 API Key、无 Docker、无网络连接的离线环境下即可完整验证：
```bash
# 一键运行全部离线冒烟测试
uv run python tests/run_smoke.py
```

测试包含以下 6 个测试套件（共 37 个场景）：
1. `tests/smoke_architecture.py`：分层单向依赖、语法禁令、空 `__init__`、提示词单例、专家规格、轻量隔离导入、无 Key 安全导入与 MCP 路径校验（8 个场景）。
2. `tests/smoke_file_agent.py`：FileAgent 离线冒烟测试（含 MarkItDown 路径安全拦截、越界隔离、HITL 恢复映射等 16 个场景）。
3. `tests/smoke_skills.py`：Skills 规范与 Backend 虚拟路由冒烟测试。
4. `tests/smoke_amap_contract.py`：高德地图 16 个本地工具契约与 Schema 快照测试。
5. `tests/smoke_amap_http.py`：高德工具 HTTP 异常与密钥脱敏测试。
6. `tests/smoke_chat_service.py`：多轮 HITL、发件箱图片连续性、并发忙提示、超时恢复与 `/reset` 测试（6 个场景）。

### 8.1 路由行为评测

冒烟测试验证的是**架构契约**（工具名、依赖方向、导入隔离），不涉及模型决策。若要衡量「orchestrator 把 query 派给哪个专家」的行为表现，使用路由评测：

```bash
# 关键词基线（离线、零成本，仅用于验证评测集与管线）
uv run python tests/eval/run_routing_eval.py

# 真实模型（产生 API 费用；subagent 全部替换为空壳，不会写文件/动桌面/发请求）
uv run python tests/eval/run_routing_eval.py --live -v

# 只跑指定用例组，或保存基线用于 A/B 对比
uv run python tests/eval/run_routing_eval.py --live --only G,S --save baseline.json
```

- **用例集**：`tests/eval/routing_cases.json`，每条含 `expected`（必须委派）与 `forbidden`（绝不能委派）两个方向。
- **用途**：修改提示词、拆分/合并专家后，用它验证行为是否变化；这是判断「架构调整是否真的改善结果」的依据。
- **注意**：`--live` 会消耗 Gemini 配额（免费档约 15 请求/分钟），单条超时可用 `EVAL_CASE_TIMEOUT` 调整。

---

## 9. 已知限制

1. **DesktopAgent 不稳定**: 桌面操作依赖网格视觉识别，受分辨率和缩放比例影响较大，可能存在点击漂移。
2. **MemorySaver 会话重启丢失**: 当前状态保存器为内存级 `MemorySaver`，机器人进程重启后进行中的上下文与未决 HITL 中断会丢失。长期事实依赖 Milvus 后台抽取持久化。
3. **FileAgent 工具调用步数限制**: 单次 FileAgent 运行串行工具调用上限约 19 次（带 HITL 中间件每次工具调用占 3 个 superstep）；`write_file` 对已存在文件会整体覆盖。
