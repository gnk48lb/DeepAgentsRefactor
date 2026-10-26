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
1. **KnowledgeAgent**: 知识检索专家，优先检索本地 Milvus 知识库 (`rag`)，无果且具时效性时通过 Tavily 全网搜索 (`web_search`)；另提供 `get_current_time`（主 Agent 也直接挂载同一个工具，时间类问题由主 Agent 就地回答，不再委派）。
2. **MediaAgent**: 成人影视专家，同时挂载 Neo4j 知识图谱检索 (`av_graph_rag`) 与 MySQL 只读查询 (`execute_sql`)：图谱负责实体 / 合作关系网络与题材标签，数据库负责身高、罩杯、出生年份等精确属性过滤与发行年份统计，复合问题由它自己组合完成。
3. **MapAgent**: 地理与出行专家，挂载高德地图本地工具（共 12 个高德本地 Tool），支持天气查询、周边检索、路径规划与地理编码。约定：路线、周边搜索、测距的位置参数既可传文字地址也可传'经度,纬度'，工具内部自动解析（文字地址跨城有歧义时会报错要求补城市）；只有 maps_regeocode 必须传坐标。
4. **CodeAgent**: 代码执行专家，通过 Docker 隔离沙箱安全运行 Python 脚本，支持数据分析与 Matplotlib 绘图（绘图规范由 Skill `matplotlib-chart` 提供）。
5. **BrowserAgent**: 浏览器自动化专家，基于 Playwright/browser-use 操控真实或无头浏览器，处理网页端交互。
6. **FileAgent (HITL)**: 本地文件专家，内置 FilesystemMiddleware + FilesystemBackend + 原生 HITL。提供工具：`ls`/`glob`/`grep`/`read_file` 免授权，`convert_to_markdown`（由 markitdown-mcp 提供，支持富文档转 Markdown），`write_file`/`edit_file`/`create_directory`/`move_file` 需授权，删除 = 移动到 archive_trash 归档。只能在 workspace 工作区内操作。转换结果过大时会暂存到 `/_artifacts/large_tool_results/<id>`，FileAgent 用 `read_file` 分页读回。
7. **DesktopAgent (HITL)**: 桌面操作专家，基于网格坐标截屏、模拟点击与键鼠交互。

### 2.4 单用户模式
系统为单用户专属模式，仅供本人使用：
- 所有专家（含具有本地文件读写权限的 `FileAgent`、物理桌面控制的 `DesktopAgent` 以及自动化操作网页的 `BrowserAgent` 等全量 7 个专家）始终挂载。
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
- **注入控制**：注入条数由 config.MEMORY_TOP_K 控制（默认 3），相似度阈值由 MEMORY_MIN_SCORE 控制，可参考日志里的 [记忆检索] 分数调整。

### 2.7 Skills 扩展机制
系统采用 deepagents 官方 Skills 机制扩展专家的专业技能：
- **组织形式**：技能存放于 `skills/<name>/SKILL.md`，文件采用 YAML frontmatter 定义元数据，其中 `name` 必须与所在目录名完全一致，`description` 明确说明何时使用。
- **专家装配**：自定义子专家在声明式 spec 中显式配置 `skills=[SKILLS_ROUTE]`。
- **虚拟路由**：由 `app/agents/skills.py` 构建 `CompositeBackend`，将虚拟路径 `/skills/` 路由到本地真实的 `skills/` 磁盘目录。

### 2.8 专家步骤预算（独立停止条件）
每个专家在**一次 `task` 委派内**最多可调用的工具次数由 `app/config.py` 的 `SUBAGENT_TOOL_LIMITS` 配置（未登记的专家使用 `SUBAGENT_TOOL_LIMIT_DEFAULT`；统计包含 `read_file`、`write_todos` 等框架内置工具），实现见 `app/agents/budget.py`，两级止损：
- **软停止**：第 N+1 次起的工具调用不再执行，而是以 `Tool call limit exceeded` 错误回给模型，模型据已有结果收尾汇报（`WORKER_BASE_PROMPT` 的【步骤预算】一节教它这样做）。
- **硬停止**：模型无视软停止继续要工具时，在第 N+2 次模型调用后强制结束，`task` 返回 `Model call limits exceeded`。
- **实现方式**：声明式专家与 FileAgent 用 langchain 的 `ToolCallLimitMiddleware` + `ModelCallLimitMiddleware`；DesktopAgent 是手写子图，按已执行的工具步数手动计数，超限直接返回确定性汇报（不再调用模型）。
- **FileAgent 与 HITL**：授权恢复后计数重新开始；超预算被拦下的写入类调用不再弹授权。
- 预算是“止损阈值”而不是常规用量：日志里出现 `Tool call limit exceeded` / `Model call limits exceeded` 说明某个专家触顶，按真实用量调整 `SUBAGENT_TOOL_LIMITS`。

---

## 3. 目录结构

```text
DeepAgentsRefactor/
├── app/                      # 核心业务逻辑
│   ├── config.py             # L0: 全局配置中心 (路径与环境变量规范)
│   ├── bootstrap.py          # 启动环境补丁 (stdout/stderr UTF-8、第三方日志降噪、Windows os.rename 兜底)
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
│   │   ├── budget.py         # 专家工具调用预算（软 / 硬两级停止条件）
│   │   ├── prompts.py        # 主 Agent 提示词与唯一的 WORKER_BASE_PROMPT
│   │   ├── skills.py         # Skills 虚拟文件系统 Backend 路由
│   │   ├── orchestrator.py   # 主管智能体构建与多专家编排
│   │   └── experts/          # 7 个垂直领域子专家实现
│   │       ├── knowledge_agent.py
│   │       ├── media_agent.py
│   │       ├── map_agent.py
│   │       ├── code_agent.py
│   │       ├── browser_agent.py
│   │       ├── file_agent.py
│   │       └── desktop_agent.py
│   └── services/             # L5: 应用服务与业务流程调度
│       ├── startup.py        # 全局数据库与服务初始化入口
│       └── chat_service.py   # 单用户多模态会话状态机与消息调度服务
├── deploy/                   # 部署配置
│   └── docker-compose.yml    # Neo4j 等后台服务容器编排
├── mcp_servers/              # 独立 MCP 服务实现
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
│   ├── smoke_budget.py       # 专家工具调用预算（独立停止条件）冒烟测试
│   ├── smoke_skills.py       # Skills 规范与 Backend 路由冒烟测试
│   ├── smoke_amap_contract.py# 高德工具快照与契约冒烟测试
│   ├── smoke_amap_http.py    # 高德工具 HTTP 异常与密钥脱敏测试
│   └── smoke_chat_service.py # chat_service 服务层离线冒烟测试
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
  - 系统默认使用嵌入式 **Milvus Lite**（本地文件模式，存储于 `storage/gnk48_agent_collection.db`），无需启动 Docker 即可运行知识库检索与长期记忆存储。
  - `deploy/docker-compose.yml` 里的 Milvus/etcd/MinIO 服务仅在将 `MILVUS_DB_PATH` 配置为远程 HTTP URI 时才需要启用。
  - `--rebuild` 参数触发的物理索引清理仅对本地 Milvus Lite 生效。
- **Neo4j**:
  - 提供知识图谱存储与 Cypher 关系查询，供 `MediaAgent` 使用。可通过 `docker compose -f deploy/docker-compose.yml up -d neo4j` 启动。
- **MySQL**:
  - 存储女优与作品结构化关系数据，供 `MediaAgent` 的 `execute_sql` 工具精确筛选查询。MySQL 需用户自备（`deploy/docker-compose.yml` 中不包含）。建议为该工具分配专用只读账号（如 `GRANT SELECT ON av_db.* TO 'readonly'@'%'`），并在 `.env` 中配置对应的 `MYSQL_USER` 与 `MYSQL_PASSWORD`。
- **Docker Python 沙箱**:
  - 供 `CodeAgent` 启动轻量容器 (`python-ds:latest`) 安全运行数据分析与绘图代码。

### 4.4 QQ 机器人配置 (仅 bot.py 需要)
若使用 QQ 官方机器人（通过 `bot.py` 启动），需在 `.env` 中配置以下环境变量（NoneBot2 自动读取）：
- `DRIVER`：NoneBot 驱动，推荐 `~fastapi+~httpx+~websockets`。
- `QQ_IS_SANDBOX`：是否连接 QQ 开放平台沙箱环境（生产环境填 `false`，测试环境填 `true`）。
- `QQ_BOTS`：JSON 数组格式的机器人凭证列表（来源为 QQ 开放平台申请的机器人 AppID、Token、AppSecret）：
  - `id`：机器人 AppID。
  - `token`：机器人 Token。
  - `secret`：机器人 AppSecret。
  - `intent`：订阅事件权限字典，如 `{"c2c_group_at_messages": true, "direct_message": true}`（以已安装的 `nonebot/adapters/qq/config.py` 中的 `Intents` 字段名为准）。
  - `use_websocket`：是否使用 WebSocket 长连接接收事件（推荐 `true`）。

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
2. **定义专家规格**：在 `app/agents/experts/<x>_agent.py` 中定义 `NAME`、`DESCRIPTION`、`PROMPT`（基于统一的 `WORKER_BASE_PROMPT` 拼接），并导出 `SPEC = ExpertSpec(...)`，并在 `app/config.py` 的 `SUBAGENT_TOOL_LIMITS` 里登记它的工具调用预算（未登记则使用 `SUBAGENT_TOOL_LIMIT_DEFAULT`；`tests/smoke_budget.py` 的 A2 与 `tests/smoke_orchestrator.py` 的 D 会校验预算表与专家集合一致，需同步更新）。
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

测试包含以下 10 个测试套件（共 80 个场景）：
1. `tests/smoke_architecture.py`：分层单向依赖、语法禁令、空 `__init__`、提示词单例、专家规格、轻量隔离导入、无 Key 安全导入与 MCP 路径校验（8 个场景）。
2. `tests/smoke_file_agent.py`：FileAgent 离线冒烟测试（含 MarkItDown 路径安全拦截、越界隔离、HITL 恢复映射等 16 个场景）。
3. `tests/smoke_skills.py`：Skills 规范与 Backend 虚拟路由冒烟测试。
4. `tests/smoke_amap_contract.py`：高德地图 12 个本地工具契约与 Schema 快照测试（5 个场景）。
5. `tests/smoke_amap_http.py`：高德工具 HTTP 异常与密钥脱敏测试（13 个场景）。
6. `tests/smoke_chat_service.py`：多轮 HITL、发件箱图片连续性、并发忙提示、超时恢复与 `/reset` 测试（6 个场景）。
7. `tests/smoke_knowledge.py`：知识库建库、pending 增量入库、混合检索与长期记忆离线冒烟测试（5 个场景）。
8. `tests/smoke_orchestrator.py`：编排层主 Agent 装配、专家顺序、规格与 HarnessProfile 排除写工具安全测试（4 个场景，含主管时间工具与专家预算装配）。
9. `tests/smoke_units.py`：纯函数与小模块离线单元冒烟测试（路径解析、os.rename 兜底、MIME 判断、截图历史、未应答 tool_calls、SQL 加固与超时只读、rag 同源去重等 7 个场景）。
10. `tests/smoke_budget.py`：专家独立停止条件（工具调用预算）离线冒烟测试（声明式专家 / FileAgent / DesktopAgent 的软硬停止、计数器隔离、超预算写入不弹授权、生产递归上限下失控保护等 15 个场景）。

---

## 9. 已知限制

1. **DesktopAgent 不稳定**: 桌面操作依赖网格视觉识别，受分辨率和缩放比例影响较大，可能存在点击漂移。
2. **MemorySaver 会话重启丢失**: 当前状态保存器为内存级 `MemorySaver`，机器人进程重启后进行中的上下文与未决 HITL 中断会丢失。长期事实依赖 Milvus 后台抽取持久化。
3. **专家单次委派有工具调用预算**: 每个专家单次委派内的工具调用次数受 `config.SUBAGENT_TOOL_LIMITS` 限制（见 2.8；FileAgent 默认 20 次，授权恢复后重新计数），触顶后调用被拦截、专家以已有结果收尾，复杂任务可能因此汇报不完整；`write_file` 对已存在文件会整体覆盖。
