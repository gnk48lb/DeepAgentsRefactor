# GNK48-Agent (DeepAgents Refactor)

基于 **deepagents** (0.7.x) 与 **LangGraph** 构建的多智能体 (Multi-Agent) 协作系统与 QQ 机器人助理。

---

## 1. 项目简介

GNK48-Agent 采用声明式主智能体配合垂直领域子专家架构。主 Agent 基于 ReAct 循环进行意图分析与全局任务规划，通过 deepagents 内置的 `task` 工具将子任务委派给各个领域专家；针对涉及敏感操作的专家，通过编译式 LangGraph 子图实现单步 Human-in-the-loop (HITL) 动态授权与状态隔离。

---

## 2. 系统架构

### 2.1 主 Agent 与 Task 委派
- **Main Agent**: 运行 ReAct 决策循环，负责分析用户意图、协调任务依赖、汇总专家汇报并最终直接回复用户。挂载 `ToolCallLimitMiddleware` 防止单轮死循环。
- **任务委派**: 通过 deepagents 的 `task` 工具按需调遣特定的子专家，执行完成后仅将纯净战报回传主 Agent。
- **图片发件箱 (Outbox)**: 子 Agent 产出的图片（如代码绘图）通过旁路发件箱系统收集并递交给接入层，避免将海量 Base64 塞入主 Agent 上下文。

### 2.2 子专家清单
1. **KnowledgeAgent**: 知识检索专家，优先检索本地 Milvus 知识库 (`rag`)，无果且具时效性时通过 Tavily 全网搜索 (`web_search`)；另提供 `get_current_time`，回答当前日期 / 时间 / 星期类问题。
2. **MediaAgent**: 专有图谱检索专家，基于 Neo4j 知识图谱 (`av_graph_rag`) 进行实体与关系网络查询。
3. **MapAgent**: 地理与出行专家，挂载高德地图本地工具（原 MCP 已平滑迁移为本地 Tool），支持天气查询、周边检索、路径规划与地理编码。
4. **CodeAgent**: 代码执行专家，通过 Docker 隔离沙箱安全运行 Python 脚本，支持数据分析与 Matplotlib 绘图（绘图规范由 Skill `matplotlib-chart` 提供）。
5. **SQLAgent**: 关系数据库专家，负责将自然语言转为 SQL 并查询本地 MySQL (`execute_sql`)。
6. **BrowserAgent**: 浏览器自动化专家，基于 Playwright/browser-use 操控真实或无头浏览器，处理网页端交互。
7. **FileAgent (HITL)**: 本地文件专家，内置 FilesystemMiddleware + FilesystemBackend + 原生 HITL。提供工具：`ls`/`glob`/`grep`/`read_file` 免授权，`convert_to_markdown`（由 markitdown-mcp 提供，支持富文档转 Markdown），`write_file`/`edit_file`/`create_directory`/`move_file` 需授权，删除 = 移动到 archive_trash 归档。只能在 workspace 工作区内操作。转换结果过大时会暂存到 `/_artifacts/large_tool_results/<id>`，FileAgent 用 `read_file` 分页读回。
8. **DesktopAgent (HITL)**: 桌面操作专家，基于网格坐标截屏、模拟点击与键鼠交互。

### 2.3 单用户模式
系统为单用户专属模式，仅供本人使用：
- 所有专家（含具有本地文件读写权限的 `FileAgent`、物理桌面控制的 `DesktopAgent` 以及自动化操作网页的 `BrowserAgent` 等全量 8 个专家）始终挂载。
- 系统内部没有任何用户身份或权限校验逻辑。
- 对话会话使用固定的 `thread_id`（`"main"`），私聊与群聊共享同一份对话上下文。
- 发送 `/reset`（或 `/new`、`/新对话`）可清空固定会话，含挂起中的授权。

> **警告**：机器人必须保持私有；若将来对外开放，需要重新引入用户隔离与权限层。

### 2.4 MCP 常驻会话与容错机制
系统通过 Model Context Protocol (MCP) 集成外部能力（如文档转换 `markitdown`、桌面控制 `desktop`）。
各 MCP 服务进程在机器人启动时由专属后台 Runner 统一拉起并常驻运行，退出时自动回收，大幅降低工具调用开销。
运行时支持并发调用与调用超时熔断（默认 60s）；若检测到服务异常断开或进程挂死，将自动触发重连与会话重建，确保长时间运行的稳定性。
markitdown 通过 uvx 运行，首次启动要下载 markitdown[all] 的依赖，可能超过 MCP_STARTUP_TIMEOUT_SECONDS（180 秒）；建议首次先手动预热：`uvx --from markitdown-mcp==0.0.1a7 --with "markitdown[all]" markitdown-mcp --help`（若它没有立刻退出，Ctrl+C 即可）。

### 2.5 长期记忆
系统基于 Milvus 向量库维护跨会话长期记忆：
- **检索过滤**：检索时通过 `config.MEMORY_MIN_SCORE` 进行相似度阈值过滤，低于阈值的记忆视为与当前问题无关，不注入对话上下文。
- **写入去重**：写入时通过 `config.MEMORY_DEDUP_SCORE` 进行语义去重，与已有记忆相似度高于该阈值时自动覆盖旧记忆。
- **日志审计**：记忆检索阶段输出 `[记忆检索]` 标签日志，明确标注每条命中条目的分数与采用/丢弃状态，便于阈值校准。

### 2.6 Skills
系统采用 deepagents 官方 Skills 机制扩展专家的专业技能：
- **组织形式**：技能存放于 `skills/<name>/SKILL.md`，文件采用 YAML frontmatter 定义元数据，其中 `name` 必须与所在目录名完全一致，`description` 明确说明何时使用。
- **专家装配**：自定义子专家不会自动继承主 Agent 的 skills，需在自身声明式 spec 中显式配置 `skills=[SKILLS_ROUTE]`。
- **虚拟路由**：由 `app/skills.py` 构建 `CompositeBackend`，将虚拟路径 `/skills/` 路由到本地真实的 `skills/` 磁盘目录；子专家在需要出图时通过内置 `read_file` 按需读取技能规范。
- **新增技能步骤**：
  1. 在 `skills/<skill-name>/` 下创建 `SKILL.md`，编写规范的 frontmatter（name 与目录一致，明确 description）及正文；
  2. 若给子专家使用，在 `app/deep_agent.py` 对应专家的 spec 中添加 `skills: [SKILLS_ROUTE]`，并在其 prompt 中提示触发条件与读取路径；
  3. 运行 `tests/smoke_skills.py` 进行静态规范与 backend 虚拟路由验证。
- **编写规范**：`SKILL.md` 保持精简（建议 ≤ 100 行），一次 `read_file` 调用即可完整读完。

---

## 3. 目录结构

```text
DeepAgentsRefactor/
├── app/                      # 核心业务逻辑
│   ├── amap/                 # 高德地图本地 Tool 实现与契约快照
│   ├── browser_service.py    # 浏览器自动化控制封装 (Playwright)
│   ├── chat_service.py       # 统一对话服务层（处理锁、并发、HITL与发件箱）
│   ├── database.py           # Milvus 向量库与长期记忆管理
│   ├── deep_agent.py         # 主 Agent 与声明式专家装配中心
│   ├── file_agent.py         # FileAgent 定义（内置 FilesystemMiddleware + 原生 HITL）
│   ├── graph.py              # DesktopAgent 子图定义与提示词
│   ├── graph_rag.py          # 媒体图谱检索器 (Neo4j + Milvus Hybrid)
│   ├── hitl.py               # 人工授权 (HITL) 公共辅助与中断恢复
│   ├── loader.py             # 文档与多模态数据加载、切片与图片编码
│   ├── markitdown_tool.py    # FileAgent 专属 MarkItDown 工具包装与工作区路径映射
│   ├── mcp_service.py        # MCP 协议客户端管理 (MarkItDown, Desktop)
│   ├── models.py             # LLM / VLM / 向量模型与重排器单例定义
│   ├── outbox.py             # 图片跨轮发件箱旁路
│   ├── sandbox.py            # Docker Python 沙箱代码执行环境
│   ├── skills.py             # Skills 虚拟文件系统 Backend 路由配置
│   ├── tools.py              # 本地工具集合（RAG, Docker Code, SQL 等）
│   ├── transforms.py         # 查询转换增强 (HyDE, Step-Back, Multi-Query) 与意图路由
│   ├── utils.py              # 通用辅助工具 (extract_text)
│   └── windows_mcp.py        # Windows 桌面/原生交互 MCP 工具适配
├── plugins/                  # NoneBot2 插件层
│   └── chat.py               # QQ 消息接收与 ChatService 薄封装
├── scripts/                  # 运维与工具脚本
│   ├── chat_cli.py           # 命令行多轮对话与 HITL 测试终端
│   ├── docker-compose.yml    # Neo4j / Milvus / MinIO / Etcd 本地组件容器编排
│   ├── login_setup.py        # 浏览器自动化持久化登录引导脚本
│   ├── manage_memory.py      # Milvus 长期记忆查看与清理工具
│   └── migrate_to_mysql.py   # 数据迁移至 MySQL
├── skills/                   # deepagents 官方 Skills 目录
│   └── matplotlib-chart/     # Matplotlib 绘图规范技能
│       └── SKILL.md          # 技能指令与固定模板
├── tests/                    # 测试用例
│   ├── fixtures/amap/        # 高德工具契约与 Golden 样本数据
│   ├── live_amap_golden.py   # 高德工具在线 Golden 结构对比测试
│   ├── run_smoke.py          # 一键运行全部离线冒烟测试
│   ├── smoke_amap_contract.py # 高德工具快照与契约冒烟测试
│   ├── smoke_amap_http.py     # 高德工具 HTTP 异常与密钥脱敏测试
│   ├── smoke_chat_service.py # chat_service 服务层离线冒烟测试
│   ├── smoke_file_agent.py   # FileAgent 离线冒烟测试（含 MarkItDown 路径拦截）
│   ├── smoke_markitdown_e2e.py # MarkItDown MCP 端到端验证测试
│   └── smoke_skills.py       # Skills 规范与 Backend 路由冒烟测试
├── storage/                  # 运行时持久化目录（Milvus DB、浏览器配置等，已忽略）
├── workspace/                # FileAgent 操作工作区（已忽略）
├── bot.py                    # NoneBot2 QQ 机器人启动入口
├── main.py                   # 本地控制台多轮对话入口
├── config.py                 # 全局配置中心
├── mcp_servers.json          # MCP 服务配置
├── pyproject.toml            # 依赖与项目元数据 (uv)
└── .env.example              # 环境变量配置模板
```

---

## 4. 环境准备

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

### 4.3 基础服务组件与启动
- **Docker Compose 一键启动**:
  从项目根目录执行以下命令启动后台依赖容器（Neo4j / Milvus / MinIO / Etcd）：
  ```bash
  docker compose --env-file .env -f scripts/docker-compose.yml up -d
  ```
- **各组件用途说明**:
  - **Docker**: 供 `CodeAgent` 启动轻量 Python 沙箱容器 (`python-ds:latest`) 安全运行任意代码。
  - **Neo4j**: 提供知识图谱存储与 Cypher 关系查询，供 `MediaAgent` 使用。
  - **MySQL**: 存储女优与作品结构化关系数据，供 `SQLAgent` 精确筛选查询。
  - **Milvus (Milvus-Lite)**: 本地单机向量数据库，用于知识库检索 (`rag`) 以及用户跨会话长期记忆存储。

---

## 5. 运行方式

### 5.1 启动 QQ 机器人 (生产入口)
```bash
uv run python bot.py
```

### 5.2 控制台版本 (无需 QQ)
```bash
uv run python main.py
```

### 5.3 命令行脱离 QQ 测试终端 (Chat CLI)
```bash
uv run python scripts/chat_cli.py
```

---

## 6. HITL 人工授权说明

当 `FileAgent` 或 `DesktopAgent` 尝试执行高危操作（如写文件、编辑文件、移动文件、执行高危系统按键）时，系统会自动挂起并弹出授权请求：
```text
⚠️ 敏感操作授权拦截
──────────────────
操作: [write_file]
目标: test.txt
──────────────────
回复 Y 允许，回复 N 拒绝。
```
- **回复 `Y` / `y`**：系统恢复执行并完成敏感工具调用。
- **回复 `N` / `n`**：阻断该操作，向专家注入明确拒绝信息并由其礼貌告知用户。
- **多个并行授权**：当主 Agent 在同一步派发多个涉及敏感操作的专家时，系统会统一收集全部中断请求，用户的单次 `Y`/`N` 回复将同时作用于以上全部操作。

---

## 7. 工具与测试脚本说明

- `scripts/chat_cli.py`：命令行多轮对话终端，脱离 QQ 测试 Agent 交互与 HITL 授权流程。
- `scripts/manage_memory.py`：管理 Milvus 长期记忆库，支持 `list` 查看与 `delete <id>` 删除。
- `scripts/migrate_to_mysql.py`：从 `data/graphrag/*.jsonl` 导入演员与作品结构化数据至 MySQL。
- `scripts/login_setup.py`：浏览器自动化持久化登录引导脚本，支持传入一个或多个目标 URL（默认 B站），在弹出的有头浏览器中完成登录并保存 Cookie/Profile。
- `tests/run_smoke.py`：一键运行全部离线冒烟测试（FileAgent / Skills / 高德契约 / 高德 HTTP / chat_service）。
- `tests/smoke_chat_service.py`：chat_service 服务层离线冒烟测试（多轮 HITL、发件箱图片连续性、并发忙提示、并行中断、超时恢复与 `/reset`）。

---

## 8. 已知限制

1. **DesktopAgent 不稳定**: 桌面操作依赖网格视觉识别，受分辨率和缩放比例影响较大，可能存在点击漂移。
2. **MemorySaver 会话重启丢失**: 当前状态保存器为内存级 `MemorySaver`，机器人进程重启后进行中的上下文与未决 HITL 中断会丢失。长期事实依赖 Milvus 后台抽取持久化。
3. **FileAgent 工具调用步数限制**: 单次 FileAgent 运行串行工具调用上限约 19 次（带 HITL 中间件每次工具调用占 3 个 superstep）；`write_file` 对已存在文件会整体覆盖。
