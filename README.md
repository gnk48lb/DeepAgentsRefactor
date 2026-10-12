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
1. **KnowledgeAgent**: 知识检索专家，优先检索本地 Milvus 知识库 (`rag`)，无果且具时效性时通过 Tavily 全网搜索 (`web_search`)。
2. **MediaAgent**: 专有图谱检索专家，基于 Neo4j 知识图谱 (`av_graph_rag`) 进行实体与关系网络查询。
3. **MapAgent**: 地理与出行专家，挂载高德地图 MCP (`amap`)，支持天气查询、周边检索、路径规划与地理编码。
4. **CodeAgent**: 代码执行专家，通过 Docker 隔离沙箱安全运行 Python 脚本，支持数据分析与 Matplotlib 绘图。
5. **SQLAgent**: 关系数据库专家，负责将自然语言转为 SQL 并查询本地 MySQL (`execute_sql`)。
6. **BrowserAgent**: 浏览器自动化专家，基于 Playwright/browser-use 操控真实或无头浏览器，处理网页端交互。
7. **FileAgent (HITL)**: 本地文件系统专家，挂载 Filesystem MCP，受项目工作区边界保护。
8. **DesktopAgent (HITL)**: 桌面操作专家，基于网格坐标截屏、模拟点击与键鼠交互。

### 2.3 特权版 vs 受限版
系统根据 `config.OWNER_QQ_IDS` 自动为不同权限的用户构建专属 Agent 实例：
- **特权版 (Owner)**: 挂载全量 8 个子专家（含具有本地系统/文件写操作的 `FileAgent` 和系统桌面控制的 `DesktopAgent`）。
- **受限版 (Normal)**: 移除了 `FileAgent` 和 `DesktopAgent`，普通用户无法触碰服务器文件系统或物理桌面。

---

## 3. 目录结构

```text
DeepAgentsRefactor/
├── app/                      # 核心业务逻辑
│   ├── chat_service.py       # 统一对话服务层（处理锁、并发、HITL与发件箱）
│   ├── deep_agent.py         # 主 Agent 与声明式专家装配中心
│   ├── graph.py              # FileAgent 与 DesktopAgent 的 HITL 子图定义与提示词
│   ├── models.py             # LLM / VLM / 向量模型与重排器单例定义
│   ├── tools.py              # 本地工具集合（RAG, Docker Code, SQL 等）
│   ├── mcp_service.py        # MCP 协议客户端管理 (Amap, Filesystem, Desktop)
│   ├── database.py           # Milvus 向量库与长期记忆管理
│   ├── browser_service.py    # 浏览器自动化控制封装
│   ├── outbox.py             # 图片跨轮发件箱旁路
│   └── utils.py              # 通用辅助工具 (extract_text)
├── plugins/                  # NoneBot2 插件层
│   └── chat.py               # QQ 消息接收与 ChatService 薄封装
├── scripts/                  # 运维与常用测试脚本
│   ├── chat_cli.py           # 命令行多轮对话与 HITL 测试终端
│   ├── manage_memory.py      # Milvus 长期记忆查看与清理工具
│   ├── migrate_to_mysql.py   # 数据迁移至 MySQL
│   ├── test_hitl_replay_safety.py  # HITL 重放幂等性与节点重复执行单测
│   ├── hitl_replay_repro.py  # 中断恢复回归脚本
│   ├── test_http_image.py    # 图片处理测试
│   └── archive/              # 历史一次性验证脚本归档（不保证能运行）
├── docs/migration/           # 架构迁移历史文档
├── storage/                  # 运行时持久化目录（Milvus DB、浏览器配置等，已忽略）
├── workspace/                # FileAgent 操作工作区（已忽略）
├── bot.py                    # NoneBot2 QQ 机器人启动入口
├── main.py                   # 本地控制台多轮对话入口
├── config.py                 # 全局配置中心
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
- `GEMINI_API_KEY`（或 `GOOGLE_API_KEY`）：必填，Gemini 3.5 模型 API 密钥。
- `AMAP_API_KEY`：高德地图 Web 服务 API Key。
- `TAVILY_API_KEY`：Tavily 搜索 API Key。
- `SILICON_FLOW_API_KEY`：硅基流动 Embedding / Reranker API Key。
- `HUGGINGFACEHUB_API_TOKEN`：Hugging Face 访问令牌。

### 4.3 基础服务组件用途
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
# 以特权用户身份测试
uv run python scripts/chat_cli.py --user 111
```

---

## 6. OWNER_QQ_IDS 设置说明

- QQ 官方机器人体系下给出的用户标识是 **openid**（一串字符哈希），而非传统的 5~10 位纯数字 QQ 号。
- **单聊**与**不同群聊**中，同一个真实用户对应的 openid 可能各不相同。
- **设置方法**：
  1. 启动机器人并先在私聊或群聊中发送任意测试消息。
  2. 观察控制台输出的日志：`[chat] 收到来自用户: <user_id> 的消息`。
  3. 将该 `<user_id>` 填入 `.env` 中的 `OWNER_QQ_IDS`（多个 ID 用半角逗号分隔），例如：
     ```bash
     OWNER_QQ_IDS="ABCD1234EFGH,IJKL5678MNOP"
     ```
  4. 重启机器人即可生效。

---

## 7. HITL 人工授权说明

在特权模式下，当 `FileAgent` 或 `DesktopAgent` 尝试执行高危操作（如写文件、重命名文件、创建目录、执行高危系统按键）时，系统会自动挂起并弹出授权请求：
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

---

## 8. 测试脚本说明

- `scripts/test_hitl_replay_safety.py`：单测套件，验证 LangGraph 在单步 HITL 中断恢复时不会重复执行已批准的安全/危险工具调用。
- `scripts/hitl_replay_repro.py`：微型重现脚本，用于复现与比对 LangGraph 中断恢复节点的执行次数。
- `scripts/manage_memory.py`：管理 Milvus 长期记忆库，支持 `list` 查看与 `delete <id>` 删除。

---

## 9. 已知限制

1. **DesktopAgent 不稳定**: 桌面操作依赖网格视觉识别，受分辨率和缩放比例影响较大，可能存在点击漂移。
2. **受限版专家权限**: 当前受限版仅去除了 `FileAgent` 和 `DesktopAgent`，仍然挂载了 `MediaAgent`、`SQLAgent` 与 `CodeAgent`。
3. **MemorySaver 会话重启丢失**: 当前状态保存器为内存级 `MemorySaver`，机器人进程重启后进行中的上下文与未决 HITL 中断会丢失。长期事实依赖 Milvus 后台抽取持久化。
