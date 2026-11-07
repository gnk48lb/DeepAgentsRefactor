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
L5 Services  (app/services/startup.py, chat_service.py, session_log.py)
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
1. **KnowledgeAgent**: 知识检索专家，优先检索本地 Milvus 知识库 (`rag`)，无果且具时效性时通过 Tavily 全网搜索 (`web_search`)；时间类问题由主 Agent 通过自己挂载的 get_current_time 就地回答，KnowledgeAgent 不挂载该工具。
2. **MediaAgent**: 成人影视专家，同时挂载 Neo4j 知识图谱检索 (`av_graph_rag`) 与 MySQL 只读查询 (`execute_sql`)：图谱负责实体 / 合作关系网络与题材标签，数据库负责身高、罩杯、出生年份等精确属性过滤与发行年份统计，复合问题由它自己组合完成。
3. **MapAgent**: 地理与出行专家，挂载高德地图本地工具（共 12 个高德本地 Tool），支持天气查询、周边检索、路径规划与地理编码。约定：路线、周边搜索、测距的位置参数既可传文字地址也可传'经度,纬度'，工具内部自动解析（文字地址跨城有歧义时会报错要求补城市）；只有 maps_regeocode 必须传坐标。
4. **CodeAgent**: 代码执行与数据分析沙箱工作台。基于 deepagents 的 `CompiledSubAgent` 模式，挂载 `FilesystemMiddleware`（提供 `ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`、`execute` 7 个沙箱工具，故意不含 `delete` 工具）与快捷通道 `execute_python_code`。通过 `/workspace/...` 只读访问用户工作区，沙箱专属可读写会话工作目录为 `/work/...`（脚本、数据与多文件处理），专属交付箱为 `/work/outputs/...`（生成的图表由 `SandboxOutputsMiddleware` 统一按指纹提取并自动推送到模型与用户发件箱）。绘图规范由 Skill `matplotlib-chart` 提供，数据分析规范由 Skill `data-analysis` 提供。沙箱预置中文字体、热启动支持与流式输出截断。
5. **BrowserAgent**: 浏览器自动化专家，基于 Playwright/browser-use 操控真实浏览器（可见模式），处理网页端交互。
6. **FileAgent (HITL)**: 本地文件专家，内置 FilesystemMiddleware + FilesystemBackend + 原生 HITL。提供工具：`ls`/`glob`/`grep`/`read_file` 免授权，`convert_to_markdown`（由 markitdown-mcp 提供，支持富文档转 Markdown），`write_file`/`edit_file`/`create_directory`/`move_file` 需授权，删除 = 移动到 archive_trash 归档；另有自定义工具 `run_host_command`（在用户 Windows 宿主机上执行命令，仅少数只读状态查询如 `git status` 免授权，其余需授权，见 2.10）。文件工具只能在 workspace 工作区内操作；`run_host_command` 不受路径限制（仅以工作区根目录作为 cwd）。转换结果过大时会暂存到 `/_artifacts/large_tool_results/<id>`，FileAgent 用 `read_file` 分页读回。
7. **DesktopAgent (HITL)**: 桌面操作专家，基于网格坐标截屏、模拟点击与键鼠交互。

### 2.4 单用户模式
系统为单用户专属模式，仅供本人使用：
- 所有专家（含具有本地文件读写权限的 `FileAgent`、物理桌面控制的 `DesktopAgent` 以及自动化操作网页的 `BrowserAgent` 等全量 7 个专家）始终挂载。
- 系统内部没有任何用户身份或权限校验逻辑。
- 对话按会话（session）组织：`thread_id` 等于会话 id，会话游标、事件日志与挂起的授权都持久化在 `storage/` 下（详见 2.11），进程重启后可恢复；同一个聊天窗口里可以开多个会话并切换。
- 斜杠命令（在授权判断之前拦截，按首个空白分隔的词精确匹配；其余以 `/` 开头的文本，如路径，按普通消息处理）：
  - `/new [标题]`：新建并切换到新会话，旧会话原样保留（含挂起的授权）。
  - `/sessions [N]`：列出最近 N 个会话（默认 10，最多 30），标出 `[当前]` 与 `[待授权]`。
  - `/switch <序号|id前缀>`：序号按 `/sessions` 的顺序；切到有待授权操作的会话时会附上授权内容。
  - `/reset`：只清空当前会话的对话上下文（含挂起的授权）并重置代码沙箱，**事件日志保留**。沙箱仍是进程级全局，`/new`、`/switch` 不会动它。

> **警告**：机器人必须保持私有；若将来对外开放，需要重新引入用户隔离与权限层。尤其是 `FileAgent` 的 `run_host_command` 会以机器人进程所属用户的权限在宿主机上执行命令，任何能给机器人发消息并回复 `Y` 的人都等同于拥有这台机器的该用户权限。

### 2.5 MCP 常驻会话与容错机制
系统通过 Model Context Protocol (MCP) 集成外部能力（如文档转换 `markitdown`、桌面控制 `desktop`）。
各 MCP 服务进程在机器人启动时由专属后台 Runner 统一拉起并常驻运行，退出时自动回收，大幅降低工具调用开销。
运行时支持并发调用与调用超时熔断（默认 60s）；若检测到服务异常断开或进程挂死，将自动触发重连与会话重建。

### 2.6 长期记忆 (两阶段召回对账与版本化)
系统基于 Milvus 向量库维护跨会话长期记忆，采用两阶段（提取 → 召回对账 → 版本化）架构：
- **输入隔离**：仅信任当轮用户原话输入，绝不向提取器灌入 AI 回复、工具结果或系统 `<long_term_memory>` 块；图片轮（输入为 VLM 翻译指令）及超过 `MEMORY_EXTRACT_MAX_USER_CHARS`（800 字符，疑似粘贴文本）均不提取。跨 HITL 人工授权时保持用户最初原话，不记录单字符 "y"。
- **两阶段写入**：
  - **阶段 1（原子陈述提取）**：每轮 1 次 LLM 调用，严格冷酷提取用户关于自己的原子事实断言（`facts`）或撤回请求（`retractions`），无事实则输出空列表。
  - **阶段 2（语义召回与对账）**：仅在有提取陈述且库中存在候选时触发。先按向量相似度召回 top-K（`MEMORY_RECALL_TOP_K=5`，不设分数下限）合并候选记忆并赋予内部别名（S1.., M1..），交给 LLM 仲裁判别操作类型（`ADD` / `UPDATE` / `DELETE` / `NONE`）。
- **版本化与软删除**：
  - **UPDATE**：新插入一行（`status="active"`, `supersedes=旧id`），并将旧行状态置为 `superseded`，保留版本演进链。
  - **DELETE**：将目标行状态置为 `deleted`，并记录撤回原话（`retired_evidence`）。
  - **检索与注入**：仅针对 `status == "active" and kind == "fact"` 进行检索，通过 `MEMORY_MIN_SCORE` 阈值过滤，最多注入 `MEMORY_TOP_K` 条。旧值在运行时永不物理删除。
- **代码护栏 (Fail-Closed)**：
  - **证据溯源**：每条陈述必须包含逐字摘自用户原话的 `evidence`（NFKC + 空白折叠 + casefold 子串匹配且长度 ≥ 2）。
  - **文本合规**：字符数 ≤ `MEMORY_TEXT_MAX_CHARS`（200）、无换行/制表符、无 `<` 或 `>`、不含 `long_term_memory`。
  - **别名与操作校验**：别名非法、重复占用同一 target、阶段 2 漏报陈述均安全丢弃；阶段 2 严禁产出陈述外的新文本。
  - **幂等去重**：`ADD` 写入前做规范化文本完全相同检查，相同则自动跳过。
- **并发与持久化保证**：
  - 单调客户端 INT64 主键（毫秒级微秒时间戳），避免 Milvus 服务端 autoID upsert 换主键问题。
  - 全写操作在单进程 `threading.RLock` 内执行 check-and-set，向量计算在锁外完成。
- **数据迁移与旧集合备份**：
  - 新版本集合命名为 `user_memory_v2`（基于 `MEMORY_SCHEMA_VERSION=2`），全显式声明标量字段，关闭 dynamic fields 避免 Arrow KeyError。
  - 启动时自动从旧集合 `user_memory` 进行幂等、非破坏性平滑迁移（复用向量，不重复调用 embedding；旧集合永久保留作为备份）。
  - 迁移成功后在 `storage/memory_v2_migrated.flag` 写入标记文件，标记存在时永不重复迁移；`CLEAR_MEMORY_ON_STARTUP=True` 清空新库并写标记，避免旧数据复活；`REFRESH_COLLECTION` 不影响记忆库。

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
- **预算配置与调优**：未登记专家默认 10 步；`FileAgent` 拥有 20 步预算（涉及富文档转换、分页读取与复杂目录操作）；`CodeAgent` 在升级为沙箱工作台后预算提升至 15 步（支持“写文件 → 运行 → 读结果 → 微调代码 → 再次运行”等多步自动化流水线）。预算是“止损阈值”而不是常规用量，日志里出现 `Tool call limit exceeded` / `Model call limits exceeded` 说明某个专家触顶，可按真实用量微调 `SUBAGENT_TOOL_LIMITS`。

### 2.9 代码沙箱设计取舍与演进路径
系统在代码沙箱的设计上遵循严格的权责分离、纵深防御与平滑演进原则：
- **从 v1 单一执行器到 Phase 2 沙箱工作台**：
  - **v1 设计**：初期坚持单一 `execute_python_code` 快速通道，保证代码执行轻量、低认知负荷与零文件副作用。
  - **Phase 2 演进需求**：随着复杂数据分析、批量文本处理与长链任务增多，模型需要编写多文件模块、局部微调脚本（`edit_file`）、运行 Linux 命令行工具（`rg` / `jq` / `unzip` / `file` / `tree` 等）以及排查环境状态，单一 Python 字符串执行已无法满足需求。
- **权责边界：CodeAgent 与 FileAgent 的本质差异**：
  - **`FileAgent` (唯一工程改写者)**：系统中**唯一**能够读写、修改与删除宿主本地磁盘工程（用户真实 `workspace`）的专家。具有本地文件系统副作用，因此破坏性写操作必须强受 HITL 人工授权约束。
  - **`CodeAgent` (隔离沙箱工作台)**：在隔离 Docker 容器中运行，其内部的 `/workspace` 通过只读镜像（`bind: "/workspace", mode: "ro"`）挂载，**严禁向用户工作区写入任何内容**；所有写、改、执行操作仅限于沙箱专属内存工作区 `/work`（256MB tmpfs），超时或重置即清空，绝无可能损坏或污染宿主项目，因此沙箱内文件操作无需弹 HITL 授权。
- **架构装配与真实签名**：
  - `DockerSandbox` 继承并完整实现 `deepagents.backends.sandbox.BaseSandbox` 抽象基类契约（包含 `id`、`execute`、`upload_files`、`download_files`、`ls`、`read`、`write`、`edit`、`glob`、`grep` 等核心协议）。
  - 在装配层通过 `CompositeBackend` 进行路由编排，真实签名严格对齐 deepagents 官方规范：
    `CompositeBackend(default=docker_sandbox, routes={SKILLS_ROUTE: FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True), "/_artifacts/": StateBackend()}, artifacts_root="/_artifacts")`
  - 既让 CodeAgent 拥有了沙箱文件系统与 Linux 命令执行能力，又让只读技能目录 `/skills/` 走宿主虚拟路径，两不冲突。
  - 工具白名单显式剔除 `delete` 工具（`SANDBOX_FS_TOOLS = ["ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"]`），防止模型在沙箱内随意破坏未完成任务的文件。
- **交互语义与独立 subshell**：
  - 每次 `execute` 命令均由独立 subshell 运行（通过 `["timeout", "-k", "3", str(effective_timeout), "bash", "-c", command]` 启动），环境变量（`export`）与工作目录（`cd`）不跨调用持久化；多条依赖命令应使用 `&&` 显式串联。默认超时 60s，可通过参数扩展至 300s。
- **产物统一投递管线**：
  - 由 `SandboxOutputsMiddleware` 拦截 `execute` 调用，调用底层 `collect_new_outputs()` 按 `(路径: (大小, mtime))` 文件指纹扫描 `/work/outputs/`。
  - 生成的新图表通过 `publish_images()` 实现双份投递：一份推送到接入层图片发件箱（Outbox）直发给用户，一份压缩编码为低分辨率预览 block 反哺给模型进行图表自检；同一指纹在会话生命周期内不重复投递。

### 2.10 宿主机命令与沙箱的分工
`FileAgent` 通过自定义工具 `run_host_command` 在用户的 Windows 宿主机上执行命令（`cmd.exe /d /s /c`），`CodeAgent` 仍在 Docker 沙箱里执行。主管按“要不要联网 / 要不要碰用户这台电脑 / 要不要真正改 workspace”选择：

| 场景 | 联网 | 碰宿主机 | 改 workspace | 派给 |
| --- | --- | --- | --- | --- |
| 计算、统计、绘图、数据分析、验证纯代码 | 否 | 否 | 否 | CodeAgent（无需授权） |
| git、pip/npm 安装、下载、构建/测试脚本、运行 workspace 里的程序、查看本机进程/端口/环境 | 可能 | 是 | 可能 | FileAgent（绝大多数命令需授权） |
| 改代码 → 运行 → 看报错 → 再改 | - | 是 | 是 | FileAgent（一次委派内闭环） |

- **为什么不用 `LocalShellBackend`**：它用 `subprocess.run(shell=True, text=True)`，Windows 上超时只杀 `cmd.exe`、孙进程占着管道会让调用一直挂住；标准输入继承自机器人进程；按本地代码页解码；输出在内存里无上限。因此运行器自己写在 `app/infra/host_shell.py`：`stdin=DEVNULL`、stdout/stderr 合并、超时与输出洪泛时终止整个进程树（Windows 用 `taskkill /T`）、读线程 + 16MB 内存硬上限、先严格按 UTF-8 解码再回退 OEM 代码页、按“头 60% + 尾 40%”截断。工具不叫 `execute`，是为了与 CodeAgent 的 `execute` 区分。
- **免授权 = 整条命令精确匹配**：整条命令（空白折叠、忽略大小写）必须与 `config.HOST_SHELL_AUTO_APPROVE_COMMANDS` 中某一项完全一致才免授权，不解析 cmd 语法、不做前缀匹配；其余一律人工授权。命令长度上限 `HOST_SHELL_MAX_COMMAND_CHARS`（默认 2000），低于 `QQ_MAX_REPLY_CHARS`，保证被批准的命令一定被完整展示过。
- **环境变量白名单**：子进程只继承系统运行必需项、代理和常见工具链目录，绝不整体继承 `os.environ`（`.env` 中的 API Key、`QQ_BOTS` 等都不会传入）。需要额外放行时设置 `HOST_SHELL_ENV_ALLOW=JAVA_HOME,GOPATH`。
- **审计日志**：每条实际执行的命令追加一行到 `storage/host_shell_audit.log`（时间、退出码、耗时、命令前 500 字符）。
- **加免授权命令**：在 `app/config.py` 的 `HOST_SHELL_AUTO_APPROVE_COMMANDS` 里按“整条命令”添加；含 `& | < > ^ % ! ; ` $ ( )` 和引号的条目会被运行时忽略。只应加无副作用的只读查询。

### 2.11 会话与事件日志
设计依据是 Anthropic《Scaling Managed Agents》：**会话（session）不是上下文窗口**。会话是持久化的事件日志，模型上下文只是其中一部分的一次读取；Harness（本进程）无状态，跨轮、跨授权的状态只放在存储里。

**文章接口 ↔ 本项目**

| 文章 | 本项目 |
| --- | --- |
| `getSession` | `SessionStore.get_session` |
| `getEvents` | `get_events` / `get_last_events` / `get_pending_events` |
| `emitEvent` | `SessionStore.emit_event`（带 `key` 时幂等） |
| `wake(session_id)`（无载荷） | `chat_service.wake`：载荷全部从事件日志里读 |
| Sandbox / Tools | 已满足，未改动；沙箱仍是进程级全局，不按会话隔离 |

**两个库文件（都在被 git 忽略的 `storage/`，路径可由 `SESSIONS_DB_PATH` / `CHECKPOINT_DB_PATH` 覆盖）**
- `sessions.db`（标准库 `sqlite3`，WAL）：`sessions`（含处理游标 `processed_seq`）、`events`（只追加的事件日志）、`turn_state`（授权挂起期间暂存的用户原话与已生成的图片）、`meta`（活动会话指针）。
- `checkpoints.db`（官方 `AsyncSqliteSaver`）：LangGraph checkpoint，负责“续跑”（模型上下文与 HITL 中断）。`thread_id == session_id`。
- 两者分工：checkpoint 负责续跑，事件日志负责记录与回溯；不要试图从事件日志重建模型上下文。

**事件类型**：`session_created`、`user_message`（原话 / 翻译后文本 / 是否带图）、`approval`、`approval_request`、`assistant_message`（文本 + 图片张数）、`expert_report`（task 返回的专家完整战报）、`tool_call`、`tool_result`、`turn_failed`、`turn_abandoned`、`session_reset`。事件只追加、不改写；“待处理”由会话游标 `processed_seq` 表达（入站事件 `user_message` / `approval` 的 seq 大于游标即待处理）。图片字节永远不进事件日志，只记张数。

**wake 与重启恢复语义**
- wake 只做按需触发（用户消息 / Y-N 审批），不做定时、cron、后台主动唤醒，也不做崩溃自动重试。
- 收到消息：写入 `user_message` 事件 → `wake(会话)` → 读取未处理的入站事件逐条处理 → 推进游标（成功失败都推进）。
- 挂起授权：状态在 checkpoint 里，原话与已生成的图片落在 `turn_state`。进程重启后回复 Y/N 照常恢复，已完成的专家不会重跑，挂起前生成的图片仍随最终回复发出。
- 崩溃遗留：启动时对每个有未处理入站事件的会话先镜像已落盘的消息，再追加 `turn_abandoned{process_restart}` 并推进游标，**不自动重跑**（回复通道已断，且重放可能让已批准的宿主机命令再执行一次）。
- 主 Agent 的回溯：`search_session_history` / `get_session_events` 两个只读工具直接挂在主 Agent 上，查的是事件日志（用户原话 + 专家完整汇报），不受上下文压缩影响。

**`/reset` 与脚本删除的区别**
- `/reset`：只清上下文（删 checkpoint 线程 + 挂起态 + 重置沙箱），事件日志保留并追加 `session_reset`。
- `uv run python scripts/manage_sessions.py delete <id前缀> [--yes]`：彻底删除（事件日志 + 挂起态 + checkpoint 线程），不可恢复。另有 `list`、`show <id前缀> [--from N --limit N --types a,b]`。

**嵌套 checkpoint 清理**：DesktopAgent 截图的 base64 会让子专家的嵌套 checkpoint 迅速膨胀，而一轮正常结束后它们不会再被读取。每轮**正常结束**时清理 `checkpoint_ns != ''` 的行（`CHECKPOINT_PRUNE_NESTED`，默认开启）；失败 / 超时 / 挂起时不清理，根命名空间永不清理，也不调用 `aprune()`。

**隐私提示**：`sessions.db` 含完整对话文本（用户原话与专家汇报），`checkpoints.db` 含完整模型上下文，都位于被 git 忽略的 `storage/`；备份、分享前请留意。

**严格 msgpack**：checkpointer 创建时显式传入 `allowed_msgpack_modules=None`（等价于 `LANGGRAPH_STRICT_MSGPACK=true`），checkpoint 反序列化只允许 LangGraph 内置的安全类型；重启恢复测试覆盖了本项目用到的状态类型。

---

## 3. 目录结构

```text
DeepAgentsRefactor/
├── .github/                  # GitHub Actions CI 工作流
│   └── workflows/
│       └── ci.yml
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
│   │   ├── host_shell.py     # 宿主机命令运行器 / 环境变量白名单 / 免授权判定（FileAgent run_host_command）
│   │   ├── session_store.py  # 会话 / 事件日志 / 挂起态 / 活动会话指针（标准库 sqlite3，sessions.db）
│   │   ├── checkpointer.py   # LangGraph checkpoint 的 AsyncSqliteSaver 打开 / 关闭 / 嵌套命名空间清理（checkpoints.db）
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
│   │   ├── store.py          # Milvus 记忆集合初始化、版本化维护与语义检索
│   │   ├── injector.py       # 对话初次消息记忆注入与过滤
│   │   ├── extractor.py      # 后台两阶段记忆抽取、对账与异步持久化
│   │   └── guards.py         # 记忆纯函数代码护栏（证据校验/文本合规/对账规则）
│   ├── tools/                # L3: 工具集合与适配器
│   │   ├── knowledge_tools.py# RAG / Tavily / 时间查询工具
│   │   ├── session_tools.py  # 主 Agent 的会话记录回溯工具（get_session_events / search_session_history，只读）
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
│       ├── chat_service.py   # 单用户多模态会话状态机、斜杠命令与 wake 调度服务
│       └── session_log.py    # 事件日志镜像（mirror_messages）与命令输出格式化
├── deploy/                   # 部署配置
│   ├── docker-compose.yml    # Neo4j 等后台服务容器编排
│   └── sandbox/              # Docker 沙箱镜像定义
│       └── Dockerfile
├── mcp_servers/              # 独立 MCP 服务实现
│   └── windows_desktop.py
├── plugins/                  # NoneBot2 插件层
│   └── chat.py               # QQ 机器人消息接收与 ChatService 调度
├── scripts/                  # 运维与工具脚本
│   ├── login_setup.py        # 浏览器自动化持久化登录引导脚本
│   ├── manage_memory.py      # Milvus 长期记忆管理工具 (list/delete/restore)
│   ├── manage_sessions.py    # 会话管理工具 (list/show/delete，彻底删除会话与其 checkpoint)
│   └── migrate_to_mysql.py   # 数据迁移至 MySQL
├── skills/                   # deepagents 官方 Skills 目录
│   └── matplotlib-chart/     # Matplotlib 绘图规范技能
├── tests/                    # 测试用例
│   ├── fixtures/amap/        # 高德工具契约与 Golden 样本数据
│   ├── live_amap_golden.py   # 高德在线 Golden 数据集比对测试
│   ├── run_smoke.py          # 一键运行全部离线冒烟测试
│   ├── smoke_amap_contract.py# 高德工具快照与契约冒烟测试
│   ├── smoke_amap_http.py    # 高德工具 HTTP 异常与密钥脱敏测试
│   ├── smoke_architecture.py # 架构分层、规范与契约冒烟测试
│   ├── smoke_budget.py       # 专家工具调用预算（独立停止条件）冒烟测试
│   ├── smoke_chat_service.py # chat_service 服务层离线冒烟测试
│   ├── smoke_file_agent.py   # FileAgent 离线冒烟测试
│   ├── smoke_host_shell.py   # 宿主机命令运行器离线冒烟测试
│   ├── smoke_knowledge.py    # 知识库离线冒烟测试
│   ├── smoke_markitdown_e2e.py # MarkItDown 端到端离线冒烟测试
│   ├── smoke_memory.py       # 长期记忆两阶段召回对账与版本化离线冒烟测试
│   ├── smoke_orchestrator.py # 主管编排与专家装配冒烟测试
│   ├── smoke_session.py      # 会话存储 / 事件日志镜像 / 回溯工具离线冒烟测试
│   ├── smoke_skills.py       # Skills 规范与 Backend 路由冒烟测试
│   └── smoke_units.py        # 纯函数单元冒烟测试
├── storage/                  # 运行时持久化目录（Milvus DB、sessions.db、checkpoints.db、浏览器配置等，已忽略）
├── workspace/                # FileAgent 操作工作区（已忽略）
├── bot.py                    # NoneBot2 QQ 机器人启动入口
├── main.py                   # 本地控制台多轮对话入口 (合并了原 CLI 功能)
├── mcp_servers.json          # MCP 服务配置文件
├── pyproject.toml            # 依赖与项目元数据 (uv)
└── requirements.txt          # 导出依赖列表 (pip 兼容)
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
- `GEMINI_API_KEY`（或 `GOOGLE_API_KEY`）：必填，轻量工具调用、聊天图片意图翻译与记忆提取仍使用 Gemini。
- `AMAP_API_KEY`：高德地图 Web 服务 API Key。
- `TAVILY_API_KEY`：Tavily 搜索 API Key。
- `SILICON_FLOW_API_KEY`：硅基流动 Embedding / Reranker API Key。
- `HUGGINGFACEHUB_API_TOKEN`：Hugging Face 访问令牌。
- `HOST_SHELL_*`（可选）：FileAgent 宿主机命令的超时、输出上限、命令长度上限与额外放行的环境变量名，见 2.10 与 `.env.example`。
- `SESSIONS_DB_PATH` / `CHECKPOINT_DB_PATH` / `CHECKPOINT_PRUNE_NESTED`（可选）：会话库、checkpoint 库路径与嵌套 checkpoint 清理开关，默认 `storage/sessions.db`、`storage/checkpoints.db`、开启，见 2.11。

#### LLM 角色与 Magpie 网关

较重的 agentic 角色默认通过 **Anthropic Messages 协议**访问本机 Magpie；轻量、低延迟或当前已稳定的结构化调用保留 Gemini。Magpie 的 base URL 是 `http://127.0.0.1:3425`（不要在后面追加 `/v1`），本地网关默认使用占位 key `local`。

| 角色配置前缀 | 默认 provider | 用途 |
|---|---|---|
| `ORCHESTRATOR_LLM` | Magpie | 主管规划、委派与综合 |
| `WORKER_LLM` | Magpie | Media / Map / 外层 Browser / Code / File |
| `KNOWLEDGE_LLM` | Magpie | KnowledgeAgent，RAG 结果可能含图 |
| `INGESTION_VLM` | Magpie | 知识库图片摘要（批量） |
| `BROWSER_VLM` | Magpie | 嵌套 browser_use 视觉浏览器自动化 |
| `DESKTOP_VLM` | Magpie | DesktopAgent 视觉桌面控制 |
| `UTILITY_LLM` | 固定 Gemini | 媒体意图解析与查询改写 |
| `VLM` | 固定 Gemini | 聊天图片到文本的低延迟意图翻译 |
| `EXTRACTOR_LLM` | 固定 Gemini | 后台记忆结构化提取 |

公共网关配置：`MAGPIE_BASE_URL`、`MAGPIE_API_KEY`、`MAGPIE_MODEL`、`MAGPIE_MAX_TOKENS`、`MAGPIE_TIMEOUT_SECONDS`、`MAGPIE_MAX_RETRIES`。默认模型为 `minimax-code-global/MiniMax-M3.1-Flash`，每个迁移角色可用 `<ROLE>_MODEL` 单独覆盖，模型 ID 直接取 Magpie `/v1/models` 返回的值，不加 `anthropic:` 前缀。

每个迁移角色都可独立回退。例如将 `WORKER_LLM_PROVIDER=gemini` 写入 `.env` 并重启进程即可；若设置过 `WORKER_LLM_MODEL`，同时删除它或改为原 Gemini 型号。未显式覆盖模型时，provider 会自动选择对应的默认型号。回退后请开启新会话，不混用不同 provider 的历史推理块。

可替换为其他 Magpie 模型，但工具型 agent 必须支持工具调用；`WORKER_LLM`（图表预览）、`KNOWLEDGE_LLM`、`INGESTION_VLM`、`BROWSER_VLM`、`DESKTOP_VLM` 必须支持图片输入。思考模型需要足够的输出预算，避免整轮只有 thinking 而没有正文或工具调用。

当前应用进程运行在宿主机，Docker 只承载无网络代码沙箱，因此默认 loopback 地址即可。若以后把应用本身容器化，需通过 `MAGPIE_BASE_URL` 改为宿主机网关地址或服务名；明文 HTTP 只应用于 loopback/可信网络。

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
  - 镜像基于 `python:3.13-slim`，预置了完整的数据科学常用依赖（pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, openpyxl, xlrd, tabulate, pyyaml, sympy 等）与开箱即用的中文字体（Noto Sans CJK SC/JP），并预热了 matplotlib 字体缓存以实现亚秒级热启动。
  - 可通过 `docker build -t python-ds:latest deploy/sandbox` 构建镜像。
  - > **注意**：修改 `deploy/sandbox/Dockerfile`、`requirements.txt` 或 `matplotlibrc` 后，需重新执行镜像构建命令；若已有常驻沙箱容器正在运行，需重启机器人进程或等待空闲 TTL（`SANDBOX_IDLE_TTL_SECONDS=1800`）触发容器自动重建后方可生效。

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

### 5.4 知识库增量入库与 BM25 维护

知识库涉及两个目录：

- `data/source/`：已归档的资料（全量建库 / 重建时读取）。
- `data/pending/`：待处理的增量资料；每次启动时自动处理，成功后移动到 `data/source/`。

#### 普通增量（`REFRESH_BM25=False`，默认）

1. 把新文件放进 `data/pending/`，正常启动即可。
2. 程序会为图片/PDF 生成 VLM 摘要、切分文本，用**现有的** `storage/bm25_model.json` 编码稀疏向量，然后写入 Milvus 并把文件归档到 `data/source/`。
3. 普通增量**不会**重新拟合 BM25 模型：新卡片用旧模型编码，模型文件保持不变。检索仍然可用（查询和文档用同一个模型），只是新词 / 新语料的权重不是最优。

#### 刷新 BM25（`REFRESH_BM25=True`）

当新增内容较多、希望 BM25 统计覆盖全量语料时，按以下顺序操作：

1. 保持 `config.REFRESH_BM25 = False`，启动一次，让 `data/pending/` 处理完，并确认该目录为空。
2. 将 `config.REFRESH_BM25 = True`，再启动一次。程序会：
   - 用 `data/source/` + `data/pending/` 的全部文本重新拟合 BM25 并保存；
   - 用新模型重新编码 Milvus 中所有历史记录的 `sparse_vector` 并局部更新（`partial_update=True`，不会重新调用 Embedding API）。
3. 看到“已成功局部更新 N 条历史数据的稀疏向量字段”后，把 `REFRESH_BM25` 改回 `False`，避免每次启动重复执行。

> **不要**在 `data/pending/` 还有图片/PDF 时开启 `REFRESH_BM25=True`：刷新分支会先把这些文件的 VLM 摘要生成好，但只用于 BM25 统计、不写入 Milvus；随后的增量步骤会把这些摘要当成“已存在”而跳过，导致摘要块永久缺失。

如果只想彻底重建知识库，也可以直接：

```bash
uv run python main.py --rebuild
```

`--rebuild` 会删除 `knowledge` 集合（不影响媒体库和记忆库），从 `data/source/` 全量重新建库并重新拟合 BM25。代价是所有片段都会重新调用 Embedding API，比重刷 BM25 慢且费额度。

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
`run_host_command` 的授权提示会完整显示命令（不截断）：
```text
⚠️ 敏感操作授权拦截
──────────────────
操作: [run_host_command]
目标: git clone https://github.com/u/r.git  （timeout=120s）
──────────────────
回复 Y 允许，回复 N 拒绝。
```
- `write_file` / `edit_file` 的授权提示会显示内容预览（前 300 字符及总字符数），因为“先 `write_file` 写脚本、再 `run_host_command` 运行”是最直接的绕过路径。
- 授权提示里的 URL 不会像普通回复那样被替换为“（链接已省略）”，而是转义成 `https[:]//github[.]com/...` 的形式，保证你能看清自己在批准什么。
- **回复 `Y` / `y`**：系统恢复执行并完成敏感工具调用。
- **回复 `N` / `n`**：阻断该操作，向专家注入明确拒绝信息并由其礼貌告知用户。
- **重启后仍可恢复**：挂起的授权状态在 checkpoint 里，授权前已生成的图片和用户原话在 `turn_state` 里。授权提示出现后即使重启进程，回复 `Y` / `N` 仍照常放行，已完成的专家不会重跑（见 2.11）。
- **授权挂起时切换会话**：`/new`、`/switch` 不会丢弃挂起的授权；用 `/sessions` 可看到 `[待授权]` 标记，`/switch` 切回该会话后回复 `Y` / `N`。
- **发送 `/reset`**：可随时丢弃当前会话未决的授权与历史上下文（事件日志保留），重新开始新对话。

---

## 8. 测试体系

本项目包含完整的离线冒烟测试套件，在无 API Key、无 Docker、无网络连接的离线环境下即可完整验证：
```bash
# 一键运行全部离线冒烟测试
uv run python tests/run_smoke.py
```

测试包含以下 15 个离线测试套件：
1. `tests/smoke_architecture.py`：分层单向依赖、语法禁令、空 `__init__`、提示词单例、专家规格、轻量隔离导入、无 Key 安全导入与 MCP 路径校验（8 个场景）。
2. `tests/smoke_file_agent.py`：FileAgent 离线冒烟测试（含 MarkItDown 路径安全拦截、越界隔离、HITL 恢复映射以及 `run_host_command` 的工具白名单、免授权、批准/拒绝、绕过变体与授权提示内容等 23 个场景）。
3. `tests/smoke_skills.py`：Skills 规范与 Backend 虚拟路由冒烟测试（2 个场景）。
4. `tests/smoke_amap_contract.py`：高德地图 12 个本地工具契约与 Schema 快照测试（5 个场景）。
5. `tests/smoke_amap_http.py`：高德工具 HTTP 异常与密钥脱敏测试（13 个场景）。
6. `tests/smoke_chat_service.py`：以生产同款 `AsyncSqliteSaver` + `SessionStore` 驱动的服务层测试，覆盖多轮 HITL、发件箱图片连续性、并发忙提示、超时恢复、`/reset`、图片轮不提取记忆，以及重启恢复、多会话 `/new` `/sessions` `/switch`、崩溃遗留、事件日志镜像、嵌套 checkpoint 清理、`/reset` 保留日志与 `wake` 幂等消费（14 个场景）。
7. `tests/smoke_session.py`：`SessionStore` 建表 / 活动指针 / 标题 / 前缀解析、事件追加与幂等 key、处理游标、`turn_state`（含 ≥1MB 发件箱）、搜索、关闭重开、并发 emit、`delete_session`，以及 `session_log.mirror_messages` 与两个回溯工具（12 个场景）。
8. `tests/smoke_knowledge.py`：知识库建库、pending 增量入库、混合检索、长期记忆与 Reranker 阈值/降级离线冒烟测试（6 个场景）。
9. `tests/smoke_memory.py`：长期记忆两阶段召回对账、版本化演进、代码护栏、输入隔离、fail-closed、多线程并发与平滑迁移离线冒烟测试（9 个场景）。
10. `tests/smoke_orchestrator.py`：编排层主 Agent 装配、专家顺序、规格与 HarnessProfile 排除写工具安全测试（4 个场景，含主管时间 / 会话回溯工具与专家预算装配）。
11. `tests/smoke_units.py`：纯函数与小模块离线单元冒烟测试（路径解析、os.rename 兜底、MIME 判断、截图历史、未应答 tool_calls、SQL 加固与超时只读、rag 同源去重、浏览器幂等关闭、Milvus客户端双检锁、Neo4j图约束确保、图片编码与自检、Anthropic thinking 保留、浏览器 provider 选择与授权提示 URL 转义等 18 个场景）。
12. `tests/smoke_budget.py`：专家独立停止条件（工具调用预算）离线冒烟测试（声明式专家 / FileAgent / DesktopAgent 的软硬停止、计数器隔离、超预算写入不弹授权、超预算宿主机命令不弹授权、生产递归上限下失控保护等 19 个场景）。
13. `tests/smoke_sandbox.py`：Docker 沙箱架构契约与防护离线冒烟测试（含配置参数校验、单例跨调用复用与 TTL 超时重建、执行超时防护、输出洪泛头尾截断、多模态产物解析与安全防御、Docker 未启动友好引导、只读挂载降级与软链接防御、BaseSandbox 与 CompositeBackend 抽象基类契约、资源释放幂等性、并发串行化、code_tools 工具层与 Outbox 旁路集成、P0 命令包装与输出上限、O 超时细分保留状态、P 客户端缓存与 clamp、Q 指纹去重、R 沙箱工作台装配与 S 多模态中间件等 19 个场景）。
14. `tests/smoke_models.py`：模型角色默认 provider、Gemini 回退、未知 provider、缺 key 作用域、惰性单例与 thinking 文本隔离测试（5 个场景，构造过程不访问网络）。
15. `tests/smoke_host_shell.py`：宿主机命令运行器测试（环境变量白名单、编码回退、免授权精确匹配、基础运行、子进程环境与 stdin、超时杀进程树、脱离孙进程不挂住、截断与洪泛硬上限、审计日志，以及仅 Windows 的专项用例，共 10 个场景）。

此外还提供 `tests/live_sandbox.py` 真机验收测试套件（包含 18 项端到端用例，需在已启动 Docker 守护进程的环境下执行 `uv run python tests/live_sandbox.py --live` 运行；慢测试需追加 `--slow` 参数）。

`tests/live_magpie.py` 是独立的 Magpie 协议探针，覆盖文本、工具轮次、普通图片、工具结果内图片和 browser_use 结构化输出。它不会启动真实浏览器/Desktop，也不会执行真实工具；只有显式设置 `RUN_MAGPIE_LIVE=1` 后运行 `uv run python tests/live_magpie.py` 才会访问网关，且不会加入离线 smoke 或 CI。

---

## 9. 已知限制

1. **DesktopAgent 不稳定**: 桌面操作依赖网格视觉识别，受分辨率和缩放比例影响较大，可能存在点击漂移。
2. **会话重启后的边界**: 会话、事件日志、挂起的授权与模型上下文已持久化（见 2.11），重启后上下文与未决 HITL 都能恢复。但**非 HITL 的中途崩溃轮不会自动续跑**：启动时只把崩溃前未处理的消息标记为放弃（`turn_abandoned`），需要用户重新发送。长期事实仍依赖 Milvus 后台抽取持久化。
3. **专家单次委派有工具调用预算**: 每个专家单次委派内的工具调用次数受 `config.SUBAGENT_TOOL_LIMITS` 限制（见 2.8；FileAgent 默认 20 次，授权恢复后重新计数），触顶后调用被拦截、专家以已有结果收尾，复杂任务可能因此汇报不完整；`write_file` 对已存在文件会整体覆盖。
4. **增量入库不是事务**: 图片/PDF 的 VLM 摘要生成、写入 Milvus、把文件从 `data/pending/` 移到 `data/source/` 是三个独立步骤，失败不会回滚。若 VLM 调用失败（或摘要已生成但入库前进程中断），而同一批还有其它成功入库的文档，该文件仍会被移动到 `data/source/`，其摘要块之后不会被自动补入，表现为“文件已归档，但检索不到”。恢复方式：修复后执行 `uv run python main.py --rebuild`；若文件仍卡在 `data/pending/`，先删除对应的 `data/pipeline/summaries/<hash>.md`，并确保 `REFRESH_BM25=False` 后重新启动。
5. **记忆提取仅信任用户原话**: 只记录用户原话，用户用“对/是的”确认 AI 转述不会被记住。
6. **历史会话记忆块不自动回溯清洗**: 会话历史里旧的 `<long_term_memory>` 块不会随记忆更新而清理。
7. **相似度检索局限**: 相似度检索可能漏掉与当前问题无关的重要事实。
8. **沙箱 /work 为临时内存文件系统**: 沙箱中的工作目录 `/work` 基于 256MB tmpfs 内存文件系统，机器人进程重启或会话空闲超过 `SANDBOX_IDLE_TTL_SECONDS` 触发清理重建后内容清空，非持久化存储。
9. **沙箱产物导出目前仅支持图片**: 目前沙箱多模态解析仅自动识别并导出 `/work/outputs/` 下的有效图片（PNG/JPEG/WebP/GIF），非图片产物（CSV/JSON/HTML 等中间结果）v1 暂不导出。
10. **沙箱会话级单例模式**: 沙箱在单机器人进程中采用单例设计与固定容器标签（`gnk48.sandbox=1`）。若在同一宿主机上并发启动多个机器人实例，可能发生容器抢占或清理冲突（建议单实例运行）。
11. **沙箱镜像更新需手动重新构建**: 修改 `deploy/sandbox/` 下的 Dockerfile 或配置文件后需手动执行 `docker build` 重新构建；运行中的常驻沙箱容器不会自动感知镜像变更，需重启机器人或待空闲 TTL 触发重建。
12. **宿主机命令没有沙箱**: `run_host_command` 以机器人进程所属用户的权限运行，唯一防线是人工授权（注意避免“反射性回 Y”）、环境变量白名单和提示词约束。
13. **免批 git 命令在不可信仓库里可能执行代码**: `git status` / `git diff` 等免授权命令可能经仓库配置（如 `core.fsmonitor`、外部 diff 驱动）执行代码；不要让 workspace 指向不可信仓库，或把相关条目从 `HOST_SHELL_AUTO_APPROVE_COMMANDS` 中移除。
14. **进程树清理有边界**: `taskkill /T` 杀不到父进程已退出的孙进程；超时后遗留的后台进程不会被清理。
15. **授权提示可能发不出去**: QQ 若拒收含“像域名”的文本（如命令里的 `a.b`），授权提示可能发不出，此时用 `/reset` 兜底。
16. **沙箱是进程级全局，不按会话隔离**: `/new`、`/switch` 不动 Docker 沙箱，所以不同会话共用同一个 `/work`；只有 `/reset` 会重置沙箱。
17. **根命名空间 checkpoint 不清理**: 只清理子专家的嵌套命名空间（见 2.11）；`checkpoints.db` 会随对话轮数持续增长，需要时用 `scripts/manage_sessions.py delete` 删除整个会话。
18. **刚挂起即崩溃会丢授权前已生成的图片**: 挂起态（用户原话与已生成图片）在授权提示发出的同一轮收尾时才落库；若进程恰好在 checkpoint 落盘之后、挂起态写入之前崩溃，授权仍可恢复，但授权前已生成的图片会丢失。
19. **事件时间戳是写入时间**: 事件日志里的 `ts` 是事件写入日志的时间，不是原始发生时间（镜像发生在每轮收尾，启动恢复时补镜像）。
