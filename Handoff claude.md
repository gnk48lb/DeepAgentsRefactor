# 交接文档：Deep Agents 重构项目（给新的 Claude 对话）

> 更新于 2026-10-01。新对话没有前文记忆，本文是唯一的上下文来源。
> 与仓库里的 `deepagents-refactor-plan*.md`、`walkthrough*.md` 冲突时，以本文和用户上传的**实际代码**为准（那些是历史记录，部分已过时）。

---

## 0. 你的角色与工作方式

- 你是架构顾问兼 code reviewer。用户用 **Antigravity**（Google 的 AI 编程助手，能读写仓库、跑命令）执行改动；你负责规划、写给 Antigravity 的提示词、审核它的回报。
- 你**看不到仓库**，只能看到用户上传或粘贴的内容。开始前让用户上传第 9 节的文件；审核时让用户贴 Antigravity 的**原始输出**，不要只看它的总结。
- 用户偏好：中文；先商量再动手；提示词要能整段复制；风险可接受（独立副本目录 + git，大不了推倒重来）；全部重构完成后由用户自己做最终测试，**包括 DesktopAgent，不要让 Antigravity 测 DesktopAgent**。
- 用户有时会把"另一个对话"的 review 贴来让你核对。请独立验证其中具体的事实性说法（日期、API 名、issue 编号）。此前出现过把相邻文章的日期读串的错误，结论没变，说法却是错的。
- 前任的失误也要认：`MAIN_AGENT_PROMPT` 里"图文分离"一段声称主 Agent 能在上下文里看到专家汇报中的图片，没有核实 `task` 的返回方式，已证实是错的（见 7.1）。

## 1. 项目与目标

- **项目**：GNK48-Agent，QQ 机器人（NoneBot）多智能体系统，Windows 本机运行。依赖 Milvus Lite（知识库/记忆）、Neo4j（媒体图谱）、MySQL、Docker（代码沙箱）、MCP（高德/文件系统/桌面）。
- **目标**：把原来手写的 LangGraph Supervisor+Router+8 个 Worker，迁移到 **deepagents 0.7.13**（LangChain 的 agent harness：主 Agent 的 ReAct 循环 + 内置 `task` 工具委派 subagent）。
- 在独立副本目录 `DeepAgentsRefactor` 中进行，git 分支 `deepagents-phase1`（以 `git branch` 为准）。uv 管理依赖：deepagents 0.7.13、langchain 1.4.0、langchain-core 1.6.2、langchain-google-genai 4.3.7。
- 新旧并存：旧入口 `main.py` + `app/graph.py` 仍可运行。**`app/deep_agent.py` 仍 import `app/graph.py`**（`build_file_agent_subgraph`、`build_desktop_agent_subgraph`、两个 `*_AGENT_PROMPT`），所以 graph.py 现在不能删。

## 2. 当前架构（已实现）

**入口**
- `main_deepagents.py`：新控制台入口（Windows `os.rename` 补丁、`astream` 事件映射、FileAgent 中断的 Y/N 交互、记忆后台提取）。
- `plugins/chat.py` + `bot.py`：QQ 入口，**仍是旧架构，尚未迁移**。

**`app/deep_agent.py`（核心）**
- `build_main_agent(checkpointer=None)`：初始化 MCP → 组装 subagents → `create_deep_agent(model=models.supervisor_llm, system_prompt=MAIN_AGENT_PROMPT, subagents=..., middleware=[ToolCallLimitMiddleware(tool_name="task", run_limit=12, exit_behavior="continue")], checkpointer=MemorySaver())`。
- 6 个声明式 subagent：KnowledgeAgent（`models.vlm`；`rag`、`web_search`）、MediaAgent（`av_graph_rag`）、MapAgent（amap MCP 工具）、SQLAgent（`execute_sql`）、BrowserAgent（`browser_tools`）、CodeAgent（`execute_python_code`）。除 KnowledgeAgent 外都用 `models.worker_llm`；各自 system_prompt 沿用旧 graph.py 原文。
- 2 个 `CompiledSubAgent`：FileAgent、DesktopAgent，由 `SubgraphCompiledWrapper` 包装旧 `build_*_subgraph()` 产物；子图内部（含自定义 `interrupt()`）一行未改。
- `general-purpose`：显式注册一个 `tools=[]` 的窄权限同名 subagent（硬覆盖框架默认），同时保留 `register_harness_profile(..., GeneralPurposeSubagentProfile(enabled=False))` 作第二层。
- `build_initial_messages()`：应用层调 `database.search_memory(top_k=1)`，命中则作为 SystemMessage 前置。`_background_extract_memory()`：后台提取长期记忆。

**其他新增/改动**
- `app/outbox.py`（新，**未提交**）：基于 contextvar 的图片发件箱。`execute_python_code` 产出沙箱图片时 `push`；入口每条消息在 `astream` 前 `begin()`。原因：`task` 只回传文本，图片无法经消息链路到达接入层。
- `app/utils.py::extract_text`：Gemini 3.x 返回的 `content` 是 list（含 `extras.signature` 的 text 块），统一用它取文本；`deep_agent.py`、`main_deepagents.py`、`loader.py` 已改用。
- `config.py` / `app/models.py`：所有 LLM 均为 Gemini（GitHub Models 已退役，域名 NXDOMAIN）。`worker_llm`/`supervisor_llm`/`desktop_llm` 读 config，改前改后取值一致。
- `mcp_servers.json`、`app/mcp_service.py`：未改（amap / filesystem / desktop）。

## 3. 决策记录

| 项 | 决定 | 理由 / 备注 |
|---|---|---|
| Supervisor+Router → 主 Agent + `task` | 已完成 | 约束全靠主 prompt；暂不用 `response_format`，飘了再加 |
| 死循环防护 | 官方 `ToolCallLimitMiddleware(tool_name="task", run_limit=12, exit_behavior="continue")` | 不用 `"end"`：并行工具调用会抛 `NotImplementedError`；与 checkpointer 组合有已知 bug（langchain #34159）；且只吐写死的提示字符串。放弃了自写计数 middleware（状态原地修改不会跨调用持久化） |
| FileAgent / DesktopAgent | `CompiledSubAgent` 包旧子图，内部不改 | 已用连续三次危险操作确认验证 resume 正常，deepagents Discussion #1762 的担忧在此方案下未复现 |
| HITL 协议 | 保持 "Y"/"N" 字符串 + `Command(resume=...)` | 中断是子图内部自定义的，不走 deepagents 的 `interrupt_on` 结构化格式。**旧计划里"重写 HITL 协议"一条作废** |
| 默认 general-purpose subagent | 显式同名窄权限条目 + profile 双保险 | 默认那个会拿到宽泛工具，可能绕过 FileAgent 的人工确认 |
| TodoListMiddleware | 不加 | v0.7 起默认关闭，官方评测显示默认开启无显著提升。以后要加，先试 DesktopAgent（多步 GUI 序列），不是 BrowserAgent（它委托给 browser-use 内部循环，Todo 看不进去） |
| filesystem MCP → FilesystemBackend | **暂定不换（非否决）** | 真做时注意：内置工具只有 ls/read_file/write_file/edit_file/delete/glob/grep，没有 move/mkdir，`ls` 非递归（用 `glob("**/*")`），要用 `CompositeBackend` 避免框架内部文件混进工作区，`delete` 可用工具白名单排除（契合"绝不真删，只归档"） |
| 沙箱 | 暂定；继续用自写 Docker 沙箱，作为普通 tool | 官方一线沙箱是云服务（Modal/Daytona/Runloop/LangSmith），本地 Docker 没有内置方案 |
| 长期记忆 | Milvus 保留；检索/提取在应用层 | `system_prompt` 是静态字符串，不能每轮 `.format()` |
| 图片传递 | 发件箱旁路 | `task` 只取子 Agent 最后一条非空 AIMessage 的 `.text`（已读源码确认） |
| MCP | 不变 | deepagents 用的就是 langchain-mcp-adapters |
| Async subagent / Skills | 暂不做 | Async 需要自建 Agent Protocol 服务 |

## 4. 验证状态

**已验证（有原始输出为证）**
- 天气问题路由到 MapAgent（多次）；同一 `thread_id` 两轮（"济南今天天气怎么样" → "那明天呢"）上下文接得上。
- KnowledgeAgent + RAG（命中图片描述 chunk，返回含 `image_url` 的多模态内容）在 Gemini 下没有 400。旧的 `multimodal_prompt_handler` 是为 OpenAI 的 400 写的，Gemini 下不需要。
- FileAgent：单次中断、连续三次中断（write_file → create_directory → move_file）恢复均 PASS，文件内容已核对。注意：都是脚本自动答 Y，**真人 `input()` 路径未手测**。
- subagent 名单运行时内省为 9 个（8 个专家 + 唯一的 `general-purpose`）。`task` 的 `subagent_type` 在 schema 里是纯 string，没有 enum。
- `task` 源码（`deepagents/middleware/subagents.py` 约 L701–726）：倒序找最后一条非空 AIMessage，只取 `.text`，丢弃 list 内容。
- `extract_text` 有效；真实 VLM 调用确认 `res.content` 是 list。
- 长期记忆 检索→注入→被模型引用（Phase 1 验证）。

**未验证**
- 发件箱端到端：验收 (a) 因 Docker Desktop 未启动失败，(b)(c) 没做。
- CodeAgent / BrowserAgent / MediaAgent / SQLAgent / DesktopAgent 在新架构下**都没跑过**；FileAgent、KnowledgeAgent、MapAgent 是仅有跑过的三个。
- `run_limit` 触顶行为、一次并行委派多个专家（Part 1 验收标准第 2 条，从未被报告过）。
- `general-purpose` 的行为级验证（名单里有它，但没验证它背后确实是我们的窄权限版本；依据是官方文档"显式同名 subagent 会替换默认"）。
- 主 Agent 与各 subagent 实际可见的工具清单（见 7.6）。

## 5. 仓库状态

- 已提交：Phase 1 骨架、FileAgent/DesktopAgent 接入、general-purpose 覆盖、`extract_text`、模型切换 Gemini、`models.py` 读 config、`main_deepagents.py` 初版（以 `git log` 为准）。
- **未提交**：`app/tools.py`、`main_deepagents.py`（发件箱相关）、新文件 `app/outbox.py`。
- 未跟踪的临时脚本：`scripts/test_huoshen_query.py`、`scripts/test_rag_direct.py`（排查 RAG 召回用，不要提交）。
- 原样未动：`app/graph.py`、`main.py`、`plugins/chat.py`、`bot.py`、`mcp_servers.json`。
- `requirements.txt` 已过时，以 `pyproject.toml` + `uv.lock` 为准。

## 6. 待办（按顺序）

1. **【用户】** 启动 Docker Desktop，确认 `docker images` 里有 `python-ds:latest`。
2. **【Antigravity】** 执行 `handoff-antigravity.md` 的本轮任务（环境检查 → 提交检查点 → 发件箱验收 → 只读查工具清单）。**你来审核它的回报。**
3. 若工具清单显示主 Agent / subagent 带有内置文件类工具：起草"关闭它们"的小改动（harness profile 的 `excluded_tools`，或 FilesystemMiddleware 的工具白名单；先查文档确认写法，改完必须用同样的内省重新打印确认生效）。
4. **【和用户商量后交给 Antigravity】** 主 prompt 修改（7.1–7.3）。
5. **【商量】** RAG 召回问题（7.4）。
6. **【商量】** 记忆注入的相关性阈值与线程累积（7.5）。
7. **【Antigravity】** `plugins/chat.py` 迁移（清单见第 10 节）。
8. **【Antigravity】** 小范围测试：并行委派 + `run_limit` 触顶（构造"天气 + 查技能"这类需要同时调两个专家的问题）。
9. **【Antigravity】** 收尾清理：把 FileAgent/DesktopAgent 子图构建函数和提示词从 graph.py 搬出，再考虑退役 graph.py / main.py；清理 GitHub Models 残留配置。
10. **【用户】** 最终全面测试：DesktopAgent（高危键位拦截）、BrowserAgent、MediaAgent、SQLAgent、CodeAgent、真人 Y/N 交互、QQ 端到端。
11. **【暂定，不阻塞】** FilesystemBackend、沙箱 backend、TodoListMiddleware（DesktopAgent）、SummarizationMiddleware。

## 7. 已知隐患与待决定

**7.1 `MAIN_AGENT_PROMPT` 的"图文分离"一段已过时（已确认）**
它写"专家汇报中如果包含图片，你能在上下文里直接看到"。`task` 只回传文本，这句是错的，可能诱导模型编造对图片的描述。建议改为：图片不会出现在你的上下文里，系统会自动把专家产出的图片发给用户；你只需基于专家的文字汇报回答（可以告诉用户图已生成）；不要输出图片链接、Base64 或 Markdown 图片语法。

**7.2 最终回答里出现 Markdown（已确认）**
MapAgent/KnowledgeAgent 的战报带 `*   **xxx**`，主 Agent 原样搬进最终回答，QQ 不渲染。用户说过由他来定。建议在主 prompt 加一句：最终回答是发到 QQ 的纯文本，不使用 Markdown 语法（不要 `**加粗**`、`#` 标题、表格、代码围栏），需要列表时用"1."或"·"开头的普通文本行；去掉标记符号但保留全部数据。

**7.3 "信息完整性"与"500 字"冲突（推断）**
上一轮专家战报含各技能的耗蓝/施法距离，本轮最终回答里没有。本轮战报没贴，无法确证，但最可能是 prompt 里两个"最高优先级"互相打架。建议写明取舍：数据字段优先于说明性文字，超长时先删解释性语句，保留全部数值/名称/路径；仍超出时在末尾注明"内容较长，已省略 XX，可继续追问"，不要静默丢字段。

**7.4 RAG 召回：宽泛的"列举类"问题漏项（已确认现象，原因为推断）**
"火神战姬的技能有哪些？"只返回 W/E/R，没有 Q。直接查"火神战姬 Q技能"能命中 `images/火神战姬_page_1.png` 摘要里的 Q 技能 chunk，说明知识库有数据；宽泛查询的 rerank top-3 被 page 2 的 W/E/R 汇总 chunk 占满（Antigravity 的解释，未证实）。选项：(a) `rag` 的 `top_k` 从 3 提到 5；(b) 放宽 KnowledgeAgent 提示词里"严禁拆分查询"那条，对"全部/有哪些"类问题允许补查；(c) 检查图片摘要的分块。这是旧架构里就存在的问题，不是迁移引入的。

**7.5 记忆注入的两个问题（推断自代码，未测）**
- `search_memory` 没有相关性阈值，`top_k=1` 总会返回最近的一条，所以无关记忆（比如"用户是 Python 开发者"）几乎每轮都会被注入。旧架构同样如此。
- 新架构里每轮都把记忆 SystemMessage 追加进被 checkpointer 持久化的消息列表，同一线程多轮会累积重复的记忆消息。可能的做法：加相似度阈值（先打印分数标定）；注入前检查本线程里是否已有相同内容。

**7.6 内置虚拟文件工具（推断自官方文档，需实测）**
deepagents 默认给主 Agent 和声明式 subagent 配 ls/read_file/write_file/edit_file/glob/grep（0.7 起还有 delete），默认 StateBackend，操作的是内存 state，不是真实磁盘。风险：主 Agent 可能自己调 `write_file` 以为写了真实文件，而不是委派 FileAgent；每个 subagent 也多背几个工具描述。待办 2 里的只读检查会给出实际清单。

**7.7 `run_limit` 与并行委派从未测过**
`continue` 模式理论上规避了 `NotImplementedError`，但没有实测。

**7.8 `register_harness_profile` 的 provider key 是否对直传的 `ChatGoogleGenerativeAI` 实例生效，没有证据。**
general-purpose 的硬覆盖不依赖它。以后凡是依赖 profile 的改动，都要用运行时内省验证。

**7.9 `graph.py` 暂时不能退役**，见第 1 节。

## 8. 给 Antigravity 写提示词的协议

**此前出现过的回报失真（都有证据）**
1. Part 1 自报"全部顺利完成"，复查发现 5 处问题：天气路由给了 KnowledgeAgent、记忆检索调用了不存在的函数、`MAIN_AGENT_PROMPT` 与规格不符、两处 model 写错。
2. 验证 general-purpose 是否被关掉，用的是"问模型介绍你的团队"，这证明不了任何事。
3. 要求"原样输出"，实际重新打字（战报"夜间多云"，最终回答"夜间多复"）。
4. 要求"贴最终回答原文"，用省略号和括号概括带过。
5. 验收要求"知识型问题"，实际跑了"1+1"。
6. 一轮 4 项任务只汇报了 2 项，另外 2 项整个没提。
7. 调查结论与源码不符：说 `execute_python_code` 只返回文字路径，实际代码有图片时返回含 `image_url` 的 list。

**写提示词时**
- 每条 ≤4 项，编号；每项写明"可检查的交付物"。
- 要求先贴 `git status --short`，结束时贴 `git status --short`、`git diff --stat`、`git log --oneline -3`——漏项一眼可见。
- 要求逐条对应编号汇报，标 ✅/❌/⏸，不得省略。
- 要求原样粘贴终端输出，同时保存到 `scratch/logs/`，用户可以核对文件。
- 写明"不要做"的清单；环境问题（Docker、网络）一律"停下来报告"；需要决策的事不要交给它。

**审核回报时检查**
条目数是否等于提示词条目数；有没有省略号/括号概括；战报与最终回答的文字是否一致（笔误往往暴露重新打字）；测试用例是否按规格；结论是否有对应证据；要求贴源码的是否贴了。

## 9. 新对话开始前，请用户上传

必传：`handoff-claude.md`（本文）、`app/deep_agent.py`、`main_deepagents.py`、`app/tools.py`、`app/outbox.py`、`config.py`、`app/models.py`、`pyproject.toml`
做 `plugins/chat.py` 迁移时再传：`plugins/chat.py`、`bot.py`
按需：`app/graph.py`（很大，只在涉及 FileAgent/DesktopAgent 子图时需要）、`app/database.py`、`app/sandbox.py`
建议你的第一句回复：用自己的话复述当前状态和下一步，确认理解一致，**先不要写提示词**。

## 10. 技术备忘

**deepagents 0.7.13**
- `create_deep_agent(model, system_prompt, subagents, middleware, checkpointer)`；`model=` 接受 `BaseChatModel` 实例。
- `CompiledSubAgent(name=, description=, runnable=)`；`task` 工具参数是 `description` 与 `subagent_type`。
- `ToolCallLimitMiddleware` 的导入路径是 `langchain.agents.middleware`。
- `register_harness_profile` 是 beta API，注册是叠加合并的，key 是 provider 或 `provider:model`。
- v0.7 变化：默认基础 prompt 为空；Todo 默认关闭；Backend 工厂函数被删，必须传具体实例（`StoreBackend` 要显式 `namespace`）；`write_file` 直接覆盖；新增 `delete` 工具。

**流式事件与中断**
- `astream` 的 node 名：`model`（AIMessage 带 `tool_calls` = 委派；不带 = 最终回答）、`tools`（`ToolMessage(name="task")` = 专家战报）、`*Middleware*` 节点要跳过。
- 中断：一轮 `astream` 结束后 `await agent.aget_state(config)`，`state.next` 非空即有中断；payload 在 `state.tasks[*].interrupts[*].value`（FileAgent 是 dict：`tool_name`、`tool_args`、`tool_call_id`；DesktopAgent 是字符串）；恢复用 `agent.astream(Command(resume="Y"/"N"), config=...)`，循环到 `state.next` 为空。

**其他**
- Gemini 3.x 的 `content` 是 list，一律走 `extract_text`。
- Docker 沙箱：需要 Docker Desktop 运行 + 镜像 `python-ds:latest`；无网络、256MB、30 秒超时，预装 pandas/numpy/matplotlib。
- Windows：入口文件里有 milvus-lite 的 `os.rename` 兼容补丁；Python 用 `.venv\Scripts\python.exe`。

**`plugins/chat.py` 迁移清单**
1. 启动时 `deep_agent.build_main_agent()` 取代 `graph.build_graph()`；`database.run_global_database_init()` 照旧；MCP 初始化已在 `build_main_agent()` 内。
2. 每条消息：`outbox.begin()` → `build_initial_messages()` → `astream`。
3. 最终文本：最后一条不带 `tool_calls` 的 AIMessage，经 `extract_text`；`Supervisor_Final` 这个标记已不存在。保留 600 字截断作兜底。
4. 图片从发件箱取，不再从消息里扫；`_make_image_segment` 保留。
5. 图片意图翻译里的 `vlm_res.content.strip()` 改走 `extract_text`。
6. HITL：用 `aget_state`；`_format_hitl_warning` 的 payload 结构没变，可复用；resume 值保持 "Y"/"N"。注意：用户回复 "Y" 的那条消息触发恢复，此时如果再调 `outbox.begin()`，会清掉中断前已登记的图片，需要决定。
7. 记忆提取只取"本轮用户文本 + 最终回答"两条；`database.search_memory` 是同步调用，包进 `asyncio.to_thread`。
8. 旧计划里提到的 `wechat_server.py` 在当前文件清单里没见到，以实际仓库为准。