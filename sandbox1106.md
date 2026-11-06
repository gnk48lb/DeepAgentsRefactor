# 任务（Phase 2）：把 CodeAgent 升级为“沙箱工作台”（FilesystemMiddleware + execute）

Phase 1 已验收（`DockerSandbox(BaseSandbox)`、常驻容器、`/workspace` 只读、`/work` tmpfs）。本阶段让 CodeAgent 在沙箱里获得 Coding Agent 式工作台：`ls / read_file / write_file / edit_file / glob / grep / execute`（Linux 命令），并保留 Phase 1 的 `execute_python_code` 作为“一步 Python / 绘图”的快速通道。
设计已定（§2）。**先做 §1 的前置修复**，不修的话文件工具在真容器里根本不能用。遇到与现实不符的地方先实测，在汇报里记录偏离；不要中途停下来提问；每完成一个 P0 就跑一次 `uv run python tests/run_smoke.py`。

## 0. 先读
- app/infra/sandbox.py（execute / upload_files / download_files / run_python / _ensure_container / _get_client）、app/tools/code_tools.py、app/agents/experts/code_agent.py 与 file_agent.py（file_agent 是同构参照）、app/agents/{orchestrator,budget,skills,prompts}.py、app/config.py
- tests/smoke_{sandbox,budget,orchestrator,architecture,skills}.py、tests/live_sandbox.py、deploy/sandbox/Dockerfile
- 以 .venv 里实际安装的 deepagents / langchain 源码为准：deepagents/middleware/{filesystem,skills}.py、backends/{sandbox,composite,protocol}.py。下文“已验证”的结论来自我在 deepagents 0.7.13 / langchain 1.4.9 / langgraph 1.2.20 下的实测，请在本仓库锁定的版本上复测；不一致就以复测为准，并在汇报里写明。

## 1. P0 前置修复（阻塞项；每项先写一个会失败的回归测试，再修）
**P0-1 `DockerSandbox.execute` 的命令包装是坏的。** 现在是 `cmd = f"sh -c '{command}'"`，docker-py 对 str 形式的 cmd 会再做一次 `shlex.split`。我用 BaseSandbox 真实生成的 6 条 read/ls/glob/grep/edit/write 命令验证：5/6 要么报 `No closing quotation`、要么被拆成上千段、要么引号被吃掉导致内容变形。Phase 1 的测试只跑了 `ls`/`echo`，所以没发现。
修法：传 list，不经 shlex：`["timeout","-k","3",str(T),"bash","-c",command]`，workdir=/work，user=10001:10001。用 `bash` 而不是 `sh`（模型会写 `[[ ]]`、花括号展开；BaseSandbox 的命令是 POSIX 的，在 bash 下同样可用）。
回归测试：写一个继承 BaseSandbox、重写 `execute` 来记录命令的子类，调用 read/ls/glob/grep/edit/write 拿到真实命令；再让 FakeDocker 记录 `exec_create` 的 cmd，断言它是 list 且 `cmd[-1] == 原命令`。

**P0-2 协议方法不得向外抛异常。** 实测：沙箱方法一旦抛异常，整个 CodeAgent 运行直接崩溃（ToolNode 不会把它变成错误消息）。Docker 没开时 `SandboxUnavailable` 就会这样。`execute` / `upload_files` / `download_files`（以及它们依赖的 `_ensure_container` / `_get_client`）对 `SandboxUnavailable`、`docker.errors.DockerException`、`requests` 异常、`OSError` 一律转成：`ExecuteResponse(output=<友好文本，含修复指引>, exit_code=1)`；上传/下载则逐文件返回 error 字符串。`run_python` 仍可抛 `SandboxUnavailable`（L3 工具会接住）。

**P0-3 协议路径的输出上限不能是 8000 字符。** BaseSandbox 的 `read` 单页最多 500KB，glob/grep 最多 10000 条，都以一整段 JSON 从 `execute` 的输出里解析；头尾截断会把 JSON 截坏。新增 `SANDBOX_EXECUTE_MAX_OUTPUT_CHARS`（默认 2_000_000）专用于 `execute()`；`run_python` 保持 8000/3000。上限调大后，`StreamingBuffer` 的尾部缓冲不要再逐字符 append，改成按块切片（洪泛场景下避免 CPU 打满）。给模型看的上下文由 FilesystemMiddleware 的大结果转存机制兜底（超过 20000 token 会存到 /_artifacts 并只给预览）。

**P0-4 超时不得销毁容器。** 现在 `execute` 没有容器内超时，只有宿主看门狗 `container.kill()`；`run_python` 与 `execute` 遇到 124/137 都会 `close()`。工作台里 `/work` 存着模型写的文件，每次超时都清空太伤。新规则：
- `timeout -k` 返回 124，或 137 且耗时 ≥ T → 容器保留，在输出/notes 末尾追加“命令超过 N 秒被终止；沙箱状态保留”。
- 137 且耗时 < T → 进程被信号/OOM 杀死，提示“进程被杀（可能超出内存上限 {DOCKER_MAX_MEM}）”，容器保留。
- 只有宿主看门狗（T+10s 仍未结束）才 kill 并标记重建，注明“沙箱已重置，/work 已清空”。
- 可选：超时后用 `pkill`（镜像里会有 procps）做一次残留进程清扫。

**P0-5 DockerClient 与超时参数。**
- (a) `_get_client()` 现在每次调用都 `docker.from_env()` 新建客户端（BaseSandbox 一次文件操作会触发多次调用）→ 缓存到实例上，`close()`/shutdown 时关闭。
- (b) 读超时现在固定 SANDBOX_TIMEOUT_SECONDS+30=90s，而 FilesystemMiddleware 默认允许模型传 `timeout` 到 3600s。新增 `SANDBOX_MAX_TIMEOUT_SECONDS`（默认 300）；构造 `FilesystemMiddleware(max_execute_timeout=该值)`（已实测：超出会直接给模型返回错误、不转发）；`execute` 内部再 clamp 一次（0/None 回落默认值，不得解释为“不限时”）；流式 exec 期间把 `client.api.timeout` 临时设为 T+30（调用在锁内、串行），结束后还原。

**P0-6 `run_python` 与产物语义调整。** (a) 脚本路径由 `/work/main.py` 改为 `/work/.sandbox/snippet.py`（模型很可能自己 write_file 一个 main.py，不能被覆盖）。(b) 不再每次 `rm -rf /work/outputs`；容器创建后一次性 `mkdir -p /work/outputs`（和预热字体缓存的 exec 合并）。(c) 产物改为“按指纹投递”，见 A3。

## 2. 目标架构（已定）
**A1 CodeAgent 变成 `CompiledSubAgent`**（参照 file_agent.py；用 `langchain.agents.create_agent`，所以不受主图 HarnessProfile.excluded_tools 影响；**不要改主图的 HarnessProfile**）。
- `code_agent.py` 提供 `build_code_agent_runnable(llm, system_prompt, checkpointer=None, sandbox_backend=None, max_tool_calls=None)`；保留 `NAME`、`DESCRIPTION`、`PROMPT`（须含“你是 CodeAgent”，cli_logger 靠它识别）；删除 `SPEC` 与 ExpertSpec 引用。
- orchestrator：从 `_DECLARATIVE_SPECS` 去掉 code_agent.SPEC，在 `_local_subagents()` 里构建 code 子专家；最终顺序必须仍是 `["KnowledgeAgent","MediaAgent","MapAgent","CodeAgent","BrowserAgent","FileAgent","DesktopAgent"]`。构建主 Agent 时不得连接 Docker。

**A2 装配**
```python
SANDBOX_FS_TOOLS = ["ls","read_file","write_file","edit_file","glob","grep","execute"]   # 故意不含 delete

sb = sandbox_backend if sandbox_backend is not None else sandbox.get_sandbox()
backend = CompositeBackend(
    default=sb,                                   # 必须是 BaseSandbox，FilesystemMiddleware 才会暴露 execute
    routes={SKILLS_ROUTE: FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True),
            "/_artifacts/": StateBackend()},
    artifacts_root="/_artifacts")
create_agent(
    model=llm, tools=list(code_tools.TOOLS),      # execute_python_code 保留
    system_prompt=system_prompt,
    middleware=[
        SandboxOutputsMiddleware(sb),             # 最外层，见 A3
        SkillsMiddleware(backend=backend, sources=[SKILLS_ROUTE]),   # from deepagents.middleware.skills import SkillsMiddleware
        FilesystemMiddleware(backend=backend, tools=SANDBOX_FS_TOOLS,
                             max_execute_timeout=config.SANDBOX_MAX_TIMEOUT_SECONDS),
        *build_budget_middleware(max_tool_calls),
    ],
    checkpointer=checkpointer)
```
`/skills/` 走宿主目录（加载技能不需要 Docker），沙箱 shell 里看不到 `/skills/`。不接 HITL：沙箱无网络、`/workspace` 只读，没有宿主写入路径。

**A3 图片/产物投递（统一管线）**
- **L1** `DockerSandbox.collect_new_outputs() -> OutputsResult(images, other_files, notes)`：
  - 用一次 exec 列出 `/work/outputs`（文件名可能含空格/中文，建议 `find … -printf '%P\t%s\t%T@\0'` 按 NUL 切分；目录不存在视为空），与实例内 `_seen_outputs{相对路径:(大小,mtime)}` 比较，只处理新增/变更文件。
  - 图片取回字节（复用现有 `_download_single_file`，沿用 ≤4 张 / ≤10MB / ≤40M 像素）；非图片只记名称与大小（v1 仍不导出）。
  - **所有新增/变更的指纹都记为已见**，包括超出数量上限被跳过的（notes 写明“已跳过：…，需要时请重新生成”）；容器重建/close 时清空 `_seen_outputs`。
  - `run_python` 执行后调用它，`RunResult` 沿用现有字段。
- **L3** `code_tools.py` 新增公共函数 `publish_images(...) -> (已发送文件名, 给模型的 content blocks, 失败 notes)`：用户版（长边 1600、q88）`outbox.push`，模型版（长边 768、q75）返回 blocks；`execute_python_code` 改为复用它。
- **L4** `SandboxOutputsMiddleware(AgentMiddleware)`（放在 code_agent.py，传入与 backend.default 同一个沙箱实例）：只拦 tool 名为 `execute` 的调用（`execute_python_code` 自己已投递，不重复）。流程：
  1. `result = handler(request)`；
  2. 调 `collect_new_outputs()`，**先投递**（`publish_images`）；
  3. 再**尽力改写**返回的 ToolMessage：`msg.model_copy(update={"content": [{"type":"text","text": 原文本 + "\n[系统提示] 已向用户发送 N 张图片（…）；附带的是缩略预览，仅供你检查"}, *模型缩略图 blocks]})`。
  - 投递与改写要解耦：返回值若不是 ToolMessage（如 Command），仍要完成投递，改写尽力而为；任何异常只记日志，绝不影响工具结果。对不具备 `collect_new_outputs` 的沙箱（测试替身）直接透传。
  - **必须同时实现 `wrap_tool_call` 与 `awrap_tool_call`**。已实测：只写同步钩子时，`ainvoke` 会直接抛 `NotImplementedError`，而 chat_service 用的是 ainvoke。async 版里把阻塞的 docker 调用放进 `asyncio.to_thread`，`outbox.push` 放回协程里做。
  - 用 `model_copy` 而不是新建 ToolMessage。已实测：这样才保留 `artifact={'exit_code': …}`、status、id；状态里的 ToolMessage 接受多模态 content。
  - 顺序：放在 FilesystemMiddleware **外层**（列表里更靠前）。
  - 退路（现在不做）：若实测 Gemini 对 ToolMessage 内含图片报错，就让中间件只投递、不回传缩略图；若 flash-lite 老在 `execute_python_code` 与 `execute` 之间选错，可直接去掉 `execute_python_code`，因为 `execute` 已具备同样的出图与回传能力。

**A4 预算**：`SUBAGENT_TOOL_LIMITS["CodeAgent"]` 10→15。已实测：同样的栈（Skills + Filesystem + 预算中间件，嵌在 create_deep_agent 里，recursion_limit=60），N=10/15/20 都能被硬停止体面收尾，不触发 GraphRecursionError。仍要在 tests/smoke_budget.py 补“CodeAgent 失控 + 生产递归上限”的测试把它钉住；失败就调低预算或抬高 AGENT_RECURSION_LIMIT。

**A5 镜像**：apt 追加 `ripgrep jq unzip zip file tree procps`。deepagents 的 grep 工具是字面匹配，描述里明确让模型用 `rg '<regex>'` 做正则，所以 ripgrep 是必需的。`LABEL gnk48.sandbox.version=3`，`config.SANDBOX_IMAGE_VERSION="3"`（让旧镜像被版本检查拦住并提示重建）；测试里不要再硬编码 "2"，一律引用 config。

**A6 路由与提示词**
- `code_agent.DESCRIPTION`：“代码与终端专家：在隔离 Docker 沙箱里写代码、跑 Python 与 Linux 命令（rg/jq/unzip…）、读写沙箱内文件（/work），只读访问用户工作区（/workspace）；用于计算、数据分析与绘图、批量处理文件。不能联网，不能修改用户工作区（新建/修改 workspace 文件一律找 FileAgent）。需要绘图要明确要求。”
- `code_agent.PROMPT`（保留 WORKER_BASE_PROMPT 前缀与“你是 CodeAgent”）必须写清：
  1. 全部使用绝对路径；主管若写成 `workspace/xxx`，要改成 `/workspace/xxx`。
  2. `/workspace` 只读镜像（路径与其他专家一致）；`/work` 可读写（cwd，会话内保留，沙箱重建后清空）；`/work/outputs/` 是交付箱：放进去的图片会自动发给用户、同一文件每个版本只发一次；`/skills/` 只能用 `read_file` 读（`limit=1000`），shell 里看不到。
  3. 工具决策表：
     - 一次性 Python（计算/分析/绘图）→ `execute_python_code`（不要写 `python -c`）；
     - Shell 命令（ls/rg/jq/unzip/zip/file/tree/wc/head…）→ `execute`：每次都是独立 shell，`cd`/`export` 不保留，多条命令用 `&&`，默认 60s，可用 timeout 参数至多 300s；
     - 多文件/多步脚本 → `write_file` 到 `/work/xxx.py` → `execute("python3 /work/xxx.py")` → 小改用 `edit_file` → 重跑，不要整份重写；
     - 读取/浏览 → `ls/glob/grep/read_file`；`grep` 是字面匹配，正则用 `execute("rg …")`；PDF/Word 等富文档读不了，汇报由 FileAgent 处理。
  4. 禁止：写 `/workspace`、pip install（无网络）、对 `/` 或整个 `/workspace` 递归搜索（先 ls 缩小范围，限定子目录与文件类型）。
  5. 汇报格式沿用 WORKER_BASE_PROMPT，且不向用户暴露沙箱内部路径。
- `prompts.py`（MAIN_AGENT_PROMPT）更新 CodeAgent 委派规则：
  - 要“算、统计、绘图、批量处理文件、跑命令”→ CodeAgent；仅仅“看/找/读 workspace 文件”→ FileAgent（它还负责富文档转换、写入需授权）。
  - CodeAgent 对 workspace 只读，沙箱里的产物（图片除外）不会出现在 workspace。用户要把结果落到 workspace：小文本让 CodeAgent 在汇报里原文给出，再交 FileAgent 写入（需授权）；大文件/二进制暂不支持，如实告知。
  - 同一会话里可让 CodeAgent 接着上次的 /work 中间文件继续。
- `file_agent.DESCRIPTION` 里“唯一能访问本地磁盘”的措辞改成“唯一能修改本地磁盘（workspace）的专家，也负责富文档读取”，避免与 CodeAgent 的只读访问冲突。

**A7 DockerSandbox 路径提示**：`upload_files` 保持只允许 `/work`、`/tmp`（BaseSandbox.edit 会把临时文件写到 `/tmp/.deepagents_edit_*`），但 error 字符串改成自解释，例如 `permission_denied：/workspace 是只读挂载，请写到 /work/ 下`、`invalid_path：只能写入 /work/ 或 /tmp/`（BaseSandbox 会把它原样拼进给模型的错误信息）。同步更新 smoke_sandbox 里按精确字符串断言的地方（改为 startswith）。

## 3. 其他文件
- config.py：新增 `SANDBOX_MAX_TIMEOUT_SECONDS=300`、`SANDBOX_EXECUTE_MAX_OUTPUT_CHARS=2000000`；`SANDBOX_IMAGE_VERSION="3"`；CodeAgent 预算 15。`.env.example` 补可选项。
- skills/：matplotlib-chart 与 data-analysis 按新环境修订（仍 ≤100 行，**水印规则原样保留**）：
  - 去掉“每次执行前清空 /work/outputs”的说法（含 `code_tools.execute_python_code` 的 docstring），改为“图片保存到 /work/outputs/ 即自动发送，同一文件每个版本只发一次”；
  - 写明两种跑法：execute_python_code，或 write_file 脚本 + execute；
  - data-analysis 补充可先用 `execute` 的 `file -i`、`head -c`、`wc -l` 判断编码与规模。
  - smoke_skills 的关键字断言相应更新。
- README：§2.3、§2.8（预算）、§2.9 重写。现在 CodeAgent 是工作台，写清它与 FileAgent 的边界；沙箱是临时工作台（空闲 TTL/重启即清空）；`execute` 每次独立 shell；v1 仍不导出非图片产物；改镜像要重建（版本 3）。现有 §2.9 里的 `CompositeBackend(primary=…, secondary=…)` 参数名是错的，真实签名是 `CompositeBackend(default, routes, *, artifacts_root)`。

## 4. 约束与已验证事实
1. 分层：L1 sandbox 不得 import app.runtime / app.tools / app.agents；L4 code_agent 可以 import L3 code_tools 与 L1 sandbox。code_agent.py 顶层不得 import app.infra.models（与 file_agent 一致）。
2. BaseSandbox.write 会覆盖已存在文件；它的文件操作依赖容器里有 `python3` 与 GNU `grep`（镜像已有）。
3. 构建主 Agent、加载 Skills 都不得需要 Docker；无 Docker 时文件工具应返回友好错误，不得崩溃。
4. 只在 `execute` 之后做一次产物扫描（多一次 exec）；read_file/ls/glob/grep/write_file/edit_file 之后不要做。
5. CI：ruff F821/F811/F401；`uv sync --locked`，不新增宿主依赖；离线测试不得依赖 Docker。

## 5. 测试
- **修改**：
  - `smoke_architecture.test_e`：去掉 CodeAgent 的 ExpertSpec 断言和 `_DECLARATIVE_SPECS` 预期里的 CodeAgent，改为校验 `build_code_agent_runnable`、`SANDBOX_FS_TOOLS`（无 delete）存在。
  - `smoke_orchestrator`：EXPECTED_TOOLS 去掉 CodeAgent；CodeAgent 与 File/Desktop 一起断言 `"runnable" in sub`；test_d 的预算断言循环去掉 CodeAgent。
  - `smoke_sandbox` / `live_sandbox`：版本号，以及超时语义（超时后容器保留、/work 文件仍在）。
- **新增（离线，smoke_sandbox.py）**：
  - a. P0-1 回归（list 断言）。
  - b. P0-2：FakeDocker 抛异常/镜像缺失时 execute/upload/download 返回友好结果、不抛。
  - c. P0-3：BaseSandbox 真实 read 输出（>8000 字符 JSON）经 `execute()` 不被截断。
  - d. P0-4：超时保留容器；仅看门狗路径才重建；137 且耗时 < T 的提示。
  - e. P0-5：客户端被缓存；模型传入的 timeout 被 clamp。
  - f. `collect_new_outputs`：新增→投递一次；再次调用为空；文件被改写→再投递；超上限的被记为已见；`run_python` 与中间件共享指纹、不重复投递。
  - g. CodeAgent 装配，用内存版 BaseSandbox 替身（`execute` 返回 `ExecuteResponse(output="", exit_code=0)`，`upload_files` 存字典，记录所有调用）加 ScriptedModel：
     - 绑定给模型的工具集 = 7 个 FS 工具 + execute_python_code，无 delete；
     - system prompt 含 matplotlib-chart 与 data-analysis；
     - `write_file("/work/a.py")` 路由到沙箱；
     - `read_file("/skills/…")` 读到宿主文件，且沙箱未收到任何含 `/skills` 的调用；
     - 模型传入的 `execute` timeout 被拦截或转发正确。
  - h. 中间件：同步与异步两条路径都能投递并改写 ToolMessage（content 变为 text+image_url 列表，artifact/status 保留）；大输出被转存时也能投递；对无 `collect_new_outputs` 的替身透传。
  - i. smoke_budget：CodeAgent 软停止、硬停止、生产递归上限下失控。
- **新增（真 Docker，live_sandbox.py，默认跳过）**：
  - 重建镜像后 `rg/jq/unzip/zip/file/tree/ps/pkill` 可用。
  - 对真实 DockerSandbox 依次 `write → read → edit → ls → glob → grep`（内容含单引号、双引号、中文）全部成功。
  - `execute` 的 bash 语法：`[[ ]]`、花括号展开、`$(( ))`、含引号的管道。
  - `execute("sleep 30", timeout=3)` 返回 124，`/work` 文件仍在，容器 ID 不变。
  - `execute("sleep 95", timeout=120)` 不被读超时误杀（慢测试，用 `--slow` 开关）。
  - 用 execute 跑绘图脚本后图片只投递一次，改写后再投递，与 execute_python_code 不重复。
  - `write` 到 `/workspace` 得到自解释错误。
  - `seq 1 200000` 大输出不崩。
  - Docker 不可用时文件工具返回友好文本。
  - 再加一条端到端脚本：ScriptedModel + 真 DockerSandbox + build_code_agent_runnable（write_file → execute → edit_file → execute → read_file）。

## 6. 验收
`uv run python tests/run_smoke.py` 全绿；`uvx ruff check --select F821,F811,F401 app plugins scripts tests bot.py main.py mcp_servers` 零报错；有 Docker 时 `uv run python tests/live_sandbox.py --live`（含新增项，慢测试单独说明）通过。

## 7. 不做
Jupyter 式变量持久化；跨 `execute` 保留 cd/环境变量的“真终端会话”；非图片产物导出到宿主；开放网络；给沙箱工具加 HITL；`delete` 工具；`write_todos`；修改主图 HarnessProfile、FileAgent、DesktopAgent 与 HITL 逻辑。

## 8. 汇报格式
改动文件清单；P0-1…P0-6 与 A1…A7 逐条“已落地 / 偏离及原因”（尤其预算 15 的复测结果、ToolMessage 多模态在你环境里的表现）；离线与 ruff 结果；真 Docker 每一项的实测结果（无法运行的明说“未验证”）；遗留问题。
