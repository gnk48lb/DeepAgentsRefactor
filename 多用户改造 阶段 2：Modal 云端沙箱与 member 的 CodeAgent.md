# 多用户改造 阶段 2：Modal 云端沙箱与 member 的 CodeAgent

Nov 19, 2026 · @big white dog

## 0. 背景与目标

用法：整篇复制给 Claude Code，先用 plan 模式出计划，再按 M0 → M7 的顺序做。前提：阶段 1 已完成，当前 HEAD 是 `a88d4f5e788502e749f11e0d4588e54ab4cbff51`（已含多用户改造与 `/stop`），`.env` 里已有 `OWNER_USER_IDS`。

你要改造的是 DeepAgentsRefactor（GNK48-Agent，基于 deepagents + LangGraph 的多智能体 QQ 机器人，跑在我的 Windows 电脑上；官方机器人只有好友私聊，好友上限 20 人）。动手前先读完：`README.md`（§2.1 分层规则、§2.4 多用户模式、§2.9 代码沙箱设计取舍、§2.10 宿主机命令与沙箱的分工）、`tests/smoke_architecture.py`（它在机械地强制分层与导入规则），以及后文各里程碑涉及的源文件。

**现状（阶段 1 之后）**

- owner（我：QQ 主号 + 本地 CLI）与 member（其他好友）在编译期就是两张主图，共用一个 checkpointer；请求上下文 `RequestContext(user_id, session_id, is_owner)` 放在 ContextVar 里（`app/utils/context.py`），失败即关闭。
- owner 图有 7 个专家，其中 CodeAgent 用本机 Docker 沙箱；member 图只有 KnowledgeAgent、MediaAgent、MapAgent，没有任何代码执行能力，也绝不能落到本机 Docker 上。
- 会话库 schema v2、按用户的 Milvus 记忆集合、按用户的聊天锁、`MAX_CONCURRENT_MEMBER_TURNS`、每用户 `/stop` 都已就位。16 个离线冒烟套件全绿，`MAIN_AGENT_PROMPT` 是稳定基线。

**本阶段目标**

给 member 加上云端 CodeAgent：在 Modal Sandbox 里跑 Python 与 shell 命令，有公网，功能尽量满。每个 `(user_id, session_id)` 一个隔离沙箱，惰性创建，用完靠空闲回收；图片产物沿用 `/work/outputs/` 的约定，由系统自动发给用户。owner 的 CodeAgent 继续用本机 Docker，行为完全不变。

能力矩阵：

| 专家 | owner 图 | member 图（本阶段） |
| --- | --- | --- |
| KnowledgeAgent、MediaAgent、MapAgent | 有 | 有 |
| CodeAgent | 有（本机 Docker，不变） | 有（Modal 云端沙箱；仅在 Modal 可用时） |
| FileAgent、BrowserAgent、DesktopAgent | 有 | 无 |

**本阶段不做**：给 member 任何本机资源（Docker、`/workspace`、宿主机命令、桌面、浏览器）；QQ 文件上传到沙箱；非图片产物回传；GPU、持久卷、端口隧道、Modal Functions 部署；每用户次数或额度限制；任何改变 owner 现有行为的重构；除 `modal` 之外的新依赖。

**成本与安全背景**：Modal 工作区的预算上限与 $0 超额上限在 Modal 控制台侧设置，代码里不做额度逻辑，只做资源护栏（见 §1）。Modal 凭证只存在宿主机 `.env`，绝不进入沙箱。

## 1. 已定决策与硬约束

### 1.1 已定决策

这些是我已经定下的，不要重新讨论；如果发现它们与仓库里的实际代码冲突，在最终报告里指出，不要自行改方向。

1. **后端与凭证**。只用 Modal Sandbox（`modal>=1.6.4`，工作区 `dabaigou8964`）。`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` 只放宿主机 `.env`，由 SDK 自己从环境变量读取（`app/config.py` 已经 `load_dotenv`）。代码里只允许存一个布尔量 `config.MODAL_TOKENS_PRESENT`，不读取、不保存、不打印这两个值，也不写进日志、异常文本、审计日志或沙箱环境。
2. **编译期门控，永不回退 Docker**。member 图是否带 CodeAgent，在编译期由静态函数 `member_code_status() -> (bool, reason)` 决定，不联网。不可用时 member 图与阶段 1 逐字等价。member 路径上任何情况下都不得触达 Docker（`get_sandbox`、`DockerSandbox`、`reset_sandbox`、`interrupt_sandbox`、`shutdown_sandbox`、`set_docker_client`），也不得触达宿主机命令（`host_shell`）。
3. **粒度与生命周期**。一个 `(user_id, session_id)` 一个沙箱，惰性创建（第一次真的要执行时才创建）。`/new`、`/switch` 不终止沙箱；`/reset` 终止当前会话的沙箱；owner 执行 `/block` 时终止该用户的全部沙箱；进程关闭时终止全部。闲置的沙箱由 Modal 按 `idle_timeout` 自己回收。
4. **资源护栏**，全部走 `config`，默认值如下：CPU `(0.25, 2.0)`；内存 `(512, 2048)` MiB；空闲超时 600 s；最长存活 3600 s；每用户同时最多 2 个、全局最多 8 个（满了先 LRU 回收空闲的，全部在忙才拒绝）；单次 exec 默认 120 s、上限 300 s；创建沙箱超时 120 s；回合上限 `AGENT_TURN_TIMEOUT_SECONDS = 600` 不变。**不做**应用层的按用户额度或次数限制；Modal 控制台里的预算上限与 $0 超额限制是代码之外的最后一道兜底。
5. **网络默认开放**（`MODAL_BLOCK_NETWORK=false`，可配置关闭）。不挂 Secret、Volume、端口、PTY；沙箱里不放任何凭证。传给沙箱的环境变量只有白名单：`PYTHONUNBUFFERED=1`、`PYTHONIOENCODING=utf-8`、`LANG=C.UTF-8`、`HOME=/work`。
6. **熔断**。连续 3 次创建沙箱失败后，60 s 内直接拒绝新的创建并给出友好提示；任何一次成功即清零。
7. **审计日志** `storage/modal_audit.log`（`RotatingFileHandler`，5 MB × 3）：记录 创建 / 执行 / 终止 / 拒绝 / 熔断 事件；用户与会话 ID 一律哈希；命令截断到 500 字符；不记录输出内容，不记录任何凭证。
8. **标签**。每个沙箱打 tags：`svc=gnk48`、`role=member-code`、`u=sha256(user_id)[:12]`、`s=sha256(session_id)[:8]`，用于人工排查与清理，不含明文 ID。
9. **镜像**。命名镜像 `gnk48-python-ds:v1`，由我手动运行 `scripts/build_modal_image.py` 构建并发布一次；运行时只做 `Image.from_name(...)`，**绝不在运行时构建镜像**。沙箱里以 root 运行，不建非 root 用户（隔离靠 Modal 沙箱本身）。
10. **不做启动时孤儿清理**。同一台机器上可能同时跑着另一个进程（比如我在另一个终端开了 CLI），启动时杀「看起来没人管」的沙箱会误杀它的沙箱；Modal 会按 `idle_timeout` / `timeout` 自己回收。只提供手动脚本 `scripts/manage_modal_sandboxes.py`：`list`、`kill --all`、`kill --user <hash>`。
11. **`/skills/` 对 member 只读**。member 的技能目录物理分离为 `skills_member/`，并用只读后端包装（§7.3）。这是阶段 1 之后新发现的一个洞：member 的 CodeAgent 一旦有 `write_file` / `edit_file`，而 `/skills/` 路由到宿主机真实目录，member 就能改写 owner 的 Agent 会读取的技能文件。

### 1.2 项目硬规则

违反任何一条都算未完成。

- **分层**由 `tests/smoke_architecture.py` 机械强制。L0（`config`、`utils`）< L1（`runtime`、`infra`）< L2（`knowledge`、`memory`）< L3（`tools`）< L4（`agents`）< L5（`services`），只能向下依赖；同层的不同子包互不导入；`app/agents/experts/*` 之间互不导入（所以 member 的 CodeAgent 构建函数必须写在 `code_agent.py` 里，不能借道别的专家）；`tools/*_tools.py` 互不导入；只允许 `from app import config`；禁止相对导入；子包的 `__init__.py` 保持为空。
- **`modal` 只能惰性导入**，即在函数体内 `import modal`。架构测试 F（轻量导入）与 G（不带任何密钥、用 `walk_packages` 导入全部 `app` 模块）要求没有 `modal`、没有密钥时全部模块也能被 import。
- **阻塞的 Modal 调用绝不在事件循环线程上执行**；异步路径一律 `asyncio.to_thread`。
- **`MAIN_AGENT_PROMPT` 一个字节都不能变**；owner 路径的行为不变（含 owner 的 Docker CodeAgent、`/stop`、`host_shell` 清理）。
- **member 路径上不得出现**：`/workspace`、宿主机命令、Docker、FileAgent、DesktopAgent、BrowserAgent。
- **CI 命令必须保持绿**：`uv sync --locked`；`uvx ruff check --select F821,F811,F401 app plugins scripts tests bot.py main.py mcp_servers`；`uv run python tests/run_smoke.py`。CI 没有任何令牌，所有测试必须完全离线；`pyproject.toml` 与 `uv.lock` 必须一起修改。
- **既有测试只能适配，不能削弱**。不得为了让测试通过而删除或放宽断言；确需改动的断言，在最终报告里逐条列出原因。
- 日志、异常文本、审计日志、测试输出里不得出现令牌值；任何可能带出令牌的异常，先脱敏再记录。

## 2. 工作方式与边界

**开始之前**

- 先进 plan 模式。读完 §0 指向的文件和 M0 要求核对的源码，再出计划；计划里按里程碑写出：要改的文件、要新增的测试、你认为最容易出错的点。
- 一次只做一个里程碑（M0 → M7）。每个里程碑结束时跑一遍 §1.2 里的 ruff 命令和该里程碑相关的 smoke 套件；全部做完后再跑 `uv run python tests/run_smoke.py` 全套。
- 每个里程碑开头先读一遍它要改的文件的**当前内容**，不要凭本文的转述改代码：本文给的是契约与意图，行号、字段名以仓库为准。

**你可以做**：读代码；改 §4–§10 列出的文件、新增文件；`uv add 'modal>=1.6.4'`（会改 `pyproject.toml` 与 `uv.lock`）；运行离线测试与 ruff。

**你不要做**（硬性）

- 不要读取、打印、复制 `.env` 或任何 `.env.*`，也不要把它们的内容写进任何文件、日志或报告。`.env.example` 可以改。
- 不要读取、修改、删除 `storage/` 与 `data/` 下的任何文件（里面是我的会话库与记忆）。测试一律用临时目录。
- 不要 `git commit`、`git push`、`git stash`、`git reset`、`git checkout -- <路径>`。所有改动留在工作区，由我审阅后提交。
- 不要调用真实的 Modal：任何测试、脚本、临时试探都不得带着真实令牌联网。涉及 Modal 的验证一律用 `tests/_fake_modal.py`。
- 不要运行 `scripts/build_modal_image.py`（它会构建并发布镜像、产生费用）和 `tests/live_modal.py --live`（真机验收），这两个由我来跑。你只负责写好，并保证不带 `--live` / 带 `--help` 时完全离线可用。
- 不要碰 owner 的 Docker 沙箱路径：不要启动、停止或调试 Docker；不要改 `deploy/sandbox/` 里的 Dockerfile 与 `matplotlibrc`（`deploy/sandbox/requirements.txt` 是依赖版本的唯一来源，只读引用）。
- 不要在 `site-packages` 里给 SDK 或 deepagents 打补丁；不要用私有属性绕开公开接口，除非本文明确要求。
- 不要扩大范围：本文没写的重构、改名、顺手优化，一律不做。

**本文与现实不一致时**：以你在仓库里、已安装的包里实际核对到的为准，按「最接近本文意图」的方式实现，并在最终报告的「与本文的差异」一栏逐条写清楚（事实、你的处理、影响）。没有验证过的，写进「未验证」，不要把猜测写成事实放进代码注释或 README。

**沟通**：不要中途停下来问我，除非遇到不可逆、且确实有两种合理做法的决定；其余一律按 §1 的决策推进。每个里程碑完成时，用两三句话说明做了什么、有没有偏离本文，然后直接进入下一个里程碑。

## 3. M0 前置核对（只读，不改代码）

目的：确认仓库状态与本文引用的接口一致，把本文里「读源码确认过的」和「推断的」区分开。结果写进最终报告的「M0 核对结果」一栏；M0 本身不产生代码改动，核对完直接进入 M1，不必等我确认。

### 3.1 仓库状态与基线

- `git rev-parse HEAD` 应为 `a88d4f5e788502e749f11e0d4588e54ab4cbff51`，`git status --short` 应为空。不一致就先停下，在报告里说明，不要硬做。
- 跑一遍基线：`uv sync --locked`、§1.2 的 ruff 命令、`uv run python tests/run_smoke.py`。应当是 16 个套件全绿；基线不绿时只记录失败项，不要在本任务里顺手修。

### 3.2 本仓库的接口（读源码确认；行号以仓库为准）

逐项在报告里写一句「已确认」或「有差异」：

1. `app/infra/sandbox.py`：数据类 `RunResult`（`stdout, stderr, exit_code, timed_out, duration, images, other_files, work_listing, notes`）、`OutputsResult`（`images, other_files, notes`）、`SandboxImage(name, data)`；`StreamingBuffer(max_chars)` 的 `feed` / `get_result`（**不是线程安全的**）；`SandboxUnavailable`。`DockerSandbox` 的 `run_python(code, *, timeout=None)`、`execute(command, *, timeout=None) -> ExecuteResponse`、`upload_files`、`download_files`、`collect_new_outputs`（用「路径 → (大小, mtime)」指纹只取新增或修改的产物，非图片只报告不导出）、`interrupt`（不取锁、永不抛异常）、`reset` / `close`；模块级的 `get_sandbox`、`reset_sandbox`、`interrupt_sandbox`、`shutdown_sandbox`、`set_docker_client` 与 `atexit.register(shutdown_sandbox)`。
2. `app/tools/code_tools.py`：`execute_python_code` 写死了 `sandbox.get_sandbox().run_python(code)`，说明文本写的是 Docker / 无网络 / 60 秒；`publish_images`；`TOOLS`。
3. `app/agents/experts/code_agent.py`：`NAME`、`DESCRIPTION`、`PROMPT`、`SANDBOX_FS_TOOLS`；`SandboxOutputsMiddleware`（只在 `execute` 工具调用之后收集产物）；`build_code_agent_runnable` 里的 `CompositeBackend(default=sb, routes={SKILLS_ROUTE: FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True), '/_artifacts/': StateBackend()}, artifacts_root='/_artifacts')`。
4. `app/agents/skills.py`；`app/agents/orchestrator.py`（`_member_subagents`、`_assert_member_graph_safe`、`_MEMBER_FORBIDDEN_TOOL_NAMES` 共 13 个、`_build_member_agent`、`# STAGE2` 标记）；`app/agents/prompts.py`（`MEMBER_MAIN_AGENT_PROMPT`、`WORKER_BASE_PROMPT`）。
5. `app/services/chat_service.py`：`init_service`、`shutdown_service`、`_cmd_reset`、`_cmd_whoami`、`_cmd_block`、`_kill_side_effects`、`_stop_current_turn`、`wake` 里的 `# STAGE2` 标记与现有逻辑。
6. 测试：`tests/smoke_architecture.py`（分层与导入规则）、`tests/run_smoke.py`（套件清单）、`tests/_support.py`（`owner_context` / `member_context`）、`tests/smoke_orchestrator.py`、`tests/smoke_chat_service.py`、`tests/smoke_skills.py`、`tests/smoke_sandbox.py`（里面的 FakeDocker 是写 FakeModal 的参照）。

### 3.3 deepagents 的接口（用 `uv run python` + `inspect` 核对已安装版本，不要只信文档）

- `CompositeBackend.__init__(default, routes, *, artifacts_root)`，实例属性 `default`、`routes`、`sorted_routes`、`artifacts_root`（§7.4 的启动期断言会读它们）。
- `BaseSandbox` 的抽象成员恰为 `execute(command, *, timeout=None)`、`id`（property）、`upload_files`、`download_files`；`ls / read / write / edit / glob / grep / delete` 都是通过 `execute()` 里的 `python3 -c ...` 实现的（所以沙箱镜像里必须有 `python3`）；`write()` 先做一次 `os.makedirs` 预检再走 `upload_files`。
- `FilesystemBackend(root_dir=..., virtual_mode=True)` 的 `write / awrite / edit / aedit / delete / adelete / upload_files / aupload_files` 的签名与返回类型（`WriteResult` / `EditResult` / `DeleteResult` 带 `error=` 字段，`FileUploadResponse(path, error=)`）。它**没有任何只读保护**——这正是 §7.3 要补的洞。
- `FilesystemMiddleware(backend, tools=..., max_execute_timeout=...)`：`execute` 工具收到的 `timeout` 超过 `max_execute_timeout` 会直接报错；后端的 `execute` 必须带 `timeout` 形参（`execute_accepts_timeout` 会检查）。

### 3.4 Modal SDK 事实表（M1 的 `uv add` 之后补做，M0 此刻还没有装）

下面是我读 `modal` 1.6.4 源码确认的事实。请用 `inspect.signature` 或直接读已安装包的源码逐条核对；有出入就以实际为准、调整 M2 的实现，并写进报告。

| 主题 | 事实 |
| --- | --- |
| 创建沙箱 | `modal.Sandbox.create(*args, app=, name=, tags=, image=, env=, timeout=, idle_timeout=, workdir=, cpu=, memory=, block_network=, ...)`。`*args` 是入口 CMD（覆盖镜像的 CMD），我们显式传 `'sleep', 'infinity'`；`timeout`（最长存活）默认只有 300 秒，必须显式传；`cpu` / `memory` 可以传 `(请求, 上限)`，内存单位 MiB；`app` 来自 `modal.App.lookup(name, create_if_missing=True)`。 |
| 执行命令 | `sb.exec(*args, stdout=, stderr=, timeout=, workdir=, env=, text=, bufsize=)` 返回 `ContainerProcess`；用 `text=False` 取字节流；`p.stdout` / `p.stderr` 可迭代；`p.wait()` 返回退出码。**`p.returncode` 在 `wait()` 之前访问会抛 `InvalidError`，不要依赖它。** |
| exec 超时 | exec 的 `timeout` 同时是服务端与客户端的截止时间；到点后 `wait()` / `poll()` 返回 **-1**，输出流静默结束，**不会抛 `ExecTimeoutError`**。所以主超时靠容器里的 `timeout -k 3 N`（退出码 124），`exec(timeout=N+12)` 只是兜底，看到 -1 就说明兜底触发了。 |
| 终止进程 | `ContainerProcess` 没有 `kill()`。中止只能再 exec 一个命令去杀进程，或 `sb.terminate()`。`terminate()` 之后输出流的行为我没有验证过，读流的线程要宽泛捕获异常。 |
| 沙箱状态 | `sb.poll()` 沙箱还活着时返回 `None`；`sb.terminate(wait=False)`；`sb.object_id`；`sb.set_tags` / `sb.get_tags`；`modal.Sandbox.list(app_id=..., tags=...)`。 |
| 文件系统 | `sb.filesystem.write_bytes(data, remote_path)`、`read_bytes(remote_path)`、`make_directory(remote_path, create_parents=True)`、`remove(remote_path, recursive=False)`、`list_files`、`stat`；只接受绝对路径；超大文件抛 `SandboxFilesystemFileTooLargeError`。1.6.0 起旧的 `Sandbox.open / ls / mkdir / rm / watch` 已移除，不要用。 |
| 异常 | `modal.exception` 里有 `Error`、`AuthError`、`ResourceExhaustedError`、`NotFoundError`、`ConnectionError`、`ServiceError`、`SandboxTerminatedError`、`SandboxTimeoutError`、`ExecTimeoutError`、`InvalidError`、`OutputExpiredError`、`SandboxFilesystemError` 及其 NotFound / IsADirectory / FileTooLarge / Permission 子类。其中 `ConnectionError`、`TimeoutError` 与 Python 内建同名：不要 `from modal.exception import *`，也不要和内建异常混在同一个 `except` 里。 |
| 镜像 | `modal.Image.debian_slim(python_version=...)`、`.apt_install`、`.pip_install` / `.uv_pip_install`、`.run_commands`、`.env({...})`、`.workdir`、`.add_local_file(local, remote, copy=True)`；先 `image.build(app)` 再 `image.publish(name)`（`name[:tag]`，缺省 `:latest`）；运行时 `modal.Image.from_name(name)` 是惰性的，镜像不存在只能在 `Sandbox.create` 失败时才知道（NotFound 一类异常）。 |
| 凭证 | SDK 自己从环境变量 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` 读取。`modal.Client.from_credentials(...)` 存在，但本项目不用——不碰令牌值。 |
| 阻塞 | 以上调用全是阻塞的同步接口。在事件循环线程里调用会卡死整个机器人，必须放线程里。 |
| 输出保留 | exec 的输出只在服务端保留约 10 分钟，过期读流会抛 `OutputExpiredError`；我们的单次 exec 上限 300 秒，正常用不到，但读流线程要能容忍它。 |

### 3.5 M0 的产出

一张「项 / 已确认或差异 / 你的处理」的简表，进最终报告。只有当差异会改变本文某条已定决策（§1.1）时才需要在报告里单独标红，其余差异按「最接近本文意图」自行处理。

## 4. M1 依赖与配置

顺序：① `uv add 'modal>=1.6.4'`（会改 `pyproject.toml` 与 `uv.lock`；之后 `uv sync --locked` 必须通过；立刻回到 §3.4 补做 SDK 核对）→ ② 改 `app/config.py` → ③ 在 `app/infra/modal_sandbox.py` 里先放下 `member_code_status()`（其余内容在 M2）。

**导出依赖文件**：做完 ① 之后马上重新生成 `requirements.txt`（给不用 uv 的 pip 环境用；文件头写着由 `uv export` 生成、请勿手改）：`uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`。然后只读地看 `git diff -- requirements.txt`：应该只多出 `modal` 与它新带来的传递依赖；如果还有别的行变了（版本漂移），把原因写进最终报告，不要手工去改。`deploy/sandbox/requirements.txt` 是 Docker 沙箱镜像自己的依赖文件，不在这一步里，一个字都不要动。

### 4.1 `app/config.py` 新增常量

放在现有 `SANDBOX_*` 常量之后、单独一段“Modal 云端沙箱（member 的 CodeAgent）”注释下。解析风格沿用文件里的写法（`int(os.getenv(...))`、布尔用 `.lower() in ('1', 'true', 'yes')`），并对非法值做下表的夹紧，不要让一个写错的数字把 import 搞崩。

| 常量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODAL_DISABLED` | `false` | 总开关。`true` 时 member 图不带 CodeAgent（阶段 1 行为） |
| `MODAL_TOKENS_PRESENT` | 派生布尔 | `bool(os.getenv('MODAL_TOKEN_ID')) and bool(os.getenv('MODAL_TOKEN_SECRET'))`；**只存布尔值**。import 时计算（在 `load_dotenv` 之后） |
| `MODAL_APP_NAME` | `gnk48-code-sandbox` | `modal.App.lookup(..., create_if_missing=True)` 用。去掉首尾空白，空串回退默认 |
| `MODAL_IMAGE_NAME` | `gnk48-python-ds:v1` | 运行时 `Image.from_name`；也是构建脚本发布的名字。空串回退默认 |
| `MODAL_CPU_REQUEST` / `MODAL_CPU_LIMIT` | `0.25` / `2.0` | float。request 大于 limit 时夹到 limit |
| `MODAL_MEMORY_MB_REQUEST` / `MODAL_MEMORY_MB_LIMIT` | `512` / `2048` | int，单位 MiB。request 大于 limit 时夹到 limit；limit 至少 256 |
| `MODAL_IDLE_TIMEOUT_SECONDS` | `600` | 空闲超时，至少 60 |
| `MODAL_MAX_LIFETIME_SECONDS` | `3600` | 沙箱最长存活，至少 `MODAL_EXEC_MAX_TIMEOUT_SECONDS + 60` |
| `MODAL_EXEC_TIMEOUT_SECONDS` | `120` | 单次 exec 默认超时，至少 1，不超过上限 |
| `MODAL_EXEC_MAX_TIMEOUT_SECONDS` | `300` | 单次 exec 上限（也是 member 的 `FilesystemMiddleware(max_execute_timeout=...)`） |
| `MODAL_STARTUP_TIMEOUT_SECONDS` | `120` | 创建沙箱（含冷启动）的整体超时 |
| `MODAL_MAX_SANDBOXES_PER_USER` | `2` | 至少 1 |
| `MODAL_MAX_SANDBOXES_TOTAL` | `8` | 至少等于每用户上限 |
| `MODAL_BLOCK_NETWORK` | `false` | `true` 时沙箱断网 |
| `MODAL_AUDIT_LOG` | `storage/modal_audit.log` | 用文件里已有的 `_resolve_path`，和 `SESSIONS_DB_PATH` 一样落在被 git 忽略的 `storage/` |
| `MODAL_INTERRUPT_MODE` | `soft` | 中止方式（/stop、卡住的命令）：`soft` 只杀沙箱里我们自己起的命令进程、保留 `/work`，失败自动回退到 `terminate`；`terminate` 直接终止沙箱（`/work` 丢失）。去掉首尾空白并转小写后只认这两个值，其余按 `soft` 处理。见 §5.3 |

补充规则：

- 输出上限、图片上限沿用现有的 `SANDBOX_MAX_STDOUT_CHARS`、`SANDBOX_MAX_STDERR_CHARS`、`SANDBOX_EXECUTE_MAX_OUTPUT_CHARS`、`SANDBOX_MAX_IMAGES`、`SANDBOX_MAX_IMAGE_MB`、`SANDBOX_USER_IMAGE_MAX_SIDE`、`SANDBOX_MODEL_IMAGE_MAX_SIDE`，**不要**为 Modal 另设一套。现有 `DOCKER_*`、`SANDBOX_*` 常量一个都不改。
- `AGENT_TURN_TIMEOUT_SECONDS = 600` 不变；单次 exec 上限 300 秒小于它，一轮里多次执行的总和仍由这 600 秒兜底。
- `config.py` 里不得 `import modal`，也不得有任何网络或文件 IO（除了已有的 `load_dotenv`）。

### 4.2 `member_code_status() -> tuple[bool, str]`

放在 `app/infra/modal_sandbox.py`（L1）。纯静态判断，**不联网、不 import `modal`**，每次调用都现读 `config.*`（测试里直接 monkeypatch `config.MODAL_DISABLED` / `config.MODAL_TOKENS_PRESENT` 就能改变结果）：

1. `config.MODAL_DISABLED` 为真 → `(False, 'MODAL_DISABLED=true')`。
2. `not config.MODAL_TOKENS_PRESENT` → `(False, '未配置 MODAL_TOKEN_ID / MODAL_TOKEN_SECRET')`。
3. `importlib.util.find_spec('modal') is None` → `(False, '未安装 modal 包')`。
4. `importlib.metadata.version('modal')` 用正则 `^(\d+)\.(\d+)` 取前两段，低于 1.6 → `(False, 'modal 版本过低（需要 >= 1.6）')`；版本号解析不出来也按不可用处理（失败即关闭）。
5. 其余 → `(True, '')`。

返回的原因是给运营者看的中文短句，不得带任何令牌值。它的语义是“**可以编译进 member 图**”，而不是“现在一定能用”：令牌是否有效、镜像是否已发布、Modal 是否可达，都要到第一次真的创建沙箱时才知道，那时的失败按 §5.4 处理（友好文案 + 熔断），不影响机器人的其他功能。

**可测性**：上面的检查必须写成模块属性调用——`importlib.util.find_spec('modal')` 与 `importlib.metadata.version('modal')`，不要写 `from importlib.util import find_spec` 这类直接导入名字的形式，文件里显式 `import importlib.util` 与 `import importlib.metadata`。这样测试里才能 `monkeypatch.setattr(importlib.util, 'find_spec', ...)` 与 `monkeypatch.setattr(importlib.metadata, 'version', ...)` 来改变结果（§9.3 的 B 就这么做）。

## 5. M2 云端沙箱后端（`app/infra/modal_sandbox.py`）

### 5.1 模块边界

- **两个新文件，都在 L1**：`app/infra/modal_sandbox.py`（`ModalSandbox`、`ModalSandboxPool`、`ModalRoutedSandbox`、审计、`member_code_status`、模块级函数）与 `app/infra/modal_image.py`（镜像定义，见 §6）。不要为了篇幅拆包，`sandbox.py` 也是单文件。
- **复用** `app.infra.sandbox` 里的纯数据 / 工具类：`RunResult`、`OutputsResult`、`SandboxImage`、`StreamingBuffer`、`SandboxUnavailable`，这样 `code_tools.publish_images` 与 `SandboxOutputsMiddleware` 不用改。除这几个名字外，`modal_sandbox.py` 与 `modal_image.py` 里**不得出现** `DockerSandbox`、`get_sandbox`、`reset_sandbox`、`interrupt_sandbox`、`shutdown_sandbox`、`set_docker_client` 这些标识符（§9 的 AST 测试会扫）。
- **惰性导入 `modal`**：只在公开函数 `load_modal()` 的函数体里写 `import modal` 语句（整个文件里只有这一处）。它返回 modal 模块，或者 `set_modal_module` 注入的假模块；导入失败就让 `ImportError` 往上抛。构建脚本、管理脚本与 `get_pool()` 一律通过 `load_modal()` 拿 modal（§6.2、§5.8、§5.5）。提供测试钩子 `set_modal_module(mod_or_None)`，注入 FakeModal（传 `None` 恢复真实导入），并同时清空池单例、关闭并重置审计 handler（下一次写审计会按当时的 `config.MODAL_AUDIT_LOG` 重新创建）。
- 依赖方向：只能导入 `app.config`、`app.utils.*`、`app.infra.sandbox`（同包）、deepagents 的 `BaseSandbox` 与 protocol 类型、标准库、PIL。不得导入 L2 及以上。

### 5.2 `ModalSandbox`：一个 `(user_id, session_id)` 的沙箱

包装一个 Modal 沙箱，接口与 `DockerSandbox` 对齐（读 `sandbox.py` 里对应方法的实现再写，语义、提示文案风格尽量一致）：

- `run_python(code, *, timeout=None) -> RunResult`：代码超过 256 KiB 直接拒绝（返回 `exit_code=1` 的 `RunResult`）；把代码写到 `/work/.sandbox/snippet.py`（用 `sb.filesystem.make_directory(..., create_parents=True)` + `write_bytes`）再运行；返回时附带 `collect_new_outputs()` 的图片 / 非图片产物与 `work_listing`（`ls -1 /work` 的前 20 项）。创建沙箱失败时抛 `SandboxUnavailable`（`execute_python_code` 工具已经会把它转成文本）。
- `execute(command, *, timeout=None) -> ExecuteResponse`：**绝不向外抛异常**，创建失败也返回 `ExecuteResponse(output='[云端沙箱不可用] ...', exit_code=1, truncated=False)`。`timeout` 为 `None` 或 ≤ 0 取 `config.MODAL_EXEC_TIMEOUT_SECONDS`，否则 `min(int(timeout), config.MODAL_EXEC_MAX_TIMEOUT_SECONDS)`；`run_python` 的 `timeout` 同理。
- `upload_files(files)` / `download_files(paths)`：逐个文件分别捕获异常、绝不抛（`FileUploadResponse(path, error=)` / `FileDownloadResponse(path, content=, error=)`）。上传路径规则：`posixpath.normpath` 后必须是绝对路径，且 `== '/work'`、`startswith('/work/')`、`== '/tmp'`、`startswith('/tmp/')` 之一（注意不要沿用 Docker 版那个 `startswith('/work')` 会误放 `/workfoo` 的写法）；`/workspace` 开头的返回 `permission_denied：云端沙箱没有 /workspace，请写到 /work/ 下`；其余返回 `invalid_path：只能写入 /work/ 或 /tmp/`；单个文件超过 5 MiB 返回 `file_too_large`（若 SDK 的 `write_bytes` 实际上限更小，取更小值并写进报告）；写入前用 `make_directory(parent, create_parents=True)`。下载的错误码：`file_not_found`、`is_directory`、`file_too_large`、`permission_denied`，其余用脱敏、截断到 200 字的异常文本。
- `collect_new_outputs() -> OutputsResult`：逻辑与 Docker 版一致——用 `exec` 跑 `find /work/outputs -mindepth 1 -type f -printf '%P\t%s\t%T@\0'`，用「路径 → (大小, mtime)」指纹只取新增或修改的文件；图片扩展名 `.png .jpg .jpeg .webp .gif`；单张 ≤ `SANDBOX_MAX_IMAGE_MB`、每次 ≤ `SANDBOX_MAX_IMAGES` 张、像素 ≤ 40M（用 PIL 校验），超限的记 notes（文案沿用 Docker 版）；非图片只报告「非图片产物未导出」不导出；图片字节用 `sb.filesystem.read_bytes('/work/outputs/<rel>')` 读。额外加两条保护：解析 `find` 输出时最多处理 500 条（超出记 note），相对路径仍要拒绝 `..` 与绝对路径。沙箱还没创建时返回空的 `OutputsResult()`，**不得为了扫描产物而创建沙箱**。
- `interrupt() -> bool`、`close() -> None`、`id`（属性）、`alive`（属性）：见 §5.3。

并发规则与 Docker 版相同：每个 `ModalSandbox` 有一把 `RLock`，`run_python` / `execute` / `upload_files` / `download_files` / `collect_new_outputs` 持锁执行（同一沙箱里的并行工具调用被串行化）；`interrupt()` 与 `close()` **不取这把锁**（否则中断要等被中断的命令跑完，等于没中断）。锁里绝不保持池的锁。

### 5.3 命令执行、超时、中止与关闭

**命令包装**（用 list 传参，不经 shell 拼接）：`execute` 用 `['timeout', '-k', '3', str(N), 'bash', '-c', command]`，`run_python` 用 `['timeout', '-k', '3', str(N), 'python', '-u', '/work/.sandbox/snippet.py']`；`sb.exec(*argv, workdir='/work', timeout=N + 12, text=False)`。主超时是容器里的 `timeout`，`exec(timeout=N+12)` 只是客户端兜底（原因见 §3.4：兜底触发时 `wait()` 返回 -1、不抛异常）。

**读流**：stdout 与 stderr 各用一个线程并发读，用增量 UTF-8 解码器（`codecs.getincrementaldecoder('utf-8')(errors='replace')`）处理被分块切开的多字节字符，再喂给 `StreamingBuffer`。`run_python` 用两个独立缓冲（`SANDBOX_MAX_STDOUT_CHARS` / `SANDBOX_MAX_STDERR_CHARS`）；`execute` 用**一个共享缓冲（`SANDBOX_EXECUTE_MAX_OUTPUT_CHARS`）并加锁**（`StreamingBuffer` 不是线程安全的）。每个流设原始字节硬上限（模块级常量 `_STREAM_HARD_CAP_BYTES = 64 * 1024 * 1024`；测试里会把它调小，所以读流代码必须在运行时读这个模块全局，不要把它绑成默认参数）：超过后触发一次 `interrupt()`、结果里加一条「输出量过大，命令已终止」note。读流线程里的任何异常（含 `OutputExpiredError`、沙箱被终止后流的各种怪行为）都要宽泛捕获并转成 note，不得让线程带着异常退出而主线程永远等下去；主线程 `p.wait()` 之后给读流线程最多 5 秒收尾。退出码只用 `wait()` 的返回值，不访问 `returncode`。

**退出码与超时判定**（对照 Docker 版的 P0-4）：

| 退出码 | 含义 | 处理 |
| --- | --- | --- |
| 124，或 137 且耗时 ≥ N | 容器内 `timeout` 触发 | `timed_out=True`，note「命令执行超过 N秒被终止；沙箱状态保留」，然后做一次残留进程清理（同下「软杀」） |
| 137 且耗时 < N | 进程被杀，很可能是内存超限 | note「进程被杀（可能超出内存上限 {MODAL\_MEMORY\_MB\_LIMIT} MiB）」。**真实 Modal 里超内存会只杀进程还是连沙箱一起杀、我没验证过**，由 `tests/live_modal.py` 实测（§10.1） |
| -1 | 客户端兜底 deadline 到点（没有异常，流静默结束） | `timed_out=True`、`exit_code=None`，note「执行超时，云端看门狗终止了该命令」；随后做健康检查（`exec('true', timeout=15)` 退出码 0），失败则 `terminate()` 并标记沙箱已死，note 再加「沙箱已重置，/work 已清空」 |
| exec 抛异常（沙箱被回收 / 终止 / 连接断开） | 命令没跑完 | 标记沙箱已死；note「云端沙箱已被回收（空闲超时或被终止），本次执行未完成；/work 里的文件已丢失，请重试」。**不自动重试**同一条命令（无法确定它有没有已经执行过），让模型自己重试 |

**重建**：每次命令前先检查 `sb.poll() is None`；已死（空闲超时被 Modal 回收 / 上一次标记为死）就惰性重建，并在这次结果里带一条一次性的 note「云端沙箱已重建，/work 里之前的文件已不存在」（类似 Docker 版 `_ensure_container()` 返回的 `init_notes`）；重建时清空 `_seen_outputs`。

**中止与关闭**（`/stop`、卡住的命令、`/reset`）：

- `interrupt() -> bool`：**不取锁、永不抛异常**。沙箱还没创建返回 `False`。按 `config.MODAL_INTERRUPT_MODE` 分两种（该常量见 §4.1 的配置表，默认 `soft`，其它合法值 `terminate`，非法值回退 `soft`）：
  - `soft`（默认，保留 `/work`）：在沙箱里再 `exec` 一段 `python3 -c` 脚本，扫 `/proc`，**只杀掉我们自己的命令包装进程（cmdline 以 `timeout -k 3` 开头的 `timeout`）及其全部后代**（用 `/proc/<pid>/stat` 的 ppid 建树，注意 comm 字段可能含空格与括号，要按最后一个 `)` 切），绝不碰 PID 1、脚本自己、`sleep infinity` 与任何不属于这棵树的进程（它们可能是 Modal 自己的基础设施进程）。等脚本跑完后做一次健康检查（`exec('true', timeout=15)` 退出码 0）；任何一步失败或超时，立即回退到 `terminate`。

    **实现要求**：这段脚本的逻辑不是临时拼出来的字符串，而是 `modal_sandbox.py` 里一个自包含的函数 `_kill_wrapper_trees(proc_root='/proc', *, self_pid=None, kill=None) -> list[int]`：只用标准库，函数体里自己 `import os, signal`，不引用任何模块级名字；`self_pid` 默认 `os.getpid()`，`kill` 默认 `os.kill`（发 SIGKILL），返回被杀的 pid 列表。包装进程的精确判定：`/proc/<pid>/cmdline` 按 NUL 切开后，argv\[0\] 的 basename 是 `timeout` 且 argv\[1:3\] 恰为 `['-k', '3']`；先算出整棵树（包装进程加全部后代）的 pid 集合，再一起杀，名单里永远排除 PID 1 与 `self_pid`。`python3 -c` 的脚本文本由 `inspect.getsource(_kill_wrapper_trees)` 加一条调用语句拼出。这样同一份逻辑既能在真实沙箱里跑，也能在测试里对着伪造的 `/proc` 目录跑（§9.3 的 I）。
  - `terminate`（更简单，`/work` 会丢）：`sb.terminate(wait=False)` 并标记已死，下一次命令惰性重建。
  - **软杀在真实 Modal 里是否安全（会不会把 exec 通道或沙箱弄坏）我没验证过**，由 `tests/live_modal.py` 实测（§10.1）。实现两条路径、默认软杀，结果出来后我用 `MODAL_INTERRUPT_MODE` 切换，不需要改代码。
- `close() -> None`：幂等，不取锁；`sb.terminate(wait=False)`，异常全吞；标记已死、清空 `_seen_outputs`。被它杀掉的正在跑的命令会自己返回一个带「沙箱已被终止」note 的结果，不得卡死。
- `/stop` 走的是池的 `interrupt_session_nowait`（§5.5）：起一个守护线程去跑 `interrupt()`，调用方立即返回。

### 5.4 创建、护栏、熔断与错误文案

**创建**（惰性，在沙箱的锁内、第一次需要时且仅此一次）：

1. 先问熔断（见下）；池里缓存 `app = modal.App.lookup(config.MODAL_APP_NAME, create_if_missing=True)`（线程安全、只查一次）。
2. `modal.Sandbox.create('sleep', 'infinity', app=app, image=modal.Image.from_name(config.MODAL_IMAGE_NAME), env=ENV_WHITELIST, timeout=config.MODAL_MAX_LIFETIME_SECONDS, idle_timeout=config.MODAL_IDLE_TIMEOUT_SECONDS, workdir='/work', cpu=(request, limit), memory=(request, limit), block_network=config.MODAL_BLOCK_NETWORK, tags=...)`。不传 `secrets`、`volumes`、任何端口、`pty`、`name`。`ENV_WHITELIST` 恰为 `PYTHONUNBUFFERED=1`、`PYTHONIOENCODING=utf-8`、`LANG=C.UTF-8`、`HOME=/work`。`tags` 见 §1.1（哈希用 `hashlib.sha256(...).hexdigest()` 取前 12 / 8 位）。
3. 把 `Sandbox.create` 放在一个辅助线程里，用 `config.MODAL_STARTUP_TIMEOUT_SECONDS` 作为 join 超时；超时即视为创建失败。**辅助线程晚于超时才创建成功时，必须立刻把这个迟到的沙箱 `terminate`**（否则它会变成孤儿一直计费到 idle / lifetime 到点）。
4. 创建成功后 `exec('mkdir', '-p', '/work/outputs', '/work/.sandbox')`；记审计 `create_ok`；熔断计数清零。

**护栏**（池里实现，见 §5.5）：每用户 `MODAL_MAX_SANDBOXES_PER_USER`、全局 `MODAL_MAX_SANDBOXES_TOTAL`。

**熔断**：连续 3 次创建失败后进入 60 秒的“断开”状态，期间立即拒绝新的创建（审计 `reject:breaker`）；60 秒后放行一次试探，成功即恢复、失败则再断开 60 秒。状态存在池里，用可注入的时钟（测试里不要真等 60 秒）。已有沙箱上的命令失败**不**计入熔断。

**错误文案分两层**——这是重点，因为工具返回的文字会进入 member 的对话：

- **给模型 / member 看的**（`SandboxUnavailable` 的消息、`ExecuteResponse.output`）一律是通用、不泄露基础设施细节的短句：创建失败 / 熔断用「云端代码沙箱暂时不可用，请稍后再试」；超每用户上限且都在忙用「你同时运行的代码沙箱已达上限（N 个），请等当前任务结束后再试」；超全局上限且都在忙用「云端代码沙箱现在比较繁忙，请稍后再试」。
- **给运营者看的**（控制台 `print` 与审计日志）带分类原因与可操作的提示：`NotFoundError` →「镜像可能没有发布，请运行 scripts/build\_modal\_image.py」；`AuthError` →「凭证无效或已过期，请检查 .env 里的 Modal 令牌」；`ResourceExhaustedError` →「Modal 资源或额度不足」；`ConnectionError` / `ServiceError` / 超时 →「连不上 Modal」；其它 → 类名 + 脱敏、截断到 200 字的文本。
- **脱敏**：所有写进日志 / 控制台 / 返回值的异常文本都先过 `_redact()`——用正则把形如 `ak-…`、`as-…`（跨过 10 个字母数字的令牌形状）与 `MODAL_TOKEN_(ID|SECRET)\s*[=:]\s*\S+` 替换成 `***`。它不需要读取令牌值本身。

### 5.5 `ModalSandboxPool` 与模块级函数

```
class ModalSandboxPool:
    def __init__(self, modal_mod, *, clock=time.monotonic): ...
    def lease(self, user_id, session_id, *, create=True) -> ContextManager[ModalSandbox | None]
    def reset_session(self, user_id, session_id) -> None
    def reset_user(self, user_id) -> int
    def interrupt_session(self, user_id, session_id) -> bool
    def shutdown(self, *, deadline_s=15.0) -> None
    def snapshot(self) -> list[dict]      # 诊断 / 测试用：哈希后的 id、状态、busy、last_used
```

- 池用一把普通 `threading.Lock` 保护 `dict[(user_id, session_id)] -> ModalSandbox`。**池锁绝不跨网络调用持有**（终止、创建、健康检查都在锁外）。
- **`lease` 是唯一的取用方式**：在池锁内查找 / 新建条目并把它的 `busy` 计数 +1，退出上下文时 -1 并刷新 `last_used`。正在被租用（`busy > 0`）的条目永远不会被驱逐，这样就不会出现“刚选中它当空闲牺牲品、别的线程却刚开始用它”。`create=False` 时条目不存在就交出 `None`（给产物扫描、`interrupt` 用，不创建任何东西）。
- **上限与驱逐**（新建条目时）：先在锁内剪掉已死且不在租用的条目，以及 `busy == 0` 且距 `last_used` 超过 `MODAL_IDLE_TIMEOUT_SECONDS + 60` 的条目（Modal 应该已经回收它们）；然后如果该用户的条目数 ≥ 每用户上限，就从**这个用户**的空闲条目里选最久没用的驱逐；全局条目数 ≥ 全局上限，就从**所有用户**的空闲条目里选最久没用的驱逐。没有可驱逐的（全在忙）才拒绝，抛 `SandboxUnavailable`（文案见 §5.4），审计 `reject:cap_user` / `reject:cap_total`。被驱逐的条目先从字典里摘掉（锁内），再在锁外 `close()`（审计 `terminate:evict`）。创建失败的条目立刻从池里移除，不占名额。
- `reset_session` 摘掉并 `close()` 当前会话的条目（不存在就是空操作）；`reset_user` 对该用户的全部条目这样做并返回个数；`interrupt_session` 对存在的条目调 `interrupt()`（不创建、不租用计数也行，注意不取池锁以外的任何锁）。
- `shutdown(deadline_s)`：先设 `closing`（之后的 `lease` 直接抛 `SandboxUnavailable`），摘下全部条目，用多线程并行 `close()`，整体最多等 `deadline_s` 秒（超时就走，不阻塞退出）。
- **模块级函数**（`chat_service` 只通过它们访问，且必须用模块属性调用——`modal_sandbox.reset_session(...)`，不要 `from ... import reset_session`，否则测试里替换不到）：`load_modal()`（§5.1）、`get_pool()`（惰性单例，首次调用才经 `load_modal()` 导入 `modal`；导入失败抛 `SandboxUnavailable`）、`get_routed_sandbox()`（`ModalRoutedSandbox` 的进程级单例，惰性创建；创建它既不导入 `modal` 也不联网）、`reset_session(user_id, session_id)`、`reset_user(user_id) -> int`、`interrupt_session(user_id, session_id) -> bool`、`interrupt_session_nowait(user_id, session_id) -> None`（起守护线程跑 `interrupt_session` 并立即返回）、`shutdown_pool()`、`member_code_status()`。**除 `load_modal()`、`get_pool()`、`get_routed_sandbox()` 这三个取用函数外，其余函数在池还没创建时都是空操作，不创建池、不 import `modal`、不联网，永不抛异常**（没配 Modal 的环境里，`/reset`、`/stop`、关闭都会走到它们）。`atexit.register(shutdown_pool)` 做进程退出兜底。`ModalRoutedSandbox` 调 `get_pool()` 时，它抛出的 `SandboxUnavailable` 与 `lease()` 抛出的同一类错误一视同仁，都走 §5.6 的拒绝路径。

  **时钟规则**（§9.1 的 FakeClock 靠它成立）：`ModalSandboxPool(modal_mod, clock=...)` 把 `clock` 传给它创建的每个 `ModalSandbox`；沙箱用它量耗时（填 `RunResult.duration`、判断「137 且耗时 ≥ N」），池用它判断空闲、驱逐与熔断。整个文件里不得直接调用 `time.monotonic()` / `time.time()`——只允许出现在 `clock=` 参数的默认值里，以及审计日志的 `ts` 字段里（§9.3 的 N 会扫）。需要等待的地方一律用 `threading.Event.wait(timeout)` / `Thread.join(timeout)`，不要用 `time.sleep` 猜时序。

### 5.6 `ModalRoutedSandbox(BaseSandbox)`：按请求上下文路由

是一个**无状态**的单例式对象，同时当 `CompositeBackend` 的 `default` 与 `execute_python_code` / `SandboxOutputsMiddleware` 的“沙箱”用。每次调用都现解析上下文：

- `ctx = get_request_context()`：没有上下文 → 拒绝；`ctx.is_owner` 为真 → 拒绝（「云端沙箱仅供成员使用」，owner 永远不该走到这里，这是纵深防御）；`validate_user_id(ctx.user_id)` 不过 → 拒绝；`ctx.session_id` 为空 → 拒绝。拒绝时记审计 `reject:context`。
- `execute(command, *, timeout=None)` / `upload_files` / `download_files` / `collect_new_outputs()` / `interrupt()`：解析出 `(user_id, session_id)` 后用 `get_pool().lease(...)` 取沙箱并转发；拒绝或创建失败时，`execute` 返回 `ExecuteResponse(output='[云端沙箱不可用] …', exit_code=1, truncated=False)`，`upload_files` / `download_files` 返回逐项 `error=`，`collect_new_outputs()` 返回空 `OutputsResult()`（且用 `create=False`），`interrupt()` 返回 `False`。
- `run_python(code, *, timeout=None) -> RunResult`：拒绝或创建失败时**抛 `SandboxUnavailable`**（和 Docker 版一致，由 `execute_python_code` 转文本）。
- `id` 返回固定字符串 `'modal-routed'`。不要在实例上缓存任何用户或会话相关的状态。
- 同时要在两条线程路径下都能拿到上下文：deepagents 同步路径下工具跑在 LangChain 的线程池里（`copy_context`），异步路径下 `BaseSandbox.aexecute` 走 `asyncio.to_thread`。测试要分别覆盖。

### 5.7 审计日志

- `logging.getLogger('gnk48.modal_audit')`，`propagate = False`，用 `RotatingFileHandler(config.MODAL_AUDIT_LOG, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')`；**惰性**创建（第一次写才建目录与 handler，import 时不做任何 IO）；handler 创建失败就关掉审计并在控制台提示一次，**绝不能让审计失败影响代码执行**。
- 一行一个 JSON：`{ts, event, u, s, ...}`，`u` / `s` 是哈希。事件：`create_ok`、`create_fail`（带分类与类名）、`exec`（`command[:500]`、超时、退出码、耗时、是否截断）、`run_python`（`code_len`、`code[:500]`、退出码、耗时）、`upload` / `download`（路径、字节数）、`interrupt`（模式、结果）、`terminate`（`reason`：`reset` / `reset_user` / `evict` / `stop` / `shutdown` / `dead`）、`reject`（`reason`：`cap_user` / `cap_total` / `breaker` / `context`）、`breaker_open` / `breaker_close`。
- **绝不记**：输出内容、环境变量、任何令牌、明文用户 / 会话 ID。每个字符串字段都先过 `_redact()`。

### 5.8 手动管理脚本 `scripts/manage_modal_sandboxes.py`

没有启动时孤儿清理（见 §1.1）10），所以我需要一个手动工具：`list`（列出 `svc=gnk48` 标签的沙箱：对象 id、`u` / `s` 标签）、`kill --all`、`kill --user <哈希前缀>`；删除类操作默认要交互确认，`--yes` 跳过。用 `modal.App.lookup(config.MODAL_APP_NAME, create_if_missing=False)`，`modal.Sandbox.list(app_id=app.app_id, tags={'svc': 'gnk48'})`（`app_id` 是属性；签名以已安装版本为准）。开头沿用现有脚本的写法（`from app import bootstrap` + `bootstrap.apply()`，参照 `scripts/manage_sessions.py`），用 `argparse`，`--help` 完全离线；绝不打印令牌。通过 `set_modal_module` 注入 FakeModal 即可离线测试。

## 6. M3 命名镜像与构建脚本

原则：镜像由我**手动**构建并发布一次，运行时只用 `Image.from_name(config.MODAL_IMAGE_NAME)`，运行时代码里不允许出现 `.build(` / `.publish(` / `debian_slim(`（§9 的 AST 测试会扫 `modal_sandbox.py`）。这样用户的第一条消息不会触发一次几分钟的构建，构建的费用也完全由我掌控。

### 6.1 `app/infra/modal_image.py`（L1，纯定义，不 import `modal`）

对外提供下列常量和函数，构建脚本与测试都从这里取：

- `IMAGE_PYTHON_VERSION = '3.13'`（和现有 Docker 镜像一致，保证 `deploy/sandbox/requirements.txt` 的固定版本能装）。
- `APT_PACKAGES`：`fonts-noto-cjk ripgrep jq unzip zip file tree procps git curl wget ca-certificates build-essential poppler-utils sqlite3 graphviz`（前 7 个与 Docker 镜像一致，其余是云端有网络后常用的）。`APT_OPTIONAL`：`ffmpeg nodejs npm`（体积大、不一定用得到，默认不装，由 `--with-optional` 开启）。
- `PIP_EXTRA`：`httpx requests beautifulsoup4 lxml pillow pypdf pdfplumber python-docx python-pptx xlsxwriter networkx statsmodels`。
- `IMAGE_ENV`：`PYTHONUNBUFFERED=1`、`PYTHONDONTWRITEBYTECODE=1`、`PYTHONIOENCODING=utf-8`、`LANG=C.UTF-8`、`MPLBACKEND=Agg`、`MATPLOTLIBRC=/opt/mplrc`、`MPLCONFIGDIR=/opt/mplcache`、`HOME=/work`。（与 Docker 镜像的差异：`MPLCONFIGDIR` 直接用构建时预热过字体缓存的 `/opt/mplcache`，因为云端沙箱以 root 运行，该目录可写，不需要 Docker 那样另开 `/tmp/mpl`。）
- `FONT_SELFCHECK_PY`：构建期自检脚本字符串，内容就是 `deploy/sandbox/Dockerfile` 里那段（渲染含中文与负号的图，断言没有 `Glyph missing` 警告与 `findfont` 噪声）。
- `read_pinned_requirements() -> list[str]`：读 `deploy/sandbox/requirements.txt`（忽略空行与 `#` 注释）。路径用 `config` 里现有的项目根目录常量拼（读 `config.py` 确认名字）。**这个文件是依赖版本的唯一来源**，Docker 与 Modal 两个镜像都从它来。
- `build_image(modal, *, with_optional=False, force=False)`：接收**已经导入的 `modal` 模块**、返回未构建的 `modal.Image`，内容依次为：`modal.Image.debian_slim(python_version=IMAGE_PYTHON_VERSION, force_build=force)` → `.apt_install(*APT_PACKAGES, *(APT_OPTIONAL if with_optional else ()))` → `.pip_install(*pinned, *PIP_EXTRA)`（同一层，让 pip 一次解析）→ `.add_local_file(<仓库>/deploy/sandbox/matplotlibrc, '/opt/mplrc/matplotlibrc', copy=True)`（**必须 `copy=True`**，否则文件不会被烘进发布的镜像）→ `.env(IMAGE_ENV)` → `.run_commands(...)`（下面的步骤）→ `.workdir('/work')`。
- `run_commands` 里依次做：① `mkdir -p /work/outputs /opt/mplrc /opt/mplcache`；② 防御性保证 `python3` 存在（`command -v python3 || ln -s "$(command -v python)" /usr/local/bin/python3`，因为 deepagents 的文件工具都靠 `python3 -c`）；③ 预热字体缓存 `python -c 'import matplotlib.pyplot'`；④ 跑 `FONT_SELFCHECK_PY`。脚本要进 shell，引号转义容易出错，**用 base64 传**：`echo <b64> | base64 -d | python`。
- **不建非 root 用户**，不设 `USER`。

### 6.2 `scripts/build_modal_image.py`

用法：`uv run python scripts/build_modal_image.py [--name NAME[:TAG]] [--with-optional] [--force] [--check-only] [--yes]`。写成 `main(argv=None) -> int`（方便测试），开头沿用现有脚本的 `from app import bootstrap` + `bootstrap.apply()`，`modal` 一律通过 `modal_sandbox.load_modal()` 拿（这样测试里 `set_modal_module(FakeModal)` 就能离线跑整个脚本）。`--help` 必须完全离线。

流程（非 `--check-only`）：

1. 前置检查：`config.MODAL_TOKENS_PRESENT` 为假或 `modal` 导入失败 → 打印中文提示、退出码 2。
2. 打印计划（镜像名、基础、包列表、是否 `--force`）与警告「这会在 Modal 上构建镜像，产生少量费用，可能需要几分钟」；没有 `--yes` 时交互确认。
3. `app = modal.App.lookup(config.MODAL_APP_NAME, create_if_missing=True)` → `image = build_image(modal, ...)` → `with modal.enable_output(): image.build(app)`（命中缓存时 Modal 不会重建；`--force` 才强制重建）→ `image.publish(name)`。构建失败时把异常脱敏后打印、退出码 1。
4. 继续做验收（下）。

`--check-only`：只做验收。这也是判断“镜像到底发布了没有”的唯一方式（`Image.from_name` 是惰性的，不存在要等 `Sandbox.create` 失败才知道），失败时要明确说“镜像不存在，请去掉 --check-only 重新运行”。

**验收**（在一个临时沙箱里，用运行时同样的创建参数：`Image.from_name(name)`、`workdir='/work'`、env 白名单、资源护栏，`timeout=600`、`idle_timeout=120`；**沙箱必须在 `finally` 里 `terminate()`**，包括 Ctrl-C），逐项打印 ✅ / ❌，有任何 ❌ 就以退出码 1 结束：

- `python --version` 与 `python3 --version` 都是 3.13；`id -u` 是 0，`pwd` 是 `/work`，`echo $HOME` 是 `/work`，`/work/outputs` 可写。
- 一次 `python -c 'import ...'` 导入全部库：`pandas numpy matplotlib scipy seaborn openpyxl xlrd tabulate yaml sympy sklearn httpx requests bs4 lxml PIL pypdf pdfplumber docx pptx xlsxwriter networkx statsmodels`。
- `pip freeze` 的结果里，`requirements.txt` 里每一项的版本都与实际一致；把完整的 `pip freeze` 打印出来（这样 `PIP_EXTRA` 之后想钉版本时有据可查）。
- 命令行工具都在：`rg jq unzip zip file tree git curl wget pdftotext sqlite3 dot timeout bash`；`find /work -maxdepth 0 -printf ''` 退出码 0（GNU find 的 `-printf` 是 §5.2 产物收集的依赖）。
- 环境变量：`MPLBACKEND=Agg`、`MATPLOTLIBRC=/opt/mplrc`、`MPLCONFIGDIR=/opt/mplcache`。
- 中文与负号出图：在沙箱里跑 `FONT_SELFCHECK_PY` 的运行时版本，把 PNG 写到 `/work/outputs/selftest.png`，用 `sb.filesystem.read_bytes` 取回、在宿主机上用 PIL 打开校验；再用 `rm` 删掉。
- 公网（仅当 `MODAL_BLOCK_NETWORK` 为假时）：`python -c "import urllib.request; print(urllib.request.urlopen('https://pypi.org', timeout=10).status)"`；失败只报 ⚠️，不算失败。

最后打印「镜像已发布并通过验收：\<name>。如果与 `.env` 里的 `MODAL_IMAGE_NAME` 不同，请改成它」。

### 6.3 版本与升级

镜像内容变了就升 tag（`:v2`）并改 `.env` 里的 `MODAL_IMAGE_NAME`，不要原地覆盖 `:v1`（正在运行的沙箱不受影响，但新旧行为混用时很难排查）。§10.2 的 README 要把这一点写进去；§5.4 的“镜像可能没有发布”提示也应指向这个脚本的 `--check-only`。

## 7. M4 工具、CodeAgent、只读技能、编排与提示词

这一步要同时保证两件事：member 图拿到了 CodeAgent，且 member 图的编译期隔离一点不退化。按 §7.1 → §7.6 的顺序做；§7.1、§7.2、§7.5 里凡是写到联网、超时、内存、闲置分钟数的句子，都按 §7.6 现算，不要写死。

### 7.1 `app/tools/code_tools.py`（L3）

- 把 `execute_python_code` 函数体里「执行 + 格式化成文本 / 多模态列表」的部分抽成共享实现 `_run_code(get_sandbox, code, *, workspace_hint)`（绝不向外抛异常、`SandboxUnavailable` 转文本、状态行 / Stdout / Stderr / 图片 / 非图片产物 / `/work` 清单 / 提示信息的拼装，全部保持原样）。`workspace_hint` 为真时才带「`/workspace` 只读」那段提示（只有 owner 的 Docker 版需要）。
- **owner 的 `execute_python_code` 与 `TOOLS` 必须原样保留**：同样的 `@tool('execute_python_code')`、同样的 docstring（它就是工具描述，一个字不许改）、同样的行为。它必须在**调用时**才去取 `sandbox.get_sandbox()`（传 `lambda: sandbox.get_sandbox()`，不要在 import 时绑定），因为现有测试会在运行时 monkeypatch `sandbox.get_sandbox`。
- 新增 `build_execute_python_code(get_sandbox, *, description, workspace_hint=False)`：用 `StructuredTool.from_function(func=..., name='execute_python_code', description=description)` 返回一个工具（参数签名仍是 `code: str`）。
- 新增 `member_execute_python_description() -> str`：从 `config` 现读数字渲染出 member 版工具描述，结构与 owner 版一致（「环境规范与事实约束」三条），但要改成云端事实：① Python 3.13，预装 pandas / numpy / matplotlib / scipy / seaborn / scikit-learn / statsmodels / sympy / openpyxl / networkx / Pillow / requests / httpx 等，**沙箱可以联网**，缺少的库可以 pip install；② 目录只有 `/work`（cwd 与 HOME，同一会话内跨调用保留，沙箱闲置约 N 分钟后被回收，文件随之丢失）与 `/work/outputs/`（图片自动发送，写法与 owner 版相同：`savefig` 到 `/work/outputs/<英文名>.png`，禁止 `plt.show()`）；③ 限制：单次执行默认 `MODAL_EXEC_TIMEOUT_SECONDS` 秒、上限 `MODAL_EXEC_MAX_TIMEOUT_SECONDS` 秒，内存约 `MODAL_MEMORY_MB_LIMIT / 1024` GiB，单次最多 `SANDBOX_MAX_IMAGES` 张图（单张 ≤ `SANDBOX_MAX_IMAGE_MB` MB、总像素 ≤ 40M），输出过多会被截断。**不得出现** `Docker`、`/workspace`、`workspace`、`宿主机`、`FileAgent`、`DesktopAgent`、`BrowserAgent`。
- 新增 `build_member_tools() -> list`：返回 `[build_execute_python_code(lambda: modal_sandbox.get_routed_sandbox(), description=member_execute_python_description())]`。`modal_sandbox.get_routed_sandbox()` 是返回 `ModalRoutedSandbox` 进程级单例的函数（写在 `modal_sandbox.py` 里，和 `get_pool()` 一样惰性）。

### 7.2 `app/agents/experts/code_agent.py`（L4）

owner 部分（`NAME`、`DESCRIPTION`、`PROMPT`、`SANDBOX_FS_TOOLS`、`SandboxOutputsMiddleware`、`build_code_agent_runnable`）一个字都不动。新增：

- `MEMBER_DESCRIPTION`（进主管的 `task` 工具描述）：代码与计算专家——在云端隔离沙箱里写代码、跑 Python 与 Linux 命令，用于数学计算、数据处理与分析、绘图、批量处理文本与数据；沙箱可以联网（下载公开数据、pip 安装缺少的库）；**访问不到用户电脑上的任何文件，要处理的数据必须直接写进任务描述**；沙箱闲置约 N 分钟会被回收，之前生成的文件随之丢失；需要绘图要明确要求。（N 从 `config.MODAL_IDLE_TIMEOUT_SECONDS` 现算。）
- `MEMBER_PROMPT`：以 `WORKER_BASE_PROMPT` 开头，必须含「你是 CodeAgent」（`cli_logger` 靠它识别角色）。下面是设计稿，措辞可以微调，但**每一条事实、每一条红线、以及「禁用词」规则都要保留**；大括号里的数字由 `config` 现算（用普通字符串拼接或 `.replace`，不要让提示词里的花括号和 `str.format` 打架）。

```
你是 CodeAgent，专职在云端隔离沙箱里编写执行代码、运行 Linux 命令行工具、解决数学计算、数据处理与分析、绘图任务。
【运行环境 - 强制】
1. 全部使用绝对路径。/work 是沙箱的可读写工作目录（cwd 与 HOME），同一会话内保留；沙箱闲置约 {N} 分钟或被重置后会被回收，/work 里的文件随之丢失，重要的中间结果要写进汇报。
2. /work/outputs/ 是专用交付箱：保存到这里的图片会自动发送给用户，同一文件每个版本只发一次。
3. 你看不到用户电脑上的任何文件，沙箱里也没有用户的文件；要处理的数据只会出现在任务描述里。数据缺失时如实汇报缺什么，不要编造数据。
4. /skills/ 是只读技能目录，只能用 read_file(limit=1000) 读，它不属于沙箱文件系统，在沙箱 shell 里看不到。
5. 沙箱可以联网（有公网）：可以用 requests / httpx / curl 下载公开数据；缺少的库可以 pip install（先用 execute('pip list | rg 名称') 确认没有预装）。下载来的数据先看大小再读。
6. 环境：Python 3.13；预装 pandas、numpy、matplotlib、scipy、seaborn、scikit-learn、statsmodels、sympy、openpyxl、networkx、Pillow、requests、httpx、beautifulsoup4、lxml、pypdf、pdfplumber、python-docx、python-pptx 等，以及 rg、jq、unzip、zip、file、tree、git、curl、wget、pdftotext 等命令行工具。
7. 资源限制：单次执行默认 {exec_default} 秒、最长 {exec_max} 秒（execute 可用 timeout 参数调整）；内存上限约 {mem_gib} GiB；单次最多导出 {max_images} 张图片。超时或超内存的命令会被终止。
【工具决策表】
1. 一次性 Python（计算/分析/绘图）→ execute_python_code（无需写 python -c）。
2. Shell 命令（ls/rg/jq/unzip/zip/file/tree/wc/head/curl/pip…）→ execute：每次调用都是独立 subshell，cd 与 export 环境变量不跨调用保留，多条命令请用 '&&' 串联；默认 {exec_default} 秒，可用 timeout 参数至多 {exec_max} 秒。
3. 多文件/多步脚本 → 用 write_file 写到 /work/xxx.py → execute('python3 /work/xxx.py') 运行 → 局部微调优先用 edit_file 避免整文件重写 → 再次运行。
4. 读取与检索 → 用 ls / glob / grep / read_file；grep 是字面精确匹配，正则表达式搜索请用 execute('rg <regex> ...')；PDF/Word 等富文档可用 pypdf / pdfplumber / python-docx 在沙箱里读（前提是文件已从公开网址下载到 /work）。
【行为红线与禁止事项】
1. 严禁用沙箱做以下事：挖矿；对任何第三方的端口扫描、攻击、压力测试、暴力破解；批量注册、刷量、发送垃圾信息；绕过他人的访问限制或抓取受保护的内容；编写或投放恶意软件。遇到这类要求直接拒绝，并如实汇报“该任务不在允许范围内”。
2. 严禁探测沙箱之外的内容，或打听云平台的账号、令牌、配置。
3. 严禁对根目录 '/' 盲目递归搜索（必须先用 ls 缩小范围，限定子目录与文件类型）。
4. 不要后台挂起长时间进程或开启服务端口（沙箱不对外开放端口）；命令必须在时限内结束。
5. 纯计算任务无需读取技能；绘图任务必须先 read_file('/skills/matplotlib-chart/SKILL.md', limit=1000)；数据分析任务必须先 read_file('/skills/data-analysis/SKILL.md', limit=1000)。
【汇报格式】汇报遵循工作专家标准格式，以“【内部专家汇报 - 来源：CodeAgent】”开头；不向用户暴露沙箱内部临时路径。
```

**说明**：上面是联网版设计稿。`MEMBER_DESCRIPTION` 与 `MEMBER_PROMPT` 不是写死的字符串，而是 `build_member_description()` 与 `build_member_prompt()` 现算结果的快照（`MODAL_BLOCK_NETWORK=true` 时会生成断网版）；数字与联网句子的规则见 §7.6。

- `MemberCodeAgentBuild`（`@dataclass(frozen=True)`）：`runnable`、`backend`（构建时用的 `CompositeBackend`）、`tool_names: frozenset[str]`（工具名集合 = `{t.name for t in build_member_tools()} ∪ set(SANDBOX_FS_TOOLS)`，恰为 `execute_python_code, ls, read_file, write_file, edit_file, glob, grep, execute` 八个）。
- `build_member_code_agent_runnable(llm, system_prompt, checkpointer=None, max_tool_calls=None) -> MemberCodeAgentBuild`：结构照抄 `build_code_agent_runnable`，只改四个地方——① 沙箱用 `modal_sandbox.get_routed_sandbox()`，不接受外部传入的 `sandbox_backend`；② `CompositeBackend(default=<路由沙箱>, routes={SKILLS_ROUTE: ReadOnlyFilesystemBackend(root_dir=SKILLS_MEMBER_DIR, virtual_mode=True), '/_artifacts/': StateBackend()}, artifacts_root='/_artifacts')`；③ `tools=build_member_tools()`；④ `FilesystemMiddleware(..., max_execute_timeout=config.MODAL_EXEC_MAX_TIMEOUT_SECONDS)`。中间件列表、预算中间件、缓存中间件与 owner 版一致。**绝不调用、也不引用 owner 那个 `build_code_agent_runnable` 与 `sandbox.get_sandbox`。**
- member 的工具集合必须**恰好是上述 8 个**，不含 `delete`。

### 7.3 `/skills/` 对 member 只读（修补一个洞）

**问题**：owner 的 CodeAgent 把 `/skills/` 路由到宿主机真实目录 `skills/`（`FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)`），而 `FilesystemBackend` 的 `write / edit / delete / upload_files` 没有任何只读保护，`write_file` / `edit_file` 又在 `SANDBOX_FS_TOOLS` 里。member 的 CodeAgent 一旦拿到这两个工具，就能改写 owner 的 Agent 之后会读的技能文件（提示词注入的一种持久化形式）。这不在阶段 1 的范围里，必须在本阶段堵上。

**做法**（`app/agents/skills.py`，其余约定不变）：

- 新增 `SKILLS_MEMBER_DIR`（仓库根目录下的 `skills_member/`，写法和 `SKILLS_DIR` 一致）。
- 新增 `ReadOnlyFilesystemBackend(FilesystemBackend)`：覆盖 `write` / `awrite` / `edit` / `aedit` / `delete` / `adelete` / `upload_files` / `aupload_files`（签名与返回类型照 M0 核对到的实际为准），一律不碰磁盘、直接返回带 `error=` 的结果（`WriteResult` / `EditResult` / `DeleteResult`；`upload_files` 的每一项返回 `FileUploadResponse(path, error=...)`），错误文案：“错误：/skills/ 是只读目录，不能写入、修改或删除。”。读类方法（`ls` / `read` / `grep` / `glob` / `download_files`）保持父类行为。
- 新增 `build_member_skills_backend()`：`CompositeBackend(default=StateBackend(), routes={SKILLS_ROUTE: ReadOnlyFilesystemBackend(root_dir=SKILLS_MEMBER_DIR, virtual_mode=True)})`，给 member 的**主管**用（见 §7.4）。`build_skills_backend()` 不动（owner 用）。
- **`skills_member/`**：新目录，以 `skills/matplotlib-chart/SKILL.md` 与 `skills/data-analysis/SKILL.md` 为底稿改写（先读现有文件，保持 frontmatter 的 `name` / `description` 结构）。`matplotlib-chart`：`description` 里的“Docker 沙箱”改成“云端沙箱”，正文按沙箱现实检查一遍（保存路径 `/work/outputs/<英文名>.png`、`plt.close(fig)`、禁 `plt.show()`、≤ 4 张、水印规则都保留）。`data-analysis`：整篇围绕 `/workspace` 写的，要改成围绕 `/work`——数据来源是「任务描述里给出的数据、代码生成的数据，以及（仅限沙箱可以联网时）从公开网址下载到 /work 的文件」，保留“探查先行、编码探测、大数据集安全读取、中间结果存 /work、精炼汇报”这些通用规则。两篇里**不得出现** `Docker`、`/workspace`、`workspace`、`宿主机`、`FileAgent`、`DesktopAgent`、`BrowserAgent`。`skills/`（owner 用）一个字都不动。两篇里凡是涉及联网、下载、pip install 的句子，一律按 §7.6 写成有条件的。

### 7.4 `app/agents/orchestrator.py`

只改 member 一侧，`_build_owner_agent` 与 `_local_subagents` 一行不动。

- `_member_subagents(code=None)`：默认（`code is None`）与现在一模一样。`code` 是 `MemberCodeAgentBuild` 时，在 `# STAGE2: 云端 CodeAgent` 标记处追加 `CompiledSubAgent(name=code_agent.NAME, description=code_agent.MEMBER_DESCRIPTION, runnable=code.runnable)`（顺序：Knowledge、Media、Map、Code）。
- `_assert_member_graph_safe(supervisor_tools, subagents, vetted_code=None)`：原有的全部检查保持，**13 个禁用工具名一个不减**，原有的报错文案也不改。只增加一个严格的例外：预编译子专家（带 `runnable`）仅在同时满足以下条件时放行：`vetted_code is not None`；它的 `name == code_agent.NAME`；`sa['runnable'] is vetted_code.runnable`（身份比较，不是相等）；整张图里只有这一个预编译专家；且 `vetted_code` 通过下面的构建物断言。专家名白名单在 `vetted_code` 不为空时才额外允许 `code_agent.NAME`。
- **构建物断言**（失败抛 `RuntimeError`，阻止启动）：`vetted_code.tool_names == frozenset({'execute_python_code', 'ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'execute'})`；`isinstance(vetted_code.backend, CompositeBackend)`；`isinstance(backend.default, modal_sandbox.ModalRoutedSandbox)`（不是 `DockerSandbox`）；`set(backend.routes) == {'/skills/', '/_artifacts/'}`；`/skills/` 路由的后端是 `ReadOnlyFilesystemBackend` 且根目录是 `SKILLS_MEMBER_DIR`（root 存在哪个属性上，读 `FilesystemBackend.__init__` 确认）；`/_artifacts/` 路由是 `StateBackend`。
- `_build_member_agent(checkpointer)`：① `ok, reason = modal_sandbox.member_code_status()`；② `ok` 时 `code = code_agent.build_member_code_agent_runnable(llm=models.worker_llm, system_prompt=code_agent.MEMBER_PROMPT)`（和 owner 一样不传 checkpointer），否则 `code = None`；③ `subagents = _member_subagents(code)`，`_assert_member_graph_safe(supervisor_tools, subagents, vetted_code=code)`；④ 主管提示词 `MEMBER_MAIN_AGENT_PROMPT_CODE if code else MEMBER_MAIN_AGENT_PROMPT`；⑤ `backend=build_member_skills_backend()`（替换原来的 `build_skills_backend()`，只在 member 这一侧）。返回的专家名元组要包含 `CodeAgent`（启用时）。`member_code_status()` 不可用时**不调用任何 CodeAgent 构建函数**。
- 主管 `tools` 不变（`get_current_time` + 会话回溯工具）。member 图的 `ToolCallLimitMiddleware(tool_name='task', run_limit=12, ...)` 不变。

### 7.5 `app/agents/prompts.py`

`MAIN_AGENT_PROMPT` 一个字节都不能变（M0 先算出它当前的 sha256，写成常量放进测试里钉住）。针对 member：

- 新增 `build_member_main_prompt(code_enabled: bool) -> str`，并令 `MEMBER_MAIN_AGENT_PROMPT = build_member_main_prompt(False)`、`MEMBER_MAIN_AGENT_PROMPT_CODE = build_member_main_prompt(True)`。建议把现有文本切成带 `@@占位符@@` 的模板、用 `str.replace` 填充（不要用 `str.format` / f-string，提示词里可能有花括号）。
- **两个变体都要做的一处修正**：“委派规范”里那句例子里的真实台北地址换成中性占位：把 `用户当前位置在台北市信义区市府路1号，请搜索附近的菜市场` 改成 `用户当前位置在（此处写用户给出的具体地址），请搜索附近的菜市场`。两个变体里都不得再出现 `台北市信义区市府路1号`（owner 的 `MAIN_AGENT_PROMPT` 里继续保留它，那是 owner 自己的地址）。除此之外，`code_enabled=False` 的文本与阶段 1 的 `MEMBER_MAIN_AGENT_PROMPT` **逐字相同**。
- **`code_enabled=True` 的四处差异**：
  1. 【委派规范】末尾（在 MediaAgent 那条及其“严格数据边界”子条之后）追加两条：“计算、数据处理与分析、绘图、批量处理文本或数据、需要运行代码或命令才能得到结果的任务，委派给 CodeAgent。它在云端沙箱里运行，沙箱可以联网（可以下载公开数据），但访问不到用户电脑上的任何文件。要处理的数据必须原样写进 task 的 description（数据太多就让用户缩小范围，或改为让它从公开网址下载）；绘图要写清图的类型、数据、标题与坐标含义，并明确要求出图。同一个任务里需要的计算与绘图合并成一次 task。”“CodeAgent 的云端沙箱闲置一段时间后会被回收，里面的文件随之丢失：不要让它“继续处理之前生成的文件”，需要延续时把上一步的关键结果写进新的 description。”
  2. 【能力边界】改写：“你没有读写用户电脑文件、操作桌面或浏览器的能力。用户提出这类要求时，如实说明做不到，并在可行时给出文字建议（例如告诉用户可以怎么自己操作）；不要尝试委派——系统里没有对应的专家。计算、数据分析、绘图、运行代码可以委派给 CodeAgent（它在云端运行，看不到用户电脑上的文件）。”——保留“不要尝试委派”这几个字。
  3. 在【过程不可见】之后、【长期记忆】之前插入 owner 提示词里的【图文分离】段（原样复制）。
  4. 【回复要求】末尾“不要向用户提及内部专家…”那句改为列上 CodeAgent（“如 KnowledgeAgent、MediaAgent、MapAgent、CodeAgent 等”），并在其后追加一行：“不要向用户提及沙箱内部的文件路径（如 /work/outputs/...）；图片由系统另行发送，你只用文字交代结论。”
- 两个变体都必须保留「你是一个多智能体系统的主 Agent」这句话（`cli_logger._identify_agent` 靠它识别角色）。
- **禁用词**：`code_enabled=False` 的变体沿用阶段 1 的 H 测试（不得含 `FileAgent`、`CodeAgent`、`DesktopAgent`、`BrowserAgent`、`workspace`、`宿主机`、`Docker`、`沙箱`）；`code_enabled=True` 的变体**不得含** `FileAgent`、`DesktopAgent`、`BrowserAgent`、`workspace`、`宿主机`、`Docker`，但允许出现 `CodeAgent` 与 `沙箱`。（不要把 `本机` 加进禁用词：现有文本里的“访问不到用户的任何本机文件”是有意保留的。）

### 7.6 提示词与工具描述里「随配置变化」的句子

**要解决的问题**：§7.1、§7.2、§7.5 里有两类句子取决于配置而不是常量——数字（超时、内存、闲置分钟数），以及「沙箱能不能联网」。`MODAL_BLOCK_NETWORK=true` 时沙箱是断网的，如果提示词还写着「可以联网、可以 pip install」，模型会白白浪费工具调用去下载，再拿报错去糊弄用户。所以这几段文字必须由函数现算，不能写成写死的字符串。

**四个构建函数**，都在调用时才读 `config`（不要在 import 时把 `config.MODAL_*` 缓存到别处）：

- `code_tools.member_execute_python_description() -> str`（§7.1）。
- `code_agent.build_member_description() -> str` 与 `code_agent.build_member_prompt() -> str`（§7.2；后者以 `WORKER_BASE_PROMPT` 开头）。
- `prompts.build_member_main_prompt(code_enabled: bool) -> str`（§7.5；`code_enabled=False` 的文本里没有任何联网句子，与配置无关）。

模块常量 `MEMBER_DESCRIPTION`、`MEMBER_PROMPT`、`MEMBER_MAIN_AGENT_PROMPT`、`MEMBER_MAIN_AGENT_PROMPT_CODE` 就是它们在 import 时的返回值（快照）。编译进图的用这些常量：一个进程里 config 不会变，所以和现算的一致，§9.4 里对提示词的身份比较（`is`）也才成立。测试要验证「文本随配置变化」时，先 monkeypatch config，再直接调用这四个函数（§9.3 的 P）。

**现算的数字**（统一写法，测试按它断言）：§7.2 设计稿里的 `{N}` 取 `max(1, (config.MODAL_IDLE_TIMEOUT_SECONDS + 30) // 60)`；`{mem_gib}` 取 `f'{config.MODAL_MEMORY_MB_LIMIT / 1024:g}'`（2048 得 `2`，768 得 `0.75`）；`{exec_default}`、`{exec_max}`、`{max_images}` 分别取 `config.MODAL_EXEC_TIMEOUT_SECONDS`、`config.MODAL_EXEC_MAX_TIMEOUT_SECONDS`、`config.SANDBOX_MAX_IMAGES`；单张图片上限取 `config.SANDBOX_MAX_IMAGE_MB`。

**联网句子的两个版本**（`config.MODAL_BLOCK_NETWORK` 为假用联网版，为真用断网版）：

| 位置 | 联网版 | 断网版 |
| --- | --- | --- |
| `member_execute_python_description()` 的第 ① 条 | 沙箱可以联网，缺少的库可以 pip install | 沙箱不能联网，不能下载数据，也不能 pip install，只能用预装的库 |
| `build_member_description()` | 沙箱可以联网（下载公开数据、pip 安装缺少的库）；访问不到用户电脑上的任何文件，要处理的数据必须直接写进任务描述 | 沙箱不能联网（不能下载数据，也不能 pip install）；访问不到用户电脑上的任何文件，要处理的数据必须直接写进任务描述 |
| `build_member_prompt()` 运行环境第 5 条 | §7.2 设计稿原文（开头是「沙箱可以联网（有公网）：」） | 沙箱不能联网，不能下载数据，也不能 pip install；只能使用预装的库，数据必须直接写进任务描述。 |
| `build_member_prompt()` 工具决策表第 2 条括号里的命令清单 | ls/rg/jq/unzip/zip/file/tree/wc/head/curl/pip… | ls/rg/jq/unzip/zip/file/tree/wc/head…（去掉 curl 与 pip） |
| `build_member_prompt()` 工具决策表第 4 条末尾的括号 | （前提是文件已从公开网址下载到 /work） | （用户的文件进不来，只有代码生成的文件可读） |
| `build_member_main_prompt(True)` 的【委派规范】新增的第一条 | …它在云端沙箱里运行，沙箱可以联网（可以下载公开数据），但访问不到用户电脑上的任何文件。要处理的数据必须原样写进 task 的 description（数据太多就让用户缩小范围，或改为让它从公开网址下载）… | …它在云端沙箱里运行，沙箱不能联网，也访问不到用户电脑上的任何文件。要处理的数据必须原样写进 task 的 description（数据太多就让用户缩小范围）… |

**断言用的固定短语**（§9.3 的 P 靠它们判断，措辞不能漂）：

- 联网版：上面四个函数的返回值，每个都恰含一次 `沙箱可以联网`，且不含 `沙箱不能联网`。
- 断网版：每个都恰含一次 `沙箱不能联网`，且不含这些肯定句片段：`沙箱可以联网`、`有公网`、`缺少的库可以`、`pip 安装缺少的库`、`下载公开数据`、`从公开网址下载`、`requests / httpx / curl`、`curl/pip`。断网版自己的否定句「不能下载数据，也不能 pip install」里会出现「下载」与 `pip install` 这两个词，这是允许的，所以断言用上面这些整句片段，不用单个词。
- `build_member_main_prompt(False)` 在两种配置下都既不含 `沙箱可以联网` 也不含 `沙箱不能联网`。

**skills\_member 里的联网表述**：那两篇是静态文件，随不了配置，所以写成有条件的——凡涉及联网、下载、pip install 的句子，都加上前提「仅当 CodeAgent 的运行环境说明写明沙箱可以联网」；数据来源清单写成「任务描述里给出的数据、代码生成的数据，以及（仅限可以联网时）从公开网址下载到 /work 的文件」。

**自查**：写完用 `rg -n '沙箱可以联网|沙箱不能联网' app` 看一遍，只应命中上面四个函数所在的三个文件（`code_tools.py`、`code_agent.py`、`prompts.py`）。

## 8. M5 chat\_service 接入点（L5）

阶段 1 在 `app/services/chat_service.py` 里留了两处 `# STAGE2:` 标记（`_cmd_reset` 与 `_stop_current_turn`），本节把它们补上，再加几处必要的钩子。先 `rg -n "STAGE2" app tests`（pattern 里的引号按你的 shell 调整）列出全部标记位置，逐一对照下面的清单。**只改 member 一侧**：owner 的 Docker / host\_shell 分支（`sandbox.reset_sandbox`、`_kill_side_effects`、`sandbox.shutdown_sandbox`）一行不动，也不给 owner 路径新增任何 Modal 调用。

### 8.1 通用规则

- 调用一律写成 `modal_sandbox.<函数>(...)`（先 `from app.infra import modal_sandbox`）。**模块属性调用**，测试靠替换模块属性来观察；不要写 `from app.infra.modal_sandbox import reset_session`。导入放在函数内或文件顶部都行，但不能让 `import modal` 在模块导入时发生（`modal_sandbox` 自己保证惰性，见 §5.1）。
- 这些钩子都是尽力而为：`try / except Exception`，失败只 `print` 一行「⚠️ …（非致命）」，绝不改变命令或回合的结果。会阻塞的（`reset_session`、`reset_user`、`shutdown_pool`）用 `asyncio.to_thread` 调；`interrupt_session_nowait` 本身立即返回，直接调。
- 池的键是 `(user_id, session_id)`，必须与 `wake()` 放进 `RequestContext` 的那一对一致：`user_id` 是内部 id（`identity.user_id`），`session_id` 是被操作的那个会话。
- **不要用 `_member_code_enabled` 去门控这些钩子**：池没创建时它们本来就是空操作（不 import modal、不联网），门控反而会在测试替身里漏掉调用。`_member_code_enabled` 只用于启动提示与 `/whoami`。
- 不要从 `MainAgents` 读新增属性：`smoke_chat_service` 的 P14 用 `types.SimpleNamespace(owner, member, owner_experts, member_experts)` 伪造它。

### 8.2 改动清单

- **模块级状态**：新增 `_member_code_enabled: bool = False`。在 `init_service` 成功之后赋值；在 `_reset_runtime_state()` 里复位为 `False`（`shutdown_service` 与测试清理都会调它，记得补 `global` 声明）。
- **`init_service`**（打印两行「图专家」之后）：`_member_code_enabled = 'CodeAgent' in built.member_experts`（字面量就是 `code_agent.NAME`；以真正编译出来的结果为准，不要自己再算一遍门控）。随后打印一行：启用时「☁️ \[chat\_service\] member 云端 CodeAgent：已启用（Modal）」；未启用时「☁️ \[chat\_service\] member 云端 CodeAgent：未启用（原因）」，原因取 `modal_sandbox.member_code_status()[1]`（静态、不联网、不抛，只用于打印）。
- **`shutdown_service`**：在 `_stop_all_turns()` 的 try 块之后、`mcp_service.shutdown_mcp()` 之前，新增一个独立的 try 块：`await asyncio.to_thread(modal_sandbox.shutdown_pool)`，失败打印「⚠️ 警告：云端沙箱销毁失败 (非致命): …」。放在前面是因为云端沙箱按时计费：后面的 MCP / 浏览器 / Milvus 关闭万一抛异常，不能把它漏掉（`atexit` 兜底只管正常退出）。
- **`_cmd_reset`**（替换 `# STAGE2: member 的云端沙箱重置`）：`if identity.is_owner:` 的原分支不动；新增 `else:` 分支 `await asyncio.to_thread(modal_sandbox.reset_session, uid, sid)`（`sid` 就是函数开头取到的当前会话 id），失败打印「⚠️ 警告：云端沙箱重置失败 (非致命): …」。`/reset` 遇到繁忙时 `process_message` 会先 `_stop_current_turn` 再进 `_cmd_reset`，两者可以并发，池保证幂等。
- **`_stop_current_turn`**（替换 `# STAGE2: member 的云端沙箱中止`）：在 `if user_id == OWNER_USER_ID:` 之后加 `else:`，调 `modal_sandbox.interrupt_session_nowait(user_id, handle.session_id)`，包 try / except，失败打印「⚠️ \[/stop\] 中断云端沙箱失败 (非致命): …」。顺序仍是**先 `task.cancel()` 再杀进程**，现有顺序不要动。**member 的 /stop 绝不能走到 `_kill_side_effects`、`host_shell.kill_all_running()`、`sandbox.interrupt_sandbox()`**：它们是进程级全局的，会杀掉 owner 正在跑的命令。`_stop_all_turns`（关闭服务）与 `_cmd_block` 都经由 `_stop_current_turn`，不需要再单独处理。
- **`wake()` 的失败分支**（`except (Exception, asyncio.CancelledError) as e:` 里，在 `raise` 之前）：回合因超时 / 递归上限 / 异常 / 非 /stop 的取消而失败时，已经在沙箱里起了的命令不会因为 asyncio 任务被取消而停下（它跑在线程里）。所以加：`if not is_owner and reason != 'user_stop':` → `modal_sandbox.interrupt_session_nowait(user_id, session_id)`（try / except，失败打印警告）。`user_stop` 已由 `_stop_current_turn` 处理，不重复。
- **`_cmd_block`**（`blocked=True`，在 `_stop_current_turn` 之后）：`n = await asyncio.to_thread(modal_sandbox.reset_user, user.user_id)`（try / except）；`n > 0` 时在回复末尾追加「已回收它的 {n} 个云端沙箱。」。owner 走不到这里（`set_user_blocked` 对 owner 抛 `ValueError`，上面已经返回）。`/unblock` 不用动。
- **`_cmd_whoami`**：「CodeAgent 后端」一行，owner 仍是「本机 Docker」；member 改成 `'云端沙箱（Modal）' if _member_code_enabled else '未启用'`。
- **`/new`、`/switch`**：不改，**不终止沙箱**。沙箱按 `(user, session)` 存在，切走再切回来时文件还在，闲置到期后回收。
- **文案同步（三处，别的措辞不动）**：`config.py` 里 `OWNER_USER_IDS` 上方的注释、`chat_service._startup_notices()` 里「未配置 OWNER\_USER\_IDS」那一条、`main.py` 里 `--member` 的 help，都写着 member「没有……代码能力」，启用 CodeAgent 后不再准确：把「文件 / 命令 / 桌面 / 浏览器 / 代码能力」改成「文件 / 命令 / 桌面 / 浏览器能力；代码只能在云端沙箱里跑」。`test_p12_startup_notices` 断言的关键词（`OWNER_USER_IDS`、`/whoami`、`root`、`只读`）都保留。README 里的同类描述在 M7 处理。

### 8.3 不用改的与验收

- 不用改：`plugins/`、`bot.py`、`main.py` 的关闭流程（它们已经调用 `shutdown_service()`）；`recover_orphans()`（只处理会话事件，不涉及沙箱；§1.1 已定启动时不清理孤儿沙箱）；`MEMBER_WELCOME_TEXT` 的默认值（是否在 `.env` 里加一句「运行代码时数据会发到第三方云端」由用户自己决定，见 §11）。
- 验收：`rg -n "modal_sandbox" app/services/chat_service.py` 只出现在上面列出的位置；`rg -n "reset_sandbox|interrupt_sandbox|kill_all_running|shutdown_sandbox" app/services/chat_service.py` 的命中点全都仍在 owner 分支里；owner 的 `/reset`、`/stop`、关闭流程里没有任何 `modal_sandbox` 调用。

## 9. M6 测试：FakeModal 与各套件的新增 / 适配断言

本节规定新增什么、适配什么。全部离线；现有断言只适配、不削弱。

### 9.1 地基规则

- **一切离线、确定、不依赖机器环境**：不联网、不调用真实 Modal、不读 `.env` 的内容；调用 `modal` 包里任何会联网的函数都算违规。测试自己控制配置：直接 monkeypatch `config.MODAL_*` 属性（`modal_sandbox` 每次调用都现读 `config.*`），不要靠环境变量，因为用户机器上的 `.env` 里会有真实令牌。
- **会编译真实 member 图的套件**（目前只有 `tests/smoke_orchestrator.py`；用 `rg -n 'build_main_agents|_build_member_agent' tests` 再确认一遍），在文件最顶部、任何 `import app...` 之前加 `os.environ['MODAL_DISABLED'] = 'true'`（`load_dotenv` 默认不覆盖已有的环境变量，所以这一行能压过 `.env`）。否则在有真实令牌的机器上 member 图会多出 CodeAgent，A–H 全红。
- **时间**：池、熔断、沙箱的耗时 / 空闲 / 驱逐判断全部走可注入的 `clock`（`ModalSandboxPool(modal_mod, clock=...)` 把它传给它创建的每个 `ModalSandbox`），测试里用 `FakeClock`（`now()`、`advance(s)`），不要真等。真正要等的只有线程同步，用 `threading.Event`，不要用 `sleep` 猜时序。
- **不挂死**：任何会起线程、会阻塞的场景都带总时限（`Event.wait(timeout)`、`Thread.join(timeout)` 之后断言线程已结束），挂死要变成失败而不是卡住 CI；`smoke_modal.py` 整个套件在普通笔记本上 < 60 秒。
- 新套件沿用现有写法：开头 docstring 列出场景代码，`main()` 里逐个场景运行并统计，有失败就 `sys.exit(1)`，输出强制 UTF-8；用 `from _support import member_context, owner_context`（`tests/` 在 `sys.path` 里）。在 `tests/run_smoke.py` 的 `SMOKES` 里把 `tests/smoke_modal.py` 加在 `tests/smoke_sandbox.py` 之后。
- 现有断言**只适配、不削弱**。改动或新增的每一条现有断言，都要列进最终报告的「已适配断言」：文件::测试名、旧断言、新断言、原因。

### 9.2 `tests/_fake_modal.py`：一个忠实的 FakeModal

它的价值取决于有多像真的 SDK，尤其要重现 §3.4 里那些会让天真实现出错的行为。`FakeModal()` 的实例就当作 `modal` 模块，通过 `modal_sandbox.set_modal_module(fake)` 注入（传 `None` 恢复真实导入；它同时清空池单例并关闭审计 handler）。契约：

- **异常**：`fake.exception` 下有 §3.4 列出的全部异常，都继承 `fake.exception.Error`。`ConnectionError`、`TimeoutError` 是**与内建同名的新类、不继承内建**（和真实 SDK 一样），所以实现里误写成 `except ConnectionError`（内建）会在测试里漏接。
- **未知参数一律 `TypeError`**：`Sandbox.create`、`Sandbox.exec`、`Sandbox.list`、`App.lookup`、`Image.*` 的签名精确复刻 §3.4 的参数表，多传一个 SDK 不认识的关键字就抛 `TypeError`，这样实现里的笔误或过时参数能在测试里暴露。所有收到的参数原样记录：`fake.created[i].kwargs`、`sandbox.exec_calls[j].argv / .kwargs`。
- **沙箱生命周期**：`poll()` 活着时 `None`，死后 `int`；`terminate(wait=False)` 幂等，并让该沙箱上所有还在跑的进程按 `fake.terminate_mode` 收尾——`'end'`（流静默结束、`wait()` 返回 -1）或 `'raise'`（读流与 `wait()` 抛 `SandboxTerminatedError`）。§3.4 说过真实行为没验证过，所以 §9.3 里相关场景两种模式都要各跑一遍。`fake.advance(seconds)` 推进虚拟时间，到点按 `idle_timeout` / `timeout` 让沙箱死亡（之后 `exec` 抛 `SandboxTerminatedError`、`poll()` 返回 int）；每次 `exec` 刷新空闲计时。
- **exec 与进程**：`exec` 返回 `FakeProcess`：`stdout` / `stderr` 是可迭代的**字节块**流（分块大小可配置，能把多字节字符切在块边界、能输出非法 UTF-8、能输出很大的量），`wait()` 阻塞到进程结束并返回 int；`poll()` 未结束返回 `None`；**`returncode` 在 `wait()` 返回之前访问抛 `InvalidError`**；**没有 `kill` 属性**（访问抛 `AttributeError`，实现里一旦调用 `p.kill()` 就会暴露）；`fake.fire_exec_deadline(proc)` 模拟客户端兜底到点：`wait()` / `poll()` 返回 `-1`、流静默结束、**不抛任何异常**。
- **可编程的命令行为**：不要去解释真实的 shell。FakeModal 带一张行为表，用 `sandbox.on_exec(matcher, behavior)` 注册；`matcher` 看 argv（对 `timeout -k 3 N ...` 包装，匹配的是剥掉包装之后的内层命令）；`behavior` 可以是 `Out(stdout=b'', stderr=b'', exit=0, chunks=...)`、`Hang()`（一直阻塞，直到被终止 / 被软杀脚本命中 / 兜底到点 / `fire_inner_timeout()`）、`Raise(exc)`、`Sleep(seconds, then=...)`（推进注入的 `FakeClock`，用来造「耗时」）。内置默认行为：`mkdir -p`、`true`、`ls -1 /work`、`find /work/outputs ... -printf`（从假文件系统渲染出 `相对路径 TAB 大小 TAB mtime NUL`）、`timeout -k 3 N bash -c <cmd>` 与 `timeout -k 3 N python -u /work/.sandbox/snippet.py`（内层交给行为表；`fire_inner_timeout()` 让它以 124 退出）、`python3 -c <软杀脚本>`（记录脚本，按 `sandbox.softkill_effect` 配置成功 / 失败 / 挂起 / 抛异常）。未注册的命令默认 `Out(exit=0)` 并被记录（deepagents 的文件工具靠 `python3 -c` 预检，需要它返回 0）。
- **文件系统**：`sb.filesystem` 是基于 dict 的假文件系统，只接受绝对路径（相对路径抛 `SandboxFilesystemError`）；`write_bytes(data, path)`（**data 在前**）、`read_bytes`、`make_directory(path, create_parents=True)`、`remove`、`list_files`、`stat`；读不存在抛 `SandboxFilesystemNotFoundError`，读目录抛 `...IsADirectoryError`；单文件大小上限可配置（`fs_max_bytes`），超限抛 `...FileTooLargeError`。
- **创建与镜像**：`fake.on_create(...)` 可以让第 N 次创建成功 / 抛指定异常 / 被一个 `Event` 卡住（用来测创建超时与「迟到的沙箱必须被 terminate」）。镜像是否存在取决于 `fake.published` 里有没有这个名字，不存在时 `Sandbox.create` 抛 `NotFoundError`；`fake.publish_default_image()` 一步发布 `config.MODAL_IMAGE_NAME`。`Sandbox.list(app_id=, tags=)` 只返回活着且 tags 为子集的沙箱。`App.lookup(name, create_if_missing=False)` 对未知名字且不允许创建时抛 `NotFoundError`，成功返回带 `app_id` **属性**的 `FakeApp`，并记录调用次数。
- **Image**：不可变的链式记录器——`debian_slim(python_version=, force_build=)`、`apt_install`、`pip_install`、`run_commands`、`env`、`workdir`、`add_local_file(local, remote, copy=False)` 各返回一个追加了操作记录的新对象；`build(app)` 记录它是否发生在 `enable_output()` 上下文里；`publish(name)` 在没 `build` 过时抛 `InvalidError`，成功写入 `fake.published[name]`；`Image.from_name(name)` 是惰性的。
- **线程安全**：FakeModal 自己必须线程安全（池测试会并发调用）。
- **自检**：`smoke_modal.py` 的场景 S 先验证 FakeModal 自己的这些不变量。FakeModal 自己错了，后面所有场景都没意义。

### 9.3 `tests/smoke_modal.py`：场景

- **S FakeModal 自检**：`returncode` 提前访问抛错；兜底到点返回 -1 且静默；没有 `kill`；未知参数 `TypeError`；同名异常不继承内建；`publish` 要先 `build`；`write_bytes` 的参数顺序。
- **A 真实 SDK 的表面**（`find_spec('modal')` 不为 `None` 才运行，没装就打印 SKIP、不算失败；只做 `inspect.signature` 与属性存在性检查，不调用任何会联网的函数）：我们实际调用的每个接口——`Sandbox.create` 的关键字（`app`、`tags`、`image`、`env`、`timeout`、`idle_timeout`、`workdir`、`cpu`、`memory`、`block_network`）、`Sandbox.exec` 的关键字（`timeout`、`workdir`、`text`）、`Sandbox.list(app_id, tags)`、`App.lookup(name, create_if_missing)`、`Image.from_name`、`Image.debian_slim(python_version, force_build)`、`apt_install`、`pip_install`、`run_commands`、`env`、`workdir`、`add_local_file(local, remote, copy)`、`Image.build`、`Image.publish`、`modal.enable_output`、`sb.filesystem` 的四个方法、`sb.terminate(wait=)`、`sb.poll`、`modal.exception` 里的那串异常名——真实 SDK 都接受或存在。反方向也查：FakeModal 里同名接口的每个参数名都必须在真实签名里存在（Fake 不得接受真实 SDK 不认识的参数）。`ContainerProcess` 有没有 `kill`：只打印记录，不断言（见 §3.4）。
- **B 配置与门控**：用 `smoke_multiuser.py` 里 `_config_probe` 的做法（子进程里设环境变量、打印 `config.MODAL_*`）验证 §4.1 的夹紧与回退——非法数字回默认、request 大于 limit 夹到 limit、内存 limit ≥ 256、idle ≥ 60、lifetime ≥ exec 上限 + 60、total ≥ per\_user、`MODAL_INTERRUPT_MODE=bogus` → `soft`、空串名字回退默认；`MODAL_TOKENS_PRESENT` 只在两个令牌都非空时为真，且 `config` 的 `vars()` 里没有任何值等于假令牌的项。`member_code_status()` 的 5 个分支各自的返回值与原因（monkeypatch `config.MODAL_DISABLED` / `MODAL_TOKENS_PRESENT`、`importlib.util.find_spec`、`importlib.metadata.version`；版本串 `1.5.9` 与 `abc` 不可用，`1.6.0` 与 `2.0` 可用；§4.2 的实现要用这两个函数的模块属性调用写法，否则替换不到）；原因里不含令牌值；调用期间把 `socket.socket.connect` 换成抛异常的替身，证明它不联网。**惰性导入**：开子进程执行 `import app.infra.modal_sandbox, app.infra.modal_image, app.tools.code_tools, app.agents.experts.code_agent`，断言 `modal` 不在 `sys.modules`；同一子进程里再调 `member_code_status()`，`modal` 仍不在。
- **C 创建参数与环境**：往 `os.environ` 里塞一批假的敏感变量（`MODAL_TOKEN_ID`、`MODAL_TOKEN_SECRET`、`GEMINI_API_KEY`、`MYSQL_PASSWORD` 等），触发一次创建，断言 `Sandbox.create` 收到的恰是 §5.4 列出的那一组：`args == ('sleep', 'infinity')`、`app` 是 `lookup` 返回的那个、`image` 是 `Image.from_name(config.MODAL_IMAGE_NAME)`、`env` 恰为 4 项白名单且不含任何假敏感变量、`timeout` / `idle_timeout` / `workdir` / `cpu` / `memory` / `block_network` 与 `config` 一致、`tags` 含 `svc=gnk48` 与 `u` / `s` 的 sha256 前缀（不含明文 id）、**没有** `secrets` / `volumes` / 端口 / `pty` / `name`；创建后做过 `mkdir -p /work/outputs /work/.sandbox`；多次创建里 `App.lookup` 只调用一次。镜像不存在（`fake.published` 为空）→ 创建失败，运营者文案指向 `scripts/build_modal_image.py`，给模型的文案是通用短句。
- **D 池、租约与驱逐**：同一 `(u, s)` 复用同一沙箱，不同会话各一个；每用户上限（默认 2）→ 第 3 个会话创建时驱逐该用户最久空闲且不忙的那个；全局上限同理跨用户按 LRU；**全部都在忙**时拒绝，且两种拒绝文案不同（每用户上限 / 全局繁忙）、审计里是 `reject:cap_user` / `reject:cap_total`；**忙的条目绝不被驱逐**（线程 A 持有租约不放，线程 B 触发驱逐，A 的沙箱仍存活且未被 terminate）；`lease(create=False)` 没有条目时给 `None` 且不创建；`reset_session` / `reset_user` 的返回值与被 terminate 的次数；`shutdown()` 之后再 `lease` 抛 `SandboxUnavailable`，`shutdown` 并行关闭，一个沙箱的 `terminate` 卡住也不会让整体超过 `deadline_s`；**池锁绝不跨网络调用持有**（让一次创建被 Event 卡住，同时从另一线程 `reset_user` 其他用户，必须立即完成）。
- **E 执行语义**（§5.3 的表逐行）：argv 恰为 `['timeout', '-k', '3', str(N), 'bash', '-c', cmd]` 与 `['timeout', '-k', '3', str(N), 'python', '-u', '/work/.sandbox/snippet.py']`，`workdir='/work'`、`timeout=N+12`、`text=False`；退出码 0 / 非零 / 124（`timed_out=True` + note）/ 137 且耗时 < N（内存 note）/ 137 且耗时 ≥ N（`timed_out`）/ 兜底到点 -1（`timed_out=True`、`exit_code=None`，随后做健康检查：通过 → 沙箱保留；失败 → terminate 并带「沙箱已重置」note）；`exec` 抛 `SandboxTerminatedError` → 沙箱标记已死、返回带 note 的结果、**同一条命令只发出一次**（`exec_calls` 里不重复），下一次调用惰性重建并带一次性「已重建」note；`terminate_mode` 的 `'end'` 与 `'raise'` 都不能让运行中的命令卡死；`timeout` 参数夹紧（`None` / `0` / `-5` → 默认，`9999` → 上限）；输出超过 `SANDBOX_EXECUTE_MAX_OUTPUT_CHARS` → `truncated=True`；被切在块边界的多字节字符与非法 UTF-8 不抛、不丢；流的原始字节硬上限（模块级常量 `_STREAM_HARD_CAP_BYTES`，测试里调小）触发一次中断并带「输出量过大」note；`run_python` 的 256 KiB 代码上限；同一沙箱里并行的两次 `execute` 被 RLock 串行化。
- **F 永不抛异常**：对 `Sandbox.create`、`exec`、读流、`wait`、`poll`、`terminate`、`filesystem.*` 逐个注入各类异常（`Error`、modal 的 `ConnectionError`、`OutputExpiredError`、`SandboxTerminatedError`、`RuntimeError`、`MemoryError`），断言 `execute` / `upload_files` / `download_files` / `collect_new_outputs` / `interrupt` / `close` 都不抛、返回值形状正确，读流线程里的异常不会让主线程等不到结果（总时限）；`run_python` 在创建失败时抛的是 `SandboxUnavailable`，消息就是 §5.4 的通用短句（不含异常类名、令牌、`modal` 内部细节）。
- **G 路径规则**：上传 `/work/a.txt`（成功）、`/work/../etc/x`、`/workfoo/x`、`/work`、`/tmp/x`（成功）、`/tmpx/x`、`/workspace/x`（`permission_denied`，文案提示写到 /work）、相对路径、空串、含 NUL、`/work/./a/../../etc/passwd`（规范化后越界）→ 各自的错误码；单文件超过 5 MiB → `file_too_large`；写入前对父目录做过 `make_directory(create_parents=True)`；下载的 `file_not_found` / `is_directory` / `file_too_large` / `permission_denied`。
- **H 产物收集**：指纹增量（没变的文件不重发、改动后再发）、图片扩展名白名单、非图片只报告不导出、单张大小 / 张数 / 像素上限与 notes（文案与 Docker 版一致）、500 条上限与 note、`find` 输出里的 `..` / 绝对路径被拒绝、沙箱未创建时用 `create=False` 且不创建沙箱、图片字节来自 `read_bytes`。
- **I 中止**：软杀脚本的逻辑写成 `modal_sandbox.py` 里一个自包含函数 `_kill_wrapper_trees(proc_root='/proc', *, self_pid=None, kill=None) -> list[int]`（只用标准库、不引用模块级名字），`python3 -c` 的脚本由 `inspect.getsource` 加一条调用语句拼出（§5.3 已补这一条）。测试在 `tmp` 目录里伪造一棵 `/proc`：PID 1（`sleep infinity`）、一个 `timeout -k 3 120 bash -c ...` 包装进程及其子孙、一个不相干的守护进程、脚本自己，其中一个进程的 `comm` 含空格和括号（如 `(my) proc (x)`）；注入假的 `kill`；断言杀掉的恰是包装树、没碰其余任何进程（尤其是 PID 1 与脚本自己）。池层面：`MODAL_INTERRUPT_MODE` 取 `soft` / `terminate` / 非法值（→ `soft`）；软杀后健康检查通过 → 沙箱与 `/work` 保留；脚本失败 / 抛异常 / 健康检查失败 / 超时 → 回退 `terminate`；沙箱还没创建时 `interrupt()` 返回 `False` 且不创建；**`interrupt()` 不取沙箱的 RLock**（一个线程持锁跑着命令，另一个线程调 `interrupt()` 必须在时限内返回，被中断的命令随后返回带 note 的结果）；`interrupt_session_nowait` 立即返回（即使底层 `interrupt()` 被卡住）且用的是守护线程；`close()` 幂等、不取锁。
- **J 熔断、创建超时与脱敏**：连续 3 次创建失败 → 开路，之后的创建请求不再调用 `Sandbox.create`（审计 `reject:breaker`）；`FakeClock` 推进 60 秒 → 放行一次试探，成功则恢复、失败则再断开 60 秒；已有沙箱上的命令失败不计数；运营者文案的分类（`NotFoundError`、`AuthError`、`ResourceExhaustedError`、`ConnectionError` / `ServiceError` / `TimeoutError`、其它 → 类名 + 截断）。**创建超时**：把 `config.MODAL_STARTUP_TIMEOUT_SECONDS` patch 成 0.3，fake 的 create 被 Event 卡住 → 创建失败并计入熔断；放开 Event 让迟到的沙箱创建成功 → 它必须被 terminate（`fake.live()` 为空）。**脱敏**：注入的异常文本里夹带 `ak-` / `as-` 形状的令牌串和 `MODAL_TOKEN_SECRET=xxx`，断言捕获到的控制台输出、审计日志文件内容、`SandboxUnavailable` 的消息、`ExecuteResponse.output` 里都没有这些串；给模型看的文案**精确等于** §5.4 的三句之一。
- **K 路由与 BaseSandbox 契约**：`ModalRoutedSandbox` 在没有上下文、owner 上下文、非法 `user_id`、空 `session_id` 时一律拒绝（审计 `reject:context`、不碰池）；member 上下文落到对应的 `(user_id, session_id)`；两个 member 并发使用互不串（各自 `/work` 里的文件不同）；**两条线程路径都要拿到上下文**：①异步路径 `await backend.aexecute(...)`（`asyncio.to_thread`）；②同步路径在 `ThreadPoolExecutor` 里用 `contextvars.copy_context().run(...)`（模拟 LangChain 的线程池）；反例：线程里**没有**带上下文时 fail-closed 拒绝；`id == 'modal-routed'`；实例不缓存任何用户 / 会话状态；契约：`isinstance(inst, SandboxBackendProtocol)`、`upload_files` / `download_files` 经 FakeModal 文件系统往返一致、`write()` 先预检再上传。
- **L 审计日志**：惰性创建（导入后、第一次写之前目录与文件都不存在）；一行一个 JSON、字段齐全；覆盖 §5.7 列出的全部事件；日志里没有明文 `user_id` / `session_id`、没有输出内容、没有环境变量、没有令牌形状的串（命令里夹带一个，断言已被 `_redact`）；命令只记前 500 字符；`RotatingFileHandler` 参数是 5 MiB × 3；`config.MODAL_AUDIT_LOG` 指向不可写位置（例如把一个普通文件当目录用）→ 审计自动关闭、控制台只提示一次、代码执行不受影响；`propagate` 为 `False`。
- **M 镜像定义与脚本**：`modal_image` 的常量；`build_image(fake, ...)` 产出的操作链顺序与内容（`debian_slim(python_version='3.13', force_build=...)` → `apt_install` → `pip_install`（同一层，含 `read_pinned_requirements()` 的全部条目与 `PIP_EXTRA`）→ `add_local_file(..., copy=True)` → `env` → `run_commands` → `workdir('/work')`）；`--with-optional` 追加 `APT_OPTIONAL`；`read_pinned_requirements()` 忽略空行与注释。`scripts/build_modal_image.main(argv)`（用 FakeModal 注入）：`--help` 离线、退出 0，且子进程里 `modal` 不在 `sys.modules`；无令牌 → 退出码 2；正常流程的调用顺序是 `App.lookup(create_if_missing=True)` → `build`（在 `enable_output` 里）→ `publish(name)`；没有 `--yes` 时要交互确认（monkeypatch `input`）；构建失败 → 退出码 1 且消息已脱敏；`--check-only` 不 build 也不 publish，镜像不存在 → 明确文案 + 退出码 1；验收每一项失败都 → 退出码 1；**任何路径下临时沙箱都被 terminate**（含验收中途抛异常与 `KeyboardInterrupt`）。`scripts/manage_modal_sandboxes.main(argv)`：`list`（只列 `svc=gnk48` 标签的，只显示 id 与哈希标签）、`kill --all`、`kill --user <前缀>`（前缀不匹配时不杀任何东西）、没有 `--yes` 时交互确认、`--help` 离线、任何输出都不含令牌。
- **N 静态扫描**（AST）：扫 `app/infra/modal_sandbox.py` 与 `app/infra/modal_image.py`——不得出现标识符 `DockerSandbox`、`get_sandbox`、`reset_sandbox`、`interrupt_sandbox`、`shutdown_sandbox`、`set_docker_client`；`modal_sandbox.py` 里不得调用 `.build(`、`.publish(`、`debian_slim(`；`import modal` 只允许出现在 `load_modal()` 函数体里；不得有裸名 `ConnectionError` / `TimeoutError` 的 `except`（要么用 modal 的同名类，要么用 `Exception`；确有合理用法就在报告里说明并加白名单）；`time.monotonic` / `time.time` 只允许出现在 `clock=` 的默认值与审计的 `ts` 里。全仓 `app/` 里读取 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` 的只有 `config.py`，并且只转成布尔值。
- **O member 的 `execute_python_code` 工具**（`code_tools.build_member_tools()`）：在 `member_context()` 里调用 → 返回含状态行与 Stdout 的文本；FakeModal 在 `/work/outputs/` 里放一张 PNG → 返回多模态内容、`outbox` 里有那张图、同一文件第二次不重发；在 `owner_context()` 里或没有上下文时 → 拒绝文本，池里没有创建任何沙箱；创建失败 → 返回通用短句文本、不抛；`description` 是 member 版（含 `config` 里的数字，不含 Docker、/workspace、workspace、宿主机、FileAgent、DesktopAgent、BrowserAgent）；owner 的 `execute_python_code` 对象与描述不受影响。
- **P 提示词与描述的事实核对**：`code_agent.build_member_prompt()`、`build_member_description()`、`code_tools.member_execute_python_description()` 里的数字与 `config` 一致（改 `MODAL_EXEC_MAX_TIMEOUT_SECONDS`、`MODAL_IDLE_TIMEOUT_SECONDS`、`MODAL_MEMORY_MB_LIMIT` 后重新调用，文本随之变化）；联网相关的句子随 `config.MODAL_BLOCK_NETWORK` 切换（按 §7.6 的固定短语断言：联网版的四个函数返回值各恰含一次 沙箱可以联网，断网版各恰含一次 沙箱不能联网，且不含 §7.6 列出的肯定句片段；四个函数是上面三个再加 prompts.build\_member\_main\_prompt(True)；build\_member\_main\_prompt(False) 在两种配置下都不含这两个短语）；`你是 CodeAgent` 恰好一次；§7.2 的行为红线都在；禁用词一个都没有；`/skills/` 被描述为只读。

### 9.4 现有套件的适配与新增断言

**`tests/smoke_orchestrator.py`**（A–H 针对 `MODAL_DISABLED=true` 的状态，保持原样通过）：

- 顶部加 `os.environ['MODAL_DISABLED'] = 'true'`（§9.1）。
- `_build()` 里 `spy` 对 member 图的识别，从 `prompt is MEMBER_MAIN_AGENT_PROMPT` 改成 `prompt is MEMBER_MAIN_AGENT_PROMPT or prompt is MEMBER_MAIN_AGENT_PROMPT_CODE`（身份比较）。
- F 里的 `poisoned()` 要接受参数：`def poisoned(code=None): subs = real_subs(code) ...`（`_build_member_agent` 现在会把 `code` 传给 `_member_subagents`）。F 的其它断言（13 个黑名单、没有 `vetted_code` 时预编译子专家一律 `RuntimeError`、报错文案）一个字不改。
- G 的 `patched` 列表追加 `(orchestrator.code_agent, 'build_member_code_agent_runnable')`：禁用状态下它也不应被调用。
- H 保持原样，并新增 H2：`MEMBER_MAIN_AGENT_PROMPT_CODE` 含识别短语、`【能力边界】`、`不要尝试委派`、`CodeAgent`、`沙箱`、`【图文分离】`、`【严格数据边界】`、`<long_term_memory>`、三个专家名与会话工具名，**不含** `FileAgent`、`DesktopAgent`、`BrowserAgent`、`workspace`、`宿主机`、`Docker`；两个 member 变体都不含 `台北市信义区市府路1号`。再新增 H3：把阶段 1 的 `MAIN_AGENT_PROMPT` 与 `MEMBER_MAIN_AGENT_PROMPT` 的 sha256（M0 里先算好）写成常量，断言 `MAIN_AGENT_PROMPT` 的 sha256 不变、`MEMBER_MAIN_AGENT_PROMPT` 把占位句换回台北例句之后的 sha256 等于阶段 1 的（证明只改了那一句）。
- 新增 I「启用 CodeAgent 的 member 图」：monkeypatch `config.MODAL_DISABLED=False`、`config.MODAL_TOKENS_PRESENT=True`，`modal_sandbox.set_modal_module(FakeModal())`（`find_spec` 与版本检查在 CI 里是真的、modal 已装；没装就先 monkeypatch 它们），再用会捕获参数的 `create_deep_agent` 替身调用 `orchestrator._build_member_agent(MemorySaver())`，断言：专家名恰为 Knowledge / Media / Map / Code；`system_prompt is MEMBER_MAIN_AGENT_PROMPT_CODE`；主管工具不变；子专家列表里恰有一个预编译的，名字是 `CodeAgent` 且 `runnable` 就是 `MemberCodeAgentBuild.runnable`；主管 `backend` 的 `/skills/` 路由是指向 `SKILLS_MEMBER_DIR` 的 `ReadOnlyFilesystemBackend`；同一次测试里 owner 图与禁用状态下完全一样（7 个专家、`MAIN_AGENT_PROMPT`）。然后依次令 `MODAL_DISABLED=True` / 令牌缺失 / `find_spec` 返回 `None` / 版本过低，每一种都回到 3 个专家与 `MEMBER_MAIN_AGENT_PROMPT`，且 `build_member_code_agent_runnable` 没被调用。
- 新增 F2「预编译子专家的严格例外」：拿一个真实的 `MemberCodeAgentBuild`（用 FakeModal 构建）当 `vetted_code`，基线通过；然后逐个破坏并断言 `RuntimeError`——`runnable` 身份对不上、名字不是 `CodeAgent`、整张图里多一个预编译子专家、`tool_names` 多出 `delete` 或少一个、`backend.default` 换成 `DockerSandbox` 的替身、路由多出 `/workspace/`、`/skills/` 路由换成可写的 `FilesystemBackend` 或指向 owner 的 `SKILLS_DIR`、`/_artifacts/` 路由不是 `StateBackend`；再确认 13 个黑名单工具注入主管 / 声明式子专家时，**即使带着 `vetted_code`** 也照样 `RuntimeError`。
- 新增 G2：CodeAgent 启用时，member 构建也**不得**调用 `build_code_agent_runnable`（owner 的）、`sandbox.get_sandbox`、`sandbox.set_docker_client`、`markitdown_tool.build_tools`，也不得读取任何 MCP 工具（`_mcp_requests == []`）。

**`tests/smoke_skills.py`**：对 `skills_member/*/SKILL.md` 做与 `skills/` 同样的 frontmatter 规范检查；断言其中不含 `Docker`、`/workspace`、`workspace`、`宿主机`、`FileAgent`、`DesktopAgent`、`BrowserAgent`；`build_member_skills_backend()` 的 `/skills/` 路由是 `ReadOnlyFilesystemBackend`、根目录是 `skills_member/`；owner 的 `build_skills_backend()` 不变。`ReadOnlyFilesystemBackend` 直接测：`write` / `awrite` / `edit` / `aedit` / `delete` / `adelete` / `upload_files` / `aupload_files` 全部返回带 `error` 的结果，且**磁盘上没有任何变化**（对 `skills_member/` 目录做前后哈希快照）；读类方法（`ls` / `read` / `grep` / `glob` / `download_files`）仍可用；经 `CompositeBackend` 做路径穿越（`/skills/../.env`、`/skills/a/../../x`、`/skills/~`）读 / 写 / 列目录都拿不到 `skills_member/` 之外的东西。

**驱动真实编译图的越权测试**（放进 `smoke_modal.py` 的场景 K2，或 `smoke_sandbox.py`，二选一）：用 `build_member_code_agent_runnable(llm=<脚本化假模型>, system_prompt=code_agent.MEMBER_PROMPT)` 编译出真实的 member CodeAgent（模型替身照抄 `smoke_chat_service.py` 里 `ScriptedModel` 的写法，约 20 行；组装与取工具表的方式照抄 `smoke_sandbox.py` 的 `test_r_code_agent_assembly_and_routing`），在 member 上下文里让它依次发出：`write_file` 写 `/skills/matplotlib-chart/SKILL.md`、`edit_file` 改同一路径、`read_file` 读 `/skills/../.env`、`ls` 列 `/skills/..`、一个名为 `delete` 的工具调用、一个会触发大结果转存的调用（阈值难触发就只断言路由配置，并在报告里说明）。断言：所有写 / 改都返回只读错误文本；`skills/` 与 `skills_member/` 下每个文件的哈希前后一致；越界读取拿不到内容；`delete` 工具不存在；转存只写进了 `StateBackend`，宿主机磁盘与 FakeModal 的文件系统里都没有它；工具名集合恰为 8 个。

**`tests/smoke_chat_service.py`**（P4、P6、P12、P14 与 O / Q 系列的现有断言继续有效）：新增场景 R，通过替换 **`app.infra.modal_sandbox` 模块上的属性**（`reset_session` / `reset_user` / `interrupt_session_nowait` / `shutdown_pool` / `member_code_status`）来观察钩子，每个替身记录调用参数和**调用时所在的线程**。

- R1 `/reset`：member → `reset_session(member.user_id, 当前 session_id)` 恰好一次，且在事件循环线程之外执行，`sandbox.reset_sandbox` 没被碰；owner 反过来。钩子抛异常时回复文本不变。
- R2 `/stop`：member 的 `/stop` → `interrupt_session_nowait(member.user_id, handle.session_id)` 一次，`host_shell.kill_all_running` 与 `sandbox.interrupt_sandbox` 都没被调用；owner 的 `/stop` → `modal_sandbox` 的任何函数都没被调用（扩展 Q2）；member 的 `/stop` 不影响 owner 与其他 member 的轮（Q1 / Q3 保持）。
- R3 `wake` 失败分支：member 的回合超时（把 `AGENT_TURN_TIMEOUT_SECONDS` 调小）→ `interrupt_session_nowait` 恰好一次；用户 `/stop` 引起的失败 → 总共仍然只有一次（来自 `_stop_current_turn`，不重复）；owner 失败 → 没有调用。
- R4 `/block`：member 有一轮在跑 → 先中止、再 `reset_user(member.user_id)` 一次；`reset_user` 返回 2 → 回复里有「已回收它的 2 个云端沙箱」；返回 0 → 回复与阶段 1 逐字一致；对 owner 的 `/block` → 「管理员不可被封禁。」且没有任何调用。
- R5 关闭：`shutdown_service()` 调用 `shutdown_pool` 恰好一次，且在 `mcp_service.shutdown_mcp` 之前；`_stop_all_turns` 抛异常或 `shutdown_pool` 自己抛异常时，后续的关闭步骤（checkpointer 等）照样执行。
- R6 启动与 `/whoami`：沿用 P14 的 `fake_build`，让 `member_experts` 带上 `CodeAgent` → `cs._member_code_enabled is True`、启动横幅已打印、member 的 `/whoami` 里是「云端沙箱（Modal）」、owner 的仍是「本机 Docker」；不带 `CodeAgent` → 「未启用」，并打印 `member_code_status()` 的原因；`shutdown_service()` 之后标志复位。
- R7 文案：`_startup_notices()` 在 `OWNER_USER_IDS` 为空时的那一条不再含「代码能力」、含「云端沙箱」，P12 原有断言的关键词都在。

**`tests/smoke_architecture.py`**：D 里追加 `ident(code_agent.MEMBER_PROMPT) == 'CodeAgent'`、`ident(MEMBER_MAIN_AGENT_PROMPT_CODE) == 'MainAgent (Orchestrator)'`，并把 `MEMBER_MAIN_AGENT_PROMPT_CODE`、`code_agent.MEMBER_PROMPT`、`code_agent.MEMBER_DESCRIPTION` 加进「不得出现 SQLAgent」的循环；`code_agent.MEMBER_PROMPT` 里 `你是 CodeAgent` 恰好出现一次。E 里追加 `build_member_code_agent_runnable`、`MemberCodeAgentBuild`、`MEMBER_DESCRIPTION`、`MEMBER_PROMPT` 存在。F 的轻量导入清单加上 `app.infra.modal_sandbox` 与 `app.infra.modal_image`，并把 `modal` 加进它的 `forbidden` 检查。A / B 会自动覆盖新文件的分层与导入规则，**不要为了让新文件通过而放宽这些规则**，新文件违规就改新文件。

**`tests/smoke_sandbox.py`**：`DockerSandbox` 与 `code_tools` 的现有测试一律不改；新增一个断言：模块级的 `code_tools.TOOLS` 与 `execute_python_code.description`（docstring，字节级）和阶段 1 完全一致——M0 先把该描述算成 sha256 常量写进测试。

### 9.5 验收

- `uv run python tests/run_smoke.py` 全绿（17 个套件：原 16 个加 `smoke_modal.py`），§1.2 的 ruff 命令干净。
- 整套测试不依赖网络、令牌与 `.env`：在没有 `.env`、也没有任何 `MODAL_*` 环境变量的干净 checkout 里，和在把 `MODAL_TOKEN_ID`、`MODAL_TOKEN_SECRET` 设成任意假值的环境里，结果一致，并确认没有任何测试尝试联网。
- 最终报告的「已适配断言」表与 `git diff` 里被改动的现有断言一一对应，没有削弱。

## 10. M7 真机验收脚本、README 与 .env.example 收尾

### 10.1 `tests/live_modal.py`（真机验收，只由用户运行，Claude Code 不得运行）

目的：把我在 §3.4 / §5.3 里「没验证过」的几件事在真实 Modal 上量出来，结果交给我决定 `MODAL_INTERRUPT_MODE` 等默认值。写法照 `tests/live_sandbox.py`：

- 文件头 docstring、强制 UTF-8 输出；`main()` 里**没有 `--live` 就打印用法并以退出码 0 退出，此时不 import `modal`、不联网**；`--help` 离线。用法：`uv run python tests/live_modal.py --live [--slow]`，`--slow` 开启需要等待空闲回收的那一步。开头沿用 `from app import bootstrap` + `bootstrap.apply()`（参照 `scripts/manage_sessions.py`）。
- 前置检查：`member_code_status()` 必须为 `(True, '')`，否则打印原因并以退出码 2 退出。
- 身份：用 `cli_member_identity('livetest')` 得到的 `user_id`，在真实的 member 上下文里调用真实的 `ModalRoutedSandbox` 与池（不注入 FakeModal）。需要直接探 SDK 的步骤（下面的 5、8），再用 `modal_sandbox.load_modal()` 单独建一个「原始沙箱」（同样的镜像与资源护栏，`timeout=300`、`idle_timeout=60`，带 `svc=gnk48` 标签，漏了也能被 `manage_modal_sandboxes.py` 看到），用完在 `finally` 里 terminate。
- 输出：每个步骤打印 `[Case N] 描述` 与 PASS / FAIL；所有量出来的值都用一行 `MEASURE key=value` 打印，结尾再集中打印一个 `=== MEASUREMENTS ===` 块（用户整块复制给我）。\*\*不打印令牌，也不打印环境变量的值（只打印名字）。\*\*任何步骤 FAIL 不中断其余步骤，最后汇总并以退出码 1 结束；`KeyboardInterrupt` 也要走到 `finally` 清理。
- 步骤：
  1. 冷启动：第一次 `execute('echo hi')` 的耗时（含 `App.lookup` + 创建 + `mkdir`）→ `MEASURE cold_start_s`；第二次热调用 → `MEASURE warm_exec_s`。
  2. 退出码与输出：`exit 3` → 退出码 3；输出中文与 stderr 的往返是 UTF-8 正确的。
  3. 环境核对：`python --version`、`id -u`、`pwd`、`$HOME`、`/work/outputs` 可写；沙箱环境里**没有**任何名字含 `MODAL`、`KEY`、`TOKEN`、`SECRET`、`PASSWORD` 的变量（只打印变量名清单）。
  4. 文件往返：`upload_files` 一个含中文的文件到 `/work`，`download_files` 取回字节一致；写 `/workspace/x` 被拒；5 MiB 边界（`5 MiB` 成功、`5 MiB + 1` 拒绝）。
  5. 超时（两个层次）：① 通过我们的封装 `execute('sleep 30', timeout=5)` → `timed_out=True`、退出码 124、总耗时 → `MEASURE timeout_wall_s`，且沙箱与 `/work` 里的文件还在 → `MEASURE work_survives_timeout`；② 用「原始沙箱」直接 `exec('sleep', '60', timeout=5)`，验证 §3.4 说的“到点后 `wait()` 返回 -1、流静默结束、不抛异常” → `MEASURE raw_exec_timeout_returncode`、`MEASURE raw_exec_timeout_raises`。
  6. 产物：`run_python` 画一张含中文与负号的图到 `/work/outputs/a.png` → `collect_new_outputs` 拿到 1 张 PNG，第二次扫描不重发。
  7. 内存上限：`run_python` 里分配 `MODAL_MEMORY_MB_LIMIT` 的 1.5 倍（别再大）→ 观察退出码与 note；之后沙箱是否还活着、`/work` 是否还在 → `MEASURE oom_exit_code`、`MEASURE oom_sandbox_survives`。
  8. **软杀实测**（决定 `MODAL_INTERRUPT_MODE` 默认值是否安全）：在受测沙箱里开一个线程跑 `execute('sleep 300')`，3 秒后主线程 `interrupt()`（`soft`）。测：中断耗时 → `MEASURE soft_interrupt_s`；被中断的命令返回了什么（退出码与 note）；紧接着 `execute('echo alive; ls /work')` 是否正常且文件还在 → `MEASURE soft_survives`；**连续重复 3 轮**（看 exec 通道会不会被逐渐弄坏）→ `MEASURE soft_repeat_ok`。再用 `terminate` 模式重复一次：`MEASURE terminate_interrupt_s`、重建耗时 `MEASURE rebuild_s`、重建 note 文案。再用「原始沙箱」试一次 `terminate()` 之后正在读的流的行为（到底是静默结束还是抛异常）→ `MEASURE terminate_stream_behavior`。
  9. 网络：`python -c` 里 `urllib.request.urlopen('https://pypi.org', timeout=10)` → `MEASURE network`（与 `MODAL_BLOCK_NETWORK` 对照）；`pip --version` 能跑。
  10. 池护栏：同一用户开 3 个会话（上限 2）→ 第 3 个创建时驱逐最久空闲的（审计里有 `terminate` / `evict`）→ `MEASURE evicted`。
  11. 审计日志：`config.MODAL_AUDIT_LOG` 存在、一行一 JSON、不含明文 user\_id / session\_id、不含令牌形状的串。
  12. （`--slow`）空闲回收：等 `MODAL_IDLE_TIMEOUT_SECONDS + 30` 秒后再执行 → 出现「已重建」note、`/work` 被清空 → `MEASURE idle_rebuild`。
  13. 收尾：`shutdown_pool()`，用 `Sandbox.list(app_id=..., tags=...)`（签名以已安装版本为准）检查没有残留（包括「原始沙箱」）→ `MEASURE leftover`；打印本次创建了多少个沙箱、沙箱总运行大约多少分钟（让用户对费用有数）。

### 10.2 README.md

用中文、沿用现有文风，逐处改，不重写无关段落：

- **2.3 子专家清单**里的 CodeAgent：加一句——owner 用本机 Docker 沙箱，member 用 Modal 云端沙箱（见 2.4），两者代码路径完全分开。
- **2.4 多用户模式**：① 能力矩阵表里 CodeAgent 一行改成「owner：有（本机 Docker）；member：有（Modal 云端沙箱，需配置令牌）」；② 「member 永远碰不到」那条把「member 本阶段没有任何代码执行能力」改成「member 的代码只能在 Modal 云端沙箱里跑，绝不会落到本机 Docker」；③ 「启动期断言」那条改成「member 图的专家名是 Knowledge / Media / Map 再加可选的一个经过验证的 CodeAgent；预编译子专家只允许这一个，且工具集、后端路由都严格校验」；④ 新增小节「**member 的云端 CodeAgent（阶段 2）**」：编译期门控（`member_code_status()`：总开关、令牌、modal 包、版本）、永不回退 Docker、每个 (用户, 会话) 一个惰性创建的沙箱及其生命周期（`/new`、`/switch` 不动它，`/reset` 终止当前会话的沙箱，空闲超时由 Modal 回收，owner `/block` 回收该用户全部沙箱，关闭服务时全部终止）、资源与并发护栏、沙箱可联网、令牌只在宿主机 `.env`、`/skills/` 对 member 只读（`skills_member/`）、审计日志、`/stop` 只中止自己那个会话的命令、费用上限要在 Modal 控制台设（应用层没有按用户的配额）。
- **2.9 / 2.10**：读一遍，凡是写着「沙箱只属于 owner」「member 没有沙箱」的句子，改成与现状一致：Docker 沙箱仍然只属于 owner，member 用 Modal（见 2.4）；如果「演进路径」里写过这件事是计划，改成「已在阶段 2 实现」。
- **3 目录结构**：按现有风格补 `app/infra/modal_sandbox.py`、`app/infra/modal_image.py`、`skills_member/`（member 只读技能）、`scripts/build_modal_image.py`、`scripts/manage_modal_sandboxes.py`、`tests/_fake_modal.py`、`tests/smoke_modal.py`、`tests/live_modal.py`，各带一句说明；`requirements.txt` 那行不动。
- **4.2 环境变量**：加一条 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` + `MODAL_*` 旁路可调项（不配令牌则 member 没有代码能力、行为与阶段 1 一致），指向 `.env.example`。
- **5.1** 的 `--member` 注释把「没有……代码能力」改成「没有文件 / 命令 / 桌面 / 浏览器能力；代码只能在云端沙箱里跑」，与 `main.py` 的 help 一致。新增 **5.5 构建 Modal 镜像与管理云端沙箱**：`uv run python scripts/build_modal_image.py`、`--check-only`、`scripts/manage_modal_sandboxes.py list / kill`、`tests/live_modal.py --live`，并写明“镜像内容变了就升 tag（`:v2`）并改 `MODAL_IMAGE_NAME`，不要原地覆盖”。
- **8 测试体系**：套件总数与清单加上 `smoke_modal.py`（以及 `_fake_modal.py` 的作用），写明 `live_modal.py` 不在一键运行里，需要真实令牌、会产生少量 Modal 费用。
- **9 已知限制**：改三条现有的——第 16 条（沙箱是进程级全局、只属于 owner）改成「这是 owner 的 Docker 沙箱；member 的云端沙箱按 (用户, 会话) 隔离，`/reset` 只重置当前会话的那个」；第 20 条补一句 member 的 `/stop` 只中断自己会话的云端命令；第 23 条把「member 没有……代码能力」改成「member 的代码与其中的数据会发到 Modal 云端处理（沙箱默认可联网）」。再新增几条（编号接在现有最后一条之后）：费用上限只能在 Modal 控制台设、应用层无按用户配额；进程被强杀（没走正常关闭流程）时云端沙箱最多存活 `MODAL_MAX_LIFETIME_SECONDS`，可用 `manage_modal_sandboxes.py` 清理；沙箱的 `/work` 在空闲回收、`/reset`、软杀回退为 terminate 时会丢失；第一条代码请求有冷启动延迟；中止方式（`MODAL_INTERRUPT_MODE`）以 `tests/live_modal.py` 的实测为准；镜像升级需手动重建并升 tag。

### 10.3 `.env.example`

在 `DOCKER_* / SANDBOX_*` 那一段之后加一段（写法、注释风格和现有的一致；**只放占位、没有任何真实值**）：

```
# ── Modal 云端沙箱（member 的 CodeAgent）──
# 不配置令牌则 member 没有代码能力，行为与阶段 1 一致。令牌只放在这个文件里，绝不会传进沙箱。
# 费用上限请在 Modal 控制台设置（应用层没有按用户的配额）。
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
# MODAL_DISABLED=false
# MODAL_APP_NAME=gnk48-code-sandbox
# MODAL_IMAGE_NAME=gnk48-python-ds:v1
# MODAL_CPU_REQUEST=0.25
# MODAL_CPU_LIMIT=2.0
# MODAL_MEMORY_MB_REQUEST=512
# MODAL_MEMORY_MB_LIMIT=2048
# MODAL_IDLE_TIMEOUT_SECONDS=600
# MODAL_MAX_LIFETIME_SECONDS=3600
# MODAL_EXEC_TIMEOUT_SECONDS=120
# MODAL_EXEC_MAX_TIMEOUT_SECONDS=300
# MODAL_STARTUP_TIMEOUT_SECONDS=120
# MODAL_MAX_SANDBOXES_PER_USER=2
# MODAL_MAX_SANDBOXES_TOTAL=8
# MODAL_BLOCK_NETWORK=false
# soft=只杀命令进程、保留 /work；terminate=直接终止沙箱（/work 丢失）。以 tests/live_modal.py 的实测为准
# MODAL_INTERRUPT_MODE=soft
# MODAL_AUDIT_LOG=storage/modal_audit.log
```

### 10.4 其它收尾

- `.gitignore`：审计日志默认在已被忽略的 `storage/` 下（轮转文件 `modal_audit.log.1` 等也在里面），不需要改；用 `git check-ignore -v storage/modal_audit.log storage/modal_audit.log.1` 确认即可。
- `requirements.txt` 是 `uv export` 生成的（文件头写了命令，请勿手改）。M1 的 `uv add` 之后用它头部给出的那条命令重新导出，`git diff requirements.txt` 里应当只有 modal 及其传递依赖的增量。
- 不要动 `deploy/sandbox/*`（owner 的 Docker 镜像）。阶段 1 的报告文件（仓库根目录的「多用户改造 阶段 1…」那份 .md）也不要改。
- 所有改动留在工作区，不要 commit、push、stash。

## 11. 最终报告、需要用户手动做的事、验收清单

### 11.1 最终报告（全部完成后在对话里输出，不要写进仓库）

用中文、按下面 10 项的顺序写，每项都要有，没有内容的写「无」。用户会把这份报告整份拿去跟我复核，所以写事实、不写过程，不要复述这份文档。

1. **一句话结论**：做到哪一步、测试是否全绿。
2. **M0 核对表**：§3 的每一条核对项→实际结果（一致 / 有出入，出入要写清楚你怎么调整的）。必须包括：基线 HEAD 与 `git status`；M0 里算出的三个 sha256（`MAIN_AGENT_PROMPT`、阶段 1 的 `MEMBER_MAIN_AGENT_PROMPT`、`execute_python_code` 的描述）；deepagents 签名核对结果；Modal SDK 事实表逐条核对结果（尤其是 `returncode` 在 `wait()` 之前、exec 超时返回 -1、`ContainerProcess` 有没有 `kill`、`terminate` 之后读流的行为——能从源码读出多少就写多少）。
3. **与本文的差异**：任何偏离本文的地方（设计、命名、参数），每条写「本文怎么写 / 实际怎么做 / 为什么」。
4. **未验证 / 无法离线验证的事项**：至少列出软杀在真实 Modal 上的安全性、`terminate` 之后流的行为、内存上限触发时的表现、冷启动耗时、镜像能否真的构建成功——这些都要靠用户运行脚本来定。
5. **改动清单**：新增文件、修改文件（各一行说明）；`pyproject.toml` / `uv.lock` / `requirements.txt` 的变化（新增的依赖及版本）。
6. **测试结果**：`uv run python tests/run_smoke.py` 的完整汇总（套件数、通过数）、§1.2 的 ruff 命令结果、`smoke_modal` 的场景通过情况、以及在「把 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` 设成任意假值」的环境里的一次复跑结果。
7. **已适配断言表**（§9.1）：文件::测试名 / 旧断言 / 新断言 / 原因。
8. **安全自查**：对照 §1.2 的硬规则逐条给证据——令牌不在沙箱环境里（测试名）、`rg` 结果证明 Modal 路径里没有 `DockerSandbox` / `get_sandbox` / `host_shell`、member CodeAgent 的工具集合、`/skills/` 只读的测试名、没有读取 `.env`、没有 commit。
9. **用户要手动做的事**：照 §11.2 写，但按你实际实现的脚本参数把每条命令写成可直接复制的形式。
10. **已知限制 / 后续建议**（≤ 5 条）。

### 11.2 用户要手动做的事（Claude Code 不能代做）

按顺序：

1. **开始前备份** `storage/`（尤其 `sessions.db*`、`checkpoints.db*`）。阶段 2 不改数据库结构，但备份永远值得。
2. **Modal 账户**：保持你已经设好的工作区 `dabaigou8964` 预算 $30 与 $0 超额支出上限不变；在 Modal 控制台（Settings → API Tokens）生成令牌，把 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` 写进本机 `.env`。令牌只写在 `.env` 里，不要发给 Claude 或 Claude Code，也不要贴到聊天里。
3. Claude Code 完成后：先 `git status` / `git diff --stat` 看一眼改动范围，然后跑离线测试：

```
uv sync
uv run python tests/run_smoke.py
```

4. **构建镜像（只做一次，要几分钟、产生少量费用）**：

```
uv run python scripts/build_modal_image.py
uv run python scripts/build_modal_image.py --check-only
```

第一条会先要你确认；第二条的验收应当全部 ✅。如果脚本最后提示镜像名与 `.env` 里的 `MODAL_IMAGE_NAME` 不同，按提示改 `.env`。

5. **真机验收脚本**：

```
uv run python tests/live_modal.py --live
```

可以再加 `--slow`（多等几分钟，测空闲回收）。把终端里 `=== MEASUREMENTS ===` 那一整块贴回给我，我来定 `MODAL_INTERRUPT_MODE`：默认 `soft`（只杀命令进程、保留 `/work`）；如果实测显示软杀会弄坏 exec 通道或不可靠，就在 `.env` 里改成 `MODAL_INTERRUPT_MODE=terminate`。

6. **QQ 真机验收**（用一个 member 账号，外加 owner 账号）：
   - member 让机器人算一道题、画一张图：应收到图片，回复文字里没有 `/work/...` 之类的路径。
   - member 发一个要跑很久的任务后 `/stop`：只停自己的，此时 owner 同时在跑的任务不受影响。
   - member `/reset` 之后再让它读之前生成的文件：应该没有了（沙箱被重置）。
   - 两个 member 账号并发各用一次：互相看不到对方的文件。
   - owner 发 `/whoami`：CodeAgent 后端显示「本机 Docker」，owner 的代码仍然走 Docker；member 发 `/whoami` 显示「云端沙箱（Modal）」。
   - member 要求读写本机文件或操作桌面：应如实说做不到。
   - 把机器人正常关闭后，用 `uv run python scripts/manage_modal_sandboxes.py list` 确认没有残留沙箱（也可以在 Modal 控制台看）。
   - 阶段 1 的真机验收清单再过一遍，确认没有退化。
7. **自己提交**：确认无误后自己 `git add -A`（要包含 `uv.lock`、`pyproject.toml`、`requirements.txt`）并 commit。Claude Code 不会 commit。
8. **（可选）欢迎语**：member 的代码与其中的数据会发到 Modal 云端处理。如果想告知朋友，在 `.env` 里用 `MEMBER_WELCOME_TEXT` 自己追加一句（默认值不由 Claude Code 改）。
9. **费用观察**：第一周留意 Modal 控制台的用量。沙箱按运行秒数计费，空闲 `MODAL_IDLE_TIMEOUT_SECONDS`（默认 600 秒）后自动回收；用量异常时跑 `uv run python scripts/manage_modal_sandboxes.py list`，必要时 `kill --all`。
10. **开新对话复核**：把 Claude Code 的最终报告与几个核心文件（`app/infra/modal_sandbox.py`、`app/agents/experts/code_agent.py`、`app/agents/orchestrator.py`、`app/services/chat_service.py` 的相关改动）连同 `live_modal.py` 的实测输出一起发给我，用来复核和调参。

### 11.3 验收清单（全部满足才算完成；Claude Code 自查，报告里逐条标明）

**功能**

- `member_code_status()` 为真时，member 图有 CodeAgent，代码请求走云端沙箱（FakeModal 测试证明，真机由用户验证）。
- `member_code_status()` 为假时，member 图与阶段 1 完全一致（专家集合与提示词除台北例句的替换外字节级相同）。
- owner 图不变：7 个专家、`MAIN_AGENT_PROMPT` 的 sha256 不变、`execute_python_code` 的描述与行为不变；owner 路径上零 Modal 调用。

**安全**

- 令牌：只在宿主机 `.env` 与 SDK 自己的环境变量读取，`config` 里只有布尔值；不进沙箱环境；不出现在日志 / 审计 / 异常文案 / 提示词里（场景 B、C、J、L 证明）。
- member 的 CodeAgent 从不触达 Docker / host\_shell / owner 的工具（G2、N 证明）。
- `/skills/` 对 member 只读、路径穿越被挡住（`smoke_skills` 与 K2 证明），owner 的 `skills/` 不被 member 图引用。
- member 图的编译期断言：工具名集合恰为 8 个、没有 `delete`、`backend.default` 是 `ModalRoutedSandbox`、路由集合恰为两条（F2 证明），13 个禁用工具名一个不减。
- 没有上下文 / owner 上下文 / 非法 id 时，`ModalRoutedSandbox` 一律拒绝（K 证明）。

**稳健性**

- 任何 Modal 异常都不会让工具调用抛出（F），也不会让机器人崩；熔断、创建超时、迟到的沙箱被终止（J）。
- 正常退出时所有沙箱被终止（R5 + 池的 `shutdown`）；`/reset`、`/block` 终止对应沙箱（R1、R4）。
- 池护栏：每用户 2、全局 8，忙的条目不被驱逐（D）。

**工程**

- `uv run python tests/run_smoke.py` 全绿（17 个套件），ruff 干净，`uv sync --locked` 通过（`uv.lock` 与 `pyproject.toml` 同步），`requirements.txt` 已重新导出。
- 架构测试 A / B / C / F / G 全绿且没有放宽规则；`modal` 是惰性导入（场景 B 的子进程证明）。
- 没有读取或打印 `.env`，没有动 `storage/`、`data/`，没有 `git commit` / `push` / `stash` / `reset`，没有运行 `scripts/build_modal_image.py` 和 `tests/live_modal.py --live`，没有调用真实 Modal。
- README 与 `.env.example` 已更新，新增文件都符合分层规则，最终报告的 10 项齐全。

### 11.4 卡住时怎么办

与本文的事实核对有出入时，按 §2 的「差异处理」往下做并写进报告，不要停下来等用户。遇到只有用户能拍板、且选错了很难撤回的决定（例如要放宽某条安全规则才能继续），选更安全的那一侧、把该项标成「未完成」并在报告里说明原因，而不是去放宽规则。
