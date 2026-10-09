# Deep Agents 迁移 · 下一轮提示词（A → B）

> 先发 **Prompt A**，验收通过后再发 **Prompt B**。
> 发 A 之前，先把同目录的 `hitl_replay_repro.py` 放进项目的 `scripts/` 目录。
> Prompt B 的 B2 第 3 条（纯文本、不用 Markdown）是我建议加的，不想要就删掉那一条再发。

---

## Prompt A —— HITL 重放安全修复 + 只读检查

**（从这里到 "## Prompt B" 之前，整段复制给 Antigravity）**

这轮有一个需要修的真实 bug（涉及 HITL 安全），外加几项只读检查。范围：只改 `app/graph.py` 里的两个子图构建函数以及同文件内的清理；不碰 `plugins/chat.py`、`main.py`、`mcp_servers.json`、`app/mcp_service.py`、`app/deep_agent.py`；测试不要用真实 MCP、真实桌面。之前"不动子图内部逻辑"的约束在这里解除，仅限下面列出的改动。

### 一、问题与证据

LangGraph 恢复中断时，会把被中断的**节点从头重新执行一遍**：`interrupt()` 立刻返回已有的恢复值，但它前面的代码（包括前几个已批准的工具调用）会再执行一次。

`fa_dangerous_tools_node` 在同一次节点执行里循环处理一条 AIMessage 的全部危险 tool_call，每个先 `interrupt()` 再执行。于是每恢复一次，之前已批准的工具就被重新执行一次。证据在 `walkthrough922.md` 的双重中断轨迹里：3 个危险调用，"✅ 执行成功"出现了 write_file×3、create_directory×2、move_file×1。那次测试"通过"，只是因为这几个工具恰好幂等。

后果：
- 非幂等工具（`edit_file`、`move_file`）：第二次执行时 oldText 已被替换，会报错，而这条报错会覆盖第一次成功的结果，模型会被告知"编辑失败"。
- `da_tools_node` 是同样的写法：一条消息里"点击 + 输入文字 + 高危快捷键"，快捷键触发 `interrupt()` 后恢复，点击和输入会被重复执行。
- 路由问题：`_route_after_llm` 只要有危险调用就只走 `fa_dangerous_tools`，同一条消息里的安全调用既不执行也没有 ToolMessage（tool_call 悬空）。

我在沙箱里（langgraph 1.2.12）用最小复现验证过这三点，也验证过下面的修复方案，见 `scripts/hitl_replay_repro.py`。

### 二、先复现

在你们的 `.venv` 里运行 `python scripts/hitl_replay_repro.py`，贴出输出。确认 OLD 的执行次数与下面一致：S1 write_file 3 / create_directory 2 / move_file 1；Desktop 的 click、safe_input_text 各 2。**如果你们环境里 OLD 没有复现，先停下来报告，不要继续改。**

### 三、修改要求（`app/graph.py`）

原则：**每次节点执行里最多出现一个 `interrupt()`，且在它之前不做任何有副作用的事；每个 tool_call 最终恰好得到一条 ToolMessage。**

模块级 helper（新增）：

```python
def _last_ai_with_calls(messages):
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            return i, m
    return -1, None

def _unanswered_tool_calls(messages):
    """最近一条带 tool_calls 的 AIMessage 里，后面还没有对应 ToolMessage 的调用。"""
    i, ai = _last_ai_with_calls(messages)
    if ai is None:
        return []
    answered = {m.tool_call_id for m in messages[i + 1:] if isinstance(m, ToolMessage)}
    return [tc for tc in ai.tool_calls if tc["id"] not in answered]
```

**FileAgent（`build_file_agent_subgraph`）**

- `fa_safe_tools_node`：只处理**尚未回复**的安全调用（用 `_unanswered_tool_calls`）。未找到的工具返回 `"工具 'x' 未找到。"`，保持现有行为。
- `fa_dangerous_tools_node`：取**尚未回复**的危险调用里的**第一个**，`interrupt()` 作为第一个动作，批准则执行、拒绝则返回现有的 CRITICAL ERROR 文本，**返回恰好 1 条 ToolMessage**。`interrupt` payload 的格式（`tool_name`/`tool_args`/`tool_call_id`）和 Y/N 判定保持不变：

```python
pending = [tc for tc in _unanswered_tool_calls(state["messages"]) if tc["name"] in dangerous_names]
tc = pending[0]                                  # 每次节点执行只处理 1 个
decision = interrupt({"tool_name": tc["name"], "tool_args": tc["args"], "tool_call_id": tc["id"]})
# ……批准 → 执行；拒绝 → 现有的 CRITICAL ERROR 文本（保持原样）
return {"messages": [ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"])]}
```

- 路由：`fa_llm`、`fa_safe_tools`、`fa_dangerous_tools` 三处的条件边都用同一个函数（替换现有的 `_route_after_llm`/`_route_after_tools`）：

```python
def _next_step(state):
    msgs = state["messages"]
    if isinstance(msgs[-1], AIMessage) and not msgs[-1].tool_calls:
        return "fa_summarize"
    pending = _unanswered_tool_calls(msgs)
    if any(tc["name"] not in dangerous_names for tc in pending):
        return "fa_safe_tools"
    if any(tc["name"] in dangerous_names for tc in pending):
        return "fa_dangerous_tools"
    return "fa_llm"
```

- 给 `build_file_agent_subgraph` 和 `build_desktop_agent_subgraph` 各加一个可选参数 `checkpointer=None`，原样传给 `compile(checkpointer=checkpointer)`。`app/deep_agent.py` 的调用不变、继续不传，让子图继承父图的 checkpointer；这个参数只为单元测试。

**DesktopAgent（`build_desktop_agent_subgraph`）**

- `da_tools_node` 每次执行只处理**尚未回复**的第 1 个调用：高危就先 `interrupt()` 再执行，非高危直接执行，返回 1 条 ToolMessage。
- 把 `subgraph.add_edge("da_tools", "da_llm")` 换成条件边：

```python
def _after_tools(state):
    return "da_tools" if _unanswered_tool_calls(state["messages"]) else "da_llm"
```

- 风险判定规则、`interrupt` 的提示文本字符串、截图转 HumanMessage 的逻辑都不变。

**同文件清理**

- 删除 `build_file_agent_subgraph` 里第一个 `return subgraph.compile()` 之后的整块不可达重复代码（重复的 `_route_after_llm`/`_route_after_tools`/子图组装/第二个 return）。
- `build_graph()` 里有 `FILE_AGENT_PROMPT` / `DESKTOP_AGENT_PROMPT` 的局部重复定义（模块级已有一份）。先 diff 两份文本：完全一致就删局部副本；不一致就贴出 diff 并停下来问我。
- 确认 `from app.graph import build_graph` 仍可正常导入并 build（旧入口要继续可用）。

### 四、单元测试（真实构建函数 + 假工具，无网络/MCP/桌面）

新建 `scripts/test_hitl_replay_safety.py`，直接调用 `graph.py` 里改好的两个构建函数（`checkpointer=MemorySaver()`），工具用与真实同名的 `StructuredTool` 假实现并计数（`write_file`/`edit_file`/`move_file`/`create_directory`/`read_text_file`；`click_grid_center`/`safe_input_text`/`press_hotkey`），LLM 用下面这个：

```python
class ScriptedLLM:
    def __init__(self, scripted): self._it = iter(scripted)
    def bind_tools(self, tools, **kw): return self
    async def ainvoke(self, messages, **kw):
        try: return next(self._it)
        except StopIteration: return AIMessage(content="done")
```

驱动方式参考 `scratch/test_double_hitl.py`，但**判断是否还有待批准的中断，请用 `state.tasks[*].interrupts`，不要只看 `state.next`**（直接对子图调用时，我遇到过 `next` 为空但 tasks 里仍挂着 interrupt 的情况）。每个用例都断言：实际执行次数，以及每个 tool_call_id 恰有一条 ToolMessage。

1. **FA-1**：一条消息里 `[write_file, create_directory, move_file]`，全部 Y → 每个工具恰执行 1 次；询问顺序与调用顺序一致。
2. **FA-2**：`[edit_file, move_file]` 全部 Y，其中 `edit_file` 假工具第二次被调用就抛异常 → `edit_file` 的 ToolMessage 必须是成功结果，不是异常文本。
3. **FA-3**：`[read_text_file(安全), edit_file(危险)]` → read 执行 1 次且不触发中断；edit 一次中断，Y 后执行 1 次。
4. **FA-4**：`[write_file, edit_file, move_file]`，回答 Y、N、Y → `edit_file` 执行 0 次，其 ToolMessage 含 `USER PERMISSION DENIED`；其余各 1 次。
5. **DA-1**：`[click_grid_center, safe_input_text, press_hotkey(keys="ctrl+s")]`，回答 Y → 三个工具各 1 次；回答 N → click、input 各 1 次，`press_hotkey` 0 次且 ToolMessage 含 `USER PERMISSION DENIED`。

贴出完整测试输出。

### 五、真实回归一次（仅 FileAgent）

重新跑 `scratch/test_double_hitl.py`（write_file + create_directory + move_file 那个场景），贴出终端轨迹，并统计 `✅ [FileAgent HITL] 执行成功` 出现的次数：应当等于被批准的危险调用数（3），而不是之前的 6。**不要测 DesktopAgent 的真实桌面操作。**

### 六、只读检查（不要改任何代码）

- **(a) 工具盘点**：`build_main_agent()` 之后，打印主 Agent 和每个声明式子 Agent 实际拥有的工具名（做法参考 `scripts/verify_deep_agent.py` Step 3 里已经写过的闭包提取；每个子 Agent graph 的 tools 节点里取 `tools_by_name`，属性名以你装的版本为准）。特别报告：`ls`/`read_file`/`write_file`/`edit_file`/`glob`/`grep`/`delete`/`execute`/`write_todos` 这些 deepagents 内置工具，是否出现在**主 Agent** 上、是否出现在各**声明式子 Agent** 上。
- **(b)** 直接调用 `rag` 工具查"火神战姬 Q技能"，贴原始返回，说明知识库里有没有 Q 技能的 chunk。
- **(c)** 通过主 Agent 再问一次"火神战姬的技能有哪些？"，把最终回答**原样整段**贴出来，不要省略号或括号说明。

### 验收

提交：`git diff app/graph.py` 全文；`hitl_replay_repro.py` 在你们环境的输出；单元测试输出；真实回归轨迹；(a)(b)(c) 的结果。报告里不要写"原逻辑未改动"这类概括，图里改了什么就写什么。

---

## Prompt B —— 图片发件箱 + 提示词/描述修订 + FileAgent 工具过滤

**（A 验收通过后，从这里往下整段复制给 Antigravity）**

这轮做四件事：图片发件箱（让 CodeAgent 的图表还能送到用户手里）、提示词/描述修订（恢复迁移时漏掉的规则）、FileAgent 工具过滤、验证。范围：`app/outbox.py`（新建）、`app/tools.py`（只改 `execute_python_code`）、`main_deepagents.py`、`app/deep_agent.py`。不碰 `plugins/chat.py`、`app/graph.py`、`mcp_servers.json`。不要测 DesktopAgent。

开始前先检查 `app/outbox.py` 是否已存在、`execute_python_code` 里是否已有 `outbox.push`：如果已有，对照下面的要求检查并修正，不要重复创建。

### B1 图片发件箱

背景：deepagents 的 `task` 工具只把子 Agent **最后一条消息的文本**返回给主 Agent，子 Agent 内部工具产生的图片到不了主 Agent 的消息链。所以改成旁路：产图的工具直接登记到"本轮发件箱"，接入层读发件箱。旧代码有意排除 `rag`/`av_graph_rag` 的图（只给 VLM 看，不发给用户），所以只登记 `execute_python_code` 的沙箱图片。

`app/outbox.py`：

```python
import contextvars
from typing import Optional

_outbox: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("image_outbox", default=None)

def begin() -> list:
    """处理每条用户消息前调用，返回本轮的图片收集列表。"""
    lst: list = []
    _outbox.set(lst)
    return lst

def push(item: dict) -> None:
    """没有调用过 begin() 的上下文（旧入口、单元测试）里是空操作。"""
    lst = _outbox.get()
    if lst is not None:
        lst.append(item)
```

- `execute_python_code`：对每张沙箱图片 `outbox.push({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})`；原来返回给子 Agent 的多模态内容保持不变。
- `main_deepagents.py`：每条用户消息处理前、调用 `astream` 之前，在同一个协程里调 `outbox.begin()`；同一条用户消息内的 HITL 恢复（resume）**不要**再次调用 `begin()`。一轮结束后打印发件箱张数和每张 base64 的长度（不要打印内容），并把图片解码保存到 `storage/outbox_debug/` 方便肉眼检查。
- 用 contextvar 而不是全局变量，是因为 QQ 多用户并发时各请求要隔离。已知限制：中断前登记的图片，如果恢复发生在另一条用户消息里就会丢失，暂不处理。

验证：
- **(a)** 先不经过 Agent，直接调 `execute_python_code.invoke({"code": ...})`，代码用 matplotlib 画 x=[1,2,3], y=[1,4,9] 的折线图并 savefig 到 `/workspace/outputs/a.png`；打印返回值类型和其中 `image_url` 的个数。Docker 沙箱起不来或镜像不存在时，贴出原始错误，跳过 (b)(c)，其余项目继续做，**不要伪造结果**。
- **(b)** 通过 `main_deepagents.py` 让主 Agent 委派 CodeAgent 画同一张图：报告发件箱张数；遍历整个 `result["messages"]`（含嵌套 list content）统计 `image_url` 出现次数（预期：消息链里 0 个、发件箱 ≥1 个）。
- **(c)** 发件箱是 0 但 (a) 能产图，说明 contextvar 没传进工具执行的上下文，如实报告现象，**不要擅自改成全局变量**。

### B2 `MAIN_AGENT_PROMPT` 修订（`app/deep_agent.py`）

三处文本修改，其余不动：

1. 【委派规范】末尾新增一条：

```
- task 的 description 是子专家能看到的全部信息——它看不到用户的原话、对话历史和长期记忆。完成任务所需的上下文（用户位置/地址、具体名称、约束条件、本地文件路径、上一步专家返回的关键数据）都必须写进 description。例如派 MapAgent 搜附近的菜市场，要写"用户当前位置在济南市历下区文化东路42号，请搜索附近的菜市场"，不能只写"帮用户查去哪买菜"。
```

2. 【图文分离】整段替换为：

```
【图文分离】
专家汇报只会以文字形式回到你这里，你看不到图片本身。如果汇报提到已生成图表或图片，系统会自动把图片发给用户：你只需要用文字交代结论，不要生成图片链接、Base64 或 Markdown 图片语法，也不要说"无法展示图片"。
```

3. 【回复要求】末尾新增一条：

```
最终回答使用纯文本，不要使用 Markdown 语法（不要 **加粗**、# 标题、代码围栏、行首的 * 列表符）；需要分点时写"1. 2. 3."或换行加"- "。专家汇报里的 Markdown 标记要去掉，但数据和数字必须原样保留。
```

### B3 恢复迁移时漏掉的路由规则（description）

旧 Supervisor prompt 里有几条"极端重要"的路由规则，迁移成 description 时漏了。更新如下（FileAgent、DesktopAgent 的 description 在 `build_main_agent` 里内联的 `CompiledSubAgent(...)` 中，BrowserAgent 的在 `SUBAGENT_DESCRIPTIONS` 中）：

- **FileAgent**：`"本地文件专家，是唯一能访问本地磁盘的专家：读取/写入/编辑项目内的文本类文件（.py/.json/.md/.txt 等），涉及危险操作（写入/编辑/移动）会暂停并请求人工授权。不要让它读取图片、音视频等二进制文件（它无法解析，会导致死循环）；网页上传文件的任务不要先派它读取，直接把路径交给 BrowserAgent。"`
- **BrowserAgent**：`"浏览器操作专家。处理网页点击、登录、动态数据抓取、文件上传与自动化发帖。它可以直接接收本地文件路径（如 workspace/1.png）完成网页上传，无需先派 FileAgent 读取；只要任务涉及网页端发布/登录/点击等交互，就直接派给它，不要提前结束并让用户手动操作。复杂规划中，需要先派这个拿到地址等信息，再派 MapAgent。"`
- DesktopAgent 的 description 保持不变。

### B4 FileAgent 的工具集去掉 `read_media_file`

`build_main_agent()` 里构造 FileAgent 子图之前：

```python
filesystem_tools = [t for t in filesystem_tools if t.name != "read_media_file"]
```

原因：该工具返回 base64 媒体内容，而 `mcp_service._wrap_tool` 会把非文本块用 `str(item)` 拼成文本，base64 会被整段塞进上下文；FileAgent 本来也不该读二进制文件。只影响新架构，旧的 `build_graph` 不变。

### B5 验证

1. B1 的 (a)(b)(c)。
2. 先 `database.insert_memory("用户家住在济南市历下区文化东路42号")`，再问"我家附近有菜市场吗"：贴出终端里 `[派发任务给 MapAgent]` 那一行——description 里应当带上具体地址（验证 B2-1）。
3. 让 CodeAgent 画折线图后，贴主 Agent 最终回答原文：应是纯文本，没有 Markdown 图片语法，也没有"无法展示图片"（验证 B2-2、B2-3）。
4. 重跑"济南今天天气怎么样"，确认仍路由给 MapAgent，且最终回答里没有 `**` 或行首 `*` 之类的 Markdown 标记。

所有输出原样贴，不要转述。贴出 `git diff app/deep_agent.py`。
