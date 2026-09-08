# Deep Agents 迁移 · Phase 0/1 交付说明 + 完整路线图

> **Part 1** 是可以直接整段复制给 Antigravity 的任务指令。**Part 2** 是留着你自己追踪剩余计划的参考，不用给 Antigravity。这份文件随决策推进持续更新。

---

## Part 1 —— 复制给 Antigravity 的任务

**（从这里到 "## Part 2" 之前，整段复制过去）**

你正在对一个基于 LangChain/LangGraph 的多智能体 QQ 机器人项目做框架迁移，目标是把现有的 Supervisor+Router 手动路由架构，改造成 deepagents 的"主 Agent ReAct 循环 + task 工具"架构。这是分阶段迁移的第一阶段，只做骨架搭建和工具型专家迁移。**不要动 FileAgent/DesktopAgent 的内部逻辑、不要改 filesystem MCP、不要碰生产入口文件**——这些是后续阶段的事，本次范围之外的一律不要顺手改。完成后按文末的验收标准给出结果。

### 背景（帮助理解设计意图，不需要照抄进代码）

现有架构：`app/graph.py` 里的 Supervisor 节点每轮被迫通过工具调用产出结构化的 `next_agent` 字段，手动路由到某个 Worker 节点，Worker 跑完以 `ToolMessage` 汇报回 Supervisor，循环直到 `FINISH`。新架构：用 deepagents 的 `create_deep_agent` 构建一个自由的 ReAct 循环，工具型专家改造成声明式 subagent（通过内置的 `task` 工具委派），已有的 `FileAgent`/`DesktopAgent` 编译子图会在下一阶段包成 `CompiledSubAgent` 原样接入。

### 第 0 步：环境准备（先做，确认无误再往下走）

1. 新建独立分支，执行 `uv add "deepagents>=0.7,<0.8"`，让它把 `langchain`/`langgraph`/`langchain-core` 等依赖一起升级到匹配版本，不要手动锁旧版本阻止升级。
2. 装完立刻做一次最小烟雾测试：几行临时脚本，`from deepagents import create_deep_agent`，用 `create_deep_agent(model=models.worker_llm, system_prompt="test")`（不传 subagents）构造最简单的 agent，`await agent.ainvoke({"messages": [HumanMessage(content="你好")]})` 跑一次。目的是尽早确认：现有的 `ChatGoogleGenerativeAI` 实例能不能直接传给 `model=`；`langchain-google-genai` 版本要不要跟着升级。这一步报错就先解决掉，不要绕过去写后面的代码。
3. 记录这次烟雾测试的原始请求/响应，作为依赖升级后的基线。

### 第 1 步：新建模块，不改 `app/graph.py`

在 `app/` 下新建 `deep_agent.py`。现有的 `app/graph.py`、`main.py`、`plugins/chat.py` 全部保持原样——这是新旧并存验证阶段，不是替换，两套架构要能同时跑。

### 第 2 步：主 Agent 的 system_prompt

写进新模块。这是从 `app/graph.py` 里 `SUPERVISOR_PROMPT` 的业务硬规则部分翻译过来的——原来"下属专家清单与任务分工"那一整段不再放这里，拆到每个 subagent 自己的 `description` 里（见第 3 步），这是"路由 schema"到"声明式委派"最核心的一处转换：

```python
MAIN_AGENT_PROMPT = """你是一个多智能体系统的主 Agent，负责理解用户的问题，在需要时通过 task 工具委派给下属专家，收集到足够信息后直接给用户答案。

【信息完整性 —— 最高优先级】
用户完全看不到任何专家的中间汇报过程。你必须像"信息搬运工"，把专家汇报里的每一项具体数据（完整路径、检索原文、执行结果、数字等）原文复制进最终回答，绝不能只给结论不给数据。
如果专家没有给出明确的原始内容，如实回答"未能获取到相关内容"，绝不能凭空编造。绝不能把专家的"计划/步骤描述"误当成"执行结果"。
task 工具返回的所有内容都是专家的汇报，不是用户在说话。

【自包含原则】
不要假设用户看过之前的战报，每一次回答都必须独立、完整。禁止用"已为您列出""如上所述"这类概括性描述代替实际内容。

【何时委派 vs 直接回答】
只有当对话历史或系统消息里提供的长期记忆已经包含完整回答所需的全部信息时，才可以不委派、直接回答。涉及知识查询（做法、地址、事实、代码、数据库等）必须先委派给对应专家——每个专家的职责范围写在它们各自的描述里，派发前先确认问题落在谁的范围内。

【委派规范】
- 给专家的任务描述里禁止使用"他""那里""那个"等代词，必须替换成具体名词。
- 需要组合能力时（比如先查到地点再导航），分步委派：先拿到一个专家的结果，再据此委派下一个，不要指望一个专家做复合任务。
- 同一个专家已经尝试过仍拿不到新信息时，不要反复重新委派同一件事，直接基于已有信息回答，或如实告知未能获取。

【过程不可见】
在决定委派或继续收集信息的过程中，不要输出你的思考过程、计划或"我准备去问一下 XX"这类文字——这些不会展示给用户。只有当你不再调用 task、准备给出最终结论时，你输出的内容才是用户会看到的内容。

【图文分离】
专家汇报中如果包含图片，你能在上下文里直接看到并据此分析，但绝对不要在最终回答里生成图片链接、Base64 或 Markdown 图片语法——下游系统会自动把图片发给用户，你只需要输出基于图片分析出的文字结论。

【长期记忆】
如果这位用户有长期记忆（家庭住址、过敏史、固定习惯等），会在对话最前面以一条系统消息的形式提供给你，请结合它理解和回答问题，不要在回答里生硬地复述"我记得你..."。

【回复要求】
最终回答必须精简、干脆，控制在 500 字以内，以适配 QQ 机器人的显示限制。
"""
```

### 第 3 步：六个工具型专家的声明式定义

**只需要新写 `description`，各专家自己的 `system_prompt`（`WORKER_BASE_PROMPT` + 各自专属那部分）、`tools`、`model`，直接从 `app/graph.py` 里对应的 `make_worker_node(...)` 调用原样搬过来，一个字都不改**：

```python
SUBAGENT_DESCRIPTIONS = {
    "KnowledgeAgent": "知识百科专家。处理菜谱、游戏攻略、历史、科学等纯知识性咨询，优先检索本地知识库。不处理地理位置相关问题（在哪买/怎么去/天气），也不处理代码、本地文件、目录浏览。",
    "MediaAgent": "成人影视专家（高优先级）。处理 AV、女优、番号等综合性或模糊搜索、关系网络咨询。不涉及精确条件过滤的问题优先派给这个，而不是 SQLAgent。",
    "MapAgent": "地理出行专家，挂载高德地图工具。处理地理编码、路线规划（步行/骑行/驾车）、周边搜索（POI）、天气查询。",
    "SQLAgent": "关系数据库专家，用于精确过滤、统计或多条件检索女优、作品及关联表关系。不涉及精确条件的模糊/综合查询交给 MediaAgent。",
    "BrowserAgent": "浏览器操作专家。处理网页点击、动态数据抓取、文件上传与自动化发帖。复杂规划中，需要先派这个拿到地址等信息，再派 MapAgent。",
    "CodeAgent": "代码执行专家，用 Python 解决数学计算、数据处理和绘图任务。只能写代码运行，不能帮你查资料；需要绘图时必须明确指示它绘图。",
}
```

对照 `app/graph.py` 逐个抄成声明式 dict，例如 `KnowledgeAgent`（对照原文件里 `knowledge_agent = make_worker_node("KnowledgeAgent", [tools.rag, tools.web_search], WORKER_BASE_PROMPT + (...), llm=models.vlm)`）：

```python
{
    "name": "KnowledgeAgent",
    "description": SUBAGENT_DESCRIPTIONS["KnowledgeAgent"],
    "system_prompt": WORKER_BASE_PROMPT + "你是 KnowledgeAgent...",  # 原样抄 app/graph.py 里那一段
    "tools": [tools.rag, tools.web_search],
    "model": models.vlm,
}
```

其余五个按同样模式做，`tools`/`model` 分别对照 `app/graph.py` 里 `media_agent`/`map_agent`/`sql_agent`/`browser_agent`/`code_agent` 各自的构造参数（除 KnowledgeAgent 用 `models.vlm` 外，其余都没有显式传 `llm=`，即默认 `models.worker_llm`）。

**本阶段不接入 `FileAgent` 和 `DesktopAgent`**，在 subagents 列表末尾留一条注释占位，写清楚它们会在下一阶段以 `CompiledSubAgent` 包装 `app/graph.py` 里已有的 `build_file_agent_subgraph()` / `build_desktop_agent_subgraph()` 产物接入，本阶段不动它们的内部逻辑。

### 第 4 步：死循环/超量委派防护

**用 langchain 自带的 `ToolCallLimitMiddleware`，不要自己写计数中间件**——这是官方维护的组件，参数名和签名我核实过是真实存在的：

```python
from langchain.agents.middleware import ToolCallLimitMiddleware

task_limit = ToolCallLimitMiddleware(
    tool_name="task",
    run_limit=12,              # 单次用户请求里 task 总共最多调用 12 次，先按这个跑，不够再调
    exit_behavior="continue",  # 不要用 "end"，理由见下
)
```

**`exit_behavior` 明确用 `"continue"`，不要用 `"end"`**——查到两个具体的已知问题，不是保守起见：
1. 官方文档写明 `"end"` 遇到并行工具调用（同一个 AIMessage 里一次性发出多个 `tool_calls`）时会直接抛 `NotImplementedError`。你们恰恰是为了让主 Agent 能一次委派多个专家才换成自由 ReAct 循环的，这个风险不是边缘情况，是设计目标本身会触发的场景。
2. LangChain 仓库有个还开着的 issue（#34159）：`run_limit` + `exit_behavior="end"` 配合 checkpointer 使用时，会产出"assistant message 的 tool_calls 没有对应 tool 回复"这种非法状态，后续请求报 400。你们的 `checkpointer=MemorySaver()` 正好命中这个组合。

`"continue"` 的行为是"超限的工具调用被拦下、其他执行正常继续"，配合 `MAIN_AGENT_PROMPT` 里"同一个专家拿不到新信息就别再问"这条软约束，模型会自己把被拦截的委派请求转化成一个真实的收尾回答，而不是像 `"end"` 那样吐出一句框架写死的 "Tool call limits exceeded" 糊弄用户——这个体验比原来 Supervisor 强制 FINISH 时你们自己写的友好提示要差一截，没必要为了省事换成更差的。

这个中间件按"`task` 总共被调了几次"计数，不区分委派给了哪个专家——跟原来 `agent_call_counts`"同一个专家连续调 2 次就强制结束"不是完全同一个语义，但作为防止跑飞的硬 backstop 已经足够，配合上面的软约束，双保险。**先只用这一个，不要现在就去实现"同一子 Agent 连续调用 N 次"这种更精细的自定义 middleware。** 如果实测发现总量限制不够精细（反复纠缠同一个专家但总数没超限），再考虑加自定义 `wrap_tool_call` 中间件——到时候记得用官方 `state_schema` 声明计数字段、通过返回 `Command(update=...)` 更新状态，不要直接对 `request.state` 做 in-place 赋值再指望它跨调用持久化，这类中间件的状态持久化机制不是简单的可变字典，需要对照你本地安装版本的源码确认具体写法。

### 第 5 步：长期记忆挪到应用层，不用 deepagents 的 backend/Store

`create_deep_agent` 的 `system_prompt` 是构造时定死的静态字符串，不是每轮重新填充的模板，不能照搬原来 `{long_term_memories}.format()` 的用法。把 Milvus 记忆检索/写入的触发点从"图节点"改成"调用前后的普通函数"，`database.search_memory` / `database.insert_memory` 的实现本身一行都不用动：

```python
async def build_initial_messages(user_text: str, user_id: str) -> list:
    memories = database.search_memory(user_text, top_k=1)
    messages = []
    if memories:
        messages.append(SystemMessage(content="【用户长期记忆】\n" + "\n".join(memories)))
    messages.append(HumanMessage(content=user_text, name="user"))
    return messages

# 调用方式（本阶段只在下面第 7 步的验证脚本里用，不接入 plugins/chat.py）：
messages = await build_initial_messages(user_text, user_id)
result = await main_agent.ainvoke(
    {"messages": messages},
    config={"recursion_limit": 25, "configurable": {"thread_id": user_id}},
)

# 调用后的后台记忆提取，直接复用 app/graph.py 里 _background_extract_memory 的逻辑：
# 从 result["messages"] 里取最后几条拼 recent_chat，asyncio.create_task 触发即可，不用改。
```

### 第 6 步：组装

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents.middleware import ToolCallLimitMiddleware
from . import models, tools, mcp_service

# 注意：MapAgent 需要的 amap_tools 依赖 mcp_service 已完成初始化，
# 构造 subagents 列表前确认 await mcp_service.initialize_mcp() 已经跑过，
# 否则 get_tools_by_server 会静默返回空列表，MapAgent 会变成没有工具可用而不报错。
amap_tools = mcp_service.get_tools_by_server("amap")

subagents = [
    {"name": "KnowledgeAgent", "description": SUBAGENT_DESCRIPTIONS["KnowledgeAgent"], ...},
    {"name": "MediaAgent", "description": SUBAGENT_DESCRIPTIONS["MediaAgent"], ...},
    {"name": "MapAgent", "description": SUBAGENT_DESCRIPTIONS["MapAgent"], ...},
    {"name": "SQLAgent", "description": SUBAGENT_DESCRIPTIONS["SQLAgent"], ...},
    {"name": "BrowserAgent", "description": SUBAGENT_DESCRIPTIONS["BrowserAgent"], ...},
    {"name": "CodeAgent", "description": SUBAGENT_DESCRIPTIONS["CodeAgent"], ...},
    # FileAgent / DesktopAgent：下一阶段用 CompiledSubAgent 包
    # build_file_agent_subgraph() / build_desktop_agent_subgraph() 的产物接入，本阶段不做。
]

main_agent = create_deep_agent(
    model=models.supervisor_llm,
    system_prompt=MAIN_AGENT_PROMPT,
    subagents=subagents,
    middleware=[ToolCallLimitMiddleware(tool_name="task", run_limit=12, exit_behavior="continue")],
    checkpointer=MemorySaver(),
)
```

### 第 7 步：验证脚本（新建，不接入生产入口）

新建 `scripts/validate_deep_agent.py`，构造上面的 `main_agent`，跑 5-8 个覆盖不同专家的真实问题（每个专家职责范围里各挑一两个典型问题，包含至少一个需要多步委派的组合类问题），记录：路由对不对、最终回答是不是完整复述了专家数据而不是空泛结论、耗时、token 用量。这份记录留着跟旧架构对比，不用很正式，能看出问题就行。

### 明确不要做的事

- 不动 `FileAgent` / `DesktopAgent` 现有的 `build_file_agent_subgraph()` / `build_desktop_agent_subgraph()` 内部实现，也不把它们接进 `subagents` 列表。
- 不改 `mcp_servers.json` 或 `app/mcp_service.py`，filesystem MCP 保持现状。
- 不碰 `main.py`、`plugins/chat.py` 的生产调用路径——这次只搭新骨架、单独验证，不做流量切换。
- 不加 `TodoListMiddleware`。
- 不实现精细的"同一子 Agent 连续调用 N 次"自定义中间件，除非第 4 步的 `ToolCallLimitMiddleware` 在验证脚本里明显不够用。

### 验收标准

1. 第 0 步烟雾测试和第 7 步验证脚本都能跑通，不报错。
2. 至少验证一次：故意连续问同一个专家能回答、但反复问不出新信息的问题，确认到达 `run_limit` 后 agent 不报错、不卡死，并给出基于已有信息的真实收尾回答（不是框架写死的提示字符串）。另外专门测一次主 Agent 一次性并行委派给两个以上专家的场景，确认不会触发 `NotImplementedError`。
3. 贴出验证脚本跑出来的几个真实回答，需要能看出"专家数据有没有被完整复述进最终回答"（对应 `MAIN_AGENT_PROMPT` 的信息完整性要求）。
4. 完成后总结：环境搭建/依赖升级踩了什么坑，`model=` 直接传现有实例是否真的无需改造，`ToolCallLimitMiddleware` 好不好用。

---

## Part 2 —— 剩余计划（自己看，不用给 Antigravity）

### 下一步 A：FileAgent / DesktopAgent 接入 + HITL 验收测试
1. 把现有 `build_file_agent_subgraph()` / `build_desktop_agent_subgraph()` 的编译产物包成 `CompiledSubAgent`，内部 `fa_llm`/`fa_safe_tools`/`fa_dangerous_tools`/`fa_summarize` 节点逻辑不动。
2. **验收门槛（必须先做，不是可选项）**：构造一个会在同一次 FileAgent 调用里连续触发两次危险操作确认的场景（比如"改一个文件然后再删一个文件"），验证 `Command(resume=...)` 能不能正确处理第二次中断。deepagents 的 GitHub Discussion #1762 报告过子图内部连续 `interrupt()` 时第二次不会正确抛出的问题——维护者没有确认复现，一位社区网友给出的解释是 `Command(resume=...)` 是从检查点重放而非从当前中断点续接，规避方案是把 `interrupt()` 挪到子 Agent 外层节点，或者改用编译期 `interrupt_before` + 专门的人工输入节点。
   - 测试通过 → HITL 机制在新架构下是稳的，继续往下走。
   - 测试不通过 → 按上面两种思路调整 `fa_dangerous_tools_node` 结构，调整完成后再验证。

### 下一步 B：filesystem MCP → FilesystemBackend（待定，非否决）
挂在"下一步 A"验收通过之后，届时纯粹是"省不省这块代码"的低风险取舍。真做的时候注意：
- 内置工具集固定 `ls/read_file/write_file/edit_file/delete/glob/grep`，没有 `move_file`/`create_directory`，"删除=移动到 archive_trash"这条规则得自己补一个 `move_file` 工具。
- 必须用 `CompositeBackend` 把项目路径单独路由出去，否则框架自己的 `/large_tool_results/`、`/conversation_history/` 会混进真实工作区。
- `delete` 是 v0.7 才加进默认工具集的；如果"绝不真删除"是硬要求，用 `FilesystemMiddleware` 的工具白名单直接把 `delete` 排除掉，比"允许但每次拦截"更彻底——`DANGEROUS_FS_TOOLS` 也要跟着新工具名重新核对一遍（`create_directory` 没了，`delete` 是新增的）。
- v0.7 同时把 `read_file` 的分页做得更完善：会报告总行数、剩余行数和下一个 offset，比现在 `_truncate_if_needed` 直接截断更方便模型自己判断要不要继续翻页，真做这块迁移时可以顺带省掉这段自定义截断逻辑。

沙箱同样待定，不卡主线，随时可以单独评估。

### 下一步 C：长期记忆彻底切到应用层
Phase 0/1 验证脚本里的应用层记忆模式（第 5 步）跑稳之后，等接入生产环境（下一步 E）时，把 `app/graph.py` 里 `retrieve_memory_node`/`extract_memory_node` 这两个图节点删掉即可，Milvus 部分不改一行。

### 下一步 D：可选中间件，不阻塞任何前面阶段
- `TodoListMiddleware`：默认不加。v0.7 发布说明里明确写了三种值得开启的场景——长程多步骤任务、能力较弱的模型、需要给用户看进度的场景。`DesktopAgent` 的网格点击序列比较符合第一种，如果测试中发现它容易"跑到一半忘了目标"，优先在这个 Agent 上试，而不是 `BrowserAgent`（它的活儿本质是委托给 `browser-use` 内部的黑盒循环在做，Deep Agents 层面加 Todo 也看不进那个黑盒）。
- `SummarizationMiddleware`：v0.7 支持自定义触发阈值和摘要 prompt。如果以后 `BrowserAgent`/`DesktopAgent` 的工具调用历史变长导致上下文膨胀，按需加，不用现在决定。
- Async Subagent、Skills：维持之前的判断，基础设施成本不低，等核心跑通再评估。

### 下一步 E：接入层重写（`main.py` / `plugins/chat.py`）
手写的 `astream` 事件解析换成 Deep Agents 的 Streaming API；HITL 的 "Y"/"N" 字符串协议要跟着结构化的 resume 格式重写，`_format_hitl_warning` 和恢复逻辑一起改。这是最后一步——前面所有验证都通过、确认新架构稳定之后再做真正的流量切换。

---

## 决策状态总表

| 项 | 状态 | 备注 |
|---|---|---|
| Supervisor/Router → 主 Agent ReAct+task | 已定，Part 1 交付 | 先纯 prompt 约束，不够再加 response_format |
| 工具型 Worker → 声明式 SubAgent | 已定，Part 1 交付 | 六个一起做，模式完全一致，风险相同 |
| 死循环/超量委派拦截 | 已定，Part 1 交付 | 优先用内置 ToolCallLimitMiddleware；自定义精细计数作为备选，非必需 |
| FileAgent/DesktopAgent → CompiledSubAgent | 已定，下一步 A | 含 HITL 双重中断验收测试 |
| filesystem MCP → FilesystemBackend | 待定，挂在下一步 A 验收后 | 非否决 |
| MCP（amap/desktop） | 已定，不变 | — |
| 沙箱 | 待定 | 不卡主线 |
| 长期记忆 | 已定，Milvus 不变 | 接入方式挪到应用层，下一步 C |
| TodoListMiddleware | 已定，暂不加 | 之后优先试 DesktopAgent |
| 接入层重写 | 已定，靠后 | 下一步 E，最后做流量切换 |
