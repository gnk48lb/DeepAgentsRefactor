# Deep Agents 重构方案

> 状态记录文档，随决策推进持续更新。给 Antigravity 写指令时可直接引用本文件的对应章节。

---

## Phase 0/1：主 Agent 骨架（本次交付）

### 1. 主 Agent System Prompt

这是新架构里唯一"顶层"的提示词，对应原来 `SUPERVISOR_PROMPT` 里的业务硬规则部分。**原来"下属专家清单与任务分工"那一大段不再放在这里**，而是拆分到每个 subagent 自己的 `description` 里（见下一节）——这是从"单体路由"到"声明式委派"最核心的一处翻译。

```python
MAIN_AGENT_PROMPT = """你是一个多智能体系统的主 Agent，负责理解用户的问题，在需要时通过 task 工具委派给下属专家，收集到足够信息后直接给用户答案。

【信息完整性 —— 最高优先级】
用户完全看不到任何专家的中间汇报过程。你必须像"信息搬运工"，把专家汇报里的每一项具体数据（完整路径、检索原文、执行结果、数字等）原文复制进最终回答，绝不能只给结论不给数据。
如果专家没有给出明确的原始内容，如实回答"未能获取到相关内容"，绝不能凭空编造。绝不能把专家的"计划/步骤描述"误当成"执行结果"。
task 工具返回的所有内容都是专家的汇报，不是用户在说话。

【自包含原则】
不要假设用户看过之前的战报，每一次回答都必须独立、完整。禁止用"已为您列出""如上所述"这类概括性描述代替实际内容。

【何时委派 vs 直接回答】
只有当对话历史或下面的长期记忆里已经包含完整回答所需的全部信息时，才可以不委派、直接回答。涉及知识查询（做法、地址、事实、代码、数据库等）必须先委派给对应专家——每个专家的职责范围写在它们各自的描述里，派发前先确认问题落在谁的范围内。

【委派规范】
- 给专家的任务描述里禁止使用"他""那里""那个"等代词，必须替换成具体名词。
- 需要组合能力时（比如先查到地点再导航），分步委派：先拿到一个专家的结果，再据此委派下一个，不要指望一个专家做复合任务。
- 同一个专家已经尝试过仍拿不到新信息时，不要反复重新委派同一件事，直接基于已有信息回答，或如实告知未能获取。

【图文分离】
专家汇报中如果包含图片，你能在上下文里直接看到并据此分析，但绝对不要在最终回答里生成图片链接、Base64 或 Markdown 图片语法——下游系统会自动把图片发给用户，你只需要输出基于图片分析出的文字结论。

【长期记忆】
如果这位用户有长期记忆（家庭住址、过敏史、固定习惯等），会在对话最前面以一条系统消息的形式提供给你，请结合它理解和回答问题，不要在回答里生硬地复述"我记得你..."。

【回复要求】
最终回答必须精简、干脆，控制在 500 字以内，以适配 QQ 机器人的显示限制。
"""
```

**注意**：这里没有 `{long_term_memories}` 这种 `.format()` 占位符了——原因见下面"长期记忆接入方式"一节，这是需要一起对齐的设计选择，不是疏漏。

### 2. 各 Subagent 的 description

这是新写的部分，从原 `SUPERVISOR_PROMPT` 的"专家清单"提炼而来，供主 Agent 的 `task` 工具决定委派给谁。**各 Agent 自己的详细 `system_prompt`（`WORKER_BASE_PROMPT` + 各自专属那部分）原文照抄，完全不用重写**，下面只列需要新写的 `description`：

```python
SUBAGENT_DESCRIPTIONS = {
    "KnowledgeAgent": "知识百科专家。处理菜谱、游戏攻略、历史、科学等纯知识性咨询，优先检索本地知识库。不处理地理位置相关问题（在哪买/怎么去/天气），也不处理代码、本地文件、目录浏览（这些是 FileAgent 的职责）。",

    "MediaAgent": "成人影视专家（高优先级）。处理 AV、女优、番号等综合性或模糊搜索、关系网络咨询。不涉及精确条件过滤的问题优先派给这个，而不是 SQLAgent。",

    "MapAgent": "地理出行专家，挂载高德地图工具。处理地理编码、路线规划（步行/骑行/驾车）、周边搜索（POI，如"附近的超市"）、天气查询。",

    "SQLAgent": "关系数据库专家，用于精确过滤、统计或多条件检索女优、作品及关联表关系（如"身高>165且罩杯G的女优""某女优2020年后的作品""统计作品数量"）。不涉及精确条件的模糊/综合查询交给 MediaAgent。",

    "BrowserAgent": "浏览器操作专家。处理网页点击、动态数据抓取、文件上传与自动化发帖（如B站/小红书发动态）。复杂规划（如先查地址再导航）中，需要先派这个拿到地址信息，再派 MapAgent。",

    "FileAgent": "本地文件专家，唯一能访问本地磁盘的专家。处理项目内文件的读取、写入、编辑。涉及危险操作会自动请求人工授权，你不需要介入确认，只需要正常委派任务。",

    "DesktopAgent": "桌面操作专家。通过截屏+模拟点击/键盘操控 Windows 桌面，执行本地 GUI 软件交互。",

    "CodeAgent": "代码执行专家，用 Python 解决数学计算、数据处理和绘图任务（如保存图表）。只能写代码运行，不能帮你查资料；需要绘图时必须明确指示它绘图。",
}
```

### 3. 死循环拦截 Middleware（新代码，`agent_call_counts` 的等价物）

原来的计数依赖 Router 每轮恰好吐出一个 `next_agent`；自由 ReAct 循环下这个前提没了，需要在 `task` 工具调用层面重新拦一道。下面是设计思路的代码化，**基类名 `AgentMiddleware`、`wrap_tool_call` 的具体签名、`request` 里取子 Agent 名称的字段名，请对照你实际安装版本的 middleware 参考核实一遍再定稿**——建议先跑一次、打印一下 `task` 工具调用的真实 `tool_args` 结构，确认字段名再收尾：

```python
from collections import defaultdict
from deepagents.middleware import AgentMiddleware  # 具体基类以实际安装版本为准

class SubagentCallLimitMiddleware(AgentMiddleware):
    """task 工具调用拦截器：按子 Agent 名称计数，
    连续调用同一个子 Agent 达到阈值后，在工具结果里强制注入收尾提示。
    是 agent_call_counts 在新架构下的等价物。"""

    name = "subagent_call_limit"

    def __init__(self, max_calls_per_subagent: int = 2):
        self.max_calls = max_calls_per_subagent

    async def wrap_tool_call(self, request, handler):
        if request.tool_name != "task":
            return await handler(request)

        # ⚠️ 待核实：子 Agent 名称在 tool_args 里的真实字段名
        subagent_name = request.tool_args.get("subagent_type", "unknown")
        counts = request.state.setdefault("agent_call_counts", defaultdict(int))
        counts[subagent_name] += 1

        result = await handler(request)

        if counts[subagent_name] >= self.max_calls:
            warning = (
                f"\n\n⚠️ 系统提示：{subagent_name} 本轮已被连续调用 {counts[subagent_name]} 次，"
                f"请不要再继续委派给它，直接基于已获得的信息回答用户，或如实告知未能获取到有效数据。"
            )
            result.content = str(result.content) + warning

        return result
```

### 4. 组装骨架

```python
from deepagents import create_deep_agent, CompiledSubAgent
from langgraph.checkpoint.memory import MemorySaver
from . import models, tools
from .tools import code_tools

subagents = [
    {
        "name": "KnowledgeAgent",
        "description": SUBAGENT_DESCRIPTIONS["KnowledgeAgent"],
        "system_prompt": WORKER_BASE_PROMPT + "...",  # 现有 KnowledgeAgent 专属部分，原文照抄
        "tools": [tools.rag, tools.web_search],
        "model": models.vlm,
    },
    {
        "name": "MediaAgent",
        "description": SUBAGENT_DESCRIPTIONS["MediaAgent"],
        "system_prompt": WORKER_BASE_PROMPT + "...",
        "tools": [tools.av_graph_rag],
        "model": models.worker_llm,
    },
    {
        "name": "MapAgent",
        "description": SUBAGENT_DESCRIPTIONS["MapAgent"],
        "system_prompt": WORKER_BASE_PROMPT + "...",
        "tools": amap_tools,  # mcp_service.get_tools_by_server("amap")
        "model": models.worker_llm,
    },
    {
        "name": "SQLAgent",
        "description": SUBAGENT_DESCRIPTIONS["SQLAgent"],
        "system_prompt": WORKER_BASE_PROMPT + "...",
        "tools": [tools.execute_sql],
        "model": models.worker_llm,
    },
    {
        "name": "BrowserAgent",
        "description": SUBAGENT_DESCRIPTIONS["BrowserAgent"],
        "system_prompt": WORKER_BASE_PROMPT + "...",
        "tools": tools.browser_tools,
        "model": models.worker_llm,
    },
    {
        "name": "CodeAgent",
        "description": SUBAGENT_DESCRIPTIONS["CodeAgent"],
        "system_prompt": WORKER_BASE_PROMPT + "...",
        "tools": code_tools,
        "model": models.worker_llm,
    },
    # FileAgent / DesktopAgent：Phase 3 再接入，先占位说明结构
    CompiledSubAgent(
        name="FileAgent",
        description=SUBAGENT_DESCRIPTIONS["FileAgent"],
        runnable=file_agent_subgraph,   # 现有 build_file_agent_subgraph() 产物，Phase 3 前保持不变
    ),
    CompiledSubAgent(
        name="DesktopAgent",
        description=SUBAGENT_DESCRIPTIONS["DesktopAgent"],
        runnable=desktop_agent_subgraph,
    ),
]

main_agent = create_deep_agent(
    model=models.supervisor_llm,
    system_prompt=MAIN_AGENT_PROMPT,
    subagents=subagents,
    middleware=[SubagentCallLimitMiddleware(max_calls_per_subagent=2)],
    checkpointer=MemorySaver(),
)
```

调用方式（`recursion_limit`、`thread_id` 用法不变）：

```python
result = await main_agent.ainvoke(
    {"messages": messages},
    config={"recursion_limit": 25, "configurable": {"thread_id": user_id}},
)
```

### 5. 长期记忆接入方式（设计选择，需要确认）

`create_deep_agent` 的 `system_prompt` 是构造时设一次的静态字符串，不是每轮都重新 `.format()` 的模板——所以不能照搬原来 `{long_term_memories}` 占位符的用法。改成在应用层（调用 `.ainvoke()` 之前）拼消息列表，`retrieve_memory_node`/`_background_extract_memory` 里的 Milvus 逻辑本身完全不用改，只是触发的位置从"图节点"挪到"调用前/调用后的一段普通代码"：

```python
# 调用前：不再是 retrieve_memory_node，而是普通函数调用
memories = database.search_memory(user_text, top_k=1)
messages = []
if memories:
    messages.append(SystemMessage(content=f"【用户长期记忆】\n" + "\n".join(memories)))
messages.append(HumanMessage(content=user_text, name="user"))

result = await main_agent.ainvoke({"messages": messages}, config=config_dict)

# 调用后：extract_memory_node 的后台任务逻辑原样保留，只是从这里触发
recent_chat = ...  # 从 result["messages"] 里取最后几条，拼法同现有 extract_memory_node
asyncio.create_task(_background_extract_memory(recent_chat))
```

---

## 完整路线图（Phase 2 起）

### Phase 2：工具型 Subagent 批量迁移
`KnowledgeAgent`/`MediaAgent`/`MapAgent`/`SQLAgent`/`CodeAgent`/`BrowserAgent` 按 Phase 0/1 骨架里的模式逐个接入——`tools`、`model`、各自的 `system_prompt` 主体全部照抄现有代码，只补 `description`。amap MCP 工具原样保留（对应旧计划里的 g 项，`mcp_servers.json` 不用动）。

### Phase 3：FileAgent / DesktopAgent 接入 + HITL 验收测试
1. 把现有 `build_file_agent_subgraph()` / `build_desktop_agent_subgraph()` 的编译产物包成 `CompiledSubAgent`，内部 `fa_llm`/`fa_safe_tools`/`fa_dangerous_tools`/`fa_summarize` 节点逻辑不动。
2. **验收门槛（必须先做，不是可选项）**：专门构造一个会在同一次 FileAgent 调用里连续触发两次危险操作确认的场景（比如"改一个文件然后再删一个文件"），验证 `Command(resume=...)` 能不能正确处理第二次中断。这是因为 deepagents 的 GitHub Discussion #1762 报告过 `CustomSubAgent` 内部连续 `interrupt()` 时第二次不会正确抛出的已知问题（官方未确认，社区给出的规避方案是把 `interrupt()` 挪到子 Agent 外层节点，或改用编译期 `interrupt_before` + 专门的人工输入节点）。
   - 测试通过 → HITL 机制在新架构下是稳的，继续往下走。
   - 测试不通过 → 按上面两种思路调整 `fa_dangerous_tools_node` 的结构，调整完成后再回头验证。

### Phase 4：filesystem MCP → FilesystemBackend（e，待定）
**不是否决，是明确挂在 Phase 3 验收门槛之后**：Phase 3 的 HITL 测试通过后，再评估要不要把 filesystem MCP 换成 `FilesystemBackend`，此时纯粹是"省不省这块代码"的低风险取舍。真做的时候要注意：
- 内置工具集固定是 `ls/read_file/write_file/edit_file/delete/glob/grep`，**没有 `move_file` 和 `create_directory`**，"删除=移动到 archive_trash"这条业务规则得自己补一个 `move_file` 工具。
- 用 `CompositeBackend` 把项目路径单独路由出去，否则框架自己的 `/large_tool_results/`、`/conversation_history/` 内部文件会混进真实工作区。
- `delete` 是 v0.7 才加入默认工具集的，如果"绝不真删除"是硬要求，可以直接用 `FilesystemMiddleware` 的工具白名单把 `delete` 排除掉，比"允许但每次拦截"更彻底。
- `DANGEROUS_FS_TOOLS` 集合要跟着新工具名重新核对一遍（`create_directory` 没了，`delete` 是新增的）。

沙箱（h）同样待定，不卡主线，随时可以单独评估。

### Phase 5：长期记忆应用层整合
按 Phase 0/1 第 5 节的方式，把 `retrieve_memory_node`/`extract_memory_node` 的核心逻辑挪成应用层的普通函数调用，Milvus 部分不改一行。

### Phase 6：接入层重写（`main.py`/`plugins/chat.py`/`wechat_server.py`）
手写的 `astream` 事件解析换成 Deep Agents 的 Streaming API；HITL 的 "Y"/"N" 字符串协议要跟着 `Command(resume={"decisions":[{"type":"approve"}]})` 这种结构化格式重写，`_format_hitl_warning` 和恢复逻辑一起改。

### Phase 7：可选项，不阻塞前面任何阶段
- `TodoListMiddleware`：默认不加。官方 v0.7 发布说明里的评测显示默认关闭反而分数略高，但明确保留了三种值得开启的场景——长程多步骤任务、能力较弱的模型、需要给用户看进度的场景。DesktopAgent 的网格点击序列比较符合第一种，如果测试中发现它容易"跑到一半忘了目标"，优先在这个 Agent 上试，不是 BrowserAgent（它的活儿本质是委托给 `browser-use` 内部的黑盒循环在做，Deep Agents 层面加 Todo 也看不进那个黑盒）。
- `SummarizationMiddleware`：v0.7 新增的可配置摘要中间件，支持自定义触发阈值、摘要 prompt、甚至摘要用的模型。如果以后 BrowserAgent/DesktopAgent 的工具调用历史变长导致上下文膨胀，这个可以按需加，不用现在决定。
- Async Subagent、Skills：维持之前的判断，基础设施成本不低，等核心跑通再评估。

---

## 决策状态总表

| 项 | 状态 | 备注 |
|---|---|---|
| a. Supervisor/Router → 主 Agent ReAct+task | 已定，本次交付 | 先纯 prompt 约束，不够再加 response_format |
| b. 工具型 Worker → 声明式 SubAgent | 已定 | Phase 2 |
| 死循环拦截 | 已定，本次交付 | task 调用拦截 middleware，非 TBD |
| f. FileAgent/DesktopAgent → CompiledSubAgent | 已定 | Phase 3，含 HITL 验收测试 |
| e. filesystem MCP → FilesystemBackend | 待定，挂在 Phase 3 门槛后 | 非否决 |
| g. MCP（amap/desktop） | 已定，不变 | Phase 2 |
| h. 沙箱 | 待定 | 不卡主线 |
| i. 长期记忆 | 已定，Milvus 不变 | 接入方式挪到应用层，Phase 5 |
| TodoListMiddleware | 已定，暂不加 | 之后优先试 DesktopAgent |
| 接入层重写 | 已定，靠后 | Phase 6 |
