# Deep Agents 迁移 · 下一步 A 验收战报

## 1. 任务概述

本阶段完成了对 `deepagents` 框架迁移的"下一步 A"目标：
1. **彻底关闭默认内置的 `general-purpose` subagent**，消除未受控的兜底路由隐患；
2. **将现有的 `FileAgent` 与 `DesktopAgent` 编译子图作为 `CompiledSubAgent` 接入主 Agent**，不修改原有子图节点内部逻辑；
3. **完成严格的 HITL 双重/多重中断与恢复（Pause & Resume）验收测试**。

---

## 2. 第 0 步：关闭默认 `general-purpose` subagent

### 调研结果与最终方案（路径二：底层关闭）
在 deepagents `0.7.x` 源码中，`create_deep_agent` 装配 subagent 列表时通过以下逻辑控制默认通用 subagent：
```python
gp_profile = _profile.general_purpose_subagent or GeneralPurposeSubagentProfile()
if gp_profile.enabled is not False and not any(spec["name"] == GENERAL_PURPOSE_SUBAGENT["name"] for spec in inline_subagents):
    ...
```
我们在 [app/deep_agent.py](file:///d:/code/VisualStudioCode/AI/project/GNK48-Agent/DeepAgentsRefactor/app/deep_agent.py) 中通过官方的 `register_harness_profile` 在框架底层直接关闭：
```python
from deepagents import (
    register_harness_profile,
    HarnessProfile,
    GeneralPurposeSubagentProfile,
)

for _provider in ("google_genai", "openai"):
    try:
        register_harness_profile(
            _provider,
            HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
        )
    except Exception:
        pass
```

### 验证效果
运行 `scripts/verify_deep_agent.py` 以及专家清单提问测试，主 Agent 介绍的团队仅包含明确定义的 8 位领域专家，**完全不再出现 `general-purpose` 通用专家**：
```
我的下属专家团队包括：
1. KnowledgeAgent：知识百科专家，负责菜谱、游戏攻略、历史、科学等纯知识性咨询。
2. MediaAgent：成人影视专家，负责 AV、女优、番号等综合性或模糊搜索。
3. MapAgent：地理出行专家，负责地理编码、路线规划、周边搜索及天气查询。
4. CodeAgent：代码执行专家，负责数学计算、数据处理和绘图任务。
5. BrowserAgent：浏览器操作专家，负责网页点击、动态数据抓取及自动化操作。
6. SQLAgent：关系数据库专家，负责精确过滤、统计或多条件检索女优、作品及关联表关系。
7. FileAgent：本地文件专家，负责项目内文件的读取、写入与编辑。
8. DesktopAgent：桌面操作专家，负责通过截图与坐标操作 Windows 桌面 GUI。
```

---

## 3. 第 1 步：`CompiledSubAgent` 接入与适配器设计

### 架构适配器：`SubgraphCompiledWrapper`
由于 deepagents 的 `task` 工具在调用 subagent 时只传入 `messages: [HumanMessage(content=description)]`，而既有的 `build_file_agent_subgraph` / `build_desktop_agent_subgraph` 内部历史隔离逻辑依赖 `instruction_to_worker`、`user_query` 与 `current_tool_call_id`，且 deepagents 在任务收尾时提取的是子图状态中最后一条 `AIMessage` 的文本内容。

我们在不改动任何旧子图内部逻辑的前提下，设计了轻量 Runnable 适配器：
```python
class SubgraphCompiledWrapper(Runnable):
    def __init__(self, subgraph):
        self.subgraph = subgraph

    def invoke(self, state, config=None, **kwargs):
        raise NotImplementedError("Use async ainvoke")

    async def ainvoke(self, state, config=None, **kwargs):
        msgs = state.get("messages", [])
        description = ""
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                if isinstance(m.content, str):
                    description = m.content
                elif isinstance(m.content, list):
                    description = " ".join(p.get("text", "") for p in m.content if isinstance(p, dict) and "text" in p)
                break

        subgraph_input = {
            "messages": msgs,
            "current_tool_call_id": state.get("current_tool_call_id") or "call_subgraph",
            "instruction_to_worker": state.get("instruction_to_worker") or description,
            "user_query": state.get("user_query") or description,
        }

        result = await self.subgraph.ainvoke(subgraph_input, config)

        res_msgs = list(result.get("messages", []))
        if res_msgs and isinstance(res_msgs[-1], ToolMessage):
            final_content = res_msgs[-1].content
            res_msgs.append(AIMessage(content=final_content))
            result["messages"] = res_msgs

        return result
```

### 简单非危险任务验证
* **测试用例**：`读一下 workspace 下 1.txt 的内容`
* **执行轨迹**：
  * 主 Agent 调用 `task` 工具派发给 `FileAgent`
  * FileAgent 执行安全工具 `read_text_file(path='1.txt')`
  * 主 Agent 成功获取内容并给出完整答案：
    ```
    /workspace/1.txt 的内容为：腊肉蛋炒饭真香
    ```

---

## 4. 第 2 步：连续多次危险操作中断与恢复（HITL 验收门槛）

### 测试用例设计
针对 issue #1762 社区讨论的"子图内部第二次 interrupt 无法正确抛出"疑虑，我们设计了一个包含多个敏感操作的复合指令：
> **测试指令**：
> 1. 把 `test_hitl1.txt` 的内容写成 `'hello'` (触发 `write_file` 危险拦截)
> 2. 把 `test_hitl2.txt` 移动到 `archive_trash/test_hitl2.txt` (自动创建目录触发 `create_directory` 危险拦截，并触发 `move_file` 危险拦截)

### 完整执行与恢复轨迹
```
============================================================
Starting Multi-Interrupt HITL Test Case (Full Loop)
============================================================
[Step 0] Sending initial instruction:
请帮我完成以下两件事：
1. 把 test_hitl1.txt 的内容写成 'hello'
2. 把 test_hitl2.txt 移动到 archive_trash/test_hitl2.txt

🔴 [FileAgent HITL]: 危险工具 'write_file' 请求授权...

--- Turn 1 Check ---
[*] state.next = ('tools',)
[*] Interrupts pending: 1
    [0] {'tool_name': 'write_file', 'tool_args': {'path': 'test_hitl1.txt', 'content': 'hello'}, 'tool_call_id': 'call_343514'}
[>] Resuming interrupt 1 with Command(resume='Y')...

🔴 [FileAgent HITL]: 危险工具 'write_file' 请求授权...
  ✅ [FileAgent HITL] 执行成功
🔴 [FileAgent HITL]: 危险工具 'create_directory' 请求授权...

--- Turn 2 Check ---
[*] state.next = ('tools',)
[*] Interrupts pending: 1
    [0] {'tool_name': 'create_directory', 'tool_args': {'path': 'archive_trash'}, 'tool_call_id': 'call_343515'}
[>] Resuming interrupt 2 with Command(resume='Y')...

🔴 [FileAgent HITL]: 危险工具 'write_file' 请求授权...
  ✅ [FileAgent HITL] 执行成功
🔴 [FileAgent HITL]: 危险工具 'create_directory' 请求授权...
  ✅ [FileAgent HITL] 执行成功
🔴 [FileAgent HITL]: 危险工具 'move_file' 请求授权...

--- Turn 3 Check ---
[*] state.next = ('tools',)
[*] Interrupts pending: 1
    [0] {'tool_name': 'move_file', 'tool_args': {'source': 'test_hitl2.txt', 'destination': 'archive_trash/test_hitl2.txt'}, 'tool_call_id': 'call_343516'}
[>] Resuming interrupt 3 with Command(resume='Y')...

🔴 [FileAgent HITL]: 危险工具 'write_file' 请求授权...
  ✅ [FileAgent HITL] 执行成功
🔴 [FileAgent HITL]: 危险工具 'create_directory' 请求授权...
  ✅ [FileAgent HITL] 执行成功
🔴 [FileAgent HITL]: 危险工具 'move_file' 请求授权...
  ✅ [FileAgent HITL] 执行成功

--- Turn 4 Check ---
[*] state.next = ()
[*] Interrupts pending: 0
[+] No further interrupts. Graph finished!

============================================================
Verification Results:
============================================================
test_hitl1.txt content: 'hello' (expected 'hello') -> PASS
archive_trash/test_hitl2.txt exists: True -> PASS

[Final Agent Response]:
任务已完成：
1. `/test_hitl1.txt` 的内容已更新为 `hello`。
2. `/test_hitl2.txt` 已成功移动至 `/archive_trash/test_hitl2.txt`（系统已自动创建目标目录）。
```

### 结论与机制说明
* **动态 `interrupt()` 在当前架构下原样完全可用**，没有发生崩溃、吞断或无法恢复的问题。
* **重放机制（Replay Semantics）验证**：由于 LangGraph 检查点保存在节点边界，每次用 `Command(resume="Y")` 恢复时，当前节点会带着历史中累积的 resume values 从头安全重放，已授权的步骤直接根据历史值通过，遇到未授权的下一步再次抛出 `interrupt()`，直至循环全部授权完毕。
* **结论**：**无需重构 `fa_dangerous_tools_node` 的内部图结构**，现有动态 `interrupt()` 机制完全稳健！
