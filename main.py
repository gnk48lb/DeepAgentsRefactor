# Windows os.rename compatibility monkey-patch for milvus-lite FileExistsError (WinError 183)
import sys
import os
import logging

# Suppress all third-party verbose INFO logs immediately
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("milvus_lite").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("pymilvus").setLevel(logging.WARNING)

if sys.platform == "win32":
    _orig_rename = os.rename
    def _safe_rename(src, dst):
        try:
            _orig_rename(src, dst)
        except (FileExistsError, OSError) as e:
            try:
                os.replace(src, dst)
            except Exception:
                raise e
    os.rename = _safe_rename

from app import graph
from app import database
import config
from langchain_core.messages import HumanMessage
import asyncio
import traceback
from app import mcp_service
import sys

async def run_agent(query: str, app):
    """运行 Agent 并在控制台打印炫酷动态流式输出"""
    inputs = {"messages": [HumanMessage(content=query)]}
    
    print(f"\n--- 🚀 Running Agent for Query: '{query}' ---")
    
    # 启用防死循环配置 recursion_limit，并传入必填的 thread_id (LangGraph Checkpointer 要求)
    config_dict = {
        "recursion_limit": 15,
        "configurable": {"thread_id": "console_user"}
    }
    try:
        async for event in app.astream(inputs, config=config_dict):
            for node_name, state_update in event.items():
                if node_name == "Supervisor":
                    plan = state_update.get("plan", "")
                    if plan:
                        print(f"\n🟢 \033[92m[Supervisor 反思]\033[0m: {plan}")
                    
                    next_agent = state_update.get("next_agent", "")
                    messages = state_update.get("messages", [])
                    
                    if next_agent == "FINISH":
                        if messages and hasattr(messages[-1], "name") and messages[-1].name == "Supervisor_Final":
                            print(f"\n🌟 \033[96m[最终回答]\033[0m: {messages[-1].content}\n")
                    elif next_agent:
                        # 尝试从最新消息中提取分发的指令
                        instruction = ""
                        if messages and hasattr(messages[-1], "name") and messages[-1].name == "Supervisor":
                            instruction = messages[-1].content.replace("[主管指令] ", "")
                        print(f"\n🔵 \033[94m[派发任务给 {next_agent}]\033[0m: {instruction}")
                        
                elif node_name in ["MapAgent", "KnowledgeAgent", "MediaAgent", "FileAgent", "CodeAgent", "BrowserAgent"]:
                    messages = state_update.get("messages", [])
                    if messages:
                        # 最后一个消息应该是 Worker 返回的战报（或者是子图产生的最后一条消息）
                        last_msg = messages[-1]
                        content = last_msg.content
                        
                        # 提取战报，处理多模态列表
                        if isinstance(content, list):
                            has_image = any(isinstance(c, dict) and c.get("type") == "image_url" for c in content)
                            text_parts = []
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    t = c.get("text", "")
                                    text_parts.append(str(t) if not isinstance(t, list) else str(t[0]))
                            display_text = " ".join(text_parts)
                            if len(display_text) > 800:
                                display_text = display_text[:800] + "\n... [内容已截断]"
                            
                            label = f"{node_name} 战报 (图文多模态)" if has_image else f"{node_name} 战报"
                            print(f"\n🟡 \033[93m[{label}]\033[0m: \n{display_text}")
                        else:
                            display_text = str(content)
                            # 为了避免打印太长，可以截断展示
                            if len(display_text) > 800:
                                display_text = display_text[:800] + "\n... [内容已截断]"
                            print(f"\n🟡 \033[93m[{node_name} 战报]\033[0m: \n{display_text}")
    except Exception as e:
        # 捕捉可能是因为 recursion_limit 触发的 GraphRecursionError
        print(f"\n❌ \033[91m[Agent 执行异常/中断]\033[0m: {str(e)}")

async def main_loop():
    try:
        print("启动 Supervisor Multi-Agent 架构...")
        
        # 启动前检测命令行参数
        if "--rebuild" in sys.argv:
            config.REFRESH_COLLECTION = True
            print("🚀 检测到 --rebuild 参数，将强制重建知识库索引！")

        # 数据库全家桶一键初始化
        database.run_global_database_init()

        # 初始化 MCP 服务 (高德地图)
        await mcp_service.initialize_mcp()

        # MCP 初始化完成后，动态构建计算图
        app = graph.build_graph()

        print("\n" + "="*50)
        print("欢迎使用 GNK48-Agent ！(Supervisor Version)")
        print("输入 'exit' 退出程序。")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n您的问题: ")
                user_input = user_input.strip()
                
                # 兼容不同系统终端编码
                try:
                    user_input = user_input.encode("utf-8", "ignore").decode("utf-8")
                except Exception:
                    pass
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit']:
                    print("👋 感谢使用，再见！")
                    break
                    
                await run_agent(user_input, app)
                
            except KeyboardInterrupt:
                print("\n👋 感谢使用，再见！")
                break
            except Exception as e:
                traceback.print_exc()
                print(f"处理问题时出错: {e}")
                
    except Exception as e:
        traceback.print_exc()
        print(f"系统启动失败: {e}")

if __name__ == "__main__":
    asyncio.run(main_loop())