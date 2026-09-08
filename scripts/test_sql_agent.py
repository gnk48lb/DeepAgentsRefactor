import os
import re
import pymysql
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", message=".*create_react_agent.*")

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 加载环境变量
load_dotenv()

# 数据库配置 (请根据实际情况在 .env 中配置或直接修改)
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "8964")
DB_NAME = os.getenv("DB_NAME", "av_db")
DB_PORT = int(os.getenv("DB_PORT", 8964))

# 定义 System Prompt，包含完整的 Schema 和要求
SYSTEM_PROMPT = """你是一个专业的 Text-to-SQL 助手。你的任务是根据提供的数据库结构，将用户的自然语言问题转化为 SQL 语句并执行，最后根据查询结果回答用户。

【数据库 Schema】
数据库 `av_db` 包含日本AV资料数据，具有以下三张核心表：
1. `actresses` (演员表)
   - `name` (VARCHAR, 主键)
   - `birth_year` (INT, 出生年份)
   - `cup_size` (VARCHAR, 罩杯)
   - `height` (INT, 身高)

2. `works` (作品表)
   - `code` (VARCHAR, 主键，如番号)
   - `title` (VARCHAR, 标题)
   - `year` (INT, 发行年份)

3. `work_actress` (关联表)
   - `work_code` (VARCHAR, 外键关联 works.code)
   - `actress_name` (VARCHAR, 外键关联 actresses.name)

【核心规则】
1. **工具使用**：必须使用 `execute_sql` 工具来执行生成的 SQL。
2. **安全性**：只能生成并执行 `SELECT` 查询，严禁使用 `INSERT`, `UPDATE`, `DELETE`, `DROP` 等操作。
3. **模糊匹配**：如果遇到不知道的精确匹配词，或者用户提供的名字可能不完全准确，请考虑使用 `LIKE` 进行模糊匹配 (如 `name LIKE '%xxx%'`)。
4. **自我纠错**：如果 `execute_sql` 工具返回了错误信息，请仔细阅读错误堆栈，修复你的 SQL 语法或逻辑，并重新调用工具（你最多重试 3 次）。
5. **最终输出**：当获取到查询结果后，请综合用户的原始问题，给出一份清晰、准确的自然语言回答，不要只贴冰冷的数据。
"""

@tool
def execute_sql(sql_query: str) -> str:
    """执行 SQL SELECT 查询并返回结果。"""
    # 1. 安全性检查：只允许 SELECT
    if not re.match(r"^\s*SELECT", sql_query, re.IGNORECASE):
        return "Error: Security Check Failed. Only SELECT statements are allowed."
    
    # 2. 性能要求：如果没有 LIMIT，自动加上 LIMIT 50
    if not re.search(r"\bLIMIT\b", sql_query, re.IGNORECASE):
        sql_query = sql_query.strip().rstrip(";") + " LIMIT 50"

    connection = None
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            result = cursor.fetchall()
            return str(result)
    except Exception as e:
        # 返回详细错误信息，而不是抛出异常崩溃
        return f"Database Error: {type(e).__name__} - {str(e)}"
    finally:
        if connection:
            connection.close()

def main():
    print("========================================")
    print("      SQLAgent (Text-to-SQL) 测试终端     ")
    print("========================================")
    
    # 注意：这里默认使用 OpenAI 兼容的客户端。
    # 确保你的 .env 中配置了 OPENAI_API_KEY (如果使用中转 API，请配置 OPENAI_API_BASE)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key="AIzaSyBsbDZSna5xyiocLhLG8hRxNG_3HfCy_0U",
        temperature=0
    )

    
    tools = [execute_sql]
    
    # 使用 LangGraph 的 create_react_agent 实现 ReAct / Tool-calling 范式
    # 它天然支持工具调用的多轮交互和自我纠错
    # agent_executor = create_react_agent(
    #     llm, 
    #     tools=tools, 
    #     prompt=SYSTEM_PROMPT
    # )
    agent_executor = create_agent(
        model=llm,                            # 2. 第一个参数在新的 API 中命名为 model
        tools=tools,
        system_prompt=SYSTEM_PROMPT # 3. 原来的 prompt 参数变更为 system_prompt
    )    
    print("Agent 初始化完成！(输入 'exit' 退出)")
    
    while True:
        try:
            user_input = input("\n[User]: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
                
            print("\n[Agent 思考与执行中...]")
            # 限制图的递归深度，防止无限死循环（对应最大重试次数等）
            config = {"recursion_limit": 10} 
            response = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]}, config)
            
            # 打印完整的消息轨迹 (Message Trajectory)
            print("\n" + "="*20 + " 完整交互轨迹 (ReAct Process) " + "="*20)
            for msg in response["messages"]:
                if msg.type == "system":
                    # 系统提示词过长，跳过打印
                    continue
                elif msg.type == "human":
                    print(f"\n[User]: {msg.content}")
                elif msg.type == "ai":
                    if msg.content:
                        print(f"\n[Agent 思考/回复]:\n{msg.content}")
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"\n[Agent 动作]: 调用工具 '{tc['name']}'")
                            if tc['name'] == 'execute_sql':
                                # 使用亮青色高亮 SQL 语句
                                print(f"  --> \033[96m生成的 SQL: {tc['args'].get('sql_query')}\033[0m")
                            else:
                                print(f"  --> 参数: {tc['args']}")
                elif msg.type == "tool":
                    print(f"\n[工具执行结果 ({msg.name})]:")
                    res_str = str(msg.content)
                    if len(res_str) > 500:
                        print(f"  {res_str[:500]} ... (已省略 {len(res_str) - 500} 字符)")
                    else:
                        print(f"  {res_str}")
            print("\n" + "=" * 62)
            
        except Exception as e:
            print(f"\n[Agent 发生异常]: {str(e)}")

if __name__ == "__main__":
    main()
