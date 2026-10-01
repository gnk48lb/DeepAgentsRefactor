import sys, os, asyncio
sys.path.insert(0, os.path.abspath("."))
from app import deep_agent, database
from main_deepagents import run_agent

async def main():
    database.run_global_database_init()
    agent = await deep_agent.build_main_agent()
    config_dict = {
        "recursion_limit": 30,
        "configurable": {"thread_id": "test_huoshen_query_thread"},
    }
    query = "火神战姬的技能有哪些？"
    final_ans = await run_agent(query, agent, config_dict)
    print("\n================== [FINAL_ANSWER_FULL] ==================")
    print(final_ans)
    print("================== [END_FINAL_ANSWER] ==================\n")

if __name__ == "__main__":
    asyncio.run(main())
