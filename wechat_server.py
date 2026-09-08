import time
import hashlib
import xml.etree.ElementTree as ET
from fastapi import FastAPI, Request, Response
import uvicorn
from langchain_core.messages import HumanMessage

# --- 引入你的项目核心模块 ---
from app import graph, database, mcp_service
import config

app = FastAPI()
agent_app = None

WECHAT_TOKEN = "gnk48_agent_token_2026"

@app.on_event("startup")
async def startup_event():
    global agent_app
    print("🚀 正在初始化 Multi-Agent 系统...")
    # 数据库全家桶一键初始化
    database.run_global_database_init()
    await mcp_service.initialize_mcp()
    agent_app = graph.build_graph()
    print("✅ 系统初始化完成，FastAPI 启动，正在监听微信公众号请求...")

async def get_agent_response(query: str, user_id: str) -> str:
    global agent_app
    inputs = {"messages": [HumanMessage(content=query)]}
    config_dict = {
        "recursion_limit": 15,
        "configurable": {"thread_id": user_id} 
    }
    final_answer = "抱歉，系统暂时无法处理您的请求。"
    try:
        async for event in agent_app.astream(inputs, config=config_dict):
            for node_name, state_update in event.items():
                if node_name == "Supervisor":
                    next_agent = state_update.get("next_agent", "")
                    messages = state_update.get("messages", [])
                    if next_agent == "FINISH":
                        if messages and hasattr(messages[-1], "name") and messages[-1].name == "Supervisor_Final":
                            final_answer = messages[-1].content
    except Exception as e:
        print(f"❌ Agent 执行异常: {str(e)}")
        final_answer = f"系统内部错误: {str(e)}"
    return final_answer

# 👑 【核心修改点】路由 1：处理微信服务器的接入验证 (GET 请求)
@app.get("/wechat")
async def verify_wechat(signature: str = "", timestamp: str = "", nonce: str = "", echostr: str = ""):
    tmp_list = [WECHAT_TOKEN, timestamp, nonce]
    tmp_list.sort()
    tmp_str = "".join(tmp_list).encode('utf-8')
    hash_str = hashlib.sha1(tmp_str).hexdigest()
    
    if hash_str == signature:
        print("✅ 微信接口签名验证成功！")
        # 强制返回纯文本，且没有任何包裹，微信测试号即可秒过
        return Response(content=echostr, media_type="text/plain")
    else:
        print("❌ 微信接口签名验证失败！")
        return Response(content="Verification Failed", media_type="text/plain")

# 路由 2：处理真实用户发来的对话消息 (POST 请求)
@app.post("/wechat")
async def receive_message(request: Request):
    body = await request.body()
    xml_data = ET.fromstring(body)
    msg_type = xml_data.find("MsgType").text
    from_user = xml_data.find("FromUserName").text
    to_user = xml_data.find("ToUserName").text
    
    if msg_type == "text":
        content = xml_data.find("Content").text
        print(f"\n📩 [微信] 收到用户 {from_user[-4:]} 消息: {content}")
        
        reply_text = await get_agent_response(content, from_user)
        print(f"📤 [微信] 返回给用户: {reply_text[:100]}...")
        
        reply_xml = f"""<xml>
            <ToUserName><![CDATA[{from_user}]]></ToUserName>
            <FromUserName><![CDATA[{to_user}]]></FromUserName>
            <CreateTime>{int(time.time())}</CreateTime>
            <MsgType><![CDATA[text]]></MsgType>
            <Content><![CDATA[{reply_text}]]></Content>
        </xml>"""
        return Response(content=reply_xml, media_type="application/xml")
    else:
        return Response(content="success")

if __name__ == "__main__":
    uvicorn.run("wechat_server:app", host="0.0.0.0", port=8000, reload=False)