from langchain_core.tools import tool
from .database import get_client, execute_hybrid_search, rerank_candidates
from .graph_rag import MediaGraphRetriever
from datetime import datetime
import config
from . import models


@tool
def rag(query: str):
    """从 Milvus 本地知识库中检索相关信息（包含游戏攻略、菜谱等项目内部文档）。"""
    print(f"\n[Tool Call: rag] Original Query: {query}")
    
    try:
        import os
        client = get_client()
        coll_name = config.KNOWLEDGE_COLLECTION_NAME
        
        # 1. 查询转换路由
        queries_to_search = [query]
        if getattr(config, "ENABLE_QUERY_TRANSFORM", False):
            from . import transforms as query_transforms
            strategy = query_transforms.route_query(query)
            print(f"--> [Query Transform] Strategy selected: {strategy}")
            
            if strategy == "hyde":
                hypo_doc = query_transforms.generate_hyde(query)
                print(f"    - HyDE Document: {hypo_doc[:50]}...")
                queries_to_search.append(hypo_doc)
            elif strategy == "step_back":
                step_back_q = query_transforms.generate_step_back(query)
                print(f"    - Step-Back Query: {step_back_q}")
                queries_to_search.append(step_back_q)
            elif strategy == "multi_query":
                alt_qs = query_transforms.generate_multi_query(query)
                for i, alt_q in enumerate(alt_qs):
                    print(f"    - Alt Query {i+1}: {alt_q}")
                queries_to_search.extend(alt_qs)
                
        # 2. 多路并发/顺序检索
        all_hits = {}
        for q in queries_to_search:
            hits = execute_hybrid_search(q, client, coll_name)
            for hit in hits:
                hit_id = hit.get('id') if isinstance(hit, dict) else getattr(hit, 'id', None)
                if hit_id not in all_hits:
                    all_hits[hit_id] = hit
                    
        unique_hits = list(all_hits.values())
        
        if not unique_hits:
            print("[Tool Result] No relevant context found in Milvus.")
            return "No relevant information found in the knowledge base."
            
        # 3. 执行本地重排 (Reranking)
        # 注意：MilvusClient 返回的是 list[dict]
        candidates = [hit.get('entity', {}).get('text', '') if isinstance(hit, dict) else hit.entity.get('text', '') for hit in unique_hits]
        
        # 取 Top-3 (可根据需要调整)
        top_k = 3
        best_hits = rerank_candidates(query, candidates, unique_hits, top_k=top_k)
        
        context_docs = []
        image_paths = set()
        
        for score, hit in best_hits:
            entity = hit.get('entity', {}) if isinstance(hit, dict) else hit.entity
            text = entity.get('text', '')
            source_path = entity.get('source_path', '')
            chunk_id = entity.get('chunk_id', '')
            content_type = entity.get('content_type', 'text')
            image_path = entity.get('image_path', '')
            
            # 如果是图像描述，提取完整小作文并记录图片路径
            if content_type == "image_description" and image_path:
                full_summary_path = os.path.join("data", source_path)
                try:
                    if os.path.exists(full_summary_path):
                        with open(full_summary_path, "r", encoding="utf-8") as f:
                            text = f.read()
                except Exception as e:
                    print(f"Failed to read full summary from {full_summary_path}: {e}")
                
                image_paths.add(image_path)
            
            meta_str = f"[{source_path}] " if source_path else ""
            meta_str += f"({chunk_id}): " if chunk_id else ""
            context_docs.append(f"{meta_str}{text}")
            
        context = "\n".join(context_docs)
        
        # 4. 根据是否包含图片决定返回类型
        if image_paths:
            from .loader import compress_image_to_base64
            
            multimodal_content = []
            
            # 先添加图片，再添加文字描述，有助于模型更好地结合视觉信息
            for img_rel_path in image_paths:
                full_img_path = os.path.join("data", img_rel_path)
                if os.path.exists(full_img_path):
                    b64_img = compress_image_to_base64(full_img_path, max_size=(800, 800), quality=80)
                    if b64_img:
                        multimodal_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        })
            
            final_text = (
                context +
                "\n\n[系统提示：以上检索结果包含从图片中提取的文本描述（小作文）及原始图片。请遵循以下准则：\n"
                "1. 事实依据优先：如果文字描述中已包含问题的答案，请优先以文字内容为准进行回答。\n"
                "2. 视觉补充：文字描述仅是图片的摘要提取，可能遗漏细微特征（如角落的图标、水印、微小物件等）。如果文字中缺失用户询问的细节，严禁仅凭文字反馈“没有提到”或“不存在”，请务必直接观察并分析原始图片中的内容来回答。]"
            )
            multimodal_content.append({"type": "text", "text": final_text})
            
            print(f"[Tool Result] Reranked Top-{top_k} context (Multimodal, {len(image_paths)} images included)")
            return multimodal_content
        else:
            print(f"[Tool Result] Reranked Top-{top_k} context: {context[:200]}...")
            return context
            
    except Exception as e:
        print(f"[Tool Error] Search failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Error searching knowledge base: {e}"


@tool
def get_current_time(placeholder: str = "default") -> str:
    """Returns the current date and time."""
    print("\n[Tool Call: get_current_time]")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def web_search(query: str) -> str:
    """【优先级最低：兜底工具】只有当 rag 工具未能从本地知识库返回任何有效信息时，才允许调用此工具搜索全网实时信息或百科。"""
    print(f"\n[Tool Call: web_search] Query: {query}")
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        # 注意：TavilySearchResults 会自动读取环境变量 TAVILY_API_KEY
        search = TavilySearchResults(max_results=3)
        results = search.run(query)
        return str(results)
    except Exception as e:
        return f"网络搜索失败: {str(e)}"

@tool("av_graph_rag")
def av_graph_rag(query: str) -> str:
    """专门用于检索日本AV作品、番号、女优、男优、题材标签（如人妻、温泉、熟女等）相关信息的工具。当用户询问任何关于成人视频、特定番号作品或相关演员时，必须优先且仅能调用此工具。"""
    print(f"\n[Tool Call: av_graph_rag] Query: {query}")
    try:
        retriever = MediaGraphRetriever()
        return retriever.search(query)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error executing AV graph search: {e}"

@tool("execute_sql")
def execute_sql(sql_query: str) -> str:
    """执行 SQL SELECT 查询并返回结果。当需要精确过滤/多条件筛选/统计计算（如女优身高、罩杯统计、发行年份过滤等）本库 MySQL 数据时调用。"""
    print(f"\n[Tool Call: execute_sql] SQL: {sql_query}")
    import re
    import pymysql
    
    # 1. 安全性检查：只允许 SELECT
    if not re.match(r"^\s*SELECT", sql_query, re.IGNORECASE):
        return "Error: Security Check Failed. Only SELECT statements are allowed."
    
    # 2. 性能要求：如果没有 LIMIT，自动加上 LIMIT 50
    if not re.search(r"\bLIMIT\b", sql_query, re.IGNORECASE):
        sql_query = sql_query.strip().rstrip(";") + " LIMIT 50"

    connection = None
    host = config.MYSQL_HOST
    user = config.MYSQL_USER
    password = config.MYSQL_PASSWORD
    database = config.MYSQL_DATABASE
    port = int(config.MYSQL_PORT)

    # 自动识别初始占位配置并进行回退，确保即使没修改 .env 也能跑通
    if database == "my_database" and port == 3306:
        host = "127.0.0.1"
        password = "8964"
        database = "av_db"
        port = 8964

    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            result = cursor.fetchall()
            return str(result)
    except Exception as e:
        return f"Database Error: {type(e).__name__} - {str(e)}"
    finally:
        if connection:
            connection.close()

# 工具列表
tools = [rag, av_graph_rag, get_current_time, web_search, execute_sql]


# ── CodeAgent 专用工具 ────────────────────────────────────────────────────────

@tool("execute_python_code")
def execute_python_code(code: str) -> any:
    """
    在本地 Docker 沙箱中执行 Python 代码，用于数据分析、数学计算和图表生成。
    沙箱已预装 pandas、numpy、matplotlib。
    若需绘图，代码中必须将图片以 savefig 保存到 '/workspace/outputs/' 目录，且以 .png 结尾，禁止调用 plt.show()。
    返回执行结果（包含 stdout、stderr）及生成的图片（若有）。
    """
    print(f"\n[Tool Call: execute_python_code] 代码长度: {len(code)} 字符")
    try:
        from .sandbox import run_code_in_sandbox
        res = run_code_in_sandbox(code)

        # 记录终端日志（仅 stdout 前 200 字符，避免刷屏）
        print(f"  stdout: {res['stdout'][:200]}")
        if res["stderr"]:
            print(f"  stderr: {res['stderr'][:200]}")
        print(f"  images: {len(res['images'])} 张")

        # 【多模态组装】：有图片时返回 list，与现有拦截器无缝对接
        if res["images"]:
            content = [
                {
                    "type": "text",
                    "text": f"Stdout:\n{res['stdout']}\nStderr:\n{res['stderr']}"
                }
            ]
            for img_b64 in res["images"]:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            return content
        else:
            # 无图片时返回纯文本，保持简洁
            return f"Stdout:\n{res['stdout']}\nStderr:\n{res['stderr']}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"沙箱执行失败: {type(e).__name__}: {e}"


# CodeAgent 专用工具列表
code_tools = [execute_python_code]

# ── BrowserAgent 专用工具 ───────────────────────────────────────────────────────

from .browser_service import fetch_generic_webpage_content, fetch_bilibili_profile_data, run_browser_task

@tool
async def read_webpage_content(url: str) -> str:
    """用于分析普通网页、读取新闻/文章内容、总结网页。当用户扔来一个网页链接并要求“总结这个网页”、“这个网页说了什么”时，必须优先调用此工具。注意：输入必须是完整的 URL。"""
    return await fetch_generic_webpage_content(url)

@tool
async def get_bilibili_profile(uid: str) -> str:
    """专门用于获取B站(bilibili)用户的个人主页资料和近期动态。当用户要求“总结这个B站个人主页”、“分析这个B站UP主”时调用。输入必须是从链接中提取的纯数字 UID。"""
    return await fetch_bilibili_profile_data(uid)

@tool
async def execute_complex_browser_action(instruction: str) -> str:
    """用于需要模拟真人操作浏览器的复杂网页任务，例如：在网页上点击按钮、输入文字搜索、登录、翻页，或是处理极度复杂的动态交互网页。仅当普通的网页内容读取(read_webpage_content)无法满足需求，或用户明确要求“操作”、“点击”时才使用此工具。"""
    # 这里直接调用原有的服务函数，传入配置好的 VLM
    return await run_browser_task(instruction, models.browser_vlm)

# 请将这三个工具组合成一个列表供 Agent 使用
browser_tools = [read_webpage_content, get_bilibili_profile, execute_complex_browser_action]
