"""知识库检索与外部信息查询工具 (所在层: L3)"""

import os
import traceback
from datetime import datetime
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from app import config
from app.infra.milvus_client import get_client
from app.knowledge.search import execute_hybrid_search, rerank_candidates
from app.knowledge import query_transform
from app.utils.images import compress_image_to_base64


def _build_context(best_hits) -> tuple[list[str], list[str]]:
    context_docs = []
    image_paths = []
    seen_summary_paths = set()

    for score, hit in best_hits:
        entity = hit.get('entity', {}) if isinstance(hit, dict) else hit.entity
        text = entity.get('text', '')
        source_path = entity.get('source_path', '')
        chunk_id = entity.get('chunk_id', '')
        content_type = entity.get('content_type', 'text')
        image_path = entity.get('image_path', '')

        # 如果是图像描述，提取完整小作文并记录图片路径
        if content_type == "image_description" and image_path:
            if source_path and source_path in seen_summary_paths:
                continue

            full_summary_path = os.path.join(config.DATA_DIR, source_path)
            read_success = False
            try:
                if os.path.exists(full_summary_path):
                    with open(full_summary_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        read_success = True
            except Exception as e:
                print(f"Failed to read full summary from {full_summary_path}: {e}")

            if read_success and source_path:
                seen_summary_paths.add(source_path)

            if image_path not in image_paths:
                image_paths.append(image_path)

        meta_str = f"[{source_path}] " if source_path else ""
        meta_str += f"({chunk_id}): " if chunk_id else ""
        context_docs.append(f"{meta_str}{text}")

    return context_docs, image_paths


def is_below_threshold(best_hits: list, min_score: float) -> bool:
    """纯函数：判断 rerank 结果是否低于相关度阈值。

    规则：
    - 若 best_hits 为空：视为低于阈值 (True)。
    - 若任一命中项的分数为 None（代表 reranker 不可用并已降级退回）：不做阈值过滤，返回 False。
    - 若分数全部可用：当最高分（max(scores)）低于 min_score 时，返回 True；否则返回 False。
    """
    if not best_hits:
        return True
    scores = [s for s, _ in best_hits]
    if any(s is None for s in scores):
        return False
    return max(scores) < min_score


@tool
def rag(query: str):
    """从 Milvus 本地知识库中检索相关信息（包含游戏攻略、菜谱等项目内部文档）。"""
    print(f"\n[Tool Call: rag] Original Query: {query}")
    
    try:
        client = get_client()
        coll_name = config.KNOWLEDGE_COLLECTION_NAME
        
        # 1. 查询转换路由
        queries_to_search = [query]
        if getattr(config, "ENABLE_QUERY_TRANSFORM", False):
            strategy = query_transform.route_query(query)
            print(f"--> [Query Transform] Strategy selected: {strategy}")
            
            if strategy == "hyde":
                hypo_doc = query_transform.generate_hyde(query)
                print(f"    - HyDE Document: {hypo_doc[:50]}...")
                queries_to_search.append(hypo_doc)
            elif strategy == "step_back":
                step_back_q = query_transform.generate_step_back(query)
                print(f"    - Step-Back Query: {step_back_q}")
                queries_to_search.append(step_back_q)
            elif strategy == "multi_query":
                alt_qs = query_transform.generate_multi_query(query)
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

        # 每次检索打印前 3 名分数便于调参
        score_logs = [f"{s:.4f}" if s is not None else "None" for s, _ in best_hits[:3]]
        print(f"[RAG Rerank] Top-{len(score_logs)} 分数: {score_logs} (阈值: {config.RAG_MIN_RERANK_SCORE})")

        # 阈值判断：rerank 分数可用且最高分低于阈值时，返回固定文案
        if is_below_threshold(best_hits, config.RAG_MIN_RERANK_SCORE):
            print(f"[Tool Result] 最高重排分数低于阈值 {config.RAG_MIN_RERANK_SCORE}，判定为无相关知识。")
            return "No relevant information found in the knowledge base."
        
        context_docs, image_paths = _build_context(best_hits)
            
        context = "\n".join(context_docs)
        
        # 4. 根据是否包含图片决定返回类型
        if image_paths:
            multimodal_content = []
            # 先添加图片，再添加文字描述，有助于模型更好地结合视觉信息
            for img_rel_path in image_paths:
                full_img_path = os.path.join(config.DATA_DIR, img_rel_path)
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
        traceback.print_exc()
        return f"Error searching knowledge base: {e}"


_WEEKDAYS_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@tool
def get_current_time(placeholder: str = "default") -> str:
    """获取当前的日期、时间和星期（本机时区）。用于回答今天几号、星期几、现在几点、今年是哪一年、距离某个日期还有几天等问题。本工具不需要任何参数，placeholder 可忽略。"""
    print("\n[Tool Call: get_current_time]")
    now = datetime.now().astimezone()
    off = now.strftime("%z")  # 例如 +0800
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {_WEEKDAYS_CN[now.weekday()]} (UTC{off[:3]}:{off[3:]})"


@tool
def web_search(query: str) -> str:
    """【优先级最低：兜底工具】只有当 rag 工具未能从本地知识库返回任何有效信息时，才允许调用此工具搜索全网实时信息或百科。"""
    print(f"\n[Tool Call: web_search] Query: {query}")
    try:
        # 注意：TavilySearchResults 会自动读取环境变量 TAVILY_API_KEY
        search = TavilySearchResults(max_results=3)
        results = search.run(query)
        return str(results)
    except Exception as e:
        return f"网络搜索失败: {str(e)}"


# get_current_time 只由主管挂载，不属于 KnowledgeAgent。
TOOLS = [rag, web_search]
