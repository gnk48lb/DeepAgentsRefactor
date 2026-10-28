"""知识库混合检索与重排序逻辑 (所在层: L2)"""

from pymilvus import AnnSearchRequest, RRFRanker
from app.infra import models


def execute_hybrid_search(query: str, client, collection_name, bm25_model=None, output_fields=None, limit=10, embeddings_impl=None):
    if bm25_model is None:
        bm25_model = models.bm25_ef
    if embeddings_impl is None:
        embeddings_impl = models.embeddings
    if output_fields is None:
        output_fields = ["text", "source_path", "chunk_id", "parent_id", "content_type", "image_path"]
        
    dense_vec = embeddings_impl.embed_query(query)
    sparse_matrix = bm25_model.encode_queries([query])
    
    row_sp = sparse_matrix[0].tocsr()
    sp_dict = {int(k): float(v) for k, v in zip(row_sp.indices, row_sp.data)}
    
    search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest([dense_vec], "vector", search_params, limit=limit)
    sparse_req = AnnSearchRequest([sp_dict], "sparse_vector", search_params, limit=limit)
    
    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=[sparse_req, dense_req],
        ranker=RRFRanker(k=60),
        limit=limit,
        output_fields=output_fields
    )
    return results[0] if results and len(results) > 0 else []


def rerank_candidates(query: str, candidates: list, hits: list, top_k: int = 10):
    if not candidates or not hits:
        return []
    pairs = [[query, doc] for doc in candidates]
    try:
        scores = models.reranker.predict(pairs)
    except models.RerankerUnavailable as e:
        print(f"⚠️ 警告：Reranker 服务不可用 ({e})，退回输入顺序。")
        return [(None, hit) for hit in hits[:top_k]]
    
    scored_hits = list(zip(scores, hits))
    scored_hits.sort(key=lambda x: x[0], reverse=True)
    
    return scored_hits[:top_k]
