import logging
from typing import List, Dict
from .database import get_client, execute_hybrid_search, rerank_candidates
from .transforms import MediaQueryTranslator
import config
from . import models

logger = logging.getLogger(__name__)

class MediaGraphRetriever:
    def __init__(self):
        from neo4j import GraphDatabase
        self.neo4j_driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        self.client = get_client()
        self.collection_name = config.MEDIA_COLLECTION_NAME
        self.initialized = False
        
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            self.initialized = True
        else:
            logger.warning(f"Collection {self.collection_name} not found. MediaAgent search disabled.")

        import os
        bm25_path = "storage/av_bm25_model.json"
        if os.path.exists(bm25_path):
            try:
                models.av_bm25_ef.load(bm25_path)
            except Exception as e:
                logger.warning(f"Failed to load av_bm25_model.json: {e}")

        self.translator = MediaQueryTranslator()

    def search(self, query: str) -> str:
        if not self.initialized:
            return "❌ [错误] 专有媒体数据库尚未建立，请先执行数据构建脚本。"
            
        intent = self.translator.translate(query)
        logger.info(f"Query Intent: {intent.model_dump_json()}")
        
        graph_candidates = []
        if intent.search_mode == "hybrid_graph":
            graph_candidates = self._graph_search(intent.extracted_entities)
            
        vector_candidates = self._vector_search(query)
        
        merged_candidates = {}
        for c in graph_candidates:
            if c['code'] not in merged_candidates:
                merged_candidates[c['code']] = c
        for c in vector_candidates:
            if c['code'] not in merged_candidates:
                merged_candidates[c['code']] = c
                
        candidate_list = list(merged_candidates.values())
        logger.info(f"Merged candidate pool size: {len(candidate_list)}")
        
        if not candidate_list:
            return "未能检索到相关作品。"

        texts_for_rerank = [c['text'] for c in candidate_list]
        top_k = 10
        best_hits = rerank_candidates(query, texts_for_rerank, candidate_list, top_k=top_k)
        
        output = []
        for score, hit in best_hits:
            actresses = hit.get('actresses', '')
            actors = hit.get('actors', '')
            people = actresses
            if actors:
                people += f", {actors}" if people else actors
                
            code = hit.get('code', '未知')
            title = hit.get('title', '未知')
            tags = hit.get('tags', '')
            
            output.append(f"- **{code}**: {title} | 主要演员: {people} | 标签: {tags}")
            
        return "\n".join(output)

    def _graph_search(self, entities) -> List[Dict]:
        candidates = []
        people = entities.people
        tags = entities.tags
        
        with self.neo4j_driver.session() as session:
            if people or tags:
                q_weighted = """
                MATCH (w:Work)
                OPTIONAL MATCH (p)-[:ACTED_IN]->(w) WHERE (p:Actress OR p:Actor) AND p.name IN $people
                OPTIONAL MATCH (w)-[:HAS_TAG]->(t:Tag) WHERE t.name IN $tags
                WITH w, count(DISTINCT p) AS p_match, count(DISTINCT t) AS t_match
                WHERE (p_match > 0 OR t_match > 0)
                  AND ($target_type IS NULL OR EXISTS {
                      MATCH (target_p)-[:ACTED_IN]->(w)
                      WHERE ANY(label IN labels(target_p) WHERE label = $target_type)
                  })
                RETURN w.code AS code, w.title AS title
                ORDER BY (p_match * 10 + t_match) DESC
                LIMIT 15
                """
                records = session.run(q_weighted, people=people, tags=tags, target_type=entities.target_person_type)
                
                codes = [rec["code"] for rec in records]
                if codes:
                    codes_str = "[" + ",".join([f"'{c}'" for c in codes]) + "]"
                    expr = f"code in {codes_str}"
                    res = self.client.query(
                        collection_name=self.collection_name,
                        filter=expr,
                        output_fields=["code", "title", "actresses", "actors", "tags", "text"]
                    )
                    for r in res:
                        candidates.append(r)
        return candidates

    def _vector_search(self, query: str) -> List[Dict]:
        results = execute_hybrid_search(
            query=query, 
            client=self.client,
            collection_name=self.collection_name,
            bm25_model=models.av_bm25_ef, 
            output_fields=["code", "title", "actresses", "actors", "tags", "text"],
            limit=15,
            embeddings_impl=models.av_embeddings
        )
        
        candidates = []
        for hit in results:
            entity = hit.get('entity', {}) if isinstance(hit, dict) else hit.entity
            candidates.append({
                "code": entity.get("code", ""),
                "title": entity.get("title", ""),
                "actresses": entity.get("actresses", ""),
                "actors": entity.get("actors", ""),
                "tags": entity.get("tags", ""),
                "text": entity.get("text", "")
            })
        return candidates

if __name__ == "__main__":
    from .database import MediaDataBuilder
    builder = MediaDataBuilder()
    builder.build_from_file("data/graphrag/work.jsonl")
