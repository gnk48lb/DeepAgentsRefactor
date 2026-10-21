"""专有媒体库与知识图谱数据构建器 (所在层: L2)"""

import os
import json
import logging
import shutil
from typing import List, Dict, Any
from neo4j import GraphDatabase
from pymilvus import DataType
from app import config
from app.infra import models
from app.infra.milvus_client import get_client

logger = logging.getLogger(__name__)


def init_av_graph_rag():
    """
    初始化专有媒体数据库 (GraphRAG)：
    1. 检查 av_works_collection 集合。
    2. 如果集合不存在或为空，则触发 DataBuilder 进行初始化和数据构建。
    """
    try:
        db_builder = MediaDataBuilder()
        c = get_client()
        coll_name = db_builder.collection_name
        
        # 检查数据量，如果为空且存在源文件则构建数据
        is_empty = True
        if c.has_collection(coll_name):
            try:
                res = c.query(collection_name=coll_name, filter="id >= 0", limit=1)
                if res:
                    is_empty = False
            except Exception:
                pass
        
        if is_empty:
            data_file = config.MEDIA_SOURCE_FILE
            if os.path.exists(data_file):
                print(f"--- [初始化] 专有媒体库 '{coll_name}' 为空，正在从 {data_file} 构建数据... ---")
                db_builder.build_from_file(data_file)
                print(f"✅ 专有媒体库数据构建完成。")
            else:
                print(f"⚠️ 警告：未找到专有媒体库源文件 {data_file}，跳过构建。")
        else:
            print(f"✅ 专有媒体库 '{coll_name}' 已就绪。")
            
    except Exception as e:
        print(f"❌ [专有媒体库初始化失败]: {e}")


class MediaDataBuilder:
    def __init__(self):
        # Neo4j Setup
        self.neo4j_driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        
        # Milvus Setup
        self.client = get_client()
        self.collection_name = config.MEDIA_COLLECTION_NAME
        self._init_milvus_collection()

    def _init_milvus_collection(self):
        if getattr(config, "REFRESH_COLLECTION", False):
            coll_dir = os.path.join(config.MILVUS_DB_PATH, "collections", self.collection_name)
            if os.path.exists(coll_dir):
                logger.info(f"--- [刷新与清理] 正在手动物理清理 Windows 本地集合目录 '{self.collection_name}' ---")
                shutil.rmtree(coll_dir, ignore_errors=True)

        if not self.client.has_collection(self.collection_name):
            logger.info(f"--- [初始化] 正在 Windows 本地一键创建混合检索媒体集合 '{self.collection_name}' ---")
            
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

            schema.add_field(field_name="code", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2048)
            schema.add_field(field_name="actresses", datatype=DataType.VARCHAR, max_length=1024)
            schema.add_field(field_name="actors", datatype=DataType.VARCHAR, max_length=1024)
            schema.add_field(field_name="tags", datatype=DataType.VARCHAR, max_length=2048)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="IP",
                index_type="FLAT"
            )
            index_params.add_index(
                field_name="sparse_vector",
                metric_type="IP",
                index_type="SPARSE_INVERTED_INDEX",
                params={"drop_ratio_build": 0.2}
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"✅ 混合检索媒体集合与双路索引已在 Windows 本地一键初始化成功！")
        else:
            self.client.load_collection(self.collection_name)

    def build_from_file(self, file_path: str, batch_size: int = 500):
        records = []
        all_texts = []
        
        logger.info("First pass: collecting texts for BM25 fitting...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    text = self._format_text(row)
                    all_texts.append(text)
                    
        logger.info(f"Fitting BM25 model on {len(all_texts)} records...")
        models.av_bm25_ef.fit(all_texts)
        os.makedirs(config.STORAGE_DIR, exist_ok=True)
        models.av_bm25_ef.save(config.AV_BM25_MODEL_PATH)
        logger.info("BM25 model saved.")

        logger.info("Second pass: batch insert...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                    
                if len(records) >= batch_size:
                    self._process_batch(records)
                    records = []
                    
        if records:
            self._process_batch(records)
            
        self._compute_graph_weights()
        logger.info("Data building complete.")

    def _format_text(self, row: Dict[str, Any]) -> str:
        actresses_str = ", ".join(row.get('actresses', []))
        actors_str = ", ".join(row.get('actors', []))
        tags_str = ", ".join(row.get('tags', []))
        return f"[作品] 编号: {row.get('code', '')} | 标题: {row.get('title', '')} | 女优: {actresses_str} | 男优: {actors_str} | 标签: {tags_str}"

    def _process_batch(self, batch: List[Dict[str, Any]]):
        with self.neo4j_driver.session() as session:
            session.execute_write(self._neo4j_merge_batch, batch)
        self._milvus_insert_batch(batch)

    @staticmethod
    def _neo4j_merge_batch(tx, batch):
        query = """
        UNWIND $batch AS row
        MERGE (w:Work {code: row.code})
        SET w.title = row.title

        FOREACH (actress_name IN coalesce(row.actresses, []) |
            MERGE (a:Actress {name: actress_name})
            MERGE (a)-[:ACTED_IN]->(w)
        )

        FOREACH (actor_name IN coalesce(row.actors, []) |
            MERGE (ac:Actor {name: actor_name})
            MERGE (ac)-[:ACTED_IN]->(w)
        )

        FOREACH (tag_name IN coalesce(row.tags, []) |
            MERGE (t:Tag {name: tag_name})
            MERGE (w)-[:HAS_TAG]->(t)
        )

        FOREACH (ignoreMe IN CASE WHEN row.series IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s:Series {name: row.series})
            MERGE (w)-[:IN_SERIES]->(s)
        )

        FOREACH (ignoreMe IN CASE WHEN row.year IS NOT NULL THEN [1] ELSE [] END |
            MERGE (y:Year {value: row.year})
            MERGE (w)-[:RELEASED_IN_YEAR]->(y)
        )
        """
        tx.run(query, batch=batch)

    def _compute_graph_weights(self):
        logger.info("Computing graph weights...")
        with self.neo4j_driver.session() as session:
            co_star_query = """
            MATCH (p1)-[:ACTED_IN]->(w:Work)<-[:ACTED_IN]-(p2)
            WHERE (p1:Actress OR p1:Actor) AND (p2:Actress OR p2:Actor) AND elementId(p1) < elementId(p2)
            WITH p1, p2, count(w) AS co_star_count
            MERGE (p1)-[r:CO_STARRED_WITH]-(p2)
            SET r.co_star_count = co_star_count
            """
            session.run(co_star_query)
            
            known_for_tag_query = """
            MATCH (a:Actress)-[:ACTED_IN]->(w:Work)-[:HAS_TAG]->(t:Tag)
            WITH a, t, count(w) AS weight
            MERGE (a)-[r:KNOWN_FOR_TAG]->(t)
            SET r.weight = weight
            """
            session.run(known_for_tag_query)
        logger.info("Graph weights computed.")

    def _milvus_insert_batch(self, batch: List[Dict[str, Any]]):
        codes = []
        titles = []
        actresses_list = []
        actors_list = []
        tags_list = []
        texts = []
        
        for row in batch:
            actresses_str = ", ".join(row.get('actresses', []))
            actors_str = ", ".join(row.get('actors', []))
            tags_str = ", ".join(row.get('tags', []))
            text = self._format_text(row)
            
            codes.append(row.get('code', ''))
            titles.append(row.get('title', ''))
            actresses_list.append(actresses_str)
            actors_list.append(actors_str)
            tags_list.append(tags_str)
            texts.append(text)
            
        vectors = []
        hf_batch_size = 32
        for i in range(0, len(texts), hf_batch_size):
            sub_batch = texts[i : i + hf_batch_size]
            sub_vectors = models.av_embeddings.embed_documents(sub_batch)
            if sub_vectors is None:
                raise ValueError(f"HuggingFace Embedding failed for batch chunk {i // hf_batch_size}")
            vectors.extend(sub_vectors)
            
        sparse_matrices = models.av_bm25_ef.encode_documents(texts)
        
        if not vectors or sparse_matrices is None:
            raise ValueError(f"Embedding failed: vectors is {type(vectors)}, sparse_matrices is {type(sparse_matrices)}")
            
        data = []
        for i in range(len(texts)):
            row_sp = sparse_matrices[i].tocsr()
            indices = row_sp.indices
            values = row_sp.data
            
            if indices is None or values is None:
                sp_dict = {}
            else:
                sp_dict = {int(k): float(v) for k, v in zip(indices, values)}
            
            data.append({
                "code": codes[i],
                "title": titles[i],
                "actresses": actresses_list[i],
                "actors": actors_list[i],
                "tags": tags_list[i],
                "text": texts[i],
                "vector": vectors[i],
                "sparse_vector": sp_dict
            })
            
        self.client.insert(collection_name=self.collection_name, data=data)


if __name__ == "__main__":
    MediaDataBuilder().build_from_file(config.MEDIA_SOURCE_FILE)
