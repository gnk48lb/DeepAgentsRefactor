import os
import json
import logging
import config
from . import models
from . import loader as data_loader
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from neo4j import GraphDatabase
from typing import List, Dict, Any
import shutil

logger = logging.getLogger(__name__)

PENDING_DIR = "data/pending"
SOURCE_DIR = "data/source"

client = None

def get_client() -> MilvusClient:
    global client
    if client is None:
        # Ensure the parent directory of config.MILVUS_DB_PATH exists before initialization
        if not config.MILVUS_DB_PATH.startswith("http://") and not config.MILVUS_DB_PATH.startswith("https://"):
            db_dir = os.path.dirname(config.MILVUS_DB_PATH)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        # Use local Milvus DB Path from config, and disable keep_alive to prevent gRPC too_many_pings on Windows
        client = MilvusClient(uri=config.MILVUS_DB_PATH, keep_alive=False)
    return client

def init_knowledge_base(data_dir="data/source"):
    """
    初始化知识库：
    1. 检查集合是否存在，若开启 REFRESH_COLLECTION 则用物理清理规避 Windows 文件锁 Bug。
    2. 使用新版自适应 Schema 一键混合检索建表并建索引，100% 避开 Windows [WinError 183] 锁机制。
    """
    try:
        coll_name = config.KNOWLEDGE_COLLECTION_NAME
        
        # 1. 物理安全清理逻辑 (Windows 平台下 drop_collection 会遇到 WinError 183 文件锁 Bug 崩溃，用物理删除文件夹代替)
        if getattr(config, "REFRESH_COLLECTION", False):
            import shutil
            coll_dir = os.path.join(config.MILVUS_DB_PATH, "collections", coll_name)
            if os.path.exists(coll_dir):
                print(f"--- [刷新模式] 正在手动物理清理 Windows 本地集合目录 '{coll_name}' ---")
                shutil.rmtree(coll_dir, ignore_errors=True)
        
        c = get_client()
        
        # 2. 一键创建集合与双路索引 (自适应 Schema 与动态字段，100% 绕过 Windows 的文件锁冲突)
        if not c.has_collection(coll_name):
            print(f"--- [初始化] 正在 Windows 本地一键创建混合检索集合 '{coll_name}' ---")
            
            # 声明 Schema 启用 auto_id 和 dynamic_fields
            from pymilvus import DataType
            schema = c.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            
            # 显式声明全部检索返回字段，彻底解决 Milvus Lite 无法读取 dynamic_field 导致的 Arrow KeyError 底层 Bug
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="source_path", datatype=DataType.VARCHAR, max_length=2048)
            schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=256)
            schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=2048)
            schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=256)

            # 使用自适应 Schema 接口，直接打包多路索引参数，一次性建立
            index_params = c.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="FLAT", metric_type="IP")
            index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

            c.create_collection(
                collection_name=coll_name,
                schema=schema,
                index_params=index_params
            )
            print(f"✅ 混合检索集合与双路索引已在 Windows 本地一键初始化成功！")
        else:
            c.load_collection(collection_name=coll_name)

        # 3. 检查数据量，决定是否加载文档 (避免 REFRESH_COLLECTION=False 时重复入库)
        is_empty = True
        try:
            # 用一个永远为真的条件来探测是否有数据
            res = c.query(collection_name=coll_name, filter="id >= 0", limit=1)
            if res:
                is_empty = False
        except:
            pass
        
        if not is_empty:
            # 必须加载本地保存稀疏向量编码的 BM25 模型，否则无法对新 Query 进行编码
            bm25_path = "storage/bm25_model.json"
            if os.path.exists(bm25_path):
                models.bm25_ef.load(bm25_path)
                print(f"✅ 知识库 '{coll_name}' 已有数据且 BM25 词频模型就绪，跳过文档加载与向量化。")
                return
            else:
                print("⚠️ 警告：找到知识库数据，但丢失了 bm25_model.json！自动物理重建...")
                import shutil
                coll_dir = os.path.join(config.MILVUS_DB_PATH, "collections", coll_name)
                if os.path.exists(coll_dir):
                    shutil.rmtree(coll_dir, ignore_errors=True)
                return init_knowledge_base(data_dir)

        # 5. 执行文档加载
        print(f"--- [加载] 知识库为空，准备从 {data_dir} 目录递归挂载知识 ---")
        docs = data_loader.load_and_split_markdown(data_dir=data_dir)
        
        if docs:
            print(f"提取到 {len(docs)} 个片段，开始双路向量化并入库...")
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            
            # 生成稠密向量
            vectors = models.embeddings.embed_documents(texts)
            
            # 训练并生成稀疏向量
            print(f"正在进行 BM25 词频统计拟合...")
            models.bm25_ef.fit(texts)
            os.makedirs("storage", exist_ok=True)
            models.bm25_ef.save("storage/bm25_model.json")
            sparse_vectors = models.bm25_ef.encode_documents(texts)
            
            data = []
            for i, txt in enumerate(texts):
                # 将稀疏矩阵单行转为字典格式以供单条数据插入
                row_sp = sparse_vectors[i].tocsr()
                sp_dict = {int(k): float(v) for k, v in zip(row_sp.indices, row_sp.data)}
                
                row = {
                    "vector": vectors[i],
                    "sparse_vector": sp_dict,
                    "text": txt
                }
                row.update(metadatas[i])
                data.append(row)
            
            res = c.insert(collection_name=coll_name, data=data)
            c.load_collection(collection_name=coll_name)
            print(f"✅ 成功插入 {res['insert_count']} 条数据。")
        else:
            print(f"⚠️ 警告：{data_dir} 目录下未找到有效的 Markdown 文档。")
            
    except Exception as e:
        print(f"❌ [初始化失败]: {e}")

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
            except:
                pass
        
        if is_empty:
            data_file = "data/graphrag/work.jsonl"
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

def init_memory_collection():
    try:
        coll_name = config.MEMORY_COLLECTION_NAME
        
        # 1. 物理安全清理逻辑 (Windows 平台下 drop_collection 会遇到 WinError 183 文件锁 Bug 崩溃，用物理删除文件夹代替)
        if getattr(config, "CLEAR_MEMORY_ON_STARTUP", False) or getattr(config, "REFRESH_COLLECTION", False):
            import shutil
            coll_dir = os.path.join(config.MILVUS_DB_PATH, "collections", coll_name)
            if os.path.exists(coll_dir):
                print(f"--- [刷新与清理] 正在手动物理清理 Windows 本地记忆集合目录 '{coll_name}' ---")
                shutil.rmtree(coll_dir, ignore_errors=True)

        c = get_client()

        # 2. 一键创建集合与稠密向量索引 (自适应 Schema 与动态字段，100% 绕过 Windows 的文件锁冲突)
        if not c.has_collection(coll_name):
            print(f"--- [初始化] 正在 Windows 本地一键创建长期记忆集合 '{coll_name}' ---")
            
            from pymilvus import DataType
            schema = c.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            # 显式声明 text 字段，规避 Milvus Lite 的动态字段 Arrow KeyError Bug
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

            index_params = c.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="FLAT", metric_type="IP")

            c.create_collection(
                collection_name=coll_name,
                schema=schema,
                index_params=index_params
            )
            print(f"✅ 长期记忆集合与向量索引已在 Windows 本地一键初始化成功！")
        else:
            c.load_collection(collection_name=coll_name)
    except Exception as e:
        print(f"❌ [记忆集合初始化失败]: {e}")

def insert_memory(text: str):
    try:
        c = get_client()
        coll_name = config.MEMORY_COLLECTION_NAME
        vector = models.embeddings.embed_query(text)

        # ── 语义去重 (Semantic Upsert)：插入前先检索，相似度 > 0.92 则覆盖旧记忆 ──
        if c.has_collection(coll_name):
            res = c.search(
                collection_name=coll_name,
                data=[vector],
                limit=1,
                output_fields=["id", "text"],
                search_params={"metric_type": "IP"}
            )
            if res and res[0]:
                top_hit = res[0][0]
                score = top_hit["distance"]
                hit_id = top_hit["id"]
                if score > 0.92:
                    old_text = top_hit["entity"]["text"]
                    print(f"\U0001f504 \033[93m[记忆去重]\033[0m: 发现高度相似记忆 (相似度: {score:.4f})")
                    print(f"  旧记忆: {old_text}")
                    print(f"  新记忆: {text} -> 执行覆盖写入。")
                    c.delete(collection_name=coll_name, pks=[hit_id])
        # ─────────────────────────────────────────────────────────────────────────

        data = [{"vector": vector, "text": text}]
        c.insert(collection_name=coll_name, data=data)
        # 强制刷新以保证立即检索可见
        c.flush(coll_name)
    except Exception as e:
        print(f"❌ [插入记忆失败]: {e}")

def search_memory(query: str, top_k=3) -> List[str]:
    try:
        c = get_client()
        coll_name = config.MEMORY_COLLECTION_NAME
        if not c.has_collection(coll_name):
            return []
        
        # 直接检索，依赖异常处理而非不稳定的 row_count
        vector = models.embeddings.embed_query(query)
        res = c.search(
            collection_name=coll_name,
            data=[vector],
            limit=top_k,
            output_fields=["text"],
            search_params={"metric_type": "IP"}
        )
        memories = []
        for hits in res:
            for hit in hits:
                memories.append(hit["entity"]["text"])
        return memories
    except Exception as e:
        print(f"❌ [搜索记忆失败]: {e}")
        return []

def process_incremental_knowledge():
    """
    处理 pending 目录下的增量知识
    """
    os.makedirs(PENDING_DIR, exist_ok=True)
    os.makedirs(SOURCE_DIR, exist_ok=True)
    
    # 检查 PENDING_DIR 是否有文件
    has_files = False
    for root, _, files in os.walk(PENDING_DIR):
        if files:
            has_files = True
            break
            
    if not has_files and not getattr(config, "REFRESH_BM25", False):
        return

    c = get_client()
    coll_name = config.KNOWLEDGE_COLLECTION_NAME

    if getattr(config, "REFRESH_BM25", False):
        print("--- [重构模式] 检测到 REFRESH_BM25=True，开始全量拟合 BM25 并更新稀疏向量 ---")
        # 1. 扫描 source + pending 的所有文本进行 fit 并在本地 save
        all_docs = []
        if os.path.exists(SOURCE_DIR):
            all_docs.extend(data_loader.load_and_split_markdown(data_dir=SOURCE_DIR))
        if os.path.exists(PENDING_DIR):
            all_docs.extend(data_loader.load_and_split_markdown(data_dir=PENDING_DIR))
        
        all_texts = [doc.page_content for doc in all_docs]
        if all_texts:
            print(f"正在进行 BM25 词频统计全量拟合 (共 {len(all_texts)} 个片段)...")
            models.bm25_ef.fit(all_texts)
            os.makedirs("storage", exist_ok=True)
            models.bm25_ef.save("storage/bm25_model.json")
        
        # 2. 从 Milvus 捞出旧数据（只捞 id 和 text，规避内存问题）
        is_empty = True
        try:
            res = c.query(collection_name=coll_name, filter="id >= 0", limit=1)
            if res:
                is_empty = False
        except:
            pass
            
        if not is_empty:
            print("正在从 Milvus 获取历史数据...")
            old_records = c.query(collection_name=coll_name, filter="id >= 0", output_fields=["id", "text"])
            if old_records:
                old_texts = [r["text"] for r in old_records]
                # 3. 用新模型重新编码旧文本
                new_sparse_vectors = models.bm25_ef.encode_documents(old_texts)
                
                # 4. 组装并调用 upsert 接口进行局部覆盖
                upsert_data = []
                for i, r in enumerate(old_records):
                    row_sp = new_sparse_vectors[i].tocsr()
                    sp_dict = {int(k): float(v) for k, v in zip(row_sp.indices, row_sp.data)}
                    upsert_data.append({
                        "id": r["id"],
                        "sparse_vector": sp_dict
                    })
                
                batch_size = 500
                for i in range(0, len(upsert_data), batch_size):
                    c.upsert(collection_name=coll_name, data=upsert_data[i:i+batch_size])
                print(f"✅ 已成功局部更新 {len(upsert_data)} 条历史数据的稀疏向量字段！")
                print("BM25模型已全量重新拟合，旧数据的稀疏索引已局部更新完毕。请在 config.py 中将 REFRESH_BM25 改回 False，避免下次重复运行。")

    if not has_files:
        return
        
    print(f"--- [增量入库] 检测到 {PENDING_DIR} 中有新文件，开始处理增量知识 ---")
    docs = data_loader.load_and_split_markdown(data_dir=PENDING_DIR, is_incremental=True)
    
    if docs:
        print(f"提取到 {len(docs)} 个片段，开始双路向量化并入库...")
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        vectors = models.embeddings.embed_documents(texts)
        
        bm25_path = "storage/bm25_model.json"
        if os.path.exists(bm25_path):
            models.bm25_ef.load(bm25_path)
        else:
            print("⚠️ 警告：未找到 bm25_model.json！增量入库需要预先拟合的 BM25 模型。")
            return
            
        sparse_vectors = models.bm25_ef.encode_documents(texts)
        
        data = []
        for i, txt in enumerate(texts):
            row_sp = sparse_vectors[i].tocsr()
            sp_dict = {int(k): float(v) for k, v in zip(row_sp.indices, row_sp.data)}
            
            row = {
                "vector": vectors[i],
                "sparse_vector": sp_dict,
                "text": txt
            }
            row.update(metadatas[i])
            data.append(row)
        
        res = c.insert(collection_name=coll_name, data=data)
        c.load_collection(collection_name=coll_name)
        print(f"✅ 成功插入 {res.get('insert_count', len(data))} 条增量数据。")
        
        print(f"正在将已处理的文件从 {PENDING_DIR} 移动到 {SOURCE_DIR} ...")
        for root, _, files in os.walk(PENDING_DIR):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, PENDING_DIR)
                dst_path = os.path.join(SOURCE_DIR, rel_path)
                
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.move(src_path, dst_path)
        print("✅ 增量文件归档完成。")

def execute_hybrid_search(query: str, client, collection_name, bm25_model=None, output_fields=None, limit=10, embeddings_impl=None):
    if bm25_model is None:
        bm25_model = models.bm25_ef
    if embeddings_impl is None:
        embeddings_impl = models.embeddings
    if output_fields is None:
        output_fields = ["text", "source_path", "chunk_id", "parent_id", "content_type", "image_path"]
        
    from pymilvus import AnnSearchRequest, RRFRanker
    
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
    scores = models.reranker.predict(pairs)
    
    scored_hits = list(zip(scores, hits))
    scored_hits.sort(key=lambda x: x[0], reverse=True)
    
    return scored_hits[:top_k]

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
            import shutil
            coll_dir = os.path.join(config.MILVUS_DB_PATH, "collections", self.collection_name)
            if os.path.exists(coll_dir):
                logger.info(f"--- [刷新与清理] 正在手动物理清理 Windows 本地集合目录 '{self.collection_name}' ---")
                shutil.rmtree(coll_dir, ignore_errors=True)

        if not self.client.has_collection(self.collection_name):
            logger.info(f"--- [初始化] 正在 Windows 本地一键创建混合检索媒体集合 '{self.collection_name}' ---")
            
            from pymilvus import DataType
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
        os.makedirs(os.path.dirname("storage/av_bm25_model.json"), exist_ok=True)
        models.av_bm25_ef.save("storage/av_bm25_model.json")
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


def run_global_database_init():
    """一键运行系统所有数据库及知识库的流式初始化，供所有入口（控制台/微信/QQ）统一调用"""
    print("🚀 [Database] 开始执行全局多模态知识库联合初始化...")
    init_knowledge_base("data/source")
    process_incremental_knowledge()
    init_av_graph_rag()
    init_memory_collection()
    print("✅ [Database] 全局数据环境初始化就绪。")