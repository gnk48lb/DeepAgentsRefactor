"""本地多模态知识库初始化与增量入库 (所在层: L2)"""

import os
import shutil
from pymilvus import DataType
from app import config
from app.infra import models
from app.infra.milvus_client import get_client
from app.knowledge import loader as data_loader


def init_knowledge_base(data_dir=None):
    """
    初始化知识库：
    1. 检查集合是否存在，若开启 REFRESH_COLLECTION 则用物理清理规避 Windows 文件锁 Bug。
    2. 使用新版自适应 Schema 一键混合检索建表并建索引，100% 避开 Windows [WinError 183] 锁机制。
    """
    if data_dir is None:
        data_dir = config.SOURCE_DIR
    try:
        coll_name = config.KNOWLEDGE_COLLECTION_NAME
        
        # 1. 物理安全清理逻辑 (Windows 平台下 drop_collection 会遇到 WinError 183 文件锁 Bug 崩溃，用物理删除文件夹代替)
        if getattr(config, "REFRESH_COLLECTION", False):
            coll_dir = os.path.join(config.MILVUS_DB_PATH, "collections", coll_name)
            if os.path.exists(coll_dir):
                print(f"--- [刷新模式] 正在手动物理清理 Windows 本地集合目录 '{coll_name}' ---")
                shutil.rmtree(coll_dir, ignore_errors=True)
        
        c = get_client()
        
        # 2. 一键创建集合与双路索引 (自适应 Schema 与动态字段，100% 绕过 Windows 的文件锁冲突)
        if not c.has_collection(coll_name):
            print(f"--- [初始化] 正在 Windows 本地一键创建混合检索集合 '{coll_name}' ---")
            
            # 声明 Schema 启用 auto_id 和 dynamic_fields
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
        except Exception:
            pass
        
        if not is_empty:
            # 必须加载本地保存稀疏向量编码的 BM25 模型，否则无法对新 Query 进行编码
            bm25_path = config.BM25_MODEL_PATH
            if os.path.exists(bm25_path):
                models.bm25_ef.load(bm25_path)
                print(f"✅ 知识库 '{coll_name}' 已有数据且 BM25 词频模型就绪，跳过文档加载与向量化。")
                return
            else:
                print("⚠️ 警告：找到知识库数据，但丢失了 bm25_model.json！自动物理重建...")
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
            os.makedirs(config.STORAGE_DIR, exist_ok=True)
            models.bm25_ef.save(config.BM25_MODEL_PATH)
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


def process_incremental_knowledge():
    """
    处理 pending 目录下的增量知识
    """
    os.makedirs(config.PENDING_DIR, exist_ok=True)
    os.makedirs(config.SOURCE_DIR, exist_ok=True)
    
    # 检查 PENDING_DIR 是否有文件
    has_files = False
    for root, _, files in os.walk(config.PENDING_DIR):
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
        if os.path.exists(config.SOURCE_DIR):
            all_docs.extend(data_loader.load_and_split_markdown(data_dir=config.SOURCE_DIR))
        if os.path.exists(config.PENDING_DIR):
            all_docs.extend(data_loader.load_and_split_markdown(data_dir=config.PENDING_DIR))
        
        all_texts = [doc.page_content for doc in all_docs]
        if all_texts:
            print(f"正在进行 BM25 词频统计全量拟合 (共 {len(all_texts)} 个片段)...")
            models.bm25_ef.fit(all_texts)
            os.makedirs(config.STORAGE_DIR, exist_ok=True)
            models.bm25_ef.save(config.BM25_MODEL_PATH)
        
        # 2. 从 Milvus 捞出旧数据（只捞 id 和 text，规避内存问题）
        is_empty = True
        try:
            res = c.query(collection_name=coll_name, filter="id >= 0", limit=1)
            if res:
                is_empty = False
        except Exception:
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
        
    print(f"--- [增量入库] 检测到 {config.PENDING_DIR} 中有新文件，开始处理增量知识 ---")
    docs = data_loader.load_and_split_markdown(data_dir=config.PENDING_DIR, is_incremental=True)
    
    if docs:
        print(f"提取到 {len(docs)} 个片段，开始双路向量化并入库...")
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        vectors = models.embeddings.embed_documents(texts)
        
        bm25_path = config.BM25_MODEL_PATH
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
        
        print(f"正在将已处理的文件从 {config.PENDING_DIR} 移动到 {config.SOURCE_DIR} ...")
        for root, _, files in os.walk(config.PENDING_DIR):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, config.PENDING_DIR)
                dst_path = os.path.join(config.SOURCE_DIR, rel_path)
                
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.move(src_path, dst_path)
        print("✅ 增量文件归档完成。")
