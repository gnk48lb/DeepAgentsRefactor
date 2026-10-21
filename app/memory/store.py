"""长期记忆集合初始化、去重写入与语义检索 (所在层: L2)"""

import os
import shutil
from typing import List, Optional
from pymilvus import DataType
from app import config
from app.infra import models
from app.infra.milvus_client import get_client


def init_memory_collection():
    try:
        coll_name = config.MEMORY_COLLECTION_NAME
        
        # 1. 物理安全清理逻辑 (Windows 平台下 drop_collection 会遇到 WinError 183 文件锁 Bug 崩溃，用物理删除文件夹代替)
        if getattr(config, "CLEAR_MEMORY_ON_STARTUP", False) or getattr(config, "REFRESH_COLLECTION", False):
            coll_dir = os.path.join(config.MILVUS_DB_PATH, "collections", coll_name)
            if os.path.exists(coll_dir):
                print(f"--- [刷新与清理] 正在手动物理清理 Windows 本地记忆集合目录 '{coll_name}' ---")
                shutil.rmtree(coll_dir, ignore_errors=True)

        c = get_client()

        # 2. 一键创建集合与稠密向量索引 (自适应 Schema 与动态字段，100% 绕过 Windows 的文件锁冲突)
        if not c.has_collection(coll_name):
            print(f"--- [初始化] 正在 Windows 本地一键创建长期记忆集合 '{coll_name}' ---")
            
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
        if not c.has_collection(coll_name):
            init_memory_collection()
        else:
            try:
                c.load_collection(collection_name=coll_name)
            except Exception:
                pass

        vector = models.embeddings.embed_query(text)

        # ── 语义去重 (Semantic Upsert)：插入前先检索，相似度 > 0.92 则覆盖旧记忆 ──
        if c.has_collection(coll_name):
            try:
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
                    if score > config.MEMORY_DEDUP_SCORE:
                        old_text = top_hit["entity"]["text"]
                        print(f"🔄 \033[93m[记忆去重]\033[0m: 发现高度相似记忆 (相似度: {score:.4f})")
                        print(f"  旧记忆: {old_text}")
                        print(f"  新记忆: {text} -> 执行覆盖写入。")
                        c.delete(collection_name=coll_name, pks=[hit_id])
            except Exception as se:
                print(f"⚠️ [记忆查重跳过]: {se}")
        # ─────────────────────────────────────────────────────────────────────────

        data = [{"vector": vector, "text": text}]
        c.insert(collection_name=coll_name, data=data)
        # Windows Milvus Lite 下 flush 可能因 tmp 文件替换锁报 WinError 183，保护性调用
        try:
            c.flush(coll_name)
        except Exception:
            pass
    except Exception as e:
        print(f"❌ [插入记忆失败]: {e}")


def search_memory(query: str, top_k: int = 3, min_score: Optional[float] = None) -> List[str]:
    """检索长期记忆，只返回相似度 >= 阈值的条目（按相似度从高到低）。"""
    if min_score is None:
        min_score = config.MEMORY_MIN_SCORE
    try:
        c = get_client()
        coll_name = config.MEMORY_COLLECTION_NAME
        if not c.has_collection(coll_name):
            init_memory_collection()
        else:
            try:
                c.load_collection(collection_name=coll_name)
            except Exception:
                pass

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
                score = hit["distance"]
                text = hit["entity"]["text"]
                if score >= min_score:
                    memories.append(text)
                    print(f"🧠 \033[96m[记忆检索]\033[0m 采用 (score={score:.3f} ≥ {min_score}): {text}")
                else:
                    print(f"🧠 \033[90m[记忆检索]\033[0m 丢弃 (score={score:.3f} < {min_score}): {text}")
        return memories
    except Exception as e:
        print(f"❌ [搜索记忆失败]: {e}")
        return []
