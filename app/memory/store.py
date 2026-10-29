"""长期记忆集合初始化、版本化维护与语义检索 (所在层: L2)"""

import os
import time
import threading
from typing import List, Optional, Dict, Any
from pymilvus import DataType
from app import config
from app.infra import models
from app.infra.milvus_client import get_client
from app.memory.guards import normalize_text, check_memory_text

_write_lock = threading.RLock()
_last_id: int = 0


def _generate_id() -> int:
    """生成单调递增的客户端 INT64 主键 (微秒级时间戳 + last_id 防重)。必须在写锁内调用。"""
    global _last_id
    now_micros = time.time_ns() // 1000
    next_id = max(now_micros, _last_id + 1)
    _last_id = next_id
    return next_id


def collection_name() -> str:
    """返回当前版本的长期记忆集合名称。"""
    return f"{config.MEMORY_COLLECTION_NAME}_v{config.MEMORY_SCHEMA_VERSION}"


def init_memory_collection() -> None:
    """
    初始化长期记忆集合 (Schema v2)：
    1. v2 集合不存在则创建，关闭 dynamic_field，显式声明全部标量字段。
    2. 若配置 CLEAR_MEMORY_ON_STARTUP=True，则物理清空 v2 (delete filter='id >= 0')，跳过迁移并写标记文件。
    3. 若旧集合存在且标记文件不存在，且非 CLEAR_MEMORY_ON_STARTUP，则执行幂等、非破坏性平滑迁移：
       复用向量、旧集合不删除、写入标记文件。
    4. REFRESH_COLLECTION 不再影响长期记忆库。
    """
    v2_name = collection_name()
    flag_path = os.path.join(config.STORAGE_DIR, "memory_v2_migrated.flag")

    with _write_lock:
        try:
            c = get_client()

            # 1. 确保 v2 集合存在
            if not c.has_collection(v2_name):
                print(f"--- [初始化] 正在创建长期记忆集合 '{v2_name}' ---")
                schema = c.create_schema(auto_id=False, enable_dynamic_field=False)
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
                schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048)
                schema.add_field(field_name="kind", datatype=DataType.VARCHAR, max_length=16)
                schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=16)
                schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=32)
                schema.add_field(field_name="evidence", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="retired_evidence", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="created_at", datatype=DataType.INT64)
                schema.add_field(field_name="updated_at", datatype=DataType.INT64)
                schema.add_field(field_name="supersedes", datatype=DataType.INT64)

                index_params = c.prepare_index_params()
                index_params.add_index(field_name="vector", index_type="FLAT", metric_type="IP")

                c.create_collection(
                    collection_name=v2_name,
                    schema=schema,
                    index_params=index_params,
                )
                print(f"✅ 长期记忆集合 '{v2_name}' 初始化成功！")
            else:
                try:
                    c.load_collection(collection_name=v2_name)
                except Exception:
                    pass

            # 2. CLEAR_MEMORY_ON_STARTUP 逻辑
            if getattr(config, "CLEAR_MEMORY_ON_STARTUP", False):
                print(f"--- [清理模式] 检测到 CLEAR_MEMORY_ON_STARTUP=True，清空记忆集合 '{v2_name}' ---")
                try:
                    c.delete(collection_name=v2_name, filter="id >= 0")
                    try:
                        c.flush(v2_name)
                    except Exception:
                        pass
                except Exception as ce:
                    print(f"⚠️ [清空记忆异常]: {ce}")
                os.makedirs(config.STORAGE_DIR, exist_ok=True)
                with open(flag_path, "w", encoding="utf-8") as f:
                    f.write(str(int(time.time())))
                return

            # 3. 幂等平滑迁移
            old_name = config.MEMORY_COLLECTION_NAME
            if c.has_collection(old_name) and not os.path.exists(flag_path):
                print(f"--- [记忆迁移] 检测到旧集合 '{old_name}' 且未迁移，开始迁移至 '{v2_name}' ---")
                # 先清除上次可能中断残留的 legacy 行
                try:
                    c.delete(collection_name=v2_name, filter='source == "legacy"')
                    try:
                        c.flush(v2_name)
                    except Exception:
                        pass
                except Exception as de:
                    print(f"⚠️ [记忆迁移] 清理残留 legacy 行跳过: {de}")

                try:
                    c.load_collection(collection_name=old_name)
                except Exception:
                    pass

                offset = 0
                batch_size = 200
                now = int(time.time())
                migrated_count = 0
                while True:
                    old_batch = c.query(
                        collection_name=old_name,
                        filter="id >= 0",
                        limit=batch_size,
                        offset=offset,
                        output_fields=["id", "text", "vector"],
                    )
                    if not old_batch:
                        break

                    v2_rows = []
                    for item in old_batch:
                        txt = str(item.get("text", "") or "")
                        vec = item.get("vector")
                        ok, reason = check_memory_text(txt)
                        status = "active" if ok else "deleted"
                        if not ok:
                            print(f"⚠️ [记忆迁移] 旧记忆 (id={item.get('id')}) 文本不合规 ({reason})，置为 deleted: {txt[:50]}")

                        safe_txt = txt[:2048]
                        v2_rows.append({
                            "id": _generate_id(),
                            "vector": vec,
                            "text": safe_txt,
                            "kind": "fact",
                            "status": status,
                            "source": "legacy",
                            "evidence": "",
                            "retired_evidence": "",
                            "created_at": now,
                            "updated_at": now,
                            "supersedes": -1,
                        })

                    if v2_rows:
                        c.insert(collection_name=v2_name, data=v2_rows)
                        migrated_count += len(v2_rows)
                    offset += len(old_batch)

                try:
                    c.flush(v2_name)
                except Exception:
                    pass

                os.makedirs(config.STORAGE_DIR, exist_ok=True)
                with open(flag_path, "w", encoding="utf-8") as f:
                    f.write(str(now))
                print(f"✅ [记忆迁移完成] 共迁移 {migrated_count} 条记录，已生成标记文件 '{flag_path}'")

        except Exception as e:
            print(f"❌ [记忆集合初始化失败]: {e}")


def _ensure_loaded(c, coll_name: str) -> None:
    """确保集合已加载。"""
    try:
        c.load_collection(collection_name=coll_name)
    except Exception:
        pass


def search_memory(query: str, top_k: int = 3, min_score: Optional[float] = None) -> List[str]:
    """检索长期记忆，只返回相似度 >= 阈值的有效事实（status=active 且 kind=fact），按相似度从高到低。"""
    if min_score is None:
        min_score = config.MEMORY_MIN_SCORE
    try:
        c = get_client()
        coll_name = collection_name()
        if not c.has_collection(coll_name):
            init_memory_collection()
        else:
            _ensure_loaded(c, coll_name)

        vector = models.embeddings.embed_query(query)
        res = c.search(
            collection_name=coll_name,
            data=[vector],
            filter='status == "active" and kind == "fact"',
            limit=top_k,
            output_fields=["text"],
            search_params={"metric_type": "IP"},
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


def recall_similar(text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    阶段2对账候选召回：按向量相似度召回 top-K 条 active+fact 候选记忆，不设分数下限。
    返回每条候选的 dict: {"id": hit_id, "text": hit_text, "score": hit_score}
    """
    if top_k is None:
        top_k = config.MEMORY_RECALL_TOP_K
    try:
        c = get_client()
        coll_name = collection_name()
        if not c.has_collection(coll_name):
            init_memory_collection()
        else:
            _ensure_loaded(c, coll_name)

        vector = models.embeddings.embed_query(text)
        res = c.search(
            collection_name=coll_name,
            data=[vector],
            filter='status == "active" and kind == "fact"',
            limit=top_k,
            output_fields=["id", "text"],
            search_params={"metric_type": "IP"},
        )
        candidates = []
        for hits in res:
            for hit in hits:
                candidates.append({
                    "id": hit["id"],
                    "text": hit["entity"]["text"],
                    "score": hit["distance"],
                })
        return candidates
    except Exception as e:
        print(f"❌ [候选记忆召回失败]: {e}")
        return []


def add_memory(
    text: str,
    *,
    evidence: str,
    vector: Optional[List[float]] = None,
    source: str = "user_msg",
) -> Optional[int]:
    """
    新增一条原子事实记忆（status=active, supersedes=-1）。
    D5 e: ADD 前做'规范化文本完全相同'的幂等检查（与任一 active 记忆相同 → 跳过）。
    """
    if vector is None:
        vector = models.embeddings.embed_query(text)

    coll_name = collection_name()
    with _write_lock:
        try:
            c = get_client()
            if not c.has_collection(coll_name):
                init_memory_collection()
            else:
                _ensure_loaded(c, coll_name)

            # 幂等检查
            norm_new = normalize_text(text)
            active_existing = c.query(
                collection_name=coll_name,
                filter='status == "active" and kind == "fact"',
                output_fields=["id", "text"],
            )
            for item in active_existing:
                if normalize_text(item.get("text", "")) == norm_new:
                    print(f"🧠 [记忆写入] ADD 丢弃 (已存在规范化完全相同的事实, id={item.get('id')}): '{item.get('text')}'")
                    return None

            new_id = _generate_id()
            now = int(time.time())
            data = [{
                "id": new_id,
                "vector": vector,
                "text": text,
                "kind": "fact",
                "status": "active",
                "source": source,
                "evidence": evidence,
                "retired_evidence": "",
                "created_at": now,
                "updated_at": now,
                "supersedes": -1,
            }]
            c.insert(collection_name=coll_name, data=data)
            try:
                c.flush(coll_name)
            except Exception:
                pass
            print(f"🧠 \033[92m[记忆写入] ADD 成功\033[0m (id={new_id}): {text}")
            return new_id
        except Exception as e:
            print(f"❌ [添加记忆失败]: {e}")
            return None


def supersede_memory(
    old_id: int,
    new_text: str,
    *,
    evidence: str,
    vector: Optional[List[float]] = None,
) -> Optional[int]:
    """
    更新事实记忆（原子化版本演进）：
    1. 锁内确认 old 记忆仍处于 active 状态。
    2. 先插入新行 (status=active, supersedes=old_id)。
    3. 再把旧行置 superseded (partial_update upsert 改 status 和 updated_at)。
    4. 第二步失败则删除刚插入的新行作为补偿并上抛。
    """
    if vector is None:
        vector = models.embeddings.embed_query(new_text)

    coll_name = collection_name()
    with _write_lock:
        c = get_client()
        if not c.has_collection(coll_name):
            init_memory_collection()
        else:
            _ensure_loaded(c, coll_name)

        old_rows = c.query(
            collection_name=coll_name,
            filter=f"id == {old_id}",
            output_fields=["id", "status", "text"],
        )
        if not old_rows or old_rows[0].get("status") != "active":
            print(f"🧠 [记忆写入] UPDATE 丢弃: 目标记忆 (id={old_id}) 不存在或状态非 active")
            return None

        # 幂等检查：新文本与已有任一 active 记忆相同则丢弃
        norm_new = normalize_text(new_text)
        active_existing = c.query(
            collection_name=coll_name,
            filter='status == "active" and kind == "fact"',
            output_fields=["id", "text"],
        )
        for item in active_existing:
            if normalize_text(item.get("text", "")) == norm_new:
                print(f"🧠 [记忆写入] UPDATE 丢弃 (已存在规范化相同记忆, id={item.get('id')}): '{item.get('text')}'")
                return None

        new_id = _generate_id()
        now = int(time.time())
        new_row = {
            "id": new_id,
            "vector": vector,
            "text": new_text,
            "kind": "fact",
            "status": "active",
            "source": "user_msg",
            "evidence": evidence,
            "retired_evidence": "",
            "created_at": now,
            "updated_at": now,
            "supersedes": old_id,
        }
        c.insert(collection_name=coll_name, data=[new_row])

        try:
            c.upsert(
                collection_name=coll_name,
                data=[{"id": old_id, "status": "superseded", "updated_at": now}],
                partial_update=True,
            )
        except Exception as ue:
            # 补偿：删除刚插入的新行
            try:
                c.delete(collection_name=coll_name, pks=[new_id])
                try:
                    c.flush(coll_name)
                except Exception:
                    pass
            except Exception:
                pass
            print(f"❌ [记忆更新失败 - 已回滚新行]: {ue}")
            raise ue

        try:
            c.flush(coll_name)
        except Exception:
            pass
        print(f"🧠 \033[93m[记忆写入] UPDATE 成功\033[0m (old_id={old_id} -> new_id={new_id}): {new_text}")
        return new_id


def retire_memory(mem_id: int, *, evidence: str = "") -> bool:
    """撤回/删除记忆（软删除）：active -> deleted，记录 retired_evidence。"""
    coll_name = collection_name()
    with _write_lock:
        try:
            c = get_client()
            if not c.has_collection(coll_name):
                init_memory_collection()
            else:
                _ensure_loaded(c, coll_name)

            old_rows = c.query(
                collection_name=coll_name,
                filter=f"id == {mem_id}",
                output_fields=["id", "status"],
            )
            if not old_rows or old_rows[0].get("status") != "active":
                print(f"🧠 [记忆写入] DELETE 丢弃: 目标记忆 (id={mem_id}) 不存在或状态非 active")
                return False

            now = int(time.time())
            c.upsert(
                collection_name=coll_name,
                data=[{
                    "id": mem_id,
                    "status": "deleted",
                    "updated_at": now,
                    "retired_evidence": evidence,
                }],
                partial_update=True,
            )
            try:
                c.flush(coll_name)
            except Exception:
                pass
            print(f"🧠 \033[91m[记忆写入] DELETE 成功\033[0m (id={mem_id}, evidence='{evidence}')")
            return True
        except Exception as e:
            print(f"❌ [删除记忆失败]: {e}")
            return False


def restore_memory(mem_id: int) -> bool:
    """恢复已软删除的记忆：deleted -> active。"""
    coll_name = collection_name()
    with _write_lock:
        try:
            c = get_client()
            if not c.has_collection(coll_name):
                init_memory_collection()
            else:
                _ensure_loaded(c, coll_name)

            old_rows = c.query(
                collection_name=coll_name,
                filter=f"id == {mem_id}",
                output_fields=["id", "status"],
            )
            if not old_rows or old_rows[0].get("status") != "deleted":
                print(f"🧠 [记忆写入] RESTORE 丢弃: 目标记忆 (id={mem_id}) 不存在或状态非 deleted")
                return False

            now = int(time.time())
            c.upsert(
                collection_name=coll_name,
                data=[{"id": mem_id, "status": "active", "updated_at": now}],
                partial_update=True,
            )
            try:
                c.flush(coll_name)
            except Exception:
                pass
            print(f"🧠 \033[92m[记忆写入] RESTORE 成功\033[0m (id={mem_id})")
            return True
        except Exception as e:
            print(f"❌ [恢复记忆失败]: {e}")
            return False


def list_memories(include_inactive: bool = False) -> List[Dict[str, Any]]:
    """查询事实记忆列表。默认只返回 active，include_inactive=True 则包含 superseded 与 deleted。"""
    try:
        c = get_client()
        coll_name = collection_name()
        if not c.has_collection(coll_name):
            init_memory_collection()
        else:
            _ensure_loaded(c, coll_name)

        filter_expr = 'kind == "fact"' if include_inactive else 'status == "active" and kind == "fact"'
        results = c.query(
            collection_name=coll_name,
            filter=filter_expr,
            output_fields=[
                "id", "text", "kind", "status", "source",
                "evidence", "retired_evidence", "created_at", "updated_at", "supersedes",
            ],
        )
        return results or []
    except Exception as e:
        print(f"❌ [列出记忆失败]: {e}")
        return []


def purge_memories(ids: List[int]) -> int:
    """物理清除指定 ID 记忆（仅供运维脚本显式命令调用）。"""
    if not ids:
        return 0
    coll_name = collection_name()
    with _write_lock:
        try:
            c = get_client()
            if not c.has_collection(coll_name):
                return 0
            int_ids = [int(i) for i in ids]
            c.delete(collection_name=coll_name, pks=int_ids)
            try:
                c.flush(coll_name)
            except Exception:
                pass
            print(f"⚠️ [记忆物理清除] 已物理删除 {len(int_ids)} 条记录: {int_ids}")
            return len(int_ids)
        except Exception as e:
            print(f"❌ [物理清除记忆失败]: {e}")
            return 0
