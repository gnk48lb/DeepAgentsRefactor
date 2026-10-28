"""
tests/smoke_knowledge.py

知识库 / 长期记忆数据层离线冒烟测试:Milvus Lite + 确定性假 embedding,不需要任何 API Key / 网络 / Docker。
  A 建库:切分 → BM25 拟合并保存 → 入库
  B pending 增量入库并归档到 source
  C 混合检索(稀疏+稠密),校验 source_path 仍是"相对 data 目录"的相对路径(兼容已入库的旧数据)
  D 长期记忆:写入 / 语义去重 / 阈值过滤检索
  E 重复初始化幂等
  F reranker 阈值过滤与不可用降级
所有数据写入临时目录:通过改写 config 属性重定向路径,同时验证"路径只来自 config 常量"。
"""
import hashlib
import os
import random
import shutil
import sys
import tempfile
import traceback

_TMP = tempfile.mkdtemp(prefix="smoke_knowledge_")
os.environ["MILVUS_DB_PATH"] = os.path.join(_TMP, "m.db")

from app import config  # noqa: E402

config.DATA_DIR = os.path.join(_TMP, "data")
config.STORAGE_DIR = os.path.join(_TMP, "storage")
config.SOURCE_DIR = os.path.join(config.DATA_DIR, "source")
config.PENDING_DIR = os.path.join(config.DATA_DIR, "pending")
config.PIPELINE_IMAGES_DIR = os.path.join(config.DATA_DIR, "pipeline", "images")
config.PIPELINE_SUMMARIES_DIR = os.path.join(config.DATA_DIR, "pipeline", "summaries")
config.BM25_MODEL_PATH = os.path.join(config.STORAGE_DIR, "bm25_model.json")
config.MILVUS_DB_PATH = os.environ["MILVUS_DB_PATH"]


def _vec(text: str):
    """文本 → 确定性 1024 维单位向量:相同文本相似度恒为 1,不同文本近似正交。"""
    r = random.Random(int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16))
    v = [r.random() - 0.5 for _ in range(1024)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [_vec(t) for t in texts]

    def embed_query(self, text):
        return _vec(text)


from app.infra import models  # noqa: E402
from app.infra.milvus_client import get_client  # noqa: E402
from app.knowledge import knowledge_base, search  # noqa: E402
from app.memory import store  # noqa: E402

models._cache["embeddings"] = _FakeEmbeddings()   # 绕过惰性工厂,不触发任何网络请求


def _count(collection: str) -> int:
    return len(get_client().query(collection_name=collection, filter="id >= 0", output_fields=["id"]))


def _write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_a_build():
    _write(os.path.join(config.SOURCE_DIR, "recipe.md"),
           "# 红烧肉\n## 食材\n五花肉 冰糖 生抽 老抽 八角\n## 做法\n先焯水再炒糖色，小火慢炖一个小时。\n")
    _write(os.path.join(config.SOURCE_DIR, "game.md"),
           "# 艾尔登法环\n## Boss\n女武神玛莲妮亚 弹反 水鸟乱舞 需要躲避\n")
    knowledge_base.init_knowledge_base()
    n = _count(config.KNOWLEDGE_COLLECTION_NAME)
    assert n >= 2, f"入库片段数异常: {n}"
    assert os.path.exists(config.BM25_MODEL_PATH), "BM25 模型应保存到 config.BM25_MODEL_PATH"
    return n


def test_b_incremental(n_before):
    _write(os.path.join(config.PENDING_DIR, "extra.md"), "# 增量文档\n这是一篇通过 pending 目录增量入库的文章，讲述量子纠缠。\n")
    knowledge_base.process_incremental_knowledge()
    assert _count(config.KNOWLEDGE_COLLECTION_NAME) > n_before, "增量文档未入库"
    assert not os.listdir(config.PENDING_DIR), "pending 目录应已清空"
    assert os.path.exists(os.path.join(config.SOURCE_DIR, "extra.md")), "增量文件应归档到 source"


def test_c_hybrid_search():
    hits = search.execute_hybrid_search("弹反 女武神", get_client(), config.KNOWLEDGE_COLLECTION_NAME)
    assert hits, "混合检索无结果"
    paths = [os.path.normpath(h["entity"]["source_path"]) for h in hits[:2]]
    assert os.path.normpath("source/game.md") in paths, f"期望 source/game.md 位于前 2 名,实际: {paths}"
    # source_path 必须是相对 data 目录的相对路径(而不是绝对路径),否则已入库数据与 rag 读取路径都会失效
    assert not os.path.isabs(hits[0]["entity"]["source_path"]), hits[0]["entity"]["source_path"]


def test_d_memory():
    store.init_memory_collection()
    store.insert_memory("用户家住在济南市文化东路42号")
    store.insert_memory("用户家住在济南市文化东路42号")          # 相同文本:应语义去重覆盖,而不是写两条
    assert _count(config.MEMORY_COLLECTION_NAME) == 1, "语义去重失败"
    store.insert_memory("用户对海鲜严重过敏")
    assert _count(config.MEMORY_COLLECTION_NAME) == 2
    got = store.search_memory("用户家住在济南市文化东路42号", top_k=3)
    assert got == ["用户家住在济南市文化东路42号"], f"阈值过滤后应只命中相同文本,实际: {got}"


def test_e_idempotent(n_now):
    knowledge_base.init_knowledge_base()
    assert _count(config.KNOWLEDGE_COLLECTION_NAME) == n_now, "重复初始化不应重复入库"


def test_f_reranker_threshold_and_fallback():
    from app.tools.knowledge_tools import rag, is_below_threshold
    orig_reranker = models._cache.get("reranker")
    orig_threshold = config.RAG_MIN_RERANK_SCORE
    try:
        config.RAG_MIN_RERANK_SCORE = 0.1

        # 1. 纯函数断言
        assert is_below_threshold([(0.05, {})], 0.1) is True
        assert is_below_threshold([(0.15, {})], 0.1) is False
        assert is_below_threshold([(None, {})], 0.1) is False
        assert is_below_threshold([], 0.1) is True

        # 2. 注入低于阈值的假 reranker
        class LowScoreReranker:
            def predict(self, pairs):
                return [0.05] * len(pairs)

        models._cache["reranker"] = LowScoreReranker()
        res_low = rag.invoke({"query": "女武神"})
        assert res_low == "No relevant information found in the knowledge base.", f"低于阈值应返回无相关提示，实际: {res_low}"

        # 3. 注入高于阈值的假 reranker
        class HighScoreReranker:
            def predict(self, pairs):
                return [0.85] * len(pairs)

        models._cache["reranker"] = HighScoreReranker()
        res_high = rag.invoke({"query": "女武神"})
        assert "女武神" in str(res_high) and res_high != "No relevant information found in the knowledge base.", f"高于阈值应返回上下文，实际: {res_high}"

        # 4. 注入抛出 RerankerUnavailable 的假 reranker
        class UnavailableReranker:
            def predict(self, pairs):
                raise models.RerankerUnavailable("Connection refused")

        models._cache["reranker"] = UnavailableReranker()
        res_unavail = rag.invoke({"query": "女武神"})
        assert "女武神" in str(res_unavail) and res_unavail != "No relevant information found in the knowledge base.", f"Reranker不可用时应退回输入顺序且不被误判为无相关，实际: {res_unavail}"

    finally:
        config.RAG_MIN_RERANK_SCORE = orig_threshold
        if orig_reranker is not None:
            models._cache["reranker"] = orig_reranker
        else:
            models._cache.pop("reranker", None)


def main():
    print("=" * 70)
    print("开始运行知识库 / 长期记忆数据层离线冒烟测试...")
    print("=" * 70)
    passed = failed = 0
    state = {}
    steps = [
        ("A 建库 + BM25 保存 + 入库", lambda: state.update(n=test_a_build())),
        ("B pending 增量入库并归档", lambda: (test_b_incremental(state["n"]), state.update(n=_count(config.KNOWLEDGE_COLLECTION_NAME)))),
        ("C 混合检索与 source_path 相对路径", test_c_hybrid_search),
        ("D 长期记忆 写入/去重/阈值检索", test_d_memory),
        ("E 重复初始化幂等", lambda: test_e_idempotent(state["n"])),
        ("F reranker 阈值过滤与不可用降级", test_f_reranker_threshold_and_fallback),
    ]
    try:
        for name, fn in steps:
            try:
                fn()
                print(f"  [PASS] {name}")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e!r}")
                traceback.print_exc()
                failed += 1
    finally:
        try:
            from app.infra import milvus_client
            if milvus_client.client is not None:
                milvus_client.client.close()
        except Exception:
            pass
        shutil.rmtree(_TMP, ignore_errors=True)
    print("=" * 70)
    print(f"测试结果汇总: PASS: {passed}/{len(steps)}, FAIL: {failed}/{len(steps)}")
    print("=" * 70)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
