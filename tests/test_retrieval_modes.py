"""检索模式（vector / bm25 / hybrid）与融合元数据的离线单测。

全部使用手写小矩阵 / 桩对象，不加载任何模型、不联网、结果确定。
"""

import sys
import types
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_retrieval import HybridRetriever, RANK_MODES
from src.retrieval import RetrievalEngine
from src.evaluation import EvaluationEngine
from src.query_rewriting import QueryRewriter
from src.utils import load_config

CFG = load_config(str(Path(__file__).parent.parent / "config.yaml"))
# 只用 doc0 独有的术语（NF4 / normalfloat / 正态分布）：doc1 含「量化」但与这些词零重叠，
# 这样 BM25 路是干净的单命中，RRF 融合前后可断言、不产生并列分。
KEYWORD_QUERY = "NF4 normalfloat 正态分布"


def _chunks():
    return [
        {"text": "量化 INT4 NF4 normalfloat 正态分布权重", "chunk_index": 0, "source": "quant.txt"},
        {"text": "模型压缩 量化 INT8 剪枝 蒸馏", "chunk_index": 1, "source": "compress.txt"},
        {"text": "足球 篮球 体育新闻", "chunk_index": 2, "source": "sports.txt"},
    ]


def _fixture():
    """构造「向量会翻车、BM25 能救」的确定性场景。"""
    chunks = _chunks()
    doc_embeddings = np.array(
        [[0.6, 0.8], [0.7, 0.1], [0.0, 1.0]], dtype=float
    )
    query_emb = np.array([1.0, 0.0])
    return chunks, doc_embeddings, query_emb


def test_rank_modes_constant():
    assert RANK_MODES == ("vector", "bm25", "hybrid")
    print("✅ rank_modes_constant")


def test_rank_mode_vector_orders_by_cosine():
    chunks, docs, q = _fixture()
    hr = HybridRetriever(CFG, RetrievalEngine(CFG)).fit(chunks)
    res = hr.search(q, docs, chunks, top_k=3, query_text=KEYWORD_QUERY, rank_mode="vector")
    assert [r["chunk_index"] for r in res] == [1, 0, 2], "vector 模式应按余弦降序"
    assert res[0]["vector_rank"] == 1
    assert res[0]["fusion_score"] == 0.0, "非 hybrid 模式没有 RRF 分"
    print("✅ rank_mode_vector_orders_by_cosine")


def test_rank_mode_bm25_only_uses_keyword_match():
    chunks, docs, q = _fixture()
    hr = HybridRetriever(CFG, RetrievalEngine(CFG)).fit(chunks)
    res = hr.search(q, docs, chunks, top_k=3, query_text=KEYWORD_QUERY, rank_mode="bm25")
    # NF4 只出现在 doc0 → 纯 BM25 把 doc0 排第一
    assert res[0]["chunk_index"] == 0, f"实际 {[r['chunk_index'] for r in res]}"
    # 纯 BM25 只返回真正命中的文档（doc2 与查询零词重叠，不应出现）
    assert 2 not in [r["chunk_index"] for r in res], "无关键词命中的文档不应进入 BM25 结果"
    assert all(r["bm25_score"] > 0 for r in res)
    assert res[0]["bm25_rank"] == 1
    print("✅ rank_mode_bm25_only_uses_keyword_match")


def test_hybrid_metadata_exposes_both_sources():
    chunks, docs, q = _fixture()
    hr = HybridRetriever(CFG, RetrievalEngine(CFG)).fit(chunks)
    res = hr.search(q, docs, chunks, top_k=3, query_text=KEYWORD_QUERY, rank_mode="hybrid")
    assert res[0]["chunk_index"] == 0, "RRF 应把目标文档拉回第一"
    for r in res:
        for field in ("score", "bm25_score", "fusion_score", "vector_rank", "bm25_rank", "retrieved_rank"):
            assert field in r, f"结果缺少字段 {field}"
    # 目标文档在向量路上只排第 2，是 BM25 把它顶上去的 → 两路名次必须都能看到
    assert res[0]["vector_rank"] == 2 and res[0]["bm25_rank"] == 1
    assert res[0]["fusion_score"] > 0
    assert res[0]["rerank_score"] is None
    print("✅ hybrid_metadata_exposes_both_sources")


def test_invalid_rank_mode_raises():
    chunks, docs, q = _fixture()
    hr = HybridRetriever(CFG, RetrievalEngine(CFG)).fit(chunks)
    try:
        hr.search(q, docs, chunks, query_text=KEYWORD_QUERY, rank_mode="dense")
    except ValueError as e:
        assert "rank_mode" in str(e)
    else:
        raise AssertionError("非法 rank_mode 应抛 ValueError")
    print("✅ invalid_rank_mode_raises")


class _FakeReranker:
    """桩重排模型：只偏好正文含 INT8 的候选（doc1），用于验证重排真的覆盖了 RRF 排序。"""

    def predict(self, pairs):
        return np.array([10.0 if "INT8" in text else 0.0 for _, text in pairs])


def test_rerank_overrides_rrf_order_and_reports_scores():
    chunks, docs, q = _fixture()
    hr = HybridRetriever(CFG, RetrievalEngine(CFG)).fit(chunks)
    # 先确认没有重排时 RRF 的第一名是 doc0
    plain = hr.search(q, docs, chunks, top_k=3, query_text=KEYWORD_QUERY, rank_mode="hybrid")
    assert plain[0]["chunk_index"] == 0

    hr.rerank_enabled = True
    hr.reranker = _FakeReranker()
    res = hr.search(q, docs, chunks, top_k=3, query_text=KEYWORD_QUERY, rank_mode="hybrid")
    assert res[0]["chunk_index"] == 1, "重排应覆盖 RRF 的排序（这正是加一层的意义）"
    assert res[0]["rerank_score"] == 10.0, "重排分应写回结果，便于解释排序变化"
    assert all(r["rerank_score"] is not None for r in res), "Top-N 候选都应有重排分"
    print("✅ rerank_overrides_rrf_order_and_reports_scores")


def test_rerank_disabled_leaves_scores_none():
    chunks, docs, q = _fixture()
    hr = HybridRetriever(CFG, RetrievalEngine(CFG)).fit(chunks)
    res = hr.search(q, docs, chunks, top_k=3, query_text=KEYWORD_QUERY, rank_mode="hybrid")
    assert all(r["rerank_score"] is None for r in res)
    print("✅ rerank_disabled_leaves_scores_none")


# ---------- pipeline 侧：模式切换 + 改写融合（用桩 embedding，不加载模型） ----------


def _fake_pipeline(chunks, doc_embeddings, embed_fn, mode="vector"):
    """绕开模型加载，手工拼一个 pipeline 对象（只放被测代码需要的属性）。"""
    from src.pipeline import RAGPipeline

    pipe = object.__new__(RAGPipeline)
    pipe.config = {
        "retrieval": {"top_k": 5, "similarity_threshold": 0.3, "rrf_k": 60},
        "evaluation": {"retrieval_metrics": ["recall@3"]},
    }
    pipe.retriever = RetrievalEngine(pipe.config)
    pipe.embedder = types.SimpleNamespace(encode=embed_fn)
    pipe.evaluator = EvaluationEngine(pipe.config)
    pipe.chunks = chunks
    pipe.doc_embeddings = np.array(doc_embeddings, dtype=float)
    pipe.retrieval_mode = mode
    pipe.hybrid = None
    pipe.query_rewriter = None
    pipe._rewriter_initialized = False
    return pipe


def test_default_mode_matches_legacy_vector_search():
    """零回归：未开改写时，_retrieve 的结果必须与直接调 RetrievalEngine 完全一致。"""
    docs = np.array([[1.0, 0.0], [0.0, 1.0]])
    chunks = [{"text": "d0", "chunk_index": 0}, {"text": "d1", "chunk_index": 1}]
    q = np.array([[0.6, 0.8]])

    def encode(items):
        return np.tile(q, (len(items), 1))

    pipe = _fake_pipeline(chunks, docs, encode)
    got = pipe._retrieve("Q")
    expected = pipe.retriever.search(q, pipe.doc_embeddings, chunks, None)
    assert [r["chunk_index"] for r in got] == [r["chunk_index"] for r in expected]
    assert [r["score"] for r in got] == [r["score"] for r in expected]
    assert "fusion_score" not in got[0], "默认路径必须保持历史结果结构不变（零回归）"
    print("✅ default_mode_matches_legacy_vector_search")


def test_set_retrieval_mode_builds_hybrid_lazily_and_validates():
    docs = np.array([[1.0, 0.0], [0.0, 1.0]])
    chunks = [{"text": "NF4 量化", "chunk_index": 0}, {"text": "足球 篮球", "chunk_index": 1}]
    pipe = _fake_pipeline(chunks, docs, lambda items: np.tile([[1.0, 0.0]], (len(items), 1)))

    assert pipe.hybrid is None, "默认 vector 模式不应为 BM25 付任何代价"
    pipe.set_retrieval_mode("bm25")
    assert pipe.hybrid is not None, "切到 bm25/hybrid 时才懒加载"
    assert pipe.hybrid.bm25.corpus_size == 2, "懒加载时应立即用现有 chunks 拟合 BM25"

    try:
        pipe.set_retrieval_mode("dense")
    except ValueError:
        pass
    else:
        raise AssertionError("非法模式应抛 ValueError")
    print("✅ set_retrieval_mode_builds_hybrid_lazily_and_validates")


def test_multi_query_rewrites_fuse_and_flip_top1():
    """core 断言：只有靠改写查询才能被召回的目标文档，经 RRF 融合后应升到 top1。

    构造：原查询向量更接近 doc1（纯向量会把 doc1 排第一，但 doc1 不是答案）；
    两条改写查询都把 doc0 排第一 → RRF 融合后 doc0 反超。
    """
    chunks = [
        {"text": "target", "chunk_index": 0, "source": "t.txt"},
        {"text": "distractor", "chunk_index": 1, "source": "d.txt"},
    ]
    doc_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    vecs = {
        "原问题": [0.6, 0.8],      # 更接近 doc1 → 纯向量 top1 = doc1（错）
        "改写A": [1.0, 0.0],       # 更接近 doc0
        "改写B": [1.0, 0.0],
    }

    def encode(items):
        return np.array([vecs[c["text"]] for c in items], dtype=float)

    pipe = _fake_pipeline(chunks, doc_embeddings, encode)

    # 未开改写：doc1 排第一
    baseline = pipe._retrieve("原问题", top_k=2)
    assert baseline[0]["chunk_index"] == 1, "基线（纯向量）应被干扰项骗到"

    # 开改写（stub LLM 返回两条指向 doc0 的变体）
    pipe.query_rewriter = QueryRewriter(
        {"retrieval": {"query_rewrite": {"enabled": True, "strategy": "multi_query", "num_queries": 2}}},
        llm_fn=lambda p: "改写A\n改写B\n",
    )
    pipe._rewriter_initialized = True

    got = pipe._retrieve("原问题", top_k=2)
    assert got[0]["chunk_index"] == 0, "改写 + RRF 融合应把正确答案拉回第一"
    assert got[0]["rewrite_hits"] == 3, "目标文档应被三路（原查询 + 2 条改写）全部召回"
    assert got[0]["fusion_score"] > got[1]["fusion_score"]
    print("✅ multi_query_rewrites_fuse_and_flip_top1")


if __name__ == "__main__":
    test_rank_modes_constant()
    test_rank_mode_vector_orders_by_cosine()
    test_rank_mode_bm25_only_uses_keyword_match()
    test_hybrid_metadata_exposes_both_sources()
    test_invalid_rank_mode_raises()
    test_rerank_overrides_rrf_order_and_reports_scores()
    test_rerank_disabled_leaves_scores_none()
    test_default_mode_matches_legacy_vector_search()
    test_set_retrieval_mode_builds_hybrid_lazily_and_validates()
    test_multi_query_rewrites_fuse_and_flip_top1()
    print("✅ 全部检索模式单测通过")
