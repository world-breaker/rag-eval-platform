"""混合检索离线单测：BM25 / RRF / HybridRetriever，无需 GPU 或模型下载。"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_retrieval import tokenize, BM25Retriever, rrf_fusion, HybridRetriever


def test_tokenize():
    toks = tokenize("LoRA 量化 NF4 是什么")
    # 英文全词保留
    assert "lora" in toks
    assert "nf4" in toks
    # 中文单字 + 二元组
    assert "量" in toks
    assert "量化" in toks
    # 纯标点被丢弃
    assert "？" not in toks
    logger_ok("tokenize")


def test_bm25_ranks_term_doc_first():
    chunks = [
        {"text": "NF4 量化 正态分布 权重", "chunk_index": 0},          # 含唯一关键词 NF4
        {"text": "模型压缩 量化 剪枝 蒸馏", "chunk_index": 1},          # 只有 量化，无 NF4
        {"text": "足球 篮球 体育新闻", "chunk_index": 2},              # 噪声
    ]
    bm25 = BM25Retriever().fit(chunks)
    ranking = bm25.search("NF4量化格式为什么对正态分布友好", top_k=3)
    assert ranking[0]["chunk_index"] == 0, f"BM25 应把含 NF4 的文档排第一，实际 {ranking}"
    logger_ok("bm25_ranks_term_doc_first")


def test_rrf_fusion():
    # 两路排名：A 在第一路排 1，第二路排 2；B 在第一路排 2，第二路排 1
    fused = rrf_fusion([{0: 1, 1: 2}, {0: 2, 1: 1}], k=60)
    # A = 1/61 + 1/62；B = 1/62 + 1/61 → 相等
    assert abs(fused[0] - fused[1]) < 1e-9
    # 未出现在任何一路的文档贡献 0
    fused2 = rrf_fusion([{0: 1}, {0: 1}], k=60)
    assert 2 not in fused2 or fused2[2] == 0.0
    # 一路把某文档排第一会显著加分
    fused3 = rrf_fusion([{0: 1, 1: 3}, {0: 1}], k=60)
    assert fused3[0] > fused3[1], "两路都排第一应严格高于只排第三"
    logger_ok("rrf_fusion")


def test_hybrid_pulls_keyword_doc_up():
    """构造「纯向量会翻车、混合检索能救回」的确定性场景。

    - 向量余弦：distractor(doc1) 比 target(doc0) 更相似 → 纯向量把 doc1 排第一（错误）
    - BM25：query 含唯一关键词 NF4，仅 target(doc0) 拥有 → 把 doc0 拉回第一
    - 结论：hybrid 的 top1 应为 doc0（target）
    """
    chunks = [
        {"text": "量化 INT4 NF4 normalfloat 正态分布权重", "chunk_index": 0},  # target
        {"text": "模型压缩 量化 INT8 剪枝 蒸馏", "chunk_index": 1},            # distractor
        {"text": "量化 格式 模型压缩", "chunk_index": 2},                     # 半相关噪声（诱使向量偏向，但 BM25 弱于 target）
    ]
    doc_embeddings = np.array(
        [
            [0.6, 0.8],  # doc0 余弦=0.6
            [0.7, 0.1],  # doc1 余弦=0.7（向量更相似，但非目标）
            [0.0, 1.0],  # doc2 余弦=0.0
        ],
        dtype=float,
    )
    query_emb = np.array([1.0, 0.0])  # 归一化后点积=余弦

    # 纯向量排名（不依赖 RetrievalEngine，直接复算）
    vec_scores = (query_emb @ doc_embeddings.T).flatten()
    vec_order = np.argsort(vec_scores)[::-1]
    assert list(vec_order) == [1, 0, 2], f"纯向量会把 distractor 排第一，实际 {list(vec_order)}"

    # 混合检索（vector_retriever=None，top_k 走默认 5）
    hr = HybridRetriever({"retrieval": {}}, vector_retriever=None)
    hr.fit(chunks)
    res = hr.search(query_emb, doc_embeddings, chunks, top_k=3, query_text="NF4量化格式为什么对正态分布友好")

    assert res[0]["chunk_index"] == 0, (
        f"混合检索应把含 NF4 的 target 拉回第一，实际 top1={res[0]['chunk_index']}"
    )
    # 结果结构完整
    for r in res:
        assert "score" in r and "fusion_score" in r and "retrieved_rank" in r
    logger_ok("hybrid_pulls_keyword_doc_up")


def logger_ok(name):
    print(f"✅ {name}")


if __name__ == "__main__":
    test_tokenize()
    test_bm25_ranks_term_doc_first()
    test_rrf_fusion()
    test_hybrid_pulls_keyword_doc_up()
    print("✅ 全部混合检索单测通过")
