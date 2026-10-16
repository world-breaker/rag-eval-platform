"""快速验证脚本：无需 GPU，测试核心流程"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, logger
from src.chunking import DocumentChunker
from src.embedding import EmbeddingEngine
from src.retrieval import RetrievalEngine
import numpy as np


def test_chunking():
    """测试三种切分策略"""
    config = load_config()
    text = "人工智能是计算机科学的分支。\n\n它致力于创造智能机器。\n\n深度学习是机器学习的子集。"

    for strategy in ["fixed_size", "recursive", "semantic"]:
        config["chunking"]["strategy"] = strategy
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(text)
        assert len(chunks) > 0, f"{strategy} 切分失败"
        logger.info(f"✅ {strategy}: {len(chunks)} chunks")

    logger.info("✅ chunking 测试全部通过")


def test_embedding():
    """测试 embedding + 缓存"""
    config = load_config()
    embedder = EmbeddingEngine(config)

    chunks = [{"text": "人工智能是计算机科学的一个分支"}, {"text": "深度学习使用多层神经网络"}]
    emb = embedder.encode(chunks)

    assert emb.shape == (2, 512), f"期望 (2,512)，实际 {emb.shape}"
    logger.info(f"✅ embedding 测试通过，shape={emb.shape}")


def test_retrieval():
    """测试检索"""
    config = load_config()
    retriever = RetrievalEngine(config)

    # 模拟：3 个文档，query 跟 doc2 最相关
    doc_emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # 跟 query 正交
            [0.0, 1.0, 0.0, 0.0],  # 跟 query 完全匹配
            [0.7, 0.3, 0.0, 0.0],  # 部分匹配
        ]
    )
    query_emb = np.array([0.0, 1.0, 0.0, 0.0])

    chunks = [
        {"text": "doc1", "chunk_index": 0},
        {"text": "doc2", "chunk_index": 1},
        {"text": "doc3", "chunk_index": 2},
    ]

    results = retriever.search(query_emb, doc_emb, chunks, top_k=3)
    assert len(results) >= 1, "检索结果为空"
    assert results[0]["chunk_index"] == 1, f"期望 doc2 排第一，实际 {results[0]['chunk_index']}"
    logger.info(
        f"✅ retrieval 测试通过，top-1={results[0]['chunk_index']}, score={results[0]['score']}"
    )


def test_evaluation():
    """测试评测模块（BGE嵌入后检索评测）"""
    from src.evaluation import EvaluationEngine

    config = load_config()
    evaluator = EvaluationEngine(config)

    # 测试检索评测
    retrieved = [{"chunk_index": 0}, {"chunk_index": 2}, {"chunk_index": 5}]
    relevant_ids = [0, 3]
    ret_metrics = evaluator.evaluate_retrieval(retrieved, relevant_ids)
    logger.info(f"检索评测: {ret_metrics}")
    assert "recall@3" in ret_metrics

    # 测试 BLEU
    bleu = evaluator.compute_bleu("深度学习是机器学习的子集", "深度学习是机器学习的一部分")
    assert 0 <= bleu <= 1
    logger.info(f"BLEU: {bleu:.4f}")

    # 测试 ROUGE-L
    rouge = evaluator.compute_rouge_l("深度学习是机器学习的子集", "深度学习是机器学习的一部分")
    assert 0 <= rouge <= 1
    logger.info(f"ROUGE-L: {rouge:.4f}")

    logger.info("✅ evaluation 测试全部通过")


if __name__ == "__main__":
    logger.info("=" * 40)
    logger.info("RAG 评测平台 - 快速验证")
    logger.info("=" * 40)

    test_chunking()
    test_embedding()
    test_retrieval()
    test_evaluation()

    logger.info("=" * 40)
    logger.info("🎉 所有测试通过！")
