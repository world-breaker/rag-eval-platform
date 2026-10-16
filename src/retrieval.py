"""检索模块：向量搜索 + 相似度过滤"""

import numpy as np
from typing import List, Dict, Tuple
from src.utils import logger, timer


class RetrievalEngine:
    """检索引擎：向量余弦搜索 + 相似度阈值过滤（低于阈值的候选直接丢弃）。"""

    def __init__(self, config: dict):
        ret_cfg = config["retrieval"]
        self.top_k = ret_cfg["top_k"]
        self.similarity_threshold = ret_cfg["similarity_threshold"]

    @timer
    def search(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        chunks: List[Dict],
        top_k: int = None,
    ) -> List[Dict]:
        """
        向量检索 + 过滤。
        返回：[{text, score, chunk_index, metadata...}, ...]
        """
        k = top_k if top_k is not None else self.top_k

        # 余弦相似度（embedding 已归一化，点积 = 余弦）
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores = (query_embedding @ doc_embeddings.T).flatten()

        # 取 Top-K，筛掉低分
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < self.similarity_threshold:
                continue  # 低分文档不返回
            results.append(
                {
                    **chunks[idx],
                    "score": round(score, 4),
                    "retrieved_rank": len(results) + 1,
                }
            )

        logger.info(
            f"检索完成：query → {len(results)}/{k} 条结果 "
            f"（阈值={self.similarity_threshold}）"
        )
        return results
