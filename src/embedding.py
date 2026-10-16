"""Embedding 模块：文档向量化 + 离线缓存"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.utils import logger, timer


class EmbeddingEngine:
    """向量化引擎：BGE 模型封装 + 基于文本 MD5 的离线缓存（文档不变不重算）。"""

    def __init__(self, config: dict):
        emb_cfg = config["embedding"]
        self.model_name = emb_cfg["model_name"]
        self.device = emb_cfg["device"]
        self.batch_size = emb_cfg["batch_size"]
        self.cache_dir = Path(emb_cfg["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"加载 Embedding 模型：{self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    @staticmethod
    def _hash_texts(texts: List[str]) -> str:
        """根据文本内容计算缓存键"""
        joined = "|||".join(texts)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()

    @timer
    def encode(self, chunks: List[Dict]) -> np.ndarray:
        """
        将 chunk 列表转为向量矩阵。
        - 优先读缓存（文档没变 = 不重算）
        - 缓存未命中则计算并写入
        """
        texts = [c["text"] for c in chunks]
        cache_key = self._hash_texts(texts)
        cache_path = self.cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            logger.info(f"Embedding 缓存命中 → {cache_path.name}")
            return np.load(cache_path)

        logger.info(f"计算 Embedding：{len(texts)} 条文本")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # 归一化后，点积 = 余弦相似度
        )
        np.save(cache_path, embeddings)
        logger.info(f"Embedding 已缓存 → {cache_path.name}")
        return embeddings
