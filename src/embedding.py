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
    """向量化引擎：BGE 模型封装 + 离线缓存（文档不变不重算）。

    缓存键 = **模型名 + 文本内容** 的 MD5（2026-10-20 修）：
    原先键里只有文本，换了 `embedding.model_name` 之后缓存**照样命中**，
    于是拿**旧模型**的向量当新模型结果跑检索实验——指标「看起来正常」，
    实则是旧模型的。现在模型名进键，换模型即自然失效重算。
    """

    def __init__(self, config: dict):
        emb_cfg = config["embedding"]
        self.model_name = emb_cfg["model_name"]
        self.device = emb_cfg["device"]
        self.batch_size = emb_cfg["batch_size"]
        self.cache_dir = Path(emb_cfg["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"加载 Embedding 模型：{self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def _cache_key(self, texts: List[str]) -> str:
        """缓存键：模型名 + 文本内容。

        模型名必须进键：同一个文本在 A 模型与 B 模型下的向量完全不同，
        键里没有模型名就会互相命中（拿旧模型的向量冒充新模型）。
        """
        joined = "|||".join(texts)
        blob = f"{self.model_name}\n{joined}"
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    @timer
    def encode(self, chunks: List[Dict]) -> np.ndarray:
        """
        将 chunk 列表转为向量矩阵。
        - 优先读缓存（模型与文档都没变 = 不重算）
        - 缓存未命中则计算并写入（原子落盘：先写临时文件再 rename）
        """
        texts = [c["text"] for c in chunks]
        cache_key = self._cache_key(texts)
        cache_path = self.cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            cached = self._load_cached(cache_path, len(texts))
            if cached is not None:
                logger.info(f"Embedding 缓存命中 → {cache_path.name}")
                return cached
            logger.warning(
                f"Embedding 缓存形状/内容不可用（{cache_path.name}），按未命中重算"
            )

        logger.info(f"计算 Embedding：{len(texts)} 条文本")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # 归一化后，点积 = 余弦相似度
        )
        self._save_cached(cache_path, embeddings)
        logger.info(f"Embedding 已缓存 → {cache_path.name}")
        return embeddings

    def _load_cached(self, cache_path: Path, expect_rows: int) -> np.ndarray | None:
        """读缓存并校验条数；不可用（半截文件/条数不符）返回 None（调用方重算）。"""
        try:
            cached = np.load(cache_path)
        except Exception as e:  # 损坏的 .npy 不能让整轮实验挂掉
            logger.warning(f"Embedding 缓存读取失败（{type(e).__name__}: {e}），按未命中重算")
            return None
        if getattr(cached, "ndim", 0) != 2 or cached.shape[0] != expect_rows:
            logger.warning(
                f"Embedding 缓存条数不符（缓存 {getattr(cached, 'shape', None)} vs 期望 {expect_rows}），按未命中重算"
            )
            return None
        return cached

    def _save_cached(self, cache_path: Path, embeddings: np.ndarray) -> None:
        """原子落盘：先写同目录临时文件，再 os.replace 覆盖（避免写一半的坏文件被当成命中）。"""
        tmp_path = cache_path.with_name(cache_path.name + ".tmp.npy")
        try:
            np.save(tmp_path, embeddings)
            os.replace(tmp_path, cache_path)
        except Exception as e:  # 缓存写失败不该让实验挂掉
            logger.warning(f"Embedding 缓存写入失败（{type(e).__name__}: {e}），本次不落缓存")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
