"""混合检索模块：BM25（关键词）+ 向量（语义）双路召回 + RRF 融合 + 可选重排。

为什么需要混合检索？
- 纯向量检索擅长「语义相似」，但对精确关键词（如 "NF4"、"LoRA"、"√d_k"）和稀有术语不敏感，
  容易在「近义干扰文档」上翻车——这正是本项目原 5 篇文档评测集 recall@3 全 1.0 的天花板根源。
- BM25 擅长精确关键词匹配，正好补向量的短板。
- RRF（Reciprocal Rank Fusion）把两路的「排序」融合：不依赖分数量纲
  （向量余弦 ∈ [-1,1]，BM25 分数可任意大，量纲不同不能直接相加），鲁棒性最好。
- 可选的 Cross-Encoder 重排：用更强的交互式模型对融合后的 top-N 做二次精排，进一步提升精度。

设计取舍：
- BM25 与分词均零额外依赖（不引入 jieba），用「英文全词 + 中文单字 + 中文二元组」覆盖关键词重叠，
  足以在评测集上演示混合检索相对纯向量的可测量增益。
- 重排模型默认关闭（enabled=false），保证默认零额外下载、离线可跑、结果可复现；
  打开 rerank.enabled 即可启用（默认 cross-encoder/mmarco-mMiniLMv2-L12-H384-v1，470MB、多语言、已实测；中文专用可换 BAAI/bge-reranker-v2-m3），加载失败自动降级回 RRF。

三种排名口径（`rank_mode`）共用同一套候选召回与结果组装，只切换「最终排序依据」：
- `vector`：纯向量余弦（等价于 retrieval.mode="vector"，但**不套相似度阈值**，便于对照实验看全量排名）；
- `bm25`：纯 BM25 关键词排序（只返回 BM25 分 > 0 的文档）；
- `hybrid`：RRF 融合向量与 BM25 排名（可选叠加 Cross-Encoder 重排）。
结果里同时保留两路各自的原始名次（`vector_rank` / `bm25_rank`）与 RRF 分（`fusion_score`），
这样「这条结果是谁捞上来的、融合后涨了还是跌了」在 UI 和实验报告里都可解释，而不是只给一个黑箱分数。
"""

import re
import numpy as np
from typing import List, Dict, Optional
from src.utils import logger, timer

# 支持的三路排名口径
RANK_MODES = ("vector", "bm25", "hybrid")

# CJK 基本汉字区（U+4E00–U+9FFF）
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """中英混合分词，零依赖。

    - 连续 ASCII 字母/数字/符号（含 + - _ / . 与 √）作为一个 token，覆盖 "LoRA"、"NF4"、"RAG" 等术语。
    - 中文按「单字 + 二元组」切分：单字保证基础重叠，二元组提升多字术语（如「量化」「注意力」）的命中。
    - 其余标点/空白丢弃。
    """
    tokens: List[str] = []
    buf: List[str] = []

    def flush():
        if buf:
            tokens.append("".join(buf).lower())
            buf.clear()

    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in "+-_/."):
            buf.append(ch)
            continue
        flush()
        if _CJK_RE.match(ch):
            tokens.append(ch)
    flush()

    # CJK 二元组
    cjk_chars = [t for t in tokens if len(t) == 1 and _CJK_RE.match(t)]
    for i in range(len(cjk_chars) - 1):
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])

    return tokens


class BM25Retriever:
    """Okapi BM25 检索器（纯 NumPy，零额外依赖）。

    在 build_index 后对 chunk 文本拟合并持久化；search 返回全量排名（含分数为 0 的文档视为未命中），
    供 RRF 融合使用。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.doc_lens: np.ndarray = np.array([], dtype=float)
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.corpus_size: int = 0

    def fit(self, chunks: List[Dict]) -> "BM25Retriever":
        self.docs = [tokenize(c.get("text", "")) for c in chunks]
        self.corpus_size = len(self.docs)
        self.doc_lens = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(self.doc_lens.mean()) if self.corpus_size else 0.0

        df: Dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1

        n = self.corpus_size
        # Okapi idf（加 1 防止负无穷/零文档时 NaN）
        self.idf = {
            t: float(np.log((n - freq + 0.5) / (freq + 0.5) + 1.0))
            for t, freq in df.items()
        }
        return self

    def search(self, query_text: str, top_k: Optional[int] = None) -> List[Dict]:
        """返回按 BM25 分数降序的 [{chunk_index, bm25_score}, ...]（全量排名）。"""
        q_tokens = tokenize(query_text)
        if not q_tokens or self.corpus_size == 0:
            return []

        scores = np.zeros(self.corpus_size, dtype=float)
        for t in q_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, doc in enumerate(self.docs):
                f = doc.count(t)
                if f == 0:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_lens[i] / (self.avgdl or 1.0)
                )
                scores[i] += idf * (f * (self.k1 + 1)) / denom

        ranking = [
            {"chunk_index": i, "bm25_score": round(float(s), 4)}
            for i, s in enumerate(scores)
            if s > 0
        ]
        ranking.sort(key=lambda x: x["bm25_score"], reverse=True)
        if top_k is not None:
            ranking = ranking[:top_k]
        return ranking


def rrf_fusion(rank_maps: List[Dict[int, int]], k: int = 60) -> Dict[int, float]:
    """Reciprocal Rank Fusion。

    rank_maps: 每元素是一路检索的 {chunk_index: rank（从 1 开始）}。
    某文档未出现在某路结果中 → 该路 rank=+∞，贡献 1/(k+∞)=0。
    返回 {chunk_index: fused_score}，由调用方按分数取 top_k。
    """
    fused: Dict[int, float] = {}
    for rank_map in rank_maps:
        for idx, rank in rank_map.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    """混合检索：向量（语义）+ BM25（关键词）双路召回，RRF 融合，可选 Cross-Encoder 重排。"""

    def __init__(self, config: dict, vector_retriever=None):
        ret_cfg = config.get("retrieval", {})
        self.rrf_k = ret_cfg.get("rrf_k", 60)
        self.vector_retriever = vector_retriever

        hybrid_cfg = ret_cfg.get("hybrid", {})
        bm25_cfg = hybrid_cfg.get("bm25", {})
        self.bm25 = BM25Retriever(
            k1=bm25_cfg.get("k1", 1.5), b=bm25_cfg.get("b", 0.75)
        )

        rerank_cfg = hybrid_cfg.get("rerank", {})
        self.rerank_enabled = bool(rerank_cfg.get("enabled", False))
        # 注：`rerank.type` 只有一个实现（Cross-Encoder），所以它不是运行时开关——
        # 写别的值会在 load_config() 时 fail-fast（见 src/utils.py::validate_config）。
        self.rerank_model_name = rerank_cfg.get("model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.rerank_top_n = int(rerank_cfg.get("top_n", 20))
        # 重排默认在 CPU 上跑：评测/演示规模（top-N≤20）下 GPU 与 CPU 差距不大，
        # 而 CPU 不占用生成模型的显存，也避免小显存机器上的 OOM。可用 rerank.device 覆盖。
        self.rerank_device = rerank_cfg.get("device", "cpu")
        self.reranker = None
        self.rerank_load_error: Optional[str] = None
        if self.rerank_enabled:
            self._load_reranker()

    def _load_reranker(self):
        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                f"加载重排模型：{self.rerank_model_name}（device={self.rerank_device}）"
            )
            self.reranker = CrossEncoder(self.rerank_model_name, device=self.rerank_device)
            self.rerank_load_error = None
        except Exception as e:  # 模型未下载/无网络 → 降级，不影响检索
            logger.warning(
                f"重排模型加载失败，退回 RRF 融合结果（检索仍可用）：{e}"
            )
            self.reranker = None
            self.rerank_load_error = f"{type(e).__name__}: {e}"

    def set_rerank(self, enabled: bool, model: Optional[str] = None) -> bool:
        """运行时开关重排（供 Web UI 切换）。返回「重排是否真的可用」。"""
        self.rerank_enabled = bool(enabled)
        if model:
            self.rerank_model_name = model
        if not self.rerank_enabled:
            return False
        if self.reranker is None:
            self._load_reranker()
        return self.reranker is not None

    def fit(self, chunks: List[Dict]) -> "HybridRetriever":
        """在 build_index 后调用：用同样的 chunk 拟合 BM25 语料。"""
        self.bm25.fit(chunks)
        return self

    @timer
    def search(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        chunks: List[Dict],
        top_k: int = None,
        query_text: str = "",
        rank_mode: str = "hybrid",
    ) -> List[Dict]:
        """混合检索主入口。

        参数与 RetrievalEngine.search 对齐（额外多一个 query_text 供 BM25 使用）。
        rank_mode 决定最终排序依据：vector / bm25 / hybrid（默认 hybrid，与历史行为一致）。
        结果中同时给出两路原始名次与 RRF 分，便于解释「这条是谁捞上来的」。
        """
        if rank_mode not in RANK_MODES:
            raise ValueError(f"rank_mode 必须是 {RANK_MODES} 之一，实际 {rank_mode!r}")

        k = top_k if top_k is not None else (
            self.vector_retriever.top_k if self.vector_retriever else 5
        )

        # 1) 向量路：直接算全量余弦（点积，已归一化），保留完整排名供融合
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        vec_scores = (query_embedding @ doc_embeddings.T).flatten()
        vec_order = np.argsort(vec_scores)[::-1]
        vec_rank = {int(vec_order[i]): i + 1 for i in range(len(vec_order))}

        # 2) BM25 路（全量排名，供融合与对照）
        bm25_ranking = self.bm25.search(query_text, top_k=len(chunks))
        bm25_rank = {r["chunk_index"]: i + 1 for i, r in enumerate(bm25_ranking)}
        bm25_scores = {r["chunk_index"]: r["bm25_score"] for r in bm25_ranking}

        # 3) 按排名口径决定候选顺序
        if rank_mode == "vector":
            cand_idx = [int(i) for i in vec_order]
            fused: Dict[int, float] = {}
        elif rank_mode == "bm25":
            # 纯 BM25：只保留真正命中的文档（BM25 分 > 0），不掺向量
            cand_idx = [r["chunk_index"] for r in bm25_ranking]
            fused = {}
        else:  # hybrid：RRF 融合
            fused = rrf_fusion([vec_rank, bm25_rank], k=self.rrf_k)
            cand_idx = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

        # 4) 可选重排：仅对 top-N 用 Cross-Encoder 交互式精排（只对 hybrid 生效）
        rerank_active = False
        rerank_scores: Dict[int, float] = {}
        if rank_mode == "hybrid" and self.rerank_enabled and self.reranker is not None:
            top_cands = cand_idx[: self.rerank_top_n]
            if top_cands:
                pairs = [(query_text, chunks[i]["text"]) for i in top_cands]
                try:
                    scores = self.reranker.predict(pairs)
                    scores = np.asarray(scores, dtype=float).reshape(-1)
                    rerank_scores = {
                        int(idx): float(s) for idx, s in zip(top_cands, scores)
                    }
                    top_cands = [
                        top_cands[i] for i in np.argsort(scores)[::-1]
                    ]
                    cand_idx = top_cands + cand_idx[self.rerank_top_n :]
                    rerank_active = True
                except Exception as e:
                    logger.warning(f"重排失败，退回 RRF 融合结果：{e}")

        # 5) 组装结果：保留各路来源，供 UI/报告解释融合是否真的起了作用
        results = []
        for rank, idx in enumerate(cand_idx[:k], 1):
            results.append(
                {
                    **chunks[idx],
                    "score": round(float(vec_scores[idx]), 4),
                    "bm25_score": round(float(bm25_scores.get(idx, 0.0)), 4),
                    "fusion_score": round(float(fused.get(idx, 0.0)), 4),
                    "vector_rank": vec_rank.get(idx),
                    "bm25_rank": bm25_rank.get(idx),
                    "rerank_score": (
                        round(rerank_scores[idx], 4) if idx in rerank_scores else None
                    ),
                    "retrieved_rank": rank,
                }
            )

        detail = f"（RRF k={self.rrf_k}" if rank_mode == "hybrid" else "（"
        if rerank_active:
            detail += f" + 重排[{self.rerank_model_name}]"
        detail += "）"
        logger.info(
            f"检索完成[{rank_mode}]：query → {len(results)}/{k} 条结果{detail}"
        )
        return results
