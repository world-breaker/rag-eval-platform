"""RAG Pipeline：编排完整流程"""

import json
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional
from src.utils import load_config, logger, timer
from src.chunking import DocumentChunker
from src.embedding import EmbeddingEngine
from src.retrieval import RetrievalEngine
from src.generation import GenerationEngine
from src.evaluation import EvaluationEngine
from src.hybrid_retrieval import HybridRetriever, rrf_fusion, RANK_MODES
from src.query_rewriting import QueryRewriter


class RAGPipeline:
    """RAG 全链路编排器：模块化设计，各组件可独立替换与对比实验。"""

    def __init__(self, config_path: str = "config.yaml", load_generator: bool = True):
        self.config = load_config(config_path)
        logger.info("初始化 RAG Pipeline...")

        self.chunker = DocumentChunker(self.config)
        self.embedder = EmbeddingEngine(self.config)
        self.retriever = RetrievalEngine(self.config)
        # 检索模式：vector（默认，历史行为）/ bm25（纯关键词）/ hybrid（BM25+向量 RRF，可选重排）
        # 默认 "vector" 时不实例化 HybridRetriever，行为与旧版完全一致（零回归）。
        self.retrieval_mode = self.config["retrieval"].get("mode", "vector")
        if self.retrieval_mode not in RANK_MODES:
            raise ValueError(
                f"retrieval.mode 必须是 {RANK_MODES} 之一，实际 {self.retrieval_mode!r}"
            )
        self.hybrid = (
            HybridRetriever(self.config, self.retriever)
            if self.retrieval_mode != "vector"
            else None
        )
        # 查询改写（HyDE / multi-query）：默认关闭，开启后需要 LLM（延迟不然后悔）
        self.query_rewriter: Optional[QueryRewriter] = None
        self._rewriter_initialized = False
        # 生成模型体积大、加载慢：纯检索/评测场景（对比实验、指标评测）可延迟到首次问答时再加载
        self.generator = GenerationEngine(self.config) if load_generator else None
        self.evaluator = EvaluationEngine(self.config)

        self.doc_embeddings: Optional[np.ndarray] = None
        self.chunks: List[Dict] = []
        self.doc_ids: Dict[str, int] = {}  # source → 文档序号（供文档级检索评测使用）
        logger.info(f"Pipeline 初始化完成（retrieval.mode={self.retrieval_mode}）")

    def _ensure_generator(self) -> GenerationEngine:
        """按需加载生成模型（延迟初始化）"""
        if self.generator is None:
            self.generator = GenerationEngine(self.config)
        return self.generator

    # ========== 检索模式与开关 ==========

    def _ensure_hybrid(self) -> HybridRetriever:
        """按需构造混合检索器（BM25 需要先用当前 chunks 拟合语料）。

        延迟构造的原因：默认 vector 模式下不该为 BM25 付任何代价（零回归）；
        而 Web UI 允许运行时切到 bm25/hybrid，所以这里做成懒加载 + 幂等。
        """
        if self.hybrid is None:
            self.hybrid = HybridRetriever(self.config, self.retriever)
            if self.chunks:
                self.hybrid.fit(self.chunks)
        return self.hybrid

    def set_retrieval_mode(self, mode: str) -> str:
        """运行时切换检索模式（供 Web UI / 实验脚本使用）。"""
        if mode not in RANK_MODES:
            raise ValueError(f"检索模式必须是 {RANK_MODES} 之一，实际 {mode!r}")
        self.retrieval_mode = mode
        if mode != "vector":
            self._ensure_hybrid()
        return self.retrieval_mode

    def set_rerank(self, enabled: bool, model: Optional[str] = None) -> bool:
        """运行时开关 Cross-Encoder 重排（仅 hybrid 模式生效）。返回是否真的可用。"""
        return self._ensure_hybrid().set_rerank(enabled, model)

    def _ensure_rewriter(self) -> Optional[QueryRewriter]:
        """按需构造查询改写器；开关关闭时返回 None（零成本）。"""
        if self._rewriter_initialized:
            return self.query_rewriter
        self._rewriter_initialized = True

        cfg = (self.config.get("retrieval", {}) or {}).get("query_rewrite", {}) or {}
        if not bool(cfg.get("enabled", False)):
            self.query_rewriter = None
            return None

        # 改写用 LLM=生成模型（同一套权重，不额外占显存）
        llm = self._ensure_generator()
        self.query_rewriter = QueryRewriter(self.config, llm_fn=llm.complete)
        logger.info(
            f"查询改写已启用：strategy={self.query_rewriter.strategy}, "
            f"num_queries={self.query_rewriter.num_queries}"
        )
        return self.query_rewriter

    # ========== 建库 ==========

    @timer
    def build_index(self, documents: List[Dict]):
        """
        建向量库。
        documents: [{"text": "...", "source": "doc1.pdf", ...}, ...]
        """
        all_chunks = []
        doc_ids: Dict[str, int] = {}
        for doc in documents:
            doc = dict(doc)  # 浅拷贝：不修改调用方传入的数据
            text = doc.pop("text")
            metadata = doc  # 其余字段作为元数据
            source = doc.get("source", f"doc{len(doc_ids)}")
            doc_ids.setdefault(source, len(doc_ids))
            chunks = self.chunker.chunk(text, metadata)
            all_chunks.extend(chunks)

        # 分配全局唯一 chunk_index（跨文档连续），文档内序号保留为 doc_chunk_index。
        # 说明：每个文档各自从 0 编号会造成跨文档索引重复，检索结果无法唯一标识。
        for global_idx, chunk in enumerate(all_chunks):
            chunk["doc_chunk_index"] = chunk["chunk_index"]
            chunk["chunk_index"] = global_idx

        self.chunks = all_chunks
        self.doc_ids = doc_ids
        self.doc_embeddings = self.embedder.encode(all_chunks)
        # 混合检索：用同一批 chunk 拟合 BM25 语料
        if self.hybrid is not None:
            self.hybrid.fit(self.chunks)
        logger.info(
            f"索引构建完成：{len(all_chunks)} 个 chunk，"
            f"向量维度 {self.doc_embeddings.shape[1]}"
        )

    def load_documents_from_dir(self, directory: str) -> List[Dict]:
        """从目录加载文本文档"""
        docs = []
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"目录不存在：{directory}")
            return docs

        for filepath in dir_path.glob("*.txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"text": text, "source": filepath.name})

        # 也支持 JSON 格式
        for filepath in dir_path.glob("*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                docs.extend(data)
            elif isinstance(data, dict):
                docs.append(data)

        logger.info(f"从目录加载了 {len(docs)} 个文档")
        return docs

    # ========== 检索 ==========

    def _single_query_search(
        self, query_text: str, query_emb: np.ndarray, mode: str, top_k: int
    ) -> List[Dict]:
        """单条 query 的检索（不涉及改写）。"""
        if mode == "vector":
            # 纯向量走 RetrievalEngine：保留 similarity_threshold 过滤（历史行为不变）
            return self.retriever.search(
                query_emb, self.doc_embeddings, self.chunks, top_k
            )
        return self._ensure_hybrid().search(
            query_emb,
            self.doc_embeddings,
            self.chunks,
            top_k=top_k,
            query_text=query_text,
            rank_mode=mode,
        )

    def _retrieve(self, question: str, top_k: int = None, mode: str = None) -> List[Dict]:
        """统一检索入口：按 retrieval.mode 选择向量 / BM25 / 混合检索，可选叠加查询改写。

        返回值结构在各模式下保持一致（score / bm25_score / fusion_score / 各路 rank），
        供 ask / 对比实验 / 批量评测 / Web UI 复用。
        """
        if self.doc_embeddings is None or len(self.chunks) == 0:
            raise RuntimeError("请先调用 build_index() 构建向量库")

        mode = mode or self.retrieval_mode
        k = top_k if top_k is not None else self.retriever.top_k

        rewriter = self._ensure_rewriter()
        if rewriter is None or not rewriter.active:
            # 默认路径（零回归）：单查询检索，与历史版本逐比特一致
            query_emb = self.embedder.encode([{"text": question}])
            return self._single_query_search(question, query_emb, mode, k)

        return self._retrieve_with_rewrites(question, rewriter, mode, k)

    def _retrieve_with_rewrites(
        self, question: str, rewriter: QueryRewriter, mode: str, k: int
    ) -> List[Dict]:
        """改写路径：原查询 + 多条改写查询（+ HyDE 假想文档）分别召回 → RRF 融合。

        为什么用 RRF 融合多路改写：改写产出的查询没有「谁更可信」的先验，
        用排名融合比加权求和更稳（与混合检索同一套理由）；且任何一路漏掉的文档
        只要在其它路排得靠前，仍能被捞回来——这正是 multi-query 提升召回的原理。
        """
        rw = rewriter.rewrite(question)
        queries: List[str] = rw["queries"]
        hypothetical = rw.get("hypothetical_doc")

        # 每路都要全量排名，融合才有意义（top_k=len(chunks)）
        rank_maps: List[Dict[int, int]] = []
        meta: Dict[int, Dict] = {}

        def _collect(text: str, emb: np.ndarray):
            res = self._single_query_search(text, emb, mode, len(self.chunks))
            rank_maps.append(
                {r["chunk_index"]: i + 1 for i, r in enumerate(res)}
            )
            for r in res:  # 保留第一路的元数据（向量分/BM25 分等）
                meta.setdefault(r["chunk_index"], r)

        embs = self.embedder.encode([{"text": t} for t in queries])
        for text, emb in zip(queries, embs):
            _collect(text, emb.reshape(1, -1))

        if hypothetical:
            # HyDE：用假想文档的向量当作一条额外的「查询路」
            hyde_emb = self.embedder.encode([{"text": hypothetical}])
            _collect(hypothetical, hyde_emb.reshape(1, -1))

        rrf_k = self.hybrid.rrf_k if self.hybrid is not None else 60
        fused = rrf_fusion(rank_maps, k=rrf_k)
        cand_idx = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

        results = []
        for rank, idx in enumerate(cand_idx[:k], 1):
            row = dict(meta.get(idx, self.chunks[idx]))
            row["fusion_score"] = round(float(fused[idx]), 4)
            row["retrieved_rank"] = rank
            row["rewrite_hits"] = sum(1 for rm in rank_maps if idx in rm)
            results.append(row)

        logger.info(
            f"改写检索完成[{mode}]：{len(queries)} 条查询"
            + (f" + HyDE({len(hypothetical)} 字符)" if hypothetical else "")
            + f" → RRF 融合 {len(results)}/{k} 条"
        )
        return results

    # ========== 问答 ==========

    @timer
    def ask(
        self,
        question: str,
        top_k: int = None,
        return_context: bool = True,
        mode: str = None,
    ) -> Dict:
        """单次问答"""
        if self.doc_embeddings is None or len(self.chunks) == 0:
            raise RuntimeError("请先调用 build_index() 构建向量库")

        # 1. 检索（vector / bm25 / hybrid，可叠加查询改写）
        retrieved = self._retrieve(question, top_k, mode=mode)

        # 2. 生成
        result = self._ensure_generator().generate(question, retrieved)

        # 4. 评测（可选，需要标注数据）
        if return_context:
            result["context"] = "\n\n".join(
                [
                    f"[chunk {r['chunk_index']}, score={r['score']:.3f}] {r['text'][:200]}..."
                    for r in retrieved[:3]
                ]
            )

        return result

    # ========== 批量评测 ==========

    def batch_evaluate(self, test_data_path: str) -> List[Dict]:
        """
        批量评测：读取测试数据集，逐条问答+评测。
        测试数据格式：[{"question": "...", "relevant_ids": [0,3], "reference_answer": "..."}, ...]
        """
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        logger.info(f"批量评测：{len(test_cases)} 条测试数据")
        reports = []

        # 获取 judge 模型（复用 generator 的模型）
        generator = self._ensure_generator()
        judge_model = getattr(generator, "model", None)
        judge_tokenizer = getattr(generator, "tokenizer", None)

        for i, case in enumerate(test_cases, 1):
            question = case["question"]
            logger.info(f"[{i}/{len(test_cases)}] Q: {question[:50]}...")

            result = self.ask(question, return_context=False)

            report = self.evaluator.full_evaluation(
                question=question,
                answer=result["answer"],
                retrieved_docs=result["retrieved_docs"],
                relevant_ids=case.get("relevant_ids"),
                reference_answer=case.get("reference_answer", ""),
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                relevant_sources=case.get("relevant_sources"),
            )
            reports.append(report)

        # 汇总
        overall_scores = [
            r["overall_score"] for r in reports if "overall_score" in r
        ]
        if overall_scores:
            logger.info(
                f"批量评测完成：avg={np.mean(overall_scores):.2f}, "
                f"n={len(overall_scores)}"
            )

        return reports

    # ========== 对比实验 ==========

    @timer
    def compare_chunk_strategies(
        self, documents: List[Dict], questions: List[str]
    ) -> Dict:
        """
        对比不同 chunk 策略的检索效果：对同一知识库、同一批问题分别重建索引并检索。
        返回每种策略的平均相似度；带标注的 recall/MRR 指标走 EvaluationEngine。
        """
        strategies = ["fixed_size", "recursive", "semantic"]
        results = {}

        for strategy in strategies:
            logger.info(f"测试 chunk 策略：{strategy}")
            self.chunker.strategy = strategy

            # 重建索引
            self.build_index(documents)

            # 对每个问题测检索效果
            strategy_results = []
            for q in questions:
                retrieved = self._retrieve(q)
                strategy_results.append(
                    {
                        "question": q,
                        "retrieved_chunks": [
                            {"index": r["chunk_index"], "score": r["score"]}
                            for r in retrieved
                        ],
                        "avg_score": (
                            np.mean([r["score"] for r in retrieved])
                            if retrieved
                            else 0
                        ),
                    }
                )

            results[strategy] = {
                "avg_retrieval_score": round(
                    np.mean([r["avg_score"] for r in strategy_results]), 4
                ),
                "details": strategy_results,
            }

        logger.info(f"Chunk 策略对比完成：{list(results.keys())}")
        return results
