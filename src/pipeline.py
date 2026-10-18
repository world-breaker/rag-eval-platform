"""RAG Pipeline：编排完整流程"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from src.utils import load_config, logger, timer
from src.chunking import DocumentChunker
from src.embedding import EmbeddingEngine
from src.retrieval import RetrievalEngine
from src.generation import GenerationEngine
from src.evaluation import EvaluationEngine
from src.hybrid_retrieval import HybridRetriever


class RAGPipeline:
    """RAG 全链路编排器：模块化设计，各组件可独立替换与对比实验。"""

    def __init__(self, config_path: str = "config.yaml", load_generator: bool = True):
        self.config = load_config(config_path)
        logger.info("初始化 RAG Pipeline...")

        self.chunker = DocumentChunker(self.config)
        self.embedder = EmbeddingEngine(self.config)
        self.retriever = RetrievalEngine(self.config)
        # 混合检索：retrieval.mode="hybrid" 时启用（BM25+向量 RRF 融合，可选重排）；
        # 默认 "vector" 则不实例化，行为与旧版完全一致（零回归）。
        self.retrieval_mode = self.config["retrieval"].get("mode", "vector")
        self.hybrid = (
            HybridRetriever(self.config, self.retriever)
            if self.retrieval_mode == "hybrid"
            else None
        )
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

    def _retrieve(self, question: str, top_k: int = None) -> List[Dict]:
        """统一检索入口：根据 retrieval.mode 选择向量检索或混合检索（BM25+向量 RRF）。

        返回值结构与 RetrievalEngine.search 一致，供 ask / 对比实验 / 批量评测复用。
        """
        if self.doc_embeddings is None or len(self.chunks) == 0:
            raise RuntimeError("请先调用 build_index() 构建向量库")

        query_emb = self.embedder.encode([{"text": question}])

        if self.retrieval_mode == "hybrid" and self.hybrid is not None:
            return self.hybrid.search(
                query_emb, self.doc_embeddings, self.chunks, top_k=top_k, query_text=question
            )
        return self.retriever.search(query_emb, self.doc_embeddings, self.chunks, top_k)

    # ========== 问答 ==========

    @timer
    def ask(
        self, question: str, top_k: int = None, return_context: bool = True
    ) -> Dict:
        """单次问答"""
        if self.doc_embeddings is None or len(self.chunks) == 0:
            raise RuntimeError("请先调用 build_index() 构建向量库")

        # 1. 检索（向量 or 混合）
        retrieved = self._retrieve(question, top_k)

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
