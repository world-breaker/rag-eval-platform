"""评测模块：三层指标体系。

- 分三层的原因：检索错误会传导到生成，需要独立评检索；生成是否忠实/切题需要独立评生成。
- LLM-as-Judge 的一致性问题：同一条答案多次打分可能不同，可通过结构化输出约束 + 多次评分取中位数缓解。
"""

import re
import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from src.utils import logger, timer


class EvaluationEngine:
    """评测引擎。三层指标：检索质量 + 生成质量 + 传统指标。"""

    def __init__(self, config: dict):
        eval_cfg = config["evaluation"]
        self.retrieval_metrics = eval_cfg.get(
            "retrieval_metrics", ["recall@3", "recall@5", "mrr"]
        )
        self.judge_dimensions = eval_cfg.get(
            "judge_dimensions", ["faithfulness", "relevance", "completeness"]
        )
        self.traditional_metrics = eval_cfg.get(
            "traditional_metrics", ["bleu", "rouge_l"]
        )

    # ========== 检索评测 ==========

    @timer
    def evaluate_retrieval(
        self, retrieved_docs: List[Dict], relevant_ids: List[int]
    ) -> Dict[str, float]:
        """
        检索质量评测。
        relevant_ids: 标注的相关文档 ID 列表（chunk_index）
        """
        if not relevant_ids:
            return {
                "recall@3": 0.0,
                "recall@5": 0.0,
                "precision@3": 0.0,
                "mrr": 0.0,
            }

        retrieved_ids = [doc["chunk_index"] for doc in retrieved_docs]
        relevant_set = set(relevant_ids)

        metrics = {}

        # Recall@K：检索结果不足 K 条时，缺失排位视为未命中（分母是相关文档数）
        for k in [3, 5]:
            hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_set)
            metrics[f"recall@{k}"] = round(hits / len(relevant_set), 4)

        # Precision@3：分母固定为 3（不足 3 条时空缺位置按未命中计）
        hits = sum(1 for rid in retrieved_ids[:3] if rid in relevant_set)
        metrics["precision@3"] = round(hits / 3, 4)

        # MRR（Mean Reciprocal Rank）：第一个相关文档的排位倒数
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_set:
                metrics["mrr"] = round(1.0 / rank, 4)
                break
        else:
            metrics["mrr"] = 0.0

        return metrics

    def evaluate_retrieval_doc_level(
        self, retrieved_docs: List[Dict], relevant_sources: List[str]
    ) -> Dict[str, float]:
        """文档级检索评测（跨切分策略可比的口径）。

        为什么需要：chunk 边界随切分策略/参数变化，chunk 索引无法跨策略对照；
        「检索到正确文档」才是策略无关的评测目标。
        做法：把 chunk 级检索结果按 source 去重为文档级序列（保留原相关性排序，
        不相关文档占位但不命中），再复用 evaluate_retrieval 的同一套公式。
        """
        if not relevant_sources:
            return self.evaluate_retrieval([], [])

        source_to_id = {s: i for i, s in enumerate(relevant_sources)}
        seen, doc_level = set(), []
        for doc in retrieved_docs:
            source = doc.get("source", "")
            if source in seen:
                continue  # 同一文档的多个 chunk 只记一次
            seen.add(source)
            # 不相关文档分配不重复的负索引：保留排序位置（影响 MRR/precision），但永不命中
            doc_level.append({"chunk_index": source_to_id.get(source, -1 - len(doc_level))})

        return self.evaluate_retrieval(doc_level, list(range(len(relevant_sources))))

    # ========== 生成评测（LLM-as-Judge） ==========

    def evaluate_generation(
        self,
        question: str,
        answer: str,
        context: str,
        judge_model=None,
        judge_tokenizer=None,
    ) -> Dict[str, float]:
        """
        LLM-as-Judge：从多个维度打分。
        judge_model/tokenizer 传入已加载的模型，不重复加载。
        """
        scores = {}
        for dim in self.judge_dimensions:
            score = self._judge_single_dim(
                dim, question, answer, context, judge_model, judge_tokenizer
            )
            scores[dim] = score
        return scores

    def _judge_single_dim(
        self, dimension: str, question: str, answer: str, context: str, model, tokenizer
    ) -> float:
        """单维度 LLM 评判（1-5分）"""
        criteria = {
            "faithfulness": "回答中的每一条信息是否都能在参考文档中找到依据？有没有编造？",
            "relevance": "回答是否直接回应了用户的问题？有没有偏离主题？",
            "completeness": "回答是否覆盖了参考文档中与该问题相关的所有重要信息？",
        }

        judge_prompt = f"""你是一个严格的评测裁判。请根据以下标准对回答打分（1-5分，只输出数字）。

## 评测维度：{dimension}
## 评测标准：{criteria.get(dimension, "综合质量")}

## 用户问题
{question}

## 参考文档
{context[:2000]}

## 待评测回答
{answer}

## 请输出一个 1-5 的数字（5=完美，1=完全不合格）
分数："""

        try:
            if model is None:
                # 没有模型用启发式评分——不作为正式评估
                return self._heuristic_score(dimension, answer, context)

            messages = [{"role": "user", "content": judge_prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt")

            import torch

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
            ).strip()

            # 提取数字
            match = re.search(r"([1-5])", response)
            if match:
                return int(match.group(1))
            return 3.0  # 无法解析时给中性分
        except Exception as e:
            logger.warning(f"LLM-as-Judge {dimension} 评分失败: {e}")
            return self._heuristic_score(dimension, answer, context)

    def _heuristic_score(self, dimension: str, answer: str, context: str) -> float:
        """无 LLM 时的启发式评分（粗糙但可运行）"""
        if dimension == "faithfulness":
            # 简单检查：回答长度与上下文长度的比例
            # 这只是占位——正式评测必须有 LLM
            return min(5.0, max(1.0, len(answer) / 200))
        elif dimension == "relevance":
            return min(5.0, max(1.0, len(answer) / 150))
        else:
            return min(5.0, max(1.0, len(answer) / 100))

    # ========== 传统指标 ==========

    @staticmethod
    def compute_bleu(reference: str, candidate: str) -> float:
        """简化 BLEU-1：单个参考的 unigram 精度"""
        if not reference or not candidate:
            return 0.0
        # 中文按字而非按词
        ref_chars = set(reference)
        cand_chars = candidate
        if not ref_chars:
            return 0.0
        matches = sum(1 for c in cand_chars if c in ref_chars)
        # 加 brevity penalty
        bp = min(1.0, len(candidate) / max(len(reference), 1))
        precision = matches / max(len(candidate), 1)
        return round(bp * precision, 4)

    @staticmethod
    def compute_rouge_l(reference: str, candidate: str) -> float:
        """简化 ROUGE-L：最长公共子序列比例"""
        if not reference or not candidate:
            return 0.0

        # LCS 长度（DP简化，仅用于示意）
        m, n = len(reference), len(candidate)
        if m == 0 or n == 0:
            return 0.0

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if reference[i] == candidate[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

        lcs_len = dp[m][n]
        recall = lcs_len / m if m > 0 else 0
        precision = lcs_len / n if n > 0 else 0
        if recall + precision == 0:
            return 0.0
        f1 = 2 * recall * precision / (recall + precision)
        return round(f1, 4)

    # ========== 综合评测 ==========

    @timer
    def full_evaluation(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[Dict],
        relevant_ids: List[int] = None,
        reference_answer: str = "",
        judge_model=None,
        judge_tokenizer=None,
        relevant_sources: List[str] = None,
    ) -> Dict:
        """
        一站式评测：检索 + 生成 + 传统 三项全跑。
        返回完整的评测报告。

        检索评测两种口径（二选一）：
        - relevant_sources（推荐）：文档级标注，跨切分策略可比
        - relevant_ids：chunk 级标注，仅在同一套切分配置内有效
        """
        report = {"question": question, "answer": answer}

        # 层1：检索质量
        if relevant_sources:
            report["retrieval"] = self.evaluate_retrieval_doc_level(
                retrieved_docs, relevant_sources
            )
        elif relevant_ids:
            report["retrieval"] = self.evaluate_retrieval(retrieved_docs, relevant_ids)
        else:
            report["retrieval"] = {"note": "无标注相关文档，跳过了检索评测"}

        # 层2：生成质量（LLM-as-Judge）
        context = "\n".join([d["text"][:500] for d in retrieved_docs[:3]])
        report["generation_quality"] = self.evaluate_generation(
            question, answer, context, judge_model, judge_tokenizer
        )

        # 层3：传统指标
        if reference_answer:
            report["traditional"] = {
                "bleu": self.compute_bleu(reference_answer, answer),
                "rouge_l": self.compute_rouge_l(reference_answer, answer),
            }
        else:
            report["traditional"] = {"note": "无参考回答，跳过了传统指标评测"}

        # 汇总
        gen_scores = [
            v
            for v in report["generation_quality"].values()
            if isinstance(v, (int, float))
        ]
        if gen_scores:
            report["overall_score"] = round(np.mean(gen_scores), 2)

        logger.info(
            f"评测完成：overall={report.get('overall_score', 'N/A')}, "
            f"gen_dims={list(report['generation_quality'].keys())}"
        )
        return report
