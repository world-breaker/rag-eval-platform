"""评测模块：三层指标体系。

- 分三层的原因：检索错误会传导到生成，需要独立评检索；生成是否忠实/切题需要独立评生成。
- LLM-as-Judge 的一致性问题：同一条答案多次打分可能不同，可用结构化输出约束 +
  多次评分取中位数缓解（`evaluation.judge_samples`，默认 1 = 历史行为）。

裁判的两类「静默降级」都在本模块被显式标记（不再只看得到分数、看不到来源）：
1. 异常（含设备不一致）→ 启发式兜底 `_heuristic_score`；
2. 裁判输出解析不出 1-5 数字 → 中性兜底 3.0。
两者都会写进 `meta.fallbacks`（维度 + 原因 + 原始输出）并以 warning 级日志点名，
报告层可直接打印，避免「报告里有数、其实是兜底值」。
"""

import re
import json
import statistics
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from src.utils import logger, timer

# 兜底来源（出现在 meta.fallbacks 里的两种 source；其余 source 表示分数确实来自裁判）
FALLBACK_SOURCES = ("heuristic", "neutral_unparsed")


class EvaluationEngine:
    """评测引擎。三层指标：检索质量 + 生成质量 + 传统指标。"""

    def __init__(self, config: dict, judge_samples: Optional[int] = None):
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
        # 同一样本调 N 次裁判取中位数（N=1 时与历史行为完全一致：单次贪心解码）
        self.judge_samples = self.validate_judge_samples(
            eval_cfg.get("judge_samples", 1) if judge_samples is None else judge_samples
        )
        # 只有 N>1 才启用采样（N=1 保持 temperature=0/贪心，保证可复现与历史可比）
        self.judge_temperature = float(eval_cfg.get("judge_temperature", 0.8))

    @staticmethod
    def validate_judge_samples(n) -> int:
        """judge_samples 必须是 ≥1 的整数；配置写错要立刻炸，不要静默按 1 跑。"""
        try:
            n = int(n)
        except (TypeError, ValueError):
            raise ValueError(f"evaluation.judge_samples 必须是正整数，实际 {n!r}")
        if n < 1:
            raise ValueError(f"evaluation.judge_samples 必须 ≥ 1，实际 {n!r}")
        return n

    def set_judge_samples(self, n: int) -> int:
        """运行时覆盖采样次数（供 CLI `--judge-samples N` 使用）。"""
        self.judge_samples = self.validate_judge_samples(n)
        return self.judge_samples

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
        return_meta: bool = False,
    ):
        """
        LLM-as-Judge：从多个维度打分。
        judge_model/tokenizer 传入已加载的模型，不重复加载。

        返回（默认，向后兼容）：{dimension: score}
        返回（return_meta=True）：{"scores": {dim: score}, "meta": {...}}
        meta 里带 `judge_samples`（采样次数）与 `fallbacks`（哪些维度、因为什么用了兜底），
        这样调用方不必「猜」某个分是裁判给的还是兜底值。
        """
        scores: Dict[str, float] = {}
        details: Dict[str, Dict] = {}
        fallbacks: List[Dict] = []
        for dim in self.judge_dimensions:
            r = self._judge_single_dim(
                dim, question, answer, context, judge_model, judge_tokenizer
            )
            scores[dim] = r["score"]
            details[dim] = r
            if r["source"] in FALLBACK_SOURCES:
                fallbacks.append(
                    {
                        "dim": dim,
                        "source": r["source"],
                        "score": r["score"],
                        "reason": r["reason"],
                        "raws": r["raws"],
                    }
                )

        if return_meta:
            return {
                "scores": scores,
                "meta": {
                    "judge_samples": self.judge_samples,
                    "fallbacks": fallbacks,
                    "details": details,
                },
            }
        return scores

    @staticmethod
    def _resolve_input_device(model, tokenizer, torch_module):
        """输入张量该去的设备：优先模型（`model.device`），其次 tokenizer，最后 CPU。

        为什么需要兜底：包装类（如实验侧的记录器）或不带 `device` 属性的自定义模型
        未必有 `model.device`；此时若直接取属性会 AttributeError，若默认 CPU 则
        **在 GPU 上必然复现「张量在 CPU、权重在 GPU」的崩溃**。
        """
        for holder in (model, tokenizer):
            device = getattr(holder, "device", None)
            if device is not None:
                return device
        return torch_module.device("cpu")

    def _generate_judge_raw(
        self, dimension: str, judge_prompt: str, model, tokenizer, torch_module
    ) -> str:
        """跑一次裁判生成，返回解码后的原始文本（解析在调用方做）。"""
        messages = [{"role": "user", "content": judge_prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")

        # 输入必须与模型权重同设备：tokenizer 产出的是 CPU 张量，而裁判模型通常在
        # device_map="cuda" 上（见 src/generation.py::_generate_local 的同款处理）。
        # 漏掉这一步会抛 "Expected all tensors to be on the same device"，
        # 而该异常会被兜底逻辑吞掉——那就变成了「看似有分、其实是启发式」。
        target_device = self._resolve_input_device(model, tokenizer, torch_module)
        inputs = {
            k: (v.to(target_device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        gen_kwargs = {
            "max_new_tokens": 5,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if self.judge_samples > 1:
            # 多次采样才有「方差」可言：N>1 时按 temperature 采样（结构化约束仍是「只输出数字」）
            gen_kwargs.update(
                do_sample=True,
                temperature=self.judge_temperature,
                top_p=0.95,
            )
        else:
            # N=1 保持历史行为：贪心解码，可复现
            gen_kwargs.update(do_sample=False)

        with torch_module.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        return tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        ).strip()

    def _judge_single_dim(
        self, dimension: str, question: str, answer: str, context: str, model, tokenizer
    ) -> Dict:
        """单维度 LLM 评判（1-5分）。

        返回结构化结果，而不是裸分数：
        `{"score", "source", "reason", "parsed", "raws", "n"}`
        - source="llm"（N=1 单次）/ "llm_median"（N>1 取中位数）：分数来自裁判；
        - source="heuristic"：异常兜底（分数是 `_heuristic_score`，**不是**裁判给的）；
        - source="neutral_unparsed"：裁判输出解析不出数字，给中性兜底 3.0。

        N>1 时同一样本调 N 次、取**中位数**（对偶数的 N 取中间两数均值，可能是 .5）。
        """
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

        if model is None:
            # 没有模型用启发式评分——不作为正式评估，必须标记来源
            score = self._heuristic_score(dimension, answer, context)
            reason = "未传入裁判模型（judge_model=None），分数由 _heuristic_score 生成"
            logger.warning(
                f"LLM-as-Judge「{dimension}」该分数来自启发式兜底（不是 LLM 裁判）：{reason}；"
                f"score={score}"
            )
            return {
                "score": score,
                "source": "heuristic",
                "reason": reason,
                "parsed": [],
                "raws": [],
                "n": 0,
            }

        import torch

        n = self.judge_samples
        parsed: List[float] = []
        raws: List[str] = []
        failures: List[str] = []
        for i in range(n):
            try:
                raw = self._generate_judge_raw(
                    dimension, judge_prompt, model, tokenizer, torch
                )
            except Exception as e:  # 单次失败不炸整批；但要留下原因
                failures.append(f"第{i + 1}次调用异常 {type(e).__name__}: {str(e)[:200]}")
                continue
            raws.append(raw)
            match = re.search(r"([1-5])", raw)
            if match:
                parsed.append(float(int(match.group(1))))
            else:
                failures.append(f"第{i + 1}次输出未解析出 1-5 数字: {raw!r}")

        if not parsed:
            # 一次都没拿到可用分数：区分「调用异常」与「输出不可解析」。
            # 两者都不许静默——前者用启发式兜底（保持历史行为），后者沿用中性 3.0。
            reason = "；".join(failures) or "裁判未产出可用输出"
            if not raws:  # 全部走到了异常分支（不是「输出不可解析」）
                score = self._heuristic_score(dimension, answer, context)
                logger.warning(
                    f"LLM-as-Judge「{dimension}」该分数来自启发式兜底（不是 LLM 裁判）："
                    f"{reason}；score={score}"
                )
                return {
                    "score": score,
                    "source": "heuristic",
                    "reason": reason,
                    "parsed": [],
                    "raws": raws,
                    "n": n,
                }
            logger.warning(
                f"LLM-as-Judge「{dimension}」裁判输出无法解析出数字，"
                f"该分数为中性兜底值 3.0（不是 LLM 裁判给出的分）：{reason}"
            )
            return {
                "score": 3.0,
                "source": "neutral_unparsed",
                "reason": reason,
                "parsed": [],
                "raws": raws,
                "n": n,
            }

        if n == 1:
            # N=1：与历史行为完全一致——单次生成、取该次分数
            return {
                "score": parsed[0],
                "source": "llm",
                "reason": None,
                "parsed": parsed,
                "raws": raws,
                "n": n,
            }

        score = round(float(statistics.median(parsed)), 2)
        reason = None
        if failures:
            reason = f"{len(parsed)}/{n} 次拿到可用分数，其余失败：" + "；".join(failures)
            logger.warning(f"LLM-as-Judge「{dimension}」部分采样失败，中位数取自已成功的那几次：{reason}")
        return {
            "score": score,
            "source": "llm_median",
            "reason": reason,
            "parsed": parsed,
            "raws": raws,
            "n": n,
        }

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
        gen = self.evaluate_generation(
            question, answer, context, judge_model, judge_tokenizer, return_meta=True
        )
        report["generation_quality"] = gen["scores"]
        # 兜底来源单独放（不污染分数结构）：报告层可据此标出「这个分不是裁判给的」
        report["generation_meta"] = gen["meta"]

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
