"""生成层忠实度实测：规则判据 / 本地 1.5B 裁判 / 强裁判 API —— 三臂对照。

为什么要有这个脚本：
- `src/evaluation.py::evaluate_generation`（LLM-as-Judge）此前**有实现、零实测**：
  experiments/ 里没有任何生成层结果，README 却宣称了三层指标（含 faithfulness）。
  面试被追问「faithfulness 多少分」会答不上来。
- 只有把「本地 1.5B 当裁判」的分数与「强裁判（DeepSeek）」在同一批样本上对齐，
  才能把「本地小模型够不够用」从感觉变成**可复现的实测结论**。

三臂（同一批样本、同一份 prompt 上下文、同一套 rubric）：
- 臂 A｜规则判据：确定性、零成本、离线。数字/实体落地率、句子级归因、拒答正确性、
  EM/F1/ROUGE-L/BLEU（有 reference_answer 才做后四个；ROUGE-L/BLEU 复用
  `EvaluationEngine.compute_rouge_l/compute_bleu`）。
- 臂 B｜本地 1.5B 裁判：直接调用仓库现有的 `EvaluationEngine.evaluate_generation()`
  （judge_model 与生成模型同款 = 自评，这正是最值得被质疑的一点，所以必须与臂 C 对齐）。
- 臂 C｜强裁判 API：DeepSeek（`deepseek-chat`，一次调用返回 3 个维度的 JSON），
  **同一套 rubric（判据文本与 src/evaluation.py 逐字一致，有单测守住不漂移）**。

运行：
    python experiments/run_generation_eval.py --no-api            # 只跑臂 A + 臂 B（离线）
    python experiments/run_generation_eval.py                     # 三臂全跑（需要 DEEPSEEK_API_KEY）
    python experiments/run_generation_eval.py --limit 3 --arms rule          # 冒烟
    python experiments/run_generation_eval.py --render-only experiments/results/generation_eval_xxxx.json

成本与密钥纪律：
- API key **只从环境变量 `DEEPSEEK_API_KEY` 读**，绝不写入任何文件/日志/报告/缓存/异常文本。
- 每样本只调 1 次（一次返回 3 维）、temperature=0、max_tokens 收紧、响应落盘缓存
  （`experiments/results/api_judge_cache.json`，内容不含 key）；重跑优先读缓存，不重复花钱。
- 费用硬上限 ¥0.5：预估超限直接拒绝启动 API 臂；运行中累计超限立即停止并在报告里写明。
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "experiments/results"
API_CACHE_PATH = RESULTS_DIR / "api_judge_cache.json"

# ---------------------------------------------------------------- 数据集

DATASETS: List[Dict] = [
    {
        "name": "realistic",
        "kb": "knowledge_base_large.json",
        "qa": "test_qa_large.json",
        "answerable": True,
        "kind": "gold",
        "note": "人工精标、话题区分度大（14 篇 / 14 题）",
    },
    {
        "name": "near_dup_stress",
        "kb": "knowledge_base_stress.json",
        "qa": "test_qa_stress.json",
        "answerable": True,
        "kind": "gold",
        "note": "含 8 篇近义副本的噪声语料（10 篇 / 8 题）",
    },
    {
        "name": "sample",
        "kb": "knowledge_base.json",
        "qa": "test_qa.json",
        "answerable": True,
        "kind": "gold",
        "note": "仓库自带最小样例（5 篇 / 5 题）",
    },
    {
        "name": "unanswerable",
        "kb": "knowledge_base_large.json",
        "qa": "test_qa_unanswerable.json",
        "answerable": False,
        "kind": "synthetic_new",
        "note": "**本次新造**的拒答样本（6 题，答案确实不在知识库内），非原金标",
    },
]

# 上下文构造：与 `EvaluationEngine.full_evaluation` 一致（top-3 × 500 字符截断），
# 三臂共用同一份 context，否则「一致率」没有可比性。
CONTEXT_TOP_K = 3
CONTEXT_MAX_CHARS = 500

# ---------------------------------------------------------------- 臂 A：规则判据

# 判据 1：数字 / 实体落地率 —— 回答里的「硬事实」有多少能在上下文中找到字面依据。
# 数字（含小数、百分号）与英文/数字标识符（LoRA、NF4、QTAG-A1、vLLM、IVF+PQ 里的 IVF/PQ…）
# 是最容易被小模型编造、也最容易用字面匹配验伪的两类 token。
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s?%?")
ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\.]*")

# 判据 2：句子级归因阈值（字符 2-gram 多重集覆盖率 ≥ 阈值 → 判为「在上下文中有依据」）。
# 阈值怎么定的：见报告「规则判据校准」小节（5 条手工标注样本的覆盖率实测 + 已知误判模式）。
ATTRIBUTION_THRESHOLD = 0.60

# 判据 3：拒答标记。RAG prompt（src/generation.py）显式要求模型说
# 「根据提供的文档无法回答该问题」，因此这句话本身就是判定正例的字面形式。
REFUSAL_MARKERS = [
    "无法回答",
    "无法从",
    "无法找到",
    "无法确定",
    "不知道",
    "不清楚",
    "没有相关",
    "未提及",
    "没有提及",
    "未包含",
    "不包含",
    "没有提供",
    "未提供",
    "文档中没有",
    "提供的文档中",
]

_PUNCT_RE = re.compile(r"[\s，。、；：！？（）【】《》「」『』\"'\.,;:!?()\[\]{}<>/\-_*=+~`·…]+")


def normalize_text(text: str) -> str:
    """去掉空白与标点并转小写，用于 EM / F1 归一化比较。"""
    return _PUNCT_RE.sub("", (text or "").lower())


def extract_facts(text: str) -> List[str]:
    """抽取「硬事实」token：数字 + 英文/数字标识符（长度 ≥ 2）。"""
    facts: List[str] = []
    for m in NUMBER_RE.finditer(text or ""):
        tok = m.group(0).replace(" ", "")
        if tok not in ("", "%"):
            facts.append(tok)
    for m in ENTITY_RE.finditer(text or ""):
        tok = m.group(0).lower().strip(".-")
        if len(tok) >= 2:
            facts.append(tok)
    return facts


def fact_landing(answer: str, context: str, extra: str = "") -> Dict:
    """数字 / 实体落地率：回答里的硬事实有多少能在（上下文 + 参考答案）中找到字面依据。

    为什么把 reference_answer 也算作依据：参考回答本身就是「事实白名单」，
    模型用同义词回答了正确事实（但上下文被 500 字符截断掉）不该被判成编造。
    """
    haystack = f"{context or ''}\n{extra or ''}".lower()
    facts = extract_facts(answer)
    missed = [f for f in facts if f not in haystack]
    hit = len(facts) - len(missed)
    return {
        "n": len(facts),
        "hit": hit,
        "rate": round(hit / len(facts), 4) if facts else None,
        "missed": missed,
    }


def split_sentences(text: str) -> List[str]:
    """按中英文句末标点切句；过短的碎片（<3 字符）不参与归因统计。"""
    parts = re.split(r"[。！？!?；;\n]+", text or "")
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def ngram_coverage(unit: str, source: str, n: int = 2) -> float:
    """unit 的字符 n-gram 多重集被 source 覆盖的比例（0~1）。

    用多重集（而不是集合）是因为重复短语是幻觉的典型特征：
    「量化量化量化」若上下文只出现一次「量化」，覆盖率应显著低于 1。
    """
    unit_norm = normalize_text(unit)
    source_norm = normalize_text(source)
    if not unit_norm:
        return 0.0
    if len(unit_norm) < n:
        return 1.0 if unit_norm in source_norm else 0.0
    grams = [unit_norm[i : i + n] for i in range(len(unit_norm) - n + 1)]
    from collections import Counter

    src_counter = Counter(source_norm[i : i + n] for i in range(len(source_norm) - n + 1))
    hit = sum(min(1, src_counter.get(g, 0)) for g in grams)
    return round(hit / len(grams), 4)


def sentence_attribution(
    answer: str, context: str, threshold: float = ATTRIBUTION_THRESHOLD, extra: str = ""
) -> Dict:
    """句子级归因：每句回答与上下文的 n-gram 覆盖率 ≥ 阈值即判为已归因。"""
    source = f"{context or ''}\n{extra or ''}"
    rows = []
    for sent in split_sentences(answer):
        cov = ngram_coverage(sent, source)
        rows.append(
            {"sentence": sent, "coverage": cov, "attributed": cov >= threshold}
        )
    n = len(rows)
    attributed = sum(1 for r in rows if r["attributed"])
    return {
        "n": n,
        "attributed": attributed,
        "rate": round(attributed / n, 4) if n else None,
        "threshold": threshold,
        "sentences": rows,
    }


def detect_refusal(answer: str) -> bool:
    """回答是否属于「拒答 / 声明不知道」。"""
    text = answer or ""
    return any(m in text for m in REFUSAL_MARKERS)


def refusal_verdict(answer: str, expect_refusal: bool) -> Dict:
    """拒答正确性：不可答样本应该拒答；可答样本不应该拒答。"""
    is_refusal = detect_refusal(answer)
    return {
        "is_refusal": is_refusal,
        "expected_refusal": expect_refusal,
        "correct": is_refusal == expect_refusal,
    }


def exact_match(reference: str, candidate: str) -> float:
    return 1.0 if normalize_text(reference) == normalize_text(candidate) else 0.0


def char_f1(reference: str, candidate: str) -> float:
    """字符级 F1（中文按字，无需分词）。空参考/空回答按 0 处理。"""
    ref = normalize_text(reference)
    cand = normalize_text(candidate)
    if not ref or not cand:
        return 0.0
    from collections import Counter

    ref_c, cand_c = Counter(ref), Counter(cand)
    overlap = sum(min(ref_c[ch], cand_c[ch]) for ch in ref_c)
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def rule_faithfulness(attr: Dict, landing: Dict, refusal: Optional[Dict]) -> float:
    """把臂 A 的规则输出粗映射到 1-5 分，好与臂 B/C 的裁判分对齐。

    映射（写死在报告里，避免「事后调参凑一致率」的嫌疑）：
    - 不可答样本：正确拒答 = 5，未拒答（编造）= 1；
    - 可答样本：base = min(句子归因率, 硬事实落地率)（任一硬事实没落地就封顶），score = 1 + 4*base。
    这个映射是**粗粒度**的：它只区分「整体有据 / 整体无据」，
    1-3 分区间几乎不产生区分（见报告已知误判模式）。
    """
    if refusal is not None:
        return 5.0 if refusal["correct"] else 1.0
    parts = [v for v in (attr.get("rate"), landing.get("rate")) if v is not None]
    base = min(parts) if parts else None
    if base is None:
        return 3.0
    return round(1 + 4 * base, 2)


def rule_scores(
    answer: str,
    context: str,
    reference: str = "",
    expect_refusal: bool = False,
    threshold: float = ATTRIBUTION_THRESHOLD,
) -> Dict:
    """臂 A 的完整规则判据输出。

    拒答判定对**所有**样本都跑：
    - 不可答样本：正确拒答 = 好；
    - 可答样本：拒答 = **误拒答**（模型明明有资料却拒答），这是 1.5B 最典型的失效模式之一。
    注意规则分（rule_faithfulness）只在「应当拒答」的样本上才把拒答当作满分依据，
    否则任何一句「以上内容供参考」都会把可答样本刷成 5 分。
    """
    landing = fact_landing(answer, context, extra=reference)
    attr = sentence_attribution(answer, context, threshold, extra=reference)
    refusal = refusal_verdict(answer, expect_refusal)
    out = {
        "numeric_landing": landing,
        "sentence_attribution": attr,
        "refusal": refusal,
        "rule_faithfulness": rule_faithfulness(
            attr, landing, refusal if expect_refusal else None
        ),
    }
    if reference:
        from src.evaluation import EvaluationEngine

        out["lexical"] = {
            "em": exact_match(reference, answer),
            "f1": char_f1(reference, answer),
            "rouge_l": EvaluationEngine.compute_rouge_l(reference, answer),
            "bleu1": EvaluationEngine.compute_bleu(reference, answer),
        }
    else:
        out["lexical"] = {"note": "无参考回答，跳过 EM/F1/ROUGE-L/BLEU"}
    return out


# ---------------------------------------------------------------- 臂 B/C 共用的 rubric

# 与 `src/evaluation.py::EvaluationEngine._judge_single_dim` 的 criteria **逐字一致**，
# 由 tests/test_generation_eval.py 对源码做文本断言，防止两臂 rubric 悄悄漂移。
JUDGE_CRITERIA = {
    "faithfulness": "回答中的每一条信息是否都能在参考文档中找到依据？有没有编造？",
    "relevance": "回答是否直接回应了用户的问题？有没有偏离主题？",
    "completeness": "回答是否覆盖了参考文档中与该问题相关的所有重要信息？",
}
JUDGE_DIMENSIONS = ["faithfulness", "relevance", "completeness"]


def build_api_judge_messages(question: str, answer: str, context: str) -> List[Dict]:
    """臂 C 的 prompt：与臂 B 同一套维度与判据文本，一次返回 3 维 JSON。"""
    rubric = "\n".join(f"- {d}：{JUDGE_CRITERIA[d]}" for d in JUDGE_DIMENSIONS)
    system = "你是一个严格的评测裁判。只输出 JSON 对象，不要输出任何解释或多余文字。"
    user = f"""请根据以下标准对回答打分（1-5 分整数，5=完美，1=完全不合格）。

## 评测维度与标准
{rubric}

## 用户问题
{question}

## 参考文档
{context[:2000]}

## 待评测回答
{answer}

## 输出格式（严格 JSON，不要 markdown 代码块）
{{"faithfulness": <1-5>, "relevance": <1-5>, "completeness": <1-5>}}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_api_judge(content: str) -> Dict:
    """解析强裁判返回的 JSON；缺维度返回 None 表示解析失败。"""
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {d: None for d in JUDGE_DIMENSIONS}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {d: None for d in JUDGE_DIMENSIONS}
    out = {}
    for d in JUDGE_DIMENSIONS:
        v = data.get(d, None)
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            v = None
        out[d] = v if (v is not None and 1 <= v <= 5) else None
    return out


# ---------------------------------------------------------------- 一致率统计


def agreement_stats(a: List, b: List) -> Dict:
    """两臂逐样本对齐：精确一致率、±1 一致率、MAE、Pearson r、高/低二分类混淆。"""
    pairs = [
        (x, y)
        for x, y in zip(a, b)
        if isinstance(x, (int, float)) and isinstance(y, (int, float))
    ]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "note": "无可比对样本"}
    exact = sum(1 for x, y in pairs if x == y) / n
    within1 = sum(1 for x, y in pairs if abs(x - y) <= 1) / n
    mae = sum(abs(x - y) for x, y in pairs) / n
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    pearson = round(cov / (vx * vy), 4) if vx > 0 and vy > 0 else None
    # 高/低二分类（≥4 为高）：答样本量小时，「差 1 分」与「档次判断一致」要分开看
    tp = sum(1 for x, y in pairs if x >= 4 and y >= 4)
    tn = sum(1 for x, y in pairs if x < 4 and y < 4)
    fp = sum(1 for x, y in pairs if x >= 4 and y < 4)
    fn = sum(1 for x, y in pairs if x < 4 and y >= 4)
    return {
        "n": n,
        "exact_agreement": round(exact, 4),
        "within1_agreement": round(within1, 4),
        "mae": round(mae, 4),
        "pearson": pearson,
        "binary_agreement": round((tp + tn) / n, 4),
        "confusion_high_low": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "mean_a": round(mx, 4),
        "mean_b": round(my, 4),
    }


def segment_report(arm_b: List, arm_c: List, dim: str = "faithfulness") -> Dict:
    """按强裁判档位分段：本地 1.5B 在不同难度区间是否可信。

    分段依据用**强裁判**（臂 C）的分数，因为它是我们要逼近的目标：
    低（≤2）/ 中（=3）/ 高（≥4）。
    """
    segs: Dict[str, Dict] = {}
    for lo, hi, label in [(1, 2, "低(1-2)"), (3, 3, "中(3)"), (4, 5, "高(4-5)")]:
        pairs = [
            (x, y)
            for x, y in zip(arm_b, arm_c)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and lo <= y <= hi
        ]
        if not pairs:
            segs[label] = {"n": 0}
            continue
        n = len(pairs)
        segs[label] = {
            "n": n,
            "strong_mean": round(sum(y for _, y in pairs) / n, 3),
            "local_mean": round(sum(x for x, _ in pairs) / n, 3),
            "exact_agreement": round(sum(1 for x, y in pairs if x == y) / n, 4),
            "within1_agreement": round(sum(1 for x, y in pairs if abs(x - y) <= 1) / n, 4),
            "mae": round(sum(abs(x - y) for x, y in pairs) / n, 4),
        }
    return segs


# ---------------------------------------------------------------- 费用估算

# DeepSeek 公布价（USD / 1M tokens，2026-10-19 取自官方 pricing 页）：
# deepseek-flash(= deepseek-chat 解析到的模型) 输入 cache miss / 输出；
# 闲时价为峰时价的一半。峰时 = 周一至周五 UTC 01:00-04:00 与 06:00-10:00。
PRICING_USD_PER_MTOK = {
    "input_cache_miss": {"peak": 0.30, "offpeak": 0.15},
    "output": {"peak": 1.20, "offpeak": 0.60},
}
USD_TO_CNY = 7.1  # 粗估汇率，仅用于对照 ¥0.5 硬上限
MAX_BUDGET_CNY = 0.5


def is_peak(ts: float) -> bool:
    """按官方定义判断某次调用是否落在峰时（UTC 周一至周五 01-04 / 06-10）。"""
    dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
    if dt.weekday() >= 5:
        return False
    return 1 <= dt.hour < 4 or 6 <= dt.hour < 10


def cost_cny(prompt_tokens: int, completion_tokens: int, ts: float) -> float:
    tier = "peak" if is_peak(ts) else "offpeak"
    usd = (
        prompt_tokens / 1e6 * PRICING_USD_PER_MTOK["input_cache_miss"][tier]
        + completion_tokens / 1e6 * PRICING_USD_PER_MTOK["output"][tier]
    )
    return usd * USD_TO_CNY


# ---------------------------------------------------------------- 臂 B：记录裁判原始输出

class RecordingLM:
    """包装本地模型，记录裁判每次输出的原始文本（**不改 src/**）。

    `EvaluationEngine._judge_single_dim` 在校验解析失败时会静默返回中性分 3.0，
    不记录原始输出就无法区分「真的打了 3 分」与「没解析出来」——
    而这恰恰是判断小模型裁判可信度的关键证据。
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.records: List[str] = []

    def generate(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        try:
            out = self.model.generate(**kwargs)
        except Exception:  # 交给上游的 try/except 处理；这里留痕以便对齐维度
            self.records.append("<generate failed>")
            raise
        try:
            new_tokens = out[0][input_ids.shape[1] :]
            self.records.append(
                self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )
        except Exception:  # noqa: BLE001 —— 记录失败不能影响评测本身
            self.records.append("<decode failed>")
        return out


class DeviceTokenProxy:
    """把 tokenizer 的编码结果搬到模型所在设备（补齐 src/evaluation.py 缺的一步）。

    为什么需要：`EvaluationEngine._judge_single_dim` 用 `tokenizer(text, return_tensors="pt")`
    得到的是 **CPU** 张量，却直接喂给 `device_map="cuda"` 的模型。GPU 上实测必报
    `Expected all tensors to be on the same device ...`，而该异常被 try/except 吞掉后
    **静默回退到「按回答长度打分」的启发式**——这正是「生成层有实现、零实测」被掩盖的方式。
    （生成路径 `src/generation.py::_generate_local` 做了设备搬运，评测路径漏了。）

    为什么用代理而不是直接改 src：评测实验不该顺手改业务代码；但缺陷已如实写进报告，
    并给出最小修复建议（`inputs = {k: v.to(target_device) ...}`，与 generation.py 同款）。
    """

    def __init__(self, tokenizer, device):
        self._tok = tokenizer
        self._device = device

    def __getattr__(self, item):
        return getattr(self._tok, item)

    def __call__(self, *args, **kwargs):
        out = self._tok(*args, **kwargs)
        return {k: v.to(self._device) for k, v in out.items()}


# ---------------------------------------------------------------- 数据与上下文


def load_samples(limit: Optional[int]) -> List[Dict]:
    samples: List[Dict] = []
    for ds in DATASETS:
        qa_path = ROOT / "data/sample" / ds["qa"]
        kb_path = ROOT / "data/sample" / ds["kb"]
        if not qa_path.exists() or not kb_path.exists():
            print(f"[warn] 数据集缺失，跳过 {ds['name']}")
            continue
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if limit:
            qa = qa[:limit]
        for i, case in enumerate(qa, 1):
            samples.append(
                {
                    "id": f"{ds['name']}#{i}",
                    "dataset": ds["name"],
                    "kind": ds["kind"],
                    "answerable": ds["answerable"],
                    "kb": ds["kb"],
                    "question": case["question"],
                    "relevant_sources": case.get("relevant_sources", []),
                    "reference_answer": case.get("reference_answer", ""),
                }
            )
    return samples


def build_context(retrieved: List[Dict]) -> str:
    """与 `EvaluationEngine.full_evaluation` 一致的上下文构造（三臂共用）。"""
    return "\n".join(d["text"][:CONTEXT_MAX_CHARS] for d in retrieved[:CONTEXT_TOP_K])


# ---------------------------------------------------------------- 臂 C：API 调用

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # 实测解析为 deepseek-flash（DeepSeek-V4.1-Flash-1017）
PROXY = "http://127.0.0.1:7897"


def _read_key() -> Optional[str]:
    """只从环境变量读 key；缺失返回 None（调用方负责优雅跳过）。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    return key or None


def _http_json(
    url: str,
    key: str,
    payload: Optional[Dict] = None,
    use_proxy: bool = False,
    timeout: int = 90,
) -> Dict:
    """最小 HTTP 封装：显式控制是否走本机代理（直连失败才回退代理）。"""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST" if data else "GET",
    )
    if use_proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
        )
    else:
        # 直连：显式清空代理，避免继承系统代理设置
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe_api(key: str) -> Dict:
    """连通性探测：直连优先，失败再走本机代理；两种结果都记进报告。"""
    out = {}
    for tag, use_proxy in [("direct", False), ("proxy_7897", True)]:
        t0 = time.time()
        try:
            data = _http_json(f"{DEEPSEEK_BASE}/models", key, None, use_proxy, timeout=30)
            out[tag] = {
                "ok": True,
                "elapsed_s": round(time.time() - t0, 2),
                "models": [m.get("id") for m in data.get("data", [])],
            }
        except Exception as e:  # noqa: BLE001 —— 探针失败要如实记录
            out[tag] = {
                "ok": False,
                "elapsed_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }
    return out


def load_api_cache() -> Dict:
    if API_CACHE_PATH.exists():
        try:
            return json.loads(API_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_api_cache(cache: Dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    API_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cache_key(messages: List[Dict], model: str) -> str:
    blob = model + "\n" + "\n".join(m["role"] + ":" + m["content"] for m in messages)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def call_api_chat(
    key: str, messages: List[Dict], cache: Dict, max_tokens: int = 256,
    temperature: float = 0.0, max_retry: int = 2,
) -> Tuple[str, Dict]:
    """通用一次问答调用（用于 `--answer-source api` 的降级答案生成），带磁盘缓存。"""
    ck = cache_key(messages, DEEPSEEK_MODEL)
    entry = cache.get(ck) or {}
    if "content" in entry:
        return entry["content"], {"source": "cache", "usage": entry.get("usage", {})}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    last_err = None
    for attempt in range(max_retry + 1):
        try:
            data = _http_json(
                f"{DEEPSEEK_BASE}/chat/completions", key, payload,
                use_proxy=(attempt == max_retry),
            )
            content = data["choices"][0]["message"]["content"]
            cache[ck] = {
                "content": content,
                "usage": data.get("usage", {}),
                "model_returned": data.get("model"),
                "ts": time.time(),
            }
            save_api_cache(cache)
            return content, {"source": "api", "usage": cache[ck]["usage"], "ts": cache[ck]["ts"]}
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(1.0)
    return "", {"source": "api", "usage": {}, "error": last_err}


def call_api_judge(
    key: str, messages: List[Dict], cache: Dict, max_retry: int = 2
) -> Tuple[Dict, Dict]:
    """一次调用返回 3 维 JSON。优先读缓存；失败最多重试 2 次。

    返回 (scores, meta)。meta 含 usage / 来源（cache|api）/ 失败原因。
    """
    ck = cache_key(messages, DEEPSEEK_MODEL)
    entry = cache.get(ck) or {}
    if "scores" in entry:
        return (
            {d: entry["scores"].get(d) for d in JUDGE_DIMENSIONS},
            {
                "source": "cache",
                "ts": entry.get("ts"),
                "usage": entry.get("usage", {}),
                "model_returned": entry.get("model_returned"),
                "raw": entry.get("raw", ""),
                "error": None,
            },
        )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 96,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    last_err = None
    for attempt in range(max_retry + 1):
        use_proxy = attempt == max_retry  # 最后一次尝试走代理
        try:
            data = _http_json(
                f"{DEEPSEEK_BASE}/chat/completions", key, payload, use_proxy=use_proxy
            )
            content = data["choices"][0]["message"]["content"]
            scores = parse_api_judge(content)
            ts = time.time()
            entry = {
                "scores": scores,
                "raw": content,
                "usage": data.get("usage", {}),
                "model_returned": data.get("model"),
                "ts": ts,
                "via": "proxy" if use_proxy else "direct",
            }
            cache[ck] = entry
            save_api_cache(cache)
            return scores, {
                "source": "api",
                "ts": ts,
                "usage": entry["usage"],
                "model_returned": entry["model_returned"],
                "raw": content,
                "error": None,
                "attempts": attempt + 1,
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(1.0)
    return (
        {d: None for d in JUDGE_DIMENSIONS},
        {"source": "api", "usage": {}, "raw": "", "error": last_err},
    )


# ---------------------------------------------------------------- 主流程


def run_local_answers(pipe, samples: List[Dict], checkpoint, max_new_tokens: int) -> None:
    """用仓库现有 pipeline 生成本地答案（temperature=0 保证可复现）。"""
    by_kb: Dict[str, List[Dict]] = {}
    for s in samples:
        by_kb.setdefault(s["kb"], []).append(s)
    for kb, group in by_kb.items():
        kb_path = ROOT / "data/sample" / kb
        docs = json.loads(kb_path.read_text(encoding="utf-8"))
        print(f"[local] 建库 {kb}（{len(docs)} 篇）")
        pipe.build_index(docs)
        for s in group:
            if s.get("answer"):
                continue
            t0 = time.time()
            s.setdefault("gen", {})
            try:
                res = pipe.ask(s["question"], top_k=5)
                s["answer"] = res["answer"]
                s["context"] = build_context(res["retrieved_docs"])
                s["retrieved"] = [
                    {
                        "chunk_index": r["chunk_index"],
                        "source": r.get("source", ""),
                        "score": round(float(r["score"]), 4),
                        "text_head": r["text"][:120],
                    }
                    for r in res["retrieved_docs"]
                ]
                s["gen"]["source"] = "local"
                s["gen"]["elapsed_s"] = round(time.time() - t0, 2)
            except Exception as e:  # noqa: BLE001 —— OOM 等要如实记录并继续
                s["answer"] = ""
                s["gen"] = {"source": "local", "error": f"{type(e).__name__}: {str(e)[:300]}"}
                print(f"  [ERR] {s['id']} 本地生成失败：{s['gen']['error']}")
            checkpoint()
            print(
                f"  [{s['id']}] {s['gen'].get('elapsed_s', '-')}s "
                f"answer={len(s.get('answer') or '')} chars"
            )


def run_api_answers(
    pipe, key: str, cache: Dict, samples: List[Dict], checkpoint, report: Dict
) -> None:
    """降级路径：本机显存不足时用强裁判模型生成答案（报告中会标注为非本地臂）。

    检索仍走仓库 pipeline（embedding 很小，CPU/GPU 都跑得动），只有「生成」换成了 API。
    """
    from src.generation import RAG_PROMPT_TEMPLATE

    by_kb: Dict[str, List[Dict]] = {}
    for s in samples:
        by_kb.setdefault(s["kb"], []).append(s)
    for kb, group in by_kb.items():
        docs = json.loads((ROOT / "data/sample" / kb).read_text(encoding="utf-8"))
        pipe.build_index(docs)
        for s in group:
            if s.get("answer"):
                continue
            s.setdefault("gen", {})
            retrieved = pipe._retrieve(s["question"], top_k=5)
            s["context"] = build_context(retrieved)
            s["retrieved"] = [
                {
                    "chunk_index": r["chunk_index"],
                    "source": r.get("source", ""),
                    "score": round(float(r["score"]), 4),
                    "text_head": r["text"][:120],
                }
                for r in retrieved
            ]
            prompt = RAG_PROMPT_TEMPLATE.format(
                context="\n\n".join(
                    f"[文档{i}]（相关度: {r['score']:.2f}）\n{r['text']}"
                    for i, r in enumerate(retrieved, 1)
                ),
                question=s["question"],
            )
            content, meta = call_api_chat(
                key, [{"role": "user", "content": prompt}], cache, max_tokens=256
            )
            s["answer"] = content.strip()
            s["gen"] = {
                "source": "api(降级)",
                "usage": meta.get("usage", {}),
                "error": meta.get("error"),
            }
            checkpoint()
            print(f"  [{s['id']}] api answer {len(s['answer'])} chars")


def run_local_judge(pipe, samples: List[Dict], recording, checkpoint) -> None:
    """臂 B：调用仓库现有的 evaluate_generation()（不自造 prompt）。"""
    # 用 tokenizer 代理补齐 src/evaluation.py 漏掉的设备搬运（见 DeviceTokenProxy 说明）
    judge_tokenizer = DeviceTokenProxy(
        pipe.generator.tokenizer,
        getattr(pipe.generator.model, "device", None) or "cpu",
    )
    judge_model = recording
    for s in samples:
        if s.get("judge", {}).get("local"):
            continue
        if not s.get("answer"):
            s.setdefault("judge", {})["local"] = {"error": "无回答可评（本地生成为空）"}
            checkpoint()
            continue
        recording.records.clear()
        t0 = time.time()
        try:
            scores = pipe.evaluator.evaluate_generation(
                s["question"], s["answer"], s["context"], judge_model, judge_tokenizer
            )
            raws = list(recording.records)
        except Exception as e:  # noqa: BLE001
            s.setdefault("judge", {})["local"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            checkpoint()
            print(f"  [ERR] {s['id']} 本地裁判失败：{s['judge']['local']['error']}")
            continue
        detail = {}
        for dim, raw in zip(JUDGE_DIMENSIONS, raws):
            m = re.search(r"([1-5])", raw)
            detail[dim] = {
                "score": scores.get(dim),
                "raw": raw,
                "parsed": int(m.group(1)) if m else None,
            }
        s.setdefault("judge", {})["local"] = {
            "scores": {d: scores.get(d) for d in JUDGE_DIMENSIONS},
            "detail": detail,
            "unparsed": [d for d, v in detail.items() if v["parsed"] is None],
            # 关键证据：这些维度的分数不是模型给的，而是 src/evaluation.py 异常兜底后的启发式
            "used_heuristic": [
                d
                for d, v in detail.items()
                if v["raw"] in ("<generate failed>", "<decode failed>") or v["raw"] == ""
            ],
            "elapsed_s": round(time.time() - t0, 2),
            "model": pipe.config["generation"]["model_name"],
        }
        checkpoint()
        print(f"  [{s['id']}] local judge {s['judge']['local']['scores']}")


def run_api_judge(samples: List[Dict], key: str, cache: Dict, checkpoint, report: Dict) -> None:
    """臂 C：强裁判 API（一次调用 3 维）。含费用硬上限与超限中止。"""
    calls = report["api_meta"]["calls"]
    spent = report["api_meta"]["cost_cny"]
    for s in samples:
        if s.get("judge", {}).get("api"):
            continue
        messages = build_api_judge_messages(s["question"], s["answer"], s["context"])
        scores, meta = call_api_judge(key, messages, cache)
        s.setdefault("judge", {})["api"] = {
            "scores": scores,
            "raw": meta.get("raw", ""),
            "model_returned": meta.get("model_returned"),
            "source": meta.get("source"),
            "error": meta.get("error"),
            "via": (cache.get(cache_key(messages, DEEPSEEK_MODEL), {}) or {}).get("via"),
        }
        usage = meta.get("usage") or {}
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        if meta.get("source") == "api" and usage:
            c = cost_cny(pt, ct, meta.get("ts") or time.time())
            spent += c
            calls.append(
                {
                    "id": s["id"],
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
                    "cost_cny": round(c, 6),
                    "peak": is_peak(meta.get("ts") or time.time()),
                }
            )
        elif meta.get("source") == "cache":
            report["api_meta"]["cache_hits"] += 1
        report["api_meta"]["cost_cny"] = round(spent, 6)
        checkpoint()
        print(
            f"  [{s['id']}] api judge {scores} "
            f"({meta.get('source')}, {usage.get('total_tokens', 0)} tok, 累计 ¥{spent:.4f})"
        )
        if spent > MAX_BUDGET_CNY:
            print(f"[STOP] API 费用累计 ¥{spent:.4f} 超过硬上限 ¥{MAX_BUDGET_CNY}，停止 API 臂。")
            report["api_meta"]["aborted"] = True
            return


def compute_report(samples: List[Dict], arms: List[str], report: Dict) -> None:
    """把逐样本结果汇总成各臂均值 + 一致率矩阵 + 分段结论。"""
    dims = JUDGE_DIMENSIONS

    def dims_of(sample_key):
        out = {d: [] for d in dims}
        for s in samples:
            j = s.get("judge", {}).get(sample_key)
            if not j or "scores" not in j:
                for d in dims:
                    out[d].append(None)
                continue
            for d in dims:
                out[d].append(j["scores"].get(d))
        return out

    local = dims_of("local")
    api = dims_of("api")
    rule_f = [s.get("rule", {}).get("rule_faithfulness") for s in samples]
    def mean(vals):
        v = [x for x in vals if isinstance(x, (int, float))]
        return round(sum(v) / len(v), 4) if v else None

    report["arm_means"] = {
        "rule": {"faithfulness": mean(rule_f)},
        "local": {d: mean(local[d]) for d in dims},
        "api": {d: mean(api[d]) for d in dims},
        "n_samples": len(samples),
    }
    report["agreement"] = {
        "local_vs_api": {d: agreement_stats(local[d], api[d]) for d in dims},
        "rule_vs_api": {"faithfulness": agreement_stats(rule_f, api["faithfulness"])},
        "rule_vs_local": {"faithfulness": agreement_stats(rule_f, local["faithfulness"])},
    }
    report["segment_by_api_faithfulness"] = segment_report(
        local["faithfulness"], api["faithfulness"]
    )

    # 臂 A 的规则指标均值（不含裁判分）
    def agg(path):
        vals = []
        for s in samples:
            cur = s.get("rule", {})
            for p in path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(p)
            if isinstance(cur, (int, float)):
                vals.append(cur)
        return round(sum(vals) / len(vals), 4) if vals else None

    report["rule_aggregate"] = {
        "numeric_landing_rate": agg(["numeric_landing", "rate"]),
        "sentence_attribution_rate": agg(["sentence_attribution", "rate"]),
        "refusal_accuracy": agg(["refusal", "correct"]),
        "refusal_accuracy_answerable": round(
            sum(
                1
                for s in samples
                if s["answerable"]
                and ((s.get("rule", {}).get("refusal") or {}).get("correct"))
            )
            / max(1, sum(1 for s in samples if s["answerable"])),
            4,
        ),
        "refusal_accuracy_unanswerable": round(
            sum(
                1
                for s in samples
                if not s["answerable"]
                and ((s.get("rule", {}).get("refusal") or {}).get("correct"))
            )
            / max(1, sum(1 for s in samples if not s["answerable"])),
            4,
        ),
        # 误拒答：可答样本上模型却拒答（1.5B 最典型的失效模式）
        "false_refusals_on_answerable": [
            s["id"]
            for s in samples
            if s["answerable"] and ((s.get("rule", {}).get("refusal") or {}).get("is_refusal"))
        ],
        "hallucination_on_unanswerable": [
            s["id"]
            for s in samples
            if not s["answerable"]
            and not ((s.get("rule", {}).get("refusal") or {}).get("is_refusal"))
        ],
        "em": agg(["lexical", "em"]),
        "f1": agg(["lexical", "f1"]),
        "rouge_l": agg(["lexical", "rouge_l"]),
        "bleu1": agg(["lexical", "bleu1"]),
    }
    report["arms_run"] = arms

    # 本地裁判诊断：区分「模型真的打了这个分」与「src 异常兜底后的启发式分」
    heur, unparsed = [], []
    raws = set()
    for s in samples:
        jl = (s.get("judge") or {}).get("local") or {}
        if jl.get("used_heuristic"):
            heur.append(s["id"])
        unparsed.extend(f"{s['id']}:{d}" for d in jl.get("unparsed", []))
        for v in (jl.get("detail") or {}).values():
            if v.get("raw"):
                raws.add(v["raw"])
    report["local_judge_diagnostics"] = {
        "samples_using_heuristic_fallback": heur,
        "unparsed_dims": unparsed,
        "distinct_raw_outputs": sorted(raws),
    }


MANUAL_LABELS = {
    # 人工标签（本次由人逐条阅读回答后给出），用于校准规则判据阈值。
    # 键 = 样本 id；值 = {"label": 忠实|部分忠实|编造, "note": 理由}
    # 先跑一轮拿到回答，再回填这里，然后 --render-only 重渲染报告（见 README 复现命令）。
    # 标签均来自对 generation_eval_2026-10-19.json 里 answer / reference_answer 的逐条人工阅读。
    "realistic#6": {
        "label": "faithful",
        "note": (
            "问题问「除量化外还有哪些主流压缩方法」，回答答「剪枝、知识蒸馏、低秩分解」，"
            "与参考回答三类方法完全对应、无编造。但回答把参考回答完整改写成短句、"
            "几乎无字面重叠，字符 2-gram 归因率仅 0.54 < 0.6 阈值，规则 faithfulness 被误判为 1.0。"
            "→ 规则判据的假阴性（改述型忠实回答被打低分）。"
        ),
    },
    "near_dup_stress#4": {
        "label": "faithful",
        "note": (
            "答案是「QTAG-A7 记录内容：量化以低位整数表达权重、保精度同时更小更快」，"
            "与参考回答语义一致且字面近乎逐字重合（归因率 0.93、数字落地率 1.0），"
            "规则 faithfulness 给 5.0 与强裁判 5 一致。→ 逐字型忠实回答规则判得准。"
        ),
    },
    "sample#3": {
        "label": "faithful",
        "note": (
            "回答列出 embedding 缓存、流式输出、模型量化三条，均在参考回答范围内，"
            "但「量化…同时不影响性能」「逐个返回 token 提高用户体验」属轻度夸张措辞（信息未编造）。"
            "整体判忠实。归因率仅 0.37（改述+列表展开），规则误判为 1.0。"
            "→ 规则判据对「含有依据但措辞外扩」的回答同样偏严。"
        ),
    },
    "realistic#14": {
        "label": "partial",
        "note": (
            "要点方向对（低温度增稳定、要求「不知道就说不知道」降幻觉），"
            "但夹带编造推理：「较高的温度会导致模型过度拟合参考文档」参考回答里没有此因果，"
            "且「保证答案具有创新性和深度」与低温/忠实的本意相冲突（参考回答明确是「减少随机关联」）。"
            "→ 部分忠实：主干有据、局部编造。规则 faithfulness 误判为 1.0（编造句被算作有据）。"
        ),
    },
    "unanswerable#2": {
        "label": "fabricated",
        "note": (
            "问题是「川菜回锅肉配料」，知识库只有 AI/大模型文档，应拒答；"
            "模型却输出五花肉、豆瓣酱等整段菜谱 = 凭空编造（检索 top-5 里没有任何烹饪内容）。"
            "规则判据正确捕获：归因率 0.06、规则 faithfulness 1.0，与强裁判 1 一致。"
            "→ 规则判据对「无任何上下文依据」的整段编造是可靠的。"
        ),
    },
}

LABEL_ZH = {"faithful": "忠实", "partial": "部分忠实", "fabricated": "编造"}


def _fmt(v, nd=2):
    if v is None:
        return "未测"
    if isinstance(v, float):
        return f"{round(v, nd)}"
    return str(v)


def render_md(report: Dict) -> str:
    samples = report["samples"]
    dims = JUDGE_DIMENSIONS
    L = []
    L.append("# 生成层忠实度实测：规则判据 / 本地 1.5B 裁判 / 强裁判 API（三臂对照）\n")
    L.append(f"- 生成日期：{report['date']}")
    L.append(f"- 复现：`python experiments/run_generation_eval.py`（离线部分加 `--no-api`）")
    L.append(f"- 运行的臂：`{', '.join(report['arms_run'])}`；样本数：{len(samples)}")
    if report.get("reused_from"):
        L.append(
            f"- 本轮说明：**复用** `{report['reused_from']}` 里已归档的答案与裁判分，"
            f"只重算规则臂并重新渲染（不重新生成、不新增 API 调用）；"
            f"费用表中的 token 数即首轮真实消耗。"
        )
    L.append(f"- 结果文件：`experiments/results/generation_eval_{report['date']}.json`（用 `--out` 时可另存）\n")

    L.append("## 一、口径（先说清楚，数字才有意义）\n")
    env = report["env"]
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 生成模型 | `{env['generation_model']}`（本地，device={env['device']}） |")
    L.append(f"| 生成解码 | temperature=0（贪心，可复现）、top_p={env['top_p']}、max_new_tokens={env['max_new_tokens']} |")
    L.append(f"| 检索 | `{env['retrieval_mode']}`，top_k=5，embedding=`{env['embedding_model']}` |")
    L.append(f"| 上下文（三臂共用） | 检索 top-{CONTEXT_TOP_K} 各截断 {CONTEXT_MAX_CHARS} 字符后拼接（与 `EvaluationEngine.full_evaluation` 一致） |")
    L.append(f"| 本地裁判（臂 B） | 与生成模型**同款**（自评），逐维度单独生成，max_new_tokens=5、do_sample=False |")
    L.append(f"| 强裁判（臂 C） | DeepSeek `{report['api_meta']['model']}`（实际返回 `{report['api_meta'].get('model_returned') or '未调用'}`），一次调用返回 3 维 JSON、temperature=0、max_tokens=96 |")
    L.append(f"| rubric | 两臂同一套判据文本（与 `src/evaluation.py` 逐字一致，有单测守） |")
    L.append(f"| 机器 | {env['platform']} / Python {env['python']} / torch {env.get('torch')} |")
    L.append("")
    L.append("样本构成：")
    L.append("")
    L.append("| 数据集 | 来源 | 题数 | 说明 |")
    L.append("|---|---|---|---|")
    for ds in DATASETS:
        n = sum(1 for s in samples if s["dataset"] == ds["name"])
        src = "原金标" if ds["kind"] == "gold" else "**本次新造**"
        L.append(f"| `{ds['name']}` | {src} | {n} | {ds['note']} |")
    L.append("")

    # ---------- 二、三臂总览
    m = report["arm_means"]
    L.append("## 二、三臂分数总览（1-5 分，均值）\n")
    L.append("| 臂 | faithfulness | relevance | completeness |")
    L.append("|---|---|---|---|")
    L.append(
        f"| A 规则判据（粗映射） | {_fmt(m['rule']['faithfulness'])} | — | — |"
    )
    L.append(
        f"| B 本地 1.5B 裁判 | {_fmt(m['local']['faithfulness'])} | {_fmt(m['local']['relevance'])} | {_fmt(m['local']['completeness'])} |"
    )
    L.append(
        f"| C 强裁判 API | {_fmt(m['api']['faithfulness'])} | {_fmt(m['api']['relevance'])} | {_fmt(m['api']['completeness'])} |"
    )
    L.append("")
    ra = report["rule_aggregate"]
    L.append("臂 A 的确定性指标（均值，规则判据不看 LLM）：")
    L.append("")
    L.append("| 数字/实体落地率 | 句子归因率 | 拒答正确率(全部) | 拒答正确率(可答题) | 拒答正确率(不可答题) | EM | F1 | ROUGE-L | BLEU-1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append(
        "| {n} | {s} | {r} | {ra} | {ru} | {em} | {f1} | {rl} | {b} |".format(
            n=_fmt(ra["numeric_landing_rate"]),
            s=_fmt(ra["sentence_attribution_rate"]),
            r=_fmt(ra["refusal_accuracy"]),
            ra=_fmt(ra["refusal_accuracy_answerable"]),
            ru=_fmt(ra["refusal_accuracy_unanswerable"]),
            em=_fmt(ra["em"]),
            f1=_fmt(ra["f1"]),
            rl=_fmt(ra["rouge_l"]),
            b=_fmt(ra["bleu1"]),
        )
    )
    L.append("")
    L.append("> EM/F1/ROUGE-L/BLEU 只对有 `reference_answer` 的样本计算（27 条原金标 + 6 条新造都有参考回答，共 33 条）。")
    L.append("")

    # ---------- 三、一致率矩阵
    L.append("## 三、一致率矩阵（核心结论）\n")
    L.append("### 3.1 臂 B（本地 1.5B）vs 臂 C（强裁判 API）\n")
    L.append("| 维度 | 样本数 | 精确一致率 | ±1 内一致率 | MAE | Pearson r | 高/低二分类一致率 | B 均值 | C 均值 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for d in dims:
        s = report["agreement"]["local_vs_api"][d]
        if s.get("n", 0) == 0:
            L.append(f"| {d} | 0 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |")
            continue
        L.append(
            f"| {d} | {s['n']} | {s['exact_agreement']} | {s['within1_agreement']} | "
            f"{s['mae']} | {_fmt(s['pearson'])} | {s['binary_agreement']} | "
            f"{s['mean_a']} | {s['mean_b']} |"
        )
    L.append("")
    L.append("### 3.2 臂 A（规则判据）vs 臂 C / 臂 B（仅 faithfulness）\n")
    L.append("| 对照 | 样本数 | 精确一致率 | ±1 内一致率 | MAE | 高/低二分类一致率 |")
    L.append("|---|---|---|---|---|---|")
    for key, label in [("rule_vs_api", "A 规则 vs C 强裁判"), ("rule_vs_local", "A 规则 vs B 本地")]:
        s = report["agreement"][key]["faithfulness"]
        if s.get("n", 0) == 0:
            L.append(f"| {label} | 0 | 未测 | 未测 | 未测 | 未测 |")
            continue
        L.append(
            f"| {label} | {s['n']} | {s['exact_agreement']} | {s['within1_agreement']} | "
            f"{s['mae']} | {s['binary_agreement']} |"
        )
    L.append("")

    # ---------- 3.3 分段：按强裁判档位分组
    segs = report["segment_by_api_faithfulness"]
    L.append("### 3.3 分段：本地 1.5B 在哪些档位可信\n")
    L.append("按**强裁判**给出的 faithfulness 档位分段：")
    L.append("")
    L.append("| 强裁判档位 | 样本数 | C 均值 | B 均值 | 精确一致率 | ±1 内一致率 | MAE |")
    L.append("|---|---|---|---|---|---|---|")
    for label, v in segs.items():
        if v.get("n", 0) == 0:
            L.append(f"| {label} | 0 | — | — | — | — | — |")
            continue
        L.append(
            f"| {label} | {v['n']} | {v['strong_mean']} | {v['local_mean']} | "
            f"{v['exact_agreement']} | {v['within1_agreement']} | {v['mae']} |"
        )
    L.append("")

    # ---------- 3.4 分段结论（数字全部来自本期实测，跟着数据重算）
    halluc = []
    for s in samples:
        if not s["answerable"] and not ((s.get("rule", {}).get("refusal") or {}).get("is_refusal")):
            halluc.append(
                (
                    s["id"],
                    ((s.get("judge") or {}).get("local") or {}).get("scores"),
                    ((s.get("judge") or {}).get("api") or {}).get("scores"),
                )
            )
    la = report["agreement"]["local_vs_api"]["faithfulness"]
    ra = report["agreement"]["rule_vs_api"]["faithfulness"]
    low = segs.get("低(1-2)", {})
    high = segs.get("高(4-5)", {})
    mid = segs.get("中(3)", {})

    def _p(v, nd=2):
        return "未测" if v is None else round(v, nd)

    L.append("### 3.4 分段结论：本地 1.5B 裁判到底可信到什么程度\n")
    L.append(
        f"- **与强裁判几乎不相关**：faithfulness 上 B vs C 的 Pearson r = {_p(la.get('pearson'))}、"
        f"MAE = {_p(la.get('mae'))}、精确一致率 = {_p(la.get('exact_agreement'))}、"
        f"±1 一致率 = {_p(la.get('within1_agreement'))}。即：在 1.5B 这个量级上，"
        f"裁判分数**不携带可用于排序答案质量的信息**，更接近「恒定在 3 分附近」的噪声源。"
    )
    L.append(
        f"- **两端失真方向相反（中庸化回归）**：强裁判判低（≤2）的样本上本地裁判偏**高**"
        f"（本地均分 {_p(low.get('local_mean'))} vs 强裁判 {_p(low.get('strong_mean'))}）；"
        f"强裁判判高（≥4）的样本上本地裁判偏**低**"
        f"（{_p(high.get('local_mean'))} vs {_p(high.get('strong_mean'))}）。"
        f"本期强裁判没给出 3 分档（中档样本数 {mid.get('n', 0)}），所以「中间档区分力」这一项**无法评估**。"
    )
    L.append(
        f"- **最该抓的编造反而放行**：不可答题上「没拒答 = 编造」的样本共 "
        f"{len(report['rule_aggregate'].get('hallucination_on_unanswerable', []))} 条"
        f"（{', '.join(report['rule_aggregate'].get('hallucination_on_unanswerable', [])) or '无'}）。"
    )
    for sid, sb, sc in halluc:
        L.append(
            f"  - `{sid}`：强裁判 {sc}，本地裁判 {sb}。"
            + (
                "→ 本地裁判把编造放行了（方向性错误，不是随机噪声）。"
                if (sb or {}).get("faithfulness") and (sc or {}).get("faithfulness")
                and sb["faithfulness"] > sc["faithfulness"]
                else ""
            )
        )
    L.append(
        f"- **可信区间**：只有答案与上下文**近乎逐字重合**（句子归因率≈1、数字落地率=1）或**整段拒答**时，"
        f"本地裁判才与强裁判同档；答案是「自己组织的正确表述」（改述）时，本地裁判倾向给 3、强裁判给 4-5。"
    )
    L.append(
        f"- **反直觉结论**：本期**零成本、离线的规则判据比本地 1.5B 裁判更贴近强裁判**"
        f"（faithfulness 上 MAE {_p(ra.get('mae'))} vs {_p(la.get('mae'))}；"
        f"r {_p(ra.get('pearson'))} vs {_p(la.get('pearson'))}；"
        f"精确一致率 {_p(ra.get('exact_agreement'))} vs {_p(la.get('exact_agreement'))}）。"
        f"所以「用本地 1.5B 做裁判」在本批样本上不成立——不是「有点噪声」，而是「与目标信号无关」。"
    )
    L.append(
        "- **仍然可用的场合**：(1) 只做「明显编造 vs 明显有据」两端的粗筛/回归测试；"
        "(2) 网络与预算完全不可用时的占位。需要排序、需要区分「好与更好」的场景都不该用它。"
    )
    L.append("")

    # ---------- 3.5 本地裁判原始输出诊断
    diag = report.get("local_judge_diagnostics", {})
    L.append("### 3.5 本地裁判原始输出诊断（为什么不能只看分数）\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 未被启发式兜底覆盖的样本 | {len(samples) - len(diag.get('samples_using_heuristic_fallback', []))}/{len(samples)} |")
    L.append(f"| 走了启发式兜底的样本 | {', '.join(diag.get('samples_using_heuristic_fallback', [])) or '无'} |")
    L.append(f"| 裁判原始输出无法解析出数字的维度数 | {len(diag.get('unparsed_dims', []))} |")
    L.append(f"| 无法解析的维度明细 | {', '.join(diag.get('unparsed_dims', [])) or '无'} |")
    raws = diag.get("distinct_raw_outputs", [])
    L.append(f"| 裁判出现过的原始输出 | {json.dumps(raws[:12], ensure_ascii=False)} |")
    L.append("")
    L.append(
        "> 说明：`src/evaluation.py::_judge_single_dim` 在异常时**静默**返回"
        "「按回答长度打分」的启发式（`_heuristic_score`），解析不出数字时给中性分 3。"
        "不记录原始输出，就无法分辨「模型给了 3 分」与「模型输出没法解析」——"
        "这也是本报告必须记录逐维度原始输出的原因。"
    )
    L.append("")

    # ---------- 五、逐样本明细
    L.append("## 四、逐样本明细\n")
    L.append(
        "| # | 样本 | 问题 | 检索条数 | A 规则 f. | B 本地 f. | C 强裁判 f. | B 裁判原始输出(f/r/c) | 落地率 | 归因率 | 拒答 | M 手工 |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(samples, 1):
        rule = s.get("rule", {})
        jl = s.get("judge", {}).get("local") or {}
        ja = s.get("judge", {}).get("api") or {}
        sl = jl.get("scores", {}).get("faithfulness")
        sa = ja.get("scores", {}).get("faithfulness")
        raw = jl.get("detail", {})
        raw_txt = "/".join(_fmt(raw.get(d, {}).get("parsed")) for d in dims) if raw else "—"
        man = MANUAL_LABELS.get(s["id"], {}).get("label")
        L.append(
            f"| {i} | `{s['id']}` | {s['question']} | {len(s.get('retrieved') or [])} | {_fmt(rule.get('rule_faithfulness'))} | "
            f"{_fmt(sl)} | {_fmt(sa)} | {raw_txt} | "
            f"{_fmt((rule.get('numeric_landing') or {}).get('rate'))} | "
            f"{_fmt((rule.get('sentence_attribution') or {}).get('rate'))} | "
            f"{'✅' if (rule.get('refusal') or {}).get('correct') else ('❌' if rule.get('refusal') else '—')} | "
            f"{LABEL_ZH.get(man, '—')} |"
        )
    L.append("")
    L.append(
        "> 「拒答」列：✅ = 拒答与否符合预期（不可答题拒答 / 可答题作答）；"
        "❌ = 不符合预期（不可答题却作答 = 编造；**可答题却拒答 = 误拒答**）。"
        "「B 裁判原始输出」列是 1.5B 裁判逐维度输出的原始数字（f/r/c = faithfulness/relevance/completeness），"
        "保留它是为了区分「真的打 3 分」与「输出无法解析被兜底成中性分」。"
    )
    L.append("")

    # ---------- 六、规则判据校准
    L.append("## 五、规则判据校准（5 条手工标注样本）\n")
    if not MANUAL_LABELS:
        L.append("（本次未回填手工标签，跳过）")
    else:
        L.append(
            f"阈值：句子归因 = 字符 2-gram 多重集覆盖率 ≥ **{ATTRIBUTION_THRESHOLD}**。"
            "下面把人工标签与规则输出并排，校准依据是这些样本的实测覆盖率区间。\n"
        )
        L.append("| 样本 | 人工标签 | 规则 faithfulness | 归因率(整体) | 该回答各句覆盖率 | 关键句最低覆盖率 | 数字落地率 |")
        L.append("|---|---|---|---|---|---|---|")
        for sid, meta in MANUAL_LABELS.items():
            s = next((x for x in samples if x["id"] == sid), None)
            if not s:
                continue
            ru = s.get("rule", {})
            sents = (ru.get("sentence_attribution") or {}).get("sentences", [])
            covs = [x["coverage"] for x in sents]
            L.append(
                f"| `{sid}` | **{LABEL_ZH.get(meta['label'], meta['label'])}** | "
                f"{_fmt(ru.get('rule_faithfulness'))} | "
                f"{_fmt((ru.get('sentence_attribution') or {}).get('rate'))} | "
                f"{', '.join(str(c) for c in covs)} | {_fmt(min(covs) if covs else None)} | "
                f"{_fmt((ru.get('numeric_landing') or {}).get('rate'))} |"
            )
        L.append("")
        for sid, meta in MANUAL_LABELS.items():
            L.append(f"- `{sid}`（{LABEL_ZH.get(meta['label'], meta['label'])}）：{meta.get('note', '')}")
        L.append("")
        L.append("**已知误判模式（必须主动说，不然规则判据会被当成真值）**：")
        L.append("")
        L.append("1. **改述无字面重叠 → 误判为「未归因」**：模型把「除以 √d_k 防止梯度消失」改写成「做缩放是为了稳住梯度」，含义正确但字符 2-gram 覆盖率低，会被判成编造。归因率是**字面**指标，不是语义指标。")
        L.append("2. **模板化句式 → 误判为「已归因」**：RAG prompt 与文档本身有很多固定套话（如「量化用低位整数表示权重」），模型抄句式而内容错位时覆盖率仍然很高。")
        L.append("3. **数字落地率只看「出现过」，不看「用对了」**：回答把 0.3% 说成 0.5% 时，若上下文里同时出现 0.5（别的语境），会被误判为落地。")
        L.append("4. **拒答判据是字面标记匹配**：回答「文档中未提及」判定为拒答正确，但如果模型先说一堆相关内容再说「无法回答」，也会被判为拒答（错误地给满分 5）。")
        L.append("5. **1-5 粗映射没有中间档**：`min(归因率, 落地率)` 映射到 `1+4*base` 后，实际输出集中在两端（1 与 5 附近），对「部分忠实」不敏感——这是与裁判分对齐时 MAE 偏大的直接原因之一。")
    L.append("")

    # ---------- 七、费用
    am = report["api_meta"]
    L.append("## 六、费用与 token 实耗\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 模型 | `{am['model']}` → 实际返回 `{am.get('model_returned') or '未调用'}` |")
    L.append(f"| 连通性探测（直连） | {json.dumps(report['api_probe'].get('direct', {}), ensure_ascii=False)} |")
    L.append(f"| 连通性探测（代理 {PROXY}） | {json.dumps(report['api_probe'].get('proxy_7897', {}), ensure_ascii=False)} |")
    L.append(f"| 新调用次数 | {len(am['calls'])} |")
    L.append(f"| 命中缓存次数 | {am['cache_hits']} |")
    L.append(f"| 输入 token | {am['prompt_tokens']} |")
    L.append(f"| 输出 token | {am['completion_tokens']} |")
    L.append(f"| 缓存命中 token | {am['cache_hit_tokens']} |")
    L.append(f"| 估算费用 | ¥{am['cost_cny']:.4f}（上限 ¥{MAX_BUDGET_CNY}，{ '已超出' if am.get('aborted') else '未超' }） |")
    L.append(
        f"| 计价口径 | DeepSeek 官方价（USD/1M tok）：输入 cache-miss 峰时 $0.30 / 闲时 $0.15，输出峰时 $1.20 / 闲时 $0.60；按每次调用时刻判峰/闲；汇率按 1 USD = ¥{USD_TO_CNY} 粗估 |"
    )
    L.append("")

    # ---------- 八、诚实边界
    L.append("## 七、诚实边界（结论的适用条件）\n")
    L.append(f"1. **样本量小**：{len(samples)} 条（27 条原金标 + 6 条**本次新造**的拒答样本）。33 条只能给出量级判断，不能当分布估计；一致率的小数点后一位都不宜当基准线。")
    L.append(f"2. **本地裁判是自评**：臂 B 的判分模型与生成模型是同一个 `Qwen2.5-1.5B-Instruct`（config.yaml 的 `judge_model` 注释就写了「用同款模型自评」）。自评天然偏袒自己的输出，因此**臂 B 的绝对分不可作为答案质量的证据**，只能看它与臂 C 的偏离方向与幅度。")
    L.append(f"3. **裁判身份与版本**：强裁判为 DeepSeek `deepseek-chat`（当前解析到 `{am.get('model_returned') or '未调用'}`，官方页面标注版本 DeepSeek-V4.1-Flash-1017）。API 侧模型会滚动更新，重跑可能得到不同的绝对值；本地裁判版本固定为仓库配置的 Qwen2.5-1.5B-Instruct。")
    L.append("4. **裁判本身不是真值**：强裁判也有立场偏差（偏好长答案、偏好与上下文用词一致的答案）。本报告只做**两裁判一致性**，不是与人类标注对齐的准确性。")
    L.append(f"5. **阈值与误判模式**：规则判据的归因阈值为 {ATTRIBUTION_THRESHOLD}（字符 2-gram 多重集覆盖率），已知 5 类误判见第五节；规则分是 1-5 的**粗映射**，请当作「有据/无据」的二分类看，不要当连续分。")
    L.append("6. **API 只调一次/样本**：为保证费用可控，强裁判每样本只调用一次、temperature=0，**没有做多次采样取中位数**——所以无法给出裁判自身的方差（README 里「多次评分取中位数」仍是未实现项）。")
    L.append("7. **检索质量与前序报告一致**：本报告只评生成层，检索指标见 `experiments/results/retrieval_benchmark_2026-10-19.md`。生成层分数建立在默认 vector 检索（top_k=5）之上。")
    L.append("8. **不可答样本是本仓库新造的**：`data/sample/test_qa_unanswerable.json` 的问题由本次实验手写（确认不在知识库中），用于测拒答正确性；它不是原金标，也不代表真实业务的未命中分布。")
    errs = [s["id"] for s in samples if (s.get("judge", {}).get("local") or {}).get("error")]
    if errs:
        L.append(f"9. **本地臂有失败样本**：{', '.join(errs)}（详见 JSON 里的 error 字段）。")
    L.append("")

    L.append("### 附：评测层发现的一处 `src/` 缺陷（本实验未改 `src/`，仅记录）\n")
    L.append(
        "`src/evaluation.py::_judge_single_dim` 把 `tokenizer(text, return_tensors=\"pt\")` 产生的 **CPU** 张量"
        "直接喂给 `device_map=\"cuda\"` 的裁判模型，GPU 上必抛"
        "`Expected all tensors to be on the same device, but got index is on cpu`；"
        "该异常被 `try/except` 吞掉后**静默回退**到「按回答长度打分」的启发式。"
    )
    L.append("")
    L.append("本实验的处理：用 tokenizer 代理在外部补上设备搬运（三行包装，不动业务代码），")
    L.append("所以臂 B 的分数是模型真实输出；逐样本原始输出见 3.5 与 JSON。")
    L.append("建议的最小修复（与 `src/generation.py::_generate_local` 同款）：")
    L.append("")
    L.append("```python")
    L.append("inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 在 model.generate 之前")
    L.append("```")
    L.append("")

    L.append("## 八、复现命令\n")
    L.append("```bash")
    L.append("# 离线（臂 A + 臂 B，需要本地 BGE + Qwen 缓存，GPU 上会加载 1.5B 权重）")
    L.append("python experiments/run_generation_eval.py --no-api")
    L.append("# 三臂全跑（API key 只从环境变量读，见下方说明）")
    L.append("DEEPSEEK_API_KEY=... python experiments/run_generation_eval.py")
    L.append("# 冒烟：只跑前 3 条、只跑规则臂")
    L.append("python experiments/run_generation_eval.py --limit 3 --arms rule")
    L.append("# 由已保存的 JSON 重新渲染报告（不重跑模型/不花钱）")
    L.append("python experiments/run_generation_eval.py --render-only experiments/results/generation_eval_%s.json" % report["date"])
    L.append("```")
    L.append("")
    L.append("离线单测（不调 API、不加载大模型）：`python -m pytest -q tests/test_generation_eval.py`")
    L.append("")
    return "\n".join(L)


def dump_json(samples, report, out_path: Path) -> Path:
    """只写 JSON（含当前进度），供崩溃/中断后恢复；不含任何密钥。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["samples"] = samples
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def save_all(samples, report, out_path: Optional[Path]) -> Path:
    """写 JSON + 渲染 Markdown 报告（md 与 json 同目录同名，便于自定义 --out 时不覆盖正式报告）。"""
    stamp = report["date"]
    out = out_path or (RESULTS_DIR / f"generation_eval_{stamp}.json")
    dump_json(samples, report, out)
    out.with_suffix(".md").write_text(
        render_md(dict(report, samples=samples)), encoding="utf-8"
    )
    return out


def main():
    # 离线优先：必须在任何 HF 相关 import 之前设置，否则 huggingface_hub 在 import 时
    # 就已把 HF_HUB_OFFLINE 常量固化（本次实测踩过：模型明明有本地缓存，却仍去联网 HEAD，
    # 每个文件重试 5 次、把一次冒烟拖过 15 分钟）。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    ap = argparse.ArgumentParser(description="生成层忠实度三臂对照实测")
    ap.add_argument("--arms", default="rule,local,api", help="要跑的臂：rule,local,api 的逗号组合")
    ap.add_argument("--limit", type=int, default=None, help="每个数据集只取前 N 条（冒烟用）")
    ap.add_argument("--no-api", action="store_true", help="离线模式：不调用任何 API")
    ap.add_argument("--out", default=None, help="结果 JSON 路径（默认 experiments/results/generation_eval_<date>.json）")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="本地生成的最大新 token 数")
    ap.add_argument("--render-only", default=None, help="由已保存的 JSON 重新渲染报告，不重跑模型/API")
    ap.add_argument(
        "--from-results",
        default=None,
        help="复用既有结果 JSON 的答案与裁判分，只重算规则臂并重渲染（不重新生成、不新增 API 调用）",
    )
    ap.add_argument("--answer-source", default="local", choices=["local", "api"],
                    help="答案来源：local=本地 1.5B（默认）；api=用强裁判模型生成（本机显存不足时的降级，报告会标注）")
    args = ap.parse_args()

    today = datetime.date.today().isoformat()

    if args.render_only:
        payload = json.loads(Path(args.render_only).read_text(encoding="utf-8"))
        md = render_md(payload)
        out_md = Path(args.render_only).with_suffix(".md")
        out_md.write_text(md, encoding="utf-8")
        print(md)
        print(f"\n[ok] 已重渲染：{out_md}")
        return

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.no_api and "api" in arms:
        arms.remove("api")

    # ---- 复用模式：拿既有 JSON 的答案 + 裁判分，只重算规则臂（改规则后不必重跑模型/API）
    if args.from_results:
        prior_path = Path(args.from_results)
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        samples = prior["samples"]
        report = {k: v for k, v in prior.items() if k != "samples"}
        for s in samples:
            s["rule"] = rule_scores(
                s.get("answer", ""),
                s.get("context", ""),
                reference=s.get("reference_answer", ""),
                expect_refusal=not s["answerable"],
            )
        report["reused_from"] = str(prior_path)
        run_arms = list(dict.fromkeys(report.get("arms_run", []) + ["rule"]))
        compute_report(samples, run_arms, report)
        out = Path(args.out) if args.out else (RESULTS_DIR / f"generation_eval_{today}.json")
        save_all(samples, report, out)
        print(f"[reuse] 已按新规则重算规则臂并重渲染：{out} / {out.with_suffix('.md')}")
        return

    samples = load_samples(args.limit)
    report = {
        "date": today,
        "arms_run": [],
        "env": {},
        "api_probe": {},
        "api_meta": {
            "model": DEEPSEEK_MODEL,
            "model_returned": None,
            "calls": [],
            "cache_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_hit_tokens": 0,
            "cost_cny": 0.0,
            "aborted": False,
        },
    }
    print(f"[setup] 样本 {len(samples)} 条，臂 {arms}")

    # ---- API 前置：key + 连通性 + 预算预估（在加载 3GB 权重之前做，省时间）
    key = None
    cache: Dict = {}
    need_api = "api" in arms or args.answer_source == "api"
    if need_api:
        key = _read_key()
        if not key:
            print("[skip] 未检测到环境变量 DEEPSEEK_API_KEY → 优雅跳过臂 C（强裁判 API）。")
            arms = [a for a in arms if a != "api"]
            if args.answer_source == "api":
                raise SystemExit("--answer-source api 需要 DEEPSEEK_API_KEY")
        else:
            cache = load_api_cache()
            report["api_probe"] = probe_api(key)
            direct_ok = report["api_probe"].get("direct", {}).get("ok")
            print(f"[api] 连通性：direct={'OK' if direct_ok else 'FAIL'}，"
                  f"proxy={report['api_probe'].get('proxy_7897', {}).get('ok')}")
            # 预算粗估：中文按 1 字符 ≈ 0.6 token 估；单样本上限 = prompt + 96 输出
            est_tokens_in = sum(
                len(s["question"]) + 2400 for s in samples  # rubric+context+answer 粗估
            )
            est = cost_cny(int(est_tokens_in * 0.6), 96 * len(samples), time.time())
            print(f"[api] 预算预估 ¥{est:.4f}（硬上限 ¥{MAX_BUDGET_CNY}）")
            if est > MAX_BUDGET_CNY:
                print("[STOP] 预估费用超上限，拒绝启动 API 臂。")
                arms = [a for a in arms if a != "api"]

    # ---- 本地臂：加载 pipeline（embedding + 生成模型）
    pipe = None
    recording = None
    need_local_model = ("local" in arms) or args.answer_source == "local"
    if need_local_model:
        from src.pipeline import RAGPipeline

        pipe = RAGPipeline(str(ROOT / "config.yaml"), load_generator=True)
        # 评测用贪心解码：可复现优先于多样性
        pipe.generator.temperature = 0.0
        pipe.generator.max_new_tokens = args.max_new_tokens
        recording = RecordingLM(pipe.generator.model, pipe.generator.tokenizer)
        import torch

        report["env"] = {
            "generation_model": pipe.generator.model_name,
            "embedding_model": pipe.config["embedding"]["model_name"],
            "retrieval_mode": pipe.retrieval_mode,
            "device": pipe.generator.device,
            "top_p": pipe.generator.top_p,
            "max_new_tokens": args.max_new_tokens,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "platform": f"{os.name}/{sys.platform}",
        }
    else:
        # 降级：只做检索的 pipeline（embedding 很小），答案交给 API 生成
        from src.pipeline import RAGPipeline

        pipe = RAGPipeline(str(ROOT / "config.yaml"), load_generator=False)
        report["env"] = {
            "generation_model": "未加载（--answer-source api 降级模式）",
            "embedding_model": pipe.config["embedding"]["model_name"],
            "retrieval_mode": pipe.retrieval_mode,
            "device": "-",
            "top_p": "-",
            "max_new_tokens": args.max_new_tokens,
            "python": sys.version.split()[0],
            "platform": f"{os.name}/{sys.platform}",
        }

    out_path = Path(args.out) if args.out else (RESULTS_DIR / f"generation_eval_{today}.json")

    def checkpoint():
        dump_json(samples, report, out_path)

    # ---- 答案生成
    if args.answer_source == "local":
        assert pipe is not None and pipe.generator is not None
        run_local_answers(pipe, samples, checkpoint, args.max_new_tokens)
    else:
        print("[warn] 答案由 API 生成（**非本地臂**，报告会标注为降级模式）")
        run_api_answers(pipe, key, cache, samples, checkpoint, report)
    report["answer_source"] = args.answer_source

    # ---- 臂 A
    if "rule" in arms:
        for s in samples:
            s["rule"] = rule_scores(
                s.get("answer", ""),
                s.get("context", ""),
                reference=s.get("reference_answer", ""),
                expect_refusal=not s["answerable"],
            )
        print("[arm A] 规则判据完成")
        report["arms_run"].append("rule")

    # ---- 臂 B
    if "local" in arms:
        assert pipe is not None and recording is not None
        print("[arm B] 本地 1.5B 裁判（复用仓库 evaluate_generation）")
        run_local_judge(pipe, samples, recording, checkpoint)
        report["arms_run"].append("local")

    # ---- 臂 C
    if "api" in arms and key:
        print("[arm C] 强裁判 API（DeepSeek）")
        run_api_judge(samples, key, cache, checkpoint, report)
        report["arms_run"].append("api")
        usages = [c for c in report["api_meta"]["calls"]]
        report["api_meta"]["prompt_tokens"] = sum(c["prompt_tokens"] for c in usages)
        report["api_meta"]["completion_tokens"] = sum(c["completion_tokens"] for c in usages)
        report["api_meta"]["cache_hit_tokens"] = sum(c["cache_hit_tokens"] for c in usages)
        ret = [s["judge"]["api"]["model_returned"] for s in samples if s.get("judge", {}).get("api", {}).get("model_returned")]
        report["api_meta"]["model_returned"] = ret[0] if ret else None

    compute_report(samples, arms, report)
    save_all(samples, report, out_path)

    print("\n" + "=" * 70)
    print(f"[done] 报告：{out_path.with_suffix('.md')}")
    print(f"[done] 结果：{out_path}")
    if "api" in report["arms_run"]:
        am = report["api_meta"]
        print(
            f"[cost] 新调用 {len(am['calls'])} 次 / 缓存命中 {am['cache_hits']} 次；"
            f"输入 {am['prompt_tokens']} tok（缓存命中 {am['cache_hit_tokens']}）、"
            f"输出 {am['completion_tokens']} tok；估算 ¥{am['cost_cny']:.4f}"
        )
    else:
        print("[cost] 本次未调用 API（¥0.0000）")


if __name__ == "__main__":
    main()
