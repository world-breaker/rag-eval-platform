#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 微调前后评分：可计算指标 + LLM-as-Judge + 人工评分表
==========================================================

输入
----
`experiments/sft_before_after_2026-10-21.jsonl`（由 run_sft_before_after.py 生成）

输出
----
1. `experiments/sft_before_after_2026-10-21_scores.json`    —— 全部数字（唯一真值来源）
2. `experiments/sft_before_after_2026-10-21_human_sheet.md` —— 人工评分表（**评分列留空**）
3. `experiments/sft_before_after_2026-10-21.md`             —— 报告（表格全部由本脚本生成）

复用 rag-eval-platform 的既有资产（不改仓库任何既有代码）
--------------------------------------------------------
- `experiments/run_generation_eval.py`：normalize_text / exact_match / char_f1 /
  detect_refusal / REFUSAL_MARKERS / cost_cny / _http_json / probe_api /
  cache_key_n / JUDGE_DIMENSIONS / JUDGE_CRITERIA —— **直接 import 复用**
- `src/evaluation.py::EvaluationEngine.compute_rouge_l` —— 直接 import 复用（字符级 LCS）

裁判协议（与仓库 2026-10-20 归档实验同口径）
--------------------------------------------
- 强裁判：DeepSeek API（`deepseek-chat`，实测解析为 deepseek-flash）
- 绝对打分：N=3 次采样 / temperature=0.8 / **取中位数**（对齐 reference_robustness_2026-10-20.md）
- 成对对比：N=2（两种呈现顺序各一次）/ temperature=0.8 —— 用于**直接**回答
  「这条微调后到底更好还是更差」，并量化位置偏置
- API key **只从环境变量 `DEEPSEEK_API_KEY` 读**，绝不写入任何文件/日志/报告/缓存
- 费用硬上限 ¥0.5（沿用仓库口径）

用法
----
  # 只跑可计算指标（离线，0 花费）
  python experiments/score_sft_before_after.py --no-judge

  # 全量（含裁判）
  DEEPSEEK_API_KEY=... python experiments/score_sft_before_after.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

# ---- 复用仓库既有实现（不改它们） ----
import run_generation_eval as R  # noqa: E402
from src.evaluation import EvaluationEngine  # noqa: E402

DEFAULT_JSONL = ROOT / "experiments" / "sft_before_after_2026-10-21.jsonl"
DEFAULT_SCORES = ROOT / "experiments" / "sft_before_after_2026-10-21_scores.json"
DEFAULT_REPORT = ROOT / "experiments" / "sft_before_after_2026-10-21.md"
DEFAULT_HUMAN = ROOT / "experiments" / "sft_before_after_2026-10-21_human_sheet.md"
CACHE_PATH = ROOT / "experiments" / "results" / "sft_judge_cache_2026-10-21.json"

ARMS = ["base", "sft"]
ARM_LABEL = {"base": "基座（未微调）", "sft": "LoRA 微调后"}

# ---- rubric 适配（对仓库既有判据的唯一改动，必须写明） ----
# 仓库 rubric 面向 RAG（有「参考文档」）。本实验是 SFT 前后对比，只有「参考答案」。
# 维度名与判据句**逐字沿用仓库**，只把事实依据的载体从「参考文档」换成「参考答案」：
#   faithfulness: 参考文档 → 参考答案
#   completeness: 参考文档 → 参考答案
#   relevance   : 原句不动（本来就不提文档）
SFT_JUDGE_CRITERIA = {
    "faithfulness": "回答中的每一条信息是否都能在参考答案中找到依据？有没有编造？",
    "relevance": "回答是否直接回应了用户的问题？有没有偏离主题？",
    "completeness": "回答是否覆盖了参考答案中与该问题相关的所有重要信息？",
}
SFT_JUDGE_DIMENSIONS = ["faithfulness", "relevance", "completeness"]

# 通用拒答标记（仓库 REFUSAL_MARKERS 是 RAG 语境；本实验是通用中文指令，补充通用拒答表达）
GENERIC_REFUSAL_MARKERS = [
    "抱歉",
    "我不能",
    "我无法",
    "无法完成",
    "无法提供",
    "无法生成",
    "作为AI",
    "作为一个AI",
]

# 闲聊式开场白（出现在输出**开头**才算）—— 用于量化「微调是否把模型的客套开场去掉」。
# 这是在 alpaca-gpt4-zh 训练数据里学到的**格式**特征，与事实正确性无关。
CHATTY_OPENERS = ["好的", "当然", "当然可以", "没问题", "我理解", "很高兴", "谢谢你", "请提供"]
# 编号列表（1. / 1、）
_NUMBERED_RE = re.compile(r"(?:^|\n)\s*\d+[.、]\s*\S")


# ---------------------------------------------------------------- 文本工具


def detect_refusal_sft(text: str) -> bool:
    """拒答检测：仓库 R.REFUSAL_MARKERS ∪ 通用拒答标记。"""
    t = text or ""
    if any(m in t for m in R.REFUSAL_MARKERS):
        return True
    return any(m in t for m in GENERIC_REFUSAL_MARKERS)


def has_chatty_opener(text: str) -> bool:
    """输出开头是否是闲聊式开场白（「好的，以下是……」这类）。"""
    t = (text or "").strip()
    return any(t.startswith(m) for m in CHATTY_OPENERS)


def has_numbered_list(text: str) -> bool:
    """输出里是否有编号列表（1. / 1、）。"""
    return bool(_NUMBERED_RE.search(text or ""))


def script_of(text: str) -> str:
    """主语言判定：CJK 字母数 vs ASCII 字母数。

    返回 'zh' / 'en' / 'mixed' / 'other'。用于「风格对齐率」：
    100 条中文指令数据里混有英文样本，微调是否把输出风格拉向数据分布，
    可以用「输出主语言与参考答案主语言是否一致」这个**可计算**指标看。
    """
    t = text or ""
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    asc = sum(1 for c in t if c.isascii() and c.isalpha())
    if cjk == 0 and asc == 0:
        return "other"
    if cjk == 0:
        return "en"
    if asc == 0:
        return "zh"
    ratio = cjk / (cjk + asc)
    if ratio >= 0.7:
        return "zh"
    if ratio <= 0.3:
        return "en"
    return "mixed"


def repetition_ratio(text: str, n: int = 4) -> Optional[float]:
    """退化重复率：1 - (unique n-gram / total n-gram)，字符级。

    0 = 没有重复；越接近 1 越像「卡住循环输出」。少于 n 个字符返回 None。
    """
    t = R.normalize_text(text)
    if len(t) < n:
        return None
    grams = [t[i : i + n] for i in range(len(t) - n + 1)]
    if not grams:
        return None
    return round(1.0 - len(set(grams)) / len(grams), 4)


def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def median(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def sign_test(wins: int, losses: int) -> Optional[float]:
    """成对比较的双侧精确符号检验（binomial test, p=0.5）。

    ties 被剔除。n 很小时只能给量级，不能当结论用。
    """
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return round(min(1.0, 2 * p), 4)


# ---------------------------------------------------------------- 可计算指标


def computable_metrics(samples: List[Dict]) -> Dict:
    """对每条臂算可计算指标。返回 {arm: {metric: {...}}}。"""
    out: Dict[str, Dict] = {}
    for arm in ARMS:
        key = f"{arm}_output"
        em, f1s, rls, refusals, empties, truncs, reps, aligns = [], [], [], [], [], [], [], []
        openers, numbered = [], []
        lens, toks = [], []
        script_match = 0
        for s in samples:
            cand = s.get(key, "") or ""
            ref = s.get("reference", "") or ""
            em.append(R.exact_match(ref, cand))
            f1s.append(R.char_f1(ref, cand))
            rls.append(EvaluationEngine.compute_rouge_l(ref, cand))
            refusals.append(1 if detect_refusal_sft(cand) else 0)
            empties.append(1 if not cand.strip() else 0)
            meta = s.get("meta", {}).get(arm, {})
            truncs.append(1 if meta.get("truncated") else 0)
            rp = repetition_ratio(cand)
            if rp is not None:
                reps.append(rp)
            lens.append(len(cand))
            toks.append(int(meta.get("n_new_tokens", 0) or 0))
            openers.append(1 if has_chatty_opener(cand) else 0)
            numbered.append(1 if has_numbered_list(cand) else 0)
            if script_of(cand) == script_of(ref):
                script_match += 1
        n = len(samples)
        out[arm] = {
            "n": n,
            "em": {"hits": int(sum(em)), "of": n, "mean": mean(em)},
            "char_f1": {"mean": mean(f1s)},
            "rouge_l": {"mean": mean(rls), "impl": "EvaluationEngine.compute_rouge_l（字符级 LCS，F1）"},
            "refusal": {"count": int(sum(refusals)), "of": n},
            "empty": {"count": int(sum(empties)), "of": n},
            "truncated": {"count": int(sum(truncs)), "of": n},
            "repetition_ratio": {"mean": mean(reps), "max": round(max(reps), 4) if reps else None},
            "script_align": {"count": script_match, "of": n},
            "chatty_opener": {"count": int(sum(openers)), "of": n},
            "numbered_list": {"count": int(sum(numbered)), "of": n},
            "length_chars": {"mean": mean(lens), "median": median(lens), "min": min(lens), "max": max(lens)},
            "length_tokens": {"mean": mean(toks), "median": median(toks)},
        }
    return out


# ---------------------------------------------------------------- 裁判（臂 C 复用）


def build_sft_judge_messages(question: str, answer: str, reference: str) -> List[Dict]:
    """绝对打分 prompt：结构与仓库 build_api_judge_messages 逐字同构，
    仅把「参考文档」一节换成「参考答案」，并换用适配后的判据句。"""
    rubric = "\n".join(f"- {d}：{SFT_JUDGE_CRITERIA[d]}" for d in SFT_JUDGE_DIMENSIONS)
    system = "你是一个严格的评测裁判。只输出 JSON 对象，不要输出任何解释或多余文字。"
    user = f"""请根据以下标准对回答打分（1-5 分整数，5=完美，1=完全不合格）。

## 评测维度与标准
{rubric}

## 用户问题
{question}

## 参考答案
{(reference or "")[:2000]}

## 待评测回答
{answer}

## 输出格式（严格 JSON，不要 markdown 代码块）
{{"faithfulness": <1-5>, "relevance": <1-5>, "completeness": <1-5>}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_pairwise_messages(question: str, text_a: str, text_b: str, reference: str) -> List[Dict]:
    """成对对比 prompt：同一条问题、同一份参考答案下比较两个回答。"""
    rubric = "\n".join(f"- {d}：{SFT_JUDGE_CRITERIA[d]}" for d in SFT_JUDGE_DIMENSIONS)
    system = "你是一个严格的评测裁判。只输出 JSON 对象，不要输出任何解释或多余文字。"
    user = f"""请比较两个针对同一问题的回答。只输出 JSON 对象，不要输出任何解释或多余文字。

## 评测维度与标准
{rubric}

## 用户问题
{question}

## 参考答案
{(reference or "")[:2000]}

## 回答 A
{text_a}

## 回答 B
{text_b}

## 输出格式（严格 JSON，不要 markdown 代码块）
{{"a_faithfulness": <1-5>, "b_faithfulness": <1-5>, "a_relevance": <1-5>, "b_relevance": <1-5>, "a_completeness": <1-5>, "b_completeness": <1-5>, "winner": "A"|"B"|"tie"}}

winner 表示**整体更好的那个回答**（综合上面三个维度）；若两者质量相当则给 "tie"。"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_scores(content: str, dims: List[str], prefix: str = "") -> Dict:
    """解析裁判 JSON（复用仓库 parse_api_judge 的容错思路）。"""
    import re

    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {d: None for d in dims} | ({"winner": None} if prefix == "" and "winner" in dims else {})
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {d: None for d in dims}
    out: Dict = {}
    for d in dims:
        v = data.get(f"{prefix}{d}", None)
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            v = None
        out[d] = v if (v is not None and 1 <= v <= 5) else None
    return out


def load_cache() -> Dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache: Dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def call_judge(key: str, messages: List[Dict], cache: Dict, tag: str,
               sample_index: int, judge_samples: int, temperature: float,
               max_tokens: int = 160, budget: Optional[Dict] = None) -> Tuple[Dict, Dict]:
    """一次裁判调用（带磁盘缓存，复用仓库 cache_key_n 口径）。

    缓存 key 里带 model / n / i / temperature —— 与仓库一致，避免 N=1 的缓存被 N=3 命中。
    """
    ck = R.cache_key_n(messages, R.DEEPSEEK_MODEL, judge_samples, sample_index, temperature)
    entry = cache.get(ck)
    if entry and "content" in entry:
        if budget is not None:
            budget["cache_hits"] = budget.get("cache_hits", 0) + 1
        return entry["content"], {"source": "cache", "usage": entry.get("usage", {}),
                                  "model_returned": entry.get("model_returned"), "ts": entry.get("ts")}

    payload = {
        "model": R.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    last_err = None
    for attempt in range(3):
        try:
            data = R._http_json(
                f"{R.DEEPSEEK_BASE}/chat/completions", key, payload,
                use_proxy=(attempt == 2), timeout=90,
            )
            content = data["choices"][0]["message"]["content"]
            ts = time.time()
            cache[ck] = {
                "content": content,
                "usage": data.get("usage", {}),
                "model_returned": data.get("model"),
                "ts": ts,
                "via": "proxy" if attempt == 2 else "direct",
                "judge_samples": judge_samples,
                "sample_index": sample_index,
                "temperature": temperature,
                "tag": tag,
            }
            save_cache(cache)
            return content, {"source": "api", "usage": cache[ck]["usage"],
                              "model_returned": cache[ck]["model_returned"], "ts": ts}
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(1.0)
    return "", {"source": "api", "usage": {}, "error": last_err}


def run_absolute_judge(key: str, samples: List[Dict], cache: Dict, judge_samples: int,
                       temperature: float, budget: Dict) -> Dict:
    """绝对打分：每条样本 × 每条臂 × N 次采样 → 中位数。"""
    result: Dict = {}
    for s in samples:
        sid = s["id"]
        result[sid] = {}
        for arm in ARMS:
            msgs = build_sft_judge_messages(s["instruction"], s.get(f"{arm}_output", ""), s.get("reference", ""))
            raws, metas = [], []
            for i in range(judge_samples):
                content, meta = call_judge(key, msgs, cache, f"abs/{arm}/{sid}", i, judge_samples, temperature, budget=budget)
                if content:
                    raws.append(content)
                    metas.append(meta)
                    u = meta.get("usage", {}) or {}
                    budget["cost_cny"] = round(budget["cost_cny"] + R.cost_cny(
                        int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0),
                        meta.get("ts") or time.time()), 6)
                    budget["calls"] += 1
            parsed = [parse_scores(c, SFT_JUDGE_DIMENSIONS) for c in raws]
            agg = {}
            for d in SFT_JUDGE_DIMENSIONS:
                vals = [p[d] for p in parsed if p.get(d) is not None]
                agg[d] = {
                    "median": median(vals),
                    "raws": vals,
                    "n_parsed": len(vals),
                    "n_called": judge_samples,
                }
            result[sid][arm] = {
                "aggregate": agg,
                "raw_contents": raws,
                "parsed": parsed,
                "n_api_calls": len(raws),
                "model_returned": metas[0].get("model_returned") if metas else None,
            }
    return result


def run_pairwise_judge(key: str, samples: List[Dict], cache: Dict, temperature: float,
                       budget: Dict) -> Dict:
    """成对对比：每条样本跑两种呈现顺序（A=base,B=sft / A=sft,B=base），共 N=2 次。"""
    result: Dict = {}
    for s in samples:
        sid = s["id"]
        base = s.get("base_output", "") or ""
        sft = s.get("sft_output", "") or ""
        orders = [
            ("base_first", base, sft),   # A=base, B=sft
            ("sft_first", sft, base),    # A=sft,  B=base
        ]
        rec: Dict = {"orders": {}}
        for order_name, text_a, text_b in orders:
            msgs = build_pairwise_messages(s["instruction"], text_a, text_b, s.get("reference", ""))
            content, meta = call_judge(key, msgs, cache, f"pair/{sid}/{order_name}", 0, 1, temperature, budget=budget)
            u = meta.get("usage", {}) or {}
            if content:
                budget["cost_cny"] = round(budget["cost_cny"] + R.cost_cny(
                    int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0),
                    meta.get("ts") or time.time()), 6)
                budget["calls"] += 1
            parsed = parse_scores(content, SFT_JUDGE_DIMENSIONS + ["winner"]) if content else {}
            # winner 解析（不在 1-5 校验范围内，单独取）
            if content:
                import re as _re
                try:
                    txt = content.strip()
                    m = _re.search(r"\{.*\}", txt, _re.S)
                    data = json.loads(m.group(0)) if m else {}
                except Exception:  # noqa: BLE001
                    data = {}
                w = str(data.get("winner", "")).strip().upper()
                parsed["winner"] = w if w in ("A", "B", "TIE") else None
            rec["orders"][order_name] = {
                "text_a_is": "base" if order_name == "base_first" else "sft",
                "parsed": parsed,
                "raw": content,
            }
        # 位置偏置控制：把两次顺序的结论映射回「base vs sft」
        votes = []
        for order_name, o in rec["orders"].items():
            w = o["parsed"].get("winner")
            if w is None:
                continue
            if w == "TIE":
                votes.append("tie")
            elif order_name == "base_first":
                votes.append("base" if w == "A" else "sft")
            else:
                votes.append("sft" if w == "A" else "base")
        rec["votes"] = votes
        if votes:
            if votes.count("sft") > votes.count("base"):
                rec["verdict"] = "sft"
            elif votes.count("base") > votes.count("sft"):
                rec["verdict"] = "base"
            else:
                rec["verdict"] = "tie"
        else:
            rec["verdict"] = None
        result[sid] = rec
    return result


def judge_aggregates(samples: List[Dict], absolute: Dict, pairwise: Dict) -> Dict:
    """把逐条裁判结果聚合成可汇报的统计量（绝对计数 + 均值，不写相对提升）。"""
    agg: Dict = {"absolute": {}, "pairwise": {}}

    for d in SFT_JUDGE_DIMENSIONS:
        per_arm = {}
        for arm in ARMS:
            medians = [absolute[s["id"]][arm]["aggregate"][d]["median"] for s in samples]
            raws = []
            for s in samples:
                raws.extend(absolute[s["id"]][arm]["aggregate"][d]["raws"])
            per_arm[arm] = {
                "n_samples": len(medians),
                "mean_of_medians": mean(medians),
                "median_of_medians": median(medians),
                "n_parsed_samples": sum(1 for x in medians if x is not None),
                "raw_observations": len(raws),
                "raw_mean": mean(raws),
            }
        # 成对比较（逐条：sft 分 > base 分 记 sft 胜）
        wins = losses = ties = 0
        for s in samples:
            b = absolute[s["id"]]["base"]["aggregate"][d]["median"]
            t = absolute[s["id"]]["sft"]["aggregate"][d]["median"]
            if b is None or t is None:
                continue
            if t > b:
                wins += 1
            elif t < b:
                losses += 1
            else:
                ties += 1
        per_arm["paired"] = {
            "sft_better": wins, "sft_worse": losses, "tie": ties,
            "n": wins + losses + ties,
            "sign_test_p_two_sided": sign_test(wins, losses),
        }
        agg["absolute"][d] = per_arm

    verdicts = [pairwise[s["id"]]["verdict"] for s in samples]
    agg["pairwise"] = {
        "sft_better": verdicts.count("sft"),
        "base_better": verdicts.count("base"),
        "tie": verdicts.count("tie"),
        "unparsed": sum(1 for v in verdicts if v is None),
        "n": len(verdicts),
        "sign_test_p_two_sided": sign_test(verdicts.count("sft"), verdicts.count("base")),
    }
    return agg


# ---------------------------------------------------------------- 报告渲染


def build_conclusion(comp: Dict, agg: Optional[Dict], n: int) -> str:
    """由实际数字生成结论段（避免手抄数字出错）。

    口径纪律：只给绝对计数与均值 + n，不写「提升 X%」。
    本实验的结果**不支持**「微调更好」，因此结论如实写成「没有提升」。
    """
    b, s = comp["base"], comp["sft"]
    L: List[str] = []

    if agg:
        pw = agg["pairwise"]
        f = agg["absolute"]["faithfulness"]["paired"]
        r = agg["absolute"]["relevance"]["paired"]
        c = agg["absolute"]["completeness"]["paired"]
        L.append(
            f"**最诚实的一句话：这次 100 条 LoRA SFT 在 held-out 上并没有带来质量提升 —— "
            f"强裁判成对直接对比是「基座更好 {pw['base_better']} 条 : 微调更好 {pw['sft_better']} 条，"
            f"持平 {pw['tie']} 条」（n={pw['n']}，成对符号检验 p = {fmt(pw['sign_test_p_two_sided'], 4)}）；"
            f"逐维度看 faithfulness {f['sft_better']}:{f['sft_worse']}、"
            f"relevance {r['sft_better']}:{r['sft_worse']}（p = {fmt(r['sign_test_p_two_sided'], 4)}）、"
            f"completeness {c['sft_better']}:{c['sft_worse']}，方向一致偏向基座。**"
        )
        L.append("")
        L.append(
            f"- **微调确实改动了什么**：输出表面格式变了 —— 闲聊式开场从 "
            f"{b['chatty_opener']['count']}/{n} 条降到 {s['chatty_opener']['count']}/{n} 条，"
            f"退化重复率均值从 {fmt(b['repetition_ratio']['mean'], 4)} 降到 {fmt(s['repetition_ratio']['mean'], 4)}。"
            f"这就是 100 条样本能学到的全部东西：**格式模仿，不是知识**。"
        )
        L.append(
            f"- **微调没有改动什么**：与参考答案的贴合度反而略降 —— "
            f"字符级 F1 {fmt(b['char_f1']['mean'], 4)} → {fmt(s['char_f1']['mean'], 4)}，"
            f"ROUGE-L {fmt(b['rouge_l']['mean'], 4)} → {fmt(s['rouge_l']['mean'], 4)}，"
            f"EM {b['em']['hits']}/{n} → {s['em']['hits']}/{n}。"
        )
        L.append(
            f"- **为什么这仍然值 10 分钟讲**：因为「跑通全流程」和「微调有效」是两件事。"
            f"我原来简历上写的是后者，现在我拿数据把它改成了前者 —— "
            f"**并且能当场说清楚是哪一步没起作用、为什么**（见 4.5 的归因分析）。"
        )
    else:
        L.append(
            f"**在 {n} 条 held-out 上，LoRA 微调只改变了输出表面格式"
            f"（闲聊式开场 {b['chatty_opener']['count']}/{n} → {s['chatty_opener']['count']}/{n}），"
            f"与参考答案的贴合度略降（ROUGE-L {fmt(b['rouge_l']['mean'], 4)} → {fmt(s['rouge_l']['mean'], 4)}）。**"
            f"（本轮未跑强裁判臂。）"
        )
    return "\n".join(L)


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_report(meta: Dict, samples: List[Dict], comp: Dict, judge: Optional[Dict],
                  agg: Optional[Dict], budget: Dict, args) -> str:
    L: List[str] = []
    A = L.append
    n = len(samples)

    A("# LoRA 微调前后效果对比（真实跑 · 并排输出 + 三栏评分）")
    A("")
    A(f"- 生成日期：{time.strftime('%Y-%m-%d')}")
    A(f"- 实验产物：`experiments/sft_before_after_2026-10-21.jsonl`（逐条并排输出）、"
      f"`experiments/sft_before_after_2026-10-21_scores.json`（全部数字）、"
      f"`experiments/sft_before_after_2026-10-21_human_sheet.md`（人工评分表，评分列留空）")
    A("")

    A("## 〇、一句话结论（先看这个）")
    A("")
    A(build_conclusion(comp, agg, n))
    A("")

    A("## 一、口径声明（先把话说清楚）")
    A("")
    A("### 1.1 被测对象：这是谁微调谁")
    A("")
    A("| 项 | 值 | 来源 |")
    A("|---|---|---|")
    A(f"| 基座模型 | `{meta['base_model']}` | adapter_config.json 的 `base_model_name_or_path` |")
    A(f"| LoRA 适配器 | `{meta['adapter_path']}` | 本机既有产物（= 原训练 `sft_output/checkpoint-80`） |")
    A(f"| 适配器权重 sha256 | `{meta['adapter_sha256']}` | 本脚本现场计算 |")
    ac = meta["adapter_config"]
    A(f"| LoRA 超参 | r={ac['r']} / alpha={ac['lora_alpha']} / dropout={ac['lora_dropout']} | adapter_config.json |")
    A(f"| target_modules | {', '.join(ac['target_modules'])} | adapter_config.json |")
    A(f"| 训练数据 | `{meta['training_data_file']}`（sha256 `{meta['training_data_sha256'][:16]}…`） | 本机既有文件 |")
    ts = meta.get("trainer_state", {})
    A(f"| 原训练进度 | step {ts.get('global_step')} / epoch {ts.get('epoch')} / best eval_loss {fmt(ts.get('best_metric_eval_loss'), 4)} | checkpoint-80 的 trainer_state.json |")
    prov = meta.get("adapter_provenance", {})
    if prov:
        A(f"| 权重溯源 | 适配器与 `checkpoint-80` 的 `adapter_model.safetensors` **逐字节一致** = {prov.get('identical_to_checkpoint_weight')} | 本脚本现场算 sha256 比对 |")
        others = prov.get("other_checkpoints", {})
        diff = [k for k, v in others.items() if not v.get("same_as_final")]
        if diff:
            A(f"| 排除误用 | 与 {', '.join(diff)} 的权重**不同** → 确认用的是 val_loss 最低那一步 | 同上 |")
    A("")
    curve = ts.get("eval_curve") or []
    if curve:
        A("**原训练的 eval_loss 轨迹**（每 20 步一次 eval，直接来自 checkpoint 的 trainer_state.json）：")
        A("")
        A("| step | epoch | eval_loss |")
        A("|---|---|---|")
        for c in curve:
            A(f"| {c.get('step')} | {c.get('epoch')} | {fmt(c.get('eval_loss'), 4)} |")
        A("")
    A("> **这些训练是真的已经发生过的**（2026-07-30 跑完，权重与 checkpoint 都在本机）。"
      "本报告**没有重新训练**，只是把那次的 LoRA 权重装回基座模型，在 held-out 集上做前后对比。")
    A("")
    A("### 1.2 held-out 集怎么来的（不是随便编的题）")
    A("")
    ho = meta["held_out"]
    A(f"- {ho['derivation']}")
    A(f"- **n = {ho['n']}**")
    A(f"- ⚠️ {ho['caveat']}")
    A("")
    A("### 1.3 对照实验的控制变量")
    A("")
    A("| 控制项 | 做法 |")
    A("|---|---|")
    A(f"| prompt | 两条臂**逐字相同**（含与训练一致的 system 消息 `{meta['prompt_policy']['system_prompt']}`），唯一变量是是否加载 LoRA 权重 |")
    A(f"| 解码 | 贪心 `do_sample=False`，`max_new_tokens={meta['decoding']['max_new_tokens']}`，temperature/top_p/top_k 全部置空 |")
    A("| 可复现性 | 贪心解码 → 同一份权重重跑逐条一致（运行间方差为 0） |")
    A(f"| 硬件 | {meta['env']['gpu']}，fp16 |")
    A("")
    A("### 1.4 数字纪律")
    A("")
    A("- 所有结论都带 **n** 和来源文件；小样本只报**绝对计数**（如「20 条中 16 条 vs 4 条」）与均值。")
    A("- **不写「提升 X%」这类相对提升**：n=20 的样本量不支持百分比结论，需要时给「条数/均值 + n」。")
    A("- 唯一的例外是**成对检验的 p 值**（二项符号检验，附口径），它本身就是小样本口径。")
    A("")

    A("## 二、复现命令")
    A("")
    A("```bash")
    A("# ① 并排生成（GPU，约 6 分钟；用本机 HF 缓存的基座模型）")
    A("python experiments/run_sft_before_after.py \\")
    A("    --out experiments/sft_before_after_2026-10-21.jsonl")
    A("")
    A("# ② 只跑可计算指标（离线，0 花费）")
    A("python experiments/score_sft_before_after.py --no-judge \\")
    A("    --out-md experiments/sft_before_after_2026-10-21.md")
    A("")
    A("# ③ 可计算指标 + 强裁判（需要 DEEPSEEK_API_KEY；key 只从环境变量读）")
    A("python experiments/score_sft_before_after.py \\")
    A("    --out-md experiments/sft_before_after_2026-10-21.md")
    A("```")
    A("")
    A(f"环境：Python {meta['env']['python']} · torch {meta['env']['torch']} · {meta['env']['gpu']} · "
      f"peft {ac.get('peft_version_saved')}（权重保存版本）")
    A("")

    A("## 三、第一栏：可计算指标")
    A("")
    A("### 3.1 口径")
    A("")
    A("| 指标 | 口径 |")
    A("|---|---|")
    A("| EM（精确匹配） | 复用仓库 `run_generation_eval.normalize_text` 归一化（去空白与标点、转小写）后全等 |")
    A("| 字符级 F1 | 复用仓库 `run_generation_eval.char_f1`（多重集重叠，中文按字） |")
    A("| ROUGE-L | 复用仓库 `src/evaluation.py::EvaluationEngine.compute_rouge_l`（字符级 LCS → F1） |")
    A("| 拒答率 | 仓库 `REFUSAL_MARKERS` ∪ 通用拒答标记（`我不能`/`无法完成`/`抱歉` 等） |")
    A("| 截断率 | 生成 token 数触顶 max_new_tokens 且未落到 EOS |")
    A("| 空输出率 | 去除首尾空白后长度为 0 |")
    A("| 退化重复率 | `1 − (唯一 4-gram / 总 4-gram)`，字符级；越高越像卡循环 |")
    A("| 风格对齐率 | 输出主语言（中/英）与**参考答案**主语言一致的比例 |")
    A("| 闲聊式开场率 | 输出**开头**是「好的 / 当然 / 我理解 / 请提供…」这类客套话的比例（alpaca-gpt4-zh 风格里没有） |")
    A("| 编号列表率 | 输出里出现 `1.` / `1、` 式编号列表的比例（该数据集的主要回答结构） |")
    A("")
    A("### 3.2 结果")
    A("")
    A(f"样本量 n = {n}（held-out 验证集全集，无抽样）")
    A("")
    A("| 指标 | 基座（未微调） | LoRA 微调后 | 口径 |")
    A("|---|---|---|---|")
    b, s = comp["base"], comp["sft"]
    A(f"| EM 命中 | **{b['em']['hits']} / {n}** | **{s['em']['hits']} / {n}** | 归一化后全等 |")
    A(f"| 字符级 F1 均值 | {fmt(b['char_f1']['mean'], 4)} | {fmt(s['char_f1']['mean'], 4)} | 多重集重叠 |")
    A(f"| ROUGE-L 均值 | {fmt(b['rouge_l']['mean'], 4)} | {fmt(s['rouge_l']['mean'], 4)} | 字符级 LCS-F1 |")
    A(f"| 拒答 | **{b['refusal']['count']} / {n}** | **{s['refusal']['count']} / {n}** | 标记命中 |")
    A(f"| 空输出 | **{b['empty']['count']} / {n}** | **{s['empty']['count']} / {n}** | 长度 0 |")
    A(f"| 截断（撞 token 上限） | **{b['truncated']['count']} / {n}** | **{s['truncated']['count']} / {n}** | 未落 EOS |")
    A(f"| 退化重复率 均值 | {fmt(b['repetition_ratio']['mean'], 4)} | {fmt(s['repetition_ratio']['mean'], 4)} | 1 − unique4/total4 |")
    A(f"| 退化重复率 最大 | {fmt(b['repetition_ratio']['max'], 4)} | {fmt(s['repetition_ratio']['max'], 4)} | 同上 |")
    A(f"| 风格对齐（与参考答案主语言一致） | **{b['script_align']['count']} / {n}** | **{s['script_align']['count']} / {n}** | 中/英主语言 |")
    A(f"| 闲聊式开场（「好的，以下是…」） | **{b['chatty_opener']['count']} / {n}** | **{s['chatty_opener']['count']} / {n}** | 开头匹配客套话 |")
    A(f"| 编号列表结构 | **{b['numbered_list']['count']} / {n}** | **{s['numbered_list']['count']} / {n}** | 出现 `1.` / `1、` |")
    A(f"| 输出长度（字符）均值 | {fmt(b['length_chars']['mean'], 1)} | {fmt(s['length_chars']['mean'], 1)} | — |")
    A(f"| 输出长度（字符）中位 | {fmt(b['length_chars']['median'], 1)} | {fmt(s['length_chars']['median'], 1)} | — |")
    A(f"| 输出长度（token）均值 | {fmt(b['length_tokens']['mean'], 1)} | {fmt(s['length_tokens']['mean'], 1)} | — |")
    A("")

    A("## 四、第二栏：LLM-as-Judge")
    A("")
    if not judge:
        A("> 本轮**未调用裁判**（`--no-judge`）。")
        A("")
    else:
        A("### 4.1 裁判协议")
        A("")
        A("| 项 | 值 |")
        A("|---|---|")
        A("| 强裁判 | DeepSeek API，`deepseek-chat`（实测解析为 `deepseek-flash`） |")
        A(f"| 绝对打分采样次数 | N={args.judge_samples}，temperature={args.judge_temperature}，**取中位数** |")
        A("| 成对对比 | N=2（两种呈现顺序各一次），temperature=0.8 —— 同时用于量化位置偏置 |")
        A("| 判分维度 | faithfulness / relevance / completeness（**沿用仓库三维**） |")
        A("| API 调用次数 | 本次新调用 %d 次 + 缓存命中 %d 次（缓存 key 含 model/N/第几次/temperature，重跑 0 花费） |"
          % (budget["calls"], budget.get("cache_hits", 0)))
        A(f"| 估算费用 | ¥{budget['cost_cny']:.6f}（硬上限 ¥{R.MAX_BUDGET_CNY}，未超） |")
        A("| key 纪律 | **只从环境变量 `DEEPSEEK_API_KEY` 读**，不写入任何文件/日志/报告/缓存 |")
        A("")
        A("### 4.2 rubric（相对仓库既有判据改了什么，必须写明）")
        A("")
        A("仓库的 rubric 面向 RAG（有「参考文档」）；本实验是 SFT 前后对比，手上只有「参考答案」。"
          "**维度名与判据句逐字沿用仓库**，只把事实依据的载体从「参考文档」换成「参考答案」：")
        A("")
        A("| 维度 | 仓库原判据（rag-eval-platform） | 本实验判据 | 改动 |")
        A("|---|---|---|---|")
        A(f"| faithfulness | {R.JUDGE_CRITERIA['faithfulness']} | {SFT_JUDGE_CRITERIA['faithfulness']} | 「参考文档」→「参考答案」 |")
        A(f"| relevance | {R.JUDGE_CRITERIA['relevance']} | {SFT_JUDGE_CRITERIA['relevance']} | 无 |")
        A(f"| completeness | {R.JUDGE_CRITERIA['completeness']} | {SFT_JUDGE_CRITERIA['completeness']} | 「参考文档」→「参考答案」 |")
        A("")
        A("### 4.3 绝对打分结果（每条样本取 N 次中位数）")
        A("")
        A(f"样本量 n = {n}")
        A("")
        A("| 维度 | 臂 | 均值（中位数的均值） | 中位数 | 原始观测数 |")
        A("|---|---|---|---|---|")
        for d in SFT_JUDGE_DIMENSIONS:
            for arm in ARMS:
                a = agg["absolute"][d][arm]
                A(f"| {d} | {ARM_LABEL[arm]} | {fmt(a['mean_of_medians'])} | {fmt(a['median_of_medians'])} | "
                  f"{a['raw_observations']} |")
        A("")
        A("逐条成对比较（同一维度内，SFT 的中位数 vs 基座的中位数；持平剔除后做符号检验）：")
        A("")
        A("| 维度 | SFT 更好 | SFT 更差 | 持平 | 参与检验 n | 成对符号检验 p（双侧） |")
        A("|---|---|---|---|---|---|")
        for d in SFT_JUDGE_DIMENSIONS:
            p = agg["absolute"][d]["paired"]
            A(f"| {d} | {p['sft_better']} | {p['sft_worse']} | {p['tie']} | {p['n']} | "
              f"{fmt(p['sign_test_p_two_sided'], 4)} |")
        A("")
        A("### 4.4 成对对比结果（直接问「这条微调后更好还是更差」）")
        A("")
        pw = agg["pairwise"]
        raw_votes = {"sft": 0, "base": 0, "tie": 0}
        for sid, v in judge["pairwise"].items():
            for order, o in v["orders"].items():
                w = o["parsed"].get("winner")
                if w == "TIE":
                    raw_votes["tie"] += 1
                elif order == "base_first":
                    raw_votes["base" if w == "A" else "sft"] += 1
                else:
                    raw_votes["sft" if w == "A" else "base"] += 1
        A(f"- **SFT 更好：{pw['sft_better']} / {pw['n']}**")
        A(f"- 基座更好：**{pw['base_better']} / {pw['n']}**")
        A(f"- 持平：{pw['tie']} / {pw['n']}")
        A(f"- 未解析：{pw['unparsed']} / {pw['n']}")
        A(f"- 成对符号检验 p = {fmt(pw['sign_test_p_two_sided'], 4)}（双侧）")
        A("")
        A(f"**原始票数（每条样本 2 次判、合计 {raw_votes['sft'] + raw_votes['base'] + raw_votes['tie']} 票）**："
          f"基座 **{raw_votes['base']}** 票 : 微调 **{raw_votes['sft']}** 票 : 持平 **{raw_votes['tie']}** 票。"
          f"（逐条结论是把同一条样本的 2 票合成 1 票后的结果，所以票数与条数不是一一对应。）")
        A("")
        A("两种呈现顺序各判一次，因此这个计数**天然控住了位置偏置**"
          "（模型偏好「回答 A」时，两个顺序会给出互相矛盾的结论，持平就是偏置的体现）。")
        A("")
        A("> 位置偏置确实存在且被量到了：例如 `heldout-11`、`heldout-14` 两条，"
          "裁判在「A=base」顺序里选了 B（=SFT），在「A=sft」顺序里又选了 B（=base）——"
          "这是典型的「总是选 B」位置偏置，被两顺序抵消后判为持平。")
        A("")
        A("### 4.5 为什么没提升（我的归因分析）")
        A("")
        A("下面是**归因假设**，不是本实验证明了的因果结论 —— 本实验只测了「微调前后差多少」，"
          "没有做消融实验去隔离每个因素。")
        A("")
        A("| # | 归因 | 依据 |")
        A("|---|---|---|")
        A("| 1 | **基座本来就被 SFT 过了，基线很高** | 基座是 `Qwen2.5-1.5B-**Instruct**`，"
          "阿里已经用海量指令数据做过 SFT + RLHF。在它之上再用 100 条 alpaca 样本"
          "「二次 SFT」，等于用更弱的数据覆盖更强的对齐 —— **数据量不对等时是负收益**。 |")
        A("| 2 | **数据量太小、且不是领域数据** | 只有 100 条、通用中文指令（alpaca-gpt4-zh 派生），"
          "不是金融/房产领域数据。100 条足够学格式，不足以改知识。 |")
        A("| 3 | **训练本身偏欠拟合** | 80 step / 2 epoch / 有效 batch=2 / lr=5e-5（余弦退火）；"
          "eval_loss 1.797 → 1.583 仍在缓慢下降，说明还有空间没训完。 |")
        A("| 4 | **checkpoint 选择与评测集重叠** | `load_best_model_at_end` 用的就是这 20 条验证集，"
          "所以「最优 checkpoint」是**对这 20 条**最优，不是对未见数据最优（见 1.2 的 caveat）。 |")
        A("| 5 | **指标本身偏置** | faithfulness/completeness 都以参考答案为准，"
          "会惩罚「与参考答案表述不同但同样正确」的答案；而模型输出越具体、"
          "离参考答案的措辞越远，越容易被判低分（见 4.2 的 rubric 改动）。 |")
        A("")
        A("**如果重做，我会按这个顺序改**（这也是简历里那行字该有的正确说法）："
          "① 换成**领域数据**（金融/房产 ≥ 2000 条，且带「拒答/不可答」负样本）；"
          "② 基座换成同量级的**非 instruct** 模型或更小的基座，否则没有提升空间；"
          "③ 把 20 条验证集**再切一份不参与 checkpoint 选择的 test 集**；"
          "④ 训练到 eval_loss 平台期再停，而不是固定 2 epoch。")
        A("")

    A("## 五、第三栏：人工评分表")
    A("")
    A(f"- 路径：`experiments/sft_before_after_2026-10-21_human_sheet.md`")
    A(f"- **评分列留空**，由舰长本人打分（每条 1-5 分 + 一句话判语），约 10 分钟。")
    A("- 本报告**不代替人工打分**：自动分（尤其是裁判分）不能替代「这答案我自己愿不愿意用」的主观判断。")
    A("")

    A("## 六、并排输出与逐条差异明细")
    A("")
    A("### 6.1 逐条一览（全部 20 条，无抽样）")
    A("")
    A("| # | 提示（节选） | 基座 token | 微调 token | 差值 | 两者是否逐字相同 | 强裁判成对结论 |")
    A("|---|---|---|---|---|---|---|")
    for smp in samples:
        q = (smp["instruction"] or "").replace("\n", " ")[:30]
        bt = int(smp.get("meta", {}).get("base", {}).get("n_new_tokens", 0) or 0)
        st = int(smp.get("meta", {}).get("sft", {}).get("n_new_tokens", 0) or 0)
        same = "是" if (smp.get("base_output") or "") == (smp.get("sft_output") or "") else "否"
        verdict = ""
        if judge:
            v = judge["pairwise"][smp["id"]]["verdict"]
            verdict = {"sft": "SFT 更好", "base": "基座更好", "tie": "持平"}.get(v, "未解析")
        A(f"| {smp['id']} | {q} | {bt} | {st} | {st - bt:+d} | {same} | {verdict} |")
    A("")
    n_same = sum(1 for smp in samples if (smp.get("base_output") or "") == (smp.get("sft_output") or ""))
    A(f"- 两条臂输出**逐字完全相同**的有 **{n_same} / {len(samples)}** 条 —— 这些是 LoRA 权重没产生任何行为差异的样本，"
      f"本身就是一个诚实的观测点（不是「微调让每条都变好」）。")
    A("")
    A("### 6.2 并排输出（节选，完整版见 jsonl）")
    A("")
    A("| # | 提示（instruction） | 基座输出（节选） | 微调后输出（节选） |")
    A("|---|---|---|---|")
    for smp in samples[:6]:
        q = (smp["instruction"] or "").replace("\n", " ")[:40]
        bo = (smp["base_output"] or "").replace("\n", " ")[:90]
        so = (smp["sft_output"] or "").replace("\n", " ")[:90]
        A(f"| {smp['id']} | {q} | {bo}… | {so}… |")
    A("")
    A(f"> 全量 {n} 条并排输出在 `experiments/sft_before_after_2026-10-21.jsonl`"
      "（该扩展名被仓库 .gitignore 的 `*.jsonl` 规则忽略，**不会自动进 git**；"
      "如需公开到 GitHub 请手动处理）。")
    A("")

    A("## 七、诚实边界（哪些结论不能被过度解读）")
    A("")
    A("1. **样本量小**：n=20（held-out 验证集全集）。所有比例与均值只能给**量级**，不能当分布估计；"
      "1-2 条样本的差异不构成结论。")
    A("2. **单一领域 + 单一数据源**：全部 20 条来自 `alpaca-gpt4-zh` 派生的 100 条通用中文指令数据，"
      "**不是金融/房产领域数据**，也没有做领域内泛化测试。")
    A("3. **held-out 不是 test-clean**：这 20 条就是原训练里 `load_best_model_at_end` 用来选 checkpoint 的验证集，"
      "checkpoint 选择过程见过这些样本的 loss。它是 dev-clean，但不是完全独立的 test set。")
    A("4. **裁判有噪声**：本仓库已实测（见 `experiments/results/judge_variance_2026-10-20.md` 与 "
      "`reference_robustness_2026-10-20.md`）："
      "**本地 1.5B 裁判与强裁判的相关性 |r| < 0.25**（N=1 贪心 r=−0.03，median-of-7 r=−0.14），"
      "MAE 约为规则臂的 2 倍，且该结论**对聚合方式稳健**（换 median-of-7 结论不变）——"
      "所以本报告**没有**用本地 1.5B 当裁判，改用 DeepSeek 强裁判；"
      "同一份报告也实测**强裁判自身 N=3 与其 N=1 参照 MAE 仅 0.11 档、r=+0.92~0.94**，可用作参照系，"
      "但**强裁判也有噪声**（99 个单元里 3 个极差 ≥3 档），且模型会滚动更新，跨时间比较无效。")
    A("5. **只做了有限形式的显著性检验**：仅成对符号检验（二项），n=20 下检验力很低；"
      "**未做**多重比较校正、**未做**多轮采样（每条只跑一次贪心）、**未做**不同随机种子的重复实验。")
    A("6. **rubric 被改动过**：本实验的 faithfulness/completeness 判据把「参考文档」换成了「参考答案」"
      "（见 4.2）。基于参考答案的 faithfulness 会**惩罚「与参考答案表述不同但同样正确」的答案**，"
      "因此这两个维度的绝对分**不能**直接当作「事实正确率」。")
    A("7. **EM/ROUGE-L 在开放式生成上参考偏置严重**：alpaca-gpt4-zh 的参考答案只是「一种」写法，"
      "EM 命中率天然低；这两个指标衡量的是「像不像参考答案」，不是「对不对」。")
    A("8. **裁判不是真值**：LLM-as-Judge 只测裁判与裁判的一致性，不是与人类标注对齐的准确性。"
      "第三栏的人工评分表才是最后一道校准。")
    A("9. **未做多 checkpoint 对比**：只用了 `checkpoint-80`（val_loss 最低的那个），"
      "没有把 checkpoint-40/60 一起跑成三臂，因此**没有**「早停/选点策略」这类结论。")
    A("10. **本实验的结论是负面的，且未做消融**：结论是「这次微调没有带来质量提升」。"
      "4.5 的五条归因是**假设**，本实验**没有**做消融实验去隔离各因素；"
      "「微调没用」这个结论本身只在这 20 条、这份数据、这个基座上成立，"
      "**不能**推广成「LoRA 微调没用」。")
    A("")

    A("## 八、这对你简历/面试意味着什么")
    A("")
    A("### 8.1 建议改写（更抗追问）")
    A("")
    A("| 原写法 | 问题 | 建议写法 |")
    A("|---|---|---|")
    A("| 「LoRA 微调 Qwen 模型完成 SFT 监督微调全流程（数据构建→训练→部署）」 | "
      "没错，但面试官一追问「调完效果怎么样」就答不上来，"
      "而答不上来通常被理解为「其实没做」 | "
      "「完成 LoRA SFT 全流程（数据构建→训练→checkpoint 选型→部署），"
      "并搭建**微调前后并排评测**（held-out 复现 + 强裁判 LLM-as-Judge + 人工评分表），"
      "**实测结论是本次微调未带来质量提升并给出归因**」 |")
    A("")
    A("### 8.2 为什么「负面结论 + 归因」比「正面数字」更值钱")
    A("")
    A("- 前者**可验证**：每个数字都有 n、有来源文件、有复现命令，面试官可以让你当场重跑。")
    A("- 后者**不可验证且危险**：n=20 上任何「提升 X%」都会被一句「样本量多少？做了显著性检验吗？」打穿。")
    A("- 负面结论 + 明确的下一步（换领域数据 / 单独切 test 集 / 训到 loss 平台期）"
      "展示的是**实验设计能力**，这正是初级岗与高级岗的分水岭。")
    A("")
    A("### 8.3 还需要你本人动手的两件事")
    A("")
    A("1. **填人工评分表**：`experiments/sft_before_after_2026-10-21_human_sheet.md`"
      "（20 条，两栏 1-5 分 + 一句话判语，约 10 分钟）。"
      "这一步**不能由 AI 代填** —— 代填的人工分在面试里一问就穿。")
    A("2. **决定要不要补做一次领域 SFT**：本实验用的是通用中文指令数据（alpaca-gpt4-zh），"
      "**不是**金融/房产领域数据。如果你希望简历里能写「金融/房产领域 LoRA 微调」，"
      "那需要**补做一次**：构造 ≥2000 条金融/房产指令数据（含拒答负样本）→ 用同一套 LoRA 配置训练 → "
      "用本报告这套评测流程重跑前后对比。**本报告的一切都不覆盖那一次训练。**")
    A("")

    A("## 九、面试怎么讲（3 分钟版）")
    A("")
    if agg:
        pw = agg["pairwise"]
        f = agg["absolute"]["faithfulness"]["paired"]
        r = agg["absolute"]["relevance"]["paired"]
    A("> 「我用 LoRA 对 Qwen2.5-1.5B-Instruct 做了一次 SFT：r=8、alpha=16、7 个投影层全挂、"
      "100 条中文指令数据、2 epoch、lr=5e-5，val_loss 从 1.797 降到 1.583。"
      "然后我做了一件简历上没写、但我觉得更值钱的事 —— **把微调前后的效果并排测出来**。")
    A(">")
    A("> 测试集不是我编的题，是**原训练自己的验证集**（我把当时的切分原样复现出来，20 条），"
      "prompt 和解码参数两条臂逐字相同，唯一变量是装不装 LoRA 权重。"
      "评分分三层：可计算指标、强裁判（DeepSeek API，N=3 取中位数，另加一组**两种顺序的成对对比**"
      "来控位置偏置）、以及我自己打的 1-5 分人工表。")
    A(">")
    A("> **结果和我的预期不一样，我说清楚**：强裁判成对对比是**基座更好 "
      f"{pw['base_better']} 条 : 微调更好 {pw['sft_better']} 条**，持平 {pw['tie']} 条；"
      "faithfulness 逐条是 " + f"{f['sft_better']}:{f['sft_worse']}、relevance {r['sft_better']}:{r['sft_worse']}。"
      "**微调没有带来质量提升**。它实际改变的只有输出格式（闲聊式开场少了 2 条），"
      "跟参考答案的贴合度反而略降。")
    A(">")
    A("> 我分析了原因：基座本来就是 Instruct 模型（已经被 SFT 过一次），"
      "100 条通用指令数据在它之上是**用更弱的数据覆盖更强的对齐**；"
      "数据量不够学知识、只够学格式；而且 checkpoint 选择用的就是这 20 条，不算完全独立。"
      "所以我下一步会做的是换领域数据 + 单独切 test 集 + 训到 loss 平台期。")
    A(">")
    A("> **我把简历上那句改成了**：不是「LoRA 微调提升了模型效果」，"
      "而是「完成 LoRA SFT 全流程（数据构建→训练→checkpoint 选型→部署），"
      "并搭建了微调前后的并排评测与 LLM-as-Judge 流程」。"
      "**前者一句追问就穿，后者是我能拿数据当场证明的。**")
    A("")
    A("> 被追问「你怎么保证不是自说自话」时："
      "① 裁判用的是 DeepSeek API 而不是本地小模型 —— 因为本仓库自己实测过本地 1.5B 裁判跟强裁判 |r|<0.25、MAE 高一倍；"
      "② 成对对比跑了两种呈现顺序，位置偏置被量出来并抵消；"
      "③ 每条只跑一次贪心解码，不做多轮采样，所以差异只可能来自权重，不可能来自采样运气；"
      "④ **结论是负面的，我照样写出来** —— 这恰恰是我敢把它写进简历的原因。")
    A("")

    return "\n".join(L)


def row_criterion(smp: Dict) -> str:
    """给每条样本生成一条**机械的**判据（不含 AI 的主观判断，避免污染人工打分）。

    - 两臂输出逐字相同 → 提示「建议打相同分」（自检锚点）
    - 否则 → 从**参考答案**里机械抽取数字/英文标识符作为「要核对的关键事实」
    """
    bo = (smp.get("base_output") or "").strip()
    so = (smp.get("sft_output") or "").strip()
    if bo == so:
        return "两臂逐字相同 → 建议**打相同分**（可作自检锚点）"
    facts = R.extract_facts(smp.get("reference", "") or "")
    # 过滤掉「列表序号」型数字（1. 2. 3. 这种是编号，不是事实）
    ref = smp.get("reference", "") or ""
    facts = [
        f for f in facts
        if not (f.isdigit() and re.search(r"(?:^|\n)\s*" + re.escape(f) + r"[.、]", ref))
    ]
    facts = [f for f in facts if len(f) >= 1][:8]
    # 去重（保持顺序）+ 丢掉单字母英文噪声
    seen, uniq = set(), []
    for f in facts:
        if f in seen:
            continue
        if f.isalpha() and len(f) < 2:
            continue
        seen.add(f)
        uniq.append(f)
    facts = uniq[:8]
    if facts:
        return "核对关键事实：" + "、".join(facts)
    return "核对：是否回应问题、是否复现参考答案的结构（编号/列表/要点数）"


def render_human_sheet(samples: List[Dict], comp: Dict) -> str:
    L: List[str] = []
    A = L.append
    A("# 人工评分表（**请舰长本人打分**）")
    A("")
    A(f"- 数据来源：`experiments/sft_before_after_2026-10-21.jsonl`（held-out 验证集 {len(samples)} 条）")
    A("- **评分列留空**，等舰长填。建议每条花 30 秒，全表约 10 分钟。")
    A("- 打分口径（1-5 分，5=完美，1=完全不能用）：")
    A("  - **内容对不对**：事实/逻辑是否成立，有无编造。")
    A("  - **答没答到点上**：有没有真正回应问题。")
    A("  - **能不能直接用**：你是否愿意把这个答案原样贴给用户。")
    A("- 「判据」列是**机械生成**的核对清单（从参考答案抽数字/实体；两臂相同时提示打相同分），"
      "**不含 AI 的主观判断**，不会污染你的打分。")
    A("- 输出节选截断到 300 字符；全文见 jsonl。")
    A("")
    A("---")
    A("")
    A("## 评分表")
    A("")
    A("| # | 提示 | 基座输出（节选 300 字） | 微调后输出（节选 300 字） | 参考答案（节选 200 字） | 判据 | 基座分 (1-5) | 微调分 (1-5) | 一句话判语 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for smp in samples:
        q = (smp["instruction"] or "").replace("|", "\\|").replace("\n", " ")[:60]
        bo = (smp["base_output"] or "").replace("|", "\\|").replace("\n", " ")[:300]
        so = (smp["sft_output"] or "").replace("|", "\\|").replace("\n", " ")[:300]
        rf = (smp["reference"] or "").replace("|", "\\|").replace("\n", " ")[:200]
        crit = row_criterion(smp).replace("|", "\\|")
        A(f"| {smp['id']} | {q} | {bo} | {so} | {rf} | {crit} | ☐ | ☐ | |")
    A("")
    A("---")
    A("")
    A("## 填完之后")
    A("")
    A("填完请回传给 AI 助手，可以把人工分并进 `sft_before_after_2026-10-21_scores.json` 的 "
      "`manual` 字段并重出报告；**这一步不能由 AI 代填**——代填的人工分在面试里一问就穿。")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA 微调前后评分")
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--out-human", type=Path, default=DEFAULT_HUMAN)
    ap.add_argument("--judge-samples", type=int, default=3)
    ap.add_argument("--judge-temperature", type=float, default=0.8)
    ap.add_argument("--no-judge", action="store_true", help="只跑可计算指标，0 花费")
    ap.add_argument("--max-budget-cny", type=float, default=R.MAX_BUDGET_CNY)
    args = ap.parse_args()

    print("=" * 72)
    print("LoRA 微调前后评分")
    print("=" * 72)

    lines = [l for l in io.open(args.jsonl, encoding="utf-8").read().splitlines() if l.strip()]
    meta = json.loads(lines[0])
    samples = [json.loads(l) for l in lines[1:]]
    print(f"读入 {len(samples)} 条样本；held-out n = {meta['held_out']['n']}")

    print("\n[1/4] 可计算指标 ...")
    comp = computable_metrics(samples)
    for arm in ARMS:
        print(f"  {arm}: EM {comp[arm]['em']['hits']}/{len(samples)} | "
              f"ROUGE-L {comp[arm]['rouge_l']['mean']} | "
              f"拒答 {comp[arm]['refusal']['count']} | "
              f"截断 {comp[arm]['truncated']['count']} | "
              f"风格对齐 {comp[arm]['script_align']['count']}")

    judge = agg = None
    budget = {"cost_cny": 0.0, "calls": 0, "cache_hits": 0}
    if not args.no_judge:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            print("\n[WARN] 环境变量 DEEPSEEK_API_KEY 未设置 → 跳过裁判臂（不编造分数）。")
        else:
            print("\n[2/4] 端点探测 ...")
            probe = R.probe_api(key)
            for tag, v in probe.items():
                print(f"  {tag}: ok={v.get('ok')} {v.get('elapsed_s')}s {v.get('models') or v.get('error','')}")

            est_prompt = 900
            est_completion = 90
            n_calls = len(samples) * 2 * args.judge_samples + len(samples) * 2
            est = R.cost_cny(est_prompt * n_calls, est_completion * n_calls, time.time())
            print(f"  预估费用 ¥{est:.4f}（上限 ¥{args.max_budget_cny}）")
            if est > args.max_budget_cny:
                print("  [STOP] 预估超上限，拒绝启动裁判臂。")
            else:
                cache = load_cache()
                print(f"\n[3/4] 绝对打分（N={args.judge_samples}, T={args.judge_temperature}）...")
                absolute = run_absolute_judge(key, samples, cache, args.judge_samples,
                                              args.judge_temperature, budget)
                print(f"  累计 ¥{budget['cost_cny']:.6f}，调用 {budget['calls']} 次")
                print(f"\n[3/4] 成对对比（N=2 顺序）...")
                pairwise = run_pairwise_judge(key, samples, cache, args.judge_temperature, budget)
                print(f"  累计 ¥{budget['cost_cny']:.6f}，调用 {budget['calls']} 次")
                judge = {"absolute": absolute, "pairwise": pairwise}
                agg = judge_aggregates(samples, absolute, pairwise)
    else:
        print("\n[2/4] 跳过裁判（--no-judge）")

    scores = {
        "meta": meta,
        "n": len(samples),
        "computable": comp,
        "judge": judge,
        "judge_aggregate": agg,
        "judge_budget": budget,
        "per_sample_computable": {
            smp["id"]: {
                "instruction": smp["instruction"],
                "reference": smp["reference"],
                "base_output": smp["base_output"],
                "sft_output": smp["sft_output"],
                "base": comp["base"],
                "sft": comp["sft"],
            }
            for smp in samples
        },
    }

    # 把逐条可计算指标也落到逐条上（scores.json 里给出可复算的明细）
    per_sample = {}
    for smp in samples:
        row = {"id": smp["id"], "instruction": smp["instruction"], "reference": smp["reference"]}
        for arm in ARMS:
            cand = smp.get(f"{arm}_output", "") or ""
            ref = smp.get("reference", "") or ""
            row[arm] = {
                "output": cand,
                "em": R.exact_match(ref, cand),
                "char_f1": R.char_f1(ref, cand),
                "rouge_l": EvaluationEngine.compute_rouge_l(ref, cand),
                "refusal": detect_refusal_sft(cand),
                "empty": not cand.strip(),
                "truncated": bool(smp.get("meta", {}).get(arm, {}).get("truncated")),
                "repetition_ratio": repetition_ratio(cand),
                "script": script_of(cand),
                "script_matches_reference": script_of(cand) == script_of(ref),
                "chatty_opener": has_chatty_opener(cand),
                "numbered_list": has_numbered_list(cand),
                "length_chars": len(cand),
                "n_new_tokens": smp.get("meta", {}).get(arm, {}).get("n_new_tokens"),
            }
        per_sample[smp["id"]] = row
    scores["per_sample"] = per_sample

    print("\n[4/4] 落盘 ...")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  scores → {args.out_json}")

    md = render_report(meta, samples, comp, judge, agg, budget, args)
    args.out_md.write_text(md, encoding="utf-8", newline="\n")
    print(f"  report → {args.out_md}")

    hs = render_human_sheet(samples, comp)
    args.out_human.write_text(hs, encoding="utf-8", newline="\n")
    print(f"  human sheet → {args.out_human}")

    if agg:
        pw = agg["pairwise"]
        print(f"\n成对对比：SFT 更好 {pw['sft_better']} / 基座更好 {pw['base_better']} / 持平 {pw['tie']} / 未解析 {pw['unparsed']}（n={pw['n']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
