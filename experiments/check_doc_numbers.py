# -*- coding: utf-8 -*-
"""文档数字自检：把 README / 报告里引用的每个关键数字，从已提交的结果 JSON 重算一遍并比对。

为什么需要它
------------
这个仓库的 README 与报告里全是精确到小数点后四位的数字，而这些数字**全部**来自
`experiments/results/*.json`。文档是人写的、JSON 是脚本出的，两边一旦不同步，
读者拿到的就是过期结论——而"诚实"正是这个作品集的主线。所以把「文档里的数字」
变成**可执行断言**：改了 JSON 或改了文档，只要两者不自洽，脚本就退出码 1。

判定规则（每个 claim 三件事都要成立）
  1. 数字能从 JSON **重算**出来（不是抄进去的常量）；
  2. 重算结果格式化后 == 文档里写的那串字面量（防漂移）；
  3. 那串字面量**确实出现在**指定文档里（防"数字被悄悄删掉/改写"）。

用法
----
    python experiments/check_doc_numbers.py                 # 校验全部 claim
    python experiments/check_doc_numbers.py --verbose       # 逐条打印明细
退出码：0 = 全部自洽；1 = 有 claim 不自洽（打印差异）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(os.environ.get("CHECK_DOC_ROOT") or Path(__file__).resolve().parent.parent)
RESULTS = ROOT / "experiments" / "results"
GEN_JSON = RESULTS / "generation_eval_2026-10-19.json"      # N=1 基准（已提交）
VAR_JSON = RESULTS / "judge_variance_2026-10-20.json"        # 本地裁判 N=7 方差
REF_JSON = RESULTS / "reference_robustness_2026-10-20.json"  # 参照系 N=3 稳健性

DIMS = ("faithfulness", "relevance", "completeness")


# ----------------------------------------------------------------- 基础工具

def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"缺少结果文件：{path}\n"
            f"（该文件是本仓库的证据产物，必须随文档一起提交）"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


class Sample:
    """一份结果文件里的样本视图。"""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.samples = data["samples"]
        self.ids = [str(s["id"]) for s in self.samples]
        self.by_id = {str(s["id"]): s for s in self.samples}

    def arm(self, arm: str, dim: str) -> List[float]:
        out: List[float] = []
        for s in self.samples:
            if arm == "rule":
                out.append(float(s["rule"]["rule_faithfulness"]))
            else:
                out.append(float(s["judge"][arm]["scores"][dim]))
        return out

    def api_scores(self, dim: str) -> List[float]:
        return [float(s["judge"]["api"]["scores"][dim]) for s in self.samples]


# ----------------------------------------------------------------- 重算函数

def _agree(a: List[float], b: List[float]) -> Dict[str, float]:
    n = len(a)
    return {
        "mean_a": statistics.mean(a),
        "exact": sum(1 for x, y in zip(a, b) if x == y) / n,
        "within1": sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / n,
        "mae": sum(abs(x - y) for x, y in zip(a, b)) / n,
        "r": _pearson(a, b),
    }


def _seg(scores: Dict[str, float], local: Dict[str, float], lo: int, hi: int) -> Dict[str, float]:
    ids = [i for i in scores if lo <= scores[i] <= hi]
    return {
        "n": len(ids),
        "strong_mean": statistics.mean([scores[i] for i in ids]) if ids else float("nan"),
        "local_mean": statistics.mean([local[i] for i in ids]) if ids else float("nan"),
    }


# ----------------------------------------------------------------- claim 定义

class Claim:
    """一条「文档里写的数字 = 从 JSON 重算的数字」断言。"""

    def __init__(
        self,
        cid: str,
        docs: List[str],
        needle: str,
        value: float,
        fmt: str = "{:.4f}",
        note: str = "",
    ):
        self.cid = cid
        self.docs = docs
        self.needle = needle
        self.value = value
        self.fmt = fmt
        self.note = note

    def _needle_spec(self) -> Tuple[float, int]:
        """从字面量反推「期望值 + 精度」。

        文档里的数字精度不统一（r 写两位、MAE 写四位、样本数写整数），
        所以判据不是"格式串完全相等"，而是"**按字面量的小数位数四舍五入后相等**"：
            "-0.03"  → 期望 -0.03，精度 2 位
            "0.1"    → 期望  0.10，精度 1 位
            "38.4%"  → 期望  0.384，精度 1 位
        """
        s = self.needle.strip()
        pct = s.endswith("%")
        if pct:
            s = s[:-1]
        if "." in s:
            head, frac = s.split(".", 1)
            decimals = len(frac)
        else:
            decimals = 0
        value = float(s)
        if pct:
            value = value / 100.0
        return value, decimals

    def recomputed_text(self) -> str:
        value, decimals = self._needle_spec()
        return f"{value:.{decimals}f}" + ("%" if self.needle.strip().endswith("%") else "")

    def check(self) -> Tuple[bool, str]:
        expected, decimals = self._needle_spec()
        tol = 0.5 * (10 ** -decimals) + 1e-12
        if abs(self.value - expected) > tol:
            return False, (f"重算值 {self.value:.6f} 与文档写的 {self.needle} "
                           f"（按 {decimals} 位小数）不符")
        found = False
        missing = []
        for d in self.docs:
            p = ROOT / d
            if not p.exists():
                missing.append(f"{d}(文件不存在)")
                continue
            # 文档里用的是 Unicode 减号 U+2212，比对前统一成 ASCII
            body = p.read_text(encoding="utf-8").replace("−", "-")
            if self.needle in body:
                found = True
            else:
                missing.append(d)
        if not found:
            return False, "字面量未出现于：" + "、".join(missing)
        return True, "一致"


def build_claims() -> List[Claim]:
    gen = Sample(_load(GEN_JSON))
    var = _load(VAR_JSON)
    ref = _load(REF_JSON)

    claims: List[Claim] = []
    docs_main = ["README.md", "experiments/results/generation_eval_2026-10-19.md"]

    # --- 臂 B（本地 1.5B）vs 臂 C（强裁判）：faithfulness ---
    local = gen.arm("local", "faithfulness")
    api = gen.api_scores("faithfulness")
    ag = _agree(local, api)
    claims.append(Claim("local_vs_api.faithfulness.mae", docs_main, "1.8182", ag["mae"], note="臂B vs 臂C MAE"))
    claims.append(Claim("local_vs_api.faithfulness.r", docs_main, "-0.03", ag["r"], note="臂B vs 臂C r"))
    claims.append(Claim("local_vs_api.faithfulness.exact", docs_main, "0.2121", ag["exact"], note="臂B vs 臂C 精确一致"))
    claims.append(Claim("local_vs_api.faithfulness.within1", docs_main, "0.4848", ag["within1"], note="臂B vs 臂C ±1 一致"))

    # --- 臂 A（规则判据）vs 臂 C ---
    rule = gen.arm("rule", "faithfulness")
    agr = _agree(rule, api)
    claims.append(Claim("rule_vs_api.faithfulness.mae", docs_main, "0.9152", agr["mae"], note="臂A vs 臂C MAE"))
    claims.append(Claim("rule_vs_api.faithfulness.r", docs_main, "0.59", agr["r"], note="臂A vs 臂C r"))
    claims.append(Claim("rule_vs_api.faithfulness.exact", docs_main, "0.5455", agr["exact"], note="臂A vs 臂C 精确一致"))
    claims.append(Claim("rule_vs_api.faithfulness.within1", docs_main, "0.7576", agr["within1"], note="臂A vs 臂C ±1 一致"))

    # --- 另外两个维度（臂 B vs 臂 C）---
    for dim, mae, r in (("relevance", "1.7879", "0.1"), ("completeness", "1.6667", "-0.06")):
        a = gen.arm("local", dim)
        b = gen.api_scores(dim)
        m = _agree(a, b)
        claims.append(Claim(f"local_vs_api.{dim}.mae", docs_main, mae, m["mae"], note=f"臂B vs 臂C {dim} MAE"))
        claims.append(Claim(f"local_vs_api.{dim}.r", docs_main, r, m["r"], note=f"臂B vs 臂C {dim} r"))

    # --- 中庸化回归分段（N=1 参照）---
    api_by_id = {str(s["id"]): float(s["judge"]["api"]["scores"]["faithfulness"]) for s in gen.samples}
    loc_by_id = {str(s["id"]): float(s["judge"]["local"]["scores"]["faithfulness"]) for s in gen.samples}
    low = _seg(api_by_id, loc_by_id, 1, 2)
    high = _seg(api_by_id, loc_by_id, 4, 5)
    claims.append(Claim("segment.low.n", docs_main, "6", low["n"], fmt="{:.0f}", note="判低组样本数"))
    claims.append(Claim("segment.low.strong_mean", docs_main, "1.667", low["strong_mean"], fmt="{:.3f}", note="判低组强裁判均分"))
    claims.append(Claim("segment.low.local_mean", docs_main, "3.167", low["local_mean"], fmt="{:.3f}", note="判低组本地均分"))
    claims.append(Claim("segment.high.n", docs_main, "27", high["n"], fmt="{:.0f}", note="判高组样本数"))
    claims.append(Claim("segment.high.strong_mean", docs_main, "4.852", high["strong_mean"], fmt="{:.3f}", note="判高组强裁判均分"))
    claims.append(Claim("segment.high.local_mean", docs_main, "3.111", high["local_mean"], fmt="{:.3f}", note="判高组本地均分"))

    # --- 本地裁判 N=7 方差报告 ---
    docs_var = ["experiments/results/judge_variance_2026-10-20.md", "README.md"]
    noise = var["summary"]["item_noise"]
    for dim, needle in (("faithfulness", "0.414"), ("relevance", "0.322"), ("completeness", "0.285")):
        claims.append(Claim(
            f"local_variance.item_sd.{dim}", docs_var, needle,
            float(noise["per_cell_sd_mean"][dim]), fmt="{:.3f}", note="本地裁判 item 级 sd 均值"))
    claims.append(Claim(
        "local_variance.identical_ratio", docs_var, "38.4%",
        float(noise["identical_ratio"]), fmt="{:.1%}", note="7 次采样完全一致比例"))
    claims.append(Claim(
        "local_variance.range_mean", docs_var, "0.86",
        float(noise["range_mean"]), fmt="{:.2f}", note="极差均值"))
    boot = var["summary"]["bootstrap_judge_resample"]["estimators"]
    for k, needle in (("single", "0.0869"), ("median_of_3", "0.0745"),
                      ("median_of_7", "0.0619"), ("median_of_15", "0.0513")):
        claims.append(Claim(
            f"local_variance.mae_sd.{k}", docs_var, needle,
            float(boot[k]["mae_sd"]), note=f"{k} 的 MAE 抽样 sd"))
    m7 = var["summary"]["aggregate_vs_strong_judge_faithfulness"]["median_of_7"]
    claims.append(Claim("local_variance.median7.mae", docs_var, "1.9394", float(m7["mae"]), note="median-of-7 的 MAE"))
    claims.append(Claim("local_variance.median7.r", docs_var, "-0.1436", float(m7["pearson_r"]), note="median-of-7 的 r"))

    # --- 参照系 N=3 稳健性报告 ---
    docs_ref = ["experiments/results/reference_robustness_2026-10-20.md", "README.md"]
    rs = ref["summary"]
    rob = rs["robustness_vs_reference"]["faithfulness"]
    claims.append(Claim("ref_robust.local.mae", docs_ref, "1.9394", float(rob["local_vs_n3_median"]["mae"]), note="本地 vs N=3 参照 MAE"))
    claims.append(Claim("ref_robust.local.r", docs_ref, "-0.1595", float(rob["local_vs_n3_median"]["r"]), note="本地 vs N=3 参照 r"))
    claims.append(Claim("ref_robust.rule.mae", docs_ref, "1.0061", float(rob["rule_vs_n3_median"]["mae"]), note="规则臂 vs N=3 参照 MAE"))
    claims.append(Claim("ref_robust.rule.r", docs_ref, "0.5485", float(rob["rule_vs_n3_median"]["r"]), note="规则臂 vs N=3 参照 r"))
    noise_api = rs["item_noise_api"]
    claims.append(Claim("ref_robust.api_item_sd.faithfulness", docs_ref, "0.100",
                        float(noise_api["per_dim_item_sd_mean_ddof0"]["faithfulness"]), fmt="{:.3f}", note="强裁判 item 级 sd"))
    claims.append(Claim("ref_robust.api_identical_ratio", docs_ref, "71.7%",
                        float(noise_api["identical_ratio"]), fmt="{:.1%}", note="强裁判 3 次完全一致比例"))
    claims.append(Claim("ref_robust.local_same_n_identical_ratio", docs_ref, "55.6%",
                        float(rs["item_noise_local"]["identical_ratio"]), fmt="{:.1%}", note="同一次运行内本地 N=3 的完全一致比例"))

    return claims


# ----------------------------------------------------------------- 主流程

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="校验文档里的数字与已提交结果 JSON 是否自洽")
    ap.add_argument("--verbose", action="store_true", help="逐条打印明细")
    args = ap.parse_args(argv)

    try:
        claims = build_claims()
    except FileNotFoundError as e:
        print(f"[FAIL] {e}")
        return 1
    except Exception as e:  # noqa: BLE001 —— claim 定义与结果 JSON 结构不符时必须大声失败
        print(f"[FAIL] claim 定义与结果 JSON 的结构不符：{type(e).__name__}: {e}")
        print("       （多半是结果 JSON 的字段改了，或某条 claim 引用了不存在的键）")
        return 1

    print(f"文档数字自检：{len(claims)} 条 claim，结果文件来自 {RESULTS.name}/")
    print("-" * 78)
    failed = 0
    for c in claims:
        ok, why = c.check()
        if not ok:
            failed += 1
        if args.verbose or not ok:
            print(f"  [{'PASS' if ok else 'FAIL'}] {c.cid:<44} {why}")
    print("-" * 78)
    if failed:
        print(f"结果：{failed}/{len(claims)} 条不自洽 —— 文档与结果 JSON 已漂移，请修文档或重跑实验。")
        return 1
    print(f"结果：{len(claims)}/{len(claims)} 条自洽（文档里的每个数字都能从已提交结果 JSON 重算出来）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
