#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Wilson 置信区间（95%）—— 给报告里的关键比例配区间
=====================================================================
为什么需要
----------
probe 报告已经指出：「n 小，比例的 95% 置信区间很宽……绝对值不可当精确值引用」。
本轮 test n=200、其中「该拒」只有 80 条，单点比例仍然会误导读者，
所以每个关键比例都必须**同时给绝对计数和区间**。

口径
----
- 区间只算**比例**（Wilson score interval，非正态近似，小样本也不炸）；
- 每个区间都带 n、分子分母、以及源文件路径；
- 底稿：`results/{dataset}_comparison.json`（由 `scripts/score_sft.py` 生成）。

用法
----
    python scripts/wilson_ci.py --dataset test
    python scripts/wilson_ci.py --dataset probe
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 区间（95%，z=1.96）。返回 (下界, 上界)。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def entry(label: str, k: int, n: int, source: str) -> dict:
    lo, hi = wilson(k, n)
    return {
        "指标": label, "分子": k, "分母": n,
        "比例": round(k / n, 4) if n else 0.0,
        "Wilson95_下界": round(lo, 4), "Wilson95_上界": round(hi, 4),
        "源文件": source,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="test")
    args = ap.parse_args()
    ds = args.dataset
    src = str(RESULTS / f"{ds}_comparison.json")
    comp = json.loads(io.open(src, encoding="utf-8").read())

    rows = []
    for arm, res in (("base", comp["base"]), ("lora", comp["lora"])):
        n = res["n"]
        arm_src = f"{src} [{arm}]"
        rows.append(entry(f"{arm} 决策准确率", res["T1_决策口径"]["correct"], n, arm_src))
        rows.append(entry(f"{arm} 严格口径 pass_all",
                          res["T2_严格口径_编号集合完全相等"]["correct"], n, arm_src))
        rows.append(entry(f"{arm} 该拒拒答率",
                          res["该拒拒答率"]["correct"], res["该拒拒答率"]["n"], arm_src))
        rows.append(entry(f"{arm} 该转转人工率",
                          res["该转转人工率"]["correct"], res["该转转人工率"]["n"], arm_src))
        rows.append(entry(f"{arm} 过度拒答率",
                          res["过度拒答率_该答却拒或转"]["count"],
                          res["过度拒答率_该答却拒或转"]["n"], arm_src))
        rows.append(entry(f"{arm} JSON 合法率",
                          round(res["json合法率"]["strict"] * n), n, arm_src))
        for dec, v in res["T1_决策口径"]["by_expected"].items():
            rows.append(entry(f"{arm} 按期望[{dec}] 决策准确率", v["correct"], v["n"], arm_src))

    out = {
        "dataset": ds,
        "source": src,
        "method": "Wilson score interval, z=1.959963984540054 (95%)",
        "rows": rows,
    }
    p = RESULTS / f"{ds}_wilson_ci.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'指标':<42s} {'计数':>9s} {'比例':>8s}  {'Wilson 95% CI':>18s}")
    for r in rows:
        print(f"{r['指标']:<42s} {str(r['分子'])+'/'+str(r['分母']):>9s} "
              f"{r['比例']*100:>7.1f}%  [{r['Wilson95_下界']*100:>5.1f}%, {r['Wilson95_上界']*100:>5.1f}%]")
    print(f"\n→ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
