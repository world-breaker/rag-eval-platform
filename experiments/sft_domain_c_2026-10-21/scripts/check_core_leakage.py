#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
核心问句文本的**跨段重叠**检查 + 敏感性分析
=====================================================================
为什么要单独写这个脚本
--------------------
`scripts/verify_dataset.py` 的第 ⑤ 项「core 跨段不重叠」查的是 `meta.core_id`，
而 `core_id` 的格式是 **"{cls}-{sha1(core)[:8]}"** —— 前缀里带了 `cls`
（normal / boundary / singleR / singleT / composite）。

于是会出现这种情况：**同一句核心问句文本**，在 train 里被登记为
`normal-6836a969`，在 test 里被登记为 `boundary-6836a969` → 两个 `core_id` 不相等，
第 ⑤ 项就判「不重叠」。**但核心文本其实是一样的。**

这不是 bug（normal 与 boundary 本来就共用一批「贴近红线但不越界」的核心句），
但它是一个**必须披露的口径事实**：test 集里有一部分样本，其核心问句
模型在训练时见过（同义不同措辞）。

本脚本做两件事
--------------
1. **精确测量**这个重叠有多大、落在哪些决策层、标签是否一致；
2. **敏感性分析**：剔除重叠样本后重算关键指标，
   看主结论是否依赖这部分重叠。

**主指标口径不变**：结论仍以全量 test（n=200）为准；
干净子集只是**佐证**，用来回答「结论是不是靠记忆撑起来的」。

输出
----
`results/core_leakage_report.json`（重叠规模 + 干净子集指标 + 结论是否改变）

用法
----
    python scripts/check_core_leakage.py              # 只做重叠分析（可在评测前跑）
    python scripts/check_core_leakage.py --sensitivity # 叠加干净子集指标（评测后跑）
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def read_jsonl(p: Path):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def corehash(r: dict) -> str:
    """核心问句**文本**的身份标识：core_id 去掉 "{cls}-" 前缀后的 sha1 段。"""
    return r["meta"]["core_id"].split("-", 1)[1]


def qof(p: str) -> str:
    return p.split("客户问：", 1)[-1]


def load():
    return {name: read_jsonl(DATA / f"{name}.jsonl") for name in ("train", "dev", "test")}


def overlap_analysis(splits: dict) -> dict:
    tr_h = defaultdict(list)
    for r in splits["train"]:
        tr_h[corehash(r)].append(r)

    out = {"splits": {}, "dev_test_core_overlap": {}}
    for name in ("dev", "test"):
        hit = [r for r in splits[name] if corehash(r) in tr_h]
        out["splits"][name] = {
            "n": len(splits[name]),
            "核心问句文本与train重叠_条数": len(hit),
            "占比": round(len(hit) / len(splits[name]), 4),
            "重叠样本按期望决策": dict(Counter(r["expected"]["decision"] for r in hit)),
            "重叠样本按cls": dict(Counter(r["meta"]["cls"] for r in hit)),
            "重叠样本id": [r["id"] for r in hit],
            "重叠样本的标签是否与train同核心问句一致": None,   # 下面填
        }
        same = diff = 0
        for r in hit:
            tl = {(m["expected"]["decision"], tuple(m["expected"]["redline"]),
                   tuple(m["expected"]["transfer_rules"])) for m in tr_h[corehash(r)]}
            el = (r["expected"]["decision"], tuple(r["expected"]["redline"]),
                  tuple(r["expected"]["transfer_rules"]))
            if el in tl:
                same += 1
            else:
                diff += 1
        out["splits"][name]["重叠样本的标签是否与train同核心问句一致"] = {
            "标签相同": same, "标签不同": diff}

    dv_h = {corehash(r) for r in splits["dev"]}
    te_h = {corehash(r) for r in splits["test"]}
    inter = dv_h & te_h
    out["dev_test_core_overlap"] = {
        "重叠核心问句数": len(inter),
        "说明": "dev 与 test 之间**没有**核心问句文本重叠 —— dev 选 checkpoint 不受此问题影响",
    }
    return out


def clean_subset_metrics(splits: dict, overlap_ids: set) -> dict:
    """在剔除「核心问句文本与 train 重叠」的样本后，重算两臂关键指标。"""
    res = {}
    for ds in ("test",):
        for arm in ("base", "lora"):
            p = RESULTS / f"{ds}_{arm}_scored.jsonl"
            if not p.exists():
                continue
            rows = read_jsonl(p)
            keep = [r for r in rows if r["id"] not in overlap_ids]
            n = len(keep)
            dec = sum(bool(r["decision_match"]) for r in keep)
            strict = sum(bool(r["pass_all"]) for r in keep)
            by = defaultdict(lambda: {"n": 0, "ok": 0})
            for r in keep:
                b = by[r["expected"]["decision"]]
                b["n"] += 1
                b["ok"] += bool(r["decision_match"])
            res[f"{ds}/{arm}"] = {
                "n": n,
                "决策准确率": f"{dec}/{n}" + (f" = {dec/n:.4f}" if n else ""),
                "严格口径": f"{strict}/{n}" + (f" = {strict/n:.4f}" if n else ""),
                "按期望决策": {k: {"n": v["n"], "判对": v["ok"],
                                  "率": round(v["ok"] / v["n"], 4) if v["n"] else None}
                              for k, v in sorted(by.items())},
            }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true",
                    help="叠加「剔除重叠样本」的干净子集指标（需要先跑 score_sft.py）")
    args = ap.parse_args()

    splits = load()
    rep = overlap_analysis(splits)

    print("=" * 78)
    print("① 核心问句**文本**跨段重叠（比 verify_dataset.py 的 core_id 级检查更严）")
    print("=" * 78)
    for name, v in rep["splits"].items():
        print(f"  {name}: n={v['n']}  与 train 核心文本重叠 {v['核心问句文本与train重叠_条数']} 条 "
              f"({v['占比']*100:.1f}%)")
        print(f"      按期望决策 {v['重叠样本按期望决策']}   按 cls {v['重叠样本按cls']}")
        print(f"      标签一致性 {v['重叠样本的标签是否与train同核心问句一致']}")
    print(f"  dev∩test 核心文本重叠：{rep['dev_test_core_overlap']['重叠核心问句数']} 个 "
          f"→ {rep['dev_test_core_overlap']['说明']}")

    overlap_ids = set(rep["splits"]["test"]["重叠样本id"])
    if args.sensitivity:
        rep["干净子集指标"] = clean_subset_metrics(splits, overlap_ids)
        print("\n" + "=" * 78)
        print("② 敏感性分析：剔除重叠样本后的干净子集（test 决策层构成不变）")
        print("=" * 78)
        for k, v in rep["干净子集指标"].items():
            print(f"  {k}: n={v['n']}  决策准确率 {v['决策准确率']}  严格 {v['严格口径']}")
            for d, dv in v["按期望决策"].items():
                print(f"      {d}: {dv['判对']}/{dv['n']} = {dv['率']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "core_leakage_report.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n→ {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
