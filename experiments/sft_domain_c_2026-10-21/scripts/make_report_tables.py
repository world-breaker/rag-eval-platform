#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
从 results/*.json 直接生成报告里的 Markdown 表格（避免手抄数字出错）
=====================================================================

用法
----
    python scripts/make_report_tables.py --dataset test
    python scripts/make_report_tables.py --dataset probe
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

DEC_ZH = {"回答": "答（该答）", "拒答": "拒（该拒）", "转人工": "转（该转）"}
OTHER_ZH = "无法解析/其它"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="test")
    args = ap.parse_args()
    ds = args.dataset
    comp = json.loads(io.open(RESULTS / f"{ds}_comparison.json", encoding="utf-8").read())
    base, lora = comp["base"], comp["lora"]
    n = base["n"]

    print("### 表 1 · 主指标（两臂解码参数逐项相同，唯一变量 = 是否装 LoRA）\n")
    print("| 指标 | 口径 | 基座 Qwen2.5-1.5B-Instruct | 微调后（+LoRA r=8） |")
    print("|---|---|---|---|")

    def line(label, fmt, f):
        b, l = f(base), f(lora)
        print(f"| {label} | {fmt} | {b} | {l} |")

    line("决策准确率（答/拒/转）", "主指标",
         lambda r: f"{pct(r['T1_决策口径']['decision_accuracy'])}（{r['T1_决策口径']['correct']}/{r['n']}）")
    line("**该拒拒答率**", "**关键指标**",
         lambda r: f"{pct(r['该拒拒答率']['rate'])}（{r['该拒拒答率']['correct']}/{r['该拒拒答率']['n']}）")
    line("该转转人工率", "业务指标",
         lambda r: f"{pct(r['该转转人工率']['rate'])}（{r['该转转人工率']['correct']}/{r['该转转人工率']['n']}）")
    line("过度拒答率（该答却拒/转）", "业务指标（越低越好）",
         lambda r: f"{pct(r['过度拒答率_该答却拒或转']['rate'])}（{r['过度拒答率_该答却拒或转']['count']}/{r['过度拒答率_该答却拒或转']['n']}）")
    line("严格口径（决策对 + 红线/转人工编号集合**全等**）", "probe 口径 pass_all",
         lambda r: f"{pct(r['T2_严格口径_编号集合完全相等']['pass_all'])}（{r['T2_严格口径_编号集合完全相等']['correct']}/{r['n']}）")
    line("宽松口径（决策对 + **不多报**编号）", "编号幻觉口径",
         lambda r: f"{pct(r['T3_宽松口径_不多报编号']['lenient'])}（{r['T3_宽松口径_不多报编号']['correct']}/{r['n']}）")
    line("JSON 合法率（严格解析）", "格式纪律",
         lambda r: f"{pct(r['json合法率']['strict'])}（{r['json合法率']['strict'] * r['n']:.0f}/{r['n']}）")
    line("编号幻觉率（多报红线）", "越低越好",
         lambda r: pct(r["编号错误率"]["红线幻觉率_多报"]))
    line("编号漏报率（少报红线）", "越低越好",
         lambda r: pct(r["编号错误率"]["红线漏报率_少报"]))
    line("转人工编号幻觉率（多报）", "越低越好",
         lambda r: pct(r["编号错误率"]["转人工幻觉率_多报"]))

    print("\n### 表 2 · 分期望决策的准确率\n")
    print("| 期望决策 | n | 基座 | 微调后 |")
    print("|---|---|---|---|")
    for dec in ("回答", "拒答", "转人工"):
        be = base["T1_决策口径"]["by_expected"].get(dec, {})
        le = lora["T1_决策口径"]["by_expected"].get(dec, {})
        print(f"| {DEC_ZH[dec]} | {be.get('n', 0)} | "
              f"{be.get('correct', 0)}/{be.get('n', 0)} = {pct(be.get('rate', 0))} | "
              f"{le.get('correct', 0)}/{le.get('n', 0)} = {pct(le.get('rate', 0))} |")

    print("\n### 表 3 · 混淆矩阵（行 = 期望，列 = 实际）\n")
    for arm, res in (("基座", base), ("微调后", lora)):
        m = res["混淆矩阵"]["matrix"]
        labels = res["混淆矩阵"]["labels"]
        print(f"\n**{arm}**\n")
        print("| 期望＼实际 | " + " | ".join(labels) + " |")
        print("|---" * (len(labels) + 1) + "|")
        for lab, r in zip(labels, m):
            print(f"| **{lab}** | " + " | ".join(str(x) for x in r) + " |")

    print("\n### 表 4 · 按编号分组的召回率（防止「只学会了拒 R1」）\n")
    print("| 编号 | n（期望命中该编号的条数） | 基座 召回 | 微调后 召回 | 微调后 编号集合全等 |")
    print("|---|---|---|---|---|")
    for code in sorted(set(list(base["按编号分组"].keys()) + list(lora["按编号分组"].keys()))):
        b = base["按编号分组"].get(code, {})
        l = lora["按编号分组"].get(code, {})
        print(f"| {code} | {l.get('n', b.get('n', 0))} | "
              f"{pct(b.get('召回率', 0))} | {pct(l.get('召回率', 0))} | "
              f"{pct(l.get('集合完全相等率', 0))} |")

    print("\n### 表 5 · 失败模式分布\n")
    print("| 失败模式 | 基座 | 微调后 |")
    print("|---|---|---|")
    keys = sorted(set(list(base["failure_modes"].keys()) + list(lora["failure_modes"].keys())))
    for k in keys:
        print(f"| {k or 'PASS（完全通过）'} | {base['failure_modes'].get(k, 0)} | "
              f"{lora['failure_modes'].get(k, 0)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
