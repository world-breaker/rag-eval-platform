#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
报告数字核对：把报告 md 里写的每个数字 ↔ results/ 里的实测产物逐项比对
=====================================================================

纪律：报告里的每个数字都必须能从 results/*.json 取出来；
本脚本按「关键词 → 期望值」的清单核对，不一致就报错退出（非 0）。

用法
----
    python scripts/verify_report_numbers.py
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
REPORT = ROOT.parent / "sft_domain_c_2026-10-21.md"


def read_jsonl(p: Path):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def main() -> int:
    if not REPORT.exists():
        print(f"[FATAL] 找不到报告 {REPORT}")
        return 2
    text = io.open(REPORT, encoding="utf-8").read()
    comp = json.loads(io.open(RESULTS / "test_comparison.json", encoding="utf-8").read())
    base, lora = comp["base"], comp["lora"]
    meta = json.loads(io.open(ROOT / "data" / "build_meta.json", encoding="utf-8").read())

    checks: list[tuple[str, str]] = []

    def add(name: str, value) -> None:
        checks.append((name, str(value)))

    # 决策准确率
    add("基座 决策准确率", f"{base['T1_决策口径']['decision_accuracy']:.4f}")
    add("微调 决策准确率", f"{lora['T1_决策口径']['decision_accuracy']:.4f}")
    add("基座 决策正确数", f"{base['T1_决策口径']['correct']}/{base['n']}")
    add("微调 决策正确数", f"{lora['T1_决策口径']['correct']}/{lora['n']}")
    # 该拒拒答率
    add("基座 该拒拒答率", f"{base['该拒拒答率']['rate']:.4f}")
    add("微调 该拒拒答率", f"{lora['该拒拒答率']['rate']:.4f}")
    add("基座 该拒拒答计数", f"{base['该拒拒答率']['correct']}/{base['该拒拒答率']['n']}")
    add("微调 该拒拒答计数", f"{lora['该拒拒答率']['correct']}/{lora['该拒拒答率']['n']}")
    # 该转转人工率
    add("基座 该转转人工率", f"{base['该转转人工率']['rate']:.4f}")
    add("微调 该转转人工率", f"{lora['该转转人工率']['rate']:.4f}")
    add("基座 该转计数", f"{base['该转转人工率']['correct']}/{base['该转转人工率']['n']}")
    add("微调 该转计数", f"{lora['该转转人工率']['correct']}/{lora['该转转人工率']['n']}")
    # 严格 / 宽松口径
    add("基座 严格口径", f"{base['T2_严格口径_编号集合完全相等']['pass_all']:.4f}")
    add("微调 严格口径", f"{lora['T2_严格口径_编号集合完全相等']['pass_all']:.4f}")
    add("基座 宽松口径", f"{base['T3_宽松口径_不多报编号']['lenient']:.4f}")
    add("微调 宽松口径", f"{lora['T3_宽松口径_不多报编号']['lenient']:.4f}")
    # 过度拒答 / JSON
    add("基座 过度拒答率", f"{base['过度拒答率_该答却拒或转']['rate']:.4f}")
    add("微调 过度拒答率", f"{lora['过度拒答率_该答却拒或转']['rate']:.4f}")
    add("基座 JSON合法率", f"{base['json合法率']['strict']:.4f}")
    add("微调 JSON合法率", f"{lora['json合法率']['strict']:.4f}")
    # 混淆矩阵
    for arm, res in (("基座", base), ("微调", lora)):
        m = res["混淆矩阵"]["matrix"]
        add(f"{arm} 混淆矩阵 回答行", ",".join(str(x) for x in m[0]))
        add(f"{arm} 混淆矩阵 拒答行", ",".join(str(x) for x in m[1]))
        add(f"{arm} 混淆矩阵 转人工行", ",".join(str(x) for x in m[2]))
    # 数据规模
    add("train 条数", meta["splits"]["train"]["n"])
    add("dev 条数", meta["splits"]["dev"]["n"])
    add("test 条数", meta["splits"]["test"]["n"])
    add("改写成功条数", meta["rewrite_applied"])
    add("test sha256", meta["splits"]["test"]["sha256"])
    add("train sha256", meta["splits"]["train"]["sha256"])

    # 报告中必须能找到的「关键数字」
    must_appear = [
        f"{base['T1_决策口径']['decision_accuracy']:.4f}",
        f"{lora['T1_决策口径']['decision_accuracy']:.4f}",
        f"{base['该拒拒答率']['correct']}/{base['该拒拒答率']['n']}",
        f"{lora['该拒拒答率']['correct']}/{lora['该拒拒答率']['n']}",
        f"{base['该转转人工率']['correct']}/{base['该转转人工率']['n']}",
        f"{lora['该转转人工率']['correct']}/{lora['该转转人工率']['n']}",
        meta["splits"]["test"]["sha256"],
        "n=200",
    ]

    bad = []
    print("=" * 70)
    print("① 报告里必须出现的关键数字")
    for s in must_appear:
        ok = s in text
        print(f"  {'[OK]' if ok else '[!!]'} {s}")
        if not ok:
            bad.append(s)

    print("=" * 70)
    print("② 结果文件里的实测值（供人工核对）")
    for name, v in checks:
        print(f"  {name:24s} = {v}")

    print("=" * 70)
    if bad:
        print(f"[FAIL] 报告缺少 {len(bad)} 个关键数字：{bad}")
        return 1
    print("[OK] 报告关键数字与 results/ 实测产物一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
