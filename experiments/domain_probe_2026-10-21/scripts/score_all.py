#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总 4 个域的判定结果 → results/summary.json + results/summary.md
====================================================================

用法
----
  python scripts/score_all.py
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

DOMAINS = [
    ("A", "A_agent_tool_call", "Agent 工具调用", "score_A.py"),
    ("B", "B_doc_extraction", "金融/房产文档结构化抽取", "score_B.py"),
    ("C", "C_compliance_refusal", "合规拒答与风险识别", "score_C.py"),
    ("D", "D_intent_slot", "金融客服意图+槽位", "score_D.py"),
]


def main() -> int:
    for key, name, _, script in DOMAINS:
        print(f"\n===== [{key}] {name} =====")
        r = subprocess.run([sys.executable, str(HERE / script)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-3000:])
            return r.returncode

    summary = {"domains": {}}
    for key, name, title, _ in DOMAINS:
        p = RESULTS / f"{name}_scores.json"
        if p.exists():
            summary["domains"][key] = json.loads(io.open(p, encoding="utf-8").read())
            summary["domains"][key]["title"] = title

    # 统一口径的「主指标」：越低说明基座越弱 → 越值得微调
    primary = {
        "A": ("pass_all_rate", "工具名+参数全对率（能力口径）"),
        "B": ("micro_field_f1", "字段级 micro-F1（能力口径）"),
        "C": ("decision_match_rate", "答/拒/转人工决策准确率"),
        "D": ("intent_match_rate", "意图标签准确率"),
    }
    rows = []
    for key, name, title, _ in DOMAINS:
        d = summary["domains"].get(key, {})
        if key == "A":
            p = d.get("pass_all_rate")
            extra = (f"严格词口径 {d.get('pass_all_strict_rate', 0):.0%}"
                     f" / 严格格式口径 {d.get('pass_format_strict_rate', 0):.0%}")
        elif key == "B":
            p = d.get("micro_field_f1")
            extra = (f"全字段对率 {d.get('pass_all_rate', 0):.0%}"
                     f" / 严格词口径 F1 {d.get('micro_field_f1_strict', 0):.3f}")
        elif key == "C":
            p = d.get("decision_match_rate")
            extra = f"完整通过率 {d.get('pass_all_rate', 0):.0%}"
        else:
            p = d.get("intent_match_rate")
            extra = f"槽位F1 {d.get('slot_f1', 0):.3f} / 完整通过率 {d.get('pass_all_rate', 0):.0%}"
        rows.append({
            "domain": key, "title": title, "n": d.get("n"),
            "primary_metric": primary[key][0],
            "primary_value": p,
            "extra": extra,
            "json_valid_rate": d.get("json_valid_rate"),
            "failure_modes": d.get("failure_modes"),
        })
    summary["comparison"] = rows

    out_json = RESULTS / "summary.json"
    with io.open(out_json, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    md = ["| 域 | 名称 | n | 主指标 | 主指标值 | 补充口径 | JSON 合法率 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        pv = r["primary_value"]
        md.append(f"| {r['domain']} | {r['title']} | {r['n']} | {r['primary_metric']} | "
                  f"{pv:.1%} | {r['extra']} | {r['json_valid_rate']:.1%} |")
    md.append("")
    for r in rows:
        md.append(f"**域 {r['domain']} 失败模式分布**：`{json.dumps(r['failure_modes'], ensure_ascii=False)}`")
        md.append("")

    out_md = RESULTS / "summary.md"
    with io.open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\n✓ {out_json}\n✓ {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
