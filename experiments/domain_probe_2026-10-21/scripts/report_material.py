#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""报告用素材：① 含 strict_terms 的样本清单（判据松紧的两档差异）② 各域典型错例。"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"D:\programmer\LLM\projects\rag-eval-platform\experiments\domain_probe_2026-10-21"

print("=" * 72)
print("① 含 strict_terms（两档判据）的样本")
print("=" * 72)
for name in ["A_agent_tool_call", "B_doc_extraction", "C_compliance_refusal", "D_intent_slot"]:
    with io.open(os.path.join(ROOT, "data", name + ".jsonl"), encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            o = json.loads(l)
            exp = o["expected"]
            if name.startswith("A"):
                specs = list(exp["arguments"].values())
            elif name.startswith("B"):
                specs = list(exp["fields"].values())
            else:
                continue
            for sp in specs:
                if sp.get("strict_terms"):
                    print(f"  {o['id']:6s} {sp.get('type')}: 语义必要词={sp['terms']}  严格逐字词={sp['strict_terms']}")

print()
print("=" * 72)
print("② 各域典型错例（原文摘录）")
print("=" * 72)
WANT = {
    "A_agent_tool_call": ["A-09", "A-17"],
    "B_doc_extraction": ["B-13", "B-22", "B-24"],
    "C_compliance_refusal": ["C-11", "C-14", "C-20", "C-24"],
    "D_intent_slot": ["D-09", "D-10", "D-04"],
}
for name, ids in WANT.items():
    with io.open(os.path.join(ROOT, "results", name + "_scored.jsonl"), encoding="utf-8") as f:
        rows = {}
        for l in f:
            if l.strip():
                o = json.loads(l)
                if o.get("_record_type") == "sample":
                    rows[o["id"]] = o
    with io.open(os.path.join(ROOT, "data", name + ".jsonl"), encoding="utf-8") as f:
        prompts = {}
        for l in f:
            if l.strip():
                o = json.loads(l)
                prompts[o["id"]] = o["prompt"]
    for i in ids:
        r = rows[i]
        v = r["verdict"]
        p = prompts[i]
        tail = p.split("用户请求：")[-1].split("客户问：")[-1].split("用户问询：")[-1].split("文本片段：")[-1]
        print(f"\n--- {i}  failure_mode={v.get('failure_mode')}")
        print("  问/文  :", tail[:300].replace("\n", " "))
        print("  expected:", json.dumps(r["expected"], ensure_ascii=False)[:300])
        print("  raw     :", r["raw_output"][:420].replace("\n", " "))
        print("  判定    :", {k: v[k] for k in ("tool_match", "args_match", "decision_match",
                                              "redline_match", "transfer_match", "intent_match",
                                              "got_decision", "got_intent") if k in v})
