#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核对 4 个域样本的字段合规性与 source/notes 覆盖。"""
import collections
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
D = r"D:\programmer\LLM\projects\rag-eval-platform\experiments\domain_probe_2026-10-21\data"
NAMES = ["A_agent_tool_call", "B_doc_extraction", "C_compliance_refusal", "D_intent_slot"]
FIELDS = {"id", "prompt", "expected", "source", "notes"}

all_ok = True
for n in NAMES:
    with io.open(os.path.join(D, n + ".jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    bad = [r["id"] for r in rows if set(r) != FIELDS]
    empty = [r["id"] for r in rows if not r["source"].strip() or not r["notes"].strip()]
    ids = [r["id"] for r in rows]
    dup = len(ids) != len(set(ids))
    ok = not bad and not empty and not dup
    all_ok = all_ok and ok
    print(f"{n}: n={len(rows)}  字段合规={not bad}  id唯一={not dup}  source/notes非空={not empty}")
    c = collections.Counter(r["source"] for r in rows)
    for k, v in c.items():
        print(f"    {v:2d} x {k}")
    # 每条 expected 必须非空
    noexp = [r["id"] for r in rows if not r["expected"]]
    print(f"    expected 非空: {not noexp} {noexp or ''}")
    all_ok = all_ok and not noexp

print("\n全部合规" if all_ok else "\n★ 有不合规项")
sys.exit(0 if all_ok else 1)
