#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出各域的失败样例（供报告引用）。"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"D:\programmer\LLM\projects\rag-eval-platform\experiments\domain_probe_2026-10-21\results"
NAMES = ["A_agent_tool_call", "B_doc_extraction", "C_compliance_refusal", "D_intent_slot"]
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 99

for name in NAMES:
    print("=" * 70)
    print(name)
    print("=" * 70)
    rows = []
    with io.open(os.path.join(ROOT, name + "__scored.jsonl".replace("__", "_")), encoding="utf-8") as f:
        for l in f:
            if l.strip():
                o = json.loads(l)
                if o.get("_record_type") == "sample":
                    rows.append(o)
    shown = 0
    for r in rows:
        v = r["verdict"]
        if v.get("pass_all"):
            continue
        shown += 1
        if shown > LIMIT:
            break
        print(f"\n--- {r['id']}  failure_mode={v.get('failure_mode')}")
        print("  notes   :", r["notes"][:110])
        print("  expected:", json.dumps(r["expected"], ensure_ascii=False)[:400])
        print("  raw     :", r["raw_output"][:520].replace("\n", " "))
        if name == "B_doc_extraction":
            for fn, fr in (v.get("fields") or {}).items():
                if not fr["ok"]:
                    print(f"    字段 {fn}: got={fr.get('got')!r} reason={fr['reason']}")
        if name == "D_intent_slot":
            print("    got_intent:", v.get("got_intent"), " extra_slots:", v.get("extra_slot_keys"))
            for sn, sr in (v.get("slots") or {}).items():
                print(f"    槽位 {sn}: got={sr.get('got')!r} reason={sr['reason']}")
        if name == "C_compliance_refusal":
            print("    got_decision:", v.get("got_decision"),
                  " redline:", v.get("got_redline"), " transfer:", v.get("got_transfer_rules"))
    print(f"\n[失败总数 {shown} / {len(rows)}]")
