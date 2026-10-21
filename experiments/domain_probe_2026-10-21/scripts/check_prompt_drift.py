#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核对：data/*.jsonl 的 prompt 是否与 results/*_base_outputs.jsonl 里实际跑的 prompt 一致。

一致的域 → 已有基座输出可复用（只改了 expected，不必重跑）；
不一致的域 → 必须重跑（prompt 变了）。
"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"D:\programmer\LLM\projects\rag-eval-platform\experiments\domain_probe_2026-10-21"
NAMES = ["A_agent_tool_call", "B_doc_extraction", "C_compliance_refusal", "D_intent_slot"]

need_rerun = []
for name in NAMES:
    data = {}
    with io.open(os.path.join(ROOT, "data", name + ".jsonl"), encoding="utf-8") as f:
        for l in f:
            if l.strip():
                o = json.loads(l)
                data[o["id"]] = o["prompt"]
    outs = {}
    with io.open(os.path.join(ROOT, "results", name + "_base_outputs.jsonl"), encoding="utf-8") as f:
        for l in f:
            if l.strip():
                o = json.loads(l)
                if o.get("_record_type") == "sample":
                    outs[o["id"]] = o["prompt"]
    diff = [i for i in data if data[i] != outs.get(i)]
    status = "REUSE" if not diff else "RERUN"
    print(f"{name}: prompt_changed={len(diff)} -> {status} {diff}")
    if diff:
        need_rerun.append(name)

print("\n需要重跑的域：", need_rerun or "（无）")
