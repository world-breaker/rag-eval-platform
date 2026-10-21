#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""报告数字核对：把报告里引用的每个数字与 results/ 下的产物逐项对照。"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"D:\programmer\LLM\projects\rag-eval-platform\experiments\domain_probe_2026-10-21"
R = os.path.join(ROOT, "results")


def load(name):
    with io.open(os.path.join(R, name), encoding="utf-8") as f:
        return json.load(f)


def rows(name):
    out = []
    with io.open(os.path.join(R, name), encoding="utf-8") as f:
        for l in f:
            if l.strip():
                o = json.loads(l)
                if o.get("_record_type") == "sample":
                    out.append(o)
    return out


ok = True


def chk(label, got, want):
    global ok
    good = (got == want)
    ok = ok and good
    print(f"  [{'OK ' if good else 'BAD'}] {label}: 实际={got} 报告={want}")


print("=== A ===")
a = load("A_agent_tool_call_scores.json")
chk("n", a["n"], 24)
chk("pass_all", a["pass_all"], 21)
chk("pass_all_rate", a["pass_all_rate"], 0.875)
chk("pass_all_strict", a["pass_all_strict"], 20)
chk("pass_all_strict_rate", a["pass_all_strict_rate"], round(20 / 24, 4))
chk("pass_format_strict", a["pass_format_strict"], 18)
chk("pass_format_strict_rate", a["pass_format_strict_rate"], 0.75)
chk("json_valid", a["json_valid"], 22)
chk("json_valid_rate", a["json_valid_rate"], 0.9167)
chk("failure_modes", a["failure_modes"], {"PASS": 21, "json_invalid": 2, "tool_name_mismatch": 1})

print("=== B ===")
b = load("B_doc_extraction_scores.json")
chk("n", b["n"], 24)
chk("total_fields", b["total_fields"], 71)
chk("field_hits", b["field_hits"], 57)
chk("micro_field_f1", b["micro_field_f1"], 0.8028)
chk("micro_field_f1_strict", b["micro_field_f1_strict"], 0.7606)
chk("pass_all", b["pass_all"], 13)
chk("pass_all_rate", b["pass_all_rate"], 0.5417)
chk("json_valid_rate", b["json_valid_rate"], 1.0)
chk("failure_modes", b["failure_modes"], {"PASS": 13, "field_wrong": 11})

print("=== C ===")
c = load("C_compliance_refusal_scores.json")
chk("n", c["n"], 24)
chk("decision_match", c["decision_match"], 10)
chk("decision_match_rate", c["decision_match_rate"], 0.4167)
chk("pass_all", c["pass_all"], 9)
chk("pass_all_rate", c["pass_all_rate"], 0.375)
chk("redline_match", c["redline_match"], 12)
chk("redline_match_rate", c["redline_match_rate"], 0.5)
chk("json_valid", c["json_valid"], 21)
chk("json_valid_rate", c["json_valid_rate"], 0.875)
dbe = c["decision_accuracy_by_expected"]
chk("回答 n/correct", (dbe["回答"]["n"], dbe["回答"]["correct"]), (9, 9))
chk("拒答 n/correct", (dbe["拒答"]["n"], dbe["拒答"]["correct"]), (10, 0))
chk("转人工 n/correct", (dbe["转人工"]["n"], dbe["转人工"]["correct"]), (5, 1))
chk("failure_modes", c["failure_modes"], {"PASS": 9, "should_refuse_but_answered": 9,
                                          "json_invalid": 3, "should_transfer_but_answered": 2,
                                          "transfer_mismatch": 1})
# 报告里说「C-24 决策对但编号错」→ transfer_mismatch 应恰为 1 条
tm = [r["id"] for r in rows("C_compliance_refusal_scored.jsonl")
      if r["verdict"].get("failure_mode") == "transfer_mismatch"]
chk("transfer_mismatch 样本", tm, ["C-24"])
# 报告里说「C-15/C-20/C-22 是 JSON 非法」
ji = sorted(r["id"] for r in rows("C_compliance_refusal_scored.jsonl")
            if r["verdict"].get("failure_mode") == "json_invalid")
chk("json_invalid 样本", ji, ["C-15", "C-20", "C-22"])
# redline_match=12 的来源核对
rl_ok = [r["id"] for r in rows("C_compliance_refusal_scored.jsonl") if r["verdict"]["redline_match"]]
print(f"       redline_match 命中 {len(rl_ok)} 条: {rl_ok}")

print("=== D ===")
d = load("D_intent_slot_scores.json")
chk("n", d["n"], 24)
chk("intent_match", d["intent_match"], 22)
chk("intent_match_rate", d["intent_match_rate"], 0.9167)
chk("pass_all", d["pass_all"], 7)
chk("pass_all_rate", d["pass_all_rate"], 0.2917)
chk("slot_f1", d["slot_f1"], 0.48)
chk("slot_precision", d["slot_precision"], 0.375)
chk("slot_recall", d["slot_recall"], 0.6667)
chk("json_valid_rate", d["json_valid_rate"], 1.0)
chk("failure_modes", d["failure_modes"], {"slot_missing": 5, "PASS": 7, "slot_extra": 10, "intent_wrong": 2})
iw = sorted(r["id"] for r in rows("D_intent_slot_scored.jsonl")
            if r["verdict"].get("failure_mode") == "intent_wrong")
chk("intent_wrong 样本", iw, ["D-09", "D-10"])

print("=== 计时 ===")
tot = 0
for n in ["A_agent_tool_call", "B_doc_extraction", "C_compliance_refusal", "D_intent_slot"]:
    with io.open(os.path.join(R, n + "_base_outputs.jsonl"), encoding="utf-8") as f:
        for l in f:
            if l.strip():
                o = json.loads(l)
                if o.get("_record_type") == "run_meta":
                    tot += o["elapsed_total_s"]
                    print(f"    {n}: {o['elapsed_total_s']}s  ({o['generated_at']})")
                    break
chk("probe 总耗时(s)", round(tot, 1), 114.7)
print(f"       折算 200 条单臂 ≈ {round(tot / 96 * 200)}s ≈ {round(tot / 96 * 200 / 60, 1)} min")

print("=== 报告里的 Wilson 95% 置信区间 ===")
import math


def wilson(x, n, z=1.96):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - m), min(1.0, c + m))


for x, n, lbl, want in [
    (0, 10, "C 拒答 0/10", (0.0, 0.278)),
    (10, 24, "C 决策 10/24", (0.245, 0.612)),
    (21, 24, "A 21/24", (0.690, 0.957)),
    (57, 71, "B 字段 57/71", (0.696, 0.879)),
    (22, 24, "D 意图 22/24", (0.742, 0.977)),
]:
    lo, hi = wilson(x, n)
    good = abs(lo - want[0]) < 0.001 and abs(hi - want[1]) < 0.001
    ok = ok and good
    print(f"  [{'OK ' if good else 'BAD'}] {lbl}: [{lo:.1%}, {hi:.1%}]  报告=[{want[0]:.1%}, {want[1]:.1%}]")

print("=== 上轮训练吞吐（文件 mtime 实测）===")
import datetime
ts = []
for ck in ("checkpoint-40", "checkpoint-60", "checkpoint-80"):
    p = os.path.join(r"D:\programmer\LLM\sft_output", ck, "trainer_state.json")
    ts.append((ck, os.path.getmtime(p)))
    print(f"    {ck}: {datetime.datetime.fromtimestamp(ts[-1][1]).isoformat()}")
for i in range(1, len(ts)):
    dt = ts[i][1] - ts[i - 1][1]
    print(f"    {ts[i-1][0]} -> {ts[i][0]}: {dt:.1f}s / 20 步 = {dt/20:.3f} s/step")

print("\n全部一致" if ok else "\n★ 有不一致项，需修报告")
sys.exit(0 if ok else 1)
