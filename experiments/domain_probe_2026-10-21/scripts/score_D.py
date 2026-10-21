#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
域 D 判分脚本：金融客服意图 + 槽位识别（纯规则，零 LLM 依赖）
============================================================

输入：results/D_intent_slot_base_outputs.jsonl
输出：results/D_intent_slot_scored.jsonl
      results/D_intent_slot_scores.json

口径
----
  intent_accuracy : 意图标签准确率（主指标）
  slot_f1         : 槽位 micro-F1（precision / recall / F1）
  pass_all        : 意图对 **且** 槽位全对 **且** 没有多余槽位（严格）
  标签错 / 槽位漏 / 槽位值错 / 多余槽位 分开统计

用法
----
  python scripts/score_D.py
"""

from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from matchers import score_domain_d  # noqa: E402

DOMAIN = "D_intent_slot"


def read_jsonl(path: Path):
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    base = ROOT / "results" / f"{DOMAIN}_base_outputs.jsonl"
    data = ROOT / "data" / f"{DOMAIN}.jsonl"
    if not base.exists():
        print(f"[FATAL] 找不到 {base}，先跑 run_base.py")
        return 2

    samples = [r for r in read_jsonl(data) if r.get("_record_type") is None]
    outs = {r["id"]: r for r in read_jsonl(base) if r.get("_record_type") == "sample"}
    meta = next((r for r in read_jsonl(base) if r.get("_record_type") == "run_meta"), {})

    scored = []
    fm = Counter()
    slot_fm = Counter()
    conf = Counter()
    n = len(samples)
    tp = fp = fn = 0
    for s in samples:
        o = outs.get(s["id"])
        raw = o["raw_output"] if o else ""
        res = score_domain_d(s["expected"], raw)
        fm[res.get("failure_mode") or "PASS"] += 1
        tp += res.get("slot_hits", 0)
        fn += res.get("slot_misses", 0)
        fp += res.get("slot_extra", 0)
        for sname, sr in (res.get("slots") or {}).items():
            if not sr["ok"]:
                slot_fm[f"{sname}:{sr['reason']}"] += 1
        exp_i = s["expected"]["intent"]
        got_i = res.get("got_intent")
        if not res["intent_match"]:
            conf[f"{exp_i} -> {got_i}"] += 1
        scored.append({
            "_record_type": "sample",
            "id": s["id"],
            "source": s["source"],
            "notes": s["notes"],
            "expected": s["expected"],
            "raw_output": raw,
            "n_new_tokens": o["n_new_tokens"] if o else None,
            "truncated": o["truncated"] if o else None,
            "verdict": res,
        })

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    agg = {
        "domain": DOMAIN,
        "n": n,
        "pass_all": sum(1 for x in scored if x["verdict"]["pass_all"]),
        "intent_match": sum(1 for x in scored if x["verdict"]["intent_match"]),
        "json_valid": sum(1 for x in scored if x["verdict"]["json_strict_valid"]),
        "json_lenient_parsed": sum(1 for x in scored if x["verdict"]["json_lenient_parsed"]),
        "slot_tp": tp, "slot_fp": fp, "slot_fn": fn,
        "slot_precision": round(prec, 4),
        "slot_recall": round(rec, 4),
        "slot_f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        "truncated": sum(1 for x in scored if x["truncated"]),
        "failure_modes": dict(fm),
        "top_slot_failures": dict(slot_fm.most_common(20)),
        "intent_confusions": dict(conf.most_common(30)),
        "base_model": meta.get("base_model"),
        "decoding": meta.get("decoding"),
    }
    for k in ("pass_all", "intent_match", "json_valid", "json_lenient_parsed", "truncated"):
        agg[k + "_rate"] = round(agg[k] / n, 4) if n else 0.0

    out_scored = ROOT / "results" / f"{DOMAIN}_scored.jsonl"
    with io.open(out_scored, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"_record_type": "run_meta", **meta}, ensure_ascii=False) + "\n")
        for x in scored:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    out_json = ROOT / "results" / f"{DOMAIN}_scores.json"
    with io.open(out_json, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(agg, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"\n✓ {out_scored}\n✓ {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
