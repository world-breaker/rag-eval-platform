#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
域 B 判分脚本：金融/房产文档结构化抽取（纯规则，零 LLM 依赖）
================================================================

输入：results/B_doc_extraction_base_outputs.jsonl
输出：results/B_doc_extraction_scored.jsonl
      results/B_doc_extraction_scores.json

口径
----
  pass_all      ：请求的**全部**字段都判对（严格）
  field_accuracy：字段级准确率 = 判对字段数 / 总字段数（主指标，配合 micro-F1 看）
  json_valid    ：整段输出是合法 JSON

用法
----
  python scripts/score_B.py
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

from matchers import score_domain_b  # noqa: E402

DOMAIN = "B_doc_extraction"


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
    field_fm = Counter()
    n = len(samples)
    tot_fields = 0
    tot_hits = 0
    for s in samples:
        o = outs.get(s["id"])
        raw = o["raw_output"] if o else ""
        res = score_domain_b(s["expected"], raw)
        fm[res.get("failure_mode") or "PASS"] += 1
        tot_fields += res.get("n_fields", 0)
        tot_hits += res.get("field_hits", 0)
        for fname, fr in (res.get("fields") or {}).items():
            if not fr["ok"]:
                field_fm[f"{fname}:{fr['reason']}"] += 1
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

    agg = {
        "domain": DOMAIN,
        "n": n,
        "pass_all": sum(1 for x in scored if x["verdict"]["pass_all"]),
        "pass_all_strict": sum(1 for x in scored if x["verdict"]["pass_all_strict"]),
        "json_valid": sum(1 for x in scored if x["verdict"]["json_strict_valid"]),
        "json_lenient_parsed": sum(1 for x in scored if x["verdict"]["json_lenient_parsed"]),
        "total_fields": tot_fields,
        "field_hits": tot_hits,
        "field_hits_strict": sum(x["verdict"].get("field_hits_strict", 0) for x in scored),
        "micro_field_f1": round(tot_hits / tot_fields, 4) if tot_fields else 0.0,
        "micro_field_f1_strict": round(
            sum(x["verdict"].get("field_hits_strict", 0) for x in scored) / tot_fields, 4
        ) if tot_fields else 0.0,
        "mean_sample_field_accuracy": round(
            sum(x["verdict"]["field_accuracy"] for x in scored) / n, 4) if n else 0.0,
        "truncated": sum(1 for x in scored if x["truncated"]),
        "failure_modes": dict(fm),
        "top_field_failures": dict(field_fm.most_common(20)),
        "base_model": meta.get("base_model"),
        "decoding": meta.get("decoding"),
    }
    agg["pass_all_rate"] = round(agg["pass_all"] / n, 4) if n else 0.0
    agg["pass_all_strict_rate"] = round(agg["pass_all_strict"] / n, 4) if n else 0.0
    agg["json_valid_rate"] = round(agg["json_valid"] / n, 4) if n else 0.0

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
