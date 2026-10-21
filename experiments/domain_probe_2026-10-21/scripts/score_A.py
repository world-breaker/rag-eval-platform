#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
域 A 判分脚本：Agent 工具调用（纯规则，零 LLM 依赖）
======================================================

输入：results/A_agent_tool_call_base_outputs.jsonl
输出：results/A_agent_tool_call_scored.jsonl   （逐条：原始输出 + 判定）
      results/A_agent_tool_call_scores.json    （汇总：各项通过率 + 失败模式分布）

口径
----
  pass_strict  ：整段输出是合法 JSON、顶层只有 tool/arguments 两个键、只调了 1 个工具、
                 工具名对、参数全对、**参数键集合完全一致**（格式维度）
  pass_lenient ：能从输出里抽出工具调用，工具名对、参数全对（能力维度，主指标）
  json_valid   ：整段输出是合法 JSON（严格）

用法
----
  python scripts/score_A.py
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

from matchers import score_domain_a  # noqa: E402

DOMAIN = "A_agent_tool_call"


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
    n = len(samples)
    for s in samples:
        o = outs.get(s["id"])
        raw = o["raw_output"] if o else ""
        res = score_domain_a(s["expected"], raw)
        fm[res.get("failure_mode") or "PASS"] += 1
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
        "pass_format_strict": sum(1 for x in scored if x["verdict"]["pass_format_strict"]),
        "pass_lenient": sum(1 for x in scored if x["verdict"]["pass_lenient"]),
        "pass_strict": sum(1 for x in scored if x["verdict"]["pass_strict"]),
        "json_valid": sum(1 for x in scored if x["verdict"]["json_strict_valid"]),
        "json_lenient_parsed": sum(1 for x in scored if x["verdict"]["json_lenient_parsed"]),
        "tool_match": sum(1 for x in scored if x["verdict"]["tool_match"]),
        "args_match": sum(1 for x in scored if x["verdict"]["args_match"]),
        "arg_keys_exact": sum(1 for x in scored if x["verdict"]["arg_keys_exact"]),
        "truncated": sum(1 for x in scored if x["truncated"]),
        "failure_modes": dict(fm),
        "base_model": meta.get("base_model"),
        "decoding": meta.get("decoding"),
    }
    for k in ("pass_all", "pass_all_strict", "pass_format_strict", "pass_lenient",
              "pass_strict", "json_valid", "json_lenient_parsed",
              "tool_match", "args_match", "arg_keys_exact", "truncated"):
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
