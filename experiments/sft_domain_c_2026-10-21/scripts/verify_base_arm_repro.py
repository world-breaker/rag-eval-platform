#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
基座臂**可复现性**交叉核对
=====================================================================
这个脚本回答一个问题：「本次评测的基座臂，跟 probe 当初跑的是不是同一个东西？」

做法
----
把本次 `results/probe_base_outputs.jsonl`（基座臂在 probe 24 条上的输出）
与 probe 自己落盘的
`domain_probe_2026-10-21/results/C_compliance_refusal_base_outputs.jsonl`
**逐条逐字比对**。

两边应该一致，因为：
  - 同一个基座模型 `Qwen/Qwen2.5-1.5B-Instruct`（fp16, CUDA）；
  - 同一句 system message（`run_eval.py` 与 probe 的 `run_base.py` 逐字相同）；
  - 同一份 prompt 策略段（sha256 已核对相同）；
  - 同一套贪心解码参数；
  - `run_eval.py` 的生成循环与 probe `run_base.py` 的生成循环**逐行等价**（都只调
    `tokenizer(chat, return_tensors="pt")` → `model.generate(do_sample=False, …)`）。

不一致的话，要么是解码/环境漂移，要么是代码路径不同 —— 都要在这里暴露出来。

用法
----
    python scripts/verify_base_arm_repro.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

PROBE_OUT = (Path(r"D:\programmer\LLM\projects\rag-eval-platform\experiments")
              / r"domain_probe_2026-10-21" / "results"
              / "C_compliance_refusal_base_outputs.jsonl")


def read_jsonl(p: Path):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def main() -> int:
    mine_path = RESULTS / "probe_base_outputs.jsonl"
    if not mine_path.exists():
        print(f"[FATAL] 找不到 {mine_path}（先跑 run_eval.py --arm base --data probe）")
        return 2

    mine = {r["id"]: r for r in read_jsonl(mine_path) if r.get("_record_type") == "sample"}
    theirs_rows = read_jsonl(PROBE_OUT)
    theirs = {r["id"]: r for r in theirs_rows if r.get("_record_type") == "sample"}
    their_meta = next((r for r in theirs_rows if r.get("_record_type") == "run_meta"), {})
    my_meta = next((r for r in read_jsonl(mine_path) if r.get("_record_type") == "run_meta"), {})

    same_text = diff_text = 0
    diffs = []
    for sid, t in theirs.items():
        m = mine.get(sid)
        if m is None:
            diffs.append({"id": sid, "kind": "missing_in_this_run"})
            diff_text += 1
            continue
        if m["raw_output"].strip() == t["raw_output"].strip():
            same_text += 1
        else:
            diff_text += 1
            diffs.append({
                "id": sid, "kind": "text_differs",
                "probe": t["raw_output"][:220],
                "this_run": m["raw_output"][:220],
            })

    out = {
        "note": "本次基座臂在 probe 24 条上的输出，与 probe 自己落盘的原始输出逐字比对。",
        "this_run": {"file": str(mine_path), "base_model": my_meta.get("base_model"),
                     "decoding": my_meta.get("decoding"),
                     "system_prompt": my_meta.get("system_prompt"),
                     "env": my_meta.get("env")},
        "probe_original": {"file": str(PROBE_OUT), "base_model": their_meta.get("base_model"),
                           "decoding": their_meta.get("decoding"),
                           "system_prompt": their_meta.get("system_prompt"),
                           "env": their_meta.get("env")},
        "n": len(theirs),
        "逐字相同": same_text,
        "逐字不同": diff_text,
        "diffs": diffs,
        "system_prompt_逐字相同": my_meta.get("system_prompt") == their_meta.get("system_prompt"),
        "decoding_逐项相同": my_meta.get("decoding") == their_meta.get("decoding"),
        "base_model_相同": my_meta.get("base_model") == their_meta.get("base_model"),
    }
    p = RESULTS / "base_arm_repro_check.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 74)
    print(f"本次基座臂（probe 24 条） vs probe 当初落盘的原始输出")
    print(f"  n={len(theirs)}  逐字相同 {same_text}  逐字不同 {diff_text}")
    print(f"  base_model 相同：{out['base_model_相同']}")
    print(f"  decoding 逐项相同：{out['decoding_逐项相同']}")
    print(f"  system prompt 逐字相同：{out['system_prompt_逐字相同']}")
    for d in diffs[:6]:
        print(f"  ! {d['id']} ({d['kind']})")
        if d["kind"] == "text_differs":
            print(f"      probe    : {d['probe']}")
            print(f"      this_run : {d['this_run']}")
    print(f"\n→ {p}")
    return 0 if diff_text == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
