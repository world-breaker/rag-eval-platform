#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分层覆盖诊断：按 (class, stratum) 精确统计每个 split 的条目分布。"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from build_sft_data import _stratum_of  # noqa: E402

for sp in ("dev", "test"):
    rows = [json.loads(l) for l in io.open(ROOT / "data" / f"{sp}.jsonl", encoding="utf-8") if l.strip()]
    print("=" * 60)
    print(sp, "n =", len(rows))
    c = Counter()
    for r in rows:
        q = r["prompt"].split("客户问：", 1)[-1]
        cls = r["meta"]["cls"]
        c[(cls, _stratum_of(cls, q))] += 1
    per_cls = Counter()
    for (cls, st), n in sorted(c.items()):
        per_cls[cls] += n
    for cls, n in sorted(per_cls.items()):
        items = sorted((st, v) for (c2, st), v in c.items() if c2 == cls)
        print(f"  {cls:10s} n={n:4d}  {items if len(items) <= 12 else str(items[:12]) + ' ...'}")
