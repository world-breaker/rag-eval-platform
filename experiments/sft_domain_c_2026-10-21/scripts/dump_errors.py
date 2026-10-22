#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
典型错例导出：把「该拒不拒 / 该转未转 / 编号错 / 过度拒答 / JSON 非法」各挑几条
=====================================================================

用法
----
    python scripts/dump_errors.py --dataset test
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

CATEGORIES = [
    ("should_refuse_but_answered", "该拒不拒（最贵的失败模式）"),
    ("should_transfer_but_answered", "该转未转"),
    ("should_transfer_but_refused", "该转却拒"),
    ("should_refuse_but_transferred", "该拒却转人工"),
    ("over_refusal", "过度拒答（该答却拒/转）"),
    ("redline_mismatch", "决策对但红线编号错"),
    ("transfer_mismatch", "决策对但转人工编号错"),
    ("json_invalid", "JSON 非法"),
]


def read_jsonl(p: Path):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="test")
    ap.add_argument("--per-cat", type=int, default=3)
    args = ap.parse_args()

    data = {r["id"]: r for r in read_jsonl(ROOT / "data" / f"{args.dataset}.jsonl")} \
        if (ROOT / "data" / f"{args.dataset}.jsonl").exists() else {}
    out = {}
    for arm in ("base", "lora"):
        p = RESULTS / f"{args.dataset}_{arm}_scored.jsonl"
        if not p.exists():
            continue
        rows = read_jsonl(p)
        out[arm] = rows

    lines = []
    for cat, title in CATEGORIES:
        lines.append(f"\n### {title}  (`{cat}`)")
        for arm in ("base", "lora"):
            if arm not in out:
                continue
            hits = [r for r in out[arm] if r.get("failure_mode") == cat][: args.per_cat]
            if not hits:
                continue
            lines.append(f"\n**{arm}**")
            for r in hits:
                q = data.get(r["id"], {}).get("prompt", "").split("客户问：", 1)[-1]
                lines.append(f"- `{r['id']}` 期望 {r['expected']['decision']}"
                             f"/{r['expected']['redline']}/{r['expected']['transfer_rules']} → "
                             f"实际 {r['got_decision']}/{r['got_redline']}/{r['got_transfer_rules']}")
                lines.append(f"  - 问句：{q}")
                lines.append(f"  - 输出：`{r['raw_output'][:200].replace(chr(10), ' ')}`")

    txt = "\n".join(lines)
    outp = RESULTS / f"{args.dataset}_typical_errors.md"
    outp.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\n→ {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
