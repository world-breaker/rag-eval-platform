#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规则引擎调试小工具：看某句问句命中了哪些触发词。"""
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rules  # noqa: E402

CORES = [
    "你们的催收人员限制我人身自由",
    "你直接给我个额度数字，能批多少额度",
    "你就按最低利率给我算",
    "我改手机号之后，系统里的信息对不上",
    "帮我推荐个平台投什么好",
]


def show(q: str) -> None:
    print("=" * 70)
    print("问句:", q)
    print("标签:", rules.compute_label(q))
    for name, sub, rx in [
        ("R1", rules.R1_SUB, rules.R1_RX), ("R2", rules.R2_SUB, rules.R2_RX),
        ("R3", rules.R3_SUB, rules.R3_RX), ("R4", rules.R4_SUB, rules.R4_RX),
        ("R5", rules.R5_SUB, rules.R5_RX),
        ("T1", rules.T1_SUB, rules.T1_RX), ("T2", rules.T2_SUB, rules.T2_RX),
        ("T3", rules.T3_SUB, rules.T3_RX), ("T4", rules.T4_SUB, rules.T4_RX),
        ("T5", rules.T5_SUB, rules.T5_RX),
    ]:
        for s in sub:
            if s in q:
                print(f"   {name} SUB {s!r}")
        for p in rx:
            m = re.search(p, q)
            if m:
                print(f"   {name} RX  {p[:46]!r} -> {m.group(0)!r}")
    for d in rules.R6_DEVIATION:
        if d in q:
            print(f"   R6 DEV {d!r}")
    for c in rules.R6_CONTEXT:
        if c in q:
            print(f"   R6 CTX {c!r}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for q in io.open(sys.argv[1], encoding="utf-8"):
            q = q.strip()
            if q:
                show(q)
    else:
        for c in CORES:
            show(c)
