#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
判定器自检（judge self-test）
=============================

目的：证明**纯规则判定器**本身是对的 ———
  ① 用 expected 反推出的「标准答案」必须被判为通过；
  ② 各类人为错误（改工具名 / 改参数 / 漏字段 / 改标签 / 该拒未拒 …）必须被判为不通过。

如果判定器分不清「对」和「错」，后面所有通过率数字都不可信。

用法
----
  python scripts/selftest_matchers.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from matchers import SCORERS  # noqa: E402


def canonical_value(spec: Dict) -> Any:
    """由 expected 反推一个**合法**的具体值（用于构造标准答案）。"""
    t = spec["type"]
    if t == "exact":
        return spec["value"]
    if t == "terms":
        return "该内容涉及：" + "、".join(spec["terms"]) + "。"
    if t == "number":
        return spec["value"]
    if t == "number_pct":
        return spec["value"]
    if t == "enum":
        return spec["value"]
    if t == "bool":
        return spec["value"]
    if t == "list":
        return list(spec["value"])
    if t == "items_contain":
        return "；".join(alts[0] for alts in spec["value"])
    if t == "string":
        return spec["value"]
    if t == "date":
        return spec["value"]
    if t == "card4":
        return spec["value"]
    if t == "expression_value":
        return f"({spec['value']})+0"
    raise ValueError(t)


def canonical_output(domain: str, expected: Dict) -> str:
    if domain == "A_agent_tool_call":
        args = {k: canonical_value(v) for k, v in expected["arguments"].items()}
        return json.dumps({"tool": expected["tool"], "arguments": args}, ensure_ascii=False)
    if domain == "B_doc_extraction":
        fields = {k: canonical_value(v) for k, v in expected["fields"].items()}
        return json.dumps(fields, ensure_ascii=False)
    if domain == "C_compliance_refusal":
        return json.dumps({
            "decision": expected["decision"],
            "redline": expected["redline"],
            "transfer_rules": expected["transfer_rules"],
            "reason": "依据规则判断。",
        }, ensure_ascii=False)
    if domain == "D_intent_slot":
        slots = {k: canonical_value(v) for k, v in expected["slots"].items()}
        return json.dumps({"intent": expected["intent"], "slots": slots}, ensure_ascii=False)
    raise ValueError(domain)


# ---------------------------------------------------------------- 负例构造

def negatives(domain: str, expected: Dict):
    """返回 [(负例名, 输出文本)]。"""
    out = []
    if domain == "A_agent_tool_call":
        exp_tool = expected["tool"]
        exp_args = expected["arguments"]
        # 工具名错
        args = {k: canonical_value(v) for k, v in exp_args.items()}
        out.append(("工具名错", json.dumps(
            {"tool": "search_knowledge" if exp_tool != "search_knowledge" else "calculator",
             "arguments": args}, ensure_ascii=False)))
        # JSON 非法
        out.append(("JSON非法", '{"tool": "%s", "arguments": {' % exp_tool))
        # 参数错（把数值改成别的）
        args2 = {k: canonical_value(v) for k, v in exp_args.items()}
        for k, spec in exp_args.items():
            if spec["type"] in ("number", "exact") and isinstance(spec["value"], (int, float)):
                args2[k] = spec["value"] + 1
            elif spec["type"] == "expression_value":
                args2[k] = "999+1"
            elif spec["type"] == "terms":
                args2[k] = "完全无关的内容"
        if exp_args:
            # 只有期望里真的有参数时，「参数错 / 参数缺失」才是有区分度的负例
            out.append(("参数错", json.dumps({"tool": exp_tool, "arguments": args2}, ensure_ascii=False)))
            out.append(("参数缺失", json.dumps({"tool": exp_tool, "arguments": {}}, ensure_ascii=False)))
    elif domain == "B_doc_extraction":
        fields = {k: canonical_value(v) for k, v in expected["fields"].items()}
        out.append(("字段全错", json.dumps(
            {k: "乱七八糟的值" for k in expected["fields"]}, ensure_ascii=False)))
        miss = dict(fields)
        k0 = list(miss)[0]
        del miss[k0]
        out.append(("字段漏一个", json.dumps(miss, ensure_ascii=False)))
        out.append(("JSON非法", "{字段: 无效}"))
    elif domain == "C_compliance_refusal":
        good = {
            "decision": expected["decision"],
            "redline": expected["redline"],
            "transfer_rules": expected["transfer_rules"],
            "reason": "依据规则判断。",
        }
        if expected["decision"] == "拒答":
            bad = dict(good, decision="回答")
            out.append(("该拒未拒", json.dumps(bad, ensure_ascii=False)))
            out.append(("红线编号错", json.dumps(
                dict(good, redline=["R6"] if expected["redline"] != ["R6"] else ["R1"]),
                ensure_ascii=False)))
        elif expected["decision"] == "转人工":
            out.append(("该转未转", json.dumps(dict(good, decision="回答"), ensure_ascii=False)))
            out.append(("转人工编号错", json.dumps(
                dict(good, transfer_rules=["T1"] if expected["transfer_rules"] != ["T1"] else ["T2"]),
                ensure_ascii=False)))
        else:
            out.append(("过度拒答", json.dumps(dict(good, decision="拒答"), ensure_ascii=False)))
        out.append(("JSON非法", "decision: 拒答}"))
    elif domain == "D_intent_slot":
        slots = {k: canonical_value(v) for k, v in expected["slots"].items()}
        other = "投诉" if expected["intent"] != "投诉" else "结清证明"
        out.append(("标签错", json.dumps(
            {"intent": other, "slots": slots}, ensure_ascii=False)))
        if slots:
            k0 = list(slots)[0]
            bad_slots = dict(slots)
            bad_slots[k0] = "999999"
            out.append(("槽位值错", json.dumps(
                {"intent": expected["intent"], "slots": bad_slots}, ensure_ascii=False)))
            out.append(("槽位漏一个", json.dumps(
                {"intent": expected["intent"], "slots": {}}, ensure_ascii=False)))
        out.append(("多一个槽位", json.dumps(
            {"intent": expected["intent"], "slots": dict(slots, amount=1)}, ensure_ascii=False)))
        out.append(("JSON非法", "{intent: 投诉"))
    return out


def main() -> int:
    failures = []
    n_checked = 0
    for domain, scorer in SCORERS.items():
        path = ROOT / "data" / f"{domain}.jsonl"
        rows = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]
        for r in rows:
            exp = r["expected"]
            # ① 标准答案必须通过
            good = canonical_output(domain, exp)
            res = scorer(exp, good)
            n_checked += 1
            if not res.get("pass_all"):
                failures.append((domain, r["id"], "标准答案未通过", res.get("failure_mode"), good[:160]))
            # ② 负例必须不通过
            for name, bad in negatives(domain, exp):
                res2 = scorer(exp, bad)
                n_checked += 1
                if res2.get("pass_all"):
                    failures.append((domain, r["id"], f"负例通过：{name}", res2.get("failure_mode"), bad[:160]))
        print(f"[selftest] {domain}: n={len(rows)} 标准答案+负例全部检查完毕")

    print(f"\n共检查 {n_checked} 个用例，失败 {len(failures)} 个")
    for f in failures[:40]:
        print("  FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
