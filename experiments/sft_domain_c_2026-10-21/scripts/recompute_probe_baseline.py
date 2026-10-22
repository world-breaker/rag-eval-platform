#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
probe 基线**复算**（不照抄报告数字）
=====================================================================
为什么要这个脚本
------------------
probe 报告（`experiments/domain_probe_2026-10-21.md`）里写着一句关键数字：

    「基座在 n=24 上决策准确率只有 41.7%，且该拒答的 10 条里 0 条拒答成功（0/10）」

本脚本**不复述**这句话，而是用**两套独立来源**重新算一遍：

  A. **重算**：拿 probe 自己存的原始输出
     (`domain_probe_2026-10-21/results/C_compliance_refusal_base_outputs.jsonl`)
     重新喂给 **probe 自己的判定器**
     (`domain_probe_2026-10-21/scripts/matchers.py::score_domain_c`)，
     重算出决策准确率 / 完整通过率 / 该拒拒答率 / JSON 合法率 / 失败模式分布。
  B. **对账**：把 A 的结果与 probe 落盘的 `C_compliance_refusal_scores.json` 逐项比对，
     并与 probe 报告 md 里**写出来的**数字逐条对账。

这样「复算值 vs 报告原值」才是有据可查的，而不是复述。

另外，本脚本会打印 probe 原始输出的 run_meta（解码参数 / system prompt / 硬件），
供报告核对「基线数字的口径」。

输出
----
`results/probe_baseline_recompute.json`（含复算值、probe 落盘值、报告原值、三方对账结论）

用法
----
    python scripts/recompute_probe_baseline.py
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

PROBE = Path(r"D:\programmer\LLM\projects\rag-eval-platform\experiments"
             r"\domain_probe_2026-10-21")
PROBE_SCRIPTS = PROBE / "scripts"
PROBE_DATA = PROBE / "data" / "C_compliance_refusal.jsonl"
PROBE_OUT = PROBE / "results" / "C_compliance_refusal_base_outputs.jsonl"
PROBE_SCORES = PROBE / "results" / "C_compliance_refusal_scores.json"
PROBE_REPORT = PROBE.parent / "domain_probe_2026-10-21.md"

sys.path.insert(0, str(PROBE_SCRIPTS))
from matchers import score_domain_c  # noqa: E402  （probe 判定器，只读引用）


def read_jsonl(p: Path):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def main() -> int:
    data = read_jsonl(PROBE_DATA)
    rows = read_jsonl(PROBE_OUT)
    meta = next((r for r in rows if r.get("_record_type") == "run_meta"), {})
    outs = {r["id"]: r for r in rows if r.get("_record_type") == "sample"}
    stored = json.loads(io.open(PROBE_SCORES, encoding="utf-8").read())
    report_text = io.open(PROBE_REPORT, encoding="utf-8").read()

    # ---------------- A. 复算 ----------------
    n = dec_ok = strict_ok = json_ok = 0
    by_exp = defaultdict(lambda: {"n": 0, "dec_ok": 0, "strict_ok": 0})
    fm = Counter()
    for s in data:
        raw = outs.get(s["id"], {}).get("raw_output", "")
        v = score_domain_c(s["expected"], raw)
        n += 1
        dec_ok += bool(v["decision_match"])
        strict_ok += bool(v["pass_all"])
        json_ok += bool(v["json_strict_valid"])
        e = by_exp[s["expected"]["decision"]]
        e["n"] += 1
        e["dec_ok"] += bool(v["decision_match"])
        e["strict_ok"] += bool(v["pass_all"])
        fm[v.get("failure_mode") or "PASS"] += 1

    recomputed = {
        "n": n,
        "decision_match": dec_ok,
        "decision_match_rate": round(dec_ok / n, 4),
        "pass_all": strict_ok,
        "pass_all_rate": round(strict_ok / n, 4),
        "json_valid": json_ok,
        "json_valid_rate": round(json_ok / n, 4),
        "decision_accuracy_by_expected": {
            k: {"n": v["n"], "correct": v["dec_ok"], "rate": round(v["dec_ok"] / v["n"], 4)}
            for k, v in sorted(by_exp.items())
        },
        "failure_modes": dict(fm),
    }

    # ---------------- B. 对账 ----------------
    checks = []

    def chk(name: str, recomputed_v, stored_v) -> None:
        ok = (recomputed_v == stored_v)
        checks.append({"项": name, "复算值": recomputed_v, "probe落盘值": stored_v, "一致": ok})
        print(f"  {'[OK]' if ok else '[!!]'} {name:22s} 复算={recomputed_v!s:>8s}  "
              f"probe落盘={stored_v!s:>8s}")

    print("=" * 78)
    print("① 复算值 vs probe 落盘 scores.json（同一批原始输出、同一套判定器）")
    chk("n", recomputed["n"], stored["n"])
    chk("决策判对数 decision_match", recomputed["decision_match"], stored["decision_match"])
    chk("完整通过数 pass_all", recomputed["pass_all"], stored["pass_all"])
    chk("JSON 合法数", recomputed["json_valid"], stored["json_valid"])
    chk("决策准确率", recomputed["decision_match_rate"], stored["decision_match_rate"])
    chk("完整通过率", recomputed["pass_all_rate"], stored["pass_all_rate"])
    chk("JSON 合法率", recomputed["json_valid_rate"], stored["json_valid_rate"])
    chk("失败模式分布", recomputed["failure_modes"], stored["failure_modes"])
    for dec, v in recomputed["decision_accuracy_by_expected"].items():
        st = stored["decision_accuracy_by_expected"].get(dec, {})
        chk(f"按期望[{dec}] n", v["n"], st.get("n"))
        chk(f"按期望[{dec}] 判对数", v["correct"], st.get("correct"))

    # ---------------- C. 与 probe 报告 md 里写出来的数字对账 ----------------
    print("=" * 78)
    print("② probe 报告 md 里写出来的原值（正则从 md 正文里抓，不靠人工抄）")
    report_claims = {}
    m = re.search(r"决策准确率只有\s*\*\*([\d.]+)%", report_text)
    report_claims["决策准确率(%)"] = m.group(1) if m else None
    m = re.search(r"该拒答的\s*(\d+)\s*条里\s*(\d+)\s*条拒答成功", report_text)
    report_claims["该拒总数"] = int(m.group(1)) if m else None
    report_claims["该拒判对数"] = int(m.group(2)) if m else None
    m = re.search(r"完整通过率\s*([\d.]+)%", report_text)
    report_claims["完整通过率(%)"] = m.group(1) if m else None
    m = re.search(r"JSON 合法率\s*([\d.]+)%", report_text)
    if m:
        report_claims["JSON合法率(%)"] = m.group(1)
    else:
        # probe 报告里 JSON 合法率有两种写法：表头一行「| 87.5% |」，
        # 或正文「3/24 输出了非法 JSON」。两种都兜住。
        m = re.search(r"(\d+)/(\d+)\s*输出了非法\s*JSON", report_text)
        if m:
            bad, tot = int(m.group(1)), int(m.group(2))
            report_claims["JSON合法率(%)"] = f"{(tot - bad) / tot * 100:.1f}"
            report_claims["JSON合法率_来源"] = f"由「{bad}/{tot} 输出了非法 JSON」反推"
        else:
            m = re.search(r"\|\s*(\d+\.\d+)%\s*\|", report_text)
            report_claims["JSON合法率(%)"] = m.group(1) if m else None
            report_claims["JSON合法率_来源"] = "表格中的合法率列"

    def pct_eq(a, b) -> bool:
        try:
            return abs(float(a) - float(b)) < 0.05
        except (TypeError, ValueError):
            return False

    pairs = [
        ("决策准确率(%)", recomputed["decision_match_rate"] * 100),
        ("完整通过率(%)", recomputed["pass_all_rate"] * 100),
        ("JSON合法率(%)", recomputed["json_valid_rate"] * 100),
    ]
    for key, recomputed_val in pairs:
        rep = report_claims.get(key)
        ok = rep is not None and pct_eq(rep, recomputed_val)
        checks.append({"项": f"报告原值 {key}", "复算值": round(recomputed_val, 4),
                       "报告原值": rep, "一致": ok})
        print(f"  {'[OK]' if ok else '[!!]'} {key:22s} 复算={recomputed_val:.4f}%  "
              f"报告原值={rep}%")
    ok_ref = (report_claims.get("该拒总数") == by_exp["拒答"]["n"]
              and report_claims.get("该拒判对数") == by_exp["拒答"]["dec_ok"])
    checks.append({"项": "报告原值 该拒 0/10", "复算值": f"{by_exp['拒答']['dec_ok']}/{by_exp['拒答']['n']}",
                   "报告原值": f"{report_claims.get('该拒判对数')}/{report_claims.get('该拒总数')}",
                   "一致": ok_ref})
    print(f"  {'[OK]' if ok_ref else '[!!]'} {'该拒判对':22s} "
          f"复算={by_exp['拒答']['dec_ok']}/{by_exp['拒答']['n']}  "
          f"报告原值={report_claims.get('该拒判对数')}/{report_claims.get('该拒总数')}")

    all_ok = all(c["一致"] for c in checks)

    # ---------------- 落盘 ----------------
    out = {
        "_note": "probe 基线复算：用 probe 自己的判定器重算 probe 自己的原始输出，"
                 "再与 probe 落盘的 scores.json 与 probe 报告 md 里写出来的数字三方对账。",
        "source": {
            "probe_data": str(PROBE_DATA),
            "probe_base_outputs": str(PROBE_OUT),
            "probe_scores": str(PROBE_SCORES),
            "probe_report": str(PROBE_REPORT),
            "matcher": "domain_probe_2026-10-21/scripts/matchers.py::score_domain_c",
            "probe_data_sha256": hashlib.sha256(PROBE_DATA.read_bytes()).hexdigest(),
            "probe_base_outputs_sha256": hashlib.sha256(PROBE_OUT.read_bytes()).hexdigest(),
        },
        "probe_run_meta": {
            "base_model": meta.get("base_model"),
            "decoding": meta.get("decoding"),
            "system_prompt": meta.get("system_prompt"),
            "n": meta.get("n"),
            "elapsed_total_s": meta.get("elapsed_total_s"),
            "env": meta.get("env"),
            "generated_at": meta.get("generated_at"),
        },
        "复算值_recomputed": recomputed,
        "probe落盘值_stored": {
            "n": stored["n"], "decision_match": stored["decision_match"],
            "pass_all": stored["pass_all"], "json_valid": stored["json_valid"],
            "decision_match_rate": stored["decision_match_rate"],
            "pass_all_rate": stored["pass_all_rate"],
            "json_valid_rate": stored["json_valid_rate"],
            "decision_accuracy_by_expected": stored["decision_accuracy_by_expected"],
            "failure_modes": stored["failure_modes"],
        },
        "报告原值_report_claims": report_claims,
        "三方对账_checks": checks,
        "三方完全一致": all_ok,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "probe_baseline_recompute.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 78)
    print(f"[{'OK' if all_ok else 'FAIL'}] 三方对账：复算值 / probe 落盘值 / 报告原值 "
          f"→ {'完全一致' if all_ok else '存在不一致'}")
    print(f"[done] 写入 {p}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
