#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
API 花费核算（改写环节）
=====================================================================
为什么要单独核算
----------------
`results/rewrite_run.log` 只记录了**最后一次**改写运行（800 个槽位、807 次调用、
$0.0343）。但 `data/llm_rewrite_cache.json` 里最终有 **1560** 条缓存，
且日志首行写的是「缓存命中 0 条」→ 说明在日志那次运行**之后**还跑过**一次**
改写（缓存文件 mtime 22:22:59 晚于日志 21:40:54），而那次运行**没有留下日志**。

所以「实际花费」必须拆成：
  ① **有日志的部分**（实测 token 数 → 直接算钱）
  ② **无日志的部分**（按日志测得的「每槽位平均 token」外推，明确标为推算）

纪律：实测与推算**分列**，不混排；key 只从环境变量读，**本脚本不调用任何 API**。

用法
----
    python scripts/api_cost_report.py
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# deepseek-flash 价目（脚本 rewrite_with_llm.py 头部同一份来源）
PRICE = {
    "off_peak": {"in": 0.15, "in_cache": 0.003, "out": 0.60},
    "peak":     {"in": 0.30, "in_cache": 0.006, "out": 1.20},
}
USD_CNY = 7.1


def main() -> int:
    log_path = RESULTS / "rewrite_run.log"
    log = io.open(log_path, encoding="utf-8", errors="replace").read()

    # ---- 有日志的那次运行 ----
    # ⚠️ 日志里的中文汇总行有字符被写成了 '?'（写日志时的控制台编码问题），
    #    所以**只解析纯 ASCII 的进度行**：
    #      "  [800/800] ok=799 calls=807 err=0 tok_in=164330 tok_out=16137 elapsed=77.3s"
    prog = re.findall(
        r"\[(\d+)/(\d+)\]\s+ok=(\d+)\s+calls=(\d+)\s+err=(\d+)"
        r"\s+tok_in=(\d+)\s+tok_out=(\d+)", log)
    if not prog:
        raise SystemExit("[FATAL] 日志里没找到进度行，无法核算")
    done_i, tot_slots, ok_slots, calls, errs, tok_in, tok_out = prog[-1]
    ok_slots, tot_slots, calls = int(ok_slots), int(tot_slots), int(calls)
    tok_in, tok_out, errs = int(tok_in), int(tok_out), int(errs)
    tok_cache = 0
    m = re.search(r"cache-hit\s*(\d+)", log)
    if m:
        tok_cache = int(m.group(1))
    log_cost_usd = log_cost_cny = None
    m2 = re.search(r"\$([\d.]+)", log)
    if m2:
        log_cost_usd = float(m2.group(1))
    i_dollar = log.find("$0.")
    if i_dollar >= 0:
        m3 = re.findall(r"([\d.]+)", log[i_dollar:i_dollar + 60])
        if len(m3) >= 2:
            log_cost_cny = float(m3[1])

    # ---- 缓存最终条目数 → 无日志那部分有多少槽位 ----
    cache = json.loads(io.open(DATA / "llm_rewrite_cache.json", encoding="utf-8").read())
    n_cache = len(cache)
    n_ok = sum(1 for v in cache.values() if isinstance(v, dict) and v.get("ok"))
    unlogged_slots = max(0, n_cache - tot_slots)

    # 每槽位平均 token（用有日志那次实测值）
    per_slot_in = tok_in / tot_slots if tot_slots else 0.0
    per_slot_out = tok_out / tot_slots if tot_slots else 0.0
    est_in = round(per_slot_in * unlogged_slots)
    est_out = round(per_slot_out * unlogged_slots)

    def cost(price, i_tok, o_tok, c_tok=0):
        miss = max(0, i_tok - c_tok)
        return (miss / 1e6 * price["in"] + c_tok / 1e6 * price["in_cache"]
                + o_tok / 1e6 * price["out"])

    logged_off = cost(PRICE["off_peak"], tok_in, tok_out, tok_cache)
    logged_peak = cost(PRICE["peak"], tok_in, tok_out, tok_cache)
    unlogged_off = cost(PRICE["off_peak"], est_in, est_out)
    unlogged_peak = cost(PRICE["peak"], est_in, est_out)
    total_off = logged_off + unlogged_off
    total_peak = logged_peak + unlogged_peak

    out = {
        "_note": "改写环节 API 花费核算。①=有日志实测；②=无日志推算。key 只从环境变量读，本脚本零 API 调用。",
        "model": "deepseek-flash",
        "计价来源": "https://api-docs.deepseek.com/quick_start/pricing/",
        "USD_CNY": USD_CNY,
        "预算上限_CNY": 5.0,
        "①有日志的运行_实测": {
            "日志文件": str(log_path),
            "槽位_成功/总数": f"{ok_slots}/{tot_slots}",
            "API调用次数": calls,
            "输入token": tok_in, "cache_hit_token": tok_cache, "输出token": tok_out,
            "每槽位平均_输入token": round(per_slot_in, 2),
            "每槽位平均_输出token": round(per_slot_out, 2),
            "费用_USD_offpeak": round(logged_off, 6),
            "费用_CNY_offpeak": round(logged_off * USD_CNY, 4),
            "费用_USD_peak": round(logged_peak, 6),
            "费用_CNY_peak": round(logged_peak * USD_CNY, 4),
            "日志自己写的估算": {"USD": log_cost_usd, "CNY": log_cost_cny},
        },
        "②无日志的运行_推算": {
            "说明": "缓存最终条目数多于日志那次运行的槽位数 → 日志之后还跑过一次改写，"
                    "该次运行没有留下日志。token 按①实测的每槽位均值外推。",
            "缓存最终条目数": n_cache,
            "缓存成功条目数": n_ok,
            "推算槽位数": unlogged_slots,
            "推算_输入token": est_in, "推算_输出token": est_out,
            "费用_USD_offpeak": round(unlogged_off, 6),
            "费用_CNY_offpeak": round(unlogged_off * USD_CNY, 4),
            "费用_USD_peak": round(unlogged_peak, 6),
            "费用_CNY_peak": round(unlogged_peak * USD_CNY, 4),
        },
        "合计": {
            "费用_USD_offpeak": round(total_off, 6),
            "费用_CNY_offpeak": round(total_off * USD_CNY, 4),
            "费用_USD_peak": round(total_peak, 6),
            "费用_CNY_peak": round(total_peak * USD_CNY, 4),
            "占预算上限比例_offpeak": round(total_off * USD_CNY / 5.0 * 100, 2),
            "占预算上限比例_peak": round(total_peak * USD_CNY / 5.0 * 100, 2),
        },
        "其他环节": {"评测判分": "纯规则，零 API", "checkpoint选择": "dev eval_loss，零 API"},
    }

    p = RESULTS / "api_cost_report.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 74)
    print("① 有日志的运行（实测）")
    print(f"   槽位 {ok_slots}/{tot_slots}  调用 {calls} 次  in {tok_in} / out {tok_out} tok")
    print(f"   off-peak ${logged_off:.6f} = ¥{logged_off*USD_CNY:.4f}   "
          f"peak ${logged_peak:.6f} = ¥{logged_peak*USD_CNY:.4f}")
    print(f"   日志自写估算：${log_cost_usd} ≈ ¥{log_cost_cny}")
    print("② 无日志的运行（推算）")
    print(f"   缓存 {n_cache} 条（成功 {n_ok}），比日志那次多 {unlogged_slots} 槽位")
    print(f"   推算 in {est_in} / out {est_out} tok → off-peak ${unlogged_off:.6f} = ¥{unlogged_off*USD_CNY:.4f}")
    print("合计")
    print(f"   off-peak ${total_off:.6f} = ¥{total_off*USD_CNY:.4f}  "
          f"（占 ¥5 上限的 {total_off*USD_CNY/5*100:.2f}%）")
    print(f"   peak     ${total_peak:.6f} = ¥{total_peak*USD_CNY:.4f}  "
          f"（占 ¥5 上限的 {total_peak*USD_CNY/5*100:.2f}%）")
    print(f"\n→ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
