# -*- coding: utf-8 -*-
"""复算「只重抽裁判输出」的抽样波动：固定 33 条样本与强裁判分数，只重抽裁判的随机输出。

估计器：single（1 次抽样）/ median-of-3 / median-of-7 / median-of-15。
每个 item 有 7 个观测输出，用「item 内有放回重抽」做非参数 bootstrap。

数据来源（两处，都不含答案/上下文全文）：
- `--variance`：`judge_variance_2026-10-20.json` 的 `items`（每条样本每维度的 7 个裁判档位）
- `--results`：`generation_eval_2026-10-19.json` 的臂 C（强裁判 faithfulness，作为对照真值）

复算的就是 `judge_variance_2026-10-20.md` 第六节那张表（MAE/r 的均值与抽样 sd、90% 区间）。
RNG 调用顺序固定为 `for b in B: for k in (1,3,7,15): for item: k 次 random.choice`——
换 seed 只会在 1e-3 量级抖动（实测 seed=123 相对差 <2%），换顺序则数字全变，所以重写本脚本时必须保持这个嵌套顺序。

用法：
    python experiments/bootstrap_judge_variance.py            # 默认 seed=7、B=4000
    python experiments/bootstrap_judge_variance.py --boot 500 --seed 1
"""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

DIMS = ("faithfulness", "relevance", "completeness")


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return float("nan") if dx == 0 or dy == 0 else num / (dx * dy)


def load_draws(variance_path):
    """{ (id, dim): [7 个裁判档位] }，取自 items。"""
    data = json.loads(Path(variance_path).read_text(encoding="utf-8"))
    items = data["items"]
    draws = {}
    for rid, per_dim in items.items():
        for dim in DIMS:
            vals = [int(x) for x in per_dim[dim]]
            if not vals:
                raise SystemExit(f"{rid} / {dim} 没有裁判观测")
            draws[(str(rid), dim)] = vals
    return data, [str(rid) for rid in items], draws


def load_strong_judge(results_path):
    """{ id: 强裁判 faithfulness }，取自臂 C。"""
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    return {str(r["id"]): float(r["judge"]["api"]["scores"]["faithfulness"]) for r in data["samples"]}


def main():
    ap = argparse.ArgumentParser(description="复算裁判多次采样的抽样波动（bootstrap）")
    ap.add_argument("--results", default="experiments/results/generation_eval_2026-10-19.json",
                    help="提供强裁判对照分的已归档结果（臂 C）")
    ap.add_argument("--variance", default="experiments/results/judge_variance_2026-10-20.json",
                    help="提供 7 个裁判档位的方差实验归档")
    ap.add_argument("--boot", type=int, default=4000, help="bootstrap 次数 B（默认 4000）")
    ap.add_argument("--seed", type=int, default=7, help="random.seed（默认 7，与报告一致）")
    args = ap.parse_args()

    meta, ids, draws = load_draws(args.variance)
    C = load_strong_judge(args.results)
    random.seed(args.seed)

    # 每个 item 的抽样标准差（用 7 次观测估，ddof=0）
    item_sd = {k: statistics.pstdev(v) for k, v in draws.items() if len(v) > 1}
    print("=== 数据来源 ===")
    print(f"  方差实验：{args.variance}（{len(ids)} 条样本）")
    print(f"  强裁判对照：{args.results}")
    print(f"  seed={args.seed}  B={args.boot}")

    print("\n=== 每个「样本×维度」的抽样标准差（7 次观测，ddof=0）===")
    for dim in DIMS:
        sds = [sd for k, sd in item_sd.items() if k[1] == dim]
        print(f"  {dim:<14} 均值 sd={statistics.mean(sds):.3f} 最大 sd={max(sds):.3f}")
    rng_hist = {}
    for v in draws.values():
        span = max(v) - min(v)
        rng_hist[span] = rng_hist.get(span, 0) + 1
    same = sum(1 for v in draws.values() if max(v) == min(v))
    print(f"  7 次完全相同的单元 {same}/{len(draws)} = {100 * same / len(draws):.1f}%；"
          f"极差均值 {statistics.mean([max(v) - min(v) for v in draws.values()]):.2f} 档；"
          f"极差分布 {dict(sorted(rng_hist.items()))}")

    def estimate(k, dim="faithfulness"):
        """对每条样本用 k 次抽样（有放回）算估计值。"""
        out = []
        for i in ids:
            v = draws[(i, dim)]
            got = [random.choice(v) for _ in range(k)]
            out.append(statistics.median(got) if k > 1 else got[0])
        return out

    def score(a):
        b = [C[i] for i in ids]
        n = len(a)
        return {
            "mean": statistics.mean(a),
            "exact": sum(1 for x, y in zip(a, b) if x == y) / n,
            "within1": sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / n,
            "mae": sum(abs(x - y) for x, y in zip(a, b)) / n,
            "r": pearson(a, b),
        }

    B = args.boot
    res = {1: [], 3: [], 7: [], 15: []}
    for _ in range(B):
        for k in res:
            res[k].append(score(estimate(k)))

    print(f"\n=== 只重抽裁判输出（固定 {len(ids)} 条样本，B={B}）===")
    print(f"{'估计器':<16}{'MAE 均值':>10}{'MAE sd':>9}{'r 均值':>10}{'r sd':>8}{'exact sd':>10}")
    for k, label in ((1, "single（N=1）"), (3, "median-of-3"), (7, "median-of-7"), (15, "median-of-15")):
        ms = res[k]
        maes = [d["mae"] for d in ms]
        rs = [d["r"] for d in ms if not math.isnan(d["r"])]
        ex = [d["exact"] for d in ms]
        print(f"{label:<16}{statistics.mean(maes):>10.4f}{statistics.pstdev(maes):>9.4f}"
              f"{statistics.mean(rs):>10.4f}{statistics.pstdev(rs):>8.4f}{statistics.pstdev(ex):>10.4f}")

    print("\n=== 结论性对照 ===")
    sd1 = statistics.pstdev([d["mae"] for d in res[1]])
    sd3 = statistics.pstdev([d["mae"] for d in res[3]])
    sd7 = statistics.pstdev([d["mae"] for d in res[7]])
    sd15 = statistics.pstdev([d["mae"] for d in res[15]])
    print(f"  MAE 的抽样 sd：N=1 {sd1:.4f} → median-of-3 {sd3:.4f}（降 {100*(1-sd3/sd1):.0f}%）"
          f" → median-of-7 {sd7:.4f}（降 {100*(1-sd7/sd1):.0f}%）"
          f" → median-of-15 {sd15:.4f}（降 {100*(1-sd15/sd1):.0f}%）")
    r1 = [d["r"] for d in res[1] if not math.isnan(d["r"])]
    r7 = [d["r"] for d in res[7] if not math.isnan(d["r"])]
    print(f"  r 的抽样 sd：N=1 {statistics.pstdev(r1):.4f} → median-of-7 {statistics.pstdev(r7):.4f}"
          f"（降 {100*(1-statistics.pstdev(r7)/statistics.pstdev(r1)):.0f}%）")
    print(f"  单次抽样 r 的 90% 区间 = [{sorted(r1)[int(.05*len(r1))]:+.3f}, {sorted(r1)[int(.95*len(r1))]:+.3f}]")
    print(f"  median-of-7 r 的 90% 区间 = [{sorted(r7)[int(.05*len(r7))]:+.3f}, {sorted(r7)[int(.95*len(r7))]:+.3f}]")
    neg = sum(1 for x in r1 if x < 0) / len(r1)
    print(f"  单次抽样下 r < 0 的比例 = {neg:.1%}（若 100% 才能说『稳定为负』）")
    print(f"\n  数据日期：{meta.get('date')}；裁判温度：{meta.get('judge_temperature')}；兜底：{meta.get('fallbacks')}")


if __name__ == "__main__":
    main()
