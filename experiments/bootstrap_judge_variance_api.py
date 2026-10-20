# -*- coding: utf-8 -*-
"""复算「强裁判自身抽样波动」与「参照系稳健性」两张表（API 侧）。

背景：2026-10-19 的三臂对照把 DeepSeek 强裁判当作**参照系**，但那一份强裁判每样本只调了一次
（N=1 贪心）。2026-10-20 补测了一轮 `--judge-samples 3`，于是有两件事可以复算：

1. **第七节（E 表）**：强裁判自己的抽样波动有多大。做法与本地那份
   `experiments/bootstrap_judge_variance.py` 同构——固定 33 条样本与答案，在每个 item 内部
   对已采到的 3 次强裁判观测做**有放回重抽**，重算与「已归档的强裁判 N=1 参照」之间的
   MAE / r / 精确一致率，重复 B 次（`random.seed(7)`）。
2. **第五节（C 表）**：把参照系从「N=1 贪心」换成「N=3 中位数」之后，主结论是否存活。
   这部分**是确定性的**，不涉及抽样，直接算。

RNG 调用顺序固定为：
    for _ in range(B):
        for k in (1, 3):
            for i in ids:
                got = [random.choice(v) for _ in range(k)]
（k > 1 时再取 `statistics.median`）
换 seed 或换嵌套顺序，数字会变；重写本脚本时必须保持这个顺序，才能与报告逐位对照。

数据来源（都不含答案/上下文全文）：
- `--variance`：`reference_robustness_2026-10-20.json` 的 `items`
  （每条样本每维度 3 次强裁判观测 + 3 次本地裁判观测）
- `--results`：`generation_eval_2026-10-19.json` 的臂 C（强裁判 N=1 参照分）与臂 B/规则臂

用法：
    python experiments/bootstrap_judge_variance_api.py            # 默认 seed=7、B=4000
    python experiments/bootstrap_judge_variance_api.py --boot 500 --seed 1
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


def agree(a, b):
    """臂 a 对参照系 b 的一致性指标。"""
    n = len(a)
    return {
        "mean": statistics.mean(a),
        "exact": sum(1 for x, y in zip(a, b) if x == y) / n,
        "within1": sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / n,
        "mae": sum(abs(x - y) for x, y in zip(a, b)) / n,
        "r": pearson(a, b),
    }


def load_variance(path):
    """{ id: {"api": {dim: [3 观测]}, "local": {dim: [3 观测]}} }。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["items"]
    out = {}
    for rid, per_arm in items.items():
        out[str(rid)] = {
            arm: {dim: [float(x) for x in per_arm[arm][dim]] for dim in DIMS}
            for arm in ("api", "local")
        }
    return data, out


def load_results(path):
    """{ id: {"api_n1": {dim}, "local_n1": {dim}, "rule": float} }。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for r in data["samples"]:
        out[str(r["id"])] = {
            "api_n1": {d: float(r["judge"]["api"]["scores"][d]) for d in DIMS},
            "local_n1": {d: float(r["judge"]["local"]["scores"][d]) for d in DIMS},
            "rule": float(r["rule"]["rule_faithfulness"]),
        }
    return data, out


def med3(v):
    return statistics.median(v)


def main():
    ap = argparse.ArgumentParser(description="复算强裁判自身的抽样波动（API 侧 N=3）与参照系稳健性")
    ap.add_argument("--results", default="experiments/results/generation_eval_2026-10-19.json",
                    help="提供 N=1 参照分与规则臂的已归档结果")
    ap.add_argument("--variance", default="experiments/results/reference_robustness_2026-10-20.json",
                    help="提供 N=3 观测的参照系稳健性归档")
    ap.add_argument("--boot", type=int, default=4000, help="bootstrap 次数 B（默认 4000）")
    ap.add_argument("--seed", type=int, default=7, help="random.seed（默认 7，与报告一致）")
    args = ap.parse_args()

    var_meta, draws = load_variance(args.variance)
    res_meta, base = load_results(args.results)
    ids = [str(r) for r in var_meta["items"]]
    missing = [i for i in ids if i not in base]
    if missing:
        raise SystemExit(f"两份归档的样本 id 对不上：{missing}")

    api_draws = {i: draws[i]["api"]["faithfulness"] for i in ids}
    api3 = {i: {d: med3(draws[i]["api"][d]) for d in DIMS} for i in ids}
    loc3 = {i: {d: med3(draws[i]["local"][d]) for d in DIMS} for i in ids}
    api1 = {i: {d: base[i]["api_n1"][d] for d in DIMS} for i in ids}
    loc1 = {i: {d: base[i]["local_n1"][d] for d in DIMS} for i in ids}
    rule = {i: base[i]["rule"] for i in ids}

    print("=== 数据来源 ===")
    print(f"  参照系稳健性归档：{args.variance}（{len(ids)} 条样本，API 侧每格 3 次观测）")
    print(f"  N=1 参照与规则臂：{args.results}")
    print(f"  seed={args.seed}  B={args.boot}")

    # ---------------- 第五节（C 表）：参照系稳健性（确定性） ----------------
    print("\n=== 主结论稳健性（faithfulness；参照系 = 强裁判）===")
    print(f"{'臂':<12}{'参照系':<16}{'mean':>8}{'exact':>8}{'±1':>8}{'MAE':>8}{'r':>9}")
    rows = [
        ("本地 1.5B", "N=1（已提交口径）", [loc1[i]["faithfulness"] for i in ids], [api1[i]["faithfulness"] for i in ids]),
        ("本地 1.5B", "N=3 中位数", [loc3[i]["faithfulness"] for i in ids], [api3[i]["faithfulness"] for i in ids]),
        ("规则判据", "N=1", [rule[i] for i in ids], [api1[i]["faithfulness"] for i in ids]),
        ("规则判据", "N=3 中位数", [rule[i] for i in ids], [api3[i]["faithfulness"] for i in ids]),
    ]
    for name, ref_name, a, b in rows:
        st = agree(a, b)
        print(f"{name:<12}{ref_name:<16}{st['mean']:>8.4f}{st['exact']:>8.4f}{st['within1']:>8.4f}"
              f"{st['mae']:>8.4f}{st['r']:>+9.4f}")

    print("\n=== 本地 1.5B 三维（对强裁判）===")
    print(f"{'维度':<14}{'参照系':<14}{'mean':>8}{'exact':>8}{'±1':>8}{'MAE':>8}{'r':>9}")
    for d in DIMS:
        st1 = agree([loc1[i][d] for i in ids], [api1[i][d] for i in ids])
        st3 = agree([loc3[i][d] for i in ids], [api3[i][d] for i in ids])
        print(f"{d:<14}{'N=1':<14}{st1['mean']:>8.4f}{st1['exact']:>8.4f}{st1['within1']:>8.4f}"
              f"{st1['mae']:>8.4f}{st1['r']:>+9.4f}")
        print(f"{d:<14}{'N=3 中位数':<14}{st3['mean']:>8.4f}{st3['exact']:>8.4f}{st3['within1']:>8.4f}"
              f"{st3['mae']:>8.4f}{st3['r']:>+9.4f}")

    # ---------------- 第三节：强裁判自身的 item 级噪声 ----------------
    print("\n=== 强裁判自身的 item 级抽样噪声（3 次观测 / 99 个「样本×维度」单元）===")
    for arm_name, arm in (("API（本次 N=3）", "api"), ("本地 1.5B（同一次运行）", "local")):
        same = 0
        ranges = []
        for i in ids:
            for d in DIMS:
                v = draws[i][arm][d]
                ranges.append(max(v) - min(v))
                if max(v) == min(v):
                    same += 1
        hist = {}
        for r in ranges:
            hist[int(r)] = hist.get(int(r), 0) + 1
        print(f"  {arm_name}：{same}/{len(ranges)} = {100 * same / len(ranges):.1f}% 单元三次完全相同；"
              f"极差均值 {statistics.mean(ranges):.2f} 档；极差分布 {dict(sorted(hist.items()))}")
    print("  （API 侧每格只有 3 个观测，上面是量级参考，不是稳定性的证明；"
          "极差 ≥3 档的单元数见归档 summary.item_noise_api.units_range_ge3）")

    # ---------------- 第七节（E 表）：强裁判的自洽性（bootstrap） ----------------
    def estimate(k):
        out = []
        for i in ids:
            v = api_draws[i]
            got = [random.choice(v) for _ in range(k)]
            out.append(statistics.median(got) if k > 1 else got[0])
        return out

    def score(a):
        b = [api1[i]["faithfulness"] for i in ids]
        n = len(a)
        return {
            "mae": sum(abs(x - y) for x, y in zip(a, b)) / n,
            "r": pearson(a, b),
            "exact": sum(1 for x, y in zip(a, b) if x == y) / n,
        }

    random.seed(args.seed)
    B = args.boot
    res = {1: [], 3: []}
    for _ in range(B):
        for k in (1, 3):
            res[k].append(score(estimate(k)))

    print(f"\n=== 强裁判自洽性（相对已归档的 N=1 参照；item 内有放回重抽，B={B}）===")
    print(f"{'估计器':<16}{'MAE 均值':>10}{'MAE sd':>9}{'r 均值':>10}{'r sd':>8}{'exact sd':>10}")
    for k, label in ((1, "single（N=1）"), (3, "median-of-3")):
        ms = res[k]
        maes = [d["mae"] for d in ms]
        rs = [d["r"] for d in ms if not math.isnan(d["r"])]
        ex = [d["exact"] for d in ms]
        print(f"{label:<16}{statistics.mean(maes):>10.4f}{statistics.pstdev(maes):>9.4f}"
              f"{statistics.mean(rs):>10.4f}{statistics.pstdev(rs):>8.4f}{statistics.pstdev(ex):>10.4f}")

    sd1 = statistics.pstdev([d["mae"] for d in res[1]])
    sd3 = statistics.pstdev([d["mae"] for d in res[3]])
    print(f"\n  MAE 的抽样 sd：single {sd1:.4f} → median-of-3 {sd3:.4f}（降 {100 * (1 - sd3 / sd1):.0f}%）")
    print("  对照：本地 1.5B 与参照的 MAE 是 1.94 档、r 是 −0.16 量级（见第五节）——"
          "两个裁判之间的分歧不是参照系噪声造成的。")
    print(f"\n  数据日期：{var_meta.get('date')}；参照系归档：{res_meta.get('date')}")


if __name__ == "__main__":
    main()
