# -*- coding: utf-8 -*-
"""把 `run_generation_eval.py` 的 raw 输出打包成「紧凑版方差/稳健性 JSON」。

为什么需要这个脚本（2026-10-20 补）：
`experiments/results/judge_variance_2026-10-20.json` 与 `reference_robustness_2026-10-20.json`
这两份紧凑件此前**仓库里没有任何代码能产出它们**——它们的 `script` 字段自称来自
`run_generation_eval.py`，但那个脚本的输出结构完全是另一套（raw JSON，含答案/上下文全文）。
底层数据其实都在 raw eval JSON 的 `samples[].judge.<臂>.detail[dim].parsed` /
`samples[].judge.<臂>.sample_scores` 里，所以是「数据可复现、缺打包脚本」。
本脚本补上这一环，让三步复现链闭合：

    ① run_generation_eval.py            → raw（出答案/裁判原始输出）
    ② package_variance_json.py（本脚本）→ compact（抽紧凑项 + 现场复算统计量）
    ③ bootstrap_judge_variance*.py      → 表（bootstrap 复算，引用 compact 的 items）

用法：
    python experiments/package_variance_json.py \
        --input experiments/results/<raw>.json \
        --out experiments/results/<compact>.json

两种模式（按 raw 里有没有 API 裁判臂自动判定）：
- **本地方差模式**（raw 只有 rule + local 臂）：产出与 `judge_variance_2026-10-20.json` 同构的紧凑件
  （items = 每条样本每维度的 N 次裁判档位；summary = item 噪声 / 中位数 vs 贪心 / 与强裁判的一致率 /
  bootstrap 复算）。
- **API 稳健性模式**（raw 有 api 臂）：产出与 `reference_robustness_2026-10-20.json` 同构的紧凑件
  （items = 每条样本的 api / local 两组 3 次观测；summary = 两侧 item 噪声 / 参照系漂移 /
  稳健性对照 / 分段 / 强裁判自一致性）。

说明：
- **不放答案/上下文全文**（紧凑件只留裁判档位与统计量，避免把语料/答案复制进结果文件）。
- 统计量**现场复算**（不是抄归档数字），因此 `--input` 换一份 raw 就能得到对应的紧凑件。
- `--elapsed-s`：原始那轮运行的**墙钟耗时**——raw JSON 里没有这个字段（raw 只记逐样本耗时），
  所以要显式传入；不传则记 null（并在报告里标明「未记录」），不编造数字。
- 脚本里不出现任何绝对路径：默认输入/输出都相对仓库根目录解析。
"""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DIMS = ("faithfulness", "relevance", "completeness")

# 默认参照文件（仓库内相对路径）：强裁判 N=1 的已归档结果 + 本地裁判 N=7 的紧凑件
DEFAULT_REFERENCE = "experiments/results/generation_eval_2026-10-19.json"
DEFAULT_VARIANCE = "experiments/results/judge_variance_2026-10-20.json"

# ---------- 静态文案（与已归档报告逐字一致；改动请同步改归档件） ----------

SEEDS_GENERATION = "greedy（temperature=0、do_sample=False，答案与 2026-10-19 逐条一致）"
SEEDS_NOTE = (
    "bootstrap 固定 seed=7、B=4000、item 内有放回重抽 7 个观测；"
    "更换 seed 会让表内数值在 1e-3 量级抖动（实测 seed=123：sd 最多移动 1.6e-3、相对 <2%），"
    "结论不受影响。裁判采样本身未设种子（生成端为贪心解码故无抽样方差）。"
)
SD_CONVENTION = "pstdev(ddof=0)：7 个观测的标准差除以 7（不是 6）"
AGGR_REF = (
    "臂 C 强裁判分取自 experiments/results/generation_eval_2026-10-19.json（本文件不重复存储）"
)
BOOT_SCHEME = (
    "固定 33 条样本，item 内有放回重抽裁判观测（先对每条样本取 k 次再取中位数）；"
    "B=4000、random.seed(7)"
)
BOOT_CAVEAT = (
    "7 个观测做非参数 bootstrap，只能看抽样波动的量级；seed 固定为 7，"
    "更换 seed 表内数值在 1e-3~2e-3 量级抖动（相对 <2%），结论不受影响"
)
BOOT_SCRIPT = "experiments/bootstrap_judge_variance.py（默认 --seed 7 --boot 4000，可逐位复算本表）"
JV_BOUNDARIES = [
    "33 条样本，只能给量级判断；r 不宜引用到小数点后四位（|r| < 0.25 + 90% 区间）",
    "greedy 的 N=1 不是采样分布的一次抽样（取众数、运行间方差为 0），与典型抽样在水平上不同",
    "本轮只跑离线臂（rule + local），未调用强裁判 API，费用 0",
    "本地裁判与生成模型同款（自评），绝对分不可作为质量证据",
]
JV_MEDIAN_NOTES = [
    "relevance 均值降 0.1515（-5.1%）：median-of-7 不是单向抬高，高估的被拉回"
    "（realistic#9 5->3、near_dup_stress#8 3->2），低估的被抬高（sample#4 3->4）",
    "unanswerable#2 两维反向移动：faithfulness 3->4（更放行幻觉）、relevance 5->3，"
    "所以「中位数让裁判更保守」不成立，方向由每个 item 自己的 7 次分布决定",
    "本节三个维度均为 N=1 greedy -> median-of-7；与强裁判的一致性对照仍只做 faithfulness",
]
RR_TEMP_SCOPE = (
    "judge_temperature（裁判采样温度）；生成端为贪心 temperature=0，答案与 2026-10-19 逐条一致"
)
RR_ROBUST_NOTE = (
    "臂与参照系同协议：N=1 行用 2026-10-19 已归档的两臂分数，"
    "N=3 行用本次 --judge-samples 3 跑出来的两臂中位数"
)
RR_SEGMENTS_DEF = "按参照系（强裁判 faithfulness）分组：判低 1-2 / 中 3 / 判高 4-5；arm 为本地 1.5B 裁判"
RR_SC_SCHEME = (
    "固定 33 条样本，item 内对强裁判的 3 次观测有放回重抽；B=4000、random.seed(7)；"
    "参照系为 2026-10-19 已归档的强裁判 N=1 faithfulness"
)
RR_SC_REFERENCE = (
    "experiments/results/generation_eval_2026-10-19.json -> samples[*].judge.api.scores.faithfulness"
)
RR_BOUNDARIES = [
    "样本量只有 33 条（27 条原金标 + 6 条本次新造拒答样本），所有比例、均值、相关都只能给量级，不能当分布估计。",
    "API 侧每格只有 3 个观测，item 级 sd 的估计本身不稳（同一本地裁判 N=3 时 sd=0.225、N=7 时 sd=0.285），只用于比较量级。",
    "只有同一次运行内的 API vs 本地对比才是同协议可比的；跨 run 比较（N=7 那次）只作旁证。",
    "greedy 的 N=1 不是采样分布的一次抽样（沿用 2026-10-19 / judge_variance 两份报告的口径），"
    "N=1 与 N=3 的差别是估计器不同，不是准确度排序。",
    "未做假设检验：本报告只给点估计与量级描述，不做任何统计推断（不做假设检验、不给区间估计）；"
    "r 保留四位只是为了与脚本输出逐位对齐，不代表精度声明。",
    "强裁判模型会滚动更新（deepseek-chat 实际解析为 deepseek-flash，"
    "/models 现返回 deepseek-flash 与 deepseek-v4-pro），绝对值随版本变化。",
    "本地裁判与生成模型同款（自评），绝对分不可作为答案质量证据。",
]


# ---------- 小工具 ----------


def r4(x):
    return round(x, 4)


def pearson(xs, ys):
    """与 experiments/bootstrap_judge_variance.py 同款相关系数（逐位一致用）。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return float("nan") if dx == 0 or dy == 0 else num / (dx * dy)


def stats(a, b):
    """两组分数的对照统计（mean / exact / within1 / mae / r）。"""
    n = len(a)
    return {
        "mean": r4(statistics.mean(a)),
        "exact": r4(sum(1 for x, y in zip(a, b) if x == y) / n),
        "within1": r4(sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / n),
        "mae": r4(sum(abs(x - y) for x, y in zip(a, b)) / n),
        "r": r4(pearson(a, b)),
    }


def agg_pearson(a, b):
    """与 bootstrap 脚本同口径的一致率统计（字段名用 pearson_r）。"""
    n = len(a)
    return {
        "mean": r4(statistics.mean(a)),
        "exact": r4(sum(1 for x, y in zip(a, b) if x == y) / n),
        "within1": r4(sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / n),
        "mae": r4(sum(abs(x - y) for x, y in zip(a, b)) / n),
        "pearson_r": r4(pearson(a, b)),
    }


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(path_str: str) -> Path:
    """相对路径按**仓库根目录**解析（脚本可能在任意 cwd 下被调用）。"""
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def judge_temperature() -> float:
    """裁判采样温度：读仓库 config.yaml（raw JSON 里没有这个字段）。"""
    try:
        import yaml

        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        return float((cfg.get("evaluation") or {}).get("judge_temperature", 0.8))
    except Exception:
        return 0.8


# ---------- 本地方差模式（judge_variance_*.json） ----------


def build_local_variance(raw: dict, reference: dict, elapsed_s, seed: int, boot: int) -> dict:
    samples = raw["samples"]
    base_by_id = {s["id"]: s for s in reference["samples"]}
    ids = [s["id"] for s in samples]
    for sid in ids:
        if sid not in base_by_id:
            raise SystemExit(f"参照文件里没有样本 {sid}（--reference 给错了？）")

    # ---- items：每条样本每维度的 N 次裁判档位（只留档位，不留答案/上下文）----
    items, cells = {}, []
    for s in samples:
        rec = {}
        for dim in DIMS:
            parsed = [int(x) for x in s["judge"]["local"]["detail"][dim]["parsed"]]
            rec[dim] = parsed
            cells.append((s["id"], dim, parsed))
        items[s["id"]] = rec
    n_per_cell = len(cells[0][2]) if cells else 0

    # ---- item 级噪声 ----
    ranges = [max(v) - min(v) for _, _, v in cells]
    same = sum(1 for _, _, v in cells if max(v) == min(v))
    sd_by_dim = {d: [statistics.pstdev(v) for _, dd, v in cells if dd == d] for d in DIMS}
    item_noise = {
        "n_cells": len(cells),
        "n_per_cell": n_per_cell,
        "identical_cells": same,
        "identical_ratio": r4(same / len(cells)),
        "range_mean": r4(sum(ranges) / len(ranges)),
        "range_hist": {
            str(k): v
            for k, v in sorted(
                {int(r): sum(1 for x in ranges if int(x) == r) for r in ranges}.items()
            )
        },
        "per_cell_sd_mean": {
            d: r4(sum(sd_by_dim[d]) / len(sd_by_dim[d])) for d in DIMS
        },
        "per_cell_sd_max": r4(max(statistics.pstdev(v) for _, _, v in cells)),
        "sd_convention": SD_CONVENTION,
    }

    # ---- median-of-N vs greedy N=1（逐维度）----
    median_vs_greedy = {}
    for dim in DIMS:
        n1 = [base_by_id[i]["judge"]["local"]["scores"][dim] for i in ids]
        nN = [s["judge"]["local"]["scores"][dim] for s in samples]
        changes = [
            {"id": i, "n1": a, "n7": b}
            for i, a, b in zip(ids, n1, nN)
            if abs(a - b) > 1e-9
        ]
        median_vs_greedy[dim] = {
            "changed_count": len(changes),
            "changed_total": len(ids),
            "mean_n1": r4(statistics.mean(n1)),
            "mean_n7": r4(statistics.mean(nN)),
            "changes": changes,
        }
    median_vs_greedy["notes"] = list(JV_MEDIAN_NOTES)

    # ---- 与强裁判（臂 C）的 faithfulness 一致率 ----
    C = [base_by_id[i]["judge"]["api"]["scores"]["faithfulness"] for i in ids]
    n1_f = [base_by_id[i]["judge"]["local"]["scores"]["faithfulness"] for i in ids]
    mN_f = [s["judge"]["local"]["scores"]["faithfulness"] for s in samples]

    # ---- bootstrap 复算（RNG 顺序必须与 bootstrap_judge_variance.py 完全一致）----
    draws = {i: items[i]["faithfulness"] for i in ids}
    C_map = {i: float(C[k]) for k, i in enumerate(ids)}

    def estimate(k):
        out = []
        for i in ids:
            v = draws[i]
            got = [random.choice(v) for _ in range(k)]
            out.append(statistics.median(got) if k > 1 else got[0])
        return out

    def score(a):
        b = [C_map[i] for i in ids]
        n = len(a)
        return {
            "mean": statistics.mean(a),
            "exact": sum(1 for x, y in zip(a, b) if x == y) / n,
            "within1": sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / n,
            "mae": sum(abs(x - y) for x, y in zip(a, b)) / n,
            "r": pearson(a, b),
        }

    random.seed(seed)
    res = {1: [], 3: [], 7: [], 15: []}
    for _ in range(boot):
        for k in res:
            res[k].append(score(estimate(k)))

    estimators = {}
    for k, label in ((1, "single"), (3, "median_of_3"), (7, "median_of_7"), (15, "median_of_15")):
        ms = res[k]
        maes = [d["mae"] for d in ms]
        rs = [d["r"] for d in ms if not math.isnan(d["r"])]
        estimators[label] = {
            "mae_mean": r4(statistics.mean(maes)),
            "mae_sd": r4(statistics.pstdev(maes)),
            # r 可能是 NaN（估计值零方差）→ 全 NaN 时记 null，不编造数字
            "r_mean": r4(statistics.mean(rs)) if rs else None,
            "r_sd": r4(statistics.pstdev(rs)) if rs else None,
        }

    def _rel_sd(label: str, field: str) -> float | None:
        base = estimators["single"][field]
        if not base:
            return None
        return round((estimators[label][field] - base) / base, 2)

    r1 = [d["r"] for d in res[1] if not math.isnan(d["r"])]
    r7 = [d["r"] for d in res[7] if not math.isnan(d["r"])]

    bootstrap_block = {
        "scheme": BOOT_SCHEME,
        "estimators": estimators,
        "mae_sd_reduction_vs_single": {
            "median_of_3": _rel_sd("median_of_3", "mae_sd"),
            "median_of_7": _rel_sd("median_of_7", "mae_sd"),
            "median_of_15": _rel_sd("median_of_15", "mae_sd"),
        },
        "r_sd_reduction_vs_single": {
            "median_of_7": _rel_sd("median_of_7", "r_sd"),
        },
        "r_90pct_interval": {
            "single": [
                round(sorted(r1)[int(0.05 * len(r1))], 3),
                round(sorted(r1)[int(0.95 * len(r1))], 3),
            ]
            if r1
            else [None, None],
            "median_of_7": [
                round(sorted(r7)[int(0.05 * len(r7))], 3),
                round(sorted(r7)[int(0.95 * len(r7))], 3),
            ]
            if r7
            else [None, None],
        },
        "share_r_below_zero": {
            "single": round(sum(1 for x in r1 if x < 0) / len(r1), 3) if r1 else None
        },
        "caveat": BOOT_CAVEAT,
        "seed": seed,
        "boot": boot,
        "script": BOOT_SCRIPT,
    }

    answers_identical = all(s["answer"] == base_by_id[s["id"]]["answer"] for s in samples)

    summary = {
        "elapsed_s": elapsed_s,
        "n_fallbacks": (raw.get("local_judge_diagnostics") or {}).get("n_fallbacks", 0),
        "answers_identical_to_2026-10-19": answers_identical,
        "item_noise": item_noise,
        "median_vs_greedy": median_vs_greedy,
        "aggregate_vs_strong_judge_faithfulness": {
            "reference": AGGR_REF,
            "greedy_N1": agg_pearson(n1_f, C),
            f"median_of_{n_per_cell}": agg_pearson(mN_f, C),
        },
        "bootstrap_judge_resample": bootstrap_block,
        "boundaries": list(JV_BOUNDARIES),
    }

    return {
        "date": raw.get("date"),
        "script": (
            "experiments/package_variance_json.py"
            "（输入：experiments/run_generation_eval.py 的 raw 输出）"
        ),
        "args": {
            "no_api": "api" not in (raw.get("arms_run") or []),
            "judge_samples": (raw.get("local_judge_diagnostics") or {}).get(
                "judge_samples", n_per_cell
            ),
        },
        "seeds": {
            "generation": SEEDS_GENERATION,
            "judge_sampling": None,
            "bootstrap": seed,
            "note": SEEDS_NOTE,
        },
        "n_samples": len(ids),
        "judge_temperature": judge_temperature(),
        "fallbacks": (raw.get("local_judge_diagnostics") or {}).get("n_fallbacks", 0),
        "items": items,
        "summary": summary,
    }


# ---------- API 稳健性模式（reference_robustness_*.json） ----------


def build_api_robustness(raw: dict, reference: dict, variance: dict, elapsed_s, seed: int, boot: int) -> dict:
    samples = raw["samples"]
    base_by_id = {s["id"]: s for s in reference["samples"]}
    ids = [s["id"] for s in samples]
    for sid in ids:
        if sid not in base_by_id:
            raise SystemExit(f"参照文件里没有样本 {sid}（--reference 给错了？）")

    api3 = {
        s["id"]: {d: [float(x) for x in s["judge"]["api"]["sample_scores"][d]] for d in DIMS}
        for s in samples
    }
    loc3 = {
        s["id"]: {d: [float(x) for x in s["judge"]["local"]["detail"][d]["parsed"]] for d in DIMS}
        for s in samples
    }
    api1 = {
        s["id"]: {d: float(s["judge"]["api"]["scores"][d]) for d in DIMS}
        for s in reference["samples"]
    }
    loc1 = {
        s["id"]: {d: float(s["judge"]["local"]["scores"][d]) for d in DIMS}
        for s in reference["samples"]
    }
    rule1 = {s["id"]: float(s["rule"]["rule_faithfulness"]) for s in reference["samples"]}

    items = {
        i: {
            "api": {d: [int(x) for x in api3[i][d]] for d in DIMS},
            "local": {d: [int(x) for x in loc3[i][d]] for d in DIMS},
        }
        for i in ids
    }

    # ---- item 噪声（api / local 两侧）----
    def item_noise(draws, n_obs):
        units = [(i, d) for i in ids for d in DIMS]
        same = sum(1 for i, d in units if len(set(draws[i][d])) == 1)
        ranges = [max(draws[i][d]) - min(draws[i][d]) for i, d in units]
        hist: Dict[int, int] = {}
        for r in ranges:
            hist[int(r)] = hist.get(int(r), 0) + 1
        return {
            "n_units": len(units),
            "n_observations_per_unit": n_obs,
            "identical_count": same,
            "identical_ratio": r4(same / len(units)),
            "range_mean": r4(statistics.mean(ranges)),
            "range_hist": {str(k): hist[k] for k in sorted(hist)},
            "units_range_ge3": sum(1 for r in ranges if r >= 3),
            "per_dim_item_sd_mean_ddof0": {
                d: r4(statistics.mean([statistics.pstdev(draws[i][d]) for i in ids]))
                for d in DIMS
            },
        }

    n_obs = len(api3[ids[0]]["faithfulness"]) if ids else 0
    noise_api = item_noise(api3, n_obs)
    noise_local = item_noise(loc3, n_obs)
    jv_noise = (variance.get("summary") or {}).get("item_noise") or {}
    noise_local["other_run_local_n7"] = {
        "source": DEFAULT_VARIANCE,
        "n_observations_per_unit": jv_noise.get("n_per_cell", 7),
        "identical_count": jv_noise.get("identical_cells"),
        "identical_ratio": jv_noise.get("identical_ratio"),
        "range_mean": jv_noise.get("range_mean"),
        "per_dim_item_sd_mean_ddof0": {
            d: r4((jv_noise.get("per_cell_sd_mean") or {}).get(d, 0.0)) for d in DIMS
        },
    }

    # ---- 参照系漂移（强裁判 N=1 greedy vs N=3 中位数）----
    drift = {}
    for d in DIMS:
        ch = [
            (i, api1[i][d], statistics.median(api3[i][d]))
            for i in ids
            if api1[i][d] != statistics.median(api3[i][d])
        ]
        drift[d] = {
            "n_changed": len(ch),
            "mean_n1_greedy": r4(statistics.mean([api1[i][d] for i in ids])),
            "mean_n3_median": r4(
                statistics.mean([statistics.median(api3[i][d]) for i in ids])
            ),
            "changed": [
                {"id": i, "n1_greedy": int(a), "n3_median": int(b)} for i, a, b in ch
            ],
        }

    # ---- 稳健性对照 ----
    api3med_f = {i: statistics.median(api3[i]["faithfulness"]) for i in ids}
    loc3med_f = {i: statistics.median(loc3[i]["faithfulness"]) for i in ids}
    loc1_f = {i: loc1[i]["faithfulness"] for i in ids}
    api1_f = {i: api1[i]["faithfulness"] for i in ids}

    robust = {
        "faithfulness": {
            "local_vs_n1": stats([loc1_f[i] for i in ids], [api1_f[i] for i in ids]),
            "local_vs_n3_median": stats(
                [loc3med_f[i] for i in ids], [api3med_f[i] for i in ids]
            ),
            "rule_vs_n1": stats([rule1[i] for i in ids], [api1_f[i] for i in ids]),
            "rule_vs_n3_median": stats(
                [rule1[i] for i in ids], [api3med_f[i] for i in ids]
            ),
        },
        "local_three_dims": {
            "vs_n1": {d: stats([loc1[i][d] for i in ids], [api1[i][d] for i in ids]) for d in DIMS},
            "vs_n3_median": {
                d: stats(
                    [statistics.median(loc3[i][d]) for i in ids],
                    [statistics.median(api3[i][d]) for i in ids],
                )
                for d in DIMS
            },
        },
        "note": RR_ROBUST_NOTE,
    }

    # ---- 分段（按参照系分组）----
    def segments(ref, arm):
        out = {}
        groups = {"low_1_2": [], "mid_3": [], "high_4_5": []}
        for i in ids:
            r = ref[i]
            groups["low_1_2" if r <= 2 else ("mid_3" if r == 3 else "high_4_5")].append(i)
        for g, lst in groups.items():
            if not lst:
                out[g] = {"n": 0}
                continue
            out[g] = {
                "n": len(lst),
                "ref_mean": r4(statistics.mean([ref[i] for i in lst])),
                "arm_mean": r4(statistics.mean([arm[i] for i in lst])),
                "mae": r4(statistics.mean([abs(arm[i] - ref[i]) for i in lst])),
            }
        return out

    segments_block = {
        "definition": RR_SEGMENTS_DEF,
        "local_arm_n1_reference": segments(api1_f, loc1_f),
        "local_arm_n3_median_reference": segments(api3med_f, loc3med_f),
        "rule_arm_n1_reference": segments(api1_f, rule1),
        "rule_arm_n3_median_reference": segments(api3med_f, rule1),
    }

    # ---- 强裁判自一致性（bootstrap；RNG 顺序与归档一致）----
    draws = {i: api3[i]["faithfulness"] for i in ids}

    def estimate(k):
        out = []
        for i in ids:
            v = draws[i]
            got = [random.choice(v) for _ in range(k)]
            out.append(statistics.median(got) if k > 1 else got[0])
        return out

    def score(a):
        b = [api1_f[i] for i in ids]
        n = len(a)
        return {
            "mae": sum(abs(x - y) for x, y in zip(a, b)) / n,
            "r": pearson(a, b),
            "exact": sum(1 for x, y in zip(a, b) if x == y) / n,
        }

    random.seed(seed)
    res = {1: [], 3: []}
    for _ in range(boot):
        for k in (1, 3):
            res[k].append(score(estimate(k)))

    sc = {"scheme": RR_SC_SCHEME, "reference": RR_SC_REFERENCE}
    for k, label in ((1, "single"), (3, "median_of_3")):
        ms = res[k]
        maes = [d["mae"] for d in ms]
        rs = [d["r"] for d in ms if not math.isnan(d["r"])]
        ex = [d["exact"] for d in ms]
        sc[label] = {
            "mae_mean": r4(statistics.mean(maes)),
            "mae_sd": r4(statistics.pstdev(maes)),
            "r_mean": r4(statistics.mean(rs)) if rs else None,
            "r_sd": r4(statistics.pstdev(rs)) if rs else None,
            "exact_sd": r4(statistics.pstdev(ex)),
        }
    _sd1 = statistics.pstdev([d["mae"] for d in res[1]])
    _sd3 = statistics.pstdev([d["mae"] for d in res[3]])
    sc["mae_sd_drop_ratio"] = r4(1 - _sd3 / _sd1) if _sd1 else None

    am = raw.get("api_meta") or {}
    probe = raw.get("api_probe") or {}
    api_meta = {
        "calls": len(am.get("calls") or []),
        "cache_hits": am.get("cache_hits"),
        "cost_cny": am.get("cost_cny"),
        "aborted": am.get("aborted"),
        "prompt_tokens": am.get("prompt_tokens"),
        "prompt_cache_hit_tokens": am.get("cache_hit_tokens"),
        "completion_tokens": am.get("completion_tokens"),
        "elapsed_s": elapsed_s,
        "model_returned": am.get("model_returned"),
        "probe": {
            "direct": bool((probe.get("direct") or {}).get("ok")),
            "proxy_7897": bool((probe.get("proxy_7897") or {}).get("ok")),
        },
    }

    return {
        "date": raw.get("date"),
        "script": (
            "experiments/package_variance_json.py"
            "（输入：experiments/run_generation_eval.py --judge-samples N 的 raw 输出）"
        ),
        "args": {
            "judge_samples": n_obs,
            "temperature": judge_temperature(),
            "temperature_scope": RR_TEMP_SCOPE,
        },
        "n_samples": len(ids),
        "api_meta": api_meta,
        "answers_identical_to_2026-10-19": all(
            s["answer"] == base_by_id[s["id"]]["answer"] for s in samples
        ),
        "n_fallbacks": (raw.get("local_judge_diagnostics") or {}).get("n_fallbacks", 0),
        "api_errors": sum(1 for s in samples if (s["judge"].get("api") or {}).get("error")),
        "items": items,
        "summary": {
            "item_noise_api": noise_api,
            "item_noise_local": noise_local,
            "reference_drift": drift,
            "robustness_vs_reference": robust,
            "segments": segments_block,
            "strong_judge_self_consistency": sc,
            "boundaries": list(RR_BOUNDARIES),
        },
    }


# ---------- 入口 ----------


def main():
    ap = argparse.ArgumentParser(
        description="把 run_generation_eval.py 的 raw 输出打包成紧凑版方差/稳健性 JSON"
    )
    ap.add_argument("--input", required=True, help="raw eval JSON（run_generation_eval.py 的输出）")
    ap.add_argument("--out", required=True, help="紧凑件输出路径")
    ap.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help=f"参照文件（强裁判 N=1 与贪心 N=1 的已归档结果，默认 {DEFAULT_REFERENCE}）",
    )
    ap.add_argument(
        "--variance",
        default=DEFAULT_VARIANCE,
        help=f"本地 N=7 方差紧凑件（API 模式的跨 run 旁证用，默认 {DEFAULT_VARIANCE}）",
    )
    ap.add_argument(
        "--elapsed-s",
        type=float,
        default=None,
        help="原始那轮运行的墙钟耗时（秒）；raw JSON 里没有这个字段，不传则记 null",
    )
    ap.add_argument("--bootstrap-seed", type=int, default=7, help="bootstrap 的 random.seed（默认 7）")
    ap.add_argument("--bootstrap-boot", type=int, default=4000, help="bootstrap 次数 B（默认 4000）")
    args = ap.parse_args()

    raw = load_json(Path(args.input))
    reference = load_json(_resolve(args.reference))

    has_api = any((s.get("judge") or {}).get("api") for s in raw["samples"])
    if has_api:
        variance_path = _resolve(args.variance)
        if not variance_path.exists():
            raise SystemExit(
                f"找不到方差紧凑件：{variance_path}（--variance 给错了？）"
                "——它提供 API 模式里「另一次本地裁判 N=7 运行」的跨 run 旁证数字。"
            )
        variance = load_json(variance_path)
        out = build_api_robustness(
            raw, reference, variance, args.elapsed_s, args.bootstrap_seed, args.bootstrap_boot
        )
    else:
        out = build_local_variance(
            raw, reference, args.elapsed_s, args.bootstrap_seed, args.bootstrap_boot
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[ok] 已写出紧凑件：{out_path}")
    print(f"     items={len(out['items'])} n_samples={out['n_samples']}")
    if args.elapsed_s is None:
        print("     注意：未提供 --elapsed-s，summary 里的 elapsed_s 记为 null（不编造数字）")


if __name__ == "__main__":
    main()
