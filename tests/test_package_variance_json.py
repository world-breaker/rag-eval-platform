"""「紧凑版方差/稳健性 JSON」打包脚本的离线单测（不调 API、不加载大模型）。

守的是一条复现链缺口（2026-10-20 补）：`judge_variance_2026-10-20.json` 与
`reference_robustness_2026-10-20.json` 两份紧凑件此前**仓库里没有任何代码能产出它们**
（`script` 字段自称来自 run_generation_eval.py，但那个脚本的输出结构是另一套）。
`experiments/package_variance_json.py` 补上这一环——这里断言它的**产出结构**与归档件一致。

单测用合成的极小 raw（3 条样本 × 3 维度 × 3 次观测）跑，不依赖任何已归档大文件，
因此 CI 上可离线跑。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.package_variance_json import (  # noqa: E402
    DIMS,
    build_api_robustness,
    build_local_variance,
)

IDS = ["dsA#1", "dsA#2", "dsA#3"]


def _raw_local(n_obs=3):
    """合成的 raw：只有 rule + local 臂（无 API 臂）→ 走「本地方差模式」。"""
    samples = []
    for k, sid in enumerate(IDS):
        parsed = {d: [float(2 + (k + i) % 3) for i in range(n_obs)] for d in DIMS}
        samples.append(
            {
                "id": sid,
                "answer": f"answer-{sid}",
                "judge": {
                    "local": {
                        "scores": {d: sorted(parsed[d])[len(parsed[d]) // 2] for d in DIMS},
                        "detail": {
                            d: {
                                "score": 3.0,
                                "source": "llm_median",
                                "parsed": parsed[d],
                                "raws": [str(x) for x in parsed[d]],
                                "reason": None,
                            }
                            for d in DIMS
                        },
                    }
                },
            }
        )
    return {
        "date": "2026-10-20",
        "arms_run": ["rule", "local"],
        "api_meta": {},
        "api_probe": {},
        "local_judge_diagnostics": {"n_fallbacks": 0, "judge_samples": n_obs},
        "samples": samples,
    }


def _raw_api(n_obs=3):
    """合成的 raw：有 API 臂 → 走「API 稳健性模式」。"""
    raw = _raw_local(n_obs)
    raw["arms_run"] = ["rule", "local", "api"]
    for s in raw["samples"]:
        s["judge"]["api"] = {
            "scores": {d: 4.0 for d in DIMS},
            "sample_scores": {d: [4.0, 5.0, 3.0][:n_obs] for d in DIMS},
            "error": None,
        }
    raw["api_meta"] = {
        "model": "deepseek-chat",
        "model_returned": "deepseek-flash",
        "calls": [{"id": s["id"]} for s in raw["samples"]],
        "cache_hits": 0,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cache_hit_tokens": 0,
        "cost_cny": 0.01,
        "aborted": False,
    }
    raw["api_probe"] = {"direct": {"ok": True}, "proxy_7897": {"ok": True}}
    return raw


def _reference():
    """合成的参照件：臂 C（强裁判 N=1）+ 臂 B（本地裁判 N=1）+ 规则臂。"""
    samples = []
    for k, sid in enumerate(IDS):
        samples.append(
            {
                "id": sid,
                "answer": f"answer-{sid}",
                "rule": {"rule_faithfulness": 3.0 + k},
                "judge": {
                    "local": {"scores": {d: 3.0 for d in DIMS}},
                    "api": {"scores": {d: 4.0 for d in DIMS}},
                },
            }
        )
    return {"samples": samples}


def test_local_variance_output_matches_archived_schema():
    """本地方差模式：字段名与嵌套结构必须与归档件 judge_variance_*.json 一致。"""
    out = build_local_variance(_raw_local(), _reference(), 12.5, seed=7, boot=50)

    assert set(out) == {
        "date",
        "script",
        "args",
        "seeds",
        "n_samples",
        "judge_temperature",
        "fallbacks",
        "items",
        "summary",
    }
    assert set(out["seeds"]) == {"generation", "judge_sampling", "bootstrap", "note"}
    assert set(out["args"]) == {"no_api", "judge_samples"}
    assert out["args"]["no_api"] is True
    assert out["n_samples"] == len(IDS)
    assert out["fallbacks"] == 0

    # items：每条样本每维度的 N 次裁判档位（纯数字，不含答案/上下文）
    assert set(out["items"]) == set(IDS)
    for sid, rec in out["items"].items():
        assert set(rec) == set(DIMS)
        for d in DIMS:
            assert len(rec[d]) == 3 and all(isinstance(x, int) for x in rec[d])

    s = out["summary"]
    assert set(s) == {
        "elapsed_s",
        "n_fallbacks",
        "answers_identical_to_2026-10-19",
        "item_noise",
        "median_vs_greedy",
        "aggregate_vs_strong_judge_faithfulness",
        "bootstrap_judge_resample",
        "boundaries",
    }
    assert s["elapsed_s"] == 12.5
    assert s["item_noise"]["n_cells"] == len(IDS) * len(DIMS)
    assert set(s["item_noise"]) == {
        "n_cells",
        "n_per_cell",
        "identical_cells",
        "identical_ratio",
        "range_mean",
        "range_hist",
        "per_cell_sd_mean",
        "per_cell_sd_max",
        "sd_convention",
    }
    assert set(s["median_vs_greedy"]) == set(DIMS) | {"notes"}
    assert set(s["aggregate_vs_strong_judge_faithfulness"]) == {
        "reference",
        "greedy_N1",
        "median_of_3",
    }
    boot = s["bootstrap_judge_resample"]
    assert set(boot) == {
        "scheme",
        "estimators",
        "mae_sd_reduction_vs_single",
        "r_sd_reduction_vs_single",
        "r_90pct_interval",
        "share_r_below_zero",
        "caveat",
        "seed",
        "boot",
        "script",
    }
    assert set(boot["estimators"]) == {"single", "median_of_3", "median_of_7", "median_of_15"}

    # 紧凑件不许夹带答案/上下文全文
    blob = str(out)
    assert "answer-" not in blob, "紧凑件不许含答案全文"


def test_api_robustness_output_matches_archived_schema():
    """API 稳健性模式：字段名与嵌套结构必须与归档件 reference_robustness_*.json 一致。"""
    variance = {
        "summary": {
            "item_noise": {
                "n_per_cell": 7,
                "identical_cells": 38,
                "identical_ratio": 0.3838,
                "range_mean": 0.8586,
                "per_cell_sd_mean": {d: 0.3 for d in DIMS},
            }
        }
    }
    out = build_api_robustness(_raw_api(), _reference(), variance, 20.0, seed=7, boot=50)

    assert set(out) == {
        "date",
        "script",
        "args",
        "n_samples",
        "api_meta",
        "answers_identical_to_2026-10-19",
        "n_fallbacks",
        "api_errors",
        "items",
        "summary",
    }
    assert set(out["args"]) == {"judge_samples", "temperature", "temperature_scope"}
    assert set(out["api_meta"]) == {
        "calls",
        "cache_hits",
        "cost_cny",
        "aborted",
        "prompt_tokens",
        "prompt_cache_hit_tokens",
        "completion_tokens",
        "elapsed_s",
        "model_returned",
        "probe",
    }
    assert set(out["api_meta"]["probe"]) == {"direct", "proxy_7897"}

    for sid, rec in out["items"].items():
        assert set(rec) == {"api", "local"}
        for arm in ("api", "local"):
            assert set(rec[arm]) == set(DIMS)
            assert len(rec[arm]["faithfulness"]) == 3

    s = out["summary"]
    assert set(s) == {
        "item_noise_api",
        "item_noise_local",
        "reference_drift",
        "robustness_vs_reference",
        "segments",
        "strong_judge_self_consistency",
        "boundaries",
    }
    assert set(s["item_noise_local"]) == {
        "n_units",
        "n_observations_per_unit",
        "identical_count",
        "identical_ratio",
        "range_mean",
        "range_hist",
        "units_range_ge3",
        "per_dim_item_sd_mean_ddof0",
        "other_run_local_n7",
    }
    assert set(s["reference_drift"]) == set(DIMS)
    assert set(s["robustness_vs_reference"]) == {
        "faithfulness",
        "local_three_dims",
        "note",
    }
    assert set(s["segments"]) == {
        "definition",
        "local_arm_n1_reference",
        "local_arm_n3_median_reference",
        "rule_arm_n1_reference",
        "rule_arm_n3_median_reference",
    }
    sc = s["strong_judge_self_consistency"]
    assert set(sc) == {"scheme", "reference", "single", "median_of_3", "mae_sd_drop_ratio"}
    assert set(sc["single"]) == {"mae_mean", "mae_sd", "r_mean", "r_sd", "exact_sd"}


def test_script_has_no_absolute_paths_and_no_key():
    """脚本里不许出现本机绝对路径，也不许出现任何 API key 字面量。"""
    src = (ROOT / "experiments/package_variance_json.py").read_text(encoding="utf-8")
    for bad in ("C:\\Users", "D:\\Temp", "D:\\programmer", "/home/", "sk-"):
        assert bad not in src, f"打包脚本里不许出现 {bad}"


if __name__ == "__main__":
    test_local_variance_output_matches_archived_schema()
    test_api_robustness_output_matches_archived_schema()
    test_script_has_no_absolute_paths_and_no_key()
    print("✅ 全部打包脚本单测通过")
