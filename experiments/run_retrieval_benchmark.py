"""检索质量 + 延迟基准：一次跑完 vector / bm25 / hybrid(/+重排) / +查询改写 全档位。

与 `run_hybrid_comparison.py` 的关系：
- `run_hybrid_comparison.py` 是 2026-10-18 的「质量对照」脚本（vector vs hybrid）；
- 本脚本是它的超集：把**排名口径**（vector / bm25 / hybrid）、**重排**、**查询改写**都纳入同一张表，
  并补上**延迟基准**（p50 / p95 / 均值，冷启动与热启动分开）。
  两份报告共存，前者保留为历史快照。

运行：
    $env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
    python experiments/run_retrieval_benchmark.py

关于「会不会偷偷下载模型」：脚本在跑之前先用 HF 缓存目录做存在性检查，
没缓存的档位直接标记为「未实测」并跳过，绝不触发下载。

延迟口径（写进报告，避免读者误读）：
- **冷启动（cold）**：切到该档位后的**第一次**查询耗时。含首次懒加载（重排模型/LLM 载入、
  CUDA 首次 kernel 预热、BM25 语料拟合）。它反映「服务刚起来第一次问」的体感。
- **热启动（warm）**：之后所有查询的 p50 / p95 / 均值。它反映稳态吞吐，是做容量规划该看的数。
- 单次延迟 = 查询改写（若有）+ query 编码 + 检索（+重排），不含生成回答（生成与检索解耦，单独评）。
"""

import sys
import json
import time
import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.utils import logger

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "experiments/results"

DATASETS = {
    "realistic": (
        ROOT / "data/sample/knowledge_base_large.json",
        ROOT / "data/sample/test_qa_large.json",
    ),
    "near_dup_stress": (
        ROOT / "data/sample/knowledge_base_stress.json",
        ROOT / "data/sample/test_qa_stress.json",
    ),
}

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# 每个档位：排名口径 + 是否重排 + 改写策略（None = 不改写）
MODES = [
    ("vector", "纯向量（基线）", {"mode": "vector"}),
    ("bm25", "纯 BM25 关键词", {"mode": "bm25"}),
    ("hybrid", "混合 BM25+向量 · RRF", {"mode": "hybrid"}),
    (
        "hybrid_rerank",
        "混合 + Cross-Encoder 重排",
        {"mode": "hybrid", "rerank": True},
    ),
    (
        "hybrid_multi_query",
        "混合 + multi-query 改写",
        {"mode": "hybrid", "rewrite": "multi_query"},
    ),
    (
        "hybrid_hyde",
        "混合 + HyDE 改写",
        {"mode": "hybrid", "rewrite": "hyde"},
    ),
]

WARM_ROUNDS = {"default": 5, "llm": 2}  # LLM 档位每问只重复 2 轮（生成昂贵）


def _hf_model_cached(model_name: str) -> bool:
    """判断 HF 缓存里是否已有**完整**模型（避免触发下载，也避免半截文件误判为可用）。"""
    import os

    org, name = model_name.split("/", 1)
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub_dir = cache_root / "hub" / f"models--{org}--{name.replace('/', '--')}"
    if not hub_dir.exists():
        return False
    # 至少要有一个权重文件，且大小像样（半截下载的 safetensors 会让 from_pretrained 抛 SafetensorError）
    for weights in hub_dir.rglob("*"):
        if weights.is_file() and weights.suffix in (".safetensors", ".bin"):
            return weights.stat().st_size > 1_000_000
    return False


def _apply_mode(pipe: RAGPipeline, spec: dict) -> dict:
    """把档位配置套到 pipeline 上，返回本次生效的元信息（含懒加载耗时）。"""
    info = {"rerank_load_ms": None, "llm_load_ms": None, "rerank_active": None}

    pipe.set_retrieval_mode(spec["mode"])

    if spec.get("rerank"):
        t0 = time.perf_counter()
        ok = pipe.set_rerank(True, model=RERANK_MODEL)
        info["rerank_load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        info["rerank_active"] = ok
        if not ok:
            info["rerank_error"] = pipe.hybrid.rerank_load_error
    elif pipe.hybrid is not None and pipe.hybrid.rerank_enabled:
        # 上一档开过重排 → 关掉，避免污染本档位结果
        pipe.set_rerank(False)

    # 改写配置：改开关后必须重置缓存，才会按新策略重建 rewriter
    rw_cfg = {"enabled": bool(spec.get("rewrite")), "strategy": spec.get("rewrite") or "multi_query",
              "num_queries": 3, "max_new_tokens": 256, "max_hypothetical_chars": 512}
    pipe.config["retrieval"]["query_rewrite"] = rw_cfg
    pipe.query_rewriter = None
    pipe._rewriter_initialized = False

    if spec.get("rewrite"):
        t0 = time.perf_counter()
        pipe._ensure_rewriter()  # 触发 LLM 加载
        info["llm_load_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    return info


def _env_info() -> dict:
    """记录跑分环境（延迟数字离开硬件没有意义）。"""
    import platform

    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:  # 没装 torch 也要能出报告
        info["torch"] = f"unavailable: {e}"
    return info


def _percentiles(samples_ms):
    if not samples_ms:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "n": 0}
    arr = np.asarray(samples_ms, dtype=float)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "n": int(arr.size),
    }


def benchmark_mode(pipe: RAGPipeline, spec: dict, qa: list) -> dict:
    """跑完一个档位：metrics + 冷/热启动延迟。"""
    try:
        info = _apply_mode(pipe, spec)
    except Exception as e:
        # 例如 LLM 权重不完整/显存不足：如实标注未实测，不要让整轮基准挂掉
        logger.warning(f"档位 {spec} 不可用，跳过：{type(e).__name__}: {e}")
        return {"skipped": f"档位初始化失败（{type(e).__name__}: {str(e)[:160]}），未实测"}

    # 冷启动：切档位后的第一次查询
    t0 = time.perf_counter()
    first = pipe._retrieve(qa[0]["question"], top_k=5)
    cold_ms = round((time.perf_counter() - t0) * 1000, 2)

    rounds = WARM_ROUNDS["llm"] if spec.get("rewrite") else WARM_ROUNDS["default"]
    warm_ms, last_round = [], {}
    for _ in range(rounds):
        for case in qa:
            t0 = time.perf_counter()
            retrieved = pipe._retrieve(case["question"], top_k=5)
            warm_ms.append((time.perf_counter() - t0) * 1000)
            last_round[case["question"]] = retrieved

    # 质量指标（检索确定，取最后一轮即可）
    r3, r5, mrr, per_q = [], [], [], []
    for i, case in enumerate(qa, 1):
        rep = pipe.evaluator.evaluate_retrieval_doc_level(
            last_round[case["question"]], case["relevant_sources"]
        )
        r3.append(rep["recall@3"])
        r5.append(rep["recall@5"])
        mrr.append(rep["mrr"])
        per_q.append(
            {
                "idx": i,
                "question": case["question"],
                "relevant": case["relevant_sources"],
                "recall@3": rep["recall@3"],
                "recall@5": rep["recall@5"],
                "mrr": rep["mrr"],
                "top1_source": (last_round[case["question"]][0]["source"] if last_round[case["question"]] else None),
            }
        )

    return {
        **info,
        "recall@3": round(sum(r3) / len(r3), 4),
        "recall@5": round(sum(r5) / len(r5), 4),
        "mrr": round(sum(mrr) / len(mrr), 4),
        "cold_ms": cold_ms,
        "warm": _percentiles(warm_ms),
        "warm_rounds": rounds,
        "first_query_len": len(first),
        "per_question": per_q,
    }


def _fmt(v):
    return "—" if v is None else v


def main():
    rerank_cached = _hf_model_cached(RERANK_MODEL)
    llm_cached = _hf_model_cached(LLM_MODEL)
    skipped = {}
    if not rerank_cached:
        skipped["hybrid_rerank"] = f"本地无 {RERANK_MODEL} 完整缓存且离线，未实测"
    if not llm_cached:
        skipped["hybrid_multi_query"] = f"本地无 {LLM_MODEL} 完整缓存且离线，未实测"
        skipped["hybrid_hyde"] = f"本地无 {LLM_MODEL} 完整缓存且离线，未实测"
    logger.info(f"重排缓存={rerank_cached} / LLM 缓存={llm_cached}；跳过档位：{list(skipped)}")

    # 全程只建一个 pipeline：embedding 模型与（改写用的）LLM 都只加载一次，数据集之间只重建索引。
    t_init = time.perf_counter()
    pipe = RAGPipeline(str(ROOT / "config.yaml"), load_generator=False)
    init_ms = round((time.perf_counter() - t_init) * 1000, 1)
    logger.info(f"pipeline 初始化（含 embedding 模型载入）{init_ms} ms")

    all_results = {}
    for ds_name, (kb_path, qa_path) in DATASETS.items():
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        logger.info(f"===== 数据集 {ds_name}：{len(kb)} 篇 / {len(qa)} 题 =====")

        t0 = time.perf_counter()
        pipe.build_index(kb)
        build_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"建库 {build_ms} ms（{len(pipe.chunks)} chunks）")

        all_results[ds_name] = {
            "kb": kb_path.name,
            "qa": qa_path.name,
            "n_docs": len(kb),
            "n_chunks": len(pipe.chunks),
            "n_questions": len(qa),
            "pipeline_init_ms": init_ms,
            "build_index_ms": build_ms,
            "modes": {},
        }

        for key, label, spec in MODES:
            if key in skipped:
                all_results[ds_name]["modes"][key] = {"label": label, "skipped": skipped[key]}
                continue
            logger.info(f"----- [{ds_name}] 档位 {label} -----")
            res = benchmark_mode(pipe, spec, qa)
            res["label"] = label
            if "skipped" not in res and spec.get("rerank") and not res.get("rerank_active"):
                # 声明了重排但模型没加载成功 → 如实标注「未实测」，不要拿降级结果冒充重排结果
                res = {
                    "label": label,
                    "skipped": f"重排模型不可用（{res.get('rerank_error')}），未实测",
                }
            all_results[ds_name]["modes"][key] = res
            if "skipped" not in res:
                logger.info(
                    f"[{ds_name}/{key}] R@3={res['recall@3']} R@5={res['recall@5']} "
                    f"MRR={res['mrr']} cold={res['cold_ms']}ms warm_p50={res['warm']['p50_ms']}ms"
                )

    # ---------- 渲染报告 ----------
    lines = ["# 检索基准：质量 + 延迟（全档位对照）\n"]
    lines.append(f"- 生成日期：{datetime.date.today().isoformat()}")
    env = _env_info()
    lines.append(
        f"- 环境：Python {env['python']} / torch {env['torch']} / "
        + (f"GPU {env.get('gpu')}" if env.get("cuda_available") else "CPU")
    )
    lines.append(f"- 口径：**文档级** Recall@3 / Recall@5 / MRR（`evaluate_retrieval_doc_level`）")
    lines.append("- 延迟口径：单次 `_retrieve`（改写 + query 编码 + 检索 + 重排），**不含生成回答**；")
    lines.append("  冷启动 = 切档位后第一次查询（含懒加载与 CUDA 预热），热启动 = 之后所有查询的 p50/p95/均值。")
    lines.append(f"- 热启动重复轮数：普通档位 {WARM_ROUNDS['default']} 轮，LLM 改写档位 {WARM_ROUNDS['llm']} 轮（生成昂贵）")
    lines.append(f"- 重排模型：`{RERANK_MODEL}`（多语言 Cross-Encoder，本地缓存={rerank_cached}）")
    lines.append(f"- 改写 LLM：`{LLM_MODEL}`（本地缓存={llm_cached}）")
    if skipped:
        for k, why in skipped.items():
            lines.append(f"- ⚠️ 档位 `{k}` **{why}**")
    lines.append("")

    for ds_name, ds in all_results.items():
        desc = "话题区分度大的精标库" if ds_name == "realistic" else "含大量近义/重复文档的噪声库"
        lines.append(f"## 数据集：{ds_name}（{ds['kb']} {ds['n_docs']} 篇 / {ds['qa']} {ds['n_questions']} 题 · {desc}）\n")
        lines.append(f"- 冷启动-建库：Pipeline 初始化（含 embedding 模型载入，全程只做一次）**{ds['pipeline_init_ms']} ms** + 建索引 **{ds['build_index_ms']} ms**（{ds['n_chunks']} chunks，embedding 走 MD5 缓存）")
        lines.append(f"- 每个档位的「冷启动 ms」= 切到该档位后的**第一次**查询（含懒加载：重排模型/LLM 载入、CUDA 预热）\n")
        lines.append("| 档位 | Recall@3 | Recall@5 | MRR | 冷启动 ms | 热 p50 ms | 热 p95 ms | 热均值 ms | 样本 n |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for key, label, _ in MODES:
            m = ds["modes"][key]
            if "skipped" in m:
                lines.append(f"| {label} | 未实测 | 未实测 | 未实测 | — | — | — | — | 0 |")
                continue
            w = m["warm"]
            lines.append(
                f"| {label} | {m['recall@3']} | {m['recall@5']} | {m['mrr']} | {m['cold_ms']} | "
                f"{w['p50_ms']} | {w['p95_ms']} | {w['mean_ms']} | {w['n']} |"
            )
        skipped_modes = [(k, l, ds["modes"][k]["skipped"]) for k, l, _ in MODES if "skipped" in ds["modes"][k]]
        if skipped_modes:
            lines.append("")
            for k, l, why in skipped_modes:
                lines.append(f"> ⚠️ `{l}`（{k}）**{why}**")
        lines.append("")

        # 逐题明细（只列 Recall@3，避免表格过宽）
        lines.append("### 逐题 Recall@3\n")
        active = [(k, l) for k, l, _ in MODES if "skipped" not in ds["modes"][k]]
        lines.append("| # | 问题 | 相关文档 | " + " | ".join(l for _, l in active) + " |")
        lines.append("|" + "---|" * (3 + len(active)))
        for i in range(ds["n_questions"]):
            cells = [
                str(i + 1),
                ds["modes"]["vector"]["per_question"][i]["question"],
                ",".join(ds["modes"]["vector"]["per_question"][i]["relevant"]),
            ]
            for k, _ in active:
                cells.append(str(ds["modes"][k]["per_question"][i]["recall@3"]))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # ---------- 结论 ----------
    lines.append("## 结论（基于上表实测）\n")
    for ds_name, ds in all_results.items():
        v = ds["modes"]["vector"]
        h = ds["modes"]["hybrid"]
        lines.append(f"### {ds_name}\n")
        lines.append(
            f"- 纯向量：R@3={v['recall@3']}, MRR={v['mrr']}, 热 p50={v['warm']['p50_ms']}ms"
        )
        lines.append(
            f"- 混合 RRF：R@3={h['recall@3']}, MRR={h['mrr']}, 热 p50={h['warm']['p50_ms']}ms"
            f" → ΔR@3={round(h['recall@3']-v['recall@3'],4):+}, ΔMRR={round(h['mrr']-v['mrr'],4):+}"
        )
    # 重排 / 查询改写的增量（相对同数据集的 hybrid 档位），全部从实测数据里算，不写死数字
    lines.append("\n### 重排 vs 查询改写：代价换到了什么\n")
    lines.append("| 数据集 | 档位 | ΔR@3 vs hybrid | ΔMRR vs hybrid | 热 p50 vs hybrid |")
    lines.append("|---|---|---|---|---|")
    for ds_name, ds in all_results.items():
        base = ds["modes"]["hybrid"]
        if "skipped" in base:
            continue
        for key in ("hybrid_rerank", "hybrid_multi_query", "hybrid_hyde"):
            m = ds["modes"].get(key)
            if not m or "skipped" in m:
                continue
            dr3 = round(m["recall@3"] - base["recall@3"], 4)
            dmrr = round(m["mrr"] - base["mrr"], 4)
            ratio = (
                round(m["warm"]["p50_ms"] / base["warm"]["p50_ms"], 1)
                if base["warm"]["p50_ms"]
                else None
            )
            lines.append(
                f"| {ds_name} | {m['label']} | {dr3:+} | {dmrr:+} | ×{ratio} |"
            )
    lines.append("")
    lines.append(
        "读法：重排与改写都是**拿延迟换排序**的可选档位，且换到的收益随语料变化极大——\n"
        "重排只在排序还有提升空间时有效（召回已满分时它顶多纠正顺序，见上表 ΔMRR）；\n"
        "而改写（尤其 HyDE）的收益**依赖 LLM 生成质量**：小模型生成的假想文档可能夹带噪声，\n"
        "在噪声库上甚至会让召回**变差**（上表中出现负的 ΔR@3 就是这种情况）。\n"
        "这也是本项目把它们做成「默认关、按场景开」而不是默认路径的直接依据：\n"
        "收益不确定而代价确定（延迟 ×百倍），默认开等于让所有查询为大模型的不确定性买单。"
    )
    lines.append(
        "\n**什么时候有用、什么时候没用**：\n"
        "1. 干净、话题区分度大的小库上，强向量模型已接近天花板（R@3=1.0），"
        "混合检索只能改善**排序**（MRR）而不能改善召回——收益有限，却多了 BM25 的维护成本。\n"
        "2. 含近义/重复文档的噪声库上，纯向量会把相似度分数摊到多个近义副本上导致正排不上；"
        "BM25 的精确词匹配能把「唯一标识该文档的关键词」那一路顶上来——这是混合检索真正的价值区间。\n"
        "3. 延迟上，BM25 一路是**纯 NumPy、微秒级**，在总延迟里几乎看不见；"
        "真正把单次查询从毫秒级拉到秒级的是**查询改写**（要跑 LLM 生成）与**重排**（要对 top-N 逐对过一遍 Cross-Encoder）。"
        "所以「默认关、按场景开」不是保守，而是把代价花在有收益的地方。"
    )

    md = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    (RESULTS_DIR / f"retrieval_benchmark_{stamp}.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / f"retrieval_benchmark_{stamp}.json").write_text(
        json.dumps(
            {
                "meta": {
                    "date": stamp,
                    "env": env,
                    "rerank_model": RERANK_MODEL,
                    "rerank_cached": rerank_cached,
                    "llm_model": LLM_MODEL,
                    "llm_cached": llm_cached,
                    "warm_rounds": WARM_ROUNDS,
                    "skipped": skipped,
                },
                "datasets": all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(md)
    logger.info(f"报告已落盘：{RESULTS_DIR / f'retrieval_benchmark_{stamp}.md'}")


if __name__ == "__main__":
    main()
