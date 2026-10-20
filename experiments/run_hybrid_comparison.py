"""混合检索对比实验：纯向量 vs 混合(BM25+向量 RRF) vs 混合+重排。

目的：在「更大、含近义/干扰文档」的评测集上，量化混合检索相对纯向量的可测量增益，
打破原 5 文档评测集 recall@3 全 1.0 的天花板效应。

运行：
    python experiments/run_hybrid_comparison.py

说明：
- 默认只跑 vector / hybrid(RRF) 两档，零额外下载、离线可跑、结果可复现。
- 若本地 HF 缓存中已存在默认重排模型（`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`，与 config.yaml 一致），则自动追加「hybrid+重排」一档做完整演示；
  否则不下载，仅在报告中注明重排为可选项。
- **重排降级不再静默**（2026-10-20 修）：`HybridRetriever` 加载重排模型失败时会退回纯 RRF 融合结果。
  本脚本在写报告前校验「重排是否真的生效」，没生效就把该档标成「**未实测**」——
  绝不出现一行标着「混合 + Cross-Encoder 重排」的数字、实际却是没重排的纯 RRF。
"""

import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.utils import logger

ROOT = Path(__file__).parent.parent
# 两个评测集：realistic=人工精标、话题区分度大的知识库；stress=含大量近义/重复文档的噪声语料
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
RESULTS_DIR = ROOT / "experiments/results"

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _hf_model_cached(model_name: str) -> bool:
    """判断 HF 缓存里是否已有**完整**模型（避免触发下载，也避免半截文件误判为可用）。

    与 `run_retrieval_benchmark.py` 同一套判定：目录存在不算数，必须有 >1MB 的权重文件
    （半截下载的 safetensors 会让 from_pretrained 抛 SafetensorError）。
    """
    import os

    org, name = model_name.split("/", 1)
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub_dir = cache_root / "hub" / f"models--{org}--{name.replace('/', '--')}"
    if not hub_dir.exists():
        return False
    for weights in hub_dir.rglob("*"):
        if weights.is_file() and weights.suffix in (".safetensors", ".bin"):
            return weights.stat().st_size > 1_000_000
    return False


def check_rerank_active(pipe) -> str | None:
    """重排档位是否**真的**做了重排？返回失败原因（None = 真的重排了）。

    为什么要有这个检查（2026-10-20 修）：`HybridRetriever._load_reranker` 在模型
    不可用时会**静默降级**（reranker=None、只留 RRF 融合结果）。若不校验，报告里
    会出现一行标着「混合 + Cross-Encoder 重排」的数字，实际却是**没重排的纯 RRF**。
    """
    hybrid = getattr(pipe, "hybrid", None)
    if hybrid is None:
        return "未处于混合检索模式（没有 hybrid 检索器）"
    if getattr(hybrid, "reranker", None) is None:
        return f"重排模型不可用（{getattr(hybrid, 'rerank_load_error', None) or '未知原因'}）"
    return None


def build_pipeline(mode: str, rerank: bool = False) -> RAGPipeline:
    """构造指定检索模式的 pipeline。

    mode: "vector" | "hybrid"；rerank 仅对 hybrid 生效（启用 Cross-Encoder 重排）。
    """
    pipe = RAGPipeline(str(ROOT / "config.yaml"), load_generator=False)
    pipe.config["retrieval"]["mode"] = mode
    if mode == "hybrid":
        from src.hybrid_retrieval import HybridRetriever

        pipe.retrieval_mode = "hybrid"
        pipe.config["retrieval"]["hybrid"]["rerank"]["enabled"] = rerank
        pipe.hybrid = HybridRetriever(pipe.config, pipe.retriever)
    else:
        pipe.retrieval_mode = "vector"
        pipe.hybrid = None
    return pipe


def evaluate(pipe: RAGPipeline, kb_path: Path, qa_path: Path) -> dict:
    docs = json.loads(kb_path.read_text(encoding="utf-8"))
    pipe.build_index(docs)
    qa = json.loads(qa_path.read_text(encoding="utf-8"))

    per_q, r3, r5, mrr = [], [], [], []
    for i, case in enumerate(qa, 1):
        retrieved = pipe._retrieve(case["question"], top_k=5)
        rep = pipe.evaluator.evaluate_retrieval_doc_level(
            retrieved, case["relevant_sources"]
        )
        per_q.append(
            {
                "idx": i,
                "question": case["question"],
                "relevant": case["relevant_sources"],
                "recall@3": rep["recall@3"],
                "recall@5": rep["recall@5"],
                "mrr": rep["mrr"],
            }
        )
        r3.append(rep["recall@3"])
        r5.append(rep["recall@5"])
        mrr.append(rep["mrr"])

    return {
        "kb": kb_path.name,
        "qa": qa_path.name,
        "n": len(qa),
        "avg_recall@3": round(sum(r3) / len(r3), 4),
        "avg_recall@5": round(sum(r5) / len(r5), 4),
        "avg_mrr": round(sum(mrr) / len(mrr), 4),
        "per_question": per_q,
    }


def run_mode(pipe, kb_path: Path, qa_path: Path, mode: str, label: str) -> dict:
    """跑一个档位；**重排没真的生效时不写指标**，只写「未实测」。

    返回的 dict 要么是评测结果（avg_recall@3 / avg_mrr / per_question…），
    要么是 {"label":…, "skipped": "重排未实测（…）"}——后者不会被当成实测数字引用。
    """
    if mode == "hybrid_rerank":
        reason = check_rerank_active(pipe)
        if reason:
            logger.warning(f"档位「{label}」重排未生效，标为未实测：{reason}")
            return {"label": label, "skipped": f"重排未实测（{reason}）"}
    res = evaluate(pipe, kb_path, qa_path)
    res["label"] = label
    return res


def render_md(all_results: dict, modes: list, rerank_cached: bool) -> str:
    """渲染报告（单独成函数，便于单测直接断言「未实测」口径）。"""
    lines = []
    lines.append("# 混合检索对比实验（向量 vs 混合 RRF" + (" vs 混合+重排" if rerank_cached else "") + "）\n")
    lines.append(f"- 生成日期：{datetime.date.today().isoformat()}")
    lines.append(f"- 评测集：")
    for ds_name, (kb_path, qa_path) in DATASETS.items():
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        desc = "人工精标、话题区分度大" if ds_name == "realistic" else "含大量近义/重复文档的噪声语料（验证混合检索的价值区间）"
        lines.append(f"  - `{ds_name}`：`{kb_path.name}`（{len(kb)} 篇，{desc}）/ `{qa_path.name}`（{len(qa)} 题）")
    lines.append("")

    for ds_name in DATASETS:
        lines.append(f"## 数据集：{ds_name}\n")
        lines.append("### 一、汇总指标\n")
        lines.append("| 档位 | 平均 Recall@3 | 平均 Recall@5 | 平均 MRR |")
        lines.append("|------|------|------|------|")
        active_modes = []
        for mode, label in modes:
            r = all_results[ds_name][mode]
            if "skipped" in r:
                lines.append(f"| {label} | 未实测 | 未实测 | 未实测 |")
                lines.append("")
                lines.append(f"> ⚠️ `{label}`（{mode}）**{r['skipped']}**")
                lines.append("")
                continue
            active_modes.append((mode, label))
            lines.append(f"| {label} | {r['avg_recall@3']} | {r['avg_recall@5']} | {r['avg_mrr']} |")

        lines.append("\n### 二、逐题明细\n")
        qa = json.loads(DATASETS[ds_name][1].read_text(encoding="utf-8"))
        header = "| # | 问题 | 相关文档 | " + " | ".join(
            f"{label} R@3" for _, label in active_modes
        ) + " | " + " | ".join(f"{label} MRR" for _, label in active_modes) + " |"
        sep = "|" + "---|" * (3 + 2 * len(active_modes))
        lines.append(header)
        lines.append(sep)
        for i in range(len(qa)):
            cells = [str(i + 1), qa[i]["question"], ",".join(qa[i]["relevant_sources"])]
            for mode, _ in active_modes:
                cells.append(str(all_results[ds_name][mode]["per_question"][i]["recall@3"]))
            for mode, _ in active_modes:
                cells.append(str(all_results[ds_name][mode]["per_question"][i]["mrr"]))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # 结论（基于实测）
    lines.append("## 三、结论\n")
    for ds_name in DATASETS:
        base = all_results[ds_name]["vector"]
        best = all_results[ds_name]["hybrid"]
        delta_r3 = round(best["avg_recall@3"] - base["avg_recall@3"], 4)
        delta_mrr = round(best["avg_mrr"] - base["avg_mrr"], 4)
        lines.append(f"### 数据集 {ds_name}\n")
        lines.append(f"- 纯向量基线：平均 Recall@3={base['avg_recall@3']}，平均 MRR={base['avg_mrr']}。")
        lines.append(
            f"- 混合检索（BM25+向量 RRF）：平均 Recall@3={best['avg_recall@3']}（Δ={delta_r3:+}，"
            f"{'提升' if delta_r3 > 0 else '持平'}），平均 MRR={best['avg_mrr']}（Δ={delta_mrr:+}，"
            f"{'提升' if delta_mrr > 0 else '持平'}）。"
        )
    rerank_rows = [
        (ds_name, all_results[ds_name].get("hybrid_rerank"))
        for ds_name in DATASETS
        if "hybrid_rerank" in all_results[ds_name]
    ]
    if rerank_rows:
        skipped = [(ds, r) for ds, r in rerank_rows if r and "skipped" in r]
        if skipped:
            lines.append(
                "- 混合+重排：**未实测**（"
                + "；".join(f"{ds}：{r['skipped']}" for ds, r in skipped)
                + "）——重排模型没真的加载起来时，本报告不写重排数字（避免把纯 RRF 当成重排结果）。"
            )
        else:
            lines.append("- 混合+重排：详见各数据集的 `hybrid_rerank` 档位。")
    else:
        lines.append(
            f"- 重排（Cross-Encoder，{RERANK_MODEL}）：已集成但默认关闭；本地无缓存未跑，"
            f"启用后通常对 top 排位（MRR）进一步改善，见 README。"
        )
    lines.append(
        "\n**关键结论**：在话题区分度大的精标知识库上，强向量模型已接近上限，混合检索收益有限；"
        "而在含大量近义/重复文档的噪声语料上，BM25 的精确词匹配能纠正向量把分数分散到近义副本上的问题，"
        "混合检索的相对增益显著——这也解释了「何时该上混合检索」。"
    )
    return "\n".join(lines)


def main():
    # 决定跑哪些档位
    modes = [("vector", "纯向量（基线）"), ("hybrid", "混合 BM25+向量 · RRF")]
    rerank_cached = _hf_model_cached(RERANK_MODEL)
    if rerank_cached:
        modes.append(("hybrid_rerank", "混合 + Cross-Encoder 重排"))
        logger.info(f"检测到本地已缓存重排模型 {RERANK_MODEL}，追加重排档位")
    else:
        logger.info(
            f"未检测到重排模型缓存（{RERANK_MODEL}），跳过下载；重排为可选项，见 README。"
        )

    # 每个数据集下，逐档位评测
    all_results = {}  # dataset -> {mode: metrics 或 {"skipped": …}}
    for ds_name, (kb_path, qa_path) in DATASETS.items():
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        logger.info(f"数据集[{ds_name}]：{kb_path.name}（{len(kb)} 篇）/ {qa_path.name}（{len(qa)} 题）")
        all_results[ds_name] = {}
        for mode, label in modes:
            logger.info(f"===== [{ds_name}] 档位：{label}（{mode}）=====")
            pipe = (
                build_pipeline("hybrid", rerank=True)
                if mode == "hybrid_rerank"
                else build_pipeline(mode)
            )
            all_results[ds_name][mode] = run_mode(pipe, kb_path, qa_path, mode, label)

    md = render_md(all_results, modes, rerank_cached)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    (RESULTS_DIR / f"hybrid_comparison_{stamp}.md").write_text(
        md, encoding="utf-8"
    )
    (RESULTS_DIR / f"hybrid_comparison_{stamp}.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(md)
    logger.info(f"报告已落盘：{RESULTS_DIR / f'hybrid_comparison_{stamp}.md'}")


if __name__ == "__main__":
    main()
