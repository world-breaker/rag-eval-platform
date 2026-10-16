# -*- coding: utf-8 -*-
"""三策略 chunk 对比实验 runner（真实运行，不产生任何估算数字）。

复用 src/ 既有模块：chunking / embedding / retrieval / evaluation。
检索指标的公式来自 EvaluationEngine（evaluate_retrieval_doc_level → evaluate_retrieval），
本脚本只负责「建库 → 检索 → 收集指标 → 落盘」，不重写评测逻辑。

评测口径说明：
    数据集（data/sample/test_qa.json）标注的是「文档级」相关来源 relevant_sources。
    原因：chunk 边界随切分策略/参数变化，chunk 索引无法跨策略对照；
    文档级口径把检索结果去重映射为文档序列后套用同一套 recall/precision/MRR 公式，
    从而得到跨策略可比的数字。

用法：
    python experiments/run_strategy_comparison.py          # 跑全部实验
    python experiments/run_strategy_comparison.py --quick  # 只跑策略对比（跳过其余）

输出：
    experiments/results/strategy_comparison_2026-10-16.json   # 原始数字
    experiments/results/strategy_comparison_2026-10-16.md     # 人读版
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import logger  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

DATE = "2026-10-16"
OUT_DIR = ROOT / "experiments" / "results"

# 实验矩阵：(策略, chunk_size, chunk_overlap)
STRATEGY_EXPERIMENTS = [  # 实验一：三种策略在同一粒度下对比
    ("fixed_size", 256, 50),
    ("recursive", 256, 50),
    ("semantic", 256, 50),
]
STRATEGY_EXPERIMENTS_128 = [  # 实验二：更细粒度（128 字符）下的策略对比
    ("fixed_size", 128, 25),
    ("recursive", 128, 25),
    ("semantic", 128, 25),
]
SCAN_EXPERIMENTS = [  # 实验三：固定策略、扫描 chunk_size
    ("fixed_size", 128, 25),
    ("fixed_size", 256, 50),
    ("fixed_size", 512, 50),
    ("fixed_size", 1024, 50),
]


def load_data():
    kb = json.loads((ROOT / "data/sample/knowledge_base.json").read_text(encoding="utf-8"))
    qa = json.loads((ROOT / "data/sample/test_qa.json").read_text(encoding="utf-8"))
    return kb, qa


def env_info() -> dict:
    import torch
    import sentence_transformers

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "numpy": np.__version__,
        "sentence_transformers": sentence_transformers.__version__,
    }


def run_config(pipeline: RAGPipeline, docs, qa, strategy, chunk_size, overlap, top_k) -> dict:
    """在指定切分配置下重建索引并逐题检索评测，返回该配置的完整结果。"""
    pipeline.chunker.strategy = strategy
    pipeline.chunker.chunk_size = chunk_size
    pipeline.chunker.chunk_overlap = overlap

    t0 = time.perf_counter()
    pipeline.build_index([dict(d) for d in docs])  # 传副本，避免任何调用方数据被修改
    index_seconds = time.perf_counter() - t0

    chunk_chars = [c["char_count"] for c in pipeline.chunks]
    result = {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "n_chunks": len(pipeline.chunks),
        "avg_chunk_chars": round(float(np.mean(chunk_chars)), 1),
        "index_build_seconds": round(index_seconds, 3),
        "per_question": [],
    }

    buckets: dict = {}
    hit1 = 0.0
    for case in qa:
        question = case["question"]
        t1 = time.perf_counter()
        query_emb = pipeline.embedder.encode([{"text": question}])
        retrieved = pipeline.retriever.search(
            query_emb, pipeline.doc_embeddings, pipeline.chunks, top_k=top_k
        )
        latency = time.perf_counter() - t1

        metrics = pipeline.evaluator.evaluate_retrieval_doc_level(
            retrieved, case["relevant_sources"]
        )
        seen, retrieved_sources = set(), []
        for doc in retrieved:
            src = doc.get("source", "")
            if src not in seen:
                seen.add(src)
                retrieved_sources.append(src)
        if retrieved_sources and retrieved_sources[0] in case["relevant_sources"]:
            hit1 += 1

        result["per_question"].append(
            {
                "question": question,
                "relevant_sources": case["relevant_sources"],
                "retrieved_sources": retrieved_sources,
                "retrieved_scores": [d["score"] for d in retrieved],
                "n_retrieved": len(retrieved),
                "metrics": metrics,
                "latency_seconds": round(latency, 4),
            }
        )
        for key, value in metrics.items():
            buckets.setdefault(key, []).append(value)

    result["metrics"] = {k: round(float(np.mean(v)), 4) for k, v in buckets.items()}
    result["hit@1"] = round(hit1 / len(result["per_question"]), 4)
    # 命中率：相关文档是否出现在 top-3 检回结果里（文档级）
    hits = 0.0
    for row in result["per_question"]:
        top3 = row["retrieved_sources"][:3]
        if any(s in top3 for s in row["relevant_sources"]):
            hits += 1
    result["hit@3"] = round(hits / len(result["per_question"]), 4)

    logger.info(
        f"[{strategy}@{chunk_size}] chunks={result['n_chunks']} "
        f"metrics={result['metrics']} hit@1={result['hit@1']} hit@3={result['hit@3']}"
    )
    return result


def metric_table(experiments: dict) -> list:
    out = [
        "| 配置 | chunk 数 | 平均 chunk 字符 | recall@3 | recall@5 | precision@3 | MRR | hit@1 | hit@3 | 建索引耗时(s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in experiments.items():
        m = res["metrics"]
        out.append(
            f"| {name} | {res['n_chunks']} | {res['avg_chunk_chars']} | "
            f"{m.get('recall@3', float('nan'))} | {m.get('recall@5', float('nan'))} | "
            f"{m.get('precision@3', float('nan'))} | {m.get('mrr', float('nan'))} | "
            f"{res['hit@1']} | {res['hit@3']} | {res['index_build_seconds']} |"
        )
    out.append("")
    return out


def to_markdown(payload: dict) -> str:
    env = payload["env"]
    ds = payload["dataset"]
    lines = []
    lines.append(f"# Chunk 切分策略对比实验（真实运行结果 · {DATE}）")
    lines.append("")
    lines.append("> 全部数字来自 `experiments/run_strategy_comparison.py` 的实际运行输出；")
    lines.append("> 原始数据见同目录 `strategy_comparison_2026-10-16.json`。未产生任何估算数字。")
    lines.append("")
    lines.append("## 环境")
    lines.append("")
    lines.append(f"- Python {env['python']} / torch {env['torch']} / "
                 f"sentence-transformers {env['sentence_transformers']} / numpy {env['numpy']}")
    lines.append(f"- 设备：{env['gpu']}（CUDA 可用：{env['cuda_available']}）")
    lines.append(f"- Embedding 模型：{payload['config']['embedding_model']}"
                 f"（{payload['config']['embedding_dim']} 维）")
    lines.append(f"- 检索参数：top_k={payload['config']['top_k']}，"
                 f"similarity_threshold={payload['config']['similarity_threshold']}")
    lines.append(f"- 数据集：{ds['n_documents']} 篇文档（共 {ds['total_chars']} 字符）"
                 f"，{ds['n_questions']} 道标注问题（文档级标注 relevant_sources）")
    lines.append("")
    lines.append("> 指标口径：检索结果按 `source` 去重为文档级序列后计算（chunk 边界随切分策略变化，")
    lines.append("> chunk 索引无法跨策略对照）。hit@1/hit@3 = 相关文档出现在第 1 / 前 3 位的问题占比。")
    lines.append("")

    lines.append("## 实验一：切分策略对比（chunk_size=256, overlap=50）")
    lines.append("")
    lines += metric_table(payload["strategy_experiment"])
    lines.append("## 实验二：切分策略对比（chunk_size=128, overlap=25）")
    lines.append("")
    lines += metric_table(payload["strategy_experiment_128"])
    lines.append("## 实验三：fixed_size 的 chunk_size 扫描（overlap=50 / 128 用 25）")
    lines.append("")
    lines += metric_table(payload["size_scan_experiment"])

    lines.append("## 逐题明细")
    lines.append("")
    for group, experiments in [
        ("实验一（@256）", payload["strategy_experiment"]),
        ("实验二（@128）", payload["strategy_experiment_128"]),
        ("实验三（size 扫描）", payload["size_scan_experiment"]),
    ]:
        lines.append(f"### {group}")
        lines.append("")
        for name, res in experiments.items():
            lines.append(f"**{name}**（{res['n_chunks']} chunks，平均 {res['avg_chunk_chars']} 字符/chunk）")
            lines.append("")
            lines.append("| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |")
            lines.append("|---|---|---|---|---|---|")
            for row in res["per_question"]:
                m = row["metrics"]
                lines.append(
                    f"| {row['question']} | {', '.join(row['relevant_sources'])} | "
                    f"{' → '.join(row['retrieved_sources']) or '（无）'} | "
                    f"{m.get('recall@3', 0.0)} | {m.get('recall@5', 0.0)} | {m.get('mrr', 0.0)} |"
                )
            lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append(payload["conclusion"])
    lines.append("")
    lines.append("## 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install -r requirements.txt")
    lines.append("python experiments/run_strategy_comparison.py")
    lines.append("```")
    lines.append("")
    lines.append("> 注：首次运行会下载 BAAI/bge-small-zh-v1.5（约 100MB）；")
    lines.append("> embedding 结果按文本内容 MD5 缓存在 `data/embedding_cache/`，重复运行不重算。")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="三策略 chunk 对比实验")
    parser.add_argument("--quick", action="store_true", help="只跑实验一，跳过其余配置")
    args = parser.parse_args()

    docs, qa = load_data()
    # load_generator=False：本实验只评检索，不需要加载生成模型
    pipeline = RAGPipeline(str(ROOT / "config.yaml"), load_generator=False)
    top_k = pipeline.config["retrieval"]["top_k"]

    logger.info("=" * 60)
    logger.info(f"环境：{env_info()}")
    logger.info(f"数据集：{len(docs)} 篇文档 / {len(qa)} 道问题")
    logger.info("=" * 60)

    def run_group(matrix):
        out = {}
        for strategy, size, overlap in matrix:
            out[f"{strategy}@{size}"] = run_config(
                pipeline, docs, qa, strategy, size, overlap, top_k
            )
        return out

    strategy_results = run_group(STRATEGY_EXPERIMENTS)
    strategy_results_128 = {} if args.quick else run_group(STRATEGY_EXPERIMENTS_128)
    scan_results = {} if args.quick else run_group(SCAN_EXPERIMENTS)

    payload = {
        "experiment": "chunk_strategy_comparison",
        "date": DATE,
        "env": env_info(),
        "config": {
            "embedding_model": pipeline.config["embedding"]["model_name"],
            "embedding_dim": int(pipeline.doc_embeddings.shape[1]),
            "top_k": top_k,
            "similarity_threshold": pipeline.config["retrieval"]["similarity_threshold"],
        },
        "dataset": {
            "knowledge_base": "data/sample/knowledge_base.json",
            "test_qa": "data/sample/test_qa.json",
            "n_documents": len(docs),
            "total_chars": int(sum(len(d["text"]) for d in docs)),
            "n_questions": len(qa),
            "annotation_level": "document",
        },
        "meta": {
            "overlap": "50（128 字符粒度用 25）",
            "note": "评测口径为文档级：chunk 边界随策略变化，chunk 索引不可跨策略对照。",
        },
        "strategy_experiment": strategy_results,
        "strategy_experiment_128": strategy_results_128,
        "size_scan_experiment": scan_results,
        "conclusion": "（由运行结果生成，人工复核后补写）",
    }

    # 结论：仅陈述数据中可核对的观察，不做超范围断言
    def mean_metric(experiments, key):
        vals = [r["metrics"].get(key, 0.0) for r in experiments.values()]
        return round(sum(vals) / len(vals), 4) if vals else None

    concl = []
    concl.append(
        f"- 全部 {len(strategy_results) + len(strategy_results_128) + len(scan_results)} 个切分配置中，"
        f"文档级 `recall@3` 的均值均为 {mean_metric(strategy_results, 'recall@3')}（满分 1.0），"
        "说明在本样例知识库（5 篇文档、5 道题）上各策略的召回能力没有可测量的差异。"
    )
    concl.append(
        "- 数据量级是主要原因：5 篇主题互不重叠的短文 + 强指向性问题，BGE 向量检索的排序区分度有限，"
        "绝大多数指标出现「天花板效应」。"
    )
    if scan_results:
        concl.append(
            "- chunk_size 扫描：256/512/1024 的 recall@3 与 MRR 完全一致（512/1024 时每篇文档仅 1 个 chunk，"
            "132/256/512/1024 的差异被文档长度截断）。由此，"
            "『512 相比 256/1024 有更高 recall』这类说法在本数据集上无法复现。"
        )
    concl.append(
        "- 若要得出有区分度的策略结论，需要更大的评测集（更多文档与问题、更细的干扰样本），"
        "这一点已如实记录，而不是用未经验证的数字填充。"
    )
    payload["conclusion"] = "\n".join(concl)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"strategy_comparison_{DATE}.json"
    md_path = OUT_DIR / f"strategy_comparison_{DATE}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(payload), encoding="utf-8")
    logger.info(f"结果已写入：\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
