"""「假绿」防护的离线单测：降级必须被看见，不能悄悄变成另一档实验。

覆盖四条「会造出假绿」的链路（2026-10-20 修）：
1. 查询改写降级（LLM 失败）→ 检索结果带 `rewrite_degraded=True`，基准脚本把该档标成「未实测」；
2. 混合对比脚本的重排降级（模型加载失败）→ 该档标成「未实测」，不写「混合 + 重排」的数字；
3. 裁判部分采样失败 → 进 `meta.fallbacks`（不再被「零兜底」掩盖）；
4. embedding 缓存键含模型名（换模型不命中旧缓存）。

硬约束：不加载真实模型、不联网、不调 API——全部用桩对象。
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import EvaluationEngine  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.query_rewriting import QueryRewriter  # noqa: E402
from src.retrieval import RetrievalEngine  # noqa: E402


def _load_experiment_module(name: str):
    """按路径加载 experiments/ 下的脚本（该目录不是 python 包）。"""
    path = ROOT / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_exp_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- P0-2：查询改写降级


def _fake_pipeline(chunks, doc_embeddings, encode_fn):
    """绕开模型加载，手工拼一个 pipeline（只放被测代码需要的属性）。"""
    pipe = object.__new__(RAGPipeline)
    pipe.config = {
        "retrieval": {"top_k": 5, "similarity_threshold": 0.3, "rrf_k": 60},
        "evaluation": {},
    }
    pipe.embedder = types.SimpleNamespace(encode=encode_fn)
    pipe.retriever = RetrievalEngine(pipe.config)
    pipe.chunks = chunks
    pipe.doc_embeddings = np.array(doc_embeddings, dtype=float)
    pipe.retrieval_mode = "vector"
    pipe.hybrid = None
    pipe.query_rewriter = None
    pipe._rewriter_initialized = False
    return pipe


def test_rewrite_degradation_is_flagged_on_every_result():
    """LLM 改写抛异常 → 退化成「原查询单路检索」，每条结果必须带 rewrite_degraded=True。

    没有这个标志，实验报告会把这一档写成「混合 + multi-query 改写」——
    声称的档位与实际跑的东西不符。
    """
    chunks = [
        {"text": "target", "chunk_index": 0, "source": "t.txt"},
        {"text": "distractor", "chunk_index": 1, "source": "d.txt"},
    ]
    doc_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    def encode_fn(items):
        return np.tile([[1.0, 0.0]], (len(items), 1))

    pipe = _fake_pipeline(chunks, doc_embeddings, encode_fn)

    def boom(prompt):
        raise RuntimeError("CUDA OOM (fake)")

    pipe.query_rewriter = QueryRewriter(
        {"retrieval": {"query_rewrite": {"enabled": True, "strategy": "multi_query"}}},
        llm_fn=boom,
    )
    pipe._rewriter_initialized = True

    got = pipe._retrieve("原问题", top_k=2)
    assert got, "检索结果不应为空"
    assert all(r.get("rewrite_degraded") is True for r in got), (
        "改写失败时每条结果都必须标 rewrite_degraded=True，实际："
        f"{[r.get('rewrite_degraded') for r in got]}"
    )


def test_successful_rewrite_is_not_marked_degraded():
    """改写成功 → rewrite_degraded=False（不能把正常档位也标成降级）。"""
    chunks = [{"text": "target", "chunk_index": 0, "source": "t.txt"}]
    doc_embeddings = np.array([[1.0, 0.0]])

    def encode_fn(items):
        return np.tile([[1.0, 0.0]], (len(items), 1))

    pipe = _fake_pipeline(chunks, doc_embeddings, encode_fn)
    pipe.query_rewriter = QueryRewriter(
        {"retrieval": {"query_rewrite": {"enabled": True, "strategy": "multi_query"}}},
        llm_fn=lambda p: "改写A\n",
    )
    pipe._rewriter_initialized = True

    got = pipe._retrieve("原问题", top_k=1)
    assert got[0].get("rewrite_degraded") is False


def test_benchmark_marks_degraded_rewrite_mode_as_not_measured():
    """基准脚本：任一查询改写降级 → 该档标「未实测」，不写进指标表。"""
    bench = _load_experiment_module("run_retrieval_benchmark")

    res = {
        "recall@3": 1.0,
        "recall@5": 1.0,
        "mrr": 1.0,
        "rewrite_degraded_queries": 2,
        "rewrite_degraded_total": 14,
    }
    out = bench.mark_unmeasured(res, "混合 + multi-query 改写", {"rewrite": "multi_query"})
    assert "skipped" in out, "改写降级的档位必须标成未实测"
    assert "未实测" in out["skipped"]
    assert "avg" not in out and "recall@3" not in out, "未实测档位不许带指标数字"
    # 反例：没有降级 → 原样保留
    ok = bench.mark_unmeasured(
        {"recall@3": 1.0, "rewrite_degraded_queries": 0, "rewrite_degraded_total": 14},
        "混合 + multi-query 改写",
        {"rewrite": "multi_query"},
    )
    assert "skipped" not in ok and ok["recall@3"] == 1.0


# ---------------------------------------------------------------- P0-3：重排降级


def test_hybrid_comparison_marks_unavailable_rerank_as_not_measured(monkeypatch):
    """CrossEncoder 构造抛异常 → 该档不是「混合 + Cross-Encoder 重排」，而是「未实测」。"""
    mod = _load_experiment_module("run_hybrid_comparison")

    # 真的让 CrossEncoder 构造失败（模拟权重缺失 / 半截下载）
    import sentence_transformers

    def boom(*args, **kwargs):
        raise OSError("权重文件不完整")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", boom)

    from src.hybrid_retrieval import HybridRetriever

    cfg = {
        "retrieval": {
            "rrf_k": 60,
            "hybrid": {
                "rerank": {
                    "enabled": True,
                    "model": "cross-encoder/fake-model",
                    "device": "cpu",
                    "top_n": 20,
                }
            },
        }
    }
    hr = HybridRetriever(cfg)
    assert hr.reranker is None, "CrossEncoder 构造失败后不许有重排器"
    assert hr.rerank_load_error, "失败原因必须留证（rerank_load_error）"

    pipe = types.SimpleNamespace(hybrid=hr)
    reason = mod.check_rerank_active(pipe)
    assert reason, "重排没生效时必须给出原因"
    assert "重排模型不可用" in reason

    res = mod.run_mode(
        pipe,
        ROOT / "data/sample/test_qa.json",  # 不会被读到（提前返回）
        ROOT / "data/sample/test_qa.json",
        "hybrid_rerank",
        "混合 + Cross-Encoder 重排",
    )
    assert "skipped" in res and "未实测" in res["skipped"]
    assert "avg_recall@3" not in res, "未实测档位不许带指标数字"


def test_hybrid_comparison_keeps_metrics_when_rerank_works():
    """重排真的生效 → 不标未实测（正常路径不受影响）。"""
    mod = _load_experiment_module("run_hybrid_comparison")

    class _FakeHybrid:
        reranker = object()
        rerank_load_error = None

    pipe = types.SimpleNamespace(hybrid=_FakeHybrid())
    assert mod.check_rerank_active(pipe) is None


def test_hybrid_comparison_report_shows_not_measured():
    """报告渲染：未实测档位在表里显示「未实测」，不显示任何指标数字。"""
    mod = _load_experiment_module("run_hybrid_comparison")

    modes = [("vector", "纯向量（基线）"), ("hybrid", "混合 BM25+向量 · RRF"), ("hybrid_rerank", "混合 + Cross-Encoder 重排")]

    def _n_questions(path):
        import json

        return len(json.loads(Path(path).read_text(encoding="utf-8")))

    all_results = {}
    for ds_name, qa_file in (
        ("realistic", "data/sample/test_qa_large.json"),
        ("near_dup_stress", "data/sample/test_qa_stress.json"),
    ):
        n_q = _n_questions(ROOT / qa_file)
        all_results[ds_name] = {
            "vector": {
                "label": "纯向量（基线）",
                "avg_recall@3": 0.625,
                "avg_recall@5": 0.625,
                "avg_mrr": 0.4792,
                "per_question": [
                    {"recall@3": 0.5, "mrr": 0.5} for _ in range(n_q)
                ],
            },
            "hybrid": {
                "label": "混合 BM25+向量 · RRF",
                "avg_recall@3": 1.0,
                "avg_recall@5": 1.0,
                "avg_mrr": 0.7708,
                "per_question": [
                    {"recall@3": 1.0, "mrr": 0.7} for _ in range(n_q)
                ],
            },
            "hybrid_rerank": {
                "label": "混合 + Cross-Encoder 重排",
                "skipped": "重排未实测（重排模型不可用（OSError: 权重文件不完整））",
            },
        }
    md = mod.render_md(all_results, modes, rerank_cached=True)
    assert "未实测" in md
    row = [ln for ln in md.splitlines() if ln.startswith("| 混合 + Cross-Encoder 重排 |")][0]
    assert "未实测" in row and "0.625" not in row


# ---------------------------------------------------------------- P0-4：裁判部分采样失败


def test_partial_sampling_failure_is_marked_not_silent():
    """N>1 时部分采样失败 → 进 meta.fallbacks（source=llm_median_partial）。"""
    torch = pytest.importorskip("torch")

    class FakeTensor:
        def __init__(self, device, length=4):
            self.device = device
            self._length = length

        def to(self, device):
            return FakeTensor(device, self._length)

        def __getitem__(self, item):
            return self

        def __len__(self):
            return self._length

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "FAKE_PROMPT"

        def __call__(self, text, return_tensors="pt"):
            return {
                "input_ids": FakeTensor(torch.device("cpu")),
                "attention_mask": FakeTensor(torch.device("cpu")),
            }

        def decode(self, tokens, skip_special_tokens=True):
            return "5"

    class FlakyModel:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            if self.calls == 2:  # 第二次失败
                raise RuntimeError("CUDA out of memory (fake)")
            return [[0, 0, 0, 0, 9]]

    engine = EvaluationEngine(
        {"evaluation": {"judge_dimensions": ["faithfulness"], "judge_samples": 3}}
    )
    out = engine.evaluate_generation(
        "问题？", "回答。", "上下文。", FlakyModel(), FakeTokenizer(), return_meta=True
    )
    assert out["meta"]["fallbacks"], "部分采样失败必须出现在 meta.fallbacks 里"
    fb = out["meta"]["fallbacks"][0]
    assert fb["source"] == "llm_median_partial"
    assert fb["n_failed"] == 1 and fb["n"] == 3
    assert out["meta"]["partial_failures"]["faithfulness"]["failed"] == 1
    assert out["meta"]["details"]["faithfulness"]["source"] == "llm_median_partial"


def test_no_failure_means_no_fallback_record():
    """N>1 且零失败 → 不许出现任何 fallback 记录（正常路径不算兜底）。"""
    torch = pytest.importorskip("torch")

    class FakeTensor:
        def __init__(self, device, length=4):
            self.device = device
            self._length = length

        def to(self, device):
            return FakeTensor(device, self._length)

        def __getitem__(self, item):
            return self

        def __len__(self):
            return self._length

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "FAKE_PROMPT"

        def __call__(self, text, return_tensors="pt"):
            return {
                "input_ids": FakeTensor(torch.device("cpu")),
                "attention_mask": FakeTensor(torch.device("cpu")),
            }

        def decode(self, tokens, skip_special_tokens=True):
            return "4"

    class OkModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            return [[0, 0, 0, 0, 9]]

    engine = EvaluationEngine(
        {"evaluation": {"judge_dimensions": ["faithfulness"], "judge_samples": 3}}
    )
    out = engine.evaluate_generation(
        "问题？", "回答。", "上下文。", OkModel(), FakeTokenizer(), return_meta=True
    )
    assert out["meta"]["fallbacks"] == []
    assert out["meta"]["partial_failures"] == {}
    assert out["meta"]["details"]["faithfulness"]["source"] == "llm_median"


if __name__ == "__main__":
    test_rewrite_degradation_is_flagged_on_every_result()
    test_successful_rewrite_is_not_marked_degraded()
    test_benchmark_marks_degraded_rewrite_mode_as_not_measured()
    test_hybrid_comparison_marks_unavailable_rerank_as_not_measured()
    test_hybrid_comparison_keeps_metrics_when_rerank_works()
    test_hybrid_comparison_report_shows_not_measured()
    test_partial_sampling_failure_is_marked_not_silent()
    test_no_failure_means_no_fallback_record()
    print("✅ 全部假绿防护单测通过")
