"""查询改写（HyDE / multi-query）离线单测：用 stub LLM，确定性、无需模型下载。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query_rewriting import QueryRewriter, _clean_lines


def _cfg(enabled=True, strategy="multi_query", num_queries=3, **kw) -> dict:
    qr = {"enabled": enabled, "strategy": strategy, "num_queries": num_queries}
    qr.update(kw)
    return {"retrieval": {"query_rewrite": qr}}


def test_clean_lines_strips_numbering_and_labels():
    raw = "1. 量化是什么\n2、NF4 量化格式\n- 正态分布权重\n\n   查询：LoRA 低秩分解\n"
    assert _clean_lines(raw) == [
        "量化是什么",
        "NF4 量化格式",
        "正态分布权重",
        "LoRA 低秩分解",
    ]
    print("✅ clean_lines_strips_numbering_and_labels")


def test_disabled_is_passthrough_and_never_calls_llm():
    calls = []

    def stub(prompt):
        calls.append(prompt)
        return "不应该被调用"

    rw = QueryRewriter(_cfg(enabled=False), llm_fn=stub)
    out = rw.rewrite("NF4 量化为什么友好？")
    assert out["queries"] == ["NF4 量化为什么友好？"]
    assert out["hypothetical_doc"] is None
    assert out["strategy"] == "passthrough"
    assert calls == [], "默认关闭时不应调用 LLM（零成本、零延迟）"
    print("✅ disabled_is_passthrough_and_never_calls_llm")


def test_multi_query_keeps_original_first_and_parses_variants():
    def stub(prompt):
        assert "3 个不同角度" in prompt  # 占位符被正确填充
        return "1. NF4 量化格式的数值分布\n2. 4bit 正态分布量化原理\n3. 权重量化 分位数量化\n"

    rw = QueryRewriter(_cfg(strategy="multi_query", num_queries=3), llm_fn=stub)
    out = rw.rewrite("NF4 量化格式为什么对正态分布友好？")
    assert out["queries"][0] == "NF4 量化格式为什么对正态分布友好？", "原查询必须保留在第一位"
    assert len(out["queries"]) == 4
    assert out["queries"][1] == "NF4 量化格式的数值分布"
    assert out["hypothetical_doc"] is None
    assert out["degraded"] is False
    print("✅ multi_query_keeps_original_first_and_parses_variants")


def test_multi_query_dedupes_and_limits_to_num_queries():
    def stub(prompt):
        # 第 2 行与原查询重复，第 3/4 行内容重复 → 去重后只应补 2 条
        return "原问题\n原问题\n变体A\n变体A\n变体B\n变体C\n"

    rw = QueryRewriter(_cfg(num_queries=2), llm_fn=stub)
    out = rw.rewrite("原问题")
    assert out["queries"] == ["原问题", "变体A", "变体B"], out["queries"]
    print("✅ multi_query_dedupes_and_limits_to_num_queries")


def test_hyde_returns_hypothetical_document():
    def stub(prompt):
        assert "技术说明" in prompt
        return "NF4 是一种 4 位浮点量化格式，其分位数量化点针对正态分布的权重取值集中分布……"

    rw = QueryRewriter(_cfg(strategy="hyde"), llm_fn=stub)
    out = rw.rewrite("NF4 为什么对正态分布友好？")
    assert out["queries"] == ["NF4 为什么对正态分布友好？"], "HyDE 不额外增加查询"
    assert out["hypothetical_doc"] and "NF4" in out["hypothetical_doc"]
    print("✅ hyde_returns_hypothetical_document")


def test_both_strategy_produces_queries_and_doc():
    def stub(prompt):
        if "改写查询" in prompt:
            return "1. 权重量化的数值分布\n"
        return "假想的技术说明段落……"

    rw = QueryRewriter(_cfg(strategy="both"), llm_fn=stub)
    out = rw.rewrite("Q")
    assert out["queries"] == ["Q", "权重量化的数值分布"]
    assert out["hypothetical_doc"] == "假想的技术说明段落……"
    print("✅ both_strategy_produces_queries_and_doc")


def test_llm_exception_degrades_gracefully():
    def boom(prompt):
        raise RuntimeError("CUDA OOM")

    rw = QueryRewriter(_cfg(strategy="both"), llm_fn=boom)
    out = rw.rewrite("原始问题")
    assert out["queries"] == ["原始问题"], "LLM 失败必须退化为原查询，绝不能让检索不可用"
    assert out["hypothetical_doc"] is None
    assert out["degraded"] is True
    print("✅ llm_exception_degrades_gracefully")


def test_missing_llm_is_passthrough_even_when_enabled():
    rw = QueryRewriter(_cfg(enabled=True), llm_fn=None)
    assert rw.active is False
    out = rw.rewrite("原始问题")
    assert out["queries"] == ["原始问题"] and out["hypothetical_doc"] is None
    print("✅ missing_llm_is_passthrough_even_when_enabled")


def test_hyde_truncates_long_output():
    def stub(prompt):
        return "长" * 5000

    rw = QueryRewriter(_cfg(strategy="hyde", max_hypothetical_chars=128), llm_fn=stub)
    out = rw.rewrite("Q")
    assert len(out["hypothetical_doc"]) == 128
    print("✅ hyde_truncates_long_output")


if __name__ == "__main__":
    test_clean_lines_strips_numbering_and_labels()
    test_disabled_is_passthrough_and_never_calls_llm()
    test_multi_query_keeps_original_first_and_parses_variants()
    test_multi_query_dedupes_and_limits_to_num_queries()
    test_hyde_returns_hypothetical_document()
    test_both_strategy_produces_queries_and_doc()
    test_llm_exception_degrades_gracefully()
    test_missing_llm_is_passthrough_even_when_enabled()
    test_hyde_truncates_long_output()
    print("✅ 全部查询改写单测通过")
