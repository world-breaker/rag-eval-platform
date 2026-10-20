"""生成层评测的离线单测：规则判据 / 一致率统计 / 强裁判 prompt 与解析。

硬约束（CI 必须能跑）：
- **不调用任何 API**（不 import requests、不发网络请求）；
- **不加载任何大模型**（不 import torch/transformers/sentence-transformers）；
- 只用确定性的纯函数与仓库内的样本文件。
"""

import datetime
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_generation_eval import (  # noqa: E402
    ATTRIBUTION_THRESHOLD,
    DATASETS,
    JUDGE_CRITERIA,
    JUDGE_DIMENSIONS,
    agreement_stats,
    build_api_judge_messages,
    cache_key,
    cache_key_n,
    cache_keys,
    char_f1,
    cost_cny,
    detect_refusal,
    exact_match,
    extract_facts,
    fact_landing,
    is_peak,
    ngram_coverage,
    parse_api_judge,
    refusal_verdict,
    rule_faithfulness,
    rule_scores,
    segment_report,
    sentence_attribution,
    split_sentences,
)

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------- 数字 / 实体落地


def test_extract_facts_picks_numbers_and_identifiers():
    facts = extract_facts("LoRA 典型配置 r=8，α=16，可训练参数占 0.3%。")
    assert "8" in facts
    assert "16" in facts
    assert "0.3%" in facts
    assert "lora" in facts


def test_fact_landing_all_hit():
    ctx = "LoRA 在原始权重旁添加低秩矩阵，典型配置 r=8、α=16，可训练参数仅占总量的 0.3% 左右。"
    ans = "LoRA 的典型配置是 r=8，可训练参数占 0.3% 左右。"
    land = fact_landing(ans, ctx)
    assert land["n"] >= 3
    assert land["missed"] == []
    assert land["rate"] == 1.0


def test_fact_landing_detects_fabricated_number():
    ctx = "LoRA 典型配置 r=8、α=16，可训练参数仅占总量的 0.3% 左右。"
    ans = "LoRA 的推荐配置是 r=64，可训练参数占比高达 12.5%。"
    land = fact_landing(ans, ctx)
    # 8 出现在回答里吗？没有；被编造的 64 与 12.5% 必须被点名
    assert "64" in land["missed"]
    assert "12.5%" in land["missed"]
    assert land["rate"] is not None and land["rate"] < 0.5


def test_fact_landing_rate_none_when_no_facts():
    land = fact_landing("这个模型很适合做本地部署。", "无关上下文")
    assert land["n"] == 0
    assert land["rate"] is None


def test_fact_landing_uses_reference_as_whitelist():
    """参考回答是事实白名单：上下文被截断时，模型答对但没落在上下文里不该被判编造。"""
    land = fact_landing("训练用了 8 张 A100。", "上下文里没有显卡信息。", extra="训练用了 8 张 A100。")
    assert land["missed"] == []
    assert land["rate"] == 1.0


# ---------------------------------------------------------------- 句子级归因阈值


def test_split_sentences():
    sents = split_sentences("第一句内容。第二句内容！短。")
    assert sents == ["第一句内容", "第二句内容"]  # 过短碎片被丢弃


def test_ngram_coverage_bounds():
    assert ngram_coverage("量化把权重转成低位整数。", "量化把权重转成低位整数。") == 1.0
    assert ngram_coverage("完全无关的另一句话。", "量化把权重转成低位整数。") < 0.2
    assert ngram_coverage("", "任何上下文") == 0.0


def test_sentence_attribution_both_sides_of_threshold():
    """阈值两侧：一个字面拷贝句（覆盖率 ~1.0）与一句全新表述（覆盖率很低）。"""
    ctx = "检索质量评测用 Recall@3、Recall@5、Precision@3 与 MRR 四个指标衡量。"
    copied = "检索质量评测用 Recall@3、Recall@5、Precision@3 与 MRR 四个指标衡量。"
    novel = "因此建议在预处理阶段引入停用词清洗与同义词归一流程。"
    attr = sentence_attribution(f"{copied}。{novel}。", ctx)
    assert len(attr["sentences"]) == 2
    assert attr["sentences"][0]["attributed"] is True
    assert attr["sentences"][1]["attributed"] is False
    assert attr["rate"] == 0.5
    assert attr["threshold"] == ATTRIBUTION_THRESHOLD


def test_attribution_threshold_boundary_is_inclusive():
    """阈值语义：覆盖率 ≥ 阈值即判为已归因（>= 而不是 >）。

    用字符 bigram 精确可算的三句构造：0.6（正好在阈值上，应判为已归因）、
    0.625（过）、0.4（不过）。
    """
    ctx = "abcdefghij"
    exactly = sentence_attribution("abcdxy。", ctx, threshold=0.6)  # 3/5 = 0.6
    over = sentence_attribution("abcdefxyz。", ctx, threshold=0.6)  # 5/8 = 0.625
    under = sentence_attribution("abcxyq。", ctx, threshold=0.6)  # 2/5 = 0.4
    assert exactly["sentences"][0]["coverage"] == 0.6
    assert exactly["sentences"][0]["attributed"] is True
    assert over["sentences"][0]["coverage"] == 0.625
    assert over["sentences"][0]["attributed"] is True
    assert under["sentences"][0]["coverage"] == 0.4
    assert under["sentences"][0]["attributed"] is False


def test_sentence_attribution_rate_none_when_no_sentence():
    attr = sentence_attribution("好。", "上下文")  # 全部句子过短被过滤
    assert attr["n"] == 0
    assert attr["rate"] is None


# ---------------------------------------------------------------- 拒答判定


def test_detect_refusal_positive_negative():
    assert detect_refusal("根据提供的文档无法回答该问题。") is True
    assert detect_refusal("文档中没有提到相关信息。") is True
    assert detect_refusal("不知道。") is True
    assert detect_refusal("量化把浮点权重转换为低位整数表示。") is False


def test_refusal_verdict_positive_and_negative_cases():
    # 不可答样本：拒答 → 正确；回答 → 错误
    assert refusal_verdict("无法回答该问题", expect_refusal=True)["correct"] is True
    assert refusal_verdict("量化就是把权重用低位整数表示。", expect_refusal=True)["correct"] is False
    # 可答样本：回答 → 正确；拒答 → 错误
    assert refusal_verdict("量化就是把权重用低位整数表示。", expect_refusal=False)["correct"] is True
    assert refusal_verdict("无法回答该问题", expect_refusal=False)["correct"] is False


# ---------------------------------------------------------------- 词法指标


def test_exact_match_normalization():
    assert exact_match("深度学习", " 深度学习。 ") == 1.0
    assert exact_match("深度学习", "机器学习") == 0.0


def test_char_f1_boundaries():
    assert char_f1("深度学习", "深度学习") == 1.0
    assert char_f1("深度学习", "天气预报") == 0.0
    assert char_f1("", "深度学习") == 0.0
    assert char_f1("深度学习", "") == 0.0
    assert char_f1("深度学习", "深度网络") == 0.5  # 交集「深度」2 字 / 各 4 字


def test_rule_scores_with_reference_answer():
    scores = rule_scores(
        answer="LoRA 的典型配置是 r=8。",
        context="LoRA 典型配置 r=8。",
        reference="LoRA 的典型配置是 r=8。",
    )
    assert set(["numeric_landing", "sentence_attribution", "lexical", "rule_faithfulness"]) <= set(scores)
    assert scores["lexical"]["em"] == 1.0
    assert scores["lexical"]["f1"] == 1.0
    # 拒答判定对所有样本都算：可答样本上「没拒答」= 正确
    assert scores["refusal"]["is_refusal"] is False
    assert scores["refusal"]["correct"] is True


def test_false_refusal_on_answerable_is_flagged_but_does_not_inflate_rule_score():
    """可答样本上误拒答：拒答判定要报错，但规则分不能被「拒答」刷成 5。"""
    ctx = "除以 √d_k 是为了防止点积方差随维度增长导致 softmax 梯度消失。"
    answer = "根据提供的文档无法回答该问题"
    s_answerable = rule_scores(answer, ctx, expect_refusal=False)
    assert s_answerable["refusal"]["is_refusal"] is True
    assert s_answerable["refusal"]["correct"] is False
    assert s_answerable["rule_faithfulness"] == 1.0  # 归因率 0 → 低分，没有被拒答刷分
    # 同一句回答放在「不可答」样本上就是正确拒答 → 满分
    s_unanswerable = rule_scores(answer, ctx, expect_refusal=True)
    assert s_unanswerable["refusal"]["correct"] is True
    assert s_unanswerable["rule_faithfulness"] == 5.0


def test_rule_scores_without_reference_skips_lexical():
    scores = rule_scores(answer="任何回答。", context="任何上下文。")
    assert "note" in scores["lexical"]


def test_rule_faithfulness_mapping():
    assert rule_faithfulness({"rate": 1.0}, {"rate": 1.0}, None) == 5.0
    assert rule_faithfulness({"rate": 0.0}, {"rate": 1.0}, None) == 1.0
    # 取 min：只要硬事实没落地就封顶
    assert rule_faithfulness({"rate": 1.0}, {"rate": 0.5}, None) == 3.0
    # 不可答样本只看拒答正确性
    assert rule_faithfulness({"rate": 0.0}, {"rate": None}, {"correct": True}) == 5.0
    assert rule_faithfulness({"rate": 1.0}, {"rate": 1.0}, {"correct": False}) == 1.0
    # 无句子无事实 → 中性 3
    assert rule_faithfulness({"rate": None}, {"rate": None}, None) == 3.0


# ---------------------------------------------------------------- 一致率统计


def test_agreement_stats_basic():
    s = agreement_stats([5, 4, 3, 2], [5, 5, 3, 1])
    assert s["n"] == 4
    assert s["exact_agreement"] == 0.5  # 只有第 1、3 条完全相同
    assert s["within1_agreement"] == 1.0
    assert s["mae"] == 0.5
    assert s["binary_agreement"] == 1.0
    assert s["confusion_high_low"] == {"tp": 2, "tn": 2, "fp": 0, "fn": 0}


def test_agreement_stats_skips_missing():
    s = agreement_stats([5, None, 3], [5, 4, None])
    assert s["n"] == 1
    assert s["mae"] == 0.0


def test_agreement_stats_empty():
    s = agreement_stats([None], [None])
    assert s["n"] == 0
    assert "note" in s


def test_segment_report_splits_by_strong_judge():
    segs = segment_report([5, 5, 2], [5, 4, 1])
    assert segs["高(4-5)"]["n"] == 2
    assert segs["高(4-5)"]["strong_mean"] == 4.5
    assert segs["高(4-5)"]["local_mean"] == 5.0
    assert segs["高(4-5)"]["mae"] == 0.5
    assert segs["中(3)"]["n"] == 0
    assert segs["低(1-2)"]["n"] == 1


# ---------------------------------------------------------------- 强裁判 prompt / 解析


def test_rubric_matches_src_evaluation_verbatim():
    """两臂必须用同一套 rubric：判据文本与 src/evaluation.py 逐字一致（防漂移）。"""
    src = (ROOT / "src/evaluation.py").read_text(encoding="utf-8")
    for dim, criteria in JUDGE_CRITERIA.items():
        assert dim in src
        assert criteria in src, f"{dim} 的判据文本与 src/evaluation.py 不一致（rubric 漂移）"
    assert JUDGE_DIMENSIONS == ["faithfulness", "relevance", "completeness"]


def test_api_judge_messages_contain_rubric_and_format():
    msgs = build_api_judge_messages("问题？", "回答。", "上下文。")
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    for dim in JUDGE_DIMENSIONS:
        assert dim in user
        assert JUDGE_CRITERIA[dim] in user
    assert "faithfulness" in user and "1-5" in user
    assert "问题？" in user and "回答。" in user


def test_parse_api_judge_variants():
    ok = parse_api_judge('{"faithfulness": 5, "relevance": 4, "completeness": 3}')
    assert ok == {"faithfulness": 5, "relevance": 4, "completeness": 3}
    # markdown 代码块
    fenced = parse_api_judge('```json\n{"faithfulness": 4, "relevance": 4, "completeness": 4}\n```')
    assert fenced["faithfulness"] == 4
    # 前后有多余文字
    noisy = parse_api_judge('我的评分是 {"faithfulness": 2, "relevance": 3, "completeness": 1} 以上。')
    assert noisy == {"faithfulness": 2, "relevance": 3, "completeness": 1}
    # 字符串数字
    assert parse_api_judge('{"faithfulness": "4", "relevance": 4, "completeness": 4}')["faithfulness"] == 4
    # 越界与缺失 → None
    bad = parse_api_judge('{"faithfulness": 7, "relevance": 4}')
    assert bad["faithfulness"] is None and bad["completeness"] is None and bad["relevance"] == 4
    # 完全无法解析
    assert parse_api_judge("我拒绝打分") == {d: None for d in JUDGE_DIMENSIONS}


# ---------------------------------------------------------------- 费用 / 峰时


def test_is_peak_uses_utc_weekday_hours():
    # 2026-10-19 是周一：UTC 02:00 属峰时，UTC 13:30 属闲时
    peak = datetime.datetime(2026, 10, 19, 2, 0, tzinfo=datetime.timezone.utc).timestamp()
    off = datetime.datetime(2026, 10, 19, 13, 30, tzinfo=datetime.timezone.utc).timestamp()
    assert is_peak(peak) is True
    assert is_peak(off) is False
    # 周六全天闲时
    sat = datetime.datetime(2026, 10, 24, 2, 0, tzinfo=datetime.timezone.utc).timestamp()
    assert is_peak(sat) is False


def test_cost_cny_uses_official_rates():
    peak = datetime.datetime(2026, 10, 19, 2, 0, tzinfo=datetime.timezone.utc).timestamp()
    off = datetime.datetime(2026, 10, 19, 13, 30, tzinfo=datetime.timezone.utc).timestamp()
    # 闲时：1M 输入 $0.15 → ¥1.065；1M 输出 $0.60 → ¥4.26
    assert cost_cny(1_000_000, 0, off) == pytest.approx(0.15 * 7.1, rel=1e-6)
    assert cost_cny(0, 1_000_000, off) == pytest.approx(0.60 * 7.1, rel=1e-6)
    # 峰时是闲时的两倍
    assert cost_cny(1_000_000, 0, peak) == pytest.approx(0.30 * 7.1, rel=1e-6)


# ---------------------------------------------------------------- 数据集与安全


def test_all_dataset_files_exist():
    for ds in DATASETS:
        assert (ROOT / "data/sample" / ds["kb"]).exists(), ds["kb"]
        assert (ROOT / "data/sample" / ds["qa"]).exists(), ds["qa"]


def test_unanswerable_samples_schema():
    """新造的拒答样本：6 条、有参考回答、无相关文档（确实不在知识库中）。"""
    path = ROOT / "data/sample/test_qa_unanswerable.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 6
    for case in data:
        assert case["question"].strip()
        assert case["reference_answer"].strip()
        assert case["relevant_sources"] == []


def test_unanswerable_questions_absent_from_knowledge_base():
    """拒答样本的前提：关键词确实不出现在所给知识库里。"""
    kb_text = (ROOT / "data/sample/knowledge_base_large.json").read_text(encoding="utf-8")
    for kw in ["世界杯", "回锅肉", "营收", "大兴机场", "绝育", "嫦娥"]:
        assert kw not in kb_text, f"{kw} 竟出现在知识库中，拒答样本无效"


def test_eval_script_contains_no_api_key_literal():
    """密钥纪律：脚本里不允许出现任何 sk- 形态的字面量（key 只从环境变量读）。"""
    src = (ROOT / "experiments/run_generation_eval.py").read_text(encoding="utf-8")
    assert ("sk" + "-") not in src
    assert "DEEPSEEK_API_KEY" in src
    assert "os.environ.get(\"DEEPSEEK_API_KEY\"" in src


def test_eval_script_has_no_top_level_heavy_imports():
    """CI 友好：模块顶层不得 import torch / transformers / sentence_transformers / requests。"""
    src = (ROOT / "experiments/run_generation_eval.py").read_text(encoding="utf-8")
    head = src.split("def ")[0]
    for heavy in ["import torch", "import transformers", "sentence_transformers", "import requests"]:
        assert heavy not in head, f"顶层出现重依赖：{heavy}"


# ---------------------------------------------------------------- 多次采样：缓存隔离


def test_api_cache_key_depends_on_judge_samples():
    """N 必须进缓存 key：否则 N=1 的缓存会被 N=3 命中，中位数就等于单次值。"""
    msgs = build_api_judge_messages("问题？", "回答。", "上下文。")
    k1 = cache_key_n(msgs, "deepseek-chat", 1, 0, 0.0)
    k3 = cache_key_n(msgs, "deepseek-chat", 3, 0, 0.0)
    assert k1 != k3
    # 同 N 的不同采样次数也要分开（否则 3 次采样取回同一条响应）
    assert cache_key_n(msgs, "deepseek-chat", 3, 0, 0.8) != cache_key_n(
        msgs, "deepseek-chat", 3, 1, 0.8
    )
    # temperature 也进 key（0 与 0.8 的响应不同，不能混用）
    assert cache_key_n(msgs, "deepseek-chat", 3, 0, 0.0) != cache_key_n(
        msgs, "deepseek-chat", 3, 0, 0.8
    )


def test_n1_can_still_hit_legacy_cache_but_n3_cannot():
    """N=1 允许回落到历史（无 N）缓存，避免重复付费；N=3 绝不允许命中它。"""
    msgs = build_api_judge_messages("问题？", "回答。", "上下文。")
    legacy = cache_key(msgs, "deepseek-chat")
    keys_n1 = cache_keys(msgs, "deepseek-chat", 1, 0, 0.0)
    assert legacy in keys_n1
    keys_n3 = cache_keys(msgs, "deepseek-chat", 3, 0, 0.8)
    assert legacy not in keys_n3
    assert len(keys_n3) == 1


def test_local_judge_no_longer_uses_tokenizer_device_proxy():
    """回归守卫：设备搬运已进 src/evaluation.py，实验侧不得再留 workaround 类。

    保留 workaround 会让「修复到底有没有生效」永远说不清（两条路径都通）。
    """
    src = (ROOT / "experiments/run_generation_eval.py").read_text(encoding="utf-8")
    assert "class DeviceTokenProxy" not in src
    # 直接传真实 tokenizer（不再包一层）
    assert "judge_tokenizer = pipe.generator.tokenizer" in src

