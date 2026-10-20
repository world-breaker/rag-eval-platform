# -*- coding: utf-8 -*-
"""文档数字自检（pytest 版）：README / 报告里的每个关键数字，必须能从已提交的结果 JSON 重算出来。

为什么要有这个测试
------------------
这个仓库的 README 与 `experiments/results/*.md` 里全是精确到小数点后四位的数字，
而这些数字**全部**来自同目录下的结果 JSON。文档是人写的、JSON 是脚本出的，
两边一旦不同步，读者拿到的就是过期结论——而"诚实"正是这个作品集的主线。

所以把「文档里的数字」变成**可执行断言**：改了 JSON 或改了文档，只要两者不自洽，
本测试就失败（CI 门禁直接红）。判据实现见 `experiments/check_doc_numbers.py`。

注意：本测试**不联网、不跑模型**，纯读仓库内已提交的 JSON 与文档，秒级完成。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "experiments" / "check_doc_numbers.py"


def _load_checker():
    if not SCRIPT.exists():
        raise AssertionError(f"缺少自检脚本：{SCRIPT}")
    spec = importlib.util.spec_from_file_location("check_doc_numbers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_doc_numbers_are_recomputable():
    """每条 claim：重算值必须与文档字面量一致，且该字面量确实出现在文档里。"""
    checker = _load_checker()
    claims = checker.build_claims()
    assert claims, "claim 列表为空 —— 自检脚本本身可能坏了"

    failures = []
    for claim in claims:
        ok, why = claim.check()
        if not ok:
            failures.append(f"{claim.cid}（{claim.note}）：{why}")

    assert not failures, (
        "文档里的数字与已提交结果 JSON 不自洽：\n  - "
        + "\n  - ".join(failures)
        + "\n\n处理方式：要么改文档让它跟数据一致，要么重跑实验生成新数据并同步更新文档。"
    )


def test_claim_set_is_not_empty_and_covers_all_three_reports():
    """防止有人把 claim 删光导致测试变成永真。"""
    checker = _load_checker()
    claims = checker.build_claims()
    assert len(claims) >= 30, f"claim 数量异常偏少（{len(claims)}），覆盖面可能被削掉了"
    covered = {doc for claim in claims for doc in claim.docs}
    for required in (
        "README.md",
        "experiments/results/generation_eval_2026-10-19.md",
        "experiments/results/judge_variance_2026-10-20.md",
        "experiments/results/reference_robustness_2026-10-20.md",
    ):
        assert required in covered, f"没有任何 claim 检查 {required}"
