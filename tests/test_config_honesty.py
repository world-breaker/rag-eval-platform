"""配置诚实性的离线单测：**配置里不许有永不生效的选项**。

背景（2026-10-20）：一轮只读审计发现 config.yaml 里有一批「读了不生效 / 没人读」的键——
写的人以为它能调那个行为，代码根本不看；或者代码读了却不用。
本文件把「不许有死配置」这条纪律变成可执行的断言，防止它们以后又被加回来。

覆盖：
- 已删除的死键不许复活（retrieval_metrics / traditional_metrics / persist_dir /
  evaluation.judge_model / query_rewrite.max_new_tokens）；
- 保留的扩展位必须 fail-fast（hybrid.rerank.type 只能写 cross_encoder）；
- 保留的键必须真的被代码读走（app.title、evaluation.test_data_path）。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

CONFIG_PATH = ROOT / "config.yaml"


def _cfg():
    return load_config(str(CONFIG_PATH))


def test_dead_keys_are_gone():
    """这些键曾经「写了不生效」，已删除；不许复活。"""
    cfg = _cfg()
    assert "retrieval_metrics" not in cfg["evaluation"], (
        "retrieval_metrics 读了不用（检索指标是代码里固定的一套），已删除"
    )
    assert "traditional_metrics" not in cfg["evaluation"], (
        "traditional_metrics 读了不用（BLEU/ROUGE-L 是代码里固定的一套），已删除"
    )
    assert "persist_dir" not in cfg["retrieval"], (
        "persist_dir 没有任何代码读取（vector_store=numpy 时无意义），已删除"
    )
    assert "judge_model" not in cfg["evaluation"], (
        "judge_model 没有任何代码读取（裁判恒等于生成模型），已删除"
    )
    assert "max_new_tokens" not in cfg["retrieval"]["query_rewrite"], (
        "query_rewrite.max_new_tokens 从未传给 LLM（读了不用），已删除"
    )


def test_live_keys_are_present():
    """这些键是活的：代码真的读它们。"""
    cfg = _cfg()
    assert cfg["retrieval"]["hybrid"]["rerank"]["type"] == "cross_encoder"
    assert cfg["evaluation"]["test_data_path"]
    assert cfg["app"]["title"]


def test_unknown_rerank_type_fails_loudly(tmp_path):
    """rerank.type 写别的值 → 加载配置即报错（不静默按 cross_encoder 跑）。"""
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "retrieval:\n  hybrid:\n    rerank:\n      type: colbert\n", encoding="utf-8"
    )
    with pytest.raises(ValueError) as ei:
        load_config(str(path))
    msg = str(ei.value)
    assert "colbert" in msg
    assert "cross_encoder" in msg, "报错必须给出可执行的替代方案"


def test_unknown_vector_store_still_fails(tmp_path):
    """既有行为不变：vector_store 仍然 fail-fast。"""
    path = tmp_path / "cfg.yaml"
    path.write_text("retrieval:\n  vector_store: faiss\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(path))


def test_app_title_and_test_data_path_are_actually_wired():
    """app.title / evaluation.test_data_path 必须被 app.py 读走（静态检查）。

    为什么不 import app.py 做运行时检查：import 它会加载 BGE embedding 模型并构建
    Gradio 界面（要 GPU + 几百 MB 权重），CI 上跑不动。静态检查足以守住
    「配置项真的被读走」这条纪律。
    """
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '"app"' in app_src and "title" in app_src, "app.title 必须被 app.py 读走"
    assert "test_data_path" in app_src, "evaluation.test_data_path 必须被 app.py 读走"


if __name__ == "__main__":
    import tempfile

    test_dead_keys_are_gone()
    test_live_keys_are_present()
    with tempfile.TemporaryDirectory() as _d:
        test_unknown_rerank_type_fails_loudly(Path(_d))
        test_unknown_vector_store_still_fails(Path(_d))
    test_app_title_and_test_data_path_are_actually_wired()
    print("✅ 全部配置诚实性单测通过")
