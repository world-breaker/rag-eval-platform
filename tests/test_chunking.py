"""文档切分策略的 fail-fast 单测（不加载模型、不联网）。

守的是一条会「造出假对比」的坑：策略名写错（如 `semantc`）时，
旧实现只 warning 然后回退 `fixed_size`——于是「三种策略对比」里有两个策略
其实是同一个，实验照样跑出一张看起来没问题的表。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.chunking import DocumentChunker, SUPPORTED_STRATEGIES  # noqa: E402


def _cfg(strategy="fixed_size"):
    return {
        "chunking": {
            "strategy": strategy,
            "chunk_size": 64,
            "chunk_overlap": 8,
            "separators": ["\n\n", "\n", "。", "."],
        }
    }


def test_supported_strategies_are_exactly_three():
    assert SUPPORTED_STRATEGIES == ("fixed_size", "recursive", "semantic")


def test_unknown_strategy_raises_instead_of_silently_falling_back():
    """策略名写错 → 立刻 ValueError，不许静默回退 fixed_size（否则是假对比）。"""
    with pytest.raises(ValueError) as ei:
        DocumentChunker(_cfg("semantc"))
    msg = str(ei.value)
    assert "semantc" in msg
    assert "fixed_size" in msg, "报错要给出可选值，便于直接改对"


def test_unknown_strategy_set_at_runtime_also_raises():
    """运行时被改坏（compare_chunk_strategies 直接赋值 strategy）同样要炸。"""
    chunker = DocumentChunker(_cfg("fixed_size"))
    chunker.strategy = "semantc"
    with pytest.raises(ValueError):
        chunker.chunk("一些中文文本，用于切分。")


def test_valid_strategies_still_work():
    for strategy in SUPPORTED_STRATEGIES:
        chunker = DocumentChunker(_cfg(strategy))
        chunks = chunker.chunk("人工智能是计算机科学的分支。\n\n它致力于创造智能机器。")
        assert chunks, f"{strategy} 切分结果为空"
        assert all("text" in c and "chunk_index" in c for c in chunks)


if __name__ == "__main__":
    test_supported_strategies_are_exactly_three()
    print("✅ 全部切分策略 fail-fast 单测通过")
