"""Embedding 离线缓存的键与落盘单测（不加载真实模型、不联网）。

守的是一条会「造出假绿」的坑：**缓存键里没有模型名**——
换了 `embedding.model_name` 之后缓存照样命中，于是拿**旧模型**的向量当新模型结果
跑检索实验，指标「看起来正常」，实则是旧模型的。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.embedding import EmbeddingEngine  # noqa: E402


class _FakeSTModel:
    """假 embedding 模型：按文本内容给确定性向量（维度 4）。"""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True):
        self.calls += 1
        out = np.zeros((len(texts), 4), dtype=float)
        for i, t in enumerate(texts):
            out[i] = [len(t) % 7, 1.0, 0.0, 0.5]
        return out


def _engine(tmp_path, model_name):
    """绕开 SentenceTransformer 加载，只装配被测代码需要的属性。"""
    eng = object.__new__(EmbeddingEngine)
    eng.model_name = model_name
    eng.device = "cpu"
    eng.batch_size = 8
    eng.cache_dir = tmp_path
    eng.cache_dir.mkdir(parents=True, exist_ok=True)
    eng.model = _FakeSTModel()
    return eng


def test_cache_key_includes_model_name(tmp_path):
    """同一段文本、两个不同 model_name → 不命中同一个缓存文件（也不共用向量）。"""
    text = "NF4 量化格式为什么对正态分布友好？"
    chunks = [{"text": text}]

    eng_a = _engine(tmp_path / "a", "BAAI/bge-small-zh-v1.5")
    eng_b = _engine(tmp_path / "b", "BAAI/bge-base-zh-v1.5")

    key_a = eng_a._cache_key([text])
    key_b = eng_b._cache_key([text])
    assert key_a != key_b, "缓存键必须含模型名：换模型后不能命中旧缓存"

    vec_a = eng_a.encode(chunks)
    vec_b = eng_b.encode(chunks)
    assert eng_a.model.calls == 1 and eng_b.model.calls == 1, "两个模型各自都要真的算一次"
    assert (eng_a.cache_dir / f"{key_a}.npy").exists()
    assert (eng_b.cache_dir / f"{key_b}.npy").exists()

    # 同模型同文本 → 第二次命中缓存（不再调模型）
    calls_before = eng_a.model.calls
    again = eng_a.encode(chunks)
    assert eng_a.model.calls == calls_before, "同模型同文本应命中缓存"
    assert np.allclose(again, vec_a)


def test_same_model_same_text_hits_cache(tmp_path):
    """同模型 + 同文本 → 命中缓存（缓存本身仍然有效）。"""
    eng = _engine(tmp_path / "m", "BAAI/bge-small-zh-v1.5")
    chunks = [{"text": "缓存命中测试"}]
    eng.encode(chunks)
    calls = eng.model.calls
    eng.encode(chunks)
    assert eng.model.calls == calls
    assert len(list((tmp_path / "m").glob("*.npy"))) == 1


def test_broken_cache_is_recomputed(tmp_path):
    """缓存文件条数与本次不符（半截文件/换语料）→ 按未命中重算，不返回错形状的向量。"""
    eng = _engine(tmp_path / "c", "BAAI/bge-small-zh-v1.5")
    key = eng._cache_key(["a", "b"])
    np.save(eng.cache_dir / f"{key}.npy", np.zeros((1, 4)))  # 条数故意不符

    out = eng.encode([{"text": "a"}, {"text": "b"}])
    assert out.shape == (2, 4), "形状不对的缓存必须被忽略并重算"
    assert eng.model.calls == 1


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as _d:
        _p = Path(_d)
        test_cache_key_includes_model_name(_p)
        test_same_model_same_text_hits_cache(_p)
        test_broken_cache_is_recomputed(_p)
    print("✅ 全部 embedding 缓存单测通过")
