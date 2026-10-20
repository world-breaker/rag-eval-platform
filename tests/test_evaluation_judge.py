"""LLM-as-Judge 评测引擎的离线单测：设备搬运 / 兜底可观测 / 多次采样取中位数 / 配置校验。

硬约束（CI 必须能跑）：
- **不需要 GPU、不需要网络、不加载任何真实模型**：用假模型 + 假 tokenizer 断言行为；
- 设备搬运用「假 cuda 标签」验证：`model.device` 设成字符串 `"cuda:7"`，
  假张量的 `.to()` 只记录收到什么、不做真实分配——所以 CPU-only 环境也能证明
  「送进 generate 的每个张量都搬到了模型声明的设备」。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from src.evaluation import EvaluationEngine  # noqa: E402
from src.utils import load_config  # noqa: E402

DIM = "faithfulness"


# ---------------------------------------------------------------- 假张量 / 假 tokenizer / 假模型


class FakeTensor:
    """够用的假张量：只关心「送到哪个设备」和「长度」。"""

    def __init__(self, device, length=4):
        self.device = device
        self._length = length

    def to(self, device):
        return FakeTensor(device, self._length)

    def __getitem__(self, item):  # inputs["input_ids"][0]
        return self

    def __len__(self):  # len(inputs["input_ids"][0]) —— 用来切掉 prompt token
        return self._length


class FakeTokenizer:
    """按顺序吐出预设解码结果；编码输入永远产在 CPU（真实 tokenizer 的行为）。"""

    def __init__(self, decoded, device="__unset__"):
        self.decoded = list(decoded)
        self.eos_token_id = 0
        self._i = 0
        if device != "__unset__":
            self.device = device

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "FAKE_PROMPT"

    def __call__(self, text, return_tensors="pt"):
        return {
            "input_ids": FakeTensor(torch.device("cpu")),
            "attention_mask": FakeTensor(torch.device("cpu")),
        }

    def decode(self, tokens, skip_special_tokens=True):
        out = self.decoded[min(self._i, len(self.decoded) - 1)]
        self._i += 1
        return out


class FakeModel:
    """记录每次 generate 收到的张量设备与生成参数。"""

    def __init__(self, device, fail=False):
        if device is not None:
            self.device = device
        self.fail = fail
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(
            {
                "devices": {
                    k: v.device for k, v in kwargs.items() if isinstance(v, FakeTensor)
                },
                "do_sample": kwargs.get("do_sample"),
                "temperature": kwargs.get("temperature"),
            }
        )
        if self.fail:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but got index is on cpu"
            )
        return [[0, 0, 0, 0, 9]]


def _cfg(**eval_kwargs):
    cfg = {"judge_dimensions": [DIM]}
    cfg.update(eval_kwargs)
    return {"evaluation": cfg}


def _run(model, tokenizer, evaluator=None, **kwargs):
    engine = evaluator or EvaluationEngine(_cfg(judge_samples=1))
    return engine.evaluate_generation(
        "问题？", "回答。", "上下文。", model, tokenizer, return_meta=True, **kwargs
    )


# ---------------------------------------------------------------- 任务 1：设备搬运


def test_inputs_are_moved_to_model_device():
    """真实回归点：tokenizer 产出 CPU 张量，必须搬到 model.device 再喂 generate。"""
    model = FakeModel(torch.device("cpu"))
    tok = FakeTokenizer(["4"])
    out = _run(model, tok)
    assert out["scores"][DIM] == 4
    assert len(model.calls) == 1
    assert model.calls[0]["devices"] == {
        "input_ids": torch.device("cpu"),
        "attention_mask": torch.device("cpu"),
    }


def test_inputs_are_moved_to_fake_cuda_label():
    """假 cuda 标签：不需要真 GPU，也能证明「搬到模型声明的设备」而不是硬编码 CPU。"""
    model = FakeModel("cuda:7")
    tok = FakeTokenizer(["5"])
    out = _run(model, tok)
    assert out["scores"][DIM] == 5
    assert set(model.calls[0]["devices"].values()) == {"cuda:7"}
    assert out["meta"]["fallbacks"] == [], "设备搬对了就不该有兜底"


def test_device_falls_back_to_tokenizer_then_cpu():
    """model 没有 .device 时退回 tokenizer 的设备；两者都没有则退回 CPU。"""
    no_device_model = FakeModel(None)  # 不带 device 属性
    tok = FakeTokenizer(["4"], device="cuda:3")
    out = _run(no_device_model, tok)
    assert out["scores"][DIM] == 4
    assert set(no_device_model.calls[0]["devices"].values()) == {"cuda:3"}

    model2 = FakeModel(None)
    tok2 = FakeTokenizer(["4"])  # 也没有 device
    _run(model2, tok2)
    assert set(model2.calls[0]["devices"].values()) == {torch.device("cpu")}


def test_exception_still_continues_but_is_marked_as_fallback():
    """异常不许炸整批，但必须标记来源=heuristic（不是「裁判给了这个分」）。"""
    model = FakeModel(torch.device("cpu"), fail=True)
    tok = FakeTokenizer(["4"])
    out = _run(model, tok)
    score = out["scores"][DIM]
    # 异常 → 启发式（按回答长度），不是解析出来的 4
    assert score == min(5.0, max(1.0, len("回答。") / 200))
    assert out["meta"]["fallbacks"], "异常兜底必须出现在 meta.fallbacks 里"
    fb = out["meta"]["fallbacks"][0]
    assert fb["dim"] == DIM
    assert fb["source"] == "heuristic"
    assert "RuntimeError" in fb["reason"]
    assert "Expected all tensors" in fb["reason"], "原始异常信息要留证"
    assert out["meta"]["details"][DIM]["source"] == "heuristic"


# ---------------------------------------------------------------- 任务 1/5：解析路径


@pytest.mark.parametrize(
    "decoded,expected",
    [("4", 4.0), ("分数：5", 5.0), ("\n2\n", 2.0)],
)
def test_parse_variants(decoded, expected):
    model = FakeModel(torch.device("cpu"))
    out = _run(model, FakeTokenizer([decoded]))
    assert out["scores"][DIM] == expected
    assert out["meta"]["fallbacks"] == []


def test_unparsable_output_is_neutral_but_marked_as_fallback():
    """解析不出数字：沿用历史的中性 3.0，但必须被标记为兜底（不许静默）。"""
    model = FakeModel(torch.device("cpu"))
    out = _run(model, FakeTokenizer(["abc"]))
    assert out["scores"][DIM] == 3.0
    fb = out["meta"]["fallbacks"][0]
    assert fb["dim"] == DIM
    assert fb["source"] == "neutral_unparsed"
    assert "abc" in fb["reason"]
    assert out["meta"]["details"][DIM]["parsed"] == []


# ---------------------------------------------------------------- 任务 2：多次采样取中位数


def test_median_of_three_samples():
    model = FakeModel(torch.device("cpu"))
    tok = FakeTokenizer(["1", "5", "4"])
    engine = EvaluationEngine(_cfg(judge_samples=3))
    out = _run(model, tok, engine)
    assert out["scores"][DIM] == 4.0
    d = out["meta"]["details"][DIM]
    assert d["source"] == "llm_median"
    assert d["parsed"] == [1.0, 5.0, 4.0]
    assert d["raws"] == ["1", "5", "4"]
    assert len(model.calls) == 3
    # N>1 才走采样（否则三次贪心必然同分，中位数没有意义）
    assert all(c["do_sample"] is True for c in model.calls)
    assert all(c["temperature"] == pytest.approx(0.8) for c in model.calls)


def test_n_equals_one_matches_legacy_behavior():
    """N=1：单次调用、贪心解码、分数就是那一次的输出（与修复前逐比特一致）。"""
    model = FakeModel(torch.device("cpu"))
    tok = FakeTokenizer(["4"])
    out = _run(model, tok, EvaluationEngine(_cfg(judge_samples=1)))
    assert out["scores"][DIM] == 4
    assert len(model.calls) == 1
    assert model.calls[0]["do_sample"] is False
    assert out["meta"]["details"][DIM]["source"] == "llm"
    assert out["meta"]["judge_samples"] == 1


def test_median_skips_failed_samples_and_records_reason():
    """部分采样失败：用已成功的那几次取中位数，并把失败原因记下来（不静默）。"""

    class FlakyModel(FakeModel):
        def generate(self, **kwargs):
            if len(self.calls) == 1:  # 第二次调用失败
                self.calls.append({"devices": {}, "do_sample": True, "temperature": 0.8})
                raise RuntimeError("CUDA out of memory (fake)")
            return super().generate(**kwargs)

    model = FlakyModel(torch.device("cpu"))
    out = _run(model, FakeTokenizer(["5", "2"]), EvaluationEngine(_cfg(judge_samples=3)))
    d = out["meta"]["details"][DIM]
    assert d["parsed"] == [5.0, 2.0]
    assert out["scores"][DIM] == 3.5  # 中位数 = 中间两数均值
    assert "1/3 次" in (d["reason"] or "") or "2/3 次" in (d["reason"] or "")
    assert out["meta"]["fallbacks"] == [], "有可用分数就不算兜底，但失败原因仍要留证"


def test_default_return_is_plain_scores_for_backward_compat():
    model = FakeModel(torch.device("cpu"))
    engine = EvaluationEngine(_cfg(judge_samples=1))
    scores = engine.evaluate_generation("问题？", "回答。", "上下文。", model, FakeTokenizer(["4"]))
    assert scores == {DIM: 4.0}, "默认返回值仍是纯 {dim: score}，老调用方无需改动"


def test_judge_samples_validation():
    with pytest.raises(ValueError):
        EvaluationEngine(_cfg(judge_samples=0))
    with pytest.raises(ValueError):
        EvaluationEngine(_cfg(judge_samples="abc"))
    engine = EvaluationEngine(_cfg(judge_samples=1))
    with pytest.raises(ValueError):
        engine.set_judge_samples(-2)
    assert engine.set_judge_samples(3) == 3


# ---------------------------------------------------------------- 任务 3：vector_store 校验


def test_project_config_is_valid_and_defaults_to_one_sample():
    cfg = load_config(str(Path(__file__).parent.parent / "config.yaml"))
    assert cfg["retrieval"]["vector_store"] == "numpy"
    assert cfg["evaluation"]["judge_samples"] == 1, "默认必须是 1（与历史行为一致）"


def test_chromadb_vector_store_fails_loudly(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("retrieval:\n  vector_store: chromadb\n", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_config(str(path))
    msg = str(ei.value)
    assert "chromadb" in msg
    assert "numpy" in msg, "报错必须给出可执行的替代方案"
    assert "未接入" in msg


def test_unknown_vector_store_also_fails(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("retrieval:\n  vector_store: faiss\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(path))
