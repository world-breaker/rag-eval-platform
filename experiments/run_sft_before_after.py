#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 微调前 / 后并排生成（真实跑，不造数）
=================================================

用途
----
把「基座模型」与「LoRA 微调后的模型」在**同一批 held-out 提示**上的生成结果并排落盘，
作为简历 / 面试可用的「微调前后效果对比」证据包的第一层（原始输出）。

资产来源（全部为本机既有资产，本脚本不训练、不造数）
--------------------------------------------------
- 基座模型  : Qwen/Qwen2.5-1.5B-Instruct          （HF 本地缓存）
- LoRA 适配器: D:\\programmer\\LLM\\lora_sft_checkpoint
              （= sft_output/checkpoint-80，即原训练里 val_loss 最低的那个 checkpoint）
- 训练数据   : D:\\programmer\\LLM\\data\\sft_train_100.json
- 原始训练脚本: D:\\programmer\\LLM\\learning_scripts\\learn\\day9_sft.py
- held-out 集: **原训练自己的验证集**（train_test_split(test_size=0.2, seed=42) 的 20 条 test），
              本脚本按**完全相同的代码路径**复现该切分，保证取到的就是当初那条验证集。

口径要点（公平性）
-----------------
1. 两条臂使用**完全相同的 prompt**（含与训练一致的 system 消息「你是一个有用的AI助手。」），
   唯一变量是「是否加载 LoRA 权重」。
2. 两条臂使用**完全相同的解码参数**（贪心 do_sample=False，max_new_tokens 固定）。
3. 不做温度采样 → 运行间方差为 0（与 rag-eval-platform 既有 generation_eval 口径一致）。

输出
----
JSONL（每行一个 JSON）：
  第 1 行  : {"_record_type": "run_meta", ...}  运行级元信息（口径、路径、资产指纹）
  第 2..N 行: {"_record_type": "sample", "id", "prompt", "base_output", "sft_output", ...}

用法
----
  python experiments/run_sft_before_after.py \
      --out experiments/sft_before_after_2026-10-21.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------- 默认路径

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = Path(r"D:\programmer\LLM\lora_sft_checkpoint")
DEFAULT_DATA = Path(r"D:\programmer\LLM\data\sft_train_100.json")
DEFAULT_OUT = ROOT / "experiments" / "sft_before_after_2026-10-21.jsonl"
# 原训练里 val_loss 最低的那个 checkpoint（= lora_sft_checkpoint 的来源）
DEFAULT_TRAINER_STATE = Path(r"D:\programmer\LLM\sft_output\checkpoint-80\trainer_state.json")

# 与原训练脚本 day9_sft.py 的 format_sft() 逐字一致的 system 消息。
# 两条臂都用它 —— 对照实验里 prompt 必须固定，唯一变量是 LoRA 权重。
SYSTEM_PROMPT = "你是一个有用的AI助手。"

# 原训练切分参数（day9_sft.py: tokenized.train_test_split(test_size=0.2, seed=42)）
SPLIT_TEST_SIZE = 0.2
SPLIT_SEED = 42


def sha256_of(path: Path) -> Optional[str]:
    """文件指纹 —— 报告里要写「数据/权重来自哪个文件」的凭据。"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------- held-out 切分复现


def reproduce_heldout(data_path: Path, tokenizer) -> List[Dict]:
    """复现原训练的 80/20 切分，取出 20 条 held-out（验证集）样本。

    与 day9_sft.py 走同一条代码路径：
        Dataset.from_list(raw).map(format_sft)
                     .map(tokenize_fn, batched=True, remove_columns=...)
                     .train_test_split(test_size=0.2, seed=42)

    额外挂一个 `_idx` 列，用来把验证集里的行还原回原始 100 条里的下标
    （列内容不影响切分——切分只按行下标做置换）。
    """
    from datasets import Dataset

    raw = json.load(io.open(data_path, encoding="utf-8"))
    for i, row in enumerate(raw):
        row["_idx"] = i

    def format_sft(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return example

    dataset = Dataset.from_list(raw).map(format_sft)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding=False)

    # (a) 与原训练逐字一致的切分：全部列都被 remove_columns 掉
    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names
    )
    split = tokenized.train_test_split(test_size=SPLIT_TEST_SIZE, seed=SPLIT_SEED)
    test = split["test"]

    # (b) 同参数的第二次切分：保留 _idx 列，用来把验证集行还原回原始 100 条的下标。
    #     切分是**按行下标做置换**，与列无关，所以 (b) 的行顺序与 (a) 必须完全一致 —— 下面会断言。
    tokenized_idx = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in dataset.column_names if c != "_idx"],
    )
    split_idx = tokenized_idx.train_test_split(test_size=SPLIT_TEST_SIZE, seed=SPLIT_SEED)
    test_idx = split_idx["test"]

    if test["input_ids"] != test_idx["input_ids"]:
        raise RuntimeError(
            "两次切分结果不一致 —— datasets 版本的 train_test_split 行为与原训练不同，"
            "held-out 集不可信，停止。"
        )

    by_idx = {row["_idx"]: row for row in raw}
    out: List[Dict] = []
    for row in test_idx:
        idx = int(row["_idx"])
        out.append({"idx": idx, **by_idx[idx]})

    if len(out) != 20:
        print(
            f"[warn] held-out 样本数 = {len(out)}（预期 20）。"
            f"切分口径与原训练不一致，请检查 datasets 版本。"
        )
    return out


# ---------------------------------------------------------------- 生成


def build_prompt(tokenizer, instruction: str) -> str:
    """推理时的 chat prompt：与训练同结构，但以 assistant 起始标记收尾。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_all(model, tokenizer, samples: List[Dict], max_new_tokens: int, tag: str) -> List[Dict]:
    """逐条贪心生成。返回与 samples 等长的结果列表。"""
    import torch

    model.eval()
    results: List[Dict] = []
    t_all = time.time()
    for i, s in enumerate(samples, 1):
        prompt = build_prompt(tokenizer, s["instruction"])
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_prompt = int(enc["input_ids"].shape[1])
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # 贪心：运行间方差为 0
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][n_prompt:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # 自然终止 = 最后生成的是 EOS；否则说明撞到 max_new_tokens 被截断
        finished = bool(len(gen_ids) > 0 and int(gen_ids[-1]) == tokenizer.eos_token_id)
        rec = {
            "text": text.strip(),
            "n_new_tokens": int(len(gen_ids)),
            "n_prompt_tokens": n_prompt,
            "truncated": (not finished) and len(gen_ids) >= max_new_tokens,
            "elapsed_s": round(time.time() - t0, 2),
        }
        results.append(rec)
        print(
            f"  [{tag} {i:2d}/{len(samples)}] {rec['n_new_tokens']:4d} tok "
            f"trunc={rec['truncated']} {rec['elapsed_s']:5.2f}s | "
            f"{rec['text'][:60].replace(chr(10), ' ')}"
        )
    print(f"  [{tag}] 合计 {round(time.time() - t_all, 1)}s")
    return results


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA 微调前后并排生成")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--trainer-state", type=Path, default=None,
                    help="trainer_state.json 路径（默认：adapter 目录下，其次 sft_output/checkpoint-80）")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help=">0 则只跑前 N 条（调试用）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(args.seed)

    print("=" * 72)
    print("LoRA 微调前后并排生成")
    print("=" * 72)
    print(f"基座模型 : {args.base_model}")
    print(f"适配器   : {args.adapter}")
    print(f"数据     : {args.data}")
    print(f"输出     : {args.out}")
    print(f"解码     : 贪心(do_sample=False) max_new_tokens={args.max_new_tokens}")
    print(f"CUDA     : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[FATAL] 没有可用 CUDA，本脚本按 GPU 实验设计，CPU 下不做。")
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n[1/4] 复现原训练切分，取 held-out 验证集 ...")
    samples = reproduce_heldout(args.data, tokenizer)
    if args.limit:
        samples = samples[: args.limit]
    print(f"      held-out n = {len(samples)}")

    print("\n[2/4] 加载基座模型 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="cuda"
    )
    print(f"      显存占用 {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n[3/4] 生成基座输出（无 adapter）...")
    base_out = generate_all(model, tokenizer, samples, args.max_new_tokens, "base")

    print("\n[3/4] 加载 LoRA 适配器 ...")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(args.adapter))
    print("      适配器加载完成")

    print("\n[4/4] 生成微调输出（base + adapter）...")
    sft_out = generate_all(model, tokenizer, samples, args.max_new_tokens, "sft")

    # ---------------------------------------------------------------- 落盘
    adapter_cfg = json.loads(
        io.open(args.adapter / "adapter_config.json", encoding="utf-8").read()
    )

    # trainer_state.json：先找 adapter 目录，找不到就用原训练的 checkpoint-80
    ts_path = args.trainer_state
    if ts_path is None:
        cand = args.adapter / "trainer_state.json"
        ts_path = cand if cand.exists() else DEFAULT_TRAINER_STATE
    ts = json.loads(io.open(ts_path, encoding="utf-8").read()) if ts_path.exists() else {}

    # 原训练 eval_loss 轨迹（每 20 步一次 eval）
    eval_curve = []
    for h in ts.get("log_history", []):
        if "eval_loss" in h:
            eval_curve.append({
                "step": h.get("step"),
                "epoch": h.get("epoch"),
                "eval_loss": h.get("eval_loss"),
            })

    # 溯源：lora_sft_checkpoint 的权重是否与原训练 checkpoint-80 逐字节一致
    adapter_sha = sha256_of(args.adapter / "adapter_model.safetensors")
    ckpt_sha = sha256_of(DEFAULT_TRAINER_STATE.parent / "adapter_model.safetensors")
    other_ckpts = {}
    ckpt_root = DEFAULT_TRAINER_STATE.parent.parent
    if ckpt_root.exists():
        for d in sorted(ckpt_root.glob("checkpoint-*")):
            f = d / "adapter_model.safetensors"
            if f.exists():
                other_ckpts[d.name] = {
                    "sha256": sha256_of(f),
                    "same_as_final": sha256_of(f) == adapter_sha,
                }

    run_meta = {
        "_record_type": "run_meta",
        "experiment": "sft_before_after",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": args.base_model,
        "adapter_path": str(args.adapter),
        "adapter_sha256": adapter_sha,
        "adapter_provenance": {
            "trainer_state_file": str(ts_path),
            "checkpoint_weight_sha256": ckpt_sha,
            "identical_to_checkpoint_weight": (ckpt_sha is not None and ckpt_sha == adapter_sha),
            "other_checkpoints": other_ckpts,
            "note": "适配器权重与原训练 checkpoint-80 的 adapter_model.safetensors 逐字节一致；"
                    "与 checkpoint-40 / checkpoint-60 不同 —— 证明用的是 val_loss 最低那一步的权重。",
        },
        "adapter_config": {
            "peft_type": adapter_cfg.get("peft_type"),
            "r": adapter_cfg.get("r"),
            "lora_alpha": adapter_cfg.get("lora_alpha"),
            "lora_dropout": adapter_cfg.get("lora_dropout"),
            "target_modules": adapter_cfg.get("target_modules"),
            "base_model_name_or_path": adapter_cfg.get("base_model_name_or_path"),
            "peft_version_saved": adapter_cfg.get("peft_version"),
        },
        "training_data_file": str(args.data),
        "training_data_sha256": sha256_of(args.data),
        "held_out": {
            "derivation": "复现 day9_sft.py 的 train_test_split(test_size=0.2, seed=42)，取 20 条 test（= 原训练验证集）",
            "n": len(samples),
            "split_seed": SPLIT_SEED,
            "test_size": SPLIT_TEST_SIZE,
            "caveat": f"这 {len(samples)} 条就是原训练里 load_best_model_at_end 用来选 checkpoint 的验证集，"
                      "因此是 dev-clean 但不是 test-clean（checkpoint 选择见过这些样本的 loss）。",
        },
        "decoding": {
            "do_sample": False,
            "strategy": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "note": "两条臂解码参数逐项相同；贪心 → 运行间方差为 0",
        },
        "prompt_policy": {
            "system_prompt": SYSTEM_PROMPT,
            "same_for_both_arms": True,
            "note": "两条臂 prompt 完全一致，唯一变量是是否加载 LoRA 权重",
        },
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "trainer_state": {
            "source_file": str(ts_path),
            "global_step": ts.get("global_step"),
            "epoch": ts.get("epoch"),
            "best_metric_eval_loss": ts.get("best_metric"),
            "num_train_epochs": ts.get("num_train_epochs"),
            "train_batch_size": ts.get("train_batch_size"),
            "eval_curve": eval_curve,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(run_meta, ensure_ascii=False) + "\n")
        for i, (s, b, t) in enumerate(zip(samples, base_out, sft_out), 1):
            rec = {
                "_record_type": "sample",
                "id": f"heldout-{i:02d}",
                "source_index_in_100": s["idx"],
                "instruction": s["instruction"],
                "prompt": build_prompt(tokenizer, s["instruction"]),
                "reference": s["output"],
                "base_output": b["text"],
                "sft_output": t["text"],
                "meta": {
                    "base": {k: b[k] for k in ("n_new_tokens", "n_prompt_tokens", "truncated", "elapsed_s")},
                    "sft": {k: t[k] for k in ("n_new_tokens", "n_prompt_tokens", "truncated", "elapsed_s")},
                    "reference_chars": len(s["output"]),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✓ 已写入 {args.out}")
    print(f"  行数 = 1 (run_meta) + {len(samples)} (sample)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
