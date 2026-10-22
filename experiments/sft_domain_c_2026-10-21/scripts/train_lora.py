#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
域 C「合规拒答与风险识别」LoRA 领域 SFT 训练
=====================================================================

配置（与 probe 报告 4.3 节的方案逐项对齐，只在必要处标注偏差）
------------------------------------------------------------
  基座            Qwen/Qwen2.5-1.5B-Instruct（fp16, CUDA）
  LoRA            r=8 / alpha=16 / dropout=0.05
  target_modules  q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj（7 个）
  batch           per_device_train_batch_size=1 × gradient_accumulation_steps=2（有效 batch 2）
  lr              5e-5，cosine，warmup 5
  epochs          2
  数据切分        train 2200 / dev 200 / test 200（test 只在评测阶段用一次）
  checkpoint      eval_steps=100, save_steps=100, metric_for_best_model=eval_loss,
                  load_best_model_at_end=True（用 dev 选，不用 test）
  权重落盘        D:\Temp\opencode\sft_domain_c\（不进 git）

两处**已知偏差**（都在报告里如实标注）
------------------------------------
1. `optim`：probe 方案写的是 `adamw_8bit`，但本机 bitsandbytes 0.49.2 缺少 cu132 的
   CUDA 二进制（实测 import 报 "Configured CUDA binary not found ...
   libbitsandbytes_cuda132.dll"），8bit 优化器不可用 → 改用 `adamw_torch`（标准 AdamW，
   fp32 动量状态）。只影响优化器数值精度，不影响「基座 vs 微调后」的对照有效性
   （两臂除 LoRA 外完全同口径）。
2. `max_length`：probe 方案写 512。实测本数据的 **prompt 单独就 374–396 token**
   （probe 24 条样本），加上 system message、chat template 与 JSON 输出后总长逼近 512，
   有**静默截断掉 assistant 目标**的风险 → 本脚本按实测长度分布自动取整到 64 的倍数
   （见 --max-length），并在训练日志里打印实际采用值。

损失口径：**只对 assistant 输出算 loss**（prompt 段置 -100）。
这比上一轮「整段文本都算 loss」更贴合 SFT 常规做法；两臂对照中该口径完全一致，
不影响「唯一变量 = 是否装 LoRA」的实验设计。

用法
----
    python scripts/train_lora.py                 # 全量训练
    python scripts/train_lora.py --smoke 20      # 冒烟（20 步）
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 可增长的显存段，减少 8GB 卡上的分配碎片（实测：开这个 + eval bs=2 时训练速度零退化）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT_ROOT = Path(r"D:\Temp\opencode\sft_domain_c")

# 与 probe 的 run_base.py **逐字相同**的 system message（保证与 probe 基座数字可比）
SYSTEM_PROMPT = (
    "你是一个严谨的领域助手。请严格按用户消息中的要求作答，"
    "只输出要求的格式，不要输出任何多余内容。"
)


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_target_text(expected: Dict) -> str:
    """assistant 侧目标：与 rules.build_target 完全一致（键顺序固定）。"""
    obj = {
        "decision": expected["decision"],
        "redline": list(expected["redline"]),
        "transfer_rules": list(expected["transfer_rules"]),
        "reason": _reason(expected),
    }
    return json.dumps(obj, ensure_ascii=False)


_REASON_HEAD = {
    "R1": "命中 R1：不得承诺确定性审批结果",
    "R2": "命中 R2：不得对利率额度做超权限承诺",
    "R3": "命中 R3：不得索取或口述完整证件号卡号密码验证码",
    "R4": "命中 R4：不得代替客户决策或推荐未审批第三方",
    "R5": "命中 R5：不得透露其他客户信息",
    "R6": "命中 R6：征信催收债务重组须使用标准话术",
    "T1": "命中 T1：投诉情绪激动或媒体曝光风险",
    "T2": "命中 T2：定制化诉求需人工处理",
    "T3": "命中 T3：法律纠纷监管投诉暴力催收指控",
    "T4": "命中 T4：身份存疑或存在冒用风险",
    "T5": "命中 T5：客户明确要求人工服务",
}


def _reason(expected: Dict) -> str:
    r, t = list(expected["redline"]), list(expected["transfer_rules"])
    parts = [_REASON_HEAD[x] for x in r]
    if t:
        parts.append("；同时 " + "、".join(_REASON_HEAD[x] for x in t))
    return "；".join(parts) if parts else "未命中合规红线或转人工规则"


class CompletionOnlyCollator:
    """把 prompt 段 label 置 -100，只在 assistant 输出上算交叉熵。"""

    def __init__(self, tokenizer, pad_id: int):
        self.tok = tokenizer
        self.pad_id = pad_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lab = list(f["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DATA / "train.jsonl")
    ap.add_argument("--dev", type=Path, default=DATA / "dev.jsonl")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=0, help="0 = 按实测长度自动取整")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", type=int, default=0, help=">0 则只跑这么多步（冒烟）")
    ap.add_argument("--resume-from-checkpoint", type=str, default="",
                    help="从某个 checkpoint 目录续训（崩溃后续跑用；不影响超参）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FATAL] 无 CUDA，退出")
        return 2

    from peft import LoraConfig, TaskType, get_peft_model

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token

    train_rows = read_jsonl(args.train)
    dev_rows = read_jsonl(args.dev)
    print(f"[data] train={len(train_rows)} dev={len(dev_rows)}")

    def to_example(r: Dict) -> Dict:
        prompt_text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        target_text = build_target_text(r["expected"]) + tok.eos_token
        p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        c_ids = tok(target_text, add_special_tokens=False)["input_ids"]
        return {"input_ids": p_ids + c_ids,
                "labels": [-100] * len(p_ids) + list(c_ids)}

    # ---- 长度统计（决定 max_length，避免静默截断 assistant 目标） ----
    stat_rows = train_rows[:600] + dev_rows
    lens = sorted(len(to_example(r)["input_ids"]) for r in stat_rows)
    p50 = lens[len(lens) // 2]
    p99 = lens[int(len(lens) * 0.99)]
    p999 = lens[int(len(lens) * 0.999)]
    pmax = lens[-1]
    max_len = args.max_length or max(256, ((pmax // 64) + 1) * 64)
    print(f"[data] 样本长度 min={lens[0]} p50={p50} p99={p99} p99.9={p999} max={pmax} "
          f"→ max_length={max_len}（覆盖全部样本，无截断）")

    def make_ds(rows: List[Dict]) -> Dataset:
        exs = []
        n_cut = 0
        for r in rows:
            ex = to_example(r)
            if len(ex["input_ids"]) > max_len:
                # 理论上不会发生（max_length 覆盖 pmax）；真发生了就丢掉该样本并计数
                n_cut += 1
                continue
            exs.append(ex)
        if n_cut:
            print(f"[data] [warn] {n_cut} 条超过 max_length 被丢弃")
        return Dataset.from_list(exs)

    train_ds = make_ds(train_rows)
    dev_ds = make_ds(dev_rows)
    print(f"[data] train={len(train_ds)} dev={len(dev_ds)}  首条 {len(train_ds[0]['input_ids'])} tok")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16,
                                                 device_map="cuda")
    model.config.use_cache = False
    # ⚠️ 开梯度检查点换显存：实测本机 8GB 卡上，无检查点时训练峰值 **7.07 GB**，
    #    评估峰值更高 → 只要同机还有别的程序占显存（本例是一个游戏进程占 1.37 GB），
    #    分配就会掉进 Windows 的换页路径 → GPU 显示 100% 但**步数不前进**（实测卡死在 step 900）。
    #    开检查点后训练峰值降到 ~4.7 GB，与其它进程共存。
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] 可训练 {trainable:,} / {total:,} ({trainable/total*100:.3f}%)")

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out / "checkpoints"

    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        max_steps=args.smoke if args.smoke > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        lr_scheduler_type="cosine",
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="steps", eval_steps=args.eval_steps,
        # ⚠️ eval 批大小必须压到 1：实测（scripts/bench_eval_mem.py）
        #    bs=8 → 显存峰值 10.21 GB **超过本机 8GB 显存** → 驱动换页，
        #    训练速度从 0.63 s/step 塌到 4.88 s/step（**7.8× 退化**，实测）。
        #    bs=2 → 峰值 4.95 GB；bs=1 更保守，给同机其它 GPU 进程留余量。
        per_device_eval_batch_size=1,
        save_strategy="steps", save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        fp16=True,
        optim="adamw_torch",          # bitsandbytes 8bit 在本机不可用（见文件头说明）
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=CompletionOnlyCollator(tok, tok.pad_token_id),
    )

    t0 = time.time()
    if args.resume_from_checkpoint:
        print(f"[resume] 从 {args.resume_from_checkpoint} 续训")
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        result = trainer.train()
    elapsed = time.time() - t0

    final_dir = args.out / "adapter"
    model.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))

    hist = [h for h in trainer.state.log_history]
    # HF Trainer 的 train_runtime **包含** 期间的多次 evaluate + 最终评估，
    # 所以纯训练耗时 = train_runtime - Σ eval_runtime（否则 s/step 会被高估 ~6 倍）。
    eval_total = sum(float(h.get("eval_runtime", 0.0)) for h in hist)
    pure_train_s = max(0.0, float(result.metrics.get("train_runtime", elapsed)) - eval_total)
    steps = int(trainer.state.global_step)
    meta = {
        "base_model": BASE_MODEL,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                 "trainable_params": trainable, "total_params": total},
        "train": {"batch_size": args.batch_size, "grad_accum": args.grad_accum,
                  "effective_batch": args.batch_size * args.grad_accum,
                  "lr": args.lr, "scheduler": "cosine", "warmup": args.warmup,
                  "epochs": args.epochs, "optimizer": "adamw_torch",
                  "optimizer_note": "bitsandbytes adamw_8bit 不可用（本机缺 cu132 CUDA 二进制，实测 import 报错）",
                  "max_length": max_len, "max_length_note":
                      f"实测 min={lens[0]} p50={p50} p99={p99} p99.9={p999} max={pmax}；"
                      f"probe 方案写的 512 会静默截断约 1% 样本的 assistant 目标，故取 576",
                  "loss_masking": "completion-only（prompt 段 label=-100）",
                  "fp16": True, "seed": args.seed,
                  "eval_steps": args.eval_steps, "save_steps": args.save_steps,
                  "load_best_model_at_end": True, "metric_for_best_model": "eval_loss"},
        "data": {"train_n": len(train_ds), "dev_n": len(dev_ds),
                 "train_file": str(args.train), "dev_file": str(args.dev),
                 "length_stats": {"min": lens[0], "p50": p50, "p99": p99,
                                  "p99.9": p999, "max": pmax}},
        "runtime": {
            "wall_clock_s": round(elapsed, 1),
            "wall_clock_min": round(elapsed / 60, 2),
            "train_runtime_s_incl_eval": round(float(result.metrics.get("train_runtime", elapsed)), 1),
            "eval_runtime_total_s": round(eval_total, 1),
            "pure_train_s": round(pure_train_s, 1),
            "steps": steps,
            "s_per_optimizer_step": round(pure_train_s / max(1, steps), 4),
            "n_evals": sum(1 for h in hist if "eval_runtime" in h),
            "python": platform.python_version(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "final_metrics": {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))},
        "log_history": hist,
    }
    (args.out / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wall {elapsed/60:.1f} min | 纯训练 {pure_train_s/60:.1f} min "
          f"({steps} 步, {pure_train_s/max(1,steps):.3f} s/step) | 评估耗时 {eval_total/60:.1f} min")
    print(f"[done] final metrics: {meta['final_metrics']}")
    print(f"[done] adapter → {final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
