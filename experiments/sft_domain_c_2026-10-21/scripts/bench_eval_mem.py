#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测批大小 → 显存峰值 / 吞吐 / 对后续训练速度的影响（定位 26 s/step 的根因）。"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 可增长的显存段，减少碎片
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

M = "Qwen/Qwen2.5-1.5B-Instruct"
SEQ = 576
tok = AutoTokenizer.from_pretrained(M)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float16, device_map="cuda")
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM))
model.train()


def train_steps(n, label):
    ids = torch.randint(0, 140000, (1, SEQ), device="cuda")
    labels = ids.clone()
    labels[:, : SEQ // 2] = -100
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"  {label}: {dt*1000:7.1f} ms/micro-step  "
          f"-> {dt*2:5.2f} s/optimizer-step   peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return dt


def evaluate(bs):
    ids = torch.randint(0, 140000, (bs, SEQ), device="cuda")
    labels = ids.clone()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  eval bs={bs:2d}: {dt:6.2f} s/batch -> {dt*200/bs:6.1f} s/200样本   "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB   "
          f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")


print("① 训练基线（无 eval）")
train_steps(3, "warmup")
base = train_steps(20, "训练")

for bs in (8, 4, 2):
    print(f"\n② eval bs={bs} 前后对比")
    evaluate(bs)
    after = train_steps(20, "eval 后训练")
    print(f"  → 退化倍数 = {after/base:.1f}×")

print(f"\n最终 GPU 保留显存: {torch.cuda.memory_reserved()/1e9:.2f} GB "
      f"(上限约 8 GB)")
