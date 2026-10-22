#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只跑 eval bs=2 的干净进程：训练速度会不会在 eval 之后塌掉？"""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

M, SEQ = "Qwen/Qwen2.5-1.5B-Instruct", 576
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

ids = torch.randint(0, 140000, (1, SEQ), device="cuda")
labels = ids.clone()
labels[:, : SEQ // 2] = -100


def train_steps(n):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"  训练 {n} 步: {dt*1000:7.1f} ms/micro-step -> {dt*2:5.2f} s/optimizer-step  "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB  "
          f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")
    return dt


def evaluate(bs):
    eids = torch.randint(0, 140000, (bs, SEQ), device="cuda")
    elabels = eids.clone()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        model(input_ids=eids, labels=elabels)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  eval bs={bs}: {dt:6.2f} s/batch -> {dt*200/bs:6.1f} s/200样本  "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB  "
          f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")


train_steps(3)
base = train_steps(20)
for i in range(3):
    evaluate(2)
    after = train_steps(20)
    print(f"  → 第 {i+1} 次 eval 后退化倍数 = {after/base:.2f}×")
