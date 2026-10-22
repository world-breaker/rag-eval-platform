#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练吞吐探针：定位 s/step 异常的原因。"""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

M = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
tok.pad_token = tok.eos_token

print("torch", torch.__version__)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))

model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float16, device_map="cuda")
print("attn_implementation:", getattr(model.config, "_attn_implementation", "?"))
print("use_cache:", model.config.use_cache)
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM))
model.train()

for seq in (256, 576, 768):
    ids = torch.randint(0, 140000, (1, seq), device="cuda")
    labels = ids.clone()
    labels[:, : seq // 2] = -100
    # warmup
    for _ in range(2):
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    N = 5
    for _ in range(N):
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"seq={seq:4d}  fwd+bwd {dt*1000:8.1f} ms/micro-step  "
          f"→ {dt*2:6.2f} s/optimizer-step(eff bs=2)  peak_mem={mem:.2f} GB")

print("--- SDPA vs eager ---")
import transformers.models.qwen2.modeling_qwen2 as qm  # noqa: E402
print("qwen2 supports sdpa:", hasattr(qm, "Qwen2SdpaAttention"))
