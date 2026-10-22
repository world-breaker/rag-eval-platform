#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
双臂评测：基座 vs LoRA 微调后（唯一变量 = 是否装 LoRA）
=====================================================================

口径（与 probe 的 run_base.py **逐项相同**，保证基座数字与 probe 的 41.7% 可比）：
  - 基座模型：Qwen/Qwen2.5-1.5B-Instruct（fp16, CUDA）
  - 解码：贪心 do_sample=False、temperature=None、top_p=None、top_k=None
  - max_new_tokens=512
  - system message：与 probe 同一句
  - **逐条生成（不 batch）**，顺序与数据文件一致

两臂：
  base : 不装 LoRA
  lora : 同一个基座 + 训练好的 adapter（同一份 checkpoint，同一解码参数）

评测集：
  test  200 条（正式结论）
  probe 24 条（probe 的人工撰写样本，**不是生成器产物** → 作为「生成器外」的第二组证据，
                同时验证基座数字能否复现 probe 的 41.7% = 10/24）

用法
----
    python scripts/run_eval.py --arm base --data test
    python scripts/run_eval.py --arm lora --data test
    python scripts/run_eval.py --arm both --data test,probe
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = Path(r"D:\Temp\opencode\sft_domain_c\adapter")
PROBE_DATA = Path(r"D:\programmer\LLM\projects\rag-eval-platform\experiments"
                   r"\domain_probe_2026-10-21\data\C_compliance_refusal.jsonl")

# 与 probe 的 run_base.py 逐字相同
SYSTEM_PROMPT = (
    "你是一个严谨的领域助手。请严格按用户消息中的要求作答，"
    "只输出要求的格式，不要输出任何多余内容。"
)


def read_jsonl(path: Path):
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_data(name: str):
    if name == "test":
        return read_jsonl(DATA / "test.jsonl"), DATA / "test.jsonl"
    if name == "probe":
        return read_jsonl(PROBE_DATA), PROBE_DATA
    raise SystemExit(f"未知数据集 {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["base", "lora", "both"], default="both")
    ap.add_argument("--data", default="test,probe")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("[FATAL] 无 CUDA")
        return 2
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for arm in (["base", "lora"] if args.arm == "both" else [args.arm]):
        if arm == "lora":
            from peft import PeftModel
            if not ADAPTER.exists():
                print(f"[FATAL] 找不到 adapter {ADAPTER}")
                return 2
            model = PeftModel.from_pretrained(model, str(ADAPTER))
            model.eval()
            print(f"[lora] 已加载 adapter: {ADAPTER}")

        for dname in [x.strip() for x in args.data.split(",") if x.strip()]:
            rows, src = load_data(dname)
            out_path = args.out_dir / f"{dname}_{arm}_outputs.jsonl"
            print(f"\n[{arm}/{dname}] n={len(rows)} → {out_path}")
            t_all = time.time()
            out_lines = []
            for i, r in enumerate(rows, 1):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["prompt"]},
                ]
                chat = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                enc = tokenizer(chat, return_tensors="pt").to(model.device)
                n_prompt = int(enc["input_ids"].shape[1])
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen_ids = out[0][n_prompt:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                finished = bool(len(gen_ids) > 0 and int(gen_ids[-1]) == tokenizer.eos_token_id)
                rec = {
                    "_record_type": "sample",
                    "id": r["id"],
                    "prompt": r["prompt"],
                    "raw_output": text.strip(),
                    "n_new_tokens": int(len(gen_ids)),
                    "n_prompt_tokens": n_prompt,
                    "truncated": (not finished) and len(gen_ids) >= args.max_new_tokens,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                out_lines.append(rec)
                print(f"  [{i:3d}/{len(rows)}] {rec['n_new_tokens']:4d}tok "
                      f"trunc={rec['truncated']} {rec['elapsed_s']:5.2f}s | "
                      f"{rec['raw_output'][:80].replace(chr(10), ' ')}", flush=True)

            meta = {
                "_record_type": "run_meta",
                "experiment": "sft_domain_c_2026-10-21",
                "dataset": dname,
                "arm": arm,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_model": BASE_MODEL,
                "adapter": str(ADAPTER) if arm == "lora" else None,
                "decoding": {
                    "do_sample": False, "strategy": "greedy",
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": None, "top_p": None, "top_k": None,
                },
                "system_prompt": SYSTEM_PROMPT,
                "n": len(rows),
                "elapsed_total_s": round(time.time() - t_all, 1),
                "env": {
                    "python": platform.python_version(), "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                },
                "source_data_file": str(src),
            }
            with io.open(out_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                for rec in out_lines:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [{arm}/{dname}] 写入 {out_path}  用时 {round(time.time() - t_all, 1)}s")

        if arm == "lora" and args.arm == "both":
            # peft 0.19 里 PeftModel 没有 unload()，只有 LoraModel 有；
            # base 臂在 lora 臂之前跑完，这里只是为了保险地拆掉 adapter。
            try:
                model = model.base_model.unload()
                print("[lora] 已卸载 adapter，回到基座")
            except Exception as exc:
                print(f"[lora] 卸载 adapter 跳过（不影响结果）：{exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
