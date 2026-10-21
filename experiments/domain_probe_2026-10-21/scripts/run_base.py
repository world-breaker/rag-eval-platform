#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跑基座模型 Qwen/Qwen2.5-1.5B-Instruct，产出各域原始输出
=========================================================

口径（与上一轮 experiments/run_sft_before_after.py **完全一致**，保证可比）：
  - 基座模型：Qwen/Qwen2.5-1.5B-Instruct（本地 HF 缓存）
  - 解码：贪心 do_sample=False，temperature=None，top_p=None，top_k=None
  - max_new_tokens=512
  - 逐条生成（不 batch），顺序与数据文件一致

system message 固定为 4 个域**同一句**，域内差异只来自 user 消息。

输出
----
  results/<domain>_base_outputs.jsonl
      第 1 行 {"_record_type": "run_meta", ...}
      第 2..N 行 {"_record_type":"sample", "id", "prompt", "raw_output", ...}

用法
----
  python scripts/run_base.py                 # 跑全部 4 个域
  python scripts/run_base.py --domains A     # 只跑 A
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

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = (
    "你是一个严谨的领域助手。请严格按用户消息中的要求作答，"
    "只输出要求的格式，不要输出任何多余内容。"
)

DOMAIN_FILES = {
    "A": "A_agent_tool_call",
    "B": "B_doc_extraction",
    "C": "C_compliance_refusal",
    "D": "D_intent_slot",
}


def read_jsonl(path: Path):
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--domains", default="A,B,C,D")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    # 本机代理会把 HF 请求导向本地代理导致失败；模型已缓存 → 强制离线
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(0)

    print("=" * 72)
    print("领域 probe —— 基座模型原始输出")
    print("=" * 72)
    print(f"模型   : {args.model}")
    print(f"解码   : 贪心(do_sample=False) max_new_tokens={args.max_new_tokens} temperature=None")
    print(f"system : {SYSTEM_PROMPT}")
    print(f"CUDA   : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[FATAL] 无 CUDA，退出。")
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    print(f"显存   : {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key in [x.strip().upper() for x in args.domains.split(",") if x.strip()]:
        name = DOMAIN_FILES[key]
        data_path = DATA / f"{name}.jsonl"
        rows = read_jsonl(data_path)
        if args.limit:
            rows = rows[: args.limit]
        out_path = args.out_dir / f"{name}_base_outputs.jsonl"

        print(f"\n[{key}] {name}  n={len(rows)} -> {out_path}")
        t_all = time.time()
        out_lines = []
        for i, r in enumerate(rows, 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["prompt"]},
            ]
            chat = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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
            print(f"  [{i:2d}/{len(rows)}] {rec['n_new_tokens']:4d}tok trunc={rec['truncated']} "
                  f"{rec['elapsed_s']:5.2f}s | {rec['raw_output'][:70].replace(chr(10), ' ')}")

        meta = {
            "_record_type": "run_meta",
            "experiment": "domain_probe_2026-10-21",
            "domain": name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_model": args.model,
            "decoding": {
                "do_sample": False,
                "strategy": "greedy",
                "max_new_tokens": args.max_new_tokens,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            },
            "system_prompt": SYSTEM_PROMPT,
            "n": len(rows),
            "elapsed_total_s": round(time.time() - t_all, 1),
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "source_data_file": str(data_path),
        }
        with io.open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for rec in out_lines:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [{key}] 写入 {out_path}  用时 {round(time.time() - t_all, 1)}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
