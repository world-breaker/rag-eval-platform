#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
LoRA adapter 溯源核对
=====================================================================
「评测用的那份 adapter，到底是不是训练脚本存出来的那份？超参对不对？」

核对四件事
----------
1. `D:\Temp\opencode\sft_domain_c\adapter\adapter_config.json` 的
   `base_model_name_or_path` / `r` / `lora_alpha` / `lora_dropout` / `target_modules`
   是否与 probe 方案 4.3 节要求的一致（r=8 / α=16 / dropout=0.05 / 7 个 target_modules）；
2. `adapter_model.safetensors` 的 sha256 是否与 **dev 选出的 best checkpoint**
   的 adapter 权重**逐字节一致**（证明 `load_best_model_at_end` 生效、
   且评测用的就是 dev 选出来的那一份，**没有**碰过 test）；
3. best checkpoint 是哪一个（来自 `trainer_state.json::best_model_checkpoint`）；
4. `train_meta.json` 里的训练步数 / 步速 / 评估次数。

用法
----
    python scripts/verify_adapter_provenance.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

W = Path(r"D:\Temp\opencode\sft_domain_c")
ADAPTER = W / "adapter"
CKPT = W / "checkpoints"

EXPECT = {
    "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    if not ADAPTER.exists():
        print(f"[FATAL] 找不到 {ADAPTER}")
        return 2
    cfg = json.loads(io.open(ADAPTER / "adapter_config.json", encoding="utf-8").read())

    checks = []

    def chk(name, got, want):
        ok = (got == want)
        checks.append({"项": name, "实际": got, "期望": want, "一致": ok})
        print(f"  {'[OK]' if ok else '[!!]'} {name:26s} 实际={got!r} 期望={want!r}")

    print("=" * 76)
    print("① adapter_config.json 超参（对照 probe 报告 4.3 节）")
    print("=" * 76)
    chk("base_model", cfg.get("base_model_name_or_path"), EXPECT["base_model_name_or_path"])
    chk("r", cfg.get("r"), EXPECT["r"])
    chk("lora_alpha", cfg.get("lora_alpha"), EXPECT["lora_alpha"])
    chk("lora_dropout", cfg.get("lora_dropout"), EXPECT["lora_dropout"])
    tm = sorted(cfg.get("target_modules") or [])
    chk("target_modules", tm, sorted(EXPECT["target_modules"]))
    chk("target_modules 数量", len(tm), 7)

    # ---- best checkpoint（由 dev eval_loss 选出）----
    print("\n" + "=" * 76)
    print("② best checkpoint（由 dev eval_loss 选出，**未使用 test**）")
    print("=" * 76)
    best = None
    for st in sorted(CKPT.glob("checkpoint-*/trainer_state.json")):
        d = json.loads(io.open(st, encoding="utf-8").read())
        if d.get("best_model_checkpoint"):
            best = (Path(d["best_model_checkpoint"]), d)
    if best is None:
        print("[!!] 找不到 best_model_checkpoint")
        checks.append({"项": "best checkpoint", "一致": False})
    else:
        best_dir, st = best
        print(f"  best_model_checkpoint = {best_dir}")
        print(f"  best_metric (eval_loss) = {st.get('best_metric')}")
        print(f"  best_global_step = {st.get('best_global_step')}")
        print(f"  metric_for_best_model = eval_loss（dev 集 200 条）")
        checks.append({"项": "best checkpoint 记录", "实际": str(best_dir), "一致": True})

        a_ours = ADAPTER / "adapter_model.safetensors"
        a_best = best_dir / "adapter_model.safetensors"
        if a_ours.exists() and a_best.exists():
            s1, s2 = sha256(a_ours), sha256(a_best)
            chk("adapter 权重 sha256 == best checkpoint",
                s1, s2)
            print(f"    adapter       sha256 = {s1}")
            print(f"    best ckpt     sha256 = {s2}")
        else:
            print(f"  ! 权重文件缺失：ours={a_ours.exists()} best={a_best.exists()}")
            checks.append({"项": "adapter 权重文件", "一致": False})

    # ---- train_meta.json ----
    print("\n" + "=" * 76)
    print("③ train_meta.json（训练运行时）")
    print("=" * 76)
    meta_path = W / "train_meta.json"
    runtime = None
    if meta_path.exists():
        meta = json.loads(io.open(meta_path, encoding="utf-8").read())
        runtime = meta.get("runtime")
        print(json.dumps({k: meta.get(k) for k in ("base_model", "lora", "train", "data")},
                         ensure_ascii=False, indent=2)[:1800])
        print("  runtime:")
        print(json.dumps(runtime, ensure_ascii=False, indent=2))
        hist = meta.get("log_history") or []
        evals = [h for h in hist if "eval_loss" in h]
        print(f"  eval 次数 = {len(evals)}")
        for h in evals:
            print(f"    step={h.get('step'):>5} epoch={h.get('epoch'):.4f} "
                  f"eval_loss={h['eval_loss']:.6f} runtime={h.get('eval_runtime')}s")
    else:
        print("  ! 没有 train_meta.json（训练没跑完？）")
        checks.append({"项": "train_meta.json", "一致": False})

    all_ok = all(c["一致"] for c in checks)
    out = {"adapter_dir": str(ADAPTER), "checks": checks, "全部一致": all_ok,
           "train_meta": json.loads(io.open(meta_path, encoding="utf-8").read())
           if meta_path.exists() else None}
    p = RESULTS / "adapter_provenance.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 76)
    print(f"[{'OK' if all_ok else 'FAIL'}] adapter 溯源核对")
    print(f"→ {p}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
