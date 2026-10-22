#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
LLM 改写环节：只改写**问句**，标签永远由规则算
=====================================================================

纪律
----
1. **LLM 不产生标签**。模型只被要求把问句改得更口语化；它拿到的「必须原样保留」的
   片段列表来自规则引擎（`rules.matched_fragments`）。
2. 改写结果**必须重新跑规则引擎**，`compute_label` 与原标签完全一致才采用；
   不一致 → 丢弃并重采样（最多 3 次）；3 次都失败 → 标记 ok=False，
   最终数据里回退到原始模板问句（并如实记账）。
3. 结果写进 **缓存** `data/llm_rewrite_cache.json`；重跑本脚本不重复调用 API，
   因此 `build_sft_data.py --stage finalize` 的产物字节一致。
4. API key 只从**环境变量**读（`DEEPSEEK_API_KEY` 或 `ANTHROPIC_AUTH_TOKEN`），不落盘。

用法
----
    python scripts/rewrite_with_llm.py                # 跑全部待改写槽位
    python scripts/rewrite_with_llm.py --limit 20     # 先小样本试跑
    python scripts/rewrite_with_llm.py --workers 8
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

sys.path.insert(0, str(HERE))
from rules import compute_label, label_key, matched_fragments  # noqa: E402
from build_sft_data import (  # noqa: E402
    split_candidates, plan_rewrite, load_probe_questions, ngrams, jaccard,
)

# probe 的 24 条问句 → 4-gram，改写结果与它们的相似度必须 ≤ 0.5
PROBE_NG = [ngrams(q) for q in load_probe_questions()]

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
MAX_ATTEMPTS = 3
TIMEOUT = 90

# 价格（美元 / 1M token）—— 来源：https://api-docs.deepseek.com/quick_start/pricing/
#   （2026-10-21 实测抓取；deepseek-flash = DeepSeek-V4.1-Flash-1017）
#   off-peak（默认，非工作日 01:00-04:00 & 06:00-10:00 UTC 时段）：
#     输入 cache-miss $0.15 / cache-hit $0.003 / 输出 $0.60
#   peak：输入 cache-miss $0.30 / cache-hit $0.006 / 输出 $1.20
PRICE_IN = 0.15
PRICE_IN_CACHE = 0.003
PRICE_OUT = 0.60
USD_CNY = 7.1

SYS_PROMPT = (
    "你是金融客服语料标注助手。你的唯一任务是**改写**一句客户问句，"
    "让它更像是真实客户在聊天框里打出来的话。"
    "严禁：改变客户诉求、新增任何请求、回答这个问题、输出解释或任何多余文字。"
)

USER_TMPL = """把下面这句金融客户问句改写成更自然口语的表达。

硬性要求：
1. 可以调整语序、语气、称呼，可以加客套话和情绪词，让它像真实客户打的字。
2. **必须原样保留**下面这些关键片段（一个字都不能改、不能删、不能换）：
{keeps}
3. 不要改变客户的诉求方向，不要新增任何新的请求或新的合规风险。
4. 只输出改写后的那一句问句，不要输出解释、引号、编号或任何其他内容。

原句：{q}"""


class Usage:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.in_tok = 0
        self.in_cache = 0
        self.out_tok = 0
        self.calls = 0
        self.errors = 0

    def add(self, i: int, o: int, err: bool = False, cache_hit: int = 0) -> None:
        with self.lock:
            self.in_tok += i
            self.in_cache += cache_hit
            self.out_tok += o
            self.calls += 1
            if err:
                self.errors += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {"calls": self.calls, "errors": self.errors,
                    "prompt_tokens": self.in_tok, "prompt_cache_hit_tokens": self.in_cache,
                    "completion_tokens": self.out_tok}

    def cost_usd(self) -> float:
        s = self.snapshot()
        miss = max(0, s["prompt_tokens"] - s["prompt_cache_hit_tokens"])
        return (miss / 1e6 * PRICE_IN
                + s["prompt_cache_hit_tokens"] / 1e6 * PRICE_IN_CACHE
                + s["completion_tokens"] / 1e6 * PRICE_OUT)


def api_key() -> str:
    for name in ("DEEPSEEK_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        v = os.environ.get(name)
        if v:
            return v
    raise SystemExit("[FATAL] 环境变量里没有 DEEPSEEK_API_KEY / ANTHROPIC_AUTH_TOKEN")


def call_rewrite(key: str, question: str, keeps: list, usage: Usage, attempt: int) -> str:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": USER_TMPL.format(
                keeps="\n".join(f"- {k}" for k in keeps) if keeps else "- （无特别要求，保留原句主干）",
                q=question)},
        ],
        "temperature": 0.7 if attempt else 0.3,
        "max_tokens": 300,
        # 关掉 thinking：改写任务不需要推理，关掉可避免 reasoning token 吃光预算
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        usage.add(0, 0, err=True)
        raise RuntimeError(f"api_error: {type(exc).__name__}: {exc}")
    u = d.get("usage") or {}
    usage.add(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0),
              cache_hit=int(u.get("prompt_cache_hit_tokens") or 0))
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    text = (msg.get("content") or "").strip()
    # 去掉可能的引号 / 解释性前后缀
    text = text.strip().strip("「」\"'")
    if "\n" in text:
        text = text.split("\n")[0].strip()
    # 收敛重复标点（模型偶尔会输出「？？」）
    for a, b in (("？？", "？"), ("？？？", "？"), ("。。", "。"),
                 ("！！！", "！"), ("，，", "，"), ("？？", "？")):
        text = text.replace(a, b)
    return text


def rewrite_one(key: str, item: dict, usage: Usage) -> dict:
    """对一条候选做「改写 → 规则复核 → 通过才采用」。

    三道闸门（任一不过即丢弃重采样）：
      ① 规则标签必须完全一致（`compute_label` 前后相同）；
      ② 与 probe 的 24 条问句的 4-gram Jaccard 必须 ≤ 0.5（改写不得把样本推向 probe 集）；
      ③ 改写文本必须非空、长度合理。
    """
    ck = item["cache_key"]
    q = item["question"]
    lab = item["label"]
    keeps = matched_fragments(q)
    tries = []
    for attempt in range(MAX_ATTEMPTS):
        try:
            new_q = call_rewrite(key, q, keeps, usage, attempt)
        except Exception as exc:
            tries.append({"attempt": attempt, "error": str(exc)[:200]})
            time.sleep(1.5 * (attempt + 1))
            continue
        if not new_q or len(new_q) < 4:
            tries.append({"attempt": attempt, "error": "empty_output", "out": new_q[:80]})
            continue
        lab2 = compute_label(new_q)
        same = label_key(lab2) == label_key(lab)
        jv = max((jaccard(ngrams(new_q), g) for g in PROBE_NG), default=0.0)
        tries.append({"attempt": attempt, "question": new_q,
                      "label_same": same, "probe_jaccard": round(jv, 4)})
        if same and jv <= 0.5:
            return {"pid": item["pid"], "cache_key": ck, "ok": True, "question": new_q,
                    "keeps": keeps, "attempts": attempt + 1, "tries": tries}
    return {"pid": item["pid"], "cache_key": ck, "ok": False, "question": None,
            "keeps": keeps, "attempts": MAX_ATTEMPTS, "tries": tries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=DATA / "pool.jsonl")
    ap.add_argument("--cache", type=Path, default=DATA / "llm_rewrite_cache.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    key = api_key()
    pool = [json.loads(l) for l in io.open(args.pool, encoding="utf-8")]
    splits = split_candidates(pool)
    slots = plan_rewrite(splits)
    todo = []
    for sp in ("train", "dev", "test"):
        for c in splits[sp]:
            if slots.get(c["pid"]) == "rewrite":
                todo.append(c)
    todo.sort(key=lambda c: c["pid"])
    if args.limit:
        todo = todo[: args.limit]
    print(f"[rewrite] 待改写 {len(todo)} 条  model={MODEL}")

    cache = {}
    if args.cache.exists():
        cache = json.loads(io.open(args.cache, encoding="utf-8").read())
    done = {ck for ck, v in cache.items() if v.get("ok")}
    todo = [c for c in todo if c.get("cache_key", c["pid"]) not in done]
    print(f"[rewrite] 缓存命中 {len(done)} 条，剩余 {len(todo)} 条")

    usage = Usage()
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(rewrite_one, key, c, usage) for c in todo]
        for i, f in enumerate(futs, 1):
            try:
                results.append(f.result())
            except Exception as exc:
                results.append({"pid": todo[i - 1]["pid"], "ok": False,
                                "question": None, "error": str(exc)[:200]})
            if i % 25 == 0 or i == len(futs):
                u = usage.snapshot()
                print(f"  [{i}/{len(futs)}] ok={sum(1 for r in results if r.get('ok'))} "
                      f"calls={u['calls']} err={u['errors']} "
                      f"tok_in={u['prompt_tokens']} tok_out={u['completion_tokens']} "
                      f"elapsed={round(time.time() - t0, 1)}s", flush=True)

    for r in results:
        cache[r.get("cache_key", r["pid"])] = r
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    u = usage.snapshot()
    ok = sum(1 for r in results if r.get("ok"))
    cost_usd = usage.cost_usd()
    cost_cny = cost_usd * USD_CNY
    print(f"[rewrite] 完成：成功 {ok}/{len(results)}  "
          f"调用 {u['calls']} 次（错误 {u['errors']}）  "
          f"输入 {u['prompt_tokens']} tok（cache-hit {u['prompt_cache_hit_tokens']}）/ "
          f"输出 {u['completion_tokens']} tok")
    print(f"[rewrite] 估算花费 ≈ ${cost_usd:.4f} ≈ ¥{cost_cny:.2f}  "
          f"（off-peak 单价 in ${PRICE_IN}/M, cache-hit ${PRICE_IN_CACHE}/M, out ${PRICE_OUT}/M）")
    print(f"[rewrite] 用时 {round(time.time() - t0, 1)}s  → {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
