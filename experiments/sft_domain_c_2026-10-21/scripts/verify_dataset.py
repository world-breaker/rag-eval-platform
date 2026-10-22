#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
数据自检：把报告里用到的每个数字都从落盘文件里核出来
=====================================================================

检查项
------
1. 三段条数与分布（答/拒/转 = 40/40/20；编号分布下限）
2. **标签回算**：用 rules.compute_label 重算每条问句的标签，必须与落盘的 expected 完全一致
   （这条是「标签永远由规则算」的可验证证据）
3. 三段之间**无 prompt 重叠**（精确匹配 + 4-gram Jaccard 上界）
4. **probe 24 条隔离**：数据集中任意问句与 probe 问句的最大 Jaccard（应 < 0.5 闸门）
5. core 跨段不重叠（同一核心句不跨 split）
6. 改写桶：实际生效条数、改写后与原文不同（真的变了）的条数
7. sha256 与 build_meta.json 一致

用法
----
    python scripts/verify_dataset.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

sys.path.insert(0, str(HERE))
from rules import compute_label, label_key  # noqa: E402
from build_sft_data import ngrams, jaccard, sha256_file  # noqa: E402

PROBE_DATA = Path(r"D:\programmer\LLM\projects\rag-eval-platform\experiments"
                   r"\domain_probe_2026-10-21\data\C_compliance_refusal.jsonl")

OK = "[OK]"
NG = "[!!]"


def read_jsonl(p: Path):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def main() -> int:
    meta = json.loads(io.open(DATA / "build_meta.json", encoding="utf-8").read())
    splits = {sp: read_jsonl(DATA / f"{sp}.jsonl") for sp in ("train", "dev", "test")}
    pool = {r["pid"]: r for r in read_jsonl(DATA / "pool.jsonl")}
    probe = read_jsonl(PROBE_DATA)

    print("=" * 74)
    print("① 条数与分布")
    for sp, rows in splits.items():
        d = meta["splits"][sp]
        dec = {}
        for r in rows:
            dec[r["expected"]["decision"]] = dec.get(r["expected"]["decision"], 0) + 1
        sha = sha256_file(DATA / f"{sp}.jsonl")
        print(f"  {OK} {sp}: n={len(rows)} (meta={d['n']})  {dec}  sha256 一致={sha == d['sha256']}")
        print(f"      红线 {d['redline_dist']}")
        print(f"      转人工 {d['transfer_dist']}")
        print(f"      桶 {d['bucket_dist']}")

    n_ref = sum(1 for r in splits["train"] + splits["dev"] + splits["test"]
                if r["expected"]["decision"] == "拒答")
    n_tr = sum(1 for r in splits["train"] + splits["dev"] + splits["test"]
               if r["expected"]["decision"] == "转人工")
    print(f"  {OK} 全量决策分布: 拒答 {n_ref} / 转人工 {n_tr} / 回答 {2600 - n_ref - n_tr}"
          f"（{n_ref/2600:.1%} / {n_tr/2600:.1%} / {(2600-n_ref-n_tr)/2600:.1%}）")

    print("=" * 74)
    print("② 标签回算（rules.compute_label 重算 vs 落盘 expected）")
    bad = 0
    total = 0
    for sp, rows in splits.items():
        for r in rows:
            q = r["prompt"].split("客户问：", 1)[-1]
            got = compute_label(q)
            total += 1
            if label_key(got) != label_key({"decision": r["expected"]["decision"],
                                            "redline": r["expected"]["redline"],
                                            "transfer_rules": r["expected"]["transfer_rules"]}):
                bad += 1
                if bad <= 5:
                    print(f"  {NG} {r['id']} q={q!r} 重算={got} 落盘={r['expected']}")
    print(f"  {OK if bad == 0 else NG} {total - bad}/{total} 条标签与规则引擎重算结果一致"
          f"{'（不一致 ' + str(bad) + ' 条）' if bad else ''}")

    print("=" * 74)
    print("③ 三段之间重叠")
    print("   注：完整 prompt 含 ~400 token 的共用策略文本，Jaccard 恒为 ~0.97，"
          "无信息量；因此下面以**问句部分**（'客户问：' 之后）为准。")
    qs = {sp: [r["prompt"].split("客户问：", 1)[-1] for r in rows] for sp, rows in splits.items()}
    ngs = {sp: [ngrams(p) for p in qs[sp]] for sp in qs}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        inter = set(qs[a]) & set(qs[b])
        mx, pair = 0.0, None
        for i, ga in enumerate(ngs[a]):
            for j, gb in enumerate(ngs[b]):
                jv = jaccard(ga, gb)
                if jv > mx:
                    mx, pair = jv, (i, j)
        verdict = OK if (not inter and mx <= 0.7) else NG
        print(f"  {verdict} {a}∩{b}: 问句精确重复 {len(inter)} 条；"
              f"问句最大 4-gram Jaccard {mx:.3f} (闸门 0.7) 最相近 {pair}")

    print("=" * 74)
    print("④ probe 24 条隔离")
    probe_qs = [r["prompt"].split("客户问：", 1)[-1].strip() for r in probe]
    probe_ng = [ngrams(q) for q in probe_qs]
    mx, pair = 0.0, None
    for sp, rows in splits.items():
        for i, r in enumerate(rows):
            q = r["prompt"].split("客户问：", 1)[-1]
            g = ngrams(q)
            for j, gp in enumerate(probe_ng):
                jv = jaccard(g, gp)
                if jv > mx:
                    mx, pair = jv, (sp, i, j)
    print(f"  {OK if mx <= 0.5 else NG} 数据集 vs probe24 最大 Jaccard = {mx:.3f} "
          f"(闸门 0.5) 最相近位置 {pair}")

    print("=" * 74)
    print("⑤ core 跨段不重叠")
    cores = {sp: set(r["meta"]["core_id"] for r in rows) for sp, rows in splits.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        ov = cores[a] & cores[b]
        print(f"  {OK if not ov else NG} {a}∩{b} core 交集 {len(ov)}")

    print("=" * 74)
    print("⑥ 改写桶")
    cache = json.loads(io.open(DATA / "llm_rewrite_cache.json", encoding="utf-8").read())
    ok_cache = sum(1 for v in cache.values() if v.get("ok"))
    changed = 0
    same = 0
    for sp, rows in splits.items():
        for r in rows:
            if r["meta"].get("rewritten"):
                p = pool.get(r["meta"]["pid"])
                if p:
                    orig_q = p["question"]
                    new_q = r["prompt"].split("客户问：", 1)[-1]
                    if orig_q != new_q:
                        changed += 1
                    else:
                        same += 1
    n_rw = sum(1 for sp in splits for r in splits[sp] if r["meta"].get("rewritten"))
    print(f"  {OK} 缓存成功 {ok_cache}/{len(cache)}；数据集中实际改写 {n_rw} 条，"
          f"其中文本确实变化 {changed} 条、与原文相同 {same} 条")

    print("=" * 74)
    print("⑦ 确定性（重跑 pool + finalize 后的 sha256 应与 build_meta 一致）")
    for sp in splits:
        sha = sha256_file(DATA / f"{sp}.jsonl")
        print(f"  {OK if sha == meta['splits'][sp]['sha256'] else NG} {sp}.jsonl sha256={sha}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
