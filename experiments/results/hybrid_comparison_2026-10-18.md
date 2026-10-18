# 混合检索对比实验（向量 vs 混合 RRF）

- 生成日期：2026-10-18
- 评测集：
  - `realistic`：`knowledge_base_large.json`（14 篇，人工精标、话题区分度大）/ `test_qa_large.json`（14 题）
  - `near_dup_stress`：`knowledge_base_stress.json`（10 篇，含大量近义/重复文档的噪声语料（验证混合检索的价值区间））/ `test_qa_stress.json`（8 题）

## 数据集：realistic

### 一、汇总指标

| 档位 | 平均 Recall@3 | 平均 Recall@5 | 平均 MRR |
|------|------|------|------|
| 纯向量（基线） | 1.0 | 1.0 | 0.9286 |
| 混合 BM25+向量 · RRF | 1.0 | 1.0 | 0.9286 |

### 二、逐题明细

| # | 问题 | 相关文档 | 纯向量（基线） R@3 | 混合 BM25+向量 · RRF R@3 | 纯向量（基线） MRR | 混合 BM25+向量 · RRF MRR |
|---|---|---|---|---|---|---|
| 1 | Transformer的自注意力机制里为什么要除以√d_k？ | transformer.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 2 | LoRA微调的低秩分解公式是怎样的，可训练参数大概占总量多少？ | finetuning.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 3 | NF4量化格式为什么对正态分布的权重更友好？ | quantization.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 4 | RAG系统可以从哪些方面优化生成延迟？ | rag_guide.txt | 1.0 | 1.0 | 0.5 | 0.5 |
| 5 | MQA（Multi-Query Attention）和 GQA（Grouped-Query Attention）有什么区别？ | attention_variants.txt | 1.0 | 1.0 | 0.5 | 0.5 |
| 6 | 除了量化，还有哪些主流的模型压缩方法？ | model_compression.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 7 | 大语言模型预训练常用的训练目标有哪些？ | pretraining.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 8 | ReAct范式是如何把推理和行动结合起来的？ | llm_agent.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 9 | 中文场景下有哪些值得关注的Embedding向量模型？ | embedding_models_cmp.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 10 | vLLM是通过什么机制提升大模型推理吞吐的？ | llm_serving.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 11 | 什么是思维链（Chain-of-Thought）提示？ | prompt_engineering.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 12 | 人工智能、机器学习、深度学习之间是什么关系？ | ai_intro.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 13 | NF4（NormalFloat4）这种 4 位量化格式有什么特点？ | quantization.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 14 | 在 RAG 的 Prompt 里，为什么要设置较低的温度、并要求模型'不知道就说不知道'？ | rag_guide.txt | 1.0 | 1.0 | 1.0 | 1.0 |

## 数据集：near_dup_stress

### 一、汇总指标

| 档位 | 平均 Recall@3 | 平均 Recall@5 | 平均 MRR |
|------|------|------|------|
| 纯向量（基线） | 0.625 | 0.625 | 0.4792 |
| 混合 BM25+向量 · RRF | 1.0 | 1.0 | 0.7708 |

### 二、逐题明细

| # | 问题 | 相关文档 | 纯向量（基线） R@3 | 混合 BM25+向量 · RRF R@3 | 纯向量（基线） MRR | 混合 BM25+向量 · RRF MRR |
|---|---|---|---|---|---|---|
| 1 | QTAG-A1 这条记录的内容是什么？ | quant_dup_A1.txt | 0.0 | 1.0 | 0.0 | 0.5 |
| 2 | QTAG-A3 这条记录的内容是什么？ | quant_dup_A3.txt | 0.0 | 1.0 | 0.0 | 0.3333 |
| 3 | QTAG-A5 这条记录的内容是什么？ | quant_dup_A5.txt | 0.0 | 1.0 | 0.0 | 0.3333 |
| 4 | QTAG-A7 这条记录的内容是什么？ | quant_dup_A7.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 5 | QTAG-A2 这条记录的内容是什么？ | quant_dup_A2.txt | 1.0 | 1.0 | 0.3333 | 1.0 |
| 6 | QTAG-A4 这条记录的内容是什么？ | quant_dup_A4.txt | 1.0 | 1.0 | 1.0 | 1.0 |
| 7 | QTAG-A6 这条记录的内容是什么？ | quant_dup_A6.txt | 1.0 | 1.0 | 0.5 | 1.0 |
| 8 | QTAG-A8 这条记录的内容是什么？ | quant_dup_A8.txt | 1.0 | 1.0 | 1.0 | 1.0 |

## 三、结论

### 数据集 realistic

- 纯向量基线：平均 Recall@3=1.0，平均 MRR=0.9286。
- 混合检索（BM25+向量 RRF）：平均 Recall@3=1.0（Δ=+0.0，持平），平均 MRR=0.9286（Δ=+0.0，持平）。
### 数据集 near_dup_stress

- 纯向量基线：平均 Recall@3=0.625，平均 MRR=0.4792。
- 混合检索（BM25+向量 RRF）：平均 Recall@3=1.0（Δ=+0.375，提升），平均 MRR=0.7708（Δ=+0.2916，提升）。
- 重排（Cross-Encoder，BAAI/bge-reranker-v2-m3）：已集成但默认关闭；本地无缓存未跑，启用后通常对 top 排位（MRR）进一步改善，见 README。

**关键结论**：在话题区分度大的精标知识库上，强向量模型已接近上限，混合检索收益有限；而在含大量近义/重复文档的噪声语料上，BM25 的精确词匹配能纠正向量把分数分散到近义副本上的问题，混合检索的相对增益显著——这也解释了「何时该上混合检索」。