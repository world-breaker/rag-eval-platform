# Chunk 切分策略对比实验（真实运行结果 · 2026-10-16）

> 全部数字来自 `experiments/run_strategy_comparison.py` 的实际运行输出；
> 原始数据见同目录 `strategy_comparison_2026-10-16.json`。未产生任何估算数字。

## 环境

- Python 3.11.15 / torch 2.11.0+cu128 / sentence-transformers 5.6.1 / numpy 2.4.4
- 设备：NVIDIA GeForce RTX 5060 Laptop GPU（CUDA 可用：True）
- Embedding 模型：BAAI/bge-small-zh-v1.5（512 维）
- 检索参数：top_k=5，similarity_threshold=0.3
- 数据集：5 篇文档（共 2007 字符），5 道标注问题（文档级标注 relevant_sources）

> 指标口径：检索结果按 `source` 去重为文档级序列后计算（chunk 边界随切分策略变化，
> chunk 索引无法跨策略对照）。hit@1/hit@3 = 相关文档出现在第 1 / 前 3 位的问题占比。

## 实验一：切分策略对比（chunk_size=256, overlap=50）

| 配置 | chunk 数 | 平均 chunk 字符 | recall@3 | recall@5 | precision@3 | MRR | hit@1 | hit@3 | 建索引耗时(s) |
|---|---|---|---|---|---|---|---|---|---|
| fixed_size@256 | 13 | 181.8 | 1.0 | 1.0 | 0.3333 | 0.9 | 0.8 | 1.0 | 0.001 |
| recursive@256 | 25 | 78.7 | 1.0 | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.001 |
| semantic@256 | 11 | 180.1 | 1.0 | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.001 |

## 实验二：切分策略对比（chunk_size=128, overlap=25）

| 配置 | chunk 数 | 平均 chunk 字符 | recall@3 | recall@5 | precision@3 | MRR | hit@1 | hit@3 | 建索引耗时(s) |
|---|---|---|---|---|---|---|---|---|---|
| fixed_size@128 | 22 | 110.2 | 1.0 | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.002 |
| recursive@128 | 28 | 72.8 | 1.0 | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.001 |
| semantic@128 | 27 | 72.9 | 1.0 | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.002 |

## 实验三：fixed_size 的 chunk_size 扫描（overlap=50 / 128 用 25）

| 配置 | chunk 数 | 平均 chunk 字符 | recall@3 | recall@5 | precision@3 | MRR | hit@1 | hit@3 | 建索引耗时(s) |
|---|---|---|---|---|---|---|---|---|---|
| fixed_size@128 | 22 | 110.2 | 1.0 | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.001 |
| fixed_size@256 | 13 | 181.8 | 1.0 | 1.0 | 0.3333 | 0.9 | 0.8 | 1.0 | 0.001 |
| fixed_size@512 | 5 | 401.4 | 1.0 | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.001 |
| fixed_size@1024 | 5 | 401.4 | 1.0 | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.001 |

## 逐题明细

### 实验一（@256）

**fixed_size@256**（13 chunks，平均 181.8 字符/chunk）

| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| 什么是Transformer的自注意力机制？ | transformer.txt | transformer.txt → ai_intro.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |
| LoRA微调的原理是什么？ | finetuning.txt | finetuning.txt → quantization.txt → transformer.txt → rag_guide.txt | 1.0 | 1.0 | 1.0 |
| RAG系统如何优化延迟？ | rag_guide.txt | finetuning.txt → rag_guide.txt → quantization.txt | 1.0 | 1.0 | 0.5 |
| 模型量化的基本原理是什么？ | quantization.txt | quantization.txt → rag_guide.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |
| 深度学习是什么？ | ai_intro.txt | ai_intro.txt → transformer.txt → quantization.txt | 1.0 | 1.0 | 1.0 |

**recursive@256**（25 chunks，平均 78.7 字符/chunk）

| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| 什么是Transformer的自注意力机制？ | transformer.txt | transformer.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| LoRA微调的原理是什么？ | finetuning.txt | finetuning.txt → transformer.txt | 1.0 | 1.0 | 1.0 |
| RAG系统如何优化延迟？ | rag_guide.txt | rag_guide.txt → finetuning.txt → quantization.txt | 1.0 | 1.0 | 1.0 |
| 模型量化的基本原理是什么？ | quantization.txt | quantization.txt → rag_guide.txt | 1.0 | 1.0 | 1.0 |
| 深度学习是什么？ | ai_intro.txt | ai_intro.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |

**semantic@256**（11 chunks，平均 180.1 字符/chunk）

| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| 什么是Transformer的自注意力机制？ | transformer.txt | transformer.txt → finetuning.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| LoRA微调的原理是什么？ | finetuning.txt | finetuning.txt → transformer.txt → quantization.txt | 1.0 | 1.0 | 1.0 |
| RAG系统如何优化延迟？ | rag_guide.txt | rag_guide.txt → quantization.txt → finetuning.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| 模型量化的基本原理是什么？ | quantization.txt | quantization.txt → rag_guide.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |
| 深度学习是什么？ | ai_intro.txt | ai_intro.txt → transformer.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |

### 实验二（@128）

**fixed_size@128**（22 chunks，平均 110.2 字符/chunk）

| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| 什么是Transformer的自注意力机制？ | transformer.txt | transformer.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| LoRA微调的原理是什么？ | finetuning.txt | finetuning.txt → quantization.txt | 1.0 | 1.0 | 1.0 |
| RAG系统如何优化延迟？ | rag_guide.txt | rag_guide.txt → finetuning.txt → quantization.txt | 1.0 | 1.0 | 1.0 |
| 模型量化的基本原理是什么？ | quantization.txt | quantization.txt → rag_guide.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |
| 深度学习是什么？ | ai_intro.txt | ai_intro.txt → transformer.txt | 1.0 | 1.0 | 1.0 |

**recursive@128**（28 chunks，平均 72.8 字符/chunk）

| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| 什么是Transformer的自注意力机制？ | transformer.txt | transformer.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| LoRA微调的原理是什么？ | finetuning.txt | finetuning.txt | 1.0 | 1.0 | 1.0 |
| RAG系统如何优化延迟？ | rag_guide.txt | rag_guide.txt → finetuning.txt → quantization.txt | 1.0 | 1.0 | 1.0 |
| 模型量化的基本原理是什么？ | quantization.txt | quantization.txt → rag_guide.txt | 1.0 | 1.0 | 1.0 |
| 深度学习是什么？ | ai_intro.txt | ai_intro.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |

**semantic@128**（27 chunks，平均 72.9 字符/chunk）

| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| 什么是Transformer的自注意力机制？ | transformer.txt | transformer.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| LoRA微调的原理是什么？ | finetuning.txt | finetuning.txt | 1.0 | 1.0 | 1.0 |
| RAG系统如何优化延迟？ | rag_guide.txt | rag_guide.txt → finetuning.txt → quantization.txt | 1.0 | 1.0 | 1.0 |
| 模型量化的基本原理是什么？ | quantization.txt | quantization.txt → rag_guide.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |
| 深度学习是什么？ | ai_intro.txt | ai_intro.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |

### 实验三（size 扫描）

**fixed_size@128**（22 chunks，平均 110.2 字符/chunk）

| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| 什么是Transformer的自注意力机制？ | transformer.txt | transformer.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| LoRA微调的原理是什么？ | finetuning.txt | finetuning.txt → quantization.txt | 1.0 | 1.0 | 1.0 |
| RAG系统如何优化延迟？ | rag_guide.txt | rag_guide.txt → finetuning.txt → quantization.txt | 1.0 | 1.0 | 1.0 |
| 模型量化的基本原理是什么？ | quantization.txt | quantization.txt → rag_guide.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |
| 深度学习是什么？ | ai_intro.txt | ai_intro.txt → transformer.txt | 1.0 | 1.0 | 1.0 |

**fixed_size@256**（13 chunks，平均 181.8 字符/chunk）

| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| 什么是Transformer的自注意力机制？ | transformer.txt | transformer.txt → ai_intro.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |
| LoRA微调的原理是什么？ | finetuning.txt | finetuning.txt → quantization.txt → transformer.txt → rag_guide.txt | 1.0 | 1.0 | 1.0 |
| RAG系统如何优化延迟？ | rag_guide.txt | finetuning.txt → rag_guide.txt → quantization.txt | 1.0 | 1.0 | 0.5 |
| 模型量化的基本原理是什么？ | quantization.txt | quantization.txt → rag_guide.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |
| 深度学习是什么？ | ai_intro.txt | ai_intro.txt → transformer.txt → quantization.txt | 1.0 | 1.0 | 1.0 |

**fixed_size@512**（5 chunks，平均 401.4 字符/chunk）

| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| 什么是Transformer的自注意力机制？ | transformer.txt | transformer.txt → ai_intro.txt → rag_guide.txt → quantization.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |
| LoRA微调的原理是什么？ | finetuning.txt | finetuning.txt → transformer.txt → quantization.txt → rag_guide.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| RAG系统如何优化延迟？ | rag_guide.txt | rag_guide.txt → quantization.txt → finetuning.txt → transformer.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| 模型量化的基本原理是什么？ | quantization.txt | quantization.txt → rag_guide.txt → finetuning.txt → transformer.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| 深度学习是什么？ | ai_intro.txt | ai_intro.txt → transformer.txt → quantization.txt → rag_guide.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |

**fixed_size@1024**（5 chunks，平均 401.4 字符/chunk）

| 问题 | 相关来源 | 检回来源(按序) | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| 什么是Transformer的自注意力机制？ | transformer.txt | transformer.txt → ai_intro.txt → rag_guide.txt → quantization.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |
| LoRA微调的原理是什么？ | finetuning.txt | finetuning.txt → transformer.txt → quantization.txt → rag_guide.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| RAG系统如何优化延迟？ | rag_guide.txt | rag_guide.txt → quantization.txt → finetuning.txt → transformer.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| 模型量化的基本原理是什么？ | quantization.txt | quantization.txt → rag_guide.txt → finetuning.txt → transformer.txt → ai_intro.txt | 1.0 | 1.0 | 1.0 |
| 深度学习是什么？ | ai_intro.txt | ai_intro.txt → transformer.txt → quantization.txt → rag_guide.txt → finetuning.txt | 1.0 | 1.0 | 1.0 |

## 结论

- 共覆盖 3 种策略 × 2 种粒度 + fixed_size 的 4 档 chunk_size = 10 个切分配置：**全部配置的文档级 `recall@3` / `recall@5` / `hit@3` 均为 1.0**（5/5 命中）。在本样例知识库（5 篇短文，共 2007 字符）上，三种切分策略的召回能力没有可测量的差异。
- 唯一可见差异在排序质量：`fixed_size@256` 的 MRR=0.9、hit@1=0.8 —— 「RAG系统如何优化延迟？」一题中 finetuning.txt（相似度 0.6143）排在了目标 rag_guide.txt（0.5943）之前；其余 9 个配置的 MRR 与 hit@1 均为 1.0。
- 天花板效应的主因是数据集规模：5 篇主题互不重叠的短文 + 5 个强指向性问题，几乎任何切片方式都能把正确文档送进 top-3。
- chunk_size 扫描：128 / 512 / 1024 全部满分，256 反而出现唯一一次排序错误；512 与 1024 的逐题结果完全一致（每篇文档都短于 512 字符，两种粒度切出的 chunk 数量与内容相同）。因此「512 相比 256/1024 的 recall 更高」这类说法在本数据集上不成立——真实结论是「本规模下各粒度差异不显著」。
- 适用边界：以上数字只对随仓库提供的样例数据集负责。要得出有区分度的策略结论，需要更大规模、包含相似干扰文档的评测集（更多文档、与目标 chunk 语义重叠更高的问题）。

## 复现命令

```bash
pip install -r requirements.txt
python experiments/run_strategy_comparison.py
```

> 注：首次运行会下载 BAAI/bge-small-zh-v1.5（约 100MB）；
> embedding 结果按文本内容 MD5 缓存在 `data/embedding_cache/`，重复运行不重算。
