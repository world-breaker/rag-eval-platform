# RAG 知识库问答 + 评测平台

> 一个可复现的 RAG 实验平台：模块化流水线 + 三层评测体系 + 切分策略对比实验。

**仓库地址**：https://gitee.com/tmlKevin/rag-eval-platform

---

## 项目定位

这是一个**模块化架构 + 系统化评测**的 RAG 平台，重点是把「检索 → 生成 → 评测」三层解耦：每个环节可以单独替换、单独度量、单独做对比实验。

> 架构、数据流与设计取舍的完整说明见 [`docs/architecture.md`](docs/architecture.md)。

## 核心功能

| 模块 | 功能 | 设计考量 |
|------|------|-----------|
| **文档切分** | 3 种策略（固定窗口 / 递归分隔符 / 语义段落），可配置 chunk_size / overlap | 不同文档结构适用不同策略，留配置入口与扫描实验 |
| **向量化** | BGE-small-zh-v1.5（512 维），文本 MD5 内容缓存 | 文档不变不重算，重复运行零成本 |
| **检索** | NumPy 余弦检索 + 相似度阈值过滤；**可选混合检索（BM25 + 向量 RRF 融合，可选 Cross-Encoder 重排）** | 默认纯向量（零回归）；`retrieval.mode: "hybrid"` 开启双路召回，专治「近义/重复文档」导致的向量混淆 |
| **生成** | Qwen2.5 本地推理 / OpenAI 兼容 API，结构化 RAG Prompt | Prompt 约束「只使用参考文档、不知道就说不知道」以降低幻觉 |
| **评测** | 三层指标：检索（Recall@K / Precision@3 / MRR）+ 生成（LLM-as-Judge）+ 传统（BLEU / ROUGE-L） | 检索与生成的错误分离度量，问题定位更快 |
| **对比实验** | 切分策略 × 粒度矩阵，结果落盘为 JSON + Markdown | 见 [`experiments/results/`](experiments/results/) |
| **Web 界面** | Gradio 四个 Tab（文档管理 / 问答 / 评测 / 实验对比） | 浏览器内完成全流程演示 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动（CPU 版）
python app.py

# 3. 打开浏览器 http://localhost:7860
```

`config.yaml` 中可切换：

- `embedding.device: "cpu"` → `"cuda"`（有 GPU 时）
- `generation.use_api: true` → 使用 OpenAI 兼容 API（如 Ollama）代替本地模型
- `chunking.strategy: "fixed_size"` → `"recursive"` / `"semantic"`

## 项目结构

```
rag-eval-platform/
├── app.py                  # Gradio Web 界面
├── config.yaml             # 所有配置集中管理
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── src/
│   ├── chunking.py         # 文档切分（3 种策略）
│   ├── embedding.py        # 向量化（带 MD5 缓存）
│   ├── retrieval.py        # 检索（余弦相似度 + 阈值过滤）
│   ├── hybrid_retrieval.py # 混合检索（BM25 + 向量 RRF 融合 + 可选重排）
│   ├── generation.py       # LLM 生成（Prompt 模板）
│   ├── evaluation.py       # 评测（三层指标）
│   ├── pipeline.py         # 全链路编排 + 对比实验
│   └── utils.py            # 配置 / 日志 / 计时
├── data/
│   └── sample/
│       ├── knowledge_base.json        # 示例知识库（5 篇短文，原天花板效应评测集）
│       ├── test_qa.json               # 评测数据集（文档级标注 + 参考答案）
│       ├── knowledge_base_large.json  # 扩展示例（14 篇，含近义/干扰文档）
│       ├── test_qa_large.json         # 扩大评测集（14 题）
│       ├── knowledge_base_stress.json # 近义/重复噪声语料（验证混合检索价值区间）
│       └── test_qa_stress.json        # 近义副本精确标识符查询评测集（8 题）
├── experiments/
│   ├── run_strategy_comparison.py   # 切分策略对比实验 runner
│   ├── run_hybrid_comparison.py     # 混合检索对比实验 runner（向量 vs 混合 RRF）
│   └── results/                     # 实验原始数据 + 人读报告
├── docs/
│   └── architecture.md     # 架构与设计决策（公开版）
└── tests/
    ├── test_pipeline.py      # 快速验证脚本（不依赖 GPU 也能跑通）
    └── test_hybrid_retrieval.py # 混合检索离线单测（BM25 / RRF / 融合）
```

## 用代码验证

```python
from src.pipeline import RAGPipeline

# 初始化（load_generator=False 可跳过生成模型，只做检索/评测）
pipeline = RAGPipeline("config.yaml")

# 建库
pipeline.build_index([
    {"text": "深度学习是机器学习的一个子领域...", "source": "dl.txt"},
    {"text": "Transformer架构的核心是自注意力...", "source": "transformer.txt"},
])

# 问答
result = pipeline.ask("什么是Transformer？")
print(result["answer"])

# 切分策略对比（输出各策略的平均检索分）
comparison = pipeline.compare_chunk_strategies(
    documents=[{"text": "...长文档...", "source": "doc.txt"}],
    questions=["总结本文档", "提到了哪些技术"],
)
print(comparison)
```

## 实验结果

切分策略对比实验的真实运行数据（2026-10-16）：

- 报告：[`experiments/results/strategy_comparison_2026-10-16.md`](experiments/results/strategy_comparison_2026-10-16.md)
- 原始数据：[`experiments/results/strategy_comparison_2026-10-16.json`](experiments/results/strategy_comparison_2026-10-16.json)

复现命令：

```bash
python experiments/run_strategy_comparison.py
```

**一句话结论**：在仓库自带的样例数据集（5 篇文档 / 5 个问题 / 共 2007 字符）上，三种切分策略与各 chunk_size 的文档级 `recall@3` 均为 1.0，没有可测量的差异（数据规模导致的天花板效应）；报告里如实标注了适用边界，并给出了扩展到有区分度评测集的建议。唯一的排序差异出现在 `fixed_size@256`（MRR 0.9）。

## 混合检索（Hybrid Retrieval）

`retrieval.mode` 开关控制：

- `"vector"`（默认）：原有纯向量行为，**与历史版本完全兼容、零回归**。
- `"hybrid"`：BM25（关键词）+ 向量（语义）双路召回 → **RRF（Reciprocal Rank Fusion）** 融合排序 → 可选 **Cross-Encoder 重排**。

为什么这么做：

1. **纯向量检索的短板**：擅长语义相似，但对精确关键词（"NF4" / "LoRA" / "√d_k"）和稀有术语不敏感，且在**近义/重复文档**上会把分数分散到多个副本，导致相关文档排不到前面。
2. **BM25 补关键词**：零额外依赖（不引入 jieba，用「英文全词 + 中文单字 + 二元组」分词），精确匹配术语，正好补向量短板。
3. **RRF 融合而非加权求和**：两路分数**量纲不同**（余弦 ∈ [-1,1]，BM25 可任意大），不能直接相加；RRF 用排名融合 `score = Σ 1/(k+rank)`，对量纲不敏感、鲁棒性好。
4. **可选 Cross-Encoder 重排**：对 RRF 融合后的 top-N 用交互式模型（默认 `BAAI/bge-reranker-v2-m3`，中文效果好）二次精排，进一步提升精度；**默认关闭**，加载失败自动降级回 RRF，保证离线可跑、结果可复现。

开启方式（`config.yaml`）：

```yaml
retrieval:
  mode: "hybrid"
  rrf_k: 60
  hybrid:
    bm25: { k1: 1.5, b: 0.75 }
    rerank:
      enabled: false          # 改为 true 启用重排（需下载 ~1.2GB 重排模型）
      type: "cross_encoder"
      model: "BAAI/bge-reranker-v2-m3"
      top_n: 20
```

### 混合检索评测结果（2026-10-18）

复现：`python experiments/run_hybrid_comparison.py`（报告落 `experiments/results/hybrid_comparison_2026-10-18.md`）。

| 数据集 | 档位 | 平均 Recall@3 | 平均 MRR |
|------|------|------|------|
| realistic（14 篇，话题区分度大） | 纯向量 | 1.0 | 0.9286 |
| realistic（14 篇，话题区分度大） | 混合 RRF | 1.0 | 0.9286 |
| near_dup_stress（10 篇，含近义/重复副本） | 纯向量 | 0.625 | 0.4792 |
| near_dup_stress（10 篇，含近义/重复副本） | 混合 RRF | **1.0** | **0.7708** |

**核心结论**：

- 在话题区分度大的精标知识库上，强向量模型已接近上限，**混合检索收益有限**——这本身是个重要认知。
- 在**含大量近义/重复文档的噪声语料**上，BM25 的精确词匹配纠正了向量把分数分散到近义副本的问题，混合检索相对纯向量 **Recall@3 +37.5%、MRR +29%**。这正是「何时该上混合检索」的价值区间。
- 评测口径与 `evaluation.evaluate_retrieval_doc_level` 一致（文档级标注，跨策略可比）。

## 评测口径

- 评测数据集 `data/sample/test_qa.json` 使用**文档级标注**（`relevant_sources`）：chunk 边界随切分策略变化，chunk 索引无法跨策略对照，因此以「是否检索到正确文档」为稳定口径，由 `EvaluationEngine.evaluate_retrieval_doc_level` 计算（内部复用 Recall / Precision / MRR 公式）。
- 三层指标：检索层（Recall@3 / Recall@5 / Precision@3 / MRR）、生成层（faithfulness / relevance / completeness，LLM-as-Judge）、传统层（BLEU / ROUGE-L）。
- 生成层需要 Qwen 模型或 API；未配置时回退到启发式评分（仅用于跑通流程，不作为正式指标）。

## 设计决策

**Q: chunk_size 为什么默认 512？**
> `experiments/run_strategy_comparison.py` 提供 128–1024 的完整扫描入口，可一键复跑。在仓库自带的样例数据集上各粒度 `recall@3` 均为 1.0、无显著差异；512 是工程默认值，面向「一个 chunk 覆盖一段完整技术要点」的常见文档形态，更大规模语料下的选型需要更大评测集验证。

**Q: LLM-as-Judge 打分准吗？**
> 有偏差——同一条答案多次打分可能不同。当前实现的做法：Judge Prompt 只要求输出数字、限定评分维度（结构化输出降低随机性）；后续可加多次评分取中位数。这是当前评测工程中的已知开放问题。

**Q: 为什么检索是 NumPy 而不是向量数据库？**
> 本项目的知识库规模是「单机小知识库」（几十~几百个 chunk），暴力余弦检索在这个量级下复杂度完全不构成瓶颈，且零额外依赖、行为完全可解释。config 中保留了向量库相关的配置位，规模上量后（万级以上 chunk）可平滑替换为 Faiss / Chroma 等服务。

## 技术栈

- **Embedding**：sentence-transformers / BGE-small-zh-v1.5（512 维）
- **向量检索**：NumPy 余弦相似度（暴力检索 + 阈值过滤）
- **生成**：Qwen2.5-1.5B-Instruct（本地）/ OpenAI 兼容 API
- **界面**：Gradio
- **部署**：Docker + docker-compose

## License

MIT
