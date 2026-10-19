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
| **检索** | 三档排名口径：**纯向量 / 纯 BM25 / 混合（BM25 + 向量 RRF 融合）**，可选 **Cross-Encoder 重排**，可选 **查询改写（HyDE / multi-query）**；结果携带各路排名与融合分 | 默认纯向量（零回归）；`retrieval.mode` 切换口径；混合专治「近义/重复文档」的向量混淆；重排与改写默认关（要额外模型/延迟），按场景开 |
| **生成** | Qwen2.5 本地推理 / OpenAI 兼容 API，结构化 RAG Prompt | Prompt 约束「只使用参考文档、不知道就说不知道」以降低幻觉 |
| **评测** | 三层指标：检索（Recall@K / Precision@3 / MRR）+ 生成（LLM-as-Judge）+ 传统（BLEU / ROUGE-L） | 检索与生成的错误分离度量，问题定位更快 |
| **对比实验** | 切分策略 × 粒度矩阵；检索模式 × 重排 × 改写的**质量 + 延迟**基准，结果落盘为 JSON + Markdown | 见 [`experiments/results/`](experiments/results/) |
| **Web 界面** | Gradio 四个 Tab（文档管理 / 问答 / 评测 / 实验对比）；问答页可选检索模式、重排开关，结果表直接展示 **各路排名与 RRF 融合分** | 浏览器内完成全流程演示，且能现场解释「这条是谁捞上来的」 |

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
│   ├── retrieval.py        # 纯向量检索（余弦相似度 + 阈值过滤）
│   ├── hybrid_retrieval.py # BM25 / 向量 / RRF 融合 / 可选 Cross-Encoder 重排
│   ├── query_rewriting.py  # 查询改写（HyDE / multi-query，默认关）
│   ├── generation.py       # LLM 生成（Prompt 模板 + 裸补全接口）
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
│   ├── run_hybrid_comparison.py     # 混合检索对比实验 runner（向量 vs 混合 RRF，2026-10-18 快照）
│   ├── run_retrieval_benchmark.py   # 检索基准：全档位质量 + 延迟（p50/p95，冷/热启动）
│   └── results/                     # 实验原始数据 + 人读报告
├── docs/
│   └── architecture.md     # 架构与设计决策（公开版）
└── tests/
    ├── test_pipeline.py      # 快速验证脚本（不依赖 GPU 也能跑通）
    ├── test_hybrid_retrieval.py # 混合检索离线单测（BM25 / RRF / 融合）
    ├── test_retrieval_modes.py  # 三档口径 + 重排覆盖 + 改写融合（离线确定性）
    └── test_query_rewriting.py  # 查询改写单测（stub LLM，含降级路径）
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

# 问答（默认走 config 里的检索模式，历史行为不变）
result = pipeline.ask("什么是Transformer？")
print(result["answer"])

# 运行时切换检索模式（Web UI 用的也是这条路径）
pipeline.set_retrieval_mode("hybrid")          # vector | bm25 | hybrid
pipeline.set_rerank(True)                      # 可选：叠加 Cross-Encoder 重排
docs = pipeline._retrieve("Transformer 的自注意力为什么要除以 √d_k？", top_k=5)
for d in docs:
    print(d["source"], d["score"], d["bm25_score"], d["fusion_score"],
          d["vector_rank"], d["bm25_rank"])

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
- `"bm25"`：纯关键词召回（Okapi BM25，纯 NumPy 实现）；只返回真正命中的文档，微秒级。
- `"hybrid"`：BM25（关键词）+ 向量（语义）双路召回 → **RRF（Reciprocal Rank Fusion）** 融合排序 → 可选 **Cross-Encoder 重排**（可叠加 **查询改写** HyDE / multi-query）。

为什么这么做：

1. **纯向量检索的短板**：擅长语义相似，但对精确关键词（"NF4" / "LoRA" / "√d_k"）和稀有术语不敏感，且在**近义/重复文档**上会把分数分散到多个副本，导致相关文档排不到前面。
2. **BM25 补关键词**：零额外依赖（不引入 jieba，用「英文全词 + 中文单字 + 二元组」分词），精确匹配术语，正好补向量短板。
3. **RRF 融合而非加权求和**：两路分数**量纲不同**（余弦 ∈ [-1,1]，BM25 可任意大），不能直接相加；RRF 用排名融合 `score = Σ 1/(k+rank)`，对量纲不敏感、鲁棒性好。
4. **可选 Cross-Encoder 重排**：对 RRF 融合后的 top-N 用交互式模型（默认 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`，470MB、多语言、CPU 可跑、**已实测**）二次精排，进一步提升精度；**默认关闭**，加载失败自动降级回 RRF，保证离线可跑、结果可复现。（想要中文专用可换 `BAAI/bge-reranker-v2-m3`，但**未实测**且需 ~2.27GB；换模型后应重跑基准再引用数字。）

开启方式（`config.yaml`）：

```yaml
retrieval:
  mode: "hybrid"
  rrf_k: 60
  hybrid:
    bm25: { k1: 1.5, b: 0.75 }
    rerank:
      enabled: false          # 改为 true 启用重排（需下载 ~470MB 重排模型）
      type: "cross_encoder"
      model: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
      top_n: 20
```

### 检索基准实测：质量 + 延迟（2026-10-19）

复现：`python experiments/run_retrieval_benchmark.py`（离线可跑；报告落 `experiments/results/retrieval_benchmark_2026-10-19.md` + `.json`）。
更早的 `run_hybrid_comparison.py`（2026-10-18，仅 vector vs hybrid）保留为历史快照。

口径（写清楚，数字才有意义）：

- **机器**：单机 RTX 5060 Laptop（8GB）/ Python 3.12.9 / torch 2.14.0+cu132；embedding 走 GPU，**重排走 CPU**，改写 LLM 走 GPU。
- **延迟只算检索**：单次 `_retrieve` = 查询改写（若有）+ query 编码 + 检索（+ 重排），**不含生成回答**。
- **冷启动** = 切到该档位后的**第一次**查询（含懒加载：重排模型/LLM 载入 + CUDA 预热）。
  **热启动** = 之后所有查询的 p50 / p95 / 均值（普通档位 5 轮，LLM 改写档位 2 轮）。

**数据集 A：`realistic`**（`knowledge_base_large.json` 14 篇 / 14 题，话题区分度大）

| 档位 | Recall@3 | Recall@5 | MRR | 冷启动 ms | 热 p50 ms | 热 p95 ms | 热均值 ms |
|---|---|---|---|---|---|---|---|
| 纯向量（基线） | 1.0 | 1.0 | 0.9286 | 1.2 | 0.62 | 1.06 | 0.63 |
| 纯 BM25 关键词 | 1.0 | 1.0 | 0.8929 | 2.44 | 2.07 | 3.35 | 2.04 |
| 混合 BM25+向量 · RRF | 1.0 | 1.0 | 0.9286 | 2.15 | 2.38 | 3.38 | 2.36 |
| 混合 + Cross-Encoder 重排 | 1.0 | 1.0 | **1.0** | 665.12 | 994.68 | 1118.89 | 910.92 |
| 混合 + multi-query 改写 | 1.0 | 1.0 | 0.9524 | 3580.22 | 1609.65 | 2785.19 | 1795.64 |
| 混合 + HyDE 改写 | 1.0 | 1.0 | 0.9286 | 11329.26 | 12630.11 | 15712.44 | 11733.86 |

**数据集 B：`near_dup_stress`**（`knowledge_base_stress.json` 10 篇 / 8 题，含 8 篇近义副本）

| 档位 | Recall@3 | Recall@5 | MRR | 冷启动 ms | 热 p50 ms | 热 p95 ms | 热均值 ms |
|---|---|---|---|---|---|---|---|
| 纯向量（基线） | 0.625 | 0.625 | 0.4792 | 0.88 | 0.4 | 0.85 | 0.48 |
| 纯 BM25 关键词 | **1.0** | **1.0** | **1.0** | 0.44 | 0.46 | 0.73 | 0.49 |
| 混合 BM25+向量 · RRF | **1.0** | **1.0** | 0.7708 | 0.6 | 0.43 | 1.1 | 0.51 |
| 混合 + Cross-Encoder 重排 | **1.0** | **1.0** | **1.0** | 640.99 | 74.26 | 76.63 | 74.11 |
| 混合 + multi-query 改写 | **1.0** | **1.0** | 0.8333 | 1601.55 | 1198.94 | 1374.94 | 1202.86 |
| 混合 + HyDE 改写 | **0.75** | **1.0** | 0.7375 | 4977.37 | 5569.06 | 7973.54 | 5473.57 |

**核心结论（包含对我们不利的实测）**：

1. **干净小库上向量已到天花板**（R@3 已 1.0）：混合 / 重排 / 改写**都提不了召回**，只能改排序——重排把 MRR 提到 1.0、multi-query 提到 0.9524，代价是热 p50 从 2.4ms 涨到 1.0s / 1.6s（×400 / ×680）。
2. **收益真正出现在噪声库**：混合把 R@3 从 0.625 拉到 1.0（**+37.5 个百分点 / 相对 +60%**）、MRR 0.4792 → 0.7708；重排再把 MRR 补到 1.0。
3. **但混合不是无脑更好**：stress 集上**纯 BM25 的 MRR(1.0) 高于混合(0.7708)**。查询是纯标识符（"QTAG-A1"），单路精确匹配就已最优，融合反而被向量那一路的整体排名稀释。→ 该场景应该直接选 `bm25`，而不是 `hybrid`。
4. **HyDE 在我们的设置下掉点**：噪声库上 R@3 1.0 → 0.75、MRR 0.7708 → 0.7375，且热 p50 ×13000（5.6s）。1.5B 小模型生成的假想文档会夹带泛化术语，把检索带偏。这是「改写**默认关**」最直接的实测依据，也说明 HyDE 的收益**依赖生成模型质量**，换更大/更强的 LLM 需重新实测。
5. **延迟量级对照**：单路 BM25 微秒级（在总延迟里看不见）；CPU 重排 ~74–995ms；查询改写 1.2–12.6s。所以「重排/改写默认关、按场景开」不是保守，而是把代价花在**有收益的语料**上。
6. 延迟为单机实测、存在波动：重排（CPU）在两次运行间 p50 从 510ms 变到 995ms，量级可信、具体数字不宜当基准线；质量指标（Recall / MRR）是确定性的，可复现。

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
