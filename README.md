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
| **评测** | 三层指标：检索（Recall@K / Precision@3 / MRR）+ 生成（LLM-as-Judge）+ 传统（BLEU / ROUGE-L） | 检索与生成的错误分离度量，问题定位更快；**生成层已做三臂对照实测**（规则判据 / 本地 1.5B 裁判 / 强裁判 API），见下文 |
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
├── .github/
│   └── workflows/ci.yml    # CI 门禁（GitHub Actions 格式；宿主 Gitee 不自动跑）
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
│       └── test_qa_unanswerable.json  # 拒答样本（6 题，2026-10-19 手写，测生成层拒答正确性）
├── experiments/
│   ├── run_strategy_comparison.py   # 切分策略对比实验 runner
│   ├── run_hybrid_comparison.py     # 混合检索对比实验 runner（向量 vs 混合 RRF，2026-10-18 快照）
│   ├── run_retrieval_benchmark.py   # 检索基准：全档位质量 + 延迟（p50/p95，冷/热启动）
│   ├── run_generation_eval.py       # 生成层忠实度三臂对照（规则 / 本地 1.5B 裁判 / 强裁判 API）
│   ├── package_variance_json.py     # 把 run_generation_eval 的 raw 打包成紧凑版方差/稳健性 JSON（复现链的第二步）
│   └── results/                     # 实验原始数据 + 人读报告
├── docs/
│   └── architecture.md     # 架构与设计决策（公开版）
└── tests/
    ├── test_pipeline.py      # 快速验证脚本（不依赖 GPU 也能跑通）
    ├── test_hybrid_retrieval.py # 混合检索离线单测（BM25 / RRF / 融合）
    ├── test_retrieval_modes.py  # 三档口径 + 重排覆盖 + 改写融合（离线确定性）
    ├── test_query_rewriting.py  # 查询改写单测（stub LLM，含降级路径）
    ├── test_evaluation_judge.py # 裁判引擎单测（设备搬运 / 兜底可观测 / 多次采样与部分失败 / 配置校验）
    ├── test_false_green_guards.py # 「假绿」防护单测（降级必须可见：改写降级 / 重排降级 / 部分采样失败）
    ├── test_embedding_cache.py # embedding 缓存键含模型名 + 原子落盘（不加载真实模型）
    ├── test_chunking.py      # 切分策略 fail-fast（写错策略名直接报错，不静默回退）
    ├── test_config_honesty.py # 配置诚实性（死键不许复活 / 扩展位 fail-fast / 活键被读走）
    ├── test_package_variance_json.py # 紧凑版方差/稳健性 JSON 打包脚本（产出结构与归档件一致）
    └── test_generation_eval.py  # 生成层评测离线单测（规则判据 / 缓存键 / rubric 防漂移，不调 API、不载大模型）
```

## CI 门禁（`.github/workflows/ci.yml`）

`ubuntu-latest` + Python 3.12 + `pip` 依赖缓存 + `pytest`，**离线可跑、不需要密钥、不下载模型**。

> **宿主如实说明**：本仓库托管在 **Gitee（公开仓库）**，Gitee 不会执行 GitHub Actions，也还没接 Gitee Go。
> 这份 YAML 用的是**通用 GitHub Actions 格式**——镜像一份到 GitHub 即可直接生效，但在 Gitee 侧目前不是自动跑的流水线。
> 也因此 README 里**不放 CI 徽章**：**Gitee 不执行 GitHub Actions，徽章指向 GitHub 会 404**。

**本仓库比另外两个多一个坑：装 torch。** Linux 上从默认 index 装 torch 会拉 **CUDA 版（2–3GB）**，而 runner 没有 GPU——纯浪费带宽。所以 workflow 先用 **CPU 版官方 index**（`--index-url https://download.pytorch.org/whl/cpu`）显式装 CPU-only torch，再装 `requirements.txt`；因为 `torch>=2.0.0` 已被满足，第二步不会再把 CUDA 版拉回来（本机 `pip install --dry-run -r requirements.txt` 实测输出 `Requirement already satisfied: torch>=2.0.0 ... (2.14.0+cu132)`）。

| 环境变量 | 作用 |
|---|---|
| `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` | runner 上没有 BGE 模型缓存。这两条把"模型缺失"从**静默联网下载几百 MB**变成**立刻报错**（本机清空 `HF_HOME` 实测：`OSError: We couldn't connect to 'https://huggingface.co'`）——宁可红得明确，不要绿得含糊 |

**门禁的能力缺口**：108 个用例里有 **1 个必须加载真实 BGE 模型 + GPU**（`tests/test_pipeline.py::test_embedding`：构造 `EmbeddingEngine` 即 `SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda")`），CI 上必然失败，已用 `--deselect` 在**命令行显式排除**（没改测试代码，汇总行会报 `1 deselected`，配额对不上就是配置漂了）。其余 107 个用例（切分 / BM25 / RRF / 三档检索模式 / 重排覆盖 / 查询改写 / 生成层规则判据 / 裁判引擎的设备搬运与兜底标记 / 多次采样取中位数与部分采样失败 / 配置校验与「死配置不许复活」/ 降级档位标「未实测」/ embedding 缓存键含模型名 / 切分策略 fail-fast / 每臂有效 n / 紧凑件打包脚本）都是手写小矩阵 + 桩对象，离线确定性，CI 全跑。

本地一键复现同一条门禁（照抄可跑）：

```bash
python -m pip install -r requirements.txt && python -m pytest -q -rs \
  --deselect tests/test_pipeline.py::test_embedding
```

```text
........................................................................ [ 67%]
...................................                                      [100%]
107 passed, 1 deselected in 8.58s
```

> 本机实测（2026-10-20 复跑）：耗时随机器状态浮动，**用例数才是硬指标**，别把秒数当基准线。

本地**全量**（含那 1 个需要 BGE 缓存的用例；本机有缓存 + GPU）：

```bash
python -m pytest -q -rs
```

```text
........................................................................ [ 66%]
....................................                                     [100%]
108 passed in 10.27s
```

> 本机实测（2026-10-20 复跑）：耗时随机器状态浮动，**用例数才是硬指标**，别把秒数当基准线。

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

## 生成层忠实度实测：本地 1.5B 裁判到底够不够用（2026-10-19）

前面的基准只评**检索层**。这里补上**生成层忠实度**的实测：把同一批答案交给三条互为对照的评分臂，用实测数字回答「能不能拿仓库自带的 1.5B 小模型当评测裁判」。

- 报告：[`experiments/results/generation_eval_2026-10-19.md`](experiments/results/generation_eval_2026-10-19.md)
- 原始数据：[`experiments/results/generation_eval_2026-10-19.json`](experiments/results/generation_eval_2026-10-19.json)
- 复现：`python experiments/run_generation_eval.py`（离线部分加 `--no-api`；`--from-results` 可只重算规则臂、不重跑模型/不花钱）

**方差 / 稳健性两份报告的复现链（三步，2026-10-20 补齐）**：
[`judge_variance_2026-10-20.md`](experiments/results/judge_variance_2026-10-20.md)（本地裁判 N=7）与
[`reference_robustness_2026-10-20.md`](experiments/results/reference_robustness_2026-10-20.md)（强裁判 API 侧 N=3）
此前**仓库里没有任何代码能产出它们**（紧凑件的 `script` 字段自称来自 `run_generation_eval.py`，
但那个脚本的输出结构是另一套）。现在补上了中间那一步：

```bash
# ① raw（含答案/上下文全文）
python experiments/run_generation_eval.py --no-api --judge-samples 7 --out <临时目录>/n7.json
# ② 紧凑件（只留裁判档位 + 统计量；统计量现场复算）——新增脚本
python experiments/package_variance_json.py --input <临时目录>/n7.json \
  --out <临时目录>/judge_variance.json --elapsed-s 149.3
# ③ bootstrap 表（读②的紧凑件）
python experiments/bootstrap_judge_variance.py
```

用同一份 raw 跑 ②，产出与仓库里归档的紧凑件**逐字段一致**（822 / 810 个字段，数值容差 0），
唯一差别是 `script` 字段——归档件里它写的是 `run_generation_eval.py`（不实：那个脚本产不出这个结构），
打包脚本改成指向自己。API 侧的 raw 可由已入库的 `experiments/results/api_judge_cache.json`
**零花费重放**（缓存 key 含 N / 第几次采样 / temperature）。

三臂的设置（**三臂共用同一份检索上下文**，答案一律由本地 1.5B 在 `temperature=0` 下贪心生成、可复现）：

| 臂 | 角色 | 实现 |
|---|---|---|
| A 规则判据 | 零成本、确定性基线 | 数字/实体落地率、句子归因（字符 2-gram 覆盖率 ≥ 0.6）、拒答判定、EM / F1 / ROUGE-L / BLEU，映射到 1–5 粗分 |
| B 本地 1.5B 裁判 | 被测对象 | `Qwen2.5-1.5B-Instruct`（与生成模型**同款**，即自评），沿用仓库 `evaluation.py` 的判据文本 |
| C 强裁判 API | 参照系 | DeepSeek `deepseek-chat`（实测解析到 `deepseek-flash`），一次调用返回 3 维 JSON、`temperature=0` |

**实测结果（33 条样本 = 27 条原金标 + 6 条本次新造的拒答样本）**：

| 臂 | faithfulness | relevance | completeness |
|---|---|---|---|
| A 规则判据 | 3.45 | — | — |
| B 本地 1.5B 裁判 | 3.12 | 3.00 | 2.58 |
| C 强裁判 API | 4.27 | 4.48 | 3.82 |

臂 B（本地 1.5B）vs 臂 C（强裁判）在 faithfulness 上的一致率：**精确一致 0.21、±1 一致 0.48、MAE 1.82、Pearson r ≈ −0.03**。

**一句话结论（含对我们不利的实测）**：本批样本上**本地 1.5B 的裁判分数与强裁判几乎不相关（r≈0），且"中庸化回归"**——强裁判判低（≤2）的样本它偏**高**（本地均 3.17 vs 强裁判 1.67），判高（≥4）的偏**低**（3.11 vs 4.85），最该抓的编造（`unanswerable#2` 川菜菜谱幻觉）反被放行。反而**零成本、离线的规则判据更贴近强裁判**（faithfulness MAE 0.92 vs 1.82，精确一致 0.55 vs 0.21）。因此「用本地 1.5B 做裁判」在本批样本上**不成立**——不是"有点噪声"，而是"与目标信号无关"。它仍可用于「明显编造 vs 明显有据」的两端粗筛，或网络/预算完全不可用时的占位；需要排序、需要区分"好与更好"的场景都不该用它。

**边界（详见报告第七节）**：样本量小（33 条）只能给量级判断；臂 B 是**自评**（判分模型＝生成模型），绝对分不可作为质量证据；只做两裁判一致性、不是与人类标注对齐；强裁判本身也不是真值；**强裁判自身的方差已于 2026-10-20 实测**（2026-10-19 那一轮每样本只调一次 N=1，故当时无法给出）——API 侧以 N=3 补测（**99 次新调用、0 缓存命中、¥0.0777**）：方差**很小但非零**（faithfulness 的 item 级 sd 0.100、99 个单元里 71.7% 三次完全相同、但仍有 3 个单元极差 ≥3 档）；**把参照系换成 N=3 中位数后主结论存活**——本地裁判 MAE 1.9394 / r −0.1595、规则臂 MAE 1.0061 / r +0.5485，与 N=1 口径同量级，详见 [`experiments/results/reference_robustness_2026-10-20.md`](experiments/results/reference_robustness_2026-10-20.md)。多次采样取中位数**已实现**（`evaluation.judge_samples` / `--judge-samples N`，默认 1，N=1 与历史行为一致），**本地裁判一侧已于 2026-10-20 离线实测 N=7**（裁判抽样噪声确实降低但边际递减、整体结论不变，见 [`experiments/results/judge_variance_2026-10-20.md`](experiments/results/judge_variance_2026-10-20.md)）。2026-10-19 那一轮的费用：29 次新调用 + 4 次缓存命中，输入 19,248 / 输出 638 token，估 **¥0.0232**（硬上限 ¥0.5）。

## 评测口径

- 评测数据集 `data/sample/test_qa.json` 使用**文档级标注**（`relevant_sources`）：chunk 边界随切分策略变化，chunk 索引无法跨策略对照，因此以「是否检索到正确文档」为稳定口径，由 `EvaluationEngine.evaluate_retrieval_doc_level` 计算（内部复用 Recall / Precision / MRR 公式）。
- 三层指标：检索层（Recall@3 / Recall@5 / Precision@3 / MRR）、生成层（faithfulness / relevance / completeness，LLM-as-Judge）、传统层（BLEU / ROUGE-L）。
- 生成层需要 Qwen 模型或 API；未配置时回退到启发式评分（仅用于跑通流程，不作为正式指标）。**该回退不再静默**：`evaluate_generation(return_meta=True)` 会在 `meta.fallbacks` 里标出哪些维度用了兜底、原因是什么（报告层可直接打印）。
- 生成层的判分**质量**有实测：见上文「生成层忠实度实测」与 [`experiments/results/generation_eval_2026-10-19.md`](experiments/results/generation_eval_2026-10-19.md)——本地 1.5B 裁判与强裁判几乎不相关（faithfulness r≈0），库内小规模场景建议用规则判据做回归门禁、用 API 做正式裁判，不要默认信任本地小模型当裁判。

## 设计决策

**Q: chunk_size 为什么默认 512？**
> `experiments/run_strategy_comparison.py` 提供 128–1024 的完整扫描入口，可一键复跑。在仓库自带的样例数据集上各粒度 `recall@3` 均为 1.0、无显著差异；512 是工程默认值，面向「一个 chunk 覆盖一段完整技术要点」的常见文档形态，更大规模语料下的选型需要更大评测集验证。

**Q: LLM-as-Judge 打分准吗？**
> **已实测，答案是不准——至少本机 1.5B 这一档不准。** 实测（33 条，2026-10-19）：用仓库自带的 `Qwen2.5-1.5B-Instruct`（与生成模型同款，即自评）当裁判，其 faithfulness 分与强裁判（DeepSeek）**Pearson r ≈ −0.03、MAE 1.82、精确一致率仅 0.21**，且呈"中庸化回归"（低分样本偏高、高分样本偏低），把一条明确的编造答案放行。反直觉的是：**零成本、离线的规则判据反而更贴近强裁判**（MAE 0.92、精确一致 0.55）。结论与取舍——正式裁判用强模型/API，回归门禁用确定性规则判据，本地 1.5B 裁判只在「两端粗筛/离线占位」时用。完整数据、分段结论与规则判据的 5 类已知误判见 [`experiments/results/generation_eval_2026-10-19.md`](experiments/results/generation_eval_2026-10-19.md)。
>
> 实现层面有三点值得单独说（2026-10-20）：**①「多次评分取中位数」已实现且已于 2026-10-20 以 N=7 实测**——`config.yaml` 的 `evaluation.judge_samples`（CLI `--judge-samples N`）支持同一答案调 N 次取中位数，N=1 时与历史行为逐比特一致；上面报告的 33 条数字仍全部是 N=1 跑出来的。N=7 的方差实测结论：**裁判抽样噪声确实降了（MAE 的抽样 sd 降约 29%、r 的抽样 sd 也降约 29%，N=15 时降约 41%），但边际递减、判分开销线性增长；整体结论不变**——median-of-7 与强裁判仍是近零相关（|r| ≈ 0.14 < 0.25）、MAE 1.94 仍高于规则臂一倍，且 `unanswerable#2` 的幻觉反而被放行得更松（3 → 4）。因此「本地 1.5B 不适合作裁判」对聚合方式稳健，换聚合方法救不了它。n=33 只能给量级，r 不应引用到小数点后四位。详见 [`experiments/results/judge_variance_2026-10-20.md`](experiments/results/judge_variance_2026-10-20.md)。**②裁判的静默降级已修掉**：此前 `_judge_single_dim` 缺设备搬运（CPU 张量喂 CUDA 模型）+ 异常被吞，导致 GPU 上裁判分**全部**退化成按回答长度打分的启发式且看不出来；现在兜底会写进 `meta.fallbacks`、以 warning 日志点名、并由报告层逐条打印（详见 [`docs/architecture.md`](docs/architecture.md) §6.4 附录）。**③参照系稳健性已实测（2026-10-20，强裁判 API 侧 N=3）**：强裁判那一侧也补跑了 N=3（99 次新调用、0 缓存命中、**¥0.0777**），于是可以把「参照系」从 N=1 贪心换成 N=3 中位数再看一遍主结论——**结论存活**：本地 1.5B 裁判仍是近零/负相关（faithfulness MAE 1.9394 / r −0.1595），仍是零成本规则臂（MAE 1.0061 / r +0.5485）的约 2 倍差；且强裁判与它自己的 N=1 参照几乎一致（MAE 0.11 档、r +0.92~0.94），说明**两个裁判之间的分歧不是参照系噪声造成的**。详见 [`experiments/results/reference_robustness_2026-10-20.md`](experiments/results/reference_robustness_2026-10-20.md)。

**Q: 为什么检索是 NumPy 而不是向量数据库？**
> 本项目的知识库规模是「单机小知识库」（几十~几百个 chunk），暴力余弦检索在这个量级下复杂度完全不构成瓶颈，且零额外依赖、行为完全可解释。`config.yaml` 里 `retrieval.vector_store` 目前**只接受 `numpy`**：写 `chromadb` / `faiss` 会在加载配置时直接报错并提示改回 `numpy`（不再保留「配置里有个永远不生效的选项」这种不诚实的预留）。规模上量后（万级以上 chunk）需要先真正实现向量库接入，再放开这个白名单。

**Q: 配置里那些「看起来能调」的键是真的吗？**
> 2026-10-20 起这条纪律写进了代码与单测（`tests/test_config_honesty.py`）：**配置里不许有永不生效的选项**。
> - **已删除的死键**：`evaluation.judge_model`（裁判恒等于生成模型，代码从不读它）、`evaluation.retrieval_metrics` 与 `evaluation.traditional_metrics`（读了不用：检索/传统指标是代码里固定的一套）、`retrieval.persist_dir`（没有任何代码读取）、`retrieval.query_rewrite.max_new_tokens`（从未传给 LLM，改写生成的 token 上限跟随 `generation.max_new_tokens`）。
> - **保留但 fail-fast 的扩展位**：`retrieval.vector_store` 与 `retrieval.hybrid.rerank.type` 只能写当前唯一支持的值（`numpy` / `cross_encoder`），写别的值加载配置即报错并给出替代方案——不静默按默认值跑。
> - **真的接上线的键**：`app.title` 与 `evaluation.test_data_path` 由 `app.py` 读取（此前 `app.py` 把标题与数据集路径写死在代码里，配置里的同名键是死的）。

## 技术栈

- **Embedding**：sentence-transformers / BGE-small-zh-v1.5（512 维）
- **向量检索**：NumPy 余弦相似度（暴力检索 + 阈值过滤）
- **生成**：Qwen2.5-1.5B-Instruct（本地）/ OpenAI 兼容 API
- **界面**：Gradio
- **部署**：Docker + docker-compose

## License

MIT
