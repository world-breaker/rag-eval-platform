"""Gradio Web 界面：文档上传 → 问答 → 评测报告"""

import json
import time
import gradio as gr
from pathlib import Path
from src.pipeline import RAGPipeline
from src.utils import logger

# 初始化
# load_generator=False：生成模型延迟到首次真正生成时才加载（RAGPipeline.ask 内部 _ensure_generator）。
# 好处：界面秒起；只想演示检索模式时完全不加载生成模型（未下载也能跑）。
pipeline = RAGPipeline("config.yaml", load_generator=False)
index_built = False

# 检索模式下拉：label → 内部 mode 值。默认值取 config.yaml 里的 retrieval.mode，
# 保证「默认行为 = 历史行为」（零回归）。
MODE_CHOICES = [
    ("纯向量（默认，语义相似）", "vector"),
    ("纯 BM25（关键词精确匹配）", "bm25"),
    ("混合 BM25 + 向量（RRF 融合，可加重排）", "hybrid"),
]
DEFAULT_MODE = pipeline.retrieval_mode

# 结果表头：把「这条结果是谁捞上来的、融合后涨了没」摊开给面试官看
RESULT_HEADERS = [
    "#", "来源文档", "向量余弦", "BM25 分", "RRF 融合分",
    "向量排名", "BM25 排名", "重排分", "片段预览",
]

SAMPLE_KB = Path("data/sample/knowledge_base_large.json")


def build_from_text(text: str, source_name: str = "用户上传"):
    """从文本构建索引"""
    global index_built
    if not text.strip():
        return "⚠️ 请先输入或上传文档内容"

    pipeline.build_index([{"text": text, "source": source_name}])
    index_built = True

    return f"✅ 索引构建完成：{len(pipeline.chunks)} 个 chunk"


def load_sample_kb():
    """一键载入内置示例知识库（14 篇技术文档），方便直接演示检索模式对比。"""
    global index_built
    if not SAMPLE_KB.exists():
        return "⚠️ 未找到示例知识库 data/sample/knowledge_base_large.json"

    docs = json.loads(SAMPLE_KB.read_text(encoding="utf-8"))
    pipeline.build_index(docs)
    index_built = True
    return f"✅ 已载入内置示例知识库：{len(docs)} 篇文档 → {len(pipeline.chunks)} 个 chunk"


def handle_file_upload(file):
    """处理文件上传"""
    if file is None:
        return "", "未选择文件"

    filepath = Path(file.name)
    suffix = filepath.suffix.lower()

    if suffix == ".txt":
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read()
    elif suffix == ".json":
        with open(file.name, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            content = "\n\n".join(
                [d.get("text", str(d)) for d in data if isinstance(d, dict)]
            )
        else:
            content = str(data)
    else:
        return "", f"不支持的文件格式：{suffix}（请上传 .txt 或 .json）"

    result = build_from_text(content, filepath.name)
    return content, result


def _rows_from_results(results):
    """把检索结果转成表格行，逐列暴露两路来源与融合分。"""
    rows = []
    for r in results:
        rows.append(
            [
                str(r.get("retrieved_rank", "")),
                str(r.get("source", "")),
                f"{r.get('score', 0.0):.4f}" if r.get("score") is not None else "—",
                f"{r.get('bm25_score', 0.0):.4f}" if r.get("bm25_score") is not None else "—",
                f"{r.get('fusion_score', 0.0):.4f}" if r.get("fusion_score") else "—",
                str(r.get("vector_rank") or "—"),
                str(r.get("bm25_rank") or "—"),
                f"{r['rerank_score']:.4f}" if r.get("rerank_score") is not None else "—",
                (r.get("text", "")[:60].replace("\n", " ") + "…"),
            ]
        )
    return rows


def ask_question(
    question: str,
    top_k: int = 5,
    retrieval_mode: str = DEFAULT_MODE,
    use_rerank: bool = False,
    retrieval_only: bool = False,
):
    """提问（可选检索模式 / 重排 / 只检索不生成）"""
    if not index_built:
        return "⚠️ 请先构建知识库索引", "", "", []

    if not question.strip():
        return "请输入问题", "", "", []

    rerank_note = ""
    if use_rerank and retrieval_mode != "hybrid":
        rerank_note = "⚠️ 重排只在混合模式下生效，已忽略该选项。\n"
        use_rerank = False

    try:
        pipeline.set_retrieval_mode(retrieval_mode)
        rerank_active = pipeline.set_rerank(use_rerank) if retrieval_mode == "hybrid" else False
        if use_rerank and not rerank_active:
            rerank_note += "⚠️ 重排模型不可用（未下载/离线），已退回 RRF 融合结果。\n"

        t0 = time.perf_counter()
        if retrieval_only:
            retrieved = pipeline._retrieve(question, top_k=top_k, mode=retrieval_mode)
            answer = "（仅检索模式：未调用生成模型）"
        else:
            result = pipeline.ask(question, top_k=top_k, return_context=True, mode=retrieval_mode)
            retrieved = result["retrieved_docs"]
            answer = result["answer"]
        latency_ms = (time.perf_counter() - t0) * 1000

        mode_label = dict((v, k) for k, v in MODE_CHOICES).get(retrieval_mode, retrieval_mode)
        used = [f"模式：{mode_label}"]
        if retrieval_mode == "hybrid":
            used.append("重排：开" if rerank_active else "重排：关")
        if pipeline.query_rewriter is not None and pipeline.query_rewriter.active:
            used.append(f"查询改写：{pipeline.query_rewriter.strategy}")

        rows = _rows_from_results(retrieved)
        summary = rerank_note
        summary += " | ".join(used) + "\n"
        summary += f"检索到 {len(retrieved)} 条文档；耗时 {latency_ms:.1f} ms"
        if rows:
            summary += f"；Top-1 = {rows[0][1]}"

        context = "\n\n".join(
            [
                f"[chunk {r['chunk_index']}, 向量={r.get('score')}, BM25={r.get('bm25_score')}, "
                f"RRF={r.get('fusion_score')}] {r['text'][:200]}..."
                for r in retrieved[:3]
            ]
        )
        return answer, context, summary, rows
    except Exception as e:
        logger.error(f"问答失败：{e}")
        return f"❌ 错误：{e}", "", "", []


def run_evaluation(test_json: str = ""):
    """运行评测"""
    if not index_built:
        return "⚠️ 请先构建知识库索引"

    # 评测数据集路径来自 config.yaml 的 evaluation.test_data_path（配置项真的被用上）
    cfg_eval = pipeline.config.get("evaluation", {}) or {}
    test_path = Path(cfg_eval.get("test_data_path", "data/sample/test_qa.json"))
    if test_path.exists():
        reports = pipeline.batch_evaluate(str(test_path))
        return json.dumps(reports, ensure_ascii=False, indent=2)
    else:
        return f"未找到测试数据文件 {test_path}"


def compare_strategies(text: str):
    """对比 chunk 策略"""
    if not text.strip():
        return "请先输入文档内容"

    results = pipeline.compare_chunk_strategies(
        [{"text": text, "source": "benchmark"}],
        ["请总结本文档的主要内容", "文档中提到了哪些关键技术"],
    )

    output = []
    for strategy, data in results.items():
        output.append(f"## {strategy}")
        output.append(f"平均检索分：{data['avg_retrieval_score']}")
        output.append("")

    return "\n".join(output)


# ====== Gradio 界面 ======

# 界面标题来自 config.yaml 的 app.title（配置项真的被用上）
APP_TITLE = (pipeline.config.get("app", {}) or {}).get(
    "title", "RAG 知识库问答 + 评测平台"
)

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 📚 {APP_TITLE}")
    gr.Markdown("上传文档 → 构建知识库 → 自然语言问答 → 系统化评测")

    with gr.Tab("📄 文档管理"):
        gr.Markdown("### 上传文档构建知识库")
        with gr.Row():
            with gr.Column(scale=2):
                doc_text = gr.Textbox(
                    label="文档内容（可直接粘贴或上传文件）",
                    lines=15,
                    placeholder="在此粘贴文档内容，或使用右侧上传按钮...",
                )
            with gr.Column(scale=1):
                file_upload = gr.File(
                    label="上传文档（.txt / .json）",
                    file_types=[".txt", ".json"],
                )
                build_btn = gr.Button("🔨 构建索引", variant="primary")
                sample_btn = gr.Button("📦 载入内置示例知识库（14 篇）")
                build_status = gr.Textbox(label="状态", interactive=False)

        build_btn.click(fn=build_from_text, inputs=[doc_text], outputs=[build_status])
        sample_btn.click(fn=load_sample_kb, inputs=[], outputs=[build_status])
        file_upload.change(
            fn=handle_file_upload, inputs=[file_upload], outputs=[doc_text, build_status]
        )

    with gr.Tab("💬 问答"):
        gr.Markdown("### 基于知识库的智能问答")
        with gr.Row():
            question_input = gr.Textbox(
                label="你的问题", placeholder="输入问题...", scale=4
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1, label="检索数量 Top-K", scale=1
            )
        with gr.Row():
            mode_dropdown = gr.Dropdown(
                choices=[(label, value) for label, value in MODE_CHOICES],
                value=DEFAULT_MODE,
                label="检索模式",
                scale=3,
            )
            rerank_checkbox = gr.Checkbox(
                label="启用 Cross-Encoder 重排（仅混合模式生效）", value=False, scale=2
            )
            retrieval_only_checkbox = gr.Checkbox(
                label="仅检索（不调用生成模型，便于纯评测演示）", value=False, scale=2
            )
        ask_btn = gr.Button("🔍 提问", variant="primary")

        answer_output = gr.Textbox(label="回答", lines=8)
        results_table = gr.Dataframe(
            headers=RESULT_HEADERS,
            datatype=["str"] * len(RESULT_HEADERS),
            label="检索结果明细（各路来源 + 融合分）",
            interactive=False,
            wrap=True,
        )
        with gr.Accordion("📊 检索上下文（点击展开）", open=False):
            context_output = gr.Textbox(label="相关文档片段", lines=10)
        retrieval_summary = gr.Textbox(label="检索摘要", interactive=False)

        ask_btn.click(
            fn=ask_question,
            inputs=[
                question_input,
                top_k_slider,
                mode_dropdown,
                rerank_checkbox,
                retrieval_only_checkbox,
            ],
            outputs=[answer_output, context_output, retrieval_summary, results_table],
        )

    with gr.Tab("📊 评测"):
        gr.Markdown("### 系统化评测")
        gr.Markdown("对检索质量和生成质量进行三层评估")

        eval_btn = gr.Button("🚀 运行评测", variant="primary")
        eval_output = gr.Textbox(label="评测报告（JSON）", lines=20)

        eval_btn.click(fn=run_evaluation, inputs=[], outputs=[eval_output])

    with gr.Tab("🔬 实验对比"):
        gr.Markdown("### Chunk 策略对比实验")
        gr.Markdown("对同一知识库用不同切分策略建索引，观察检索排序的变化")
        compare_btn = gr.Button("🔬 运行对比实验", variant="primary")
        compare_output = gr.Textbox(label="对比结果", lines=10)
        compare_btn.click(
            fn=compare_strategies, inputs=[doc_text], outputs=[compare_output]
        )

if __name__ == "__main__":
    cfg = pipeline.config.get("app", {})
    demo.launch(
        server_name=cfg.get("host", "0.0.0.0"),
        server_port=cfg.get("port", 7860),
        share=cfg.get("share", False),
    )
