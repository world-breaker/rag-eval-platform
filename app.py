"""Gradio Web 界面：文档上传 → 问答 → 评测报告"""

import json
import gradio as gr
from pathlib import Path
from src.pipeline import RAGPipeline
from src.utils import logger

# 初始化
pipeline = RAGPipeline("config.yaml")
index_built = False


def build_from_text(text: str, source_name: str = "用户上传"):
    """从文本构建索引"""
    global index_built
    if not text.strip():
        return "⚠️ 请先输入或上传文档内容"

    pipeline.build_index([{"text": text, "source": source_name}])
    index_built = True

    return f"✅ 索引构建完成：{len(pipeline.chunks)} 个 chunk"


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


def ask_question(question: str, top_k: int = 5):
    """提问"""
    if not index_built:
        return "⚠️ 请先构建知识库索引", "", ""

    if not question.strip():
        return "请输入问题", "", ""

    try:
        result = pipeline.ask(question, top_k=top_k, return_context=True)

        answer = result["answer"]
        context = result.get("context", "")

        # 摘要
        summary = f"检索到 {len(result['retrieved_docs'])} 条相关文档\n"
        summary += f"最高相关度：{result['retrieved_docs'][0]['score']:.3f}\n" if result["retrieved_docs"] else "无相关文档\n"

        return answer, context, summary
    except Exception as e:
        logger.error(f"问答失败：{e}")
        return f"❌ 错误：{e}", "", ""


def run_evaluation(test_json: str = ""):
    """运行评测"""
    if not index_built:
        return "⚠️ 请先构建知识库索引"

    # 使用内置测试数据
    test_path = Path("data/sample/test_qa.json")
    if test_path.exists():
        reports = pipeline.batch_evaluate(str(test_path))
        return json.dumps(reports, ensure_ascii=False, indent=2)
    else:
        return "未找到测试数据文件 data/sample/test_qa.json"


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

with gr.Blocks(title="RAG 知识库问答 + 评测平台", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 RAG 知识库问答 + 评测平台")
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
                build_status = gr.Textbox(label="状态", interactive=False)

        build_btn.click(fn=build_from_text, inputs=[doc_text], outputs=[build_status])
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
        ask_btn = gr.Button("🔍 提问", variant="primary")

        answer_output = gr.Textbox(label="回答", lines=8)
        with gr.Accordion("📊 检索上下文（点击展开）", open=False):
            context_output = gr.Textbox(label="相关文档片段", lines=10)
        retrieval_summary = gr.Textbox(label="检索摘要", interactive=False)

        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, context_output, retrieval_summary],
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
