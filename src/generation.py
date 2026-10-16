"""生成模块：Prompt 拼接 + LLM 推理"""

from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.utils import logger, timer

RAG_PROMPT_TEMPLATE = """你是一个专业的智能问答助手。请严格根据以下参考文档回答问题。

## 参考文档
{context}

## 问题
{question}

## 回答要求
1. 只使用参考文档中的信息，不要编造
2. 如果文档中没有相关信息，请明确说"根据提供的文档无法回答该问题"
3. 回答简洁、准确，用中文

## 回答
"""


class GenerationEngine:
    """生成引擎：RAG Prompt 模板 + 本地模型 / OpenAI 兼容 API 两条推理路径。"""

    def __init__(self, config: dict):
        gen_cfg = config["generation"]
        self.model_name = gen_cfg["model_name"]
        self.device = gen_cfg["device"]
        self.max_new_tokens = gen_cfg["max_new_tokens"]
        self.temperature = gen_cfg["temperature"]
        self.top_p = gen_cfg["top_p"]
        self.use_api = gen_cfg.get("use_api", False)
        self.api_base = gen_cfg.get("api_base", "")
        self.api_model = gen_cfg.get("api_model", "")

        if not self.use_api:
            logger.info(f"加载生成模型：{self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            is_gpu = (self.device != "cpu") and torch.cuda.is_available()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if is_gpu else torch.float32,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
            )
            if not is_gpu:
                self.model = self.model.cpu()
            logger.info(f"生成模型加载完成 (device={self.device})")

    @timer
    def generate(self, question: str, retrieved_docs: List[Dict]) -> Dict:
        """
        根据检索结果生成回答。
        返回：{question, answer, retrieved_docs, prompt}
        """
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[文档{i}]（相关度: {doc['score']:.2f}）\n{doc['text']}")
        context = "\n\n".join(context_parts)

        # 拼 Prompt
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # 生成
        if self.use_api:
            answer = self._generate_api(prompt)
        else:
            answer = self._generate_local(prompt)

        logger.info(f"生成完成：{len(answer)} 字符")

        return {
            "question": question,
            "answer": answer.strip(),
            "retrieved_docs": retrieved_docs,
            "prompt": prompt,
        }

    def _generate_local(self, prompt: str) -> str:
        """本地模型推理"""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        if self.device == "cpu":
            inputs = {k: v.cpu() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 只取新生成的 token（去掉 prompt 部分）
        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        return response

    def _generate_api(self, prompt: str) -> str:
        """通过 API 推理（Ollama 兼容）"""
        import requests

        resp = requests.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.api_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
