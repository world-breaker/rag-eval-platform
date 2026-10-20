"""文档切分模块：支持三种切分策略"""

import re
from typing import List, Dict
from src.utils import logger

# 支持的切分策略（未知策略直接报错，不静默回退）
SUPPORTED_STRATEGIES = ("fixed_size", "recursive", "semantic")


class DocumentChunker:
    """文档切分器：固定窗口 / 递归分隔符 / 语义段落三种策略。"""

    def __init__(self, config: dict):
        chunk_cfg = config["chunking"]
        self.strategy = chunk_cfg["strategy"]
        self.chunk_size = chunk_cfg["chunk_size"]
        self.chunk_overlap = chunk_cfg["chunk_overlap"]
        self.separators = chunk_cfg.get("separators", ["\n\n", "\n", "。", "."])
        # 配置写错策略名要立刻炸：静默回退 fixed_size 会让「三种策略对比」
        # 变成两个策略其实是同一个（假对比），而且看不出来。
        self.validate_strategy(self.strategy)

    @staticmethod
    def validate_strategy(strategy: str) -> str:
        """策略名必须在白名单里，否则 ValueError（对齐 src/utils.py 的 fail-fast 风格）。"""
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"chunking.strategy={strategy!r} 未接入代码，"
                f"当前只支持 {SUPPORTED_STRATEGIES}。"
            )
        return strategy

    def chunk(self, text: str, metadata: dict = None) -> List[Dict]:
        """切分文本，返回带元数据的 chunk 列表"""
        if self.strategy == "fixed_size":
            chunks = self._fixed_size(text)
        elif self.strategy == "recursive":
            chunks = self._recursive_split(text)
        elif self.strategy == "semantic":
            chunks = self._semantic_split(text)
        else:
            # 运行时被改坏（例如 compare_chunk_strategies 直接赋值 strategy）同样要炸
            raise ValueError(
                f"未知切分策略 {self.strategy!r}，当前只支持 {SUPPORTED_STRATEGIES}。"
            )

        # 附加元数据
        result = []
        for i, chunk_text in enumerate(chunks):
            result.append(
                {
                    "text": chunk_text.strip(),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk_text),
                    **(metadata or {}),
                }
            )
        logger.info(
            f"切分完成：{len(result)} 个 chunk（策略={self.strategy}, "
            f"size={self.chunk_size}, overlap={self.chunk_overlap}）"
        )
        return result

    def _fixed_size(self, text: str) -> List[str]:
        """固定窗口滑动切分"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _recursive_split(self, text: str) -> List[str]:
        """递归按分隔符切分，优先用大分隔符"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for sep in self.separators:
            if sep in text:
                parts = re.split(f"({re.escape(sep)})", text)

                # 合并分隔符到前面的文本块
                merged = []
                buf = ""
                for part in parts:
                    buf += part
                    if any(buf.endswith(s) for s in self.separators):
                        merged.append(buf.rstrip())
                        buf = ""
                if buf.strip():
                    merged.append(buf)

                # 对每段递归或按长度切
                result = []
                for segment in merged:
                    if len(segment) <= self.chunk_size:
                        if segment.strip():
                            result.append(segment)
                    else:
                        # 该段太长，退回 fixed_size
                        result.extend(self._fixed_size(segment))
                return result

        # 没有分隔符能切，退回固定大小
        return self._fixed_size(text)

    def _semantic_split(self, text: str) -> List[str]:
        """按自然段落切分（近似语义边界）"""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) <= self.chunk_size:
                current += "\n" + para if current else para
            else:
                if current.strip():
                    chunks.append(current)
                # 如果单个段落超长，递归处理
                if len(para) > self.chunk_size:
                    sub_chunks = self._recursive_split(para)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = para

        if current.strip():
            chunks.append(current)

        return chunks
