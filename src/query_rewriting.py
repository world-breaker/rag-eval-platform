"""查询改写（Query Rewriting）：HyDE 与 multi-query，配置开关，**默认关闭**。

为什么需要查询改写？
- 检索的失败常常不是「排序算法不行」，而是**问句和文档的表述对不上**：
  用户口语化、术语不一致、缺少上下文；而文档是书面技术表述。
  这时再强的向量模型也难救——它与文档表述不同，与真实文档的语言分布更接近。
- HyDE（Hypothetical Document Embeddings）：先让 LLM 生成一段「假想答案文档」，再用**假想文档的向量**去检索。
  假想文档的用词/句式更接近知识库里的真实文档，缓解 query–document 的表述 gap。
- multi-query：把一个问题改写成多条不同角度的子查询，各自召回后用 RRF 融合，
  提升召回率（对 BM25 这类纯词面匹配尤其明显：改写引入的同义/上下位词能命中更多文档）。

设计取舍（面试常问「为什么不默认开」）：
1. **默认关闭（enabled=false）**：改写要多跑 N 次 LLM 生成，延迟从「毫秒级」拉到「秒级」；
   而在「强向量 + 干净小库」上几乎拿不到收益（见 experiments/results 的实测表）。
   收益低而代价高，所以它是**按场景开启的可选项**，不是默认路径。
2. **LLM 依赖注入**：`llm_fn: Callable[[str], str]` 由调用方传入（生产用 GenerationEngine，
   单测用 stub）。这样改写逻辑可以离线、确定性地单测，不绑定具体模型。
3. **失败即降级**：LLM 报错、输出不可解析、生成的查询与原查询重复——一律丢弃，
   最坏情况退化成「原查询单路检索」，绝不因为改写失败导致检索不可用。
"""

import re
from typing import Callable, Dict, List, Optional

from src.utils import logger

# 指令里的占位符：{question} / {num}
MULTI_QUERY_PROMPT = """你是检索系统的查询改写助手。请把用户的问题改写成 {num} 个不同角度的检索查询，\
用于从技术知识库中召回相关文档。

要求：
1. 每条查询独立成行，只输出查询本身，不要编号、不要解释、不要引号；
2. 覆盖不同的表述方式（同义术语、上位词、具体细节角度），不要简单重复原句；
3. 保持中文，长度控制在 30 字以内。

用户问题：{question}

改写查询："""

HYDE_PROMPT = """请针对下面的问题，写一段 100~200 字的专业技术说明，\
假装它是从某篇技术文档里摘出来的正文章节。

要求：
1. 直接输出说明文字，不要标题、不要编号、不要解释你在做什么；
2. 使用技术文档的书面表述（术语、定义、机制、公式名等），不要口语化；
3. 内容可以不完全准确，目标是覆盖可能相关的术语与上下文。

问题：{question}

技术说明："""

_ITEM_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.、)]|\(\d+\))\s*")


def _clean_lines(raw: str) -> List[str]:
    """把 LLM 的自由文本输出解析为「一行一条」的查询列表。"""
    out: List[str] = []
    for line in (raw or "").splitlines():
        line = _ITEM_PREFIX_RE.sub("", line.strip()).strip().strip("“”\"'")
        if not line:
            continue
        # 有些模型会输出「1. 查询：xxx」这种带标签的格式
        for label in ("改写查询：", "查询：", "Query:", "query:"):
            if line.startswith(label):
                line = line[len(label) :].strip()
        if line:
            out.append(line)
    return out


class QueryRewriter:
    """查询改写器：multi-query / HyDE，通过注入的 LLM 生成。

    rewrite() 永远返回一个可用结果：LLM 不可用时退化为「原查询 + 无假想文档」。
    """

    def __init__(self, config: dict, llm_fn: Optional[Callable[[str], str]] = None):
        cfg = (config.get("retrieval", {}) or {}).get("query_rewrite", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.strategy = cfg.get("strategy", "multi_query")  # multi_query | hyde | both
        self.num_queries = int(cfg.get("num_queries", 3))
        self.max_hypothetical_chars = int(cfg.get("max_hypothetical_chars", 512))
        self.llm_fn = llm_fn
        # 注：改写生成的 token 上限跟随 generation.max_new_tokens——
        # 曾经这里有个 query_rewrite.max_new_tokens 键，但从未传给 LLM（读了不用），已删除。

    @property
    def active(self) -> bool:
        """改写是否真的会生效（开关打开 + 有可用 LLM）。"""
        return self.enabled and self.llm_fn is not None

    # ---------- 对外主入口 ----------

    def rewrite(self, question: str) -> Dict:
        """返回 {queries, hypothetical_doc, strategy, degraded}。

        - queries：参与检索的查询列表，**第一条永远是原查询**（保证不丢原始意图）；
        - hypothetical_doc：HyDE 生成的假想文档（未启用/失败为 None）；
        - degraded：是否发生了降级（便于实验报告如实标注）。
        """
        queries: List[str] = [question]
        hypothetical: Optional[str] = None
        degraded = False

        if not self.active:
            return {
                "queries": queries,
                "hypothetical_doc": None,
                "strategy": "passthrough",
                "degraded": False,
            }

        if self.strategy in ("multi_query", "both"):
            extra = self._multi_query(question)
            if not extra:
                degraded = True
            queries.extend(extra)

        if self.strategy in ("hyde", "both"):
            hypothetical = self._hyde(question)
            if not hypothetical:
                degraded = True

        return {
            "queries": queries,
            "hypothetical_doc": hypothetical,
            "strategy": self.strategy,
            "degraded": degraded,
        }

    # ---------- 两种改写策略 ----------

    def _call_llm(self, prompt: str) -> Optional[str]:
        if self.llm_fn is None:
            return None
        try:
            return self.llm_fn(prompt)
        except Exception as e:  # LLM 挂了不能拖垮检索
            logger.warning(f"查询改写调用 LLM 失败，退化为原查询：{e}")
            return None

    def _multi_query(self, question: str) -> List[str]:
        raw = self._call_llm(
            MULTI_QUERY_PROMPT.format(num=self.num_queries, question=question)
        )
        if not raw:
            return []

        seen = {question.strip()}
        extra: List[str] = []
        for line in _clean_lines(raw):
            line = line[:64]  # 防止模型跑偏输出长段落
            if line in seen:
                continue
            seen.add(line)
            extra.append(line)
            if len(extra) >= self.num_queries:
                break
        logger.info(f"multi-query 改写：{len(extra)} 条变体")
        return extra

    def _hyde(self, question: str) -> Optional[str]:
        raw = self._call_llm(HYDE_PROMPT.format(question=question))
        if not raw:
            return None
        doc = " ".join(_clean_lines(raw)) or raw.strip()
        doc = doc[: self.max_hypothetical_chars]
        logger.info(f"HyDE 假想文档：{len(doc)} 字符")
        return doc or None
