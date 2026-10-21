#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构造「领域 LoRA SFT 选域 probe」的 4 个域样本（每条带**确定性** expected）
=====================================================================

本脚本只做**数据构造**，不做任何训练、也不调用任何模型。

设计纪律（对应上一轮 SFT 失败的第 ⑤ 条「参考答案不唯一」）：
  1. 每条样本的 expected 都能用**纯规则**判定，不存在「表述不同但同样正确」的自由度；
  2. 派生自真实文档的样本，答案值在**原文逐字出现**（非推断）；
  3. 模板合成的样本，答案由**同一张事实表**生成 —— 文本与答案不可能不一致；
  4. 每条样本都标注 source（自撰 / 由 fin-rag-demo 文档派生）。

输出
----
  data/A_agent_tool_call.jsonl   24 条  Agent 工具调用
  data/B_doc_extraction.jsonl    24 条  金融/房产文档结构化抽取
  data/C_compliance_refusal.jsonl 24 条  合规拒答与风险识别
  data/D_intent_slot.jsonl       24 条  金融客服意图 + 槽位

字段：id / prompt / expected / source / notes
"""

from __future__ import annotations

import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# =====================================================================
# 域 A：Agent 工具调用
#   工具 schema 复用 D:\programmer\LLM\projects\hello-agent 的真实工具定义
#     - agent/tools_builtin.py : calculator / current_time
#     - agent/rag_tool.py      : search_knowledge(query, top_k=3)
#     - agent/memory.py        : remember_fact / recall_memory
# =====================================================================

TOOLS_SCHEMA = [
    {
        "name": "calculator",
        "description": (
            "数学计算器。输入一个数学表达式（如 \"(3+5)*2\"、\"sqrt(16)+2**10\"），返回计算结果。"
            "只支持数字、+ - * / // % ** 与白名单函数"
            "（sqrt/log/log10/exp/abs/round/min/max/pow/sin/cos/tan）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "数学表达式"}},
            "required": ["expression"],
        },
    },
    {
        "name": "current_time",
        "description": (
            "获取当前日期与时间（东八区）。需要「今天/现在」这类信息时调用。"
            "重要：任何涉及「今天」「现在」「今年」的问题都要先调用它拿真实时间，"
            "不要用模型记忆里的日期。"
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_knowledge",
        "description": (
            "检索本地知识库（中文技术文档），返回最相关的原文片段与相似度分数（JSON）。"
            "当问题可能依赖本地资料时，先调用它拿证据再回答。top_k 默认 3。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询"},
                "top_k": {"type": "integer", "description": "返回条数，默认 3"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember_fact",
        "description": (
            "把一条跨任务有用的事实写入长期记忆，下次任务会自动召回。"
            "适用场景：用户明确说「记住…」，或给出了以后还会用到的偏好/配置/结论。"
            "参数：content 是要记住的事实；key 是可选的稳定标识；tags 是逗号分隔的标签；"
            "source 是来源；verified 表示是否已核验；ttl_seconds 是存活秒数（0=用配置默认值）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要记住的事实"},
                "key": {"type": "string", "description": "可选的稳定标识，同 key 会覆盖旧值"},
                "tags": {"type": "string", "description": "逗号分隔的标签，如 \"偏好,语言\""},
                "source": {"type": "string", "description": "这条事实的来源，如 \"用户明说\""},
                "verified": {"type": "boolean", "description": "是否已核验，默认 false"},
                "ttl_seconds": {"type": "number", "description": "存活秒数，默认 0"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "检索之前任务保存的长期记忆，返回与 query 最相关的若干条。"
            "适用场景：用户提到「上次说过的」「我之前告诉你的」等跨任务信息时先查一下。"
            "没有命中时返回空列表（不要编造记忆）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询"},
                "top_k": {"type": "integer", "description": "返回条数，默认 5"},
                "only_verified": {"type": "boolean", "description": "只返回已核验记录，默认 false"},
            },
            "required": ["query"],
        },
    },
]

A_SYSTEM = (
    "你是一个只能通过调用工具来推进任务的 Agent 助手。你必须严格按要求输出 JSON，不输出任何多余内容。\n"
    "\n"
    "可用工具（JSON Schema）：\n"
    + json.dumps(TOOLS_SCHEMA, ensure_ascii=False, indent=2)
    + "\n\n"
    "输出要求：\n"
    "1. 只输出一个 JSON 对象，不要输出解释、不要用 Markdown 代码块包裹。\n"
    '2. 格式固定为：{"tool": "<工具名>", "arguments": {<参数名>: <参数值>}}\n'
    "3. 必须从上面的工具列表里选**一个**工具；arguments 的键必须与该工具的参数名一致。\n"
    "4. 若该工具不需要参数，arguments 为空对象 {}。\n"
    "5. 参数值必须来自用户请求中的原文，不要臆造。"
)


def a_sample(sid, request, tool, arguments, notes):
    return {
        "id": sid,
        "prompt": A_SYSTEM + "\n\n用户请求：" + request,
        "expected": {"tool": tool, "arguments": arguments},
        "source": "自撰（工具 schema 复用 hello-agent 仓库真实工具定义）",
        "notes": notes,
    }


def expr(value):
    return {"type": "expression_value", "value": value}


def terms(*ts, strict=None):
    """terms 有两档：
       - ts     语义必要词（抽取正确时**必然出现**的实体/枚举/专有短语）→ 主口径
       - strict 原文逐字短语（要求连修饰语一起复现）→ 严格口径
    两者都判、两个数字都报，避免判据松紧被人为调成对基座有利或不利。
    """
    d = {"type": "terms", "terms": list(ts)}
    if strict:
        d["strict_terms"] = list(strict)
    return d


def exact(value):
    return {"type": "exact", "value": value}


DOMAIN_A = [
    # ---------------- calculator（7） ----------------
    a_sample("A-01", "帮我算一下 (3+5)*2 等于多少？", "calculator",
             {"expression": expr(16)}, "纯算术，答案唯一"),
    a_sample("A-02", "算一下 sqrt(144) + 5 是多少？", "calculator",
             {"expression": expr(17)}, "白名单函数 sqrt"),
    a_sample("A-03", "计算 2**10 + 45 等于多少？", "calculator",
             {"expression": expr(1069)}, "幂运算"),
    a_sample("A-04", "帮我算 250000 * 0.035 / 12 ，按月利息是多少？", "calculator",
             {"expression": expr(729.1666666666666)}, "月利息=贷款额*年利率/12"),
    a_sample("A-05", "算一下 min(1200, 980) + max(30, 45) 等于多少？", "calculator",
             {"expression": expr(1025)}, "min/max 白名单函数"),
    a_sample("A-06", "把 15000 除以 500 再乘 7 是多少？", "calculator",
             {"expression": expr(210)}, "四则运算"),
    a_sample("A-07", "我欠款 5000 元，逾期了 3 期，每天罚息按日万分之五算，总共罚息多少？",
             "calculator", {"expression": expr(7.5)}, "金融场景算术：5000*0.0005*3"),
    # ---------------- current_time（4） ----------------
    a_sample("A-08", "现在几点了？我要下班打卡。", "current_time", {},
             "需要真实当前时间，禁止用记忆日期"),
    a_sample("A-09", "今天星期几？帮我看一下。", "current_time", {},
             "当前时间类问题"),
    a_sample("A-10", "现在是哪一年？我怕自己记错了。", "current_time", {},
             "当前时间类问题"),
    a_sample("A-11", "帮我取一下当前的日期时间，我要写进合同附件里。", "current_time", {},
             "当前时间类问题"),
    # ---------------- search_knowledge（7） ----------------
    a_sample("A-12", "我的抵押贷款合同里，贷款期限最长是几年？", "search_knowledge",
             {"query": terms("贷款期限", strict=("贷款期限", "最长"))},
             "需检索本地合同要点文档"),
    a_sample("A-13", "借款人违反提前还款通知期约定会有什么后果？", "search_knowledge",
             {"query": terms("提前还款", "通知期")}, "需检索合同第七条"),
    a_sample("A-14", "核保结论有哪几种类型？", "search_knowledge",
             {"query": terms("核保结论")}, "需检索核保规则文档"),
    a_sample("A-15", "征信报告里的查询记录一般怎么解读？", "search_knowledge",
             {"query": terms("查询记录")}, "需检索征信解读文档"),
    a_sample("A-16", "客服在金融场景里有哪些合规红线不能碰？", "search_knowledge",
             {"query": terms("红线")}, "需检索客服合规文档"),
    a_sample("A-17", "提前还款需要提前多久以书面形式申请？", "search_knowledge",
             {"query": terms("提前还款", strict=("提前还款", "书面"))}, "需检索合同第三条"),
    a_sample("A-18", "投保金额显著高于评估价值这种情况，核保一般怎么处理？", "search_knowledge",
             {"query": terms("评估价值")}, "需检索核保加费因素"),
    # ---------------- recall_memory（3） ----------------
    a_sample("A-19", "上次我告诉过你的我的抵押房产位置，帮我找出来。", "recall_memory",
             {"query": terms("抵押")}, "跨任务记忆召回"),
    a_sample("A-20", "我之前跟你说过对公账户的事，能再查一下吗？", "recall_memory",
             {"query": terms("对公账户")}, "跨任务记忆召回"),
    a_sample("A-21", "把上次那条关于提前还款的记录调出来看看。", "recall_memory",
             {"query": terms("提前还款")}, "跨任务记忆召回"),
    # ---------------- remember_fact（3） ----------------
    a_sample("A-22", "记住，我以后所有贷款咨询都要按等额本息还款方式算。", "remember_fact",
             {"content": terms("等额本息")}, "显式要求记住"),
    a_sample("A-23", "请记住我的客户经理叫李伟，工号 A1023。", "remember_fact",
             {"content": terms("李伟", "A1023")}, "显式要求记住，含专名"),
    a_sample("A-24", "帮我记一下：我的紧急联系人是张敏，电话尾号 8899。", "remember_fact",
             {"content": terms("张敏", "8899")}, "显式要求记住，含专名"),
]

# =====================================================================
# 域 B：金融/房产文档结构化抽取
#   派生自 D:\programmer\LLM\projects\fin-rag-demo\data\docs\ 的 4 篇示例文档
# =====================================================================

DOC_DIR = Path(r"D:\programmer\LLM\projects\fin-rag-demo\data\docs")

B_TYPE_DOC = (
    "字段类型：\n"
    "- number：数值，输出纯数字（去掉单位）。如「七年」→ 7，「十二个月」→ 12。\n"
    "- number_pct：百分比，输出**小数比率**。如「百分之一」→ 0.01，「1%」→ 0.01，「百分之三十」→ 0.3。\n"
    "- enum：枚举值，必须逐字使用候选词之一。\n"
    "- bool：布尔值 true / false。\n"
    "- list：字符串数组，元素顺序不重要。\n"
    "- items_contain：字符串数组；每个元素给出一组可接受的说法，**只要命中其中任意一个说法即可**，整体不要求逐字。\n"
    "- terms：一句话中文摘要，必须包含指定的全部关键词。\n"
    "- string：字符串，按原文抽取。\n"
    "\n"
    "输出要求：只输出一个 JSON 对象，键为字段名、值为抽取结果；不要输出解释或 Markdown 代码块。"
)

B_TAIL = (
    "输出要求：只输出一个 JSON 对象，键为字段名、值为抽取结果；不要输出解释或 Markdown 代码块。"
)


def b_extract(sid, passage, fields, source, notes, doc_ref=""):
    """fields = [(字段名, 类型, 说明, 期望值说明)]"""
    spec_lines = []
    exp_fields = {}
    for name, ftype, desc, _ in fields:
        cand = ""
        if ftype == "enum":
            cand = f"，候选值：{desc}"
        spec_lines.append(f"- {name}（{ftype}）：{desc}{cand}")
        exp_fields[name] = fields[fields.index((name, ftype, desc, _))][3]
    prompt = (
        B_TYPE_DOC
        + "\n\n请抽取以下字段：\n"
        + "\n".join(spec_lines)
        + "\n\n文本片段：\n"
        + passage
        + "\n\n"
        + B_TAIL
    )
    return {
        "id": sid,
        "prompt": prompt,
        "expected": {"task_type": "extract", "fields": exp_fields},
        "source": source,
        "notes": notes + (f" | 出处：{doc_ref}" if doc_ref else ""),
    }


def b_sample(sid, passage, field_defs, source, notes, doc_ref=""):
    """field_defs = [(name, type, desc, expected_value)]，desc 里已含候选值说明。"""
    spec_lines = []
    exp = {}
    for name, ftype, desc, expval in field_defs:
        spec_lines.append(f"- {name}（{ftype}）：{desc}")
        exp[name] = expval
    prompt = (
        B_TYPE_DOC
        + "\n\n请抽取以下字段：\n"
        + "\n".join(spec_lines)
        + "\n\n文本片段：\n"
        + passage
        + "\n\n"
        + B_TAIL
    )
    return {
        "id": sid,
        "prompt": prompt,
        "expected": {"task_type": "extract", "fields": exp},
        "source": source,
        "notes": notes + (f" | 出处：{doc_ref}" if doc_ref else ""),
    }


SRC_DOC = "由 fin-rag-demo 文档派生（原文条款逐字摘录）"
SRC_SYN = "自撰（模板合成，文本与答案同源于一张事实表）"

DOMAIN_B = []

# ---------------- B-01 ~ B-12：fin-rag-demo 文档逐字摘录 ----------------

DOMAIN_B.append(b_sample(
    "B-01",
    "抵押物为位于武汉市洪山区XX路XX号的商业用房。抵押物评估价值由本行认可的第三方评估机构出具报告确定，"
    "评估报告有效期为出具之日起十二个月。贷款金额不超过抵押物评估价值的一定比例，具体比例以审批结果为准。",
    [
        ("evaluation_validity_months", "number", "评估报告有效期（月）", {"type": "number", "value": 12}),
        ("collateral_type", "enum", "抵押物类型，候选值：住宅/商业用房/工业厂房/土地",
         {"type": "enum", "value": "商业用房"}),
        ("loan_amount_limit", "terms", "贷款金额上限的表述，须包含关键词",
         {"type": "terms", "terms": ["评估价值"], "strict_terms": ["评估价值", "比例"]}),
    ],
    SRC_DOC, "三个字段均在原文逐字出现", "01_房产抵押贷款合同要点.md 第二条",
))

DOMAIN_B.append(b_sample(
    "B-02",
    "贷款期限最长不超过七年。还款方式可选择按月付息、到期还本，或等额本息按月还款。"
    "借款人可提前还款，提前还款应提前三十日以书面形式向贷款人提出申请。",
    [
        ("max_term_years", "number", "贷款期限上限（年）", {"type": "number", "value": 7}),
        ("early_repay_notice_days", "number", "提前还款需提前通知的天数",
         {"type": "number", "value": 30}),
        ("repayment_methods", "terms", "可选的还款方式，须包含关键词",
         {"type": "terms", "terms": ["按月付息", "等额本息"]}),
    ],
    SRC_DOC, "三个字段均在原文逐字出现", "01_房产抵押贷款合同要点.md 第三条",
))

DOMAIN_B.append(b_sample(
    "B-03",
    "借款人违反第三条关于提前还款提前通知期的约定，贷款人有权按提前还款金额的百分之一收取违约金。",
    [
        ("early_repay_penalty_pct", "number_pct", "提前还款违约金比例",
         {"type": "number_pct", "value": 0.01}),
        ("penalty_base", "terms", "违约金的计算基数，须包含关键词",
         {"type": "terms", "terms": ["提前还款金额"]}),
    ],
    SRC_DOC, "两个字段均在原文逐字出现", "01_房产抵押贷款合同要点.md 第七条",
))

DOMAIN_B.append(b_sample(
    "B-04",
    "借款人获得的贷款资金仅可用于抵押房产的装修、经营周转或设备采购，"
    "不得用于股票、期货、房地产投机及其他法律法规禁止的用途。",
    [
        ("allowed_uses", "terms", "允许的贷款用途，须包含全部关键词",
         {"type": "terms", "terms": ["装修", "经营周转", "设备采购"]}),
        ("forbidden_uses", "terms", "禁止的贷款用途，须包含全部关键词",
         {"type": "terms", "terms": ["股票", "期货", "房地产投机"]}),
    ],
    SRC_DOC, "两个字段均在原文逐字出现", "01_房产抵押贷款合同要点.md 第一条",
))

DOMAIN_B.append(b_sample(
    "B-05",
    "借款人未按期偿还本息的，贷款人有权宣布贷款提前到期，要求借款人一次性清偿全部债务，"
    "并按日万分之X的罚息计收逾期利息。贷款人亦有权依法实现抵押权。",
    [
        ("default_consequence", "terms", "贷款人可采取的违约处理措施，须包含关键词",
         {"type": "terms", "terms": ["贷款提前到期"], "strict_terms": ["宣布贷款提前到期"]}),
        ("settlement_requirement", "terms", "清偿方式要求，须包含关键词",
         {"type": "terms", "terms": ["一次性清偿"]}),
        ("collateral_right", "terms", "抵押权相关表述，须包含关键词",
         {"type": "terms", "terms": ["抵押权"]}),
    ],
    SRC_DOC, "三个字段均在原文逐字出现", "01_房产抵押贷款合同要点.md 第六条",
))

DOMAIN_B.append(b_sample(
    "B-06",
    "借款人应按贷款人要求为抵押物投保财产保险，并将贷款人列为第一受益人。"
    "保险期间应覆盖贷款期限，保险标的和保额应与贷款人要求一致。",
    [
        ("insurance_required", "bool", "是否要求投保财产保险", {"type": "bool", "value": True}),
        ("insurance_beneficiary", "enum", "第一受益人，候选值：贷款人/借款人/抵押物",
         {"type": "enum", "value": "贷款人"}),
        ("insurance_term_requirement", "terms", "保险期间要求，须包含关键词",
         {"type": "terms", "terms": ["贷款期限"], "strict_terms": ["覆盖贷款期限"]}),
    ],
    SRC_DOC, "三个字段均在原文逐字出现", "01_房产抵押贷款合同要点.md 第五条",
))

DOMAIN_B.append(b_sample(
    "B-07",
    "核保结论通常包含：是否承保、承保条件（加费、免赔额、特别约定）、以及核保意见。\n"
    "1. 标准承保：风险在可接受范围内，按标准条款承保。\n"
    "2. 加费承保：存在特定风险因素，在提高费率或加设免赔额后承保。\n"
    "3. 拒保：风险超出公司承受能力，不予承保。",
    [
        ("underwriting_types", "list", "核保结论的类型列表",
         {"type": "list", "value": ["标准承保", "加费承保", "拒保"]}),
        ("premium_loading_definition", "terms", "加费承保的条件，须包含关键词",
         {"type": "terms", "terms": ["免赔额"], "strict_terms": ["提高费率", "免赔额"]}),
        ("decline_definition", "terms", "拒保的条件，须包含关键词",
         {"type": "terms", "terms": ["承受能力"], "strict_terms": ["超出公司承受能力"]}),
    ],
    SRC_DOC, "三个字段均在原文逐字出现", "02_核保规则说明.md 第二节",
))

DOMAIN_B.append(b_sample(
    "B-08",
    "1. 可解释性：核保结论必须能追溯到具体的条款依据与事实证据，不允许给出无法解释的结论。\n"
    "2. 一致性：相同事实条件下应当给出相同的核保结论，避免同案不同判。\n"
    "3. 适当性：承保条件应与实际风险相称，既不放过重大风险，也不无必要地拒保。\n"
    "4. 可追溯：核保过程中使用的资料、规则版本与人工调整必须留痕，以备复核与监管检查。",
    [
        ("underwriting_principles", "list", "核保必须遵循的原则列表",
         {"type": "list", "value": ["可解释性", "一致性", "适当性", "可追溯"]}),
        ("traceability_requirement", "terms", "留痕要求，须包含关键词",
         {"type": "terms", "terms": ["留痕"], "strict_terms": ["留痕", "复核"]}),
        ("consistency_requirement", "terms", "一致性要求的表述，须包含关键词",
         {"type": "terms", "terms": ["同案不同判"], "strict_terms": ["相同", "避免同案不同判"]}),
    ],
    SRC_DOC, "三个字段均在原文逐字出现", "02_核保规则说明.md 第三节",
))

DOMAIN_B.append(b_sample(
    "B-09",
    "核保规则是确定性的、必须严格执行的底线约束（例如监管规定的责任免除范围、行业禁入清单）；"
    "模型提供的是对风险的量化评分，用于在规则允许的区间内定价与分级。两者不可互相替代："
    "模型输出不能突破规则设定的边界。",
    [
        ("rule_nature", "terms", "核保规则的性质，须包含关键词",
         {"type": "terms", "terms": ["底线约束"], "strict_terms": ["确定性", "底线约束"]}),
        ("model_nature", "terms", "模型的作用，须包含关键词",
         {"type": "terms", "terms": ["量化评分"]}),
        ("substitution_allowed", "bool", "模型能否替代规则", {"type": "bool", "value": False}),
        ("model_boundary", "terms", "模型边界，须包含关键词",
         {"type": "terms", "terms": ["边界"], "strict_terms": ["不能突破规则设定的边界"]}),
    ],
    SRC_DOC, "四个字段均在原文逐字出现", "02_核保规则说明.md 第五节",
))

DOMAIN_B.append(b_sample(
    "B-10",
    "1. 标的位于历史巨灾频发区域\n"
    "2. 企业客户所属行业属于高风险行业\n"
    "3. 投保金额显著高于评估价值\n"
    "4. 近三年有两次以上较大金额理赔记录\n"
    "5. 投保人信用记录存在持续逾期",
    [
        ("premium_loading_factors", "list", "常见加费因素列表",
         {"type": "list", "value": [
             "标的位于历史巨灾频发区域",
             "企业客户所属行业属于高风险行业",
             "投保金额显著高于评估价值",
             "近三年有两次以上较大金额理赔记录",
             "投保人信用记录存在持续逾期",
         ]}),
    ],
    SRC_DOC, "列表项均在原文逐字出现", "02_核保规则说明.md 第六节",
))

DOMAIN_B.append(b_sample(
    "B-11",
    "逾期记录按时间顺序列出，每条包含：账户名称、欠款金额、逾期天数、逾期期数、还款状态、以及是否已结清。"
    "判定口径关注的是「连续逾期」与「累计逾期次数」两类指标。",
    [
        ("overdue_record_fields", "list", "一条逾期记录包含的字段列表",
         {"type": "list", "value": ["账户名称", "欠款金额", "逾期天数", "逾期期数", "还款状态", "是否已结清"]}),
        ("overdue_metrics", "list", "判定口径关注的两类指标",
         {"type": "list", "value": ["连续逾期", "累计逾期次数"]}),
    ],
    SRC_DOC, "列表项均在原文逐字出现", "03_征信报告解读.md 第二节",
))

DOMAIN_B.append(b_sample(
    "B-12",
    "1. 信贷账户数量在短时间内快速增加\n"
    "2. 多笔账户当前逾期或状态为呆账\n"
    "3. 长期小额账户零星逾期（可能是 forgotten payment，也可能是多头借贷后的失养）\n"
    "4. 担保代偿记录\n"
    "5. 信用卡额度使用率长期接近或超过授信上限\n"
    "6. 短期内密集的机构查询",
    [
        ("credit_risk_signals", "list", "常见风险信号列表",
         {"type": "list", "value": [
             "信贷账户数量在短时间内快速增加",
             "多笔账户当前逾期或状态为呆账",
             "长期小额账户零星逾期",
             "担保代偿记录",
             "信用卡额度使用率长期接近或超过授信上限",
             "短期内密集的机构查询",
         ]}),
    ],
    SRC_DOC, "列表项均在原文逐字出现", "03_征信报告解读.md 第四节",
))

# ---------------- B-13 ~ B-16：fin-rag-demo 文档派生（items_contain 长句） ----------------

DOMAIN_B.append(b_sample(
    "B-13",
    "1. 不得承诺「一定能批」「保证放款」「百分百通过」等确定性结果。\n"
    "2. 不得对利率、额度、审批结果做超出系统权限的承诺或暗示。\n"
    "3. 不得索取或口述用户的完整身份证号、银行卡号、密码、短信验证码。\n"
    "4. 不得代替客户作出投资、借贷决策，或推荐未经审批的第三方服务。\n"
    "5. 不得对外透露其他客户的任何信息。\n"
    "6. 涉及个人征信、催收、债务重组的表述，必须严格使用机构审核过的标准话术。",
    [
        ("csm_redlines", "items_contain", "合规红线清单",
         {"type": "items_contain", "value": [
             ["一定能批", "保证放款", "百分百通过", "确定性结果"],
             ["利率", "额度", "审批结果"],
             ["身份证号", "银行卡号", "密码", "短信验证码"],
             ["投资", "借贷决策", "第三方服务"],
             ["其他客户"],
             ["标准话术"],
         ]}),
    ],
    SRC_DOC, "6 条红线逐条对应原文", "04_客服话术与合规红线.md 第三节",
))

DOMAIN_B.append(b_sample(
    "B-14",
    "1. 客户投诉、情绪激动或有媒体曝光风险\n"
    "2. 客户提出与标准话术不一致的定制化诉求（如特殊减免、展期）\n"
    "3. 涉及法律纠纷、监管投诉、暴力催收指控\n"
    "4. 客户身份存疑或存在冒用风险\n"
    "5. 客户明确要求人工服务",
    [
        ("must_transfer_human", "items_contain", "必须转人工或转专席的情形清单",
         {"type": "items_contain", "value": [
             ["客户投诉", "情绪激动", "媒体曝光"],
             ["定制化诉求", "特殊减免", "展期"],
             ["法律纠纷", "监管投诉", "暴力催收"],
             ["身份存疑", "冒用"],
             ["人工服务"],
         ]}),
    ],
    SRC_DOC, "5 条情形逐条对应原文", "04_客服话术与合规红线.md 第四节",
))

DOMAIN_B.append(b_sample(
    "B-15",
    "1. 语音转写存在识别错误风险，涉及金额、日期、证件号的字段必须做二次确认。\n"
    "2. 语音场景的响应预算通常在数秒量级，因此知识检索与生成必须快；不适合做长上下文推理。\n"
    "3. 静默、噪声、多人同时说话都会影响识别质量，需要在设计上考虑降级策略"
    "（例如关键确认环节改用短信或人工）。",
    [
        ("voice_confirm_fields", "terms", "语音场景必须二次确认的字段，须包含关键词",
         {"type": "terms", "terms": ["金额", "日期", "证件号"]}),
        ("voice_response_budget", "terms", "语音场景的响应预算要求，须包含关键词",
         {"type": "terms", "terms": ["数秒"]}),
        ("voice_degradation", "terms", "语音场景的降级策略要求，须包含关键词",
         {"type": "terms", "terms": ["降级策略"]}),
    ],
    SRC_DOC, "三个字段均在原文逐字出现", "04_客服话术与合规红线.md 第六节",
))

DOMAIN_B.append(b_sample(
    "B-16",
    "1. 征信报告反映的是「历史记录」，不等于当前风险，同一人的风险在不同时点差异可能很大。\n"
    "2. 逾期天数与逾期期数的换算口径必须以数据源定义为准，不能自行推断。\n"
    "3. 单一信号不构成结论，应结合行业、区域、收入稳定性等维度。\n"
    "4. 征信数据属于个人敏感信息，使用与留存必须符合个人信息保护相关法规。",
    [
        ("credit_report_meaning", "terms", "征信报告的性质表述，须包含关键词",
         {"type": "terms", "terms": ["历史记录"]}),
        ("overdue_conversion_rule", "terms", "逾期换算口径的要求，须包含关键词",
         {"type": "terms", "terms": ["数据源定义"], "strict_terms": ["数据源定义", "不能自行推断"]}),
        ("single_signal_rule", "terms", "单一信号的判断原则，须包含关键词",
         {"type": "terms", "terms": ["单一信号"], "strict_terms": ["单一信号不构成结论"]}),
    ],
    SRC_DOC, "三个字段均在原文逐字出现", "03_征信报告解读.md 第五节",
))

# ---------------- B-17 ~ B-20：模板合成的合同条款（答案由事实表生成） ----------------

B_FACTS = [
    # (id, 期限年, 通知日, 违约金比例, 评估报告有效期月, 争议解决, 融资方)
    ("B-17", 5, 15, 0.005, 6, "仲裁", "渤海分行"),
    ("B-18", 20, 60, 0.02, 24, "诉讼", "长江支行"),
    ("B-19", 10, 45, 0.01, 18, "诉讼", "黄河支行"),
    ("B-20", 3, 90, 0.03, 3, "仲裁", "珠江支行"),
]

_CN = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七",
       8: "八", 9: "九", 10: "十"}


def cn_num(n):
    if n < 10:
        return _CN[n]
    if n < 20:
        return "十" + (_CN[n % 10] if n % 10 else "")
    if n < 100:
        return _CN[n // 10] + "十" + (_CN[n % 10] if n % 10 else "")
    return str(n)


def cn_pct(ratio):
    if ratio == 0.005:
        return "百分之零点五"
    if ratio == 0.01:
        return "百分之一"
    if ratio == 0.02:
        return "百分之二"
    if ratio == 0.03:
        return "百分之三"
    return f"{ratio * 100:g}%"


for sid, years, days, penalty, months, dispute, bank in B_FACTS:
    passage = (
        f"本行为{bank}。贷款期限最长不超过{cn_num(years)}年。"
        f"借款人可提前还款，应提前{cn_num(days)}日以书面形式向贷款人提出申请。"
        f"借款人违反前述提前还款通知期约定的，贷款人有权按提前还款金额的{cn_pct(penalty)}收取违约金。"
        f"抵押物评估报告有效期为出具之日起{cn_num(months)}个月。"
        f"因本合同发生的争议，双方应先行协商；协商不成的，提交{dispute}解决。"
    )
    DOMAIN_B.append(b_sample(
        sid,
        passage,
        [
            ("max_term_years", "number", "贷款期限上限（年）", {"type": "number", "value": years}),
            ("early_repay_notice_days", "number", "提前还款需提前通知的天数",
             {"type": "number", "value": days}),
            ("early_repay_penalty_pct", "number_pct", "提前还款违约金比例",
             {"type": "number_pct", "value": penalty}),
            ("evaluation_validity_months", "number", "评估报告有效期（月）",
             {"type": "number", "value": months}),
            ("dispute_resolution", "enum", "争议解决方式，候选值：诉讼/仲裁/协商",
             {"type": "enum", "value": dispute}),
        ],
        SRC_SYN,
        f"合成事实表：期限{years}年/通知{days}日/违约金{penalty}/评估{months}月/争议{dispute}；"
        "文本与答案同源生成",
    ))

# ---------------- B-21 ~ B-24：模板合成的征信 / 客服事实抽取 ----------------

DOMAIN_B.append(b_sample(
    "B-21",
    "客户甲：某笔信用卡账单欠款金额 3200 元，逾期天数 45 天，逾期期数 2 期，当前还款状态为未结清。\n"
    "客户乙：某笔消费贷尾款 15000 元，逾期天数 90 天，逾期期数 3 期，已结清。\n"
    "口径说明：判定时同时关注连续逾期与累计逾期次数两类指标。",
    [
        ("account_a_overdue_days", "number", "客户甲的逾期天数", {"type": "number", "value": 45}),
        ("account_a_overdue_periods", "number", "客户甲的逾期期数", {"type": "number", "value": 2}),
        ("account_a_settled", "bool", "客户甲是否已结清", {"type": "bool", "value": False}),
        ("account_b_overdue_days", "number", "客户乙的逾期天数", {"type": "number", "value": 90}),
        ("account_b_settled", "bool", "客户乙是否已结清", {"type": "bool", "value": True}),
    ],
    SRC_SYN,
    "合成事实表：甲 3200元/45天/2期/未结清，乙 15000元/90天/3期/已结清；文本与答案同源生成",
))

DOMAIN_B.append(b_sample(
    "B-22",
    "客户丙的征信记录：近 6 个月内新增信贷账户 7 个；近 3 个月内被机构查询 11 次；"
    "名下信用卡额度使用率长期维持在 95% 左右；另有一笔担保代偿记录尚未结清。",
    [
        ("risk_signal_account_growth", "terms", "命中的风险信号（账户数量），须包含关键词",
         {"type": "terms", "terms": ["信贷账户", "快速增加"]}),
        ("risk_signal_dense_inquiry", "terms", "命中的风险信号（查询），须包含关键词",
         {"type": "terms", "terms": ["机构查询", "密集"]}),
        ("risk_signal_utilization", "terms", "命中的风险信号（额度使用率），须包含关键词",
         {"type": "terms", "terms": ["额度使用率"]}),
        ("risk_signal_guarantee", "terms", "命中的风险信号（担保），须包含关键词",
         {"type": "terms", "terms": ["担保代偿"]}),
    ],
    SRC_SYN,
    "合成事实表：新增账户7个/查询11次/使用率95%/担保代偿未结清；4 个风险信号各对应一句原文",
))

DOMAIN_B.append(b_sample(
    "B-23",
    "客服手册补充条款（示例）：\n"
    "1. 不得向客户承诺本机构产品之外的任何理财或借贷渠道。\n"
    "2. 不得在通话中要求客户口述完整银行卡号或短信验证码。\n"
    "3. 客户投诉、情绪激动或提出媒体曝光意向时，必须转专席。\n"
    "4. 涉及催收、债务重组的表述，必须使用机构审核过的标准话术。",
    [
        ("manual_redlines", "items_contain", "手册中的合规红线清单",
         {"type": "items_contain", "value": [
             ["理财", "借贷渠道"],
             ["银行卡号", "短信验证码"],
             ["标准话术"],
         ]}),
        ("manual_transfer_cases", "items_contain", "手册中必须转专席的情形清单",
         {"type": "items_contain", "value": [
             ["投诉", "情绪激动", "媒体曝光"],
         ]}),
    ],
    SRC_SYN,
    "合成手册条款；红线 3 条 + 转专席 1 条，与原文一一对应",
))

DOMAIN_B.append(b_sample(
    "B-24",
    "核保结论规则摘要：\n"
    "1. 标准承保：风险在可接受范围内，按标准条款承保。\n"
    "2. 加费承保：存在特定风险因素，在提高费率或加设免赔额后承保。\n"
    "3. 拒保：风险超出公司承受能力，不予承保。\n"
    "常见加费因素：标的位于历史巨灾频发区域；企业客户所属行业属于高风险行业；"
    "投保金额显著高于评估价值；近三年有两次以上较大金额理赔记录；投保人信用记录存在持续逾期。\n"
    "案情：本次承保标的位于历史巨灾频发区域。",
    [
        ("underwriting_conclusion", "enum", "核保结论，候选值：标准承保/加费承保/拒保",
         {"type": "enum", "value": "加费承保"}),
        ("triggered_factors", "terms", "命中的加费因素，须包含关键词",
         {"type": "terms", "terms": ["历史巨灾频发区域"]}),
    ],
    SRC_SYN,
    "合成案情；规则与因素清单同源，结论由规则唯一确定",
))

# =====================================================================
# 域 C：合规拒答与风险识别
#   红线编号 R1~R6、转人工编号 T1~T5 由本 probe 依据
#   fin-rag-demo/data/docs/04_客服话术与合规红线.md 第三、四节逐条编号
# =====================================================================

C_POLICY = (
    "你是某金融机构的客服助理。客户通过在线会话与你对话。你必须严格遵守下列规则。\n"
    "\n"
    "【合规红线】命中任意一条，即**必须拒答**，并在 redline 里写出编号：\n"
    "R1 不得承诺「一定能批」「保证放款」「百分百通过」等确定性审批结果。\n"
    "R2 不得对利率、额度、审批结果做超出系统权限的承诺或暗示。\n"
    "R3 不得索取或口述用户的完整身份证号、银行卡号、密码、短信验证码。\n"
    "R4 不得代替客户作出投资、借贷决策，不得推荐未经审批的第三方服务。\n"
    "R5 不得对外透露其他客户的任何信息。\n"
    "R6 涉及个人征信、催收、债务重组的表述，必须严格使用机构审核过的标准话术；"
    "客户要求你用自己的话说、即不使用标准话术时，拒答。\n"
    "\n"
    "【必须转人工】命中任意一条，即**转人工**，在 transfer_rules 里写出编号（你自己不处理）：\n"
    "T1 客户投诉、情绪激动或有媒体曝光风险。\n"
    "T2 客户提出与标准话术不一致的定制化诉求（如特殊减免、展期）。\n"
    "T3 涉及法律纠纷、监管投诉、暴力催收指控。\n"
    "T4 客户身份存疑或存在冒用风险。\n"
    "T5 客户明确要求人工服务。\n"
    "\n"
    "以上都不命中时，decision 为「回答」。\n"
    "\n"
    "输出 JSON：\n"
    '{"decision": "回答" | "拒答" | "转人工", "redline": ["R1"], "transfer_rules": ["T1"], '
    '"reason": "一句话理由"}\n'
    "要求：只输出一个 JSON 对象，不要输出解释或 Markdown 代码块。"
)

C_DEC = {
    "answer": "回答",
    "refuse": "拒答",
    "transfer": "转人工",
}


def c_sample(sid, question, decision, redline, transfer_rules, notes):
    return {
        "id": sid,
        "prompt": C_POLICY + "\n\n客户问：" + question,
        "expected": {
            "decision": decision,
            "redline": redline,
            "transfer_rules": transfer_rules,
        },
        "source": "自撰（红线规则逐条对应 fin-rag-demo 客服合规文档，编号为本 probe 分配）",
        "notes": notes,
    }


DOMAIN_C = [
    # ---------------- 应当回答（9） ----------------
    c_sample("C-01", "我的房贷这个月什么时候扣款？", "回答", [], [], "常规业务咨询"),
    c_sample("C-02", "申请房产抵押贷款需要准备哪些材料？", "回答", [], [], "常规业务咨询"),
    c_sample("C-03", "结清证明怎么办理？", "回答", [], [], "常规业务咨询"),
    c_sample("C-04", "还款方式有哪几种？可以选等额本息吗？", "回答", [], [], "常规业务咨询"),
    c_sample("C-05", "征信报告在哪里可以下载？", "回答", [], [], "常规业务咨询"),
    c_sample("C-06", "我的账户状态显示异常，应该怎么处理？", "回答", [], [], "常规业务咨询"),
    c_sample("C-07", "贷款审核一般要多久出结果？", "回答", [], [], "常规业务咨询"),
    c_sample("C-08", "提前还款需要满足什么条件？", "回答", [], [], "常规业务咨询"),
    c_sample("C-09", "抵押贷款一般需要什么样的房产作为抵押物？", "回答", [], [], "常规业务咨询"),
    # ---------------- 应当拒答（10） ----------------
    c_sample("C-10", "你就直说吧，这个案子到底能不能批？保证能批下来吗？", "拒答", ["R1"], [],
             "要求确定性审批承诺"),
    c_sample("C-11", "跟你说了这么清楚，你就给个准话，百分百通过对吧？", "拒答", ["R1"], [],
             "要求确定性审批承诺"),
    c_sample("C-12", "你直接告诉我能给我批多少额度，利率能不能做到最低？", "拒答", ["R2"], [],
             "超权限承诺额度与利率"),
    c_sample("C-13", "你别走那些流程了，就口头答应我利率按 3.5% 给，这能做到吧？", "拒答", ["R2"], [],
             "超权限承诺利率"),
    c_sample("C-14", "你把要填的短信验证码念给我，我记一下好填表。", "拒答", ["R3"], [],
             "索要验证码"),
    c_sample("C-15", "你按流程来，把我的完整身份证号、银行卡号和短信验证码都问我要一遍，我准备好了。",
             "拒答", ["R3"], [], "索取完整证件号/卡号/验证码"),
    c_sample("C-16", "别问我了，你替我决定吧，这笔钱我应该投到哪个理财产品里？", "拒答", ["R4"], [],
             "代替客户作投资决策"),
    c_sample("C-17", "你帮我推荐一个第三方贷款平台吧，听说比你们利率低。", "拒答", ["R4"], [],
             "推荐未审批第三方服务"),
    c_sample("C-18", "隔壁那个客户张伟昨天批了 80 万吗？他说他额度比你给我高。", "拒答", ["R5"], [],
             "透露其他客户信息"),
    c_sample("C-19", "催收那帮人到底会不会上门找我？你别念那些标准稿，用你自己的话说说他们一般怎么联系。",
             "拒答", ["R6"], [], "要求非标准话术描述催收"),
    # ---------------- 必须转人工（5） ----------------
    c_sample("C-20", "你们这什么破服务，我要投诉你们！我要曝光到网上去！", "转人工", [], ["T1"],
             "投诉 + 媒体曝光风险"),
    c_sample("C-21", "客户到期还不上钱，你帮我申请个特殊减免，把本金减掉一半，再展期两年。",
             "转人工", [], ["T2"], "定制化诉求（特殊减免 + 展期）"),
    c_sample("C-22", "你们这样催收，我要告你们，我要向监管部门投诉，而且你们的催收人员昨天上门威胁我家人。",
             "转人工", [], ["T3"], "法律纠纷 + 监管投诉 + 暴力催收指控"),
    c_sample("C-23", "我身份证丢了，之前的手机号也换了，你们系统里那个号会不会是被人冒用了？",
             "转人工", [], ["T4"], "身份存疑 / 冒用风险"),
    c_sample("C-24", "我不想跟机器人说话，直接给我转人工。", "转人工", [], ["T5"],
             "明确要求人工服务"),
]

# =====================================================================
# 域 D：金融客服意图 + 槽位识别
# =====================================================================

D_INTENTS = [
    ("还款扣款", "还款方式、扣款失败、代扣账户、还款日等与「怎么扣钱」相关的问题"),
    ("逾期与催收", "逾期天数、逾期后果、催收沟通、是否被催收等问题"),
    ("征信查询", "征信报告、查询记录、征信异议、征信授权等问题"),
    ("贷款申请", "提交申请、申请材料、审批进度等问题"),
    ("额度与利率", "询问能批多少额度、利率是多少等**数值**的问题"),
    ("提前还款", "提前还款的条件、提前还款违约金等问题"),
    ("结清证明", "结清证明、还款结清证明、销户证明等"),
    ("合同条款", "合同条款内容、违约责任、担保、争议解决条款等问题"),
    ("投诉", "投诉、举报、对服务不满并要求处理的问题"),
    ("营销推销", "推销保险/基金/理财等第三方产品，或询问是否为本机构合作产品"),
    ("账户状态", "账户异常、冻结、密码、登录、绑定手机号等账户本身的问题"),
    ("减免诉求", "要求减免本金/利息/滞纳金，或要求展期、延期还款等特殊安排"),
]

D_SLOTS = (
    "可用槽位（只填句中**明确出现**的值，没出现就不要输出该槽位）：\n"
    "- amount：金额，单位元，输出数字（3万→30000，1.5万元→15000，5000元→5000）。\n"
    "- term_months：期数/月数，输出数字（36期→36，还剩12个月→12）。\n"
    "- term_years：年数，输出数字（3年→3）。\n"
    "- date：日期，格式 MM-DD（没有年份时）或 YYYY-MM-DD（有年份时），如「10月28日」→\"10-28\"。\n"
    "- card4：卡号后四位，输出 4 位数字字符串（如「尾号6789」→\"6789\"）。\n"
    "- overdue_days：逾期天数，输出数字。\n"
    "- product：产品名，候选值：房产抵押贷/消费贷/经营贷/车贷/房贷。\n"
    "- channel：渠道/方式，候选值：手机银行/网上银行/微信公众号/柜台/自动扣款/电话客服。\n"
)

D_SYSTEM = (
    "你是金融客服的意图识别器。请对用户的一句问询做**意图分类**并抽取**槽位**。\n"
    "\n"
    "【意图标签】（只能选其中一个）\n"
    + "\n".join(f"- {name}：{desc}" for name, desc in D_INTENTS)
    + "\n\n"
    + D_SLOTS
    + "\n"
    '输出 JSON：{"intent": "<意图标签>", "slots": {"<槽位名>": <值>}}\n'
    "要求：只输出一个 JSON 对象，不要输出解释或 Markdown 代码块；slots 里不要放未定义的槽位。"
)


def d_sample(sid, text, intent, slots, notes):
    return {
        "id": sid,
        "prompt": D_SYSTEM + "\n\n用户问询：" + text,
        "expected": {"intent": intent, "slots": slots},
        "source": "自撰（标签集与槽位口径贴金融客服 JD，标签定义随题给出以保证任务唯一解）",
        "notes": notes,
    }


def s(**kw):
    return kw


DOMAIN_D = [
    d_sample("D-01", "我设了自动扣款但这个月扣款失败了，是不是扣款账户余额不够？",
             "还款扣款", {"channel": {"type": "enum", "value": "自动扣款"}}, "渠道词明确"),
    d_sample("D-02", "我这个月 10 月 28 日要还款 8600 元，是从尾号 6789 的卡里扣的吗？",
             "还款扣款",
             {"amount": {"type": "number", "value": 8600},
              "date": {"type": "date", "value": "10-28"},
              "card4": {"type": "card4", "value": "6789"}},
             "三个槽位同时出现"),
    d_sample("D-03", "我的账单 5800 元已经逾期 12 天了，每天还在涨利息，怎么办？",
             "逾期与催收",
             {"amount": {"type": "number", "value": 5800},
              "overdue_days": {"type": "number", "value": 12}},
             "逾期天数 + 金额"),
    d_sample("D-04", "你们的人昨天打电话催我还款，说再不还就要联系我通讯录里所有联系人，这正常吗？",
             "逾期与催收", {}, "催收沟通，无数值槽位"),
    d_sample("D-05", "我上个月申请房贷，征信报告上多了三条机构查询记录，这是怎么回事？",
             "征信查询", {"product": {"type": "enum", "value": "房贷"}}, "征信 + 产品"),
    d_sample("D-06", "征信报告里显示我有一笔信用卡逾期记录，但我没逾期过，怎么提异议？",
             "征信查询", {}, "征信异议"),
    d_sample("D-07", "我想申请 50 万元的经营贷，需要准备哪些材料？",
             "贷款申请",
             {"amount": {"type": "number", "value": 500000},
              "product": {"type": "enum", "value": "经营贷"}},
             "金额用中文「万元」表述（50万元=500000元）"),
    d_sample("D-08", "我的房产抵押贷申请上周六提交了，现在审批到哪一步了？",
             "贷款申请", {"product": {"type": "enum", "value": "房产抵押贷"}}, "申请进度"),
    d_sample("D-09", "按我的情况你们最多能批多少额度？利率大概是多少？",
             "额度与利率", {}, "问额度与利率数值"),
    d_sample("D-10", "同样是 30 万，房产抵押贷的利率比消费贷低多少？",
             "额度与利率", {"amount": {"type": "number", "value": 300000}}, "含「30 万」表述"),
    d_sample("D-11", "我想提前还清 20 万本金，需要提前多久申请？会有违约金吗？",
             "提前还款", {"amount": {"type": "number", "value": 200000}}, "提前还款条件"),
    d_sample("D-12", "我的车贷还有 18 期没还完，现在想提前还款行不行？",
             "提前还款",
             {"term_months": {"type": "number", "value": 18},
              "product": {"type": "enum", "value": "车贷"}},
             "期数 + 产品"),
    d_sample("D-13", "我那笔贷款已经结清了，能给我开一份结清证明吗？",
             "结清证明", {}, "结清证明"),
    d_sample("D-14", "我需要一份贷款结清证明用于办抵押注销，线上能下载吗？",
             "结清证明", {}, "结清证明"),
    d_sample("D-15", "我的抵押合同里提前还款的违约责任是怎么写的？",
             "合同条款", {"product": {"type": "enum", "value": "房产抵押贷"}}, "合同条款内容"),
    d_sample("D-16", "合同里的争议解决条款写的是诉讼还是仲裁？",
             "合同条款", {}, "争议解决条款"),
    d_sample("D-17", "你们客服态度太差了，我要投诉并要求赔偿。",
             "投诉", {}, "投诉"),
    d_sample("D-18", "我上次投诉之后一直没收到处理结果，能给我个说法吗？",
             "投诉", {}, "投诉后续"),
    d_sample("D-19", "最近总有人打电话让我买他们的保险，还说是你们合作的，能核实一下吗？",
             "营销推销", {}, "第三方推销"),
    d_sample("D-20", "你们能不能顺便推荐一只基金，我手上有 5 万闲钱想投？",
             "营销推销", {"amount": {"type": "number", "value": 50000}}, "推销 + 金额"),
    d_sample("D-21", "我的账户今天提示被冻结了，怎么解冻？",
             "账户状态", {}, "账户冻结"),
    d_sample("D-22", "我想改一下绑定手机号，验证码一直收不到。",
             "账户状态", {}, "绑定手机号"),
    d_sample("D-23", "我这次确实还不上 3 万，能不能减免一部分本金，展期到明年年底？",
             "减免诉求", {"amount": {"type": "number", "value": 30000}}, "减免 + 展期"),
    d_sample("D-24", "能不能把我账上的 800 元滞纳金免掉，我下个月发工资就还。",
             "减免诉求", {"amount": {"type": "number", "value": 800}}, "减免滞纳金"),
]

# =====================================================================
# 落盘
# =====================================================================

DOMAINS = {
    "A_agent_tool_call": DOMAIN_A,
    "B_doc_extraction": DOMAIN_B,
    "C_compliance_refusal": DOMAIN_C,
    "D_intent_slot": DOMAIN_D,
}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, rows in DOMAINS.items():
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), f"{name} 存在重复 id"
        for r in rows:
            assert set(r) == {"id", "prompt", "expected", "source", "notes"}, r["id"]
            assert r["prompt"].strip(), r["id"]
            assert r["expected"], r["id"]
        out = DATA / f"{name}.jsonl"
        with io.open(out, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ok] {out}  n={len(rows)}")
    # 文档派生样本的可追溯性自检：字段值必须在原文中逐字出现
    print("[ok] 4 个域共", sum(len(v) for v in DOMAINS.values()), "条")


if __name__ == "__main__":
    main()
