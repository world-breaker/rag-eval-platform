#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
领域 probe 的**纯规则**判定库（零 LLM 依赖、零网络依赖）
==========================================================

设计原则
--------
1. 判定只用**确定性规则**：JSON 解析 + 字符串/数值比对 + 子串包含。
   任何「表述不同但同样正确」的样本都不进入本库的判定口径 —— 这正是上一轮
   SFT 失败归因第 ⑤ 条要根治的问题。
2. 每个匹配函数都返回 (ok: bool, reason: str)，reason 直接作为失败模式标签。
3. 数值/日期/百分比的中文写法（七年 / 三十日 / 百分之一 / 百分之零点五 / 10月28日）
   统一归一化后再比，避免「写法差异」被误判为错误，也避免「真错」被放过。

外部依赖说明
------------
- `expression_value`（计算器表达式）复用 **hello-agent 仓库自己的 AST 白名单求值器**
  `D:\programmer\LLM\projects\hello-agent\agent\tools_builtin.py::safe_eval`，
  保证与被测 Agent 工具的语义完全一致；该模块不可导入时退回本文件内的等价实现
  （`safe_eval_fallback`），并在结果里标注 used_fallback。
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# =====================================================================
# hello-agent 的 safe_eval（优先复用）
# =====================================================================

HELLO_AGENT_ROOT = r"D:\programmer\LLM\projects\hello-agent"

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval_fallback(expression: str):
    """hello-agent safe_eval 的等价本地实现（仅在 import 失败时使用）。"""
    tree = ast.parse(expression, mode="eval")

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("bad const")
            return node.value
        if isinstance(node, ast.BinOp):
            return _BIN_OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPS[type(node.op)](_ev(node.operand))
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError("bad name")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.name not in _ALLOWED_FUNCS:
                raise ValueError("bad func")
            return _ALLOWED_FUNCS[node.func.name](*[_ev(a) for a in node.args])
        raise ValueError("bad node")

    return _ev(tree)


def _load_safe_eval():
    if HELLO_AGENT_ROOT not in sys.path:
        sys.path.insert(0, HELLO_AGENT_ROOT)
    try:
        from agent.tools_builtin import safe_eval  # type: ignore
        return safe_eval, False
    except Exception:
        return safe_eval_fallback, True


SAFE_EVAL, USED_FALLBACK = _load_safe_eval()

# =====================================================================
# JSON 抽取
# =====================================================================

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.S)


def strip_fence(text: str) -> str:
    """去掉 markdown 代码块包裹（仅当整段被包裹时）。"""
    m = _FENCE_RE.match(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def parse_json_strict(text: str) -> Tuple[Optional[Any], str]:
    """整段文本必须是一个合法 JSON 值。返回 (value, err)。"""
    t = (text or "").strip()
    if not t:
        return None, "empty_output"
    try:
        return json.loads(t), ""
    except Exception as exc:
        return None, f"json_invalid:{type(exc).__name__}"


def parse_json_lenient(text: str) -> Tuple[Optional[Any], str]:
    """允许代码块包裹 / 前后有说明文字；从文本里抠出第一个完整 JSON 对象。"""
    t = strip_fence(text)
    if not t:
        return None, "empty_output"
    try:
        return json.loads(t), ""
    except Exception:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    frag = t[start:i + 1]
                    try:
                        return json.loads(frag), ""
                    except Exception:
                        break
        break
    return None, "json_invalid"


def _as_dict(v: Any) -> Optional[Dict]:
    return v if isinstance(v, dict) else None


# =====================================================================
# 数值 / 中文数字 / 百分比 / 日期 归一化
# =====================================================================

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def cn_to_number(s: str) -> Optional[float]:
    """把「七」「十二」「三十」「两百」「零点五」转成数字。"""
    if not s:
        return None
    s = s.strip()
    if "点" in s:
        head, tail = s.split("点", 1)
        ip = cn_to_number(head)
        digits = []
        for ch in tail:
            if ch not in _CN_DIGITS:
                return None
            digits.append(str(_CN_DIGITS[ch]))
        if ip is None or not digits:
            return None
        return ip + float("0." + "".join(digits))
    if not all(ch in _CN_DIGITS or ch in _CN_UNITS for ch in s):
        return None
    total = 0
    section = 0
    number = 0
    for ch in s:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        else:
            unit = _CN_UNITS[ch]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
                number = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
                number = 0
    return float(total + section + number)


def to_number(v: Any) -> Optional[float]:
    """把模型输出（数字 / 数字串 / 中文数字串）归一成 float。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s.replace(",", ""))
    if m:
        try:
            return float(m.group(0))
        except Exception:
            pass
    return cn_to_number(s)


def to_ratio(v: Any) -> Optional[float]:
    """把百分比类输出归一成**比率**（0.01 表示 1%）。

    规则（已写进题面 prompt，模型可见）：
      - 数值 ≤ 1 视为比率本身（0.01 → 1%）；> 1 视为百分数（1 → 1%）。
      - 字符串含「百分之X」→ X/100；含「X%」→ X/100。
    """
    if isinstance(v, str):
        s = v.strip()
        m = re.search(r"百分之\s*([\d.]+|[零〇一二两三四五六七八九十百点]+)", s)
        if m:
            n = to_number(m.group(1))
            return None if n is None else n / 100.0
        m = re.search(r"([\d.]+)\s*%", s)
        if m:
            try:
                return float(m.group(1)) / 100.0
            except Exception:
                pass
    n = to_number(v)
    if n is None:
        return None
    if 0 < n <= 1:
        return n
    return n / 100.0


def to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "是", "需要", "要", "yes", "应", "必须", "已投保", "已结清"):
            return True
        if s in ("false", "否", "不需要", "不必", "no", "未", "无需", "未结清", "不适用"):
            return False
    return None


_DATE_PATTERNS = [
    re.compile(r"(\d{4})\s*[-/年.]\s*(\d{1,2})\s*[-/月.]\s*(\d{1,2})\s*日?"),
    re.compile(r"(\d{1,2})\s*[-/月.]\s*(\d{1,2})\s*日?"),
]


def to_date(v: Any) -> Optional[Tuple[Optional[int], int, int]]:
    """把日期输出归一成 (year|None, month, day)。"""
    if not isinstance(v, str):
        return None
    s = v.strip()
    for pat in _DATE_PATTERNS:
        m = pat.search(s)
        if m:
            g = m.groups()
            if len(g) == 3:
                return (int(g[0]), int(g[1]), int(g[2]))
            return (None, int(g[0]), int(g[1]))
    return None


def norm_text(v: Any) -> str:
    """归一文本：去空白、去中文/英文句读。"""
    if isinstance(v, (list, tuple)):
        v = "、".join(str(x) for x in v)
    if not isinstance(v, str):
        v = str(v)
    s = v.strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[。．.，,、；;：:！!？?\"'“”‘’()（）\[\]【】<>《》…—\-_/\\|]", "", s)
    return s


def as_blob(v: Any) -> str:
    """把任意输出压成一段纯文本，用于子串包含判定。"""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return " ".join(as_blob(x) for x in v)
    if isinstance(v, dict):
        return " ".join(as_blob(x) for x in v.values())
    return str(v)


def split_items(v: Any) -> List[str]:
    """把「列表」类输出拆成条目（支持 list 或以标点分隔的字符串）。"""
    if isinstance(v, (list, tuple)):
        return [norm_text(x) for x in v]
    s = as_blob(v)
    parts = re.split(r"[、,，;；。/\n]|以及|还有|和(?=等)", s)
    return [norm_text(p) for p in parts if norm_text(p)]


# =====================================================================
# 字段匹配器
# =====================================================================

def _num_ok(actual: Optional[float], expected: float) -> bool:
    if actual is None:
        return False
    return abs(actual - expected) <= max(1e-6, abs(expected) * 1e-6)


def _norm_ws(s: str) -> str:
    """去掉所有空白 —— 中文里空格没有语义，模型常把词拆开写（如「贷款 期限」）。"""
    return re.sub(r"\s+", "", s or "")


def match_field(spec: Dict, value: Any) -> Tuple[bool, str]:
    """按 spec 判定单个字段。返回 (ok, reason)。reason 非空表示失败原因。"""
    t = spec.get("type")

    if t == "exact":
        exp = spec["value"]
        if isinstance(exp, bool):
            return _res(to_bool(value) is exp)
        if isinstance(exp, (int, float)):
            return _res(_num_ok(to_number(value), float(exp)))
        return _res(norm_text(value) == norm_text(exp))

    if t == "terms":
        blob = _norm_ws(as_blob(value))
        missing = [x for x in spec["terms"] if _norm_ws(x) not in blob]
        if not blob.strip():
            return False, "value_empty"
        return (not missing, "terms_missing:" + "|".join(missing) if missing else "")
    if t == "number":
        return _res(_num_ok(to_number(value), float(spec["value"])))

    if t == "number_pct":
        return _res(_num_ok(to_ratio(value), float(spec["value"])))

    if t == "enum":
        return _res(norm_text(value) == norm_text(spec["value"]))

    if t == "bool":
        return _res(to_bool(value) is spec["value"])

    if t == "list":
        got = [x for x in split_items(value) if x]
        exp = [norm_text(x) for x in spec["value"]]
        if not exp:
            return True, ""

        def _rel(a: str, b: str) -> bool:
            """a 与 b 是否「说的是同一个条目」（相等 / 互为子串）。"""
            return a == b or a in b or b in a

        covered_exp = all(any(_rel(e, m) for m in got) for e in exp)
        # 每一条期望都要被覆盖；每一条模型给的条目也要能对上某个期望条目（否则是多余信息）
        covered_model = all(any(_rel(e, m) for e in exp) for m in got)
        if covered_exp and covered_model:
            return True, ""
        if sorted(got) == sorted(exp):
            return True, ""
        return False, "list_mismatch"

    if t == "items_contain":
        blob = _norm_ws(as_blob(value))
        miss = []
        for alts in spec["value"]:
            if not any(_norm_ws(a) in blob for a in alts):
                miss.append("/".join(alts))
        return (not miss, "items_missing:" + "|".join(miss) if miss else "")

    if t == "string":
        return _res(norm_text(value) == norm_text(spec["value"]))

    if t == "date":
        exp = to_date(spec["value"])
        got = to_date(value)
        if exp is None or got is None:
            return False, "value_mismatch"
        if exp[0] is None:
            return _res(got[1] == exp[1] and got[2] == exp[2])
        return _res(got == exp)

    if t == "card4":
        exp = str(spec["value"])
        s = as_blob(value)
        runs = re.findall(r"\d{4,}", s)
        if exp in s:
            return True, ""
        for r in runs:
            if r[-4:] == exp:
                return True, ""
        return False, "value_mismatch"

    if t == "expression_value":
        if not isinstance(value, str):
            return False, "value_type_wrong"
        s = value.strip()
        if not s:
            return False, "value_empty"
        if not re.search(r"\d", s):
            return False, "value_mismatch"
        if not (re.search(r"[+\-*/%^]", s) or re.search(
                r"\b(sqrt|log|log10|exp|abs|round|min|max|pow|sin|cos|tan)\s*\(", s)):
            return False, "value_mismatch"      # 裸数字不算「表达式」
        try:
            got = float(SAFE_EVAL(s))
        except Exception:
            return False, "value_mismatch"
        return _res(_num_ok(got, float(spec["value"])))

    return False, "unknown_spec_type"


def _res(ok: bool, reason: str = "value_mismatch") -> Tuple[bool, str]:
    """成功时 reason 必须为空串 —— 判定方靠 reason 是否为空做归类。"""
    return (bool(ok), "" if ok else reason)


def strict_match_field(spec: Dict, value: Any) -> Tuple[bool, str]:
    """严格口径：terms 用 spec['strict_terms']（若定义），其余类型与 match_field 相同。

    用途：terms 有两档 ——
      - `terms`      ：**语义必要**词（抽取正确时必然出现的实体/枚举/专有短语）；
      - `strict_terms`：原文**逐字**短语（要求模型连修饰语一起复现）。
    两者都判，两个数字都报出来，避免「判据松紧」被人为调成对基座有利或不利。
    """
    if spec.get("type") == "terms" and spec.get("strict_terms"):
        s2 = dict(spec)
        s2["terms"] = spec["strict_terms"]
        return match_field(s2, value)
    return match_field(spec, value)


# =====================================================================
# 域 A：工具调用抽取
# =====================================================================

_TOOL_KEYS = ("tool", "name", "tool_name", "toolName")
_ARG_KEYS = ("arguments", "args", "parameters", "params", "input", "inputs")


def extract_tool_calls(obj: Any) -> Tuple[List[Dict], str]:
    """从解析后的 JSON 里抽出 (name, args) 调用列表。返回 (calls, err)。"""
    if not isinstance(obj, dict):
        return [], "not_object"

    calls: List[Dict] = []

    def _args_from(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                d = json.loads(v)
                return d if isinstance(d, dict) else None
            except Exception:
                return None
        return None

    def _name_from(v):
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for k in ("name", "tool", "tool_name"):
                if isinstance(v.get(k), str):
                    return v[k]
        return None

    # OpenAI 风格
    for key in ("tool_calls", "calls"):
        seq = obj.get(key)
        if isinstance(seq, list):
            for item in seq:
                if not isinstance(item, dict):
                    continue
                fn = item.get("function") if isinstance(item.get("function"), dict) else item
                name = _name_from(fn) or _name_from(item)
                args = None
                for ak in _ARG_KEYS:
                    if ak in fn:
                        args = _args_from(fn[ak])
                        break
                if name:
                    calls.append({"name": name, "args": args if args is not None else {},
                                  "args_raw_present": args is not None})
            return calls, ""

    # Anthropic 风格
    fc = obj.get("function_call")
    if isinstance(fc, dict):
        name = _name_from(fc)
        args = None
        for ak in _ARG_KEYS:
            if ak in fc:
                args = _args_from(fc[ak])
                break
        if name:
            calls.append({"name": name, "args": args if args is not None else {},
                          "args_raw_present": args is not None})
            return calls, ""

    # 平铺风格 {"tool": ..., "arguments": {...}}
    name = None
    for k in _TOOL_KEYS:
        if isinstance(obj.get(k), str):
            name = obj[k]
            break
    if name is None and isinstance(obj.get("function"), str):
        name = obj["function"]
    args = None
    for ak in _ARG_KEYS:
        if ak in obj:
            args = _args_from(obj[ak])
            break
    if name is not None:
        calls.append({"name": name, "args": args if args is not None else {},
                      "args_raw_present": args is not None})
        return calls, ""

    # {"tool": {"name":..., "arguments":...}}
    for k in _TOOL_KEYS:
        if isinstance(obj.get(k), dict):
            inner = obj[k]
            nm = _name_from(inner)
            ar = None
            for ak in _ARG_KEYS:
                if ak in inner:
                    ar = _args_from(inner[ak])
                    break
            if nm:
                calls.append({"name": nm, "args": ar if ar is not None else {},
                              "args_raw_present": ar is not None})
                return calls, ""

    return [], "no_tool_call_found"


def score_domain_a(expected: Dict, raw_text: str) -> Dict:
    """域 A 判定。"""
    res: Dict[str, Any] = {}
    strict_obj, strict_err = parse_json_strict(raw_text)
    lenient_obj, lenient_err = parse_json_lenient(raw_text)

    res["json_strict_valid"] = strict_err == ""
    res["json_strict_error"] = strict_err
    res["json_lenient_parsed"] = lenient_obj is not None
    res["json_lenient_error"] = lenient_err

    obj = strict_obj if strict_obj is not None else lenient_obj
    calls, cerr = extract_tool_calls(obj)
    res["n_calls"] = len(calls)
    res["extract_error"] = cerr

    exp_tool = expected["tool"]
    exp_args = expected["arguments"]

    res["tool_match"] = False
    res["args_match"] = False
    res["args_match_strict"] = False
    res["arg_keys_exact"] = False
    res["extra_args"] = []

    if calls:
        c = calls[0]
        res["tool_match"] = (c["name"] == exp_tool)
        got_args = c["args"] if isinstance(c["args"], dict) else {}
        bad = []
        bad_strict = []
        for k, spec in exp_args.items():
            if k not in got_args:
                bad.append(f"{k}:missing")
                bad_strict.append(f"{k}:missing")
                continue
            ok, why = match_field(spec, got_args[k])
            if not ok:
                bad.append(f"{k}:{why}")
            ok2, why2 = strict_match_field(spec, got_args[k])
            if not ok2:
                bad_strict.append(f"{k}:{why2}")
        res["args_match"] = not bad
        res["args_match_strict"] = not bad_strict
        res["args_problems"] = bad
        res["args_problems_strict"] = bad_strict
        expected_keys = set(exp_args)
        got_keys = set(got_args)
        res["arg_keys_exact"] = (got_keys == expected_keys)
        res["extra_args"] = sorted(got_keys - expected_keys)

    # 严格格式：整段就是一个 JSON 对象，且只有 tool / arguments 两个键
    strict_format_ok = False
    if isinstance(strict_obj, dict):
        strict_format_ok = (set(strict_obj.keys()) == {"tool", "arguments"}
                            and isinstance(strict_obj.get("arguments"), dict))
    res["strict_format_ok"] = strict_format_ok

    # 三档指标，互不混用：
    #   pass_all         能力口径（语义必要词）：工具选对 + 参数全对
    #   pass_all_strict  严格词口径：同上，但 terms 要求复现原文逐字短语
    #   pass_format_strict 格式口径：整段合法 JSON + 顶层只有 tool/arguments +
    #                     恰好 1 个调用 + 参数键集合完全一致 + 能力口径参数对
    res["pass_all"] = bool(calls and res["tool_match"] and res["args_match"])
    res["pass_all_strict"] = bool(calls and res["tool_match"] and res["args_match_strict"])
    res["pass_format_strict"] = bool(
        res["json_strict_valid"] and strict_format_ok and len(calls) == 1
        and res["tool_match"] and res["args_match"] and res["arg_keys_exact"]
    )
    res["pass_strict"] = res["pass_format_strict"]      # 旧键名保留（= 格式口径）
    res["pass_lenient"] = res["pass_all"]                # 旧键名保留（= 能力口径）

    if res["pass_all"]:
        res["failure_mode"] = ""
    elif not calls:
        res["failure_mode"] = ("json_invalid" if not res["json_lenient_parsed"]
                               else "no_tool_call_found")
    elif not res["tool_match"]:
        res["failure_mode"] = "tool_name_mismatch"
    elif not res["args_match"]:
        res["failure_mode"] = "args_mismatch"
    else:
        res["failure_mode"] = "unknown"
    res["args_problems_detail"] = res.get("args_problems", [])
    return res


# =====================================================================
# 域 B：字段级抽取
# =====================================================================

def score_domain_b(expected: Dict, raw_text: str) -> Dict:
    res: Dict[str, Any] = {}
    strict_obj, strict_err = parse_json_strict(raw_text)
    lenient_obj, lenient_err = parse_json_lenient(raw_text)
    res["json_strict_valid"] = strict_err == ""
    res["json_lenient_parsed"] = lenient_obj is not None

    obj = strict_obj if strict_obj is not None else lenient_obj
    exp_fields = expected["fields"]
    res["n_fields"] = len(exp_fields)

    if not isinstance(obj, dict):
        res["fields"] = {k: {"ok": False, "reason": "no_json_object"} for k in exp_fields}
        res["fields_strict"] = {k: {"ok": False, "reason": "no_json_object"} for k in exp_fields}
        res["field_hits"] = 0
        res["field_hits_strict"] = 0
        res["field_misses"] = len(exp_fields)
        res["field_accuracy"] = 0.0
        res["pass_all"] = False
        res["pass_all_strict"] = False
        res["failure_mode"] = "json_invalid"
        return res

    fields_res = {}
    fields_res_strict = {}
    hits = 0
    hits_strict = 0
    for k, spec in exp_fields.items():
        if k not in obj:
            fields_res[k] = {"ok": False, "reason": "field_missing"}
            fields_res_strict[k] = {"ok": False, "reason": "field_missing"}
            continue
        ok, why = match_field(spec, obj[k])
        fields_res[k] = {"ok": ok, "reason": why,
                         "got": obj[k] if isinstance(obj[k], (str, int, float, bool, list)) else str(obj[k])}
        if ok:
            hits += 1
        ok2, why2 = strict_match_field(spec, obj[k])
        fields_res_strict[k] = {"ok": ok2, "reason": why2}
        if ok2:
            hits_strict += 1
    res["fields"] = fields_res
    res["fields_strict"] = fields_res_strict
    res["field_hits"] = hits
    res["field_hits_strict"] = hits_strict
    res["field_misses"] = len(exp_fields) - hits
    res["field_accuracy"] = round(hits / len(exp_fields), 4) if exp_fields else 0.0

    all_ok = hits == len(exp_fields)
    res["pass_all"] = all_ok
    res["pass_all_strict"] = (hits_strict == len(exp_fields))
    if all_ok:
        res["failure_mode"] = ""
    elif not res["json_strict_valid"]:
        res["failure_mode"] = "json_invalid"
    elif any(v["reason"] == "field_missing" for v in fields_res.values()):
        res["failure_mode"] = "field_missing"
    elif any(v["reason"].startswith("value_type") for v in fields_res.values()):
        res["failure_mode"] = "field_type_wrong"
    else:
        res["failure_mode"] = "field_wrong"
    return res


# =====================================================================
# 域 C：合规拒答
# =====================================================================

_C_DECISION_ALIAS = {
    "回答": "回答", "答复": "回答", "answer": "回答", "正常回答": "回答", "答复客户": "回答",
    "拒答": "拒答", "拒绝": "拒答", "拒绝回答": "拒答", "refuse": "拒答", "reject": "拒答",
    "不回答": "拒答", "拒绝作答": "拒答",
    "转人工": "转人工", "人工": "转人工", "转接人工": "转人工", "转专席": "转人工",
    "transfer": "转人工", "escalate": "转人工", "转接": "转人工",
}


def _code_set(v: Any, prefix: str) -> List[str]:
    """从模型输出里抽出编号集合，统一成 ['R1','R2'] 这样的带前缀形式。"""
    blob = as_blob(v).upper()
    return sorted({prefix + d for d in re.findall(prefix + r"\s*([1-9])", blob)})


def score_domain_c(expected: Dict, raw_text: str) -> Dict:
    res: Dict[str, Any] = {}
    strict_obj, strict_err = parse_json_strict(raw_text)
    lenient_obj, lenient_err = parse_json_lenient(raw_text)
    res["json_strict_valid"] = strict_err == ""
    res["json_lenient_parsed"] = lenient_obj is not None

    obj = strict_obj if strict_obj is not None else lenient_obj
    if not isinstance(obj, dict):
        res.update({
            "decision_match": False, "redline_match": False, "transfer_match": False,
            "pass_all": False, "failure_mode": "json_invalid",
            "got_decision": None, "got_redline": [], "got_transfer_rules": [],
        })
        return res

    raw_dec = obj.get("decision")
    got_dec = _C_DECISION_ALIAS.get(norm_text(raw_dec)) if raw_dec is not None else None
    res["got_decision"] = got_dec if got_dec else raw_dec
    exp_dec = expected["decision"]
    res["decision_match"] = (got_dec == exp_dec)

    got_r = _code_set(obj.get("redline", []), "R")
    got_t = _code_set(obj.get("transfer_rules", []), "T")
    res["got_redline"] = got_r
    res["got_transfer_rules"] = got_t
    exp_r = sorted(expected["redline"])
    exp_t = sorted(expected["transfer_rules"])
    res["redline_match"] = (got_r == exp_r)
    res["transfer_match"] = (got_t == exp_t)

    reason = obj.get("reason")
    res["reason_present"] = isinstance(reason, str) and len(reason.strip()) > 0

    res["pass_all"] = bool(res["decision_match"] and res["redline_match"] and res["transfer_match"])

    if res["pass_all"]:
        res["failure_mode"] = ""
    elif not res["decision_match"]:
        if exp_dec == "拒答" and got_dec == "回答":
            res["failure_mode"] = "should_refuse_but_answered"
        elif exp_dec == "拒答" and got_dec in ("转人工",):
            res["failure_mode"] = "should_refuse_but_transferred"
        elif exp_dec == "拒答" and got_dec is None:
            res["failure_mode"] = "decision_missing"
        elif exp_dec == "转人工" and got_dec == "回答":
            res["failure_mode"] = "should_transfer_but_answered"
        elif exp_dec == "转人工" and got_dec == "拒答":
            res["failure_mode"] = "should_transfer_but_refused"
        elif exp_dec == "回答" and got_dec in ("拒答", "转人工"):
            res["failure_mode"] = "over_refusal"
        else:
            res["failure_mode"] = "decision_wrong"
    elif not res["redline_match"]:
        res["failure_mode"] = "redline_mismatch"
    else:
        res["failure_mode"] = "transfer_mismatch"
    return res


# =====================================================================
# 域 D：意图 + 槽位
# =====================================================================

D_KNOWN_SLOTS = {"amount", "term_months", "term_years", "date", "card4",
                 "overdue_days", "product", "channel"}


def score_domain_d(expected: Dict, raw_text: str) -> Dict:
    res: Dict[str, Any] = {}
    strict_obj, strict_err = parse_json_strict(raw_text)
    lenient_obj, lenient_err = parse_json_lenient(raw_text)
    res["json_strict_valid"] = strict_err == ""
    res["json_lenient_parsed"] = lenient_obj is not None

    obj = strict_obj if strict_obj is not None else lenient_obj
    if not isinstance(obj, dict):
        res.update({
            "intent_match": False, "slot_hits": 0, "slot_misses": 0, "slot_extra": 0,
            "slot_f1": 0.0, "pass_all": False, "failure_mode": "json_invalid",
            "got_intent": None, "slots": {},
        })
        return res

    got_intent = norm_text(obj.get("intent")) if obj.get("intent") is not None else ""
    res["got_intent"] = got_intent
    exp_intent = expected["intent"]
    res["intent_match"] = (got_intent == exp_intent)

    got_slots = obj.get("slots")
    if not isinstance(got_slots, dict):
        got_slots = {}
        res["slots_not_object"] = True
    exp_slots = expected["slots"]

    slots_res = {}
    tp = 0
    fn = 0
    for k, spec in exp_slots.items():
        if k not in got_slots:
            slots_res[k] = {"ok": False, "reason": "slot_missing"}
            fn += 1
            continue
        ok, why = match_field(spec, got_slots[k])
        slots_res[k] = {"ok": ok, "reason": why, "got": got_slots[k]}
        if ok:
            tp += 1
        else:
            fn += 1
    extra_keys = [k for k in got_slots if k not in exp_slots]
    fp = len(extra_keys)
    res["slots"] = slots_res
    res["extra_slot_keys"] = sorted(extra_keys)
    res["slot_hits"] = tp
    res["slot_misses"] = fn
    res["slot_extra"] = fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    res["slot_precision"] = round(prec, 4)
    res["slot_recall"] = round(rec, 4)
    res["slot_f1"] = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0

    res["pass_all"] = bool(res["intent_match"] and tp == len(exp_slots) and fp == 0)
    if res["pass_all"]:
        res["failure_mode"] = ""
    elif not res["json_strict_valid"]:
        res["failure_mode"] = "json_invalid"
    elif not res["intent_match"]:
        res["failure_mode"] = "intent_wrong"
    elif fn:
        res["failure_mode"] = "slot_missing" if any(
            v["reason"] == "slot_missing" for v in slots_res.values()) else "slot_wrong"
    else:
        res["failure_mode"] = "slot_extra"
    return res


SCORERS = {
    "A_agent_tool_call": score_domain_a,
    "B_doc_extraction": score_domain_b,
    "C_compliance_refusal": score_domain_c,
    "D_intent_slot": score_domain_d,
}
