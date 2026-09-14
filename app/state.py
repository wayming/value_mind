"""LangGraph 全局状态。

父图 state 只存"结论产物",不存消息:每个 skill 节点是独立的小 agent
(自带 {messages, remaining_steps, structured_response} 小状态),
避免消息串污染与 token 膨胀。

dict 类字段使用 merge_dicts 合并器:并行扇出的多个方法节点各自写入
skills/valuation 的不同子键,合并器保证互不覆盖。
"""

from typing import Annotated, Any, NotRequired, TypedDict
import operator
import re


# valuation 中各方法的规范键名(单一定义源):nodes 写入 / report 展示 /
# synthesize 复核权重都引用这里,避免键名漂移导致复核失效
METHOD_VALUE_KEYS = [
    "ddm_per_share",
    "fcfe_per_share",
    "excess_per_share_perpetuity",
    "excess_per_share_staged",
    "relative_pb_fair_price",
]

# LLM 常把"方法名/节点名"当作权重键(如 reg_capital_fcfe),导致该方法的结果拿不到
# 权重、被静默忽略。这里只收一个方法名唯一对应一个结果键的情形——纯拼写归一,不是判断。
# 裸 "excess_returns" 故意不收:它对应两个结果键(永续上界/均值回归中央值),归哪个是
# 估值判断,必须留给 LLM,归一不了就让它原样落进告警。
METHOD_VALUE_ALIASES = {
    "ddm": "ddm_per_share",
    "ddm_gordon": "ddm_per_share",
    "ddm_multistage": "ddm_per_share",
    "fcfe": "fcfe_per_share",
    "fcfe_valuation": "fcfe_per_share",
    "reg_capital_fcfe": "fcfe_per_share",
    "regulatory_capital_fcfe": "fcfe_per_share",
    "excess_returns_perpetuity": "excess_per_share_perpetuity",
    "excess_perpetuity": "excess_per_share_perpetuity",
    "excess_returns_staged": "excess_per_share_staged",
    "excess_staged": "excess_per_share_staged",
    "relative": "relative_pb_fair_price",
    "relative_valuation": "relative_pb_fair_price",
    "pb_regression": "relative_pb_fair_price",
}


# 构建键名时无处不在的通用词:剥掉它们后剩下的词才真正区分方法
_KEY_STOPWORDS = {"per", "share", "shares", "value"}


def _key_tokens(s: str) -> frozenset[str]:
    return frozenset(
        t for t in re.split(r"[^a-z0-9]+", str(s).strip().lower())
        if t and t not in _KEY_STOPWORDS
    )


def normalize_weight_keys(
    weights: dict[str, float] | None, valid_keys: list[str] | None = None
) -> tuple[dict[str, float], list[str]]:
    """把权重键归一到规范结果键。返回 (归一后的权重, 归一记录说明)。

    三级判断,从严到宽,绝不猜:
      1. 已是规范键 → 原样保留;
      2. 方法名/节点名 → 查 METHOD_VALUE_ALIASES(relative_valuation 这类
         与结果键词面无关的只能查表);
      3. 规范键的拼写变体 → 剥掉 per/share 这类通用词后**剩余词集完全相等**,
         且只能命中一个规范键才归位。
         例:excess_share_perpetuity → {excess, perpetuity} 唯一命中
         excess_per_share_perpetuity。而裸 excess_returns → {excess, returns}
         两个规范键都不等,不归位,原样落进告警——归哪个是估值判断,不是拼写问题。

    归一记录写进报告,便于审计"权重是怎么落到结果上的"。
    """
    valid = set(valid_keys or METHOD_VALUE_KEYS)
    canon_tokens = {k: _key_tokens(k) for k in valid}
    out: dict[str, float] = {}
    renamed: list[str] = []
    for k, v in (weights or {}).items():
        target: str | None = None
        if k in valid:
            target = k
        else:
            target = METHOD_VALUE_ALIASES.get(str(k).strip().lower())
            if target not in valid:
                hits = [c for c, toks in canon_tokens.items() if toks and toks == _key_tokens(k)]
                target = hits[0] if len(hits) == 1 else None
        if target and target in valid:
            if target != k:
                renamed.append(f"{k}→{target}")
            # 两个别名指向同一结果键时权重相加,不静默丢弃后一个
            out[target] = out.get(target, 0.0) + v
        else:
            out[k] = out.get(k, 0.0) + v
    return out, renamed


def merge_dicts(left: dict, right: dict) -> dict:
    """按子键递归合并两个 dict,右值覆盖左值(用于并行节点写入不同子键)。"""
    out = dict(left or {})
    for k, v in (right or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


class ValuationState(TypedDict, total=False):
    # ---- 输入(main.py 注入) ----
    exchange: str                      # 交易所代码,如 SHA / NYSE / ASX
    code: str                          # 股票代码,如 600036 / WFC
    company_name: str                  # 公司名(分类节点可补全)
    overrides: dict[str, Any]          # CLI 覆盖参数,如 {"beta": 0.9, "rf": 0.02}
    period_hint: str                   # 建议 LLM 抓取的数据窗口(非硬编码调用)
    methods_skip: list[str]            # CLI --skip 强制跳过的估值方法
    method: str                        # Send 扇出时携带:当前方法节点名
    # ---- 分类结论(节点 01) ----
    company_type: str                  # bank | insurance | investment_bank | investment_company | diversified
    methods_enabled: list[str]         # ["ddm","reg_capital_fcfe","excess_returns","relative"] 子集
    # ---- MCP 数据缓存(LLM 调过的原始 structuredContent,供 Python 交叉校验/统计) ----
    data_cache: Annotated[dict[str, Any], merge_dicts]
    # ---- 全局口径参数(节点 02/03 产出,方法节点共享) ----
    coefficients: Annotated[dict[str, Any], merge_dicts]
    # ---- 各 skill 产出(键互不冲突) ----
    skills: Annotated[dict[str, dict], merge_dicts]
    # ---- 最终产物 ----
    valuation: Annotated[dict[str, Any], merge_dicts]
    final_report_path: str
    status: str                        # ok | degraded
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
