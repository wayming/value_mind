"""权重键归一的单测。

背景:LLM 反复把方法名或拼写变体当权重键写进 method_weights,导致该方法的估值结果
静默拿不到权重(实测 WFC 上 FCFE 84.11、超额回报 64.45/55.78 都没进加权均值)。
归一必须"绝不猜":能唯一确定才归位,归不了就留给告警。
"""

from app.state import METHOD_VALUE_KEYS, normalize_weight_keys


def test_canonical_keys_pass_through_unchanged():
    w = {k: 0.2 for k in METHOD_VALUE_KEYS}
    out, renamed = normalize_weight_keys(w)
    assert out == w
    assert renamed == []


def test_node_and_method_names_are_mapped():
    out, renamed = normalize_weight_keys(
        {"ddm": 0.3, "reg_capital_fcfe": 0.3, "relative_valuation": 0.4})
    assert out == {"ddm_per_share": 0.3, "fcfe_per_share": 0.3,
                   "relative_pb_fair_price": 0.4}
    assert len(renamed) == 3


def test_spelling_variants_of_canonical_keys_are_mapped():
    """实测变体:少个 s、漏了 per。剥掉通用词后唯一命中,可以归。"""
    out, renamed = normalize_weight_keys(
        {"excess_share_perpetuity": 0.6, "excess_share_staged": 0.4})
    assert out == {"excess_per_share_perpetuity": 0.6, "excess_per_share_staged": 0.4}
    assert len(renamed) == 2


def test_ambiguous_bare_method_name_is_not_guessed():
    """裸 excess_returns 对应两个结果键,归哪个是估值判断 —— 不猜,留给告警。"""
    out, _ = normalize_weight_keys({"excess_returns": 0.5, "ddm": 0.5})
    assert "excess_returns" in out and "excess_returns" not in METHOD_VALUE_KEYS
    assert out["ddm_per_share"] == 0.5


def test_unmappable_keys_survive_for_warning():
    out, renamed = normalize_weight_keys({"foo": 0.5, "bar": 0.5})
    assert out == {"foo": 0.5, "bar": 0.5}
    assert renamed == []


def test_two_aliases_to_same_key_are_summed_not_dropped():
    """ddm 与 ddm_gordon 若同时出现,权重相加而不是后者覆盖前者。"""
    out, _ = normalize_weight_keys({"ddm": 0.25, "ddm_gordon": 0.25})
    assert out == {"ddm_per_share": 0.5}


def test_alias_to_key_without_result_is_left_alone():
    """别名目标不在本次有效结果里(该方法失败)时不该归位,否则会假装有权重。"""
    out, _ = normalize_weight_keys({"reg_capital_fcfe": 0.5, "ddm": 0.5},
                                   valid_keys=["ddm_per_share"])
    assert out == {"ddm_per_share": 0.5, "reg_capital_fcfe": 0.5}


def test_none_weights():
    assert normalize_weight_keys(None) == ({}, [])
