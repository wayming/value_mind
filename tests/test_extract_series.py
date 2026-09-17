"""extract_series 的离线回归:序列必须属于一家公司、且来自单次抓取。

起因是一次真实运行:ASX NAB 的报告里 `roe_series_raw_latest` = 0.0949,
而 NAB 自己的值是 0.09908 —— 0.0949 是西太平洋银行(WBC)的。skill 07 会抓同业做
对比,两家澳洲银行的报告日完全相同(3/31、9/30),旧实现只按日期合并,
于是同业的值逐日覆盖进目标公司序列,σ 成了两家公司的混合体,直接喂进 PB 回归,
报告里看不出任何异常。这两个测试锁住修复。
"""

import json

from app.mcp_tools import annual_block, cached_metrics, extract_series

NAB_EX, NAB_CODE = "ASX", "NAB"
WBC_EX, WBC_CODE = "ASX", "WBC"

DATES = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]


def _raw(values: dict[str, float | None]) -> dict:
    return {"data": {d: {"ratios": {"roe": v}} for d, v in values.items()}}


def _cache(*calls) -> dict:
    out = {}
    for exchange, code, period, values in calls:
        args = {"exchange": exchange, "code": code, "metrics": ["roe"], "period": period}
        key = f"get_financials|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        out[key] = _raw(values)
    return out


NAB_VALUES = dict(zip(DATES, [0.112, 0.11212, 0.10849, 0.11222]))
WBC_VALUES = dict(zip(DATES, [0.09438, 0.09002, 0.10504, 0.09394]))


def test_cross_company_values_do_not_merge():
    """同一天的报告日 + 同业抓取,不能把同业的值并进目标公司。"""
    cache = _cache((NAB_EX, NAB_CODE, "5y", NAB_VALUES),
                   (WBC_EX, WBC_CODE, "5y", WBC_VALUES))
    series, meta = extract_series(cache, "roe", NAB_EX, NAB_CODE, period="5y")

    assert series == NAB_VALUES                       # 逐点等于 NAB,没有 WBC 的值
    assert meta["n"] == 4
    assert series["2025-12-31"] != WBC_VALUES["2025-12-31"]
    # 反向也成立:查 WBC 拿到的必须是 WBC 的
    wbc, _ = extract_series(cache, "roe", WBC_EX, WBC_CODE, period="5y")
    assert wbc == WBC_VALUES


def test_case_insensitive_company_match():
    series, _ = extract_series(_cache((NAB_EX, NAB_CODE, "5y", NAB_VALUES)),
                               "roe", "asx", "nab")
    assert series == NAB_VALUES


def test_prefers_declared_window():
    """声明的窗口有对应抓取就必须用它 —— 否则样本数随 LLM 调过哪些 period 而变。"""
    long_values = {f"20{y:02d}-12-31": 0.10 for y in range(10, 26)}   # 16 个点,信息量更大
    cache = _cache((NAB_EX, NAB_CODE, "5y", NAB_VALUES),
                   (NAB_EX, NAB_CODE, "all", long_values))
    series, meta = extract_series(cache, "roe", NAB_EX, NAB_CODE, period="5y")
    assert meta["period"] == "5y"
    assert series == NAB_VALUES
    assert meta["n"] == 4                             # 没有被 all 那 16 个点稀释

    all_series, all_meta = extract_series(cache, "roe", NAB_EX, NAB_CODE, period="all")
    assert all_meta["period"] == "all"
    assert all_series == long_values


def test_falls_back_to_most_observations_and_reports_period():
    """没有声明窗口的抓取时取信息量最大的一份,并把实际用的 period 回传供核对。"""
    cache = _cache((NAB_EX, NAB_CODE, "1y", dict(list(NAB_VALUES.items())[:2])),
                   (NAB_EX, NAB_CODE, "all", {**NAB_VALUES, "2024-12-31": 0.114}))
    series, meta = extract_series(cache, "roe", NAB_EX, NAB_CODE, period="10y")
    assert meta["period"] == "all"                    # 实际用的是这个,不是声明的 10y
    assert meta["n"] == 5
    assert "period=all" in meta["source"]


def test_all_null_fetch_does_not_win_on_point_count():
    """全空值的抓取不算有效样本 —— 否则空序列会靠点数多胜出。"""
    padded = {f"20{y:02d}-12-31": None for y in range(10, 26)}
    cache = _cache((NAB_EX, NAB_CODE, "all", padded),
                   (NAB_EX, NAB_CODE, "5y", NAB_VALUES))
    series, meta = extract_series(cache, "roe", NAB_EX, NAB_CODE)
    assert meta["period"] == "5y" and series == NAB_VALUES


def test_no_matching_company_returns_empty():
    cache = _cache((WBC_EX, WBC_CODE, "5y", WBC_VALUES))
    series, meta = extract_series(cache, "roe", NAB_EX, NAB_CODE)
    assert series == {}
    assert meta["n"] == 0 and "无匹配抓取" in meta["source"]


def test_non_financials_cache_entries_are_ignored():
    cache = {"list_metrics|{}": {"metrics": {"ratios": ["roe"]}},
             "get_data_period|{}": {"earliest_date": "2016-09-30"}}
    series, meta = extract_series(cache, "roe", NAB_EX, NAB_CODE)
    assert series == {} and meta["n"] == 0


# ---------------------------------------------------------------------------
# 口径锚定:缓存里往往有多份同公司抓取,优先选能自己判出口径的那份
#
# 起因是真实运行:skill 07 把 pe/pb/roe(all)与 roe/payoutratio/dps(5y)拆成两次抓取,
# 声明 roe_window=5y,于是选到只有 roe 的那份 —— 里面没有 pb/pe,恒等式 pb/pe ÷ roe
# 无从计算,roe 只能按原值用(σ 于是从"判定过的年度序列"退化成"原值序列",对按季披露的
# 公司差 4 倍)。缓存是跨节点累积的,一份缺参照指标的抓取还会因为观测点更多而胜出。
# ---------------------------------------------------------------------------

ANCHOR_DATES = ["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]


def _anchored_raw(roe: float = 0.11) -> dict:
    """三指标齐全、能自证口径的抓取:pb/pe = 1/(1/roe) 精确成立是年度口径的特征。"""
    return {"data": {d: {"ratios": {"roe": roe, "pb": 1.20, "pe": 1.20 / roe}}
                     for d in ANCHOR_DATES}}


def _roe_only_raw(values: dict[str, float]) -> dict:
    return {"data": {d: {"ratios": {"roe": v}} for d, v in values.items()}}


def _raw_cache(*calls) -> dict:
    out = {}
    for period, raw, metrics in calls:
        args = {"exchange": NAB_EX, "code": NAB_CODE, "metrics": metrics, "period": period}
        out[f"get_financials|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"] = raw
    return out


def test_declared_window_without_the_anchor_yields_to_an_anchorable_fetch():
    """声明的 5y 里只有 roe(锚不住),就改用那份能锚定的 all —— 不能拿未判定的序列算 σ。"""
    cache = _raw_cache(
        ("5y", _roe_only_raw(dict(zip(DATES, [0.112, 0.11212, 0.10849, 0.11222]))), ["roe"]),
        ("all", _anchored_raw(), ["roe", "pb", "pe"]),
    )
    block, meta = annual_block(cache, NAB_EX, NAB_CODE, period="5y", metric="roe")

    assert meta["period"] == "all"                     # 实际用的窗口回传给调用方核对
    assert block["roe"].determined is True
    assert any("pb/pe" in e for e in block["roe"].evidence)


def test_anchor_beats_point_count_and_the_declared_window():
    """同一窗口里:锚得住但点少的那份 > 锚不住但点多的那份(旧实现按点数选错了)。"""
    longer = {**dict(zip(DATES, [0.112, 0.11212, 0.10849, 0.11222])),
              "2024-09-30": 0.111, "2024-12-31": 0.1105}          # 6 个点,无 pb/pe
    cache = _raw_cache(
        ("5y", _roe_only_raw(longer), ["roe"]),
        ("5y", _anchored_raw(), ["roe", "pb", "pe"]),
    )
    block, meta = annual_block(cache, NAB_EX, NAB_CODE, period="5y", metric="roe")

    assert meta["period"] == "5y"
    assert block["roe"].determined is True
    assert meta["evidence"] and any("pb/pe" in e for e in meta["evidence"])


def test_a_tiny_anchorable_fetch_does_not_hijack_the_history():
    """锚定优先不能压过信息量:一份 1 个点的抓取不该顶掉 40 个点的历史。"""
    long_values = {f"20{y:02d}-12-31": 0.10 + y / 1000 for y in range(10, 26)}
    one_point = {"data": {"2026-03-31": {"ratios": {"roe": 0.11, "pb": 1.2, "pe": 10.9}}}}
    cache = _raw_cache(
        ("all", _roe_only_raw(long_values), ["roe"]),
        ("1y", one_point, ["roe", "pb", "pe"]),
    )
    series, meta = extract_series(cache, "roe", NAB_EX, NAB_CODE)

    assert meta["period"] == "all" and meta["n"] == len(long_values)


def test_unanchorable_cache_keeps_the_old_selection():
    """一份都锚不住时退回原规则(声明的窗口优先、否则信息量最大)—— 不能因为改选法改变语义。"""
    cache = _raw_cache(
        ("5y", _roe_only_raw(dict(zip(DATES, [0.112, 0.11212, 0.10849, 0.11222]))), ["roe"]),
        ("all", _roe_only_raw({f"20{y:02d}-12-31": 0.10 for y in range(10, 26)}), ["roe"]),
    )
    series, meta = extract_series(cache, "roe", NAB_EX, NAB_CODE, period="5y")
    assert meta["period"] == "5y" and meta["n"] == 4


# ---------------------------------------------------------------------------
# 整块注入:要挑"既带这些指标、又判得出它们的口径"的那份
#
# 实测形状(ASX:NAB,2026-09-17 运行):缓存里 10 份抓取,其中一份 11 个指标的抓取几乎
# 全是资产负债表项 —— 存量指标的 determined 是白送的,旧打分让它以 11 分胜出,于是
# `describe(block, 要求的指标)` 一个都筛不出来,节点注入直接是空的(dps/epsBasic/roe
# 全没进上下文)。记账口径表也只剩两行存量指标。
# ---------------------------------------------------------------------------

def _stock_heavy_raw(n: int = 12) -> dict:
    """全是资产负债表项:指标多、点数多,但一个流量指标都没有。"""
    names = [f"stock{i}" for i in range(n)]
    return {"data": {d: {"balance-sheet": {m: 100.0 for m in names}} for d in ANCHOR_DATES}}


def test_focus_metrics_beat_a_stock_heavy_fetch():
    """整块调用带上要用的指标名单:选中能覆盖它们的那份,而不是资产负债表大杂烩。"""
    flows = ("dps", "epsBasic", "roe")
    good = {"data": {d: {"ratios": {"roe": 0.11}, "income-statement": {"dps": 1.7, "epsBasic": 1.99}}
                     for d in ANCHOR_DATES}}
    cache = _raw_cache(
        ("all", _stock_heavy_raw(), [f"stock{i}" for i in range(12)]),
        ("all", good, list(flows)),
    )
    block, meta = annual_block(cache, NAB_EX, NAB_CODE, metrics=flows)

    assert "stock0" not in block                       # 选中的不是那份资产负债表大杂烩
    assert set(flows) <= set(block)                    # 要用的指标都在
    assert block["dps"].values
    assert block["dps"].determined is False            # 缺参照,如实标未判定(不假装已判定)


def test_determined_coverage_outranks_raw_coverage():
    """覆盖全但判不出口径 的那份 vs 覆盖少但判定得出 的那份:后者胜,且理由要能看见。

    注入给 LLM 的值是让它**直接采用**的:一份覆盖 5 个指标、却一个都没判定的抓取,等于把
    可能差 4 倍的值喂进去(实测 WFC 的 roe/eps 都是单季值)。
    """
    focus = ("epsBasic", "roe", "netinccmn", "dps", "payoutratio")
    wide = {"data": {d: {"income-statement": {m: 1.0 for m in ("epsBasic", "dps", "netinccmn",
                                                              "payoutratio")},
                        "ratios": {"roe": 0.11}} for d in ANCHOR_DATES}}
    anchored = {"data": {d: {"ratios": {"roe": 0.11, "pb": 1.2, "pe": 1.2 / 0.11},
                             "income-statement": {"epsBasic": 1.99}} for d in ANCHOR_DATES}}
    cache = _raw_cache(("all", wide, list(focus)), ("all", anchored, ["roe", "pb", "pe", "epsBasic"]))

    block, meta = annual_block(cache, NAB_EX, NAB_CODE, metrics=focus)
    assert block["roe"].determined is True             # 选中的是能判定的那份
    assert block["epsBasic"].determined is True
    assert any("pb/pe" in e for e in block["roe"].evidence)
    assert "dps" not in block                          # 宁可少给,也不给判不出口径的值


def test_cached_metrics_lists_what_was_fetched_for_the_company():
    """报告用它区分"没抓到"与"抓到了但判不出口径"两件事。"""
    with_dps = {"data": {d: {"ratios": {"roe": 0.11},
                             "income-statement": {"dps": 1.7}} for d in DATES}}
    cache = _raw_cache(("all", _anchored_raw(), ["roe", "pb", "pe"]),
                       ("5y", with_dps, ["roe", "dps"]))
    assert cached_metrics(cache, NAB_EX, NAB_CODE) == {"roe", "pb", "pe", "dps"}
    # 别家公司不算进来
    assert cached_metrics(cache, "ASX", "WBC") == set()
