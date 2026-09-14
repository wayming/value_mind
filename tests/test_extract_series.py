"""extract_series 的离线回归:序列必须属于一家公司、且来自单次抓取。

起因是一次真实运行:ASX NAB 的报告里 `roe_series_raw_latest` = 0.0949,
而 NAB 自己的值是 0.09908 —— 0.0949 是西太平洋银行(WBC)的。skill 07 会抓同业做
对比,两家澳洲银行的报告日完全相同(3/31、9/30),旧实现只按日期合并,
于是同业的值逐日覆盖进目标公司序列,σ 成了两家公司的混合体,直接喂进 PB 回归,
报告里看不出任何异常。这两个测试锁住修复。
"""

import json

from app.mcp_tools import extract_series

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
