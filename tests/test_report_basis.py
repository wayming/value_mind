"""报告里的数据口径表,以及"口径由 Python 判定"这条路线在 schema 上的收口。

口径是数字的一部分:报告里出现 1.70 之前,读者得知道它是怎么来的(原始值是年报值 ÷4 的
0.425,被按邻域水平修正过)。同时锁住 schema —— 别再让 LLM 去申报序列口径:那是被
app/series.py 用尺度无关的恒等式算出来的,靠量级猜会静默错 4 倍。
"""

import json
from pathlib import Path

import pytest

from app import schemas, series as S
from app.report import _basis_table, build_report

FIXTURES = Path(__file__).parent / "fixtures"


def raw_of(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))["raw"]


def state_with(name: str, exchange: str, code: str, period: str) -> dict:
    args = {"exchange": exchange, "code": code, "period": period}
    key = f"get_financials|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    return {"exchange": exchange, "code": code, "data_cache": {key: raw_of(name)}}


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def test_basis_table_shows_basis_evidence_and_corrections():
    table = "\n".join(_basis_table(state_with("nab_all", "ASX", "NAB", "all")))
    assert "年度(12 个月)口径" in table
    assert "| dps | 年度(TTM) | 1.70(2026-06-30) |" in table
    assert "pb/pe ÷ roe" in table                       # 判定依据
    assert "0.4250→1.70" in table                       # 被修正的尾部坏点:原值 → 现值
    assert "时点余额(不年化)" in table                  # 存量余额不年化
    assert "bvps" in table and "roe" in table
    # 只要关心的那几个指标,不把整份抓取铺开
    assert "dividendGrowth" not in table


def test_basis_table_is_empty_without_data_and_never_raises():
    assert _basis_table({}) == []
    assert _basis_table({"exchange": "ASX", "code": "NAB", "data_cache": {}}) == []
    # 缓存里只有别家公司的抓取、或结构不对(不是结构化 dict):跳过即可,不能把报告生成搞崩
    other = state_with("wfc_5y", "NYSE", "WFC", "5y")
    other["exchange"], other["code"] = "ASX", "NAB"
    assert _basis_table(other) == []
    broken = {"exchange": "ASX", "code": "NAB",
              "data_cache": {"get_financials|{}": ["不是结构化结果"],
                             "list_metrics|{}": {"metrics": {"ratios": ["roe"]}}}}
    assert _basis_table(broken) == []
    # 好坏混在一起时,好的那份照常出表
    mixed = state_with("nab_all", "ASX", "NAB", "all")
    mixed["data_cache"].update(broken["data_cache"])
    assert any("年度(TTM)" in line for line in _basis_table(mixed))


def _cache(*calls) -> dict:
    """{`get_financials|<参数>`: 原始 structuredContent} —— 与真实缓存同形。"""
    out = {}
    for exchange, code, period, metrics, raw in calls:
        args = {"exchange": exchange, "code": code, "metrics": metrics, "period": period}
        out[f"get_financials|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"] = raw
    return out


def test_basis_table_picks_the_fetch_that_proves_the_metrics():
    """指标多不等于有用:一份全是资产负债表项的抓取曾经靠白送的 determined 胜出,
    于是口径表只剩两行存量指标,dps/epsBasic 的来龙去脉全没了。"""
    stock = {"data": {d: {"balance-sheet": {f"stock{i}": 100.0 for i in range(11)}}
                      for d in ("2025-12-31", "2026-03-31")}}
    state = {"exchange": "ASX", "code": "NAB",
             "data_cache": _cache(("ASX", "NAB", "all", list(f"stock{i}" for i in range(11)),
                                   stock),
                                  ("ASX", "NAB", "all",
                                   ["dps", "epsBasic", "roe", "netinccmn",
                                    "totalCommonEquity", "pb", "pe"], raw_of("nab_all")))}
    table = "\n".join(_basis_table(state))
    assert "| dps | 年度(TTM) | 1.70(2026-06-30) |" in table
    assert "0.4250→1.70" in table
    assert "stock0" not in table


def test_basis_table_distinguishes_unproven_from_unfetched():
    """没进表的指标要能区分"抓到了但锚不住口径"与"压根没抓到":处理方式完全不同。"""
    rates = {"data": {d: {"ratios": {"roe": 0.11, "pb": 1.2, "pe": 1.2 / 0.11}}
                      for d in ("2025-12-31", "2026-03-31", "2026-06-30")}}
    flows = {"data": {d: {"income-statement": {"dps": 1.7, "epsBasic": 1.99}}
                      for d in ("2025-12-31", "2026-03-31", "2026-06-30")}}
    state = {"exchange": "ASX", "code": "NAB",
             "data_cache": _cache(("ASX", "NAB", "all", ["roe", "pb", "pe"], rates),
                                  ("ASX", "NAB", "5y", ["dps", "epsBasic"], flows))}
    table = "\n".join(_basis_table(state))
    assert "未判定口径:epsBasic、dps" in table            # 抓到了,但缺参照指标
    assert "未取到:" in table and "netinccmn" in table    # 本次运行没抓
    assert "roe" in table                                 # 判定得出的照常进表


def test_report_contains_the_basis_section():
    state = state_with("wfc_5y", "NYSE", "WFC", "5y")
    state.update({"skills": {}, "coefficients": {}, "company_name": "富国银行"})
    md = build_report(state)
    assert "## 数据口径(归一化)" in md
    assert "单季(已×4)" in md                            # WFC 每点是单季值
    assert "epsBasic" in md


# ---------------------------------------------------------------------------
# schema:口径不再由 LLM 申报
# ---------------------------------------------------------------------------

def test_schema_no_longer_asks_the_llm_to_declare_the_basis():
    fields = schemas.DividendsGrowthParams.model_fields
    assert "roe_series_basis" not in fields
    # σ 的降级假设仍在(数据取不到时的兜底),但它不是"口径申报"
    assert "roe_std_dev_assumed" in schemas.RelativeParams.model_fields
    # 关键字段的说明要指向 Python 给的年化值,而不是"自己换算"
    assert "Python 判定口径后的年度数据" in fields["dps_by_year"].description
    assert "Python 判定口径后的年度数据" in schemas.DdmParams.model_fields["eps0"].description


# ---------------------------------------------------------------------------
# 同比:Python 算,且不采用数据源被污染的那个
# ---------------------------------------------------------------------------

def test_describe_adds_a_python_computed_growth_for_flows_only():
    block = S.annualize(raw_of("nab_all"))
    out = S.describe(block, ("dps", "roe", "totalCommonEquity"))
    assert out["dps"]["growth_yoy"] == pytest.approx(0.012, abs=0.02)
    assert "growth_yoy" not in out["roe"]                # 比率指标不算"金额同比"
    assert "growth_yoy" not in out["totalCommonEquity"]  # 时点余额更没有同比


# ---------------------------------------------------------------------------
# by_year:按年字段的数据来源(skill 03 连抓 13 次的根因)
# ---------------------------------------------------------------------------

def test_by_year_lists_normalized_observations_per_year():
    """逐年观测必须是**年化后**的值:0.425 → 1.70,不是原始坏点。"""
    block = S.annualize(raw_of("nab_all"))
    by_year = S.by_year(block["dps"])

    assert list(by_year) == sorted(by_year) and len(by_year) <= 6
    assert by_year["2026"]["2026-06-30"] == pytest.approx(1.70)
    assert by_year["2025"]["2025-06-30"] == pytest.approx(1.70)
    assert all(len(pts) <= 4 for pts in by_year.values())
    assert all(d.startswith(y) for y, pts in by_year.items() for d in pts)


def test_by_year_is_bounded_for_quarterly_companies():
    """按季披露的公司(每季一个点):只保留最近几年、每年最近几个点。"""
    values = {f"{y}-{m:02d}-30": 1.0 + i / 100
              for i, (y, m) in enumerate([(y, m) for y in range(2015, 2027)
                                          for m in (3, 6, 9, 12)])}
    s = S.AnnualSeries(metric="dps", kind="flow", values=values, raw=values)

    by_year = S.by_year(s)
    assert len(by_year) == 6                              # 12 年数据只给最近 6 年
    assert list(by_year)[-1] == "2026" and len(by_year["2026"]) == 4
    assert list(by_year["2026"]) == ["2026-03-30", "2026-06-30", "2026-09-30", "2026-12-30"]


def test_provenance_section_separates_mcp_data_from_assumptions():
    """报告要能让读者分清:哪个数字有数据背书,哪个是 LLM 自己填的。"""
    state = {
        "exchange": "ASX", "code": "NAB", "company_name": "NAB", "status": "ok",
        "data_cache": {"list_metrics|{}": {"metrics": {"ratios": ["roe"]}}},
        "skills": {"ddm": {"provenance": [
            {"field": "eps0", "value": 1.993248, "source": "MCP epsBasic",
             "note": "与数据年度值 1.993248(2026-06-30, 年度(TTM))一致"},
            {"field": "g_terminal", "value": 0.03, "source": "LLM 假设",
             "note": "MCP 无对应指标(如 β/Rf/ERP),依据见该字段的理由说明"},
        ]}},
    }
    report = build_report(state)

    assert "## 参数来源核对(哪些来自 MCP、哪些是 LLM 假设)" in report
    assert "| 股息贴现模型(DDM) | eps0 | 1.99 | MCP epsBasic |" in report
    assert "| g_terminal | 0.0300 | LLM 假设 |" in report
    assert "见上一节" in report


def test_growth_lag_follows_the_sampling_frequency():
    """一年一个采样点时同比不能变成"四年比"(服务器将来只给年度点)。"""
    quarterly = {f"20{y}-{m:02d}-30": 1.0 * (1.05 ** i)
                 for i, (y, m) in enumerate([(y, m) for y in range(2020, 2023)
                                             for m in (3, 6, 9, 12)])}
    annual = {"2020-12-31": 1.0, "2021-12-31": 1.05, "2022-12-31": 1.1025,
              "2023-12-31": 1.1576, "2024-12-31": 1.2155}
    assert S.growth_from_anchors(annual, 1) == pytest.approx(0.05, abs=1e-3)
    s = S.AnnualSeries(metric="dps", kind="flow", values=annual, raw=annual)
    assert s.sampling_per_year() == 1.0
    assert s.growth() == pytest.approx(0.05, abs=1e-3)   # 自动滞后 1 点 = 1 年
    assert s.growth(1) == pytest.approx(0.05, abs=1e-3)  # 显式指定仍然可用
    assert len(quarterly) == 12
