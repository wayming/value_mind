"""formulas.py 离线单测:全部用第九章书中的例子数字做断言。

书中两处印刷/翻译错误已勘误(见 formulas.py 模块 docstring),此处断言公式值。
"""

import math

import pytest

from app.formulas import (
    FormulaError,
    basis_conflict,
    composite_payout,
    cost_of_equity,
    ddm_gordon,
    ddm_multistage,
    equity_reinvestment_reg_capital,
    excess_return_per_period,
    excess_returns_perpetuity,
    excess_returns_staged,
    fair_price_from_pb,
    fcfe_from_net_income,
    fcfe_valuation,
    growth_from_roe,
    implied_pe,
    implied_pb,
    infer_periods_per_year,
    latest_nonnull,
    pb_regression,
    pct,
    pv_of_cashflows,
    ratio_series_stats,
    sample_std,
    sensitivity_grid,
    terminal_payout_ratio,
    weighted_mean,
)


def approx(a, b, rel=1e-3):
    return math.isclose(a, b, rel_tol=rel)


# ---- 股权成本(富国银行例)----

def test_cost_of_equity_wells_fargo():
    assert approx(cost_of_equity(0.036, 1.2, 0.05), 0.096)      # 3.6% + 1.2×5% = 9.6%
    assert approx(cost_of_equity(0.036, 1.0, 0.05), 0.086)      # 稳定期 β 趋 1 → 8.6%


# ---- 增长与派息 ----

def test_growth_from_roe_wells_fargo():
    # 13.51% × (1 − 54.63%) = 6.13%
    assert approx(growth_from_roe(0.1351, 0.5463), 0.0613)


def test_terminal_payout_ratio():
    # 稳定期派息率 = 1 − 3%/8.6% ≈ 65.12%
    assert approx(terminal_payout_ratio(0.03, 0.086), 0.6512)
    with pytest.raises(FormulaError):
        terminal_payout_ratio(0.09, 0.086)   # g >= ROE 非法
    with pytest.raises(FormulaError):
        terminal_payout_ratio(0.03, 0.0)     # ROE 非正非法


def test_composite_payout():
    assert approx(composite_payout(80.0, 20.0, 200.0), 0.5)
    with pytest.raises(FormulaError):
        composite_payout(80.0, 20.0, 0.0)


# ---- DDM ----

def test_ddm_gordon_diverges_when_coe_leq_g():
    with pytest.raises(FormulaError):
        ddm_gordon(1.0, 0.10, 0.09)


def test_ddm_multistage_wells_fargo():
    """富国银行例:表 9-1 为图片,eps0≈2.18 为反推值,每股 27.74±0.5 容差。"""
    r = ddm_multistage(
        eps0=2.18, payout_high=0.5463, g_high=0.0613, years_high=5,
        coe_high=0.096, g_terminal=0.03, roe_terminal=0.086,
        coe_terminal=0.086, roe_high=0.1351,
    )
    assert abs(r.value_per_share - 27.74) < 0.5
    assert len(r.dividends) == 5
    assert approx(r.payout_terminal, 0.6512)
    assert r.consistency_warning is None          # 6.13% 与 ROE×(1−派息率) 自洽


def test_ddm_multistage_consistency_warning():
    r = ddm_multistage(
        eps0=2.18, payout_high=0.5463, g_high=0.09, years_high=5,
        coe_high=0.096, g_terminal=0.03, roe_terminal=0.086,
        coe_terminal=0.086, roe_high=0.1351,
    )
    assert r.consistency_warning is not None      # g=9% 与 ROE×(1−派息率)=6.13% 不自洽


def test_pv_of_cashflows():
    assert approx(pv_of_cashflows([100.0, 100.0], 0.10), 100 / 1.1 + 100 / 1.21)
    assert approx(pv_of_cashflows([100.0, 100.0], [0.10, 0.20]), 100 / 1.1 + 100 / 1.44)
    with pytest.raises(FormulaError):
        pv_of_cashflows([100.0], [0.1, 0.2])


# ---- 监管资本 FCFE(书中银行例)----

def test_regulatory_capital_example():
    # 1 亿贷款增 10% → 1.1 亿;目标资本比率 7% → 需 770 万股权;
    # 现有股权 600 万 → 再投资 170 万;净收入 500 万 → FCFE 330 万
    reinvest = equity_reinvestment_reg_capital(100e6, 0.10, 0.07, 6e6)
    assert reinvest == pytest.approx(1.7e6)
    assert fcfe_from_net_income(5e6, reinvest) == pytest.approx(3.3e6)


def test_fcfe_valuation_basic():
    r = fcfe_valuation(
        net_income0=5e6, ni_growth=0.05, years_high=3,
        assets0=100e6, assets_growth=0.10, target_capital_ratio=0.07,
        equity0=6e6, coe=0.096, g_terminal=0.03, coe_terminal=0.086, shares=1e6,
    )
    assert len(r.fcfe) == 3
    assert len(r.capital_ratio_path) == 4            # 含期初
    assert r.capital_ratio_path[-1] >= r.capital_ratio_path[0]  # 留存使资本比率趋近目标
    assert r.value_per_share > 0


def test_fcfe_valuation_negative_terminal_fcfe():
    # 资产增速与目标资本比率过高 → 期末 FCFE 为负 → 抛错(缓冲区缺口过大)
    with pytest.raises(FormulaError):
        fcfe_valuation(
            net_income0=5e6, ni_growth=0.05, years_high=3,
            assets0=100e6, assets_growth=0.30, target_capital_ratio=0.12,
            equity0=6e6, coe=0.096, g_terminal=0.03,
        )


# ---- 超额回报模型(富国银行例,书中 58.22B/105.85B 为错误,断言公式值)----

def test_excess_return_per_period():
    assert approx(excess_return_per_period(47.63e9, 0.1351, 0.096), 47.63e9 * 0.0391)


def test_excess_returns_perpetuity_wells_fargo():
    r = excess_returns_perpetuity(47.63e9, 0.1351, 0.096, shares=2.362e9)
    assert approx(r.excess_pv, 19.40e9, rel=2e-3)
    assert approx(r.equity_value, 67.03e9, rel=2e-3)
    assert abs(r.value_per_share - 28.38) < 0.05   # 书中每股 28.38 美元 ✓


def test_excess_returns_below_book_when_roe_lt_coe():
    r = excess_returns_perpetuity(100.0, 0.05, 0.10)
    assert r.equity_value < 100.0


def test_excess_returns_staged_converges_to_perpetuity_as_floor():
    """roe_terminal == coe 时衰减版超额回报消失;永续版是上界。"""
    staged = excess_returns_staged(bv0=47.63e9, roe_high=0.1351, roe_terminal=0.096,
                                   coe=0.096, years_high=5, years_fade=10,
                                   payout=0.5463, shares=2.362e9)
    perpet = excess_returns_perpetuity(47.63e9, 0.1351, 0.096, shares=2.362e9)
    assert staged.value_per_share < perpet.value_per_share
    assert staged.value_per_share > 47.63e9 / 2.362e9          # 高于账面
    assert len(staged.excess_by_year) == 15


# ---- 相对估值 ----

def test_pb_regression_tompkins():
    """汤普金斯例:书中印 1.95 与系数矛盾,公式实算 3.2082。"""
    assert approx(pb_regression(0.2798, 0.2789), 3.2082, rel=1e-4)
    assert approx(fair_price_from_pb(pb_regression(0.2798, 0.2789), 30.0), 3.2082 * 30.0)


def test_implied_pe_and_pb():
    # 派息 54.63%、g 6.13%、COE 9.6% → PE ≈ 0.5463×1.0613/0.0347
    assert approx(implied_pe(0.5463, 0.0613, 0.096), 0.5463 * 1.0613 / 0.0347)
    with pytest.raises(FormulaError):
        implied_pe(0.5, 0.10, 0.09)
    assert approx(implied_pb(28.38, 20.0), 28.38 / 20.0)
    with pytest.raises(FormulaError):
        implied_pb(10.0, 0.0)


# ---- 统计与工具 ----

def test_sample_std():
    assert approx(sample_std([1.0, 2.0, 3.0]), 1.0)
    assert approx(sample_std([1.0, None, 3.0]), math.sqrt(2.0))
    with pytest.raises(FormulaError):
        sample_std([1.0])


def test_latest_nonnull():
    series = {"2024-06": None, "2024-12": 2.5, "2025-06": None}
    assert latest_nonnull(series) == ("2024-12", 2.5)
    assert latest_nonnull({"2024-06": None}) is None


def test_weighted_mean_and_pct():
    assert approx(weighted_mean({"a": 10.0, "b": 20.0}, {"a": 0.25, "b": 0.75}), 17.5)
    assert approx(weighted_mean({"a": 10.0, "b": 20.0}), 15.0)
    assert pct(0.096) == "9.60%"


def test_sensitivity_grid():
    fn = lambda g, coe, dps_last: ddm_gordon(dps_last, g, coe)  # noqa: E731
    grid = sensitivity_grid(fn, {"g=3%": 0.03, "g=4%": 0.04},
                            {"coe=9%": 0.09, "coe=8%": 0.08}, dps_last=1.0)
    assert grid[0]["coe=9%"] is not None
    assert grid[0]["coe=8%"] is not None
    assert grid[1]["coe=8%"] is not None
    assert grid[1]["coe=9%"] is not None


def test_infer_periods_per_year():
    """日期间隔的推断结果——现在只用于交叉核对(见 test_basis_is_not_inferred_from_date_spacing),
    不再用于决定年化倍数:它给的是采样频率,不是每个点代表多长期间。
    它仍要能识别出年度序列,因为"间隔一年却说每点是单季值"是硬矛盾。"""
    q = [f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in
         [(2021, 3, 31), (2021, 6, 30), (2021, 9, 30), (2021, 12, 31), (2022, 3, 31)]]
    assert infer_periods_per_year(q) == (4.0, "单季")

    a = ["2021-12-31", "2022-12-31", "2023-12-31"]
    assert infer_periods_per_year(a) == (1.0, "年度")

    m = [f"2024-{i:02d}-28" for i in range(1, 8)]
    assert infer_periods_per_year(m) == (12.0, "月度")

    h = ["2023-06-30", "2023-12-31", "2024-06-30"]
    assert infer_periods_per_year(h) == (2.0, "半年度")

    assert infer_periods_per_year(["2025-12-31"])[0] == 1.0  # 单点无法判断,按年度
def test_ratio_series_stats_annualizes_per_declared_basis():
    """单季 ROE 序列:按 LLM 声明的口径年化,且 σ 与均值同口径。"""
    series = {"2025-03-31": 0.029, "2025-06-30": 0.031,
              "2025-09-30": 0.025, "2025-12-31": 0.027}
    st = ratio_series_stats(series, "单季")
    assert approx(st["periods_per_year"], 4.0)
    assert approx(st["raw_latest"], 0.027)      # 原始值保留,可追溯
    assert approx(st["latest"], 0.108)          # 年化后
    assert approx(st["mean"], 0.112)
    assert st["n"] == 4
    assert st["latest_date"] == "2025-12-31"
    assert st["basis_declared"] is True
    # σ 也年化:与 latest/mean 同单位,不能被落下
    assert approx(st["std"], sample_std([0.029, 0.031, 0.025, 0.027]) * 4)


def test_ratio_series_stats_annual_basis_is_not_scaled():
    """声明为年度口径的序列不能被改动(×1),否则又是反向偏差。"""
    st = ratio_series_stats({"2023-12-31": 0.11, "2024-12-31": 0.135}, "年度")
    assert approx(st["periods_per_year"], 1.0)
    assert approx(st["latest"], 0.135)
    assert approx(st["std"], sample_std([0.11, 0.135]))


def test_basis_is_not_inferred_from_date_spacing():
    """实测反例:NAB 的 roe 是滚动年度值、WFC 的是单季值,而两者日期间隔都是 91 天。

    所以同样一份季度采样的序列,声明单季和声明年度必须得出不同的数——
    日期里根本不含"每个点代表多长期间"这个信息。
    """
    quarterly_dates = {"2025-03-31": 0.10, "2025-06-30": 0.11,
                       "2025-09-30": 0.11, "2025-12-31": 0.12}
    as_quarterly = ratio_series_stats(quarterly_dates, "单季")
    as_annual = ratio_series_stats(quarterly_dates, "年度")
    assert as_quarterly["periods_per_year"] == 4.0
    assert as_annual["periods_per_year"] == 1.0
    # 两种声明下 date_implied 完全相同 —— 证明日期判定不了口径
    assert as_quarterly["date_implied"] == as_annual["date_implied"] == "单季"
    assert not basis_conflict(as_annual)          # 季度采样年度值:合法,不是矛盾


def test_undeclared_basis_defaults_to_no_scaling_and_is_flagged():
    st = ratio_series_stats({"2025-03-31": 0.03, "2025-06-30": 0.03})
    assert approx(st["periods_per_year"], 1.0)
    assert st["basis_declared"] is False


def test_basis_conflict_only_for_annual_spacing_claimed_as_quarterly():
    """只判一种硬矛盾:采样间隔约一年却说每点是单季/月度值。"""
    annual_dates = {"2023-12-31": 0.02, "2024-12-31": 0.025, "2025-12-31": 0.03}
    assert basis_conflict(ratio_series_stats(annual_dates, "单季")) is not None
    assert basis_conflict(ratio_series_stats(annual_dates, "年度")) is None


def test_ratio_series_stats_edge_cases():
    with pytest.raises(FormulaError):
        ratio_series_stats({"2025-12-31": None}, "年度")
    with pytest.raises(FormulaError):
        ratio_series_stats({"2025-12-31": 0.03}, "每季度")   # 未定义的口径标签
    st = ratio_series_stats({"2025-12-31": 0.03}, "年度")     # 单点:无 σ,但不报错
    assert "std" not in st
    with pytest.raises(FormulaError):
        infer_periods_per_year(["not-a-date", "2025-12-31"])


def test_annualization_changes_pb_regression():
    """回归里 ROE 年化而 σ 不年化会机械抬高预测 PB —— 锁住这个反向偏差。"""
    quarterly = {"2025-03-31": 0.029, "2025-06-30": 0.031,
                 "2025-09-30": 0.025, "2025-12-31": 0.027}
    st = ratio_series_stats(quarterly, "单季")
    roe = 0.125
    consistent = pb_regression(roe, st["std"])                 # 两者都年化(正确)
    mixed = pb_regression(roe, sample_std(list(quarterly.values())))  # σ 未年化(错误)
    assert mixed > consistent
    assert approx(consistent, 1.527 + 8.63 * roe - 2.63 * st["std"])
