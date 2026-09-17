"""口径归一化(app/series.py)。

用 2026-09 从线上 MCP 抓下来的四份真实响应(tests/fixtures/)离线跑。断言分三类:

1. **外部事实**:归一化后的数字要与公司真实年报对得上 —— 这是唯一能证明"口径修对了"
   的证据。ASX:NAB 的 FY17–FY21 每股股息 1.98/1.98/1.66/0.60/1.34、法定净利润
   FY20 25.6 亿/FY21 63.6 亿,ASX:CBA 的 FY24/FY25 股息 4.65/5.05 都是公开数字。
2. **交叉自证**:以数据源自己的市价比率反推它用的每股收益(WFC 的 pb×bvps/pe = 6.5
   正好等于 epsBasic×4),证明倍数不是凑出来的。
3. **不误伤**:存量余额、无量纲比率、以及真实的口径/水平变化(WFC 2021 年股息翻倍)
   都不能被改。
"""

import copy
import json
from pathlib import Path

import pytest

from app import series as S

FIXTURES = Path(__file__).parent / "fixtures"


def raw_of(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))["raw"]


@pytest.fixture(scope="module")
def nab_all():
    return S.annualize(raw_of("nab_all"))


@pytest.fixture(scope="module")
def nab_5y():
    return S.annualize(raw_of("nab_5y"))


@pytest.fixture(scope="module")
def cba_5y():
    return S.annualize(raw_of("cba_5y"))


@pytest.fixture(scope="module")
def wfc_5y():
    return S.annualize(raw_of("wfc_5y"))


# ---------------------------------------------------------------------------
# 1. 外部事实:数字要与真实年报对得上
# ---------------------------------------------------------------------------

def test_nab_dividends_match_reported_annual_figures(nab_all):
    """NAB FY17–FY21 的年度股息(公开数字)必须能从归一化序列里读出来。

    归一化前这一段的原值是年报值 ÷ 4(0.495/0.415/0.15/0.335),DDM 直接拿去做
    eps0 会让整条预测股息序列缩到 1/4。
    """
    dps = nab_all["dps"]
    assert dps.factor == 1.0, "近期是 12 个月 TTM,基准倍数应为 1"
    for date, reported in (("2017-03-31", 1.98), ("2018-06-30", 1.98),
                           ("2019-06-30", 1.66), ("2020-06-30", 0.60),
                           ("2021-06-30", 1.34)):
        assert dps.values[date] == pytest.approx(reported, rel=0.12), date
    # 归一化前的原值确实是 1/4,证明这一段真的需要换算
    assert dps.raw["2017-03-31"] == pytest.approx(1.98 / 4, rel=0.02)


def test_nab_net_income_matches_reported_figures(nab_all):
    """法定净利润:FY20 25.6 亿、FY21 63.6 亿(公开数字)。"""
    ni = nab_all["netinccmn"]
    assert ni.values["2020-03-31"] == pytest.approx(2.56e9, rel=0.15)
    assert ni.values["2021-06-30"] == pytest.approx(6.36e9, rel=0.10)
    assert ni.raw["2021-06-30"] == pytest.approx(1.578e9)          # 原值 = 年报 ÷ 4


def test_cba_dividends_match_reported_figures(cba_5y):
    """CBA 是**半年度**口径(半年值 ×2),FY24/FY25 股息 4.65/5.05。"""
    dps = cba_5y["dps"]
    assert dps.values["2023-12-31"] == pytest.approx(4.30, rel=0.12)   # FY23 = 2.10+2.25
    assert dps.values["2024-06-30"] == pytest.approx(4.65, rel=0.05)   # FY24
    assert dps.values["2026-06-30"] == pytest.approx(5.05, rel=0.05)   # FY25
    assert '×2' in dps.evidence[-1]                                    # 证据里讲清了倍数


# ---------------------------------------------------------------------------
# 2. 交叉自证:数据源自己的比率反推出同一个倍数
# ---------------------------------------------------------------------------

def test_wfc_roe_and_eps_are_quarterly(wfc_5y):
    """WFC 的 roe/eps/dps 都是单季值,须 ×4。

    自证:用数据源自己的 pb×bvps 得价格、再 ÷pe 得它计算 pe 时用的每股收益
    (2026-03-31 = 1.97),与 epsBasic×4 = 6.48 不同量级 —— 它用的是**年度**值,
    也就是说 epsBasic 本身是单季。roe ×4 = 11.7% 也才对得上富国银行的真实 ROE。
    """
    roe = wfc_5y["roe"]
    assert roe.factor == 4.0
    assert roe.values["2026-03-31"] == pytest.approx(0.1165, abs=0.01)
    assert wfc_5y["epsBasic"].factor == 4.0
    assert wfc_5y["epsBasic"].values["2026-03-31"] == pytest.approx(6.48)
    assert wfc_5y["dps"].factor == 4.0

    # 归一化后 dps/eps 必须等于数据源自己给出的 payoutratio(0.2774)——两条独立
    # 路径算出同一个派息率,说明倍数没错。
    d, e = wfc_5y["dps"].values["2026-03-31"], wfc_5y["epsBasic"].values["2026-03-31"]
    assert d / e == pytest.approx(wfc_5y["payoutratio"].values["2026-03-31"], rel=0.02)


def test_cba_half_year_block_detected(cba_5y):
    """CBA 的 roe 只有最近 10 个点(2024 起),前半段的半年度口径靠"最近 roe 代表值"看穿。"""
    ni = cba_5y["netinccmn"]
    assert ni.raw["2022-06-30"] == pytest.approx(4.901e9)          # 半年值
    assert ni.values["2022-06-30"] == pytest.approx(9.802e9)       # ×2 后 = 年度
    assert ni.values["2026-06-30"] == pytest.approx(1.0866e10)     # 已是年度,不动
    assert ni.adjusted and all("×2" in v for v in ni.adjusted.values())


# ---------------------------------------------------------------------------
# 3. 尾部坏点与"不误伤"
# ---------------------------------------------------------------------------

def test_tail_outliers_are_corrected(nab_all, nab_5y):
    """尾部两个坏点(0.425 = 邻域 ÷4、3.400 = 邻域 ×2)都要回到 1.70。"""
    for block in (nab_all, nab_5y):
        dps = block["dps"]
        assert dps.values["2025-06-30"] == pytest.approx(1.70)
        assert dps.values["2026-06-30"] == pytest.approx(1.70)
        assert "2025-06-30" in dps.adjusted and "2026-06-30" in dps.adjusted
    # 数据源的 dividendGrowth 尾部被这两个坏点污染成 +300%,自己算的不受影响
    assert nab_all["dividendGrowth"].values["2025-09-30"] == pytest.approx(3.0)
    assert nab_all["dps"].growth() == pytest.approx(0.012, abs=0.02)


def test_real_level_change_is_not_corrected(wfc_5y):
    """WFC 2021 年股息翻倍(0.10→0.20)是**真实变化**,不是坏点。

    它与坏点的区别:换挡点两侧邻居分属两个水平,而坏点前后都站在同一水平上。
    误判的代价是把真实数据砍掉一半。
    """
    dps = wfc_5y["dps"]
    assert dps.raw["2021-06-30"] == pytest.approx(0.10)
    assert dps.values["2021-06-30"] == pytest.approx(0.40)         # 只做口径换算
    assert "2021-06-30" not in dps.adjusted
    assert dps.values["2021-09-30"] == pytest.approx(0.80)         # 之后是 0.20×4


def test_stocks_and_scale_free_ratios_are_never_touched(nab_all, cba_5y, wfc_5y):
    """存量余额(时点)与无量纲比率(pe/pb/payoutratio…)一律不缩放、不校正。"""
    for block in (nab_all, cba_5y, wfc_5y):
        for metric in ("bvps", "totalCommonEquity", "sharesOutTotalCommon",
                       "pb", "pe", "payoutratio"):
            s = block.get(metric)
            if s is None:
                continue
            assert s.factor == 1.0 and not s.adjusted, metric
            assert s.determined                                            # 不需要锚定
    # NAB 2021-06-30 的净资产(627.8 亿)是时点余额,不能被"整块翻转"带成 4 倍
    assert nab_all["totalCommonEquity"].values["2021-06-30"] == pytest.approx(6.2779e10)


def test_nab_5y_single_pre_period_point_is_fixed(nab_5y):
    """5 年窗口里只有 2021-06-30 一个前期口径点,孤零零的也要换算。"""
    ni = nab_5y["netinccmn"]
    assert ni.raw["2021-06-30"] == pytest.approx(1.578e9)
    assert ni.values["2021-06-30"] == pytest.approx(6.312e9)
    assert "2021-06-30" in ni.adjusted


# ---------------------------------------------------------------------------
# 4. 判不出来时不许装作没事
# ---------------------------------------------------------------------------

def test_uniform_ttm_series_is_a_no_op():
    """服务器若统一给 12 个月 TTM,这一层必须什么都不做、也不告警。"""
    dates = [f"20{20 + i // 4}-{3 * (i % 4) + 3:02d}-30" for i in range(8)]
    roe = {d: 0.12 + 0.001 * i for i, d in enumerate(dates)}
    pb = {d: 1.5 for d in dates}
    pe = {d: 12.5 for d in dates}                     # pb/pe ÷ roe = 1.0
    equity = {d: 60e9 for d in dates}
    netinc = {d: 60e9 * roe[d] for d in dates}
    raw = {"metric_sources": {"roe": "ratios", "pb": "ratios", "pe": "ratios",
                              "netinccmn": "income-statement",
                              "totalCommonEquity": "balance-sheet"},
           "data": {d: {"ratios": {"roe": roe[d], "pb": pb[d], "pe": pe[d]},
                        "income-statement": {"netinccmn": netinc[d]},
                        "balance-sheet": {"totalCommonEquity": equity[d]}}
                    for d in dates}}
    block = S.annualize(raw)
    assert block["netinccmn"].factor == 1.0 and block["roe"].factor == 1.0
    assert not block["netinccmn"].adjusted and not block["roe"].adjusted
    assert not block["netinccmn"].warnings and block["netinccmn"].determined


def test_unanchored_series_warns_instead_of_silently_using_raw():
    """没有 pb/pe/roe/净资产可比时,必须 determined=False + 告警,而不是静默按原值。"""
    raw = {"metric_sources": {"netinccmn": "income-statement"},
           "data": {"2024-06-30": {"income-statement": {"netinccmn": 1e9}},
                    "2024-09-30": {"income-statement": {"netinccmn": 1.1e9}}}}
    ni = S.annualize(raw)["netinccmn"]
    assert ni.factor == 1.0 and not ni.determined
    assert ni.warnings and "系统性偏小" in ni.warnings[0]


def test_annualize_is_pure_and_handles_empty():
    """不改输入;空结果不抛。"""
    raw = raw_of("nab_5y")
    before = copy.deepcopy(raw)
    S.annualize(raw)
    assert raw == before
    assert S.annualize({}) == {}
    assert S.annualize(None) == {}


# ---------------------------------------------------------------------------
# 5. 与现有统计/描述层的衔接
# ---------------------------------------------------------------------------

def test_stats_reuse_formula_layer(nab_all):
    """统计量复用 formulas.ratio_series_stats(不另起一套算术)。"""
    st = nab_all["roe"].stats()
    assert st["n"] == len(nab_all["roe"].values)
    assert st["periods_per_year"] == 1.0
    assert st["latest"] == pytest.approx(0.09908)
    assert st["std"] > 0
    assert st["basis"] == "年度(TTM)" and st["determined"] is True
    assert st["latest_date"] == "2026-03-31"


def test_describe_gives_the_llm_the_basis(nab_all):
    """交给 LLM 的上下文里必须带口径与依据,替掉"靠量级猜口径"的提示词。"""
    out = S.describe(nab_all, ("roe", "dps", "payoutratio"))
    assert out["roe"]["basis"] == "年度(TTM)"
    assert out["dps"]["latest"] == pytest.approx(1.70)
    assert "恒等式" in out["dps"]["evidence"] or "净资产" in out["dps"]["evidence"]
    assert out["payoutratio"]["basis"] == "无量纲比率(不年化)"
    assert "adjusted_points" in out["dps"] and len(out["dps"]["adjusted_points"]) <= 8
    assert set(out) == {"roe", "dps", "payoutratio"}       # 未请求的指标不出现
