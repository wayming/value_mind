"""口径归一化接入节点(app/nodes.py)。

数据层(app/series.py)已经证明能把 NAB 的 ÷4 段、CBA 的半年度段、WFC 的单季段换算到
年度口径;这里锁的是**接线**,共三件事:

1. 归一化后的年度值真的进了 LLM 的上下文(而不是让它继续靠量级猜);
2. σ、归一化 ROE 的交叉校验用的是归一化后的序列;
3. LLM 万一仍然填了未年化的值,必须报出来(只告警不覆盖 —— 它也可能是在用归一化的
   调整值,代码无从判断哪个才是本意)。

跑法:状态里放真实抓取(tests/fixtures),把 run_skill_agent 换成返回预置参数,
公式与报告仍是真实代码 —— 不需要 LLM 与 MCP。
"""

import asyncio
import json
import statistics
from pathlib import Path

import pytest

import app.nodes as nodes
from app import schemas, series as S
from app.mcp_tools import annual_block, extract_series

FIXTURES = Path(__file__).parent / "fixtures"


def raw_of(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))["raw"]


def cache_of(name: str, exchange: str, code: str, period: str, metrics: list[str],
             extra: dict | None = None) -> dict:
    """一份真实的 data_cache:{`get_financials|<参数 json>`: 原始 structuredContent}。"""
    args = {"exchange": exchange, "code": code, "metrics": metrics, "period": period}
    cache = {f"get_financials|{json.dumps(args, sort_keys=True, ensure_ascii=False)}":
             raw_of(name)}
    cache.update(extra or {})
    return cache


def state_of(cache: dict, exchange="ASX", code="NAB", **over) -> dict:
    st = {"exchange": exchange, "code": code, "overrides": {}, "period_hint": "all",
          "methods_skip": [], "skills": {}, "coefficients": {}, "data_cache": cache,
          "valuation": {}, "warnings": [], "errors": []}
    st.update(over)
    return st


NAB_CACHE = cache_of("nab_all", "ASX", "NAB", "all",
                     ["dps", "epsBasic", "roe", "netinccmn", "totalCommonEquity", "pb", "pe"])
WFC_CACHE = cache_of("wfc_5y", "NYSE", "WFC", "5y",
                     ["dps", "epsBasic", "roe", "payoutratio", "pb", "pe"])


# ---------------------------------------------------------------------------
# 1. 交给 LLM 的上下文
# ---------------------------------------------------------------------------

def test_annual_ctx_hands_the_llm_annual_values():
    """ASX:NAB 的上下文必须是年报量级的股息(1.70),不是数据源那条 ÷4 的尾部坏点。"""
    ctx = nodes._annual_ctx(state_of(NAB_CACHE), ("dps", "epsBasic", "roe"))
    data = ctx[nodes._ANNUAL_KEY]

    assert data["dps"]["basis"] == "年度(TTM)"
    assert data["dps"]["latest"] == pytest.approx(1.70)
    assert data["dps"]["latest_date"] == "2026-06-30"
    assert data["roe"]["latest"] == pytest.approx(0.09908)
    assert data["dps"]["evidence"]                    # 判定的依据要一并交给模型
    assert data["dps"]["adjusted_points"]             # 被修正的坏点也要说明
    # 按年字段(dps_by_year 等)只能从 by_year 取:实测模型为了找 2024/2025 的年度值
    # 换着 period 抓了 13 次(get_financials),因为旧注入块只给了被修正的那几个点
    by_year = data["dps"]["by_year"]
    assert list(by_year) == sorted(by_year) and len(by_year) <= 6
    assert by_year["2026"]["2026-06-30"] == pytest.approx(1.70)     # 年化后的值,不是 3.40
    assert all(len(pts) <= 4 for pts in by_year.values())
    # 只要了三个指标,别的不能白塞进上下文
    assert set(data) == {"dps", "epsBasic", "roe"}


def test_annual_ctx_annualizes_a_quarterly_company():
    """NYSE:WFC 的每点是单季值:上下文给的是 ×4 之后的 6.48,不是 1.62。"""
    ctx = nodes._annual_ctx(state_of(WFC_CACHE, "NYSE", "WFC", period_hint="5y"),
                            ("epsBasic", "roe"))
    data = ctx[nodes._ANNUAL_KEY]
    assert data["epsBasic"]["basis"] == "单季(已×4)"
    assert data["epsBasic"]["latest"] == pytest.approx(6.48)
    assert data["roe"]["latest"] == pytest.approx(0.1165, abs=0.01)


def test_annual_ctx_without_data_injects_nothing():
    """缓存里还没有该公司数据时不注入(LLM 照常自己取数),不能抛。"""
    assert nodes._annual_ctx(state_of({}), ("roe",)) == {}
    other = state_of(cache_of("wfc_5y", "NYSE", "WFC", "5y", ["roe"]))   # 别家公司
    assert nodes._annual_ctx(other, ("roe",)) == {}


def test_annual_ctx_does_not_merge_other_companies():
    """同业抓取同时在缓存里:注入的必须是本公司的值(WBC 混入过 NAB 的报告)。"""
    cache = cache_of("nab_all", "ASX", "NAB", "all", ["roe", "pb", "pe"],
                     extra=cache_of("wbc_5y", "ASX", "WBC", "all", ["roe", "pb", "pe"])
                     if (FIXTURES / "wbc_5y.json").exists() else None)
    ctx = nodes._annual_ctx(state_of(cache), ("roe",))
    assert ctx[nodes._ANNUAL_KEY]["roe"]["latest"] == pytest.approx(0.09908)


# ---------------------------------------------------------------------------
# 2. LLM 仍填了未年化的值:要报出来,但不覆盖
# ---------------------------------------------------------------------------

def test_scale_check_catches_unannualized_value_and_stays_quiet_otherwise():
    block, _ = annual_block(WFC_CACHE, "NYSE", "WFC")
    # 单季 eps 填进年度字段:恰好 1/4,报出来
    [warn] = nodes._scale_warnings(block, [("eps0", 1.62, "epsBasic")])
    assert "eps0" in warn and "4" in warn and "口径" in warn
    # 正确的年度值、以及真实的归一化调整(±15%)都不该误报
    assert nodes._scale_warnings(block, [("eps0", 6.48, "epsBasic")]) == []
    assert nodes._scale_warnings(block, [("eps0", 5.51, "epsBasic")]) == []
    assert nodes._scale_warnings(block, [("eps0", None, "epsBasic")]) == []
    assert nodes._scale_warnings(block, [("eps0", 6.48, "不存在")]) == []


def test_scale_mismatch_only_flags_whole_multiples():
    """整倍数判定:2/4 倍命中,1 倍与真实的漂移不命中。"""
    assert S.scale_mismatch(1.62, 6.48) == pytest.approx(0.25)
    assert S.scale_mismatch(13.0, 6.48) == pytest.approx(2.0)
    assert S.scale_mismatch(6.48, 6.48) is None
    assert S.scale_mismatch(4.5, 6.48) is None          # 0.69 倍,不在整倍数窗口
    assert S.scale_mismatch(None, 6.48) is None
    assert S.scale_mismatch(1.62, None) is None


# ---------------------------------------------------------------------------
# 3. 节点 03:归一化 ROE 的交叉校验 + dps_by_year 的量级核对
# ---------------------------------------------------------------------------

def _dg_params(**over) -> schemas.DividendsGrowthParams:
    kw = dict(dividends_reliable=True, dps_by_year={"2025": 1.70, "2026": 1.70},
              payout_ratio=0.72, payout_basis="dps/eps 多年均值",
              roe_normalized=0.0991, roe_normalization_reason="周期与资本要求调整",
              shares_outstanding=3.05e9, analysis="测试")
    kw.update(over)
    return schemas.DividendsGrowthParams(**kw)


def _stub(monkeypatch, params):
    calls = []

    async def fake(skill_id, ctx, schema, *a, **kw):
        calls.append(ctx)
        return params, {}

    monkeypatch.setattr(nodes, "run_skill_agent", fake)
    return calls


def test_node03_compares_roe_against_the_annualized_series(monkeypatch):
    """归一化 ROE 与数据序列的比对必须用年化后的值:0.0991 对得上,季度值(0.0248)对不上。"""
    calls = _stub(monkeypatch, _dg_params())
    out = asyncio.run(nodes.dividends_growth_node(state_of(NAB_CACHE)))

    stats = out["skills"]["dividends_growth"]["results"]["roe_stats_from_data"]
    assert stats["basis"] == "年度(TTM)"
    assert stats["periods_per_year"] == 1.0
    assert stats["latest"] == pytest.approx(0.09908)
    assert "period=all" in stats["series_source"]
    assert out["warnings"] == []                       # 口径对上了,不告警
    # 上下文里也已经有了归一化后的年度值
    assert calls[0][nodes._ANNUAL_KEY]["dps"]["latest"] == pytest.approx(1.70)


def test_node03_warns_when_normalized_roe_is_not_annual(monkeypatch):
    """LLM 把单季 ROE 当年度 ROE 填进来(0.0248 ≈ 0.0991/4):必须告警。"""
    _stub(monkeypatch, _dg_params(roe_normalized=0.0248))
    out = asyncio.run(nodes.dividends_growth_node(state_of(NAB_CACHE)))
    assert any("归一化 ROE" in w for w in out["warnings"])


def test_node03_warns_when_dps_by_year_is_not_annual(monkeypatch):
    """dps_by_year 填了尾部那个 ÷4 的坏点(0.425 对 1.70):必须告警。"""
    _stub(monkeypatch, _dg_params(dps_by_year={"2026": 0.425}))
    out = asyncio.run(nodes.dividends_growth_node(state_of(NAB_CACHE)))
    assert any("dps_by_year[2026]" in w for w in out["warnings"])


def test_node03_survives_without_any_data(monkeypatch):
    """缓存为空时不硬崩:统计量留空、没有交叉校验结论(离线端到端也走这条)。"""
    _stub(monkeypatch, _dg_params())
    out = asyncio.run(nodes.dividends_growth_node(state_of({}, code="WFC")))
    assert out["skills"]["dividends_growth"]["results"]["roe_stats_from_data"] == {}
    assert out["errors"] == [] if "errors" in out else True


def test_node03_provenance_separates_data_from_assumption(monkeypatch):
    """参数来源:roe_normalized 对得上数据、dps_by_year 逐年核对、payout_ratio 是自算的。

    报告里"哪些数字来自 MCP、哪些是 LLM 的"不能靠模型自述(它会把自己的假设说成
    "由数据计算得出"),只能由 Python 拿数据核出来。
    """
    _stub(monkeypatch, _dg_params())
    out = asyncio.run(nodes.dividends_growth_node(state_of(NAB_CACHE)))
    rows = {r["field"]: r for r in out["skills"]["dividends_growth"]["provenance"]}

    assert rows["roe_normalized"]["source"] == "MCP roe"
    assert "0.09908" in rows["roe_normalized"]["note"]
    # 这份抓取里没有 sharesBasic(LLM 没取):只能如实标成假设,不能算作数据
    assert rows["shares_outstanding"]["source"] == "LLM 假设"
    assert "payoutratio 单期值不可用" in rows["payout_ratio"]["note"]     # 自算,不是偏离
    assert "2026:1.7↔1.7" in rows["dps_by_year"]["note"]                  # 逐年核对
    assert "⚠️" not in rows["dps_by_year"]["note"]


def test_node03_provenance_flags_each_bad_year(monkeypatch):
    """填了未年化的那一格,来源核对行里要指名道姓是哪个年份。"""
    _stub(monkeypatch, _dg_params(dps_by_year={"2026": 0.425}))
    out = asyncio.run(nodes.dividends_growth_node(state_of(NAB_CACHE)))
    [row] = [r for r in out["skills"]["dividends_growth"]["provenance"]
             if r["field"] == "dps_by_year"]
    assert "2026(0.25 倍,疑似未年化)" in row["note"]


def test_dps_year_row_pairs_with_the_closest_observation_in_that_year(monkeypatch):
    """一年多个观测时按"最接近的"配对,不能拿该年最后一个去比。

    实测 CBA:每年 6/12 月各一个点,模型按财年(6 月结束)取了 4.65(2024-06-30)—— 正确,
    它就是从上下文 by_year 里按财年挑的 —— 而 max(日期) 是 12 月的 4.75,报告于是写成
    "2024:4.65↔4.75",把正确取值显示成偏离。NAB 的 2021 同年 4 个点、跨度 1.2→1.335。
    """
    _stub(monkeypatch, _dg_params(dps_by_year={"2021": 1.34}))
    out = asyncio.run(nodes.dividends_growth_node(state_of(NAB_CACHE)))

    [row] = [r for r in out["skills"]["dividends_growth"]["provenance"]
             if r["field"] == "dps_by_year"]
    assert "2021:1.34↔1.34(2021-06-30)" in row["note"]      # 配对到同值的那个观测
    assert "1.335" not in row["note"] and "⚠️" not in row["note"]
    assert not [w for w in out["warnings"] if "dps_by_year" in w]


def test_dps_year_row_says_so_when_the_value_matches_nothing(monkeypatch):
    """既不是年度值、也不是它的整倍数:得让读者看见,而不是悄悄配对到最近的观测。"""
    _stub(monkeypatch, _dg_params(dps_by_year={"2021": 1.5}))
    out = asyncio.run(nodes.dividends_growth_node(state_of(NAB_CACHE)))

    assert any("dps_by_year[2021] 1.5 与当年观测都对不上" in w for w in out["warnings"])
    [row] = [r for r in out["skills"]["dividends_growth"]["provenance"]
             if r["field"] == "dps_by_year"]
    assert "2021(对不上当年观测,最接近 1.34(2021-06-30))" in row["note"]


def test_source_row_marks_assumptions_when_data_has_no_such_metric():
    """β/Rf/ERP 这类数据源里没有的字段:必须标成 LLM 假设,不能算作数据。"""
    block, _ = annual_block(NAB_CACHE, "ASX", "NAB")
    assumed = nodes._source_row(block, None, "beta", 0.95)
    assert assumed["source"] == "LLM 假设" and "MCP 无对应指标" in assumed["note"]
    missing = nodes._source_row(block, "cet1ratio", "capital_ratio", 0.12)
    assert missing["source"] == "LLM 假设" and "cet1ratio" in missing["note"]
    # 数据里有、值一致的走 MCP;偏离超过 2% 的标成"数据 + LLM 判断"
    same = nodes._source_row(block, "roe", "roe_current", 0.09908)
    assert same["source"] == "MCP roe" and "一致" in same["note"]
    judged = nodes._source_row(block, "roe", "roe_current", 0.135)
    assert judged["source"] == "MCP roe + LLM 判断" and "+36.3%" in judged["note"]
    # 差整倍数(口径没换算)要带警告标记
    scaled = nodes._source_row(block, "dps", "dps_by_year", 0.425)
    assert "⚠️" in scaled["note"] and "0.25" in scaled["note"]


# ---------------------------------------------------------------------------
# 4. 节点 07:σ 来自归一化后的年度序列
# ---------------------------------------------------------------------------

def _rel_params(**over) -> schemas.RelativeParams:
    kw = dict(applicable=True, pb_current=1.4, pe_current=12.0, bvps=20.0, eps=1.70,
              roe_series_metric="roe", roe_window="all", roe_std_dev_assumed=None,
              comparable_notes="同业 PB 约 1.1–1.5", analysis="测试")
    kw.update(over)
    return schemas.RelativeParams(**kw)


def test_node07_sigma_comes_from_the_annualized_series(monkeypatch):
    st = state_of(NAB_CACHE, coefficients={"roe_normalized": 0.0991, "growth": 0.03,
                                           "payout": 0.7, "coe_high": 0.096})
    _stub(monkeypatch, _rel_params())
    out = asyncio.run(nodes.relative_valuation_node(st))

    res = out["skills"]["relative_valuation"]["results"]
    annual = S.annualize(raw_of("nab_all"))["roe"].values
    assert res["roe_std_dev"] == pytest.approx(statistics.stdev(list(annual.values())))
    assert res["roe_series_basis"] == "年度(TTM)"      # 不再是"声明口径 ×n"
    assert res["roe_n"] == len(annual)
    assert "年度(TTM)" in res["roe_std_source"]
    assert out["valuation"]["relative_predicted_pb"] > 0


def test_node07_falls_back_to_assumed_sigma_when_data_is_missing(monkeypatch):
    """序列取不到(指标名对不上)且有降级假设:用假设值并告警,不能整个节点失败。"""
    _stub(monkeypatch, _rel_params(roe_series_metric="returnOnEquity",
                                   roe_std_dev_assumed=0.012))
    st = state_of(NAB_CACHE, coefficients={"roe_normalized": 0.0991, "growth": 0.03,
                                           "payout": 0.7, "coe_high": 0.096})
    out = asyncio.run(nodes.relative_valuation_node(st))
    res = out["skills"]["relative_valuation"]["results"]
    assert res["roe_std_dev"] == pytest.approx(0.012)
    assert "假设值" in res["roe_std_source"]
    assert any("假设值" in w for w in out["warnings"])


def test_node07_warns_when_declared_window_is_not_available(monkeypatch):
    """声明的 2y 窗口没有对应抓取,实际用的是 all:σ 会变,必须说清楚。"""
    _stub(monkeypatch, _rel_params(roe_window="2y"))
    out = asyncio.run(nodes.relative_valuation_node(state_of(NAB_CACHE)))
    assert any("声明的 ROE 窗口 2y" in w for w in out["warnings"])


def test_node07_warns_when_llm_uses_unannualized_bvps(monkeypatch):
    """bvps 用错量级不影响 σ 那条路径,但要在报告里报出来。"""
    expected = S.annualize(raw_of("nab_all"))["bvps"].values
    latest = expected[max(expected)]
    _stub(monkeypatch, _rel_params(bvps=latest / 4))
    out = asyncio.run(nodes.relative_valuation_node(state_of(NAB_CACHE)))
    assert any("bvps" in w and "倍" in w for w in out["warnings"])


def test_node07_avoids_an_unanchorable_fetch_of_the_declared_window(monkeypatch):
    """实测形状:声明 5y,而 5y 那份只有 roe(没有 pb/pe),能锚定的在 all 那份。

    旧选法按声明的窗口取,于是 roe 只能按原值进 σ(σ 于是从"判定过的年度序列"退化成
    "原值序列");现在改用能锚定的那份,窗口不符由告警如实报出。
    """
    roe_raw = raw_of("nab_all")
    roe_points = {d: v for d, v in S.points_of(roe_raw, "roe").items() if v is not None}
    five_y = {"data": {d: {"ratios": {"roe": roe_points[d]}}
                       for d in sorted(roe_points)[-6:]}}          # 只有 roe,锚不住口径
    key = "get_financials|" + json.dumps(
        {"exchange": "ASX", "code": "NAB", "metrics": ["roe"], "period": "5y"},
        sort_keys=True, ensure_ascii=False)
    cache = cache_of("nab_all", "ASX", "NAB", "all",
                     ["dps", "epsBasic", "roe", "netinccmn", "totalCommonEquity", "pb", "pe"],
                     extra={key: five_y})

    st = state_of(cache, coefficients={"roe_normalized": 0.0991, "growth": 0.03,
                                       "payout": 0.7, "coe_high": 0.096})
    _stub(monkeypatch, _rel_params(roe_window="5y"))
    out = asyncio.run(nodes.relative_valuation_node(st))

    res = out["skills"]["relative_valuation"]["results"]
    annual = S.annualize(roe_raw)["roe"].values
    assert res["roe_std_dev"] == pytest.approx(statistics.stdev(list(annual.values())))
    assert "period=all" in res["roe_std_source"]        # 实际用的窗口如实回传
    assert not any("口径未能锚定" in w for w in out["warnings"])
    assert any("声明的 ROE 窗口 5y" in w for w in out["warnings"])   # 换窗口要告警


# ---------------------------------------------------------------------------
# 5. extract_series 仍是原始值(接入改造没有偷偷改变它的语义)
# ---------------------------------------------------------------------------

def test_extract_series_still_returns_the_raw_points():
    points, meta = extract_series(NAB_CACHE, "dps", "ASX", "NAB", period="all")
    assert points["2026-06-30"] == pytest.approx(3.400)     # 数据源的原始坏点
    assert meta["n"] and "period=all" in meta["source"]
    # 而 annual_block 给的是归一化后的 1.70
    block, meta2 = annual_block(NAB_CACHE, "ASX", "NAB", metric="dps")
    assert block["dps"].values["2026-06-30"] == pytest.approx(1.70)
    assert "口径=年度(TTM)" in meta2["source"]
