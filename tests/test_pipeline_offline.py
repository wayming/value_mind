"""离线端到端:用第九章富国银行(WFC)数据打桩 LLM 的结构化输出,验证
"LLM 参数 → Python 计算 → state → 报告" 全链路,不需要 LLM 与 MCP。

打桩的是 run_skill_agent(即 LLM 的输出),公式、状态流转、报告生成都是真实代码。
"""

import copy

import pytest

import app.nodes as nodes
from app import schemas


WELLS = {
    "classify": schemas.ClassifyParams(
        company_type="bank", business_lines=["零售银行", "商业银行"],
        company_name="富国银行",
        methods_enabled=["ddm", "reg_capital_fcfe", "excess_returns", "relative_valuation"],
        methods_disabled_reasons={}, risk_vs_peers="average:大型货币中心银行,受监管强",
        capital_buffer_note="危机中资本承压", analysis="以存贷利差为主,归为银行",
    ),
    "cost_of_equity": schemas.CostOfEquityParams(
        beta=1.2, beta_source="大型货币中心商业银行平均 β(书中)", beta_risk_adjustment=0.0,
        terminal_beta=1.0, rf=0.036, rf_tenor="美国10年期国债(书中)", erp=0.05,
        erp_rationale="成熟市场约 5%(书中)", analysis="书中富国银行例",
    ),
    "dividends_growth": schemas.DividendsGrowthParams(
        dividends_reliable=True, payout_ratio=0.5463, payout_basis="过去12个月股息/收益(书中)",
        roe_normalized=0.1351, roe_normalization_reason="监管资本比率提高约30%(书中)",
        shares_outstanding=2.362e9, analysis="书中富国银行例",
    ),
    "ddm": schemas.DdmParams(
        ddm_applicable=True, eps0=2.18, payout_high=0.5463, g_high=0.0613, stage_years=5,
        g_terminal=0.03, roe_terminal=0.086, coe_terminal=0.086, analysis="书中富国银行例",
    ),
    "reg_capital_fcfe": schemas.FcfeParams(
        fcfe_applicable=True, assets=1e11, assets_growth=0.08, target_capital_ratio=0.07,
        current_capital_ratio=0.065, equity_current=6.5e9, net_income=5e9, ni_growth=0.06,
        projection_years=5, g_terminal=0.03, coe_terminal=0.086,
        buffer_gap_note="缓冲区略紧", analysis="书中方法例",
    ),
    "excess_returns": schemas.ExcessParams(
        applicable=True, bv_equity=47.63e9, bvps=None, roe_current=0.1351, roe_terminal=0.096,
        years_high=5, years_fade=10, payout=0.5463,
        mean_reversion_rationale="竞争与监管使超额回报衰减", analysis="书中富国银行例",
    ),
    "relative": schemas.RelativeParams(
        applicable=True, pb_current=1.4, pe_current=10.0, bvps=20.16, eps=2.18,
        roe_series_metric="roe", roe_window="5y", roe_std_dev_assumed=0.012,
        comparable_notes="同业 PB 约 1.1-1.5",
        loan_loss_provision_note="危机中准备金高,收益被压低", diversification_note="业务较集中",
        analysis="书中富国银行例",
    ),
    "synthesize": schemas.SynthesisParams(
        value_range_low=25.0, value_range_high=65.0,
        method_weights={"ddm_per_share": 0.3, "fcfe_per_share": 0.2,
                        "excess_per_share_perpetuity": 0.25,
                        "excess_per_share_staged": 0.15, "relative_pb_fair_price": 0.1},
        method_reconciliation_note="各法接近", drivers={"equity_risk": "与同业相当"},
        investment_notes={"capital_buffer": "危机中不足"}, verdict="合理", conviction="medium",
        key_risks=["监管资本要求提高"], analysis="综合各法",
    ),
}


@pytest.fixture
def stubbed(monkeypatch):
    """把 run_skill_agent 换成返回预置参数(模拟 LLM 输出),公式仍走真实代码。

    每个测试拿到 WELLS 的深拷贝,避免测试间互相污染(有的测试会改桩数据)。
    """
    fixture_data = copy.deepcopy(WELLS)

    def fake(skill_id, ctx, schema):
        key = {
            "01_company_classifier": "classify",
            "02_cost_of_equity": "cost_of_equity",
            "03_dividends_growth": "dividends_growth",
            "04_ddm": "ddm",
            "05_reg_capital_fcfe": "reg_capital_fcfe",
            "06_excess_returns": "excess_returns",
            "07_relative_valuation": "relative",
            "08_synthesize": "synthesize",
        }[skill_id]
        return fixture_data[key], {}

    monkeypatch.setattr(nodes, "run_skill_agent", fake)
    return fixture_data


def _initial_state(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.REPORT_DIR", str(tmp_path))
    return {
        "exchange": "NYSE", "code": "WFC", "overrides": {}, "period_hint": "5y",
        "methods_skip": [], "skills": {}, "coefficients": {}, "data_cache": {},
        "valuation": {}, "warnings": [], "errors": [],
    }


def test_graph_runs_end_to_end_and_matches_book(stubbed, tmp_path, monkeypatch):
    from app.graph import build_graph

    state = _initial_state(tmp_path, monkeypatch)
    final = build_graph().invoke(state, config={"recursion_limit": 60})

    coef = final["coefficients"]
    # 书中数字:COE 9.6% / 稳定期 8.6% / g 6.13%
    assert coef["coe_high"] == pytest.approx(0.096)
    assert coef["coe_terminal"] == pytest.approx(0.086)
    assert coef["growth"] == pytest.approx(0.0613, rel=1e-3)

    val = final["valuation"]
    # DDM ≈ 27.74(书中)
    assert abs(val["ddm_per_share"] - 27.74) < 0.5
    # 超额回报永续 = 每股 28.38(书中)
    assert abs(val["excess_per_share_perpetuity"] - 28.38) < 0.1
    # 均值回归版应低于永续版(上界)
    assert val["excess_per_share_staged"] < val["excess_per_share_perpetuity"]
    # 监管资本 FCFE 与相对估值均产出
    assert val["fcfe_per_share"] > 0
    assert val["relative_pb_fair_price"] > 0

    assert final["status"] == "ok"
    assert not final["errors"]

    report = (tmp_path / f"NYSE_WFC_{''}").parent / final["final_report_path"].split("/")[-1]
    text = open(final["final_report_path"], encoding="utf-8").read()
    assert "富国银行" in text
    assert "估值结论" in text
    assert "三个估值驱动器" in text
    assert "不构成投资建议" in text


def test_skipped_method_is_excluded(stubbed, tmp_path, monkeypatch):
    from app.graph import build_graph

    state = _initial_state(tmp_path, monkeypatch)
    state["methods_skip"] = ["ddm"]
    final = build_graph().invoke(state, config={"recursion_limit": 60})
    assert "ddm_per_share" not in (final.get("valuation") or {})
    assert final["valuation"]["excess_per_share_perpetuity"] > 0


def test_method_error_does_not_break_graph(stubbed, tmp_path, monkeypatch):
    """DDM 参数不自洽(COE <= g)时,该法报错但整图继续,其余方法仍出结果。"""
    from app.graph import build_graph

    stubbed["ddm"] = schemas.DdmParams(
        ddm_applicable=True, eps0=2.18, payout_high=0.5463, g_high=0.15, stage_years=5,
        g_terminal=0.03, roe_terminal=0.086, analysis="故意不自洽",
    )
    state = _initial_state(tmp_path, monkeypatch)
    final = build_graph().invoke(state, config={"recursion_limit": 60})
    assert any("04 DDM" in e for e in final["errors"])
    assert "ddm_per_share" not in (final.get("valuation") or {})
    assert final["valuation"]["excess_per_share_perpetuity"] > 0


def test_weighted_mean_computed_by_python(stubbed, tmp_path, monkeypatch):
    """加权均值由 Python 算(LLM 只给权重):用规范键名时应等于手算结果。"""
    from app.graph import build_graph

    state = _initial_state(tmp_path, monkeypatch)
    final = build_graph().invoke(state, config={"recursion_limit": 60})
    val = final["valuation"]

    w = stubbed["synthesize"].method_weights
    expected = sum(w[k] * val[k] for k in w) / sum(w.values())
    assert val["value_best"] == pytest.approx(expected, rel=1e-3)
    assert not any("method_weights" in x for x in final["warnings"])


def test_method_name_weight_keys_are_normalized(stubbed, tmp_path, monkeypatch):
    """LLM 用方法名当权重键(reg_capital_fcfe)时必须归一到结果键。

    否则该方法的结果静默拿不到权重——实测 WFC 跑出过 FCFE 值 84.11 完全没参与加权。
    """
    from app.graph import build_graph

    stubbed["synthesize"] = stubbed["synthesize"].model_copy(
        update={"method_weights": {"ddm": 0.5, "reg_capital_fcfe": 0.25,
                                   "relative_valuation": 0.25},
                "value_range_low": 25.0, "value_range_high": 95.0})
    state = _initial_state(tmp_path, monkeypatch)
    final = build_graph().invoke(state, config={"recursion_limit": 60})
    val = final["valuation"]

    assert not any("没有对应的有效方法结果" in x for x in final["warnings"]), \
        "别名键归一后不应再报「该键无对应结果」"
    assert any("权重键已归一" in x for x in final["warnings"]), "归一动作应留痕"
    assert val["value_best"] == pytest.approx(
        0.5 * val["ddm_per_share"] + 0.25 * val["fcfe_per_share"]
        + 0.25 * val["relative_pb_fair_price"], rel=1e-3)


def test_unmappable_weight_keys_degrade_to_equal_weight(stubbed, tmp_path, monkeypatch):
    """归一无能为力的键(凭空造的)仍应告警并退化为等权,而不是算出错的值。"""
    from app.graph import build_graph

    stubbed["synthesize"] = stubbed["synthesize"].model_copy(
        update={"method_weights": {"foo": 0.5, "bar": 0.5}, "value_range_low": 25.0,
                "value_range_high": 65.0})
    state = _initial_state(tmp_path, monkeypatch)
    final = build_graph().invoke(state, config={"recursion_limit": 60})
    assert any("均不匹配" in x for x in final["warnings"])

    val = final["valuation"]
    vals = [val[k] for k in ("ddm_per_share", "fcfe_per_share", "excess_per_share_perpetuity",
                             "excess_per_share_staged", "relative_pb_fair_price")]
    assert val["value_best"] == pytest.approx(sum(vals) / len(vals), rel=1e-3)


def test_formula_error_triggers_repair_once(tmp_path, monkeypatch):
    """参数不自洽导致公式报错时,应把报错回喂给 LLM 修正一次,而非直接放弃该方法。

    这是必要的结构补丁:计算发生在 LLM 输出参数之后,不回喂则 LLM 永远看不到
    "期末 FCFE 为负"这类不自洽。
    """
    import app.formulas as F
    from app.graph import build_graph

    calls = {"fcfe": 0, "repair_ctx": None, "allow_tools": None}
    fixture_data = copy.deepcopy(WELLS)
    key_map = {"01_company_classifier": "classify", "02_cost_of_equity": "cost_of_equity",
               "03_dividends_growth": "dividends_growth", "04_ddm": "ddm",
               "05_reg_capital_fcfe": "reg_capital_fcfe", "06_excess_returns": "excess_returns",
               "07_relative_valuation": "relative", "08_synthesize": "synthesize"}

    def fake(skill_id, ctx, schema, extra_prompt="", allow_tools=True):
        if skill_id == "05_reg_capital_fcfe":
            calls["fcfe"] += 1
            if calls["fcfe"] == 1:
                # 第一轮:资产扩张 90% + 目标资本比率 50% → 再投资远超净利润 → 期末 FCFE 为负
                bad = fixture_data["reg_capital_fcfe"].model_copy(
                    update={"assets_growth": 0.90, "target_capital_ratio": 0.50})
                return bad, {}
            # 修正轮:记录上下文,返回合理参数
            calls["repair_ctx"] = ctx
            calls["allow_tools"] = allow_tools
            return fixture_data["reg_capital_fcfe"], {}
        return fixture_data[key_map[skill_id]], {}

    monkeypatch.setattr(nodes, "run_skill_agent", fake)
    monkeypatch.setattr("app.config.REPORT_DIR", str(tmp_path))
    state = _initial_state(tmp_path, monkeypatch)
    final = build_graph().invoke(state, config={"recursion_limit": 60})

    assert calls["fcfe"] == 2, "应恰好触发一次修正调用"
    assert calls["allow_tools"] is False, "修正轮不应再调用工具"
    assert "公式计算报错" in calls["repair_ctx"], "修正上下文应包含原始报错"
    assert "期末 FCFE" in calls["repair_ctx"]["公式计算报错"]
    # 修正后该法成功产出,不再报错
    assert final["valuation"]["fcfe_per_share"] > 0
    assert not any("05 FCFE" in e for e in final["errors"])
    assert any("修正后才通过计算" in w for w in final["warnings"])


def test_fcfe_book_example_formula(stubbed, tmp_path, monkeypatch):
    """监管资本再投资 = 目标比率×新资产 − 现有股权(书中 170 万例)已在
    test_formulas 覆盖;这里验证节点把 shares 正确用于每股换算。"""
    import app.formulas as F

    r = F.fcfe_valuation(net_income0=5e6, ni_growth=0.0, years_high=1, assets0=100e6,
                         assets_growth=0.10, target_capital_ratio=0.07, equity0=6e6,
                         coe=0.096, g_terminal=0.03, shares=1e6)
    assert r.fcfe[0] == pytest.approx(3.3e6)      # 第一年 FCFE = 500万 − 170万
    assert r.value_per_share > 0


def test_incoherent_range_triggers_one_repair(tmp_path, monkeypatch):
    """加权均值落在 LLM 自给区间外时,应回喂一次让它自己决定放宽区间还是调权重。

    这是真实的矛盾而非笔误:区间说"值在 25~26",权重却算出 30 多。放宽区间还是调权重
    属估值判断,只能由 LLM 决定,代码不能替它选,但必须逼它二者自洽。
    """
    import copy as _copy

    from app.graph import build_graph

    calls = {"synth": 0, "allow_tools": None, "ctx": None}
    fixture_data = _copy.deepcopy(WELLS)
    key_map = {"01_company_classifier": "classify", "02_cost_of_equity": "cost_of_equity",
               "03_dividends_growth": "dividends_growth", "04_ddm": "ddm",
               "05_reg_capital_fcfe": "reg_capital_fcfe", "06_excess_returns": "excess_returns",
               "07_relative_valuation": "relative", "08_synthesize": "synthesize"}

    def fake(skill_id, ctx, schema, extra_prompt="", allow_tools=True):
        if skill_id == "08_synthesize":
            calls["synth"] += 1
            if calls["synth"] == 1:
                # 区间过窄,必然不含加权均值 → 应触发修正
                bad = fixture_data["synthesize"].model_copy(
                    update={"value_range_low": 25.0, "value_range_high": 26.0})
                return bad, {}
            calls["allow_tools"] = allow_tools
            calls["ctx"] = ctx
            return fixture_data["synthesize"], {}
        return fixture_data[key_map[skill_id]], {}

    monkeypatch.setattr(nodes, "run_skill_agent", fake)
    monkeypatch.setattr("app.config.REPORT_DIR", str(tmp_path))
    state = _initial_state(tmp_path, monkeypatch)
    final = build_graph().invoke(state, config={"recursion_limit": 60})

    assert calls["synth"] == 2, "应恰好触发一次区间修正"
    assert calls["allow_tools"] is False, "修正轮不应再调用工具"
    assert "Python 用你的权重算出的加权均值" in calls["ctx"], "修正上下文应含 Python 算出的均值"

    val = final["valuation"]
    assert val["value_range_low"] <= val["value_best"] <= val["value_range_high"], \
        "修正后区间必须覆盖加权均值,否则报告自相矛盾"
    assert any("修正" in w for w in final["warnings"])
    assert not any("之外" in w for w in final["warnings"]), "修正成功后不应再有区间不符告警"
