"""8 个 skill 节点。

统一模板(用户要求的核心架构):
  1. 每个节点 = 一个独立的 create_react_agent 小 agent,只注入本 skill 的 SKILL.md;
     LLM 自主决定何时调哪个 MCP 工具、调几次,直到它认为数据够了。
  2. LLM 随后输出结构化参数(Pydantic schema),不做任何算术。
  3. Python 用参数调用 app/formulas.py 计算,结果写入 state。
  4. 节点内任何异常都被吞入 state(skills[name]["error"]),不中断整图。

父图 state 不含 messages:每个小 agent 有自己独立的 {messages, remaining_steps,
structured_response} 状态,避免消息串污染与 token 膨胀。
"""

import json
import logging
from typing import Annotated, Any, NotRequired, Sequence, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from pydantic import BaseModel

import app.config as config
import app.formulas as F
from app import schemas
from app.llm import get_model
from app.mcp_tools import extract_series, make_mcp_tools
from app.skill_loader import load_skill
from app.state import METHOD_VALUE_KEYS, ValuationState, normalize_weight_keys

logger = logging.getLogger("value_mind.nodes")

METHOD_SKILL = {
    "ddm": "04_ddm",
    "reg_capital_fcfe": "05_reg_capital_fcfe",
    "excess_returns": "06_excess_returns",
    "relative_valuation": "07_relative_valuation",
}
VALID_METHODS = set(METHOD_SKILL)

_TOOL_USAGE_GUIDE = """

---

## 工具使用规范(通用,覆盖本 skill 中的相关描述)

- **批量取数**:一次 `get_financials` 调用可在 metrics 里传多个指标,尽量一次取全本节点需要的
  所有指标,不要一个指标调用一次——每个节点能承受的工具调用次数有限,零散调用会耗尽额度。
- **先确认再取**:不确定指标名时先用 `list_metrics` 看一遍可用指标名,再一次性取数。
- **数据够用即止**:拿到足以做判断的数据后立即停止调用并输出参数;不要为了"更全"反复试探
  不同的 period 或指标。缺失的数据用行业常识假设并在理由字段中标注,这比继续调用更有价值。
- **调用失败不要重试超过一次**:工具返回错误说明时,换指标或换 period 重试一次即可,仍失败则
  改用假设并标注。
"""

_STRUCTURED_INSTRUCTION = (
    "分析完成后,输出结构化的参数对象。"
    "只输出参数值,不要重复分析过程;绝对不要自己计算任何数值——"
    "所有公式(股权成本、增长率、折现、倍数等)都由外部 Python 代码用你给的参数计算。"
    "百分比一律用小数表示(如 9.6% 写作 0.096),金额用与数据源一致的货币单位。"
    "所有文字字段(analysis、各类理由/说明字段)一律用中文撰写,"
    "公司名与指标名可保留英文原名。"
)


class _NodeState(TypedDict):
    """create_react_agent + response_format 要求的最小状态(含三个必需键)。"""

    messages: Annotated[Sequence[Any], add_messages]
    remaining_steps: NotRequired[int]
    structured_response: NotRequired[Any]


# ---------------------------------------------------------------------------
# 通用:运行一个 skill agent
# ---------------------------------------------------------------------------

async def run_skill_agent(
    skill_id: str,
    ctx: dict,
    schema: type[BaseModel],
    extra_prompt: str = "",
    allow_tools: bool = True,
) -> tuple[BaseModel, dict]:
    """跑一个 skill 节点:LLM 自主工具循环 → 结构化参数。

    allow_tools=False 且无工具时用于"修正模式":数据已取到,只需调整参数。
    返回 (参数对象, 本次调用产生的 MCP 数据缓存)。
    """
    doc = load_skill(skill_id)
    tools, cache = await make_mcp_tools()
    if not allow_tools:
        tools = []

    model = get_model()
    prompt = doc.body + _TOOL_USAGE_GUIDE + extra_prompt
    msg = HumanMessage(content=json.dumps(ctx, ensure_ascii=False, indent=2))
    limit = config.MAX_LLM_STEPS * 2 + 2


    async def _invoke(sys_prompt: str, use_tool: bool = True) -> dict:
        # 工具循环与结构化输出分两步:create_react_agent 的 response_format 内部固定
        # with_structured_output(schema) 不带 method,会撞上本网关不支持的 json_schema
        # (见 llm.structured_model 的说明)。所以结构化那步自己发,method 可降级。
        agent = create_agent(
            model=model,
            tools=tools if use_tool else [],
            response_format=schema,
            system_prompt=sys_prompt,
            name=f"skill_{skill_id}"
        )
        out = await agent.ainvoke({"messages": [msg]}, config={"recursion_limit": limit})
        params = out.get("structured_response")
        if params is None:
            messages = out.get("messages", [])
            logger.error(
                "%s: structured_response=None, keys=%s, last_message=%r",
                skill_id,
                list(out.keys()),
                messages[-1] if messages else None,
            )
            raise ValueError(
                f"{skill_id}: LLM 未返回结构化参数"
            )
        return {"structured_response": params}

    try:
        out = await _invoke(prompt)
    except GraphRecursionError:
        # 工具调用耗尽步数:降级重试,明确要求立即停止取数并输出参数
        logger.warning("%s: 工具调用达到步数上限,降级重试(禁止再调用工具)", skill_id)
        out = await _invoke(
            prompt + "\n\n【重要】你已用完工具调用额度。现在**不得再调用任何工具**,"
            "立即基于已知信息输出参数;缺失字段用合理假设代替,并在相应理由/analysis 字段中标注为假设。"            ,
            use_tool=False
        )

    params = out.get("structured_response")
    if params is None:
        raise RuntimeError(f"{skill_id}: LLM 未返回结构化参数(结构化输出失败)")
    return params, cache


def _base_ctx(state: ValuationState) -> dict:
    ctx: dict[str, Any] = {"exchange": state["exchange"], "code": state["code"]}
    if state.get("overrides"):
        ctx["用户指定的假设(优先采用,但仍须在理由字段中说明来源)"] = state["overrides"]
    if state.get("period_hint"):
        ctx["建议的数据窗口(period 参数)"] = state["period_hint"]
    return ctx


def _skill_brief(state: ValuationState, name: str) -> dict:
    """取某个已运行 skill 的结论摘要,供下游节点参考(不带 messages)。"""
    s = (state.get("skills") or {}).get(name)
    if not s:
        return {}
    out = {k: v for k, v in s.items() if k in ("params", "results", "analysis")}
    if s.get("skipped"):
        out["skipped"] = True
    if s.get("error"):
        out["error"] = s["error"]
    return out


def _write(state: ValuationState, skill: str, **update) -> dict:
    """构造节点返回值:把本节点结论并入 skills[skill],同时合并其他顶层字段。"""
    skills = dict(state.get("skills") or {})
    entry = dict(skills.get(skill) or {})
    entry.update(update)
    return {"skills": {skill: entry}}


# ---------------------------------------------------------------------------
# 01 公司分类
# ---------------------------------------------------------------------------

async def classify_node(state: ValuationState) -> dict:
    try:
        params, cache = await run_skill_agent("01_company_classifier", _base_ctx(state), schemas.ClassifyParams)
    except Exception as e:  # noqa: BLE001
        logger.exception("01 分类失败")
        return {"errors": [f"01 公司分类失败:{e}"], **_write(state, "classify", error=str(e))}

    enabled = [m for m in params.methods_enabled if m in VALID_METHODS] or sorted(VALID_METHODS)
    skill_out = _write(
        state, "classify",
        params=params.model_dump(), analysis=params.analysis, error=None,
    )
    return {
        **skill_out,
        "company_type": params.company_type,
        "company_name": params.company_name or state.get("company_name") or state["code"],
        "methods_enabled": enabled,
        "data_cache": cache,
    }


# ---------------------------------------------------------------------------
# 02 股权成本
# ---------------------------------------------------------------------------

async def cost_of_equity_node(state: ValuationState) -> dict:
    ctx = _base_ctx(state)
    ctx["公司分类结论"] = _skill_brief(state, "classify")
    try:
        params, cache = await run_skill_agent("02_cost_of_equity", ctx, schemas.CostOfEquityParams)
    except Exception as e:  # noqa: BLE001
        logger.exception("02 股权成本失败")
        return {"errors": [f"02 股权成本失败:{e}"], **_write(state, "cost_of_equity", error=str(e))}

    beta_eff = params.beta + params.beta_risk_adjustment
    try:
        coe_high = F.cost_of_equity(params.rf, beta_eff, params.erp)
        coe_terminal = F.cost_of_equity(params.rf, params.terminal_beta, params.erp)
    except Exception as e:  # noqa: BLE001
        return {"errors": [f"02 股权成本公式失败:{e}"], **_write(state, "cost_of_equity", error=str(e))}

    results = {
        "coe_high": coe_high,
        "coe_terminal": coe_terminal,
        "beta_effective": beta_eff,
        "formula": "COE = Rf + β × ERP",
    }
    skill_out = _write(state, "cost_of_equity", params=params.model_dump(), results=results,
                       analysis=params.analysis, error=None)
    return {
        **skill_out,
        "coefficients": {
            "coe_high": coe_high, "coe_terminal": coe_terminal,
            "beta": params.beta, "beta_effective": beta_eff, "terminal_beta": params.terminal_beta,
            "rf": params.rf, "erp": params.erp, "rf_tenor": params.rf_tenor,
        },
        "data_cache": cache,
    }


# ---------------------------------------------------------------------------
# 03 股息与增长
# ---------------------------------------------------------------------------

async def dividends_growth_node(state: ValuationState) -> dict:
    ctx = _base_ctx(state)
    ctx["公司分类结论"] = _skill_brief(state, "classify")
    ctx["已定的股权成本参数"] = {
        k: v for k, v in (state.get("coefficients") or {}).items()
        if k in ("coe_high", "coe_terminal", "rf", "erp")
    }
    try:
        params, cache = await run_skill_agent("03_dividends_growth", ctx, schemas.DividendsGrowthParams)
    except Exception as e:  # noqa: BLE001
        logger.exception("03 股息增长失败")
        return {"errors": [f"03 股息与增长失败:{e}"], **_write(state, "dividends_growth", error=str(e))}

    warnings: list[str] = []
    payout = params.payout_ratio
    payout_type = "普通派息率(股息/收益)"
    if params.include_buybacks and params.buyback_adjusted_payout is not None:
        payout = params.buyback_adjusted_payout
        payout_type = "复合派息率((股息+回购)/净收益,多年平均)"

    try:
        growth = F.growth_from_roe(params.roe_normalized, payout)
    except Exception as e:  # noqa: BLE001
        return {"errors": [f"03 增长公式失败:{e}"], **_write(state, "dividends_growth", error=str(e))}

    # Python 从 MCP 缓存序列计算 ROE 统计量,交叉校验 LLM 的归一化 ROE(LLM 不手算)。
    # 口径由 LLM 声明(数据源对不同公司的约定不同,日期推不出来),Python 按声明年化,
    # 保证与年度口径的归一化 ROE 可比。
    roe_stats: dict[str, Any] = {}
    try:
        series, meta = extract_series(cache, "roe", state["exchange"], state["code"])
        if any(v is not None for v in series.values()):
            roe_stats = F.ratio_series_stats(series, params.roe_series_basis)
            roe_stats["series_source"] = meta["source"]
            conflict = F.basis_conflict(roe_stats)
            if conflict:
                warnings.append(f"ROE 序列{conflict},已按声明口径计算,请核对")
            basis = (f"数据最新 ROE {F.pct(roe_stats['raw_latest'])}"
                     f"({roe_stats['frequency']}口径 ×{roe_stats['periods_per_year']:g} "
                     f"年化为 {F.pct(roe_stats['latest'])})")
            if abs(params.roe_normalized - roe_stats["latest"]) > 0.05:
                warnings.append(
                    f"归一化 ROE {F.pct(params.roe_normalized)} 与{basis} 相差超过 5 个百分点,"
                    "请确认归一化理由是否充分"
                )
    except Exception as e:  # noqa: BLE001
        logger.debug("ROE 序列统计跳过: %s", e)

    results = {
        "growth": growth,
        "payout_used": payout,
        "payout_type": payout_type,
        "roe_stats_from_data": roe_stats,
        "formula": "g = ROE × (1 − 派息率)",
    }
    skill_out = _write(state, "dividends_growth", params=params.model_dump(), results=results,
                       analysis=params.analysis, warnings=warnings, error=None)
    return {
        **skill_out,
        "coefficients": {
            "growth": growth, "payout": payout, "payout_plain": params.payout_ratio,
            "roe_normalized": params.roe_normalized,
            # 口径是 skill 03 的判断产物,存进全局参数供节点 07 的回归复用(σ 要与 ROE 同口径)
            "roe_series_basis": params.roe_series_basis,
            "shares_outstanding": params.shares_outstanding,
            "dividends_reliable": params.dividends_reliable,
        },
        "data_cache": cache,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 通用:方法节点参数校验
# ---------------------------------------------------------------------------

def _require(params: BaseModel, fields: list[str]) -> None:
    missing = [f for f in fields if getattr(params, f, None) is None]
    if missing:
        raise F.FormulaError(f"缺少必要参数: {', '.join(missing)}")


_REPAIR_PROMPT = (
    "\n\n【修正模式】你上一轮给出的参数在代入公式后计算失败(下方给出报错)。"
    "数据已经取到了,**不要再调用任何工具**:请直接调整参数使假设自洽,然后重新输出完整参数对象。"
    "务必让新参数满足公式的硬约束,并在 analysis 中说明你如何修正以及为何新假设更合理。"
)


async def compute_with_repair(
    skill_id: str,
    schema: type[BaseModel],
    ctx: dict,
    params: BaseModel,
    compute,
) -> tuple[Any, BaseModel, list[str]]:
    """跑公式;若因参数不自洽抛 FormulaError,把报错回喂给 LLM 修正一次再算。

    这是必要的:计算发生在 LLM 输出参数之后,若不回喂,LLM 永远看不到
    "COE ≤ g""期末 FCFE 为负"这类不自洽,只能让该估值方法白白失败。

    返回 (计算结果, 实际使用的参数, 附加警告)。
    """
    notes: list[str] = []
    try:
        return compute(params), params, notes
    except F.FormulaError as exc:
        # 注意:except 块结束时 `exc` 这个绑定会被删除,必须先转存到外层变量
        first_err = exc
        logger.warning("%s: 公式报错,进入修正回路: %s", skill_id, first_err)

    repair_ctx = {
        **ctx,
        "你上一轮输出的参数": params.model_dump(),
        "公式计算报错": str(first_err),
        "任务": "修正参数使其自洽(满足报错中给出的硬约束),再输出完整参数对象",
    }
    try:
        params2, _ = await run_skill_agent(
            skill_id, repair_ctx, schema, extra_prompt=_REPAIR_PROMPT, allow_tools=False
        )
        result = compute(params2)
    except Exception as e:  # noqa: BLE001 —— 修正仍失败则保留原始报错
        raise first_err from e
    notes.append(f"参数经一次修正后才通过计算(原始报错:{first_err})")
    return result, params2, notes


# ---------------------------------------------------------------------------
# 04 股息贴现模型
# ---------------------------------------------------------------------------

async def ddm_node(state: ValuationState) -> dict:
    ctx = _base_ctx(state)
    ctx["公司分类结论"] = _skill_brief(state, "classify")
    ctx["已定的全局参数"] = state.get("coefficients") or {}
    ctx["股息与增长分析结论"] = _skill_brief(state, "dividends_growth")
    try:
        params, cache = await run_skill_agent("04_ddm", ctx, schemas.DdmParams)
    except Exception as e:  # noqa: BLE001
        logger.exception("04 DDM 失败")
        return {"errors": [f"04 DDM 失败:{e}"], **_write(state, "ddm", error=str(e))}

    if not params.ddm_applicable:
        return {**_write(state, "ddm", params=params.model_dump(), skipped=True,
                         skip_reason=params.skip_reason or "不适用", analysis=params.analysis)}

    coef = state.get("coefficients") or {}

    def _compute(p: schemas.DdmParams):
        _require(p, ["eps0", "payout_high", "g_high", "stage_years", "g_terminal", "roe_terminal"])
        r = F.ddm_multistage(
            eps0=p.eps0, payout_high=p.payout_high, g_high=p.g_high,
            years_high=int(p.stage_years), coe_high=coef["coe_high"],
            g_terminal=p.g_terminal, roe_terminal=p.roe_terminal,
            coe_terminal=p.coe_terminal,
            roe_high=coef.get("roe_normalized"),
        )
        grid = F.sensitivity_grid(
            lambda g, coe: F.ddm_gordon(dps_last=p.eps0 * p.payout_high, g=g, coe=coe),
            {f"g={F.pct(g,1)}": g for g in _spread(p.g_high, 0.02, 0.01)},
            {f"COE={F.pct(c,1)}": c for c in _spread(coef["coe_high"], 0.02, 0.01)},
        )
        return r, grid

    try:
        (r, grid), params, repair_notes = await compute_with_repair(
            "04_ddm", schemas.DdmParams, ctx, params, _compute)
    except Exception as e:  # noqa: BLE001
        return {"errors": [f"04 DDM 计算失败:{e}"], **_write(state, "ddm", error=str(e))}

    warnings = ([r.consistency_warning] if r.consistency_warning else []) + repair_notes
    results = {
        "value_per_share": r.value_per_share, "dividends": r.dividends,
        "terminal_price": r.terminal_price, "pv_dividends": r.pv_dividends,
        "pv_terminal": r.pv_terminal, "payout_terminal": r.payout_terminal,
        "terminal_share_of_value": r.pv_terminal / r.value_per_share if r.value_per_share else None,
        "sensitivity": grid,
        "formula": "P0 = Σ Dt/(1+COE)^t + P_T/(1+COE)^T",
    }
    return {
        **_write(state, "ddm", params=params.model_dump(), results=results,
                 analysis=params.analysis, warnings=warnings, error=None),
        "valuation": {"ddm_per_share": r.value_per_share},
        "data_cache": cache,
        "warnings": warnings,
    }


def _spread(center: float, span: float, step: float) -> list[float]:
    """以 center 为中心的敏感性取值序列(含中心值)。"""
    out, k = [], 0
    while center - k * step >= center - span:
        out.append(round(center - k * step, 6))
        k += 1
    k = 1
    while center + k * step <= center + span:
        out.append(round(center + k * step, 6))
        k += 1
    return sorted(set(out))


# ---------------------------------------------------------------------------
# 05 监管资本 FCFE
# ---------------------------------------------------------------------------

async def reg_capital_fcfe_node(state: ValuationState) -> dict:
    ctx = _base_ctx(state)
    ctx["公司分类结论"] = _skill_brief(state, "classify")
    ctx["已定的全局参数"] = state.get("coefficients") or {}
    try:
        params, cache = await run_skill_agent("05_reg_capital_fcfe", ctx, schemas.FcfeParams)
    except Exception as e:  # noqa: BLE001
        logger.exception("05 监管资本 FCFE 失败")
        return {"errors": [f"05 监管资本 FCFE 失败:{e}"],
                **_write(state, "reg_capital_fcfe", error=str(e))}

    if not params.fcfe_applicable:
        return {**_write(state, "reg_capital_fcfe", params=params.model_dump(), skipped=True,
                         skip_reason=params.skip_reason or "不适用", analysis=params.analysis)}

    coef = state.get("coefficients") or {}

    def _compute(p: schemas.FcfeParams):
        _require(p, ["assets", "assets_growth", "target_capital_ratio",
                     "equity_current", "net_income", "ni_growth",
                     "projection_years", "g_terminal"])
        return F.fcfe_valuation(
            net_income0=p.net_income, ni_growth=p.ni_growth,
            years_high=int(p.projection_years), assets0=p.assets,
            assets_growth=p.assets_growth,
            target_capital_ratio=p.target_capital_ratio,
            equity0=p.equity_current, coe=coef["coe_high"],
            g_terminal=p.g_terminal, coe_terminal=p.coe_terminal,
            shares=coef.get("shares_outstanding"),
        )

    try:
        r, params, repair_notes = await compute_with_repair(
            "05_reg_capital_fcfe", schemas.FcfeParams, ctx, params, _compute)
    except Exception as e:  # noqa: BLE001
        return {"errors": [f"05 FCFE 计算失败:{e}"],
                **_write(state, "reg_capital_fcfe", error=str(e))}

    warnings: list[str] = list(repair_notes)
    cur = params.current_capital_ratio
    if cur is not None and params.target_capital_ratio > cur:
        warnings.append(
            f"当前资本比率 {F.pct(cur)} 低于目标 {F.pct(params.target_capital_ratio)},"
            "缓冲区不足 → 再投资压力大,估值偏低"
        )
    results = {
        "equity_value": r.equity_value, "value_per_share": r.value_per_share,
        "fcfe": r.fcfe, "equity_path": r.equity_path,
        "capital_ratio_path": r.capital_ratio_path,
        "terminal_value": r.terminal_value, "pv_fcfe": r.pv_fcfe, "pv_terminal": r.pv_terminal,
        "formula": "再投资 = 目标资本比率 × 新资产 − 现有股权;FCFE = 净收入 − 再投资",
    }
    out = _write(state, "reg_capital_fcfe", params=params.model_dump(), results=results,
                 analysis=params.analysis, warnings=warnings, error=None)
    update: dict[str, Any] = {**out, "data_cache": cache, "warnings": warnings}
    if r.value_per_share:
        update["valuation"] = {"fcfe_per_share": r.value_per_share}
    return update


# ---------------------------------------------------------------------------
# 06 超额回报模型
# ---------------------------------------------------------------------------

async def excess_returns_node(state: ValuationState) -> dict:
    ctx = _base_ctx(state)
    ctx["公司分类结论"] = _skill_brief(state, "classify")
    ctx["已定的全局参数"] = state.get("coefficients") or {}
    ctx["股息与增长分析结论"] = _skill_brief(state, "dividends_growth")
    try:
        params, cache = await run_skill_agent("06_excess_returns", ctx, schemas.ExcessParams)
    except Exception as e:  # noqa: BLE001
        logger.exception("06 超额回报失败")
        return {"errors": [f"06 超额回报失败:{e}"], **_write(state, "excess_returns", error=str(e))}

    if not params.applicable:
        return {**_write(state, "excess_returns", params=params.model_dump(), skipped=True,
                         skip_reason=params.skip_reason or "不适用", analysis=params.analysis)}

    coef = state.get("coefficients") or {}

    def _compute(p: schemas.ExcessParams):
        _require(p, ["bv_equity", "roe_current", "roe_terminal", "years_high",
                     "years_fade", "payout"])
        shares = coef.get("shares_outstanding")
        perpet = F.excess_returns_perpetuity(
            bv=p.bv_equity, roe=p.roe_current, coe=coef["coe_high"], shares=shares)
        staged = F.excess_returns_staged(
            bv0=p.bv_equity, roe_high=p.roe_current, roe_terminal=p.roe_terminal,
            coe=coef["coe_high"], years_high=int(p.years_high),
            years_fade=int(p.years_fade), payout=p.payout, shares=shares)
        return perpet, staged

    try:
        (perpet, staged), params, repair_notes = await compute_with_repair(
            "06_excess_returns", schemas.ExcessParams, ctx, params, _compute)
    except Exception as e:  # noqa: BLE001
        return {"errors": [f"06 超额回报计算失败:{e}"], **_write(state, "excess_returns", error=str(e))}

    results = {
        "perpetuity": {"excess_pv": perpet.excess_pv, "equity_value": perpet.equity_value,
                       "value_per_share": perpet.value_per_share},
        "staged": {"excess_pv": staged.excess_pv, "equity_value": staged.equity_value,
                   "value_per_share": staged.value_per_share,
                   "excess_by_year": staged.excess_by_year, "bv_by_year": staged.bv_by_year},
        "bvps_implied": (params.bv_equity / coef["shares_outstanding"]
                         if coef.get("shares_outstanding") else None),
        "formula": "股权价值 = BV + Σ(ROE_t − COE)×BV_{t-1}/(1+COE)^t",
    }
    update: dict[str, Any] = {
        **_write(state, "excess_returns", params=params.model_dump(), results=results,
                 analysis=params.analysis, warnings=repair_notes, error=None),
        "data_cache": cache,
        "warnings": repair_notes,
        "valuation": {},
    }
    if perpet.value_per_share:
        update["valuation"]["excess_per_share_perpetuity"] = perpet.value_per_share
    if staged.value_per_share:
        update["valuation"]["excess_per_share_staged"] = staged.value_per_share
    return update


# ---------------------------------------------------------------------------
# 07 相对估值
# ---------------------------------------------------------------------------

async def relative_valuation_node(state: ValuationState) -> dict:
    ctx = _base_ctx(state)
    ctx["公司分类结论"] = _skill_brief(state, "classify")
    ctx["已定的全局参数"] = state.get("coefficients") or {}
    try:
        params, cache = await run_skill_agent("07_relative_valuation", ctx, schemas.RelativeParams)
    except Exception as e:  # noqa: BLE001
        logger.exception("07 相对估值失败")
        return {"errors": [f"07 相对估值失败:{e}"],
                **_write(state, "relative_valuation", error=str(e))}

    if not params.applicable:
        return {**_write(state, "relative_valuation", params=params.model_dump(), skipped=True,
                         skip_reason=params.skip_reason or "不适用", analysis=params.analysis)}

    coef = state.get("coefficients") or {}
    warnings: list[str] = []
    merged_cache = {**(state.get("data_cache") or {}), **cache}
    try:
        _require(params, ["bvps", "roe_series_metric"])
        series, meta = extract_series(merged_cache, params.roe_series_metric,
                                      state["exchange"], state["code"],
                                      period=params.roe_window)
        # σ 优先由 Python 从真实数据序列计算;取不到才用 LLM 的降级假设(并告警)。
        # 必须与 ROE 同口径:回归里 ROE 是年度值,故 σ 也要年化,否则 σ 项被缩小 3/4,
        # 预测 PB 会机械性偏高(季度序列 ×4)。
        stats: dict[str, Any] = {}
        try:
            if meta["n"] and meta["period"].strip().lower() != (params.roe_window or "").strip().lower():
                warnings.append(
                    f"声明的 ROE 窗口 {params.roe_window} 没有对应抓取,σ 改用"
                    f"{meta['source']} 的序列——若该窗口更长,σ 会偏大/偏小,请核对"
                )
            basis = coef.get("roe_series_basis")
            if not basis:
                warnings.append(
                    "skill 03 未声明 ROE 序列口径,σ 按原始值(未年化)计算;"
                    "若数据是单季口径,σ 会被低估、预测 PB 偏高"
                )
            stats = F.ratio_series_stats(series, basis)

            logger.error(
                "DEBUG ratio_series_stats: type=%s value=%r",
                type(stats).__name__,
                stats,
            )

            conflict = F.basis_conflict(stats)
            if conflict:
                warnings.append(f"ROE 序列{conflict},已按声明口径计算,请核对")
            if "std" not in stats:
                raise F.FormulaError(f"ROE 序列仅 {stats['n']} 个观测值,无法计算 σ")
            std = stats["std"]
            std_source = (f"数据序列({params.roe_series_metric}, {meta['source']}, "
                          f"{stats['frequency']}口径年化)")
            series_roe = stats["latest"]
        except F.FormulaError:
            if params.roe_std_dev_assumed is None:
                raise
            std = params.roe_std_dev_assumed
            std_source = "假设值(数据序列不可得,已降级)"
            warnings.append(
                f"ROE 序列不可得,σ={std} 为假设值而非数据计算值,回归结果可靠性下降"
            )
            series_roe = None
        roe_for_reg = coef.get("roe_normalized") or series_roe
        if roe_for_reg is None:
            raise F.FormulaError("缺少 ROE 输入(归一化 ROE 与数据序列都不可得)")
        predicted_pb = F.pb_regression(roe_for_reg, std)
        fair_price = F.fair_price_from_pb(predicted_pb, params.bvps)
    except Exception as e:  # noqa: BLE001
        return {"errors": [f"07 相对估值计算失败:{e}"],
                **_write(state, "relative_valuation", error=str(e))}

    # 隐含 PE 是次要指标:增长率不低于股权成本时稳定增长公式不成立,
    # 只降级该指标并告警,不影响 PB 回归结论
    implied_pe = None
    if coef.get("growth") is not None and coef.get("payout") is not None:
        try:
            implied_pe = F.implied_pe(coef["payout"], coef["growth"], coef["coe_high"])
        except F.FormulaError as e:
            warnings.append(
                f"隐含 PE 无法计算:{e}——说明 ROE×(1−派息率) 的可持续增长率不低于股权成本,"
                "公司未处于稳定增长状态,稳定增长 PE 公式不适用"
            )

    results = {
        "roe_used_for_regression": roe_for_reg,
        "roe_std_dev": std, "roe_std_source": std_source, "roe_n": stats.get("n", 0),
        "roe_series_frequency": stats.get("frequency"),
        "roe_series_raw_latest": stats.get("raw_latest"),
        "predicted_pb": predicted_pb, "fair_price_per_share": fair_price,
        "pb_current": params.pb_current, "pe_current": params.pe_current,
        "implied_pe": implied_pe,
        "premium_to_fair_pct": (
            (params.pb_current / predicted_pb - 1.0)
            if params.pb_current and predicted_pb else None
        ),
        "formula": "PB = 1.527 + 8.63×ROE − 2.63×σ(ROE);公允价值 = 预测PB × BVPS",
        "regression_note": "R²仅31%,回归结果只作方向性参考,非精确公允价值;书中该例印刷结果与系数矛盾,以公式为准",
    }
    n_obs = stats.get("n", 0)
    if params.roe_series_metric != "roe" or n_obs < 3:
        warnings.append(
            f"回归所用 ROE 序列样本数 {n_obs}(指标 {params.roe_series_metric}),"
            "样本偏少时 σ 不可靠"
        )
    return {
        **_write(state, "relative_valuation", params=params.model_dump(), results=results,
                 analysis=params.analysis, warnings=warnings, error=None),
        "valuation": {"relative_pb_fair_price": fair_price, "relative_predicted_pb": predicted_pb},
        "data_cache": cache,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 08 综合
# ---------------------------------------------------------------------------

def _weighted_mean_of(
    values: dict[str, float], weights: dict[str, float] | None
) -> tuple[float | None, dict[str, float], str | None]:
    """按 LLM 给的权重算各方法结果的加权均值(算术归 Python)。

    返回 (加权均值, 实际参与的结果, 警告)。权重键与结果键全对不上时退化为等权并告警,
    避免 LLM 的权重与最终结论脱节。
    """
    notes: list[str] = []
    weights, renamed = normalize_weight_keys(weights, list(values))
    if renamed:
        notes.append(f"权重键已归一为规范结果键:{', '.join(renamed)}")
    matched = {k: v for k, v in values.items() if k in (weights or {})}
    if not matched:
        if weights:
            notes.append(f"method_weights 的键 {sorted(weights)} 与有效方法结果键 "
                         f"{sorted(values)} 均不匹配,已退化为等权平均")
        matched = values
    note = "; ".join(notes) or None
    if not matched:
        return None, {}, note
    try:
        return round(F.weighted_mean(matched, weights or None), 4), matched, note
    except F.FormulaError as e:
        return None, matched, f"加权均值计算跳过:{e}"


_RANGE_REPAIR_PROMPT = (
    "\n\n【修正模式】你上一轮给出的估值区间与自己的权重不自洽:Python 用你的权重算出的"
    "加权均值落在你给的区间之外,报告会自相矛盾。数据已经取到了,**不要再调用任何工具**:"
    "请二选一后重新输出完整参数对象——(a) 放宽 value_range_low/high 使其覆盖加权均值;"
    "或 (b) 调整 method_weights(和仍为 1)使加权均值落入你认为合理的区间。"
    "选哪个由你的估值判断决定,并在 analysis 中说明理由。"
)


async def _reconcile_range(params: Any, values: dict[str, float], ctx: dict) -> tuple[Any, str | None]:
    """区间自洽修复:加权均值落在 LLM 自给区间外时回喂一次。

    这是一个真实的矛盾,不是笔误——区间说"值在 53~70",权重却算出 71.4。放宽区间还是
    调权重属于估值判断,只能由 LLM 决定,故回喂而不是代码替它选。只修一次,避免拉锯。
    """
    if not values:
        return params, None
    mean, _, _ = _weighted_mean_of(values, params.method_weights)
    if mean is None or params.value_range_low <= mean <= params.value_range_high:
        return params, None

    logger.warning("08 综合:加权均值 %s 落在区间 [%s, %s] 之外,进入修正回路",
                   mean, params.value_range_low, params.value_range_high)
    repair_ctx = {
        **ctx,
        "你上一轮输出的区间与权重": {
            "value_range_low": params.value_range_low,
            "value_range_high": params.value_range_high,
            "method_weights": params.method_weights,
        },
        "Python 用你的权重算出的加权均值": mean,
    }
    try:
        params2, _ = await run_skill_agent(
            "08_synthesize", repair_ctx, schemas.SynthesisParams,
            extra_prompt=_RANGE_REPAIR_PROMPT, allow_tools=False,
        )
    except Exception as e:  # noqa: BLE001 —— 修正失败则保留原参数,由下面的警示兜底
        logger.warning("08 区间修正失败,保留原参数: %s", e)
        return params, None

    mean2, _, _ = _weighted_mean_of(values, params2.method_weights)
    if mean2 is not None and params2.value_range_low <= mean2 <= params2.value_range_high:
        return params2, (
            f"综合区间经一次修正才自洽:原区间 [{params.value_range_low}, "
            f"{params.value_range_high}] 不含加权均值 {mean},修正后 "
            f"[{params2.value_range_low}, {params2.value_range_high}] 含 {mean2}"
        )
    return params, (
        f"区间经一次修正后仍不含加权均值(修正后区间 "
        f"[{params2.value_range_low}, {params2.value_range_high}] vs 均值 {mean2}),已保留原参数"
    )


async def synthesize_node(state: ValuationState) -> dict:
    ctx = _base_ctx(state)
    ctx["公司分类结论"] = _skill_brief(state, "classify")
    ctx["全局参数"] = state.get("coefficients") or {}
    ctx["已计算出的各方法每股价值"] = state.get("valuation") or {}
    ctx["各 skill 结论"] = {
        name: _skill_brief(state, name)
        for name in ("cost_of_equity", "dividends_growth", "ddm", "reg_capital_fcfe",
                     "excess_returns", "relative_valuation")
    }
    try:
        params, cache = await run_skill_agent("08_synthesize", ctx, schemas.SynthesisParams)
    except Exception as e:  # noqa: BLE001
        logger.exception("08 综合失败")
        return {"errors": [f"08 综合失败:{e}"], **_write(state, "synthesize", error=str(e))}

    # Python 侧复核:加权均值、区间覆盖度、隐含 PB(LLM 只给判断,数字由代码校验)
    valuation = state.get("valuation") or {}
    values = {
        k: valuation[k] for k in METHOD_VALUE_KEYS
        if isinstance(valuation.get(k), (int, float)) and valuation.get(k)
    }
    warnings: list[str] = []
    # 先让区间与权重自洽(必要时回喂一次),再算最终加权均值
    params, range_note = await _reconcile_range(params, values, ctx)
    if range_note:
        warnings.append(range_note)

    # 加权均值由 Python 计算:LLM 只提供权重(判断),不做算术
    value_best, matched, mean_note = _weighted_mean_of(values, params.method_weights)
    if mean_note:
        warnings.append(mean_note)

    # 归一后才能判断哪些权重键真的没有对应结果(别名键已在 _weighted_mean_of 里归位)
    weights_norm, _ = normalize_weight_keys(params.method_weights, list(values))
    unused = sorted(set(weights_norm) - set(values))
    if unused:
        warnings.append(f"method_weights 中以下键没有对应的有效方法结果,已忽略:{unused}")

    if value_best is not None and not (
        params.value_range_low <= value_best <= params.value_range_high
    ):
        warnings.append(
            f"Python 算出的加权均值 {value_best} 落在 LLM 给定区间 "
            f"[{params.value_range_low}, {params.value_range_high}] 之外"
        )

    covered = [k for k, v in values.items()
               if params.value_range_low <= v <= params.value_range_high]
    if values and len(covered) < len(values) / 2:
        warnings.append(
            f"估值区间 [{params.value_range_low}, {params.value_range_high}] "
            f"仅覆盖 {len(covered)}/{len(values)} 个方法结果,区间可能过窄"
        )

    bvps = None
    rel = (state.get("skills") or {}).get("relative_valuation") or {}
    if rel.get("params"):
        bvps = rel["params"].get("bvps")
    implied_pb = {}
    if bvps:
        for k, v in values.items():
            try:
                implied_pb[k] = F.implied_pb(v, bvps)
            except F.FormulaError:
                pass

    results = {
        "weighted_mean": value_best,
        "weights_used": weights_norm,   # 归一后的权重,与加权均值实际用的口径一致
        "values_used": matched,
        "methods_inside_range": covered,
        "implied_pb_by_method": implied_pb,
        "formula": "加权均值 = Σ(权重 × 各方法每股价值) / Σ权重",
    }
    valuation = {
        "value_range_low": params.value_range_low,
        "value_range_high": params.value_range_high,
        "verdict": params.verdict,
        "conviction": params.conviction,
    }
    if value_best is not None:
        valuation["value_best"] = value_best
    return {
        **_write(state, "synthesize", params=params.model_dump(), results=results,
                 analysis=params.analysis, warnings=warnings, error=None),
        "valuation": valuation,
        "data_cache": cache,
        "warnings": warnings,
    }
