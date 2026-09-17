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
import re
from typing import Annotated, Any, NotRequired, Sequence, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from pydantic import BaseModel

import app.config as config
import app.formulas as F
import app.series as S
from app import schemas
from app.llm import get_model
from app.mcp_tools import annual_block, make_mcp_tools
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

- **取数有硬额度**:本节点最多 **2 次** `get_financials`(代码强制:第 3 次会被拒绝并提示你
  输出参数;完全相同的参数会被去重、不重复返回数据)。所以**第一次就要一次取全**:本节点
  需要的全部指标 + 参照指标 + 正确的 period。若确实缺一项,第二次补齐;两次之后无论数据
  是否完美,都必须立即输出参数,缺失部分用行业常识假设并在理由字段里标注为"假设"。
- **一份抓取要带齐参照指标**:年度口径由 Python 按**同一次返回里**的恒等式判定
  (pb/pe ÷ roe、roe ÷ (净收益/净资产)),所以取流量指标(dps、epsBasic、netinccmn)时,
  **同一次调用里要带上 roe、netinccmn(净收益)与 totalCommonEquity(净资产)**;取 roe 时
  带上 **pb 与 pe**(roe 自己的年度基准只能由 pb/pe ÷ roe 判定)。拆开取(例如先取
  pe/pb/roe、再取 roe/dps)时,缺参照的那份判不出口径,只能按原值用。
- **不要用不同 period 反复试探**:工具返回的序列**只保留最近 12 个观测**(更早的点被省略,
  返回里有 `_truncated`/`_note` 说明)。换 period 或换窗口只会换一批被省略的点,**不会**多出
  中间年份。需要逐年年度值(如 `dps_by_year`)时,直接用上下文「Python 判定口径后的年度数据」
  里每条指标的 **`by_year`**(逐年观测、已年化),不要从原始序列推算,更不要靠再抓一次去找。
- **上下文里已有的数据不必再抓**:各 skill 结论与年度数据块都是权威输入,不要为了"再确认
  一次"重复调用;重复调用会被去重并回一行提示,白白浪费一步。
- **先确认再取**:不确定指标名时先用 `list_metrics` 看一遍可用指标名,再一次性取数(同一个
  节点内不要重复调 `list_metrics`)。
- **区分"数据"与"假设"**:参数里凡不是来自 MCP 或上下文数据的(MCP 没有该指标,如 β、Rf、
  ERP;或数据缺失),都要在对应的理由/analysis 字段里写明是**假设**及其依据。报告会按
  Python 的核对结果逐字段标注来源,不要把自己的假设说成"由数据计算得出"。
- **口径以 Python 给的为准**:上下文里若出现「Python 判定口径后的年度数据」,那些数值已经
  换算到 12 个月(年度)口径,并附了口径判定依据与被修正的点。**直接采用,不要再对它做
  ×4/×2 换算,也不要靠数值量级自己猜口径**。MCP 返回的原始序列口径混杂(同一序列里可能
  前几年是年报值 ÷4、之后是 12 个月 TTM),靠量级猜会静默错 4 倍。
- **调用失败不要重试超过一次**:工具返回错误说明时,换指标或换 period 重试一次即可,仍失败则
  改用假设并标注。
- **文字字段里的引号**:输出结构化参数时,文字字段(analysis、各类理由/说明)内**不要出现英文
  双引号 `"`**,它会让整个参数对象的 JSON 解析失败、这一节点直接作废(实测 08 综合节点因此
  失败过一次);需要引号时用中文引号「」。
"""

_NO_TOOLS_NOTE = """

---

## 本节点不提供数据工具

本节点的全部输入(各方法结果、各个 skill 的结论与参数、全局参数)已在上文给全,**没有绑定
任何 MCP 工具**,调用它们不会有响应。请直接基于上文数据输出结构化参数;确实缺某个数字时
用合理假设代替,并在相应理由/analysis 字段里标注为假设。所有数据都已由上游节点取过,
重复抓取只会浪费步数与上下文。
"""


class _NodeState(TypedDict):
    """create_react_agent + response_format 要求的最小状态(含三个必需键)。"""

    messages: Annotated[Sequence[Any], add_messages]
    remaining_steps: NotRequired[int]
    structured_response: NotRequired[Any]


# ---------------------------------------------------------------------------
# 通用:运行一个 skill agent
# ---------------------------------------------------------------------------

def _invalid_tool_calls(out: dict) -> list:
    """模型发出、但参数 JSON 不合法而被 langchain 丢进 invalid_tool_calls 的结构化调用。

    这类消息不会进 tool_calls,langchain 的 ToolStrategy 分支因此既不解析也不报错,
    agent 静默返回、structured_response 为 None —— 只能自己检查。
    """
    bad = []
    for m in out.get("messages", None) or []:
        bad.extend(getattr(m, "invalid_tool_calls", None) or [])
    return bad


_BAD_JSON_PROMPT = (
    "\n\n【重要】你上一轮输出的参数对象不是合法 JSON,外部程序解析失败(最常见的原因是文字"
    "字段里用了未转义的英文双引号)。请重新输出**完整**的参数对象:保留上一轮的全部分析与"
    "判断,文字字段内一律用中文引号「」,不要出现英文双引号,换行与反斜杠也要正确转义。"
)

_MISSING_STRUCTURED_PROMPT = (
    "\n\n【重要】你上一轮没有输出结构化参数对象,只在正文里写了分析。请立即输出**一次**"
    "结构化的参数对象,不要在正文里复述分析过程。"
)

_MAX_ECHO = 8000   # 回喂原文的上限(字符),超过就退回只给通用话术


def _parse_error_excerpt(error: str) -> str:
    """从 langchain 的 invalid_tool_call.error 里取出关键那句。

    它把整份 args 也塞进了报错文案('Function X arguments:\\n\\n{...}\\n\\nare not valid JSON.
    Received ...'),整段回喂等于把输入翻倍,所以只截 'are not valid JSON.' 之后的部分。
    """
    tail = error.split("are not valid JSON.", 1)[-1]
    return tail.split("For troubleshooting", 1)[0].strip()


def _repair_message(bad: list):
    """把模型上一轮那份"差一点就对"的参数原样回喂(带出错位置),让它改转义而不是重做分析。

    为什么不直接让它重跑一遍:这次失败是 4KB 中文散文里一处转义手滑(实测落在
    '属"温和增价值"' 这种中文引号习惯上),重跑很可能在同样的措辞上再滑一次;
    看着自己的原文改引号,一次就能改对。没有非法调用、或 args 过大时返回 None,退回通用话术。
    """
    if not bad:
        return None
    tc = bad[0]
    args = tc.get("args") or ""
    if not args or len(args) > _MAX_ECHO:
        return None
    lines = [f"你上一轮输出的参数对象(原样,共 {len(args)} 字符):", args, ""]
    reason = _parse_error_excerpt(tc.get("error") or "")
    if reason:
        lines.append(f"外部程序解析它时的报错:{reason}")
    m = re.search(r"char (\d+)", reason)
    if m:
        i = int(m.group(1))
        lines.append(f"出错位置附近的原文:…{args[max(0, i - 60): i + 60]}…")
    lines.append("请修正上述问题后,重新输出**完整**的参数对象。")
    return HumanMessage(content="\n".join(lines))


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
    # 无工具节点(08 综合、修正模式)不能用"工具使用规范":那份规范在教它什么时候取数,
    # 而它压根没有工具 —— 实测 08 就是这么去调 get_data_period/get_financials 的
    prompt = doc.body + (_TOOL_USAGE_GUIDE if allow_tools else _NO_TOOLS_NOTE) + extra_prompt
    msg = HumanMessage(content=json.dumps(ctx, ensure_ascii=False, indent=2))
    limit = config.MAX_LLM_STEPS * 2 + 2


    async def _invoke(sys_prompt: str, use_tool: bool = True, extra: list | None = None) -> dict:
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
        # metadata 里的 vm_skill 是 llm.out 的归属标记:agent 内部的 langgraph_node
        # 恒为 "model",光靠它分不出这次对话属于哪个 skill。
        out = await agent.ainvoke(
            {"messages": [msg, *(extra or [])]},
            config={"recursion_limit": limit, "metadata": {"vm_skill": skill_id}},
        )
        if out.get("structured_response") is None:
            messages = out.get("messages", [])
            logger.error(
                "%s: structured_response=None, keys=%s, 非法工具调用=%d, last_message=%r",
                skill_id,
                list(out.keys()),
                len(_invalid_tool_calls(out)),
                messages[-1] if messages else None,
            )
        return out

    try:
        out = await _invoke(prompt)
    except GraphRecursionError:
        # 工具调用耗尽步数:降级重试,明确要求立即停止取数并输出参数
        logger.warning("%s: 工具调用达到步数上限,降级重试(禁止再调用工具)", skill_id)
        out = await _invoke(
            prompt + "\n\n【重要】你已用完工具调用额度。现在**不得再调用任何工具**,"
            "立即基于已知信息输出参数;缺失字段用合理假设代替,并在相应理由/analysis 字段中标注为假设。",
            use_tool=False
        )

    params = out.get("structured_response")
    if params is None:
        # 结构化缺失有两种原因,回喂的话术不同(见 _BAD_JSON_PROMPT / _MISSING_STRUCTURED_PROMPT)。
        # langchain 对"调了工具但参数 JSON 不合法"是静默的:invalid_tool_calls 不参与解析,
        # 也不触发它自己的校验重试,agent 直接返回,只能由这里补一次。
        bad = _invalid_tool_calls(out)
        repair = _repair_message(bad)
        logger.warning("%s: 未拿到结构化参数(非法工具调用 %d 个,回喂原文=%s),重试一次",
                       skill_id, len(bad), bool(repair))
        out = await _invoke(
            prompt + (_BAD_JSON_PROMPT if bad else _MISSING_STRUCTURED_PROMPT),
            extra=[repair] if repair else None,
        )
        params = out.get("structured_response")
    if params is None:
        raise RuntimeError(f"{skill_id}: LLM 未返回结构化参数(回喂重试后仍失败)")
    return params, cache


def _base_ctx(state: ValuationState) -> dict:
    ctx: dict[str, Any] = {"exchange": state["exchange"], "code": state["code"]}
    if state.get("overrides"):
        ctx["用户指定的假设(优先采用,但仍须在理由字段中说明来源)"] = state["overrides"]
    if state.get("period_hint"):
        ctx["建议的数据窗口(period 参数)"] = state["period_hint"]
    return ctx


_ANNUAL_KEY = "Python 判定口径后的年度数据(直接采用,不要自己换算)"


def _merged_cache(state: ValuationState, cache: dict) -> dict:
    """本节点新抓的 + 之前节点抓的。口径归一化要跨指标比对,手上多一份判得更准。"""
    return {**(state.get("data_cache") or {}), **cache}


def _annual_ctx(state: ValuationState, metrics: Sequence[str]) -> dict[str, Any]:
    """把**年度口径**数据放进上下文:口径由代码按恒等式判定(app/series.py),不让模型猜。

    MCP 的流量指标不保证同一种口径,同一条序列里能混着两种(实测 ASX:NAB 前几年是年报值
    ÷4、之后是 12 个月 TTM)。LLM 只能靠量级猜,而量级正是被口径污染的那个东西 —— 实测
    NAB 的 dps/eps 就这样被当成单季值、DDM 的整条预测股息序列缩到 1/4。

    这里在节点启动前就把归一化后的最新值与判定依据交给它,模型只读不换算。缓存里还没有
    该公司数据时返回空字典(工具额度仍留着,LLM 照常自己取数),注入失败也不拖垮节点。
    """
    cache = state.get("data_cache") or {}
    if not cache:
        return {}
    try:
        block, _ = annual_block(cache, state["exchange"], state["code"],
                                period=state.get("period_hint"), metrics=tuple(metrics))
        described = S.describe(block, metrics)
    except Exception as e:  # noqa: BLE001 —— 注入是加分项,不该让节点失败
        logger.warning("年度口径数据注入跳过: %s", e)
        return {}
    return {_ANNUAL_KEY: described} if described else {}


def _scale_warnings(
    block: dict[str, Any],
    pairs: Sequence[tuple[str, float | None, str | None]],
) -> list[str]:
    """核对 LLM 填的标量与数据里的年度值:pairs = [(字段名, 值, 指标名)]。

    差着整倍数(2 倍/4 倍)基本只可能是口径没换算。**只告警不覆盖**:模型可能确实在用
    归一化后的调整值(危机年 ROE、一次性重组后的 EPS),代码无从判断哪个才是它的本意,
    但"恰好 4 倍"必须报出来,否则估值会静默缩水到 1/4。
    """
    out: list[str] = []
    for label, value, metric in pairs:
        s = block.get(metric)
        latest = s.latest() if s is not None else None
        if latest is None or not value:
            continue
        factor = S.scale_mismatch(value, latest[1])
        if factor:
            out.append(
                f"{label} {value:g} 恰为数据年度值 {latest[1]:g}({metric}, {latest[0]}, "
                f"口径={s.basis})的 {factor:g} 倍,疑似把未年化的值直接代入 —— 请核对该字段"
            )
    return out


def _source_row(block: dict[str, Any], metric: str | None, label: str,
                value: float | None) -> dict[str, Any]:
    """一个参数值的来源判定:数据里来的、模型调过的、还是纯假设。"""
    if metric is None:
        return {"field": label, "value": value, "source": "LLM 假设",
                "note": "MCP 无对应指标(如 β/Rf/ERP),依据见该字段的理由说明"}
    s = block.get(metric)
    latest = s.latest() if s is not None else None
    if latest is None:
        return {"field": label, "value": value, "source": "LLM 假设",
                "note": f"MCP 没有 {metric} 或本次未取到"}
    date, ref = latest
    factor = S.scale_mismatch(value, ref)
    if factor:
        return {"field": label, "value": value, "source": f"MCP {metric}",
                "note": f"⚠️ 恰为数据年度值 {ref:g}({date})的 {factor:g} 倍,疑似未年化"}
    if value and ref and abs(value - ref) / abs(ref) > 0.02:
        return {"field": label, "value": value, "source": f"MCP {metric} + LLM 判断",
                "note": f"数据年度值 {ref:g}({date}, {s.basis}),此处偏离 "
                        f"{(value - ref) / ref * 100:+.1f}%(归一化/判断),须有理由"}
    return {"field": label, "value": value, "source": f"MCP {metric}",
            "note": f"与数据年度值 {ref:g}({date}, {s.basis})一致"}


def _dps_year_rows(params: Any, dps_series: Any) -> list[tuple[str, float, str, float, float | None]]:
    """逐年核对 LLM 填的 `dps_by_year`:返回 [(年, 值, 对上的观测日, 该观测值, 整倍数标记)]。

    配对要取**该年最接近的观测**,不是"该年最后一个":一年里可能有多个观测(实测 CBA 每年
    6 月与 12 月各一个),模型按财年取了 6 月那个(4.65,正确),而拿 max(日期) 去比就变成
    "4.65↔4.75(2024-12-31)" —— 报告把正确取值写成了偏离。它其实是从上下文 `by_year` 里
    按财年挑的,这正是我们要求它做的事。

    标记位:None = 与某个观测一致;2.0/4.0 等 = 恰好是某个观测的整倍数(疑似未年化);
    0.0 = 当年观测都对不上(那既不是年度值也不是它的整倍数,得让读者自己看)。
    """
    rows: list[tuple[str, float, str, float, float | None]] = []
    for year, value in sorted((getattr(params, "dps_by_year", None) or {}).items()):
        if dps_series is None or value is None:
            continue
        obs = {d: v for d, v in dps_series.values.items() if str(d).startswith(str(year))}
        if not obs:
            continue
        date, ref = min(obs.items(), key=lambda kv: abs(value - kv[1]))
        factor: float | None = None
        if not ref or abs(value - ref) / abs(ref) > 0.02:
            factor = next((f for v in obs.values() if (f := S.scale_mismatch(value, v))), 0.0)
        rows.append((str(year), float(value), date, float(ref), factor))
    return rows


def _data_checks(
    state: ValuationState, cache: dict, *pairs: tuple[str, float | None, str | None]
) -> tuple[list[str], list[dict[str, Any]]]:
    """节点算完后,用数据里的年度值核一遍 LLM 填的关键标量(pairs = (字段名, 值, 指标名))。

    两个用途共用同一批 pairs,因为回答的是同一个问题——这个数字是从数据里来的,还是模型
    自己填的?
      - **告警**:差着整倍数(2/4 倍)基本只可能是口径没换算,必须报出来;
      - **来源核对行**(进报告):哪些与 MCP 年度值一致、哪些是 LLM 的归一化判断、
        哪些 MCP 根本没有(β/Rf/ERP 这类只能是假设)。读报告的人得能分清数据与猜测。
    指标名给 None 表示"数据源没有对应指标",只进来源核对,不产生口径告警。
    """
    provenance: list[dict[str, Any]] = []
    try:
        block, _ = annual_block(_merged_cache(state, cache), state["exchange"], state["code"],
                                period=state.get("period_hint"))
    except Exception as e:  # noqa: BLE001 —— 核对失败不该影响已经算出来的估值
        logger.warning("年度口径核对跳过: %s", e)
        return [], [_source_row({}, m, label, v) for label, v, m in pairs if v is not None]
    for label, value, metric in pairs:
        if value is not None:
            provenance.append(_source_row(block, metric, label, value))
    return _scale_warnings(block, list(pairs)), provenance


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
    # β/Rf/ERP 数据源里一个都没有(MCP 只有公司财务指标,无市场/宏观序列):报告里必须
    # 标成假设,不能让读者以为它们是从数据算出来的 —— skill 02 也要求 beta_source 写清依据
    provenance = [_source_row({}, None, name, getattr(params, name))
                  for name in ("beta", "beta_risk_adjustment", "rf", "erp")]
    skill_out = _write(state, "cost_of_equity", params=params.model_dump(), results=results,
                       analysis=params.analysis, error=None, provenance=provenance)
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
    # 派息率与增长率全靠每股序列与 ROE,口径错了整条估值都错 —— 先把归一化后的
    # 年度值(含判定依据)放进去,LLM 只读不猜。
    ctx.update(_annual_ctx(state, ("dps", "epsBasic", "payoutratio", "roe", "dividendGrowth")))
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

    # Python 从 MCP 缓存算 ROE 统计量,交叉校验 LLM 的归一化 ROE(LLM 不手算)。
    # 口径由 app/series.py 按尺度无关的恒等式判定(Python 算,不问 LLM),序列已年化,
    # 与"必须是年度口径"的归一化 ROE 直接可比。
    roe_stats: dict[str, Any] = {}
    block: dict[str, Any] = {}
    series = None
    try:
        block, meta = annual_block(_merged_cache(state, cache), state["exchange"],
                                   state["code"], metric="roe")
        series = block.get("roe")
        if series is not None and series.values:
            roe_stats = series.stats()
            roe_stats["series_source"] = meta["source"]
            latest_date, latest_roe = series.latest()
            basis = f"数据最新 ROE {F.pct(latest_roe)}({latest_date}, {series.basis})"
            if not series.determined:
                warnings.append(
                    f"ROE 序列口径未能锚定(pb/pe/roe 缺失或吸附不上),{basis} 按原值使用,"
                    "请核对是否已是年度口径"
                )
            if abs(params.roe_normalized - latest_roe) > 0.05:
                warnings.append(
                    f"归一化 ROE {F.pct(params.roe_normalized)} 与{basis} 相差超过 5 个百分点,"
                    "请确认归一化理由是否充分"
                )
    except Exception as e:  # noqa: BLE001
        # 不再静默:这条交叉校验是唯一能拦住"归一化 ROE 用错口径"的闸,跳过必须说出来
        logger.warning("ROE 序列统计跳过(交叉校验缺失): %s", e)
        warnings.append(f"ROE 数据序列统计失败({e}),归一化 ROE 未经数据交叉校验")

    # 逐年股息对一下年度序列:LLM 若把未年化的值填进 dps_by_year,派息率会跟着错
    dps_series = (block or {}).get("dps")
    year_rows = _dps_year_rows(params, dps_series)
    for year, value, date, ref, factor in year_rows:
        if factor:
            warnings.append(
                f"dps_by_year[{year}] {value:g} 恰为数据年度值 {ref:g}({date})的 "
                f"{factor:g} 倍,疑似未年化"
            )
        elif factor == 0.0:
            warnings.append(
                f"dps_by_year[{year}] {value:g} 与当年观测都对不上(最接近 {ref:g}({date})),"
                "既不是年度值也不是它的整倍数,请核对这个数从哪来"
            )

    results = {
        "growth": growth,
        "payout_used": payout,
        "payout_type": payout_type,
        "roe_stats_from_data": roe_stats,
        "formula": "g = ROE × (1 − 派息率)",
    }
    provenance = [
        _source_row(block or {}, "roe", "roe_normalized", params.roe_normalized),
        _source_row(block or {}, "sharesBasic", "shares_outstanding",
                    params.shares_outstanding),
        # payout_ratio 故意不拿 payoutratio 指标核对:该指标单期口径失真,skill 明确要求
        # 用年度 dps/eps 自算,差异是设计而非问题(拿指标对会把正确做法标成偏离)
        {"field": "payout_ratio", "value": params.payout_ratio,
         "source": "LLM 计算(年度 dps/年度 eps)",
         "note": f"口径:{params.payout_basis or '未说明'};数据源 payoutratio 单期值不可用"},
    ]
    if year_rows:
        prov_years = [f"{y}:{v:g}↔{r:g}({d})" for y, v, d, r, _f in year_rows]
        bad = [f"{y}({f:g} 倍,疑似未年化)" if f else
               f"{y}(对不上当年观测,最接近 {r:g}({d}))"
               for y, _v, d, r, f in year_rows if f is not None]
        provenance.append({
            "field": "dps_by_year", "value": None,
            "source": "MCP dps(年度口径)",
            "note": ("逐年核对(取该年最接近的观测):" + "、".join(prov_years))
                    + (f";⚠️ {'、'.join(bad)}" if bad else ""),
        })
    skill_out = _write(state, "dividends_growth", params=params.model_dump(), results=results,
                       analysis=params.analysis, warnings=warnings, error=None,
                       provenance=provenance)
    return {
        **skill_out,
        "coefficients": {
            "growth": growth, "payout": payout, "payout_plain": params.payout_ratio,
            "roe_normalized": params.roe_normalized,
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
    # eps0 是高增长期整条股息序列的起点:口径错 4 倍,估值就错 4 倍
    ctx.update(_annual_ctx(state, ("epsBasic", "dps", "payoutratio", "bvps", "roe")))
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
    wn, provenance = _data_checks(state, cache,
                                  ("eps0", params.eps0, "epsBasic"),
                                  ("payout_high", params.payout_high, None),
                                  ("g_high", params.g_high, None),
                                  ("g_terminal", params.g_terminal, None))
    warnings.extend(wn)
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
                 analysis=params.analysis, warnings=warnings, error=None,
                 provenance=provenance),
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
    ctx.update(_annual_ctx(state, ("netinccmn", "totalCommonEquity", "totalAssets",
                                   "epsBasic", "roe")))
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
    wn, provenance = _data_checks(
        state, cache,
        ("net_income", params.net_income, "netinccmn"),
        ("equity_current", params.equity_current, "totalCommonEquity"),
        ("assets", params.assets, "totalAssets"),
        ("target_capital_ratio", params.target_capital_ratio, None),
        ("assets_growth", params.assets_growth, None),
    )
    warnings.extend(wn)
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
                 analysis=params.analysis, warnings=warnings, error=None,
                 provenance=provenance)
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
    ctx.update(_annual_ctx(state, ("totalCommonEquity", "bvps", "roe", "epsBasic", "dps")))
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
    wn, provenance = _data_checks(
        state, cache,
        ("bv_equity", params.bv_equity, "totalCommonEquity"),
        ("bvps", params.bvps, "bvps"),
        ("roe_current", params.roe_current, "roe"),
        ("roe_terminal", params.roe_terminal, None),
        ("years_high", params.years_high, None),
    )
    warnings = list(repair_notes) + wn
    update: dict[str, Any] = {
        **_write(state, "excess_returns", params=params.model_dump(), results=results,
                 analysis=params.analysis, warnings=warnings, error=None,
                 provenance=provenance),
        "data_cache": cache,
        "warnings": warnings,
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
    ctx.update(_annual_ctx(state, ("roe", "pb", "pe", "bvps", "epsBasic")))
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
    merged_cache = _merged_cache(state, cache)
    stats: dict[str, Any] = {}
    try:
        _require(params, ["bvps", "roe_series_metric"])
        block, meta = annual_block(merged_cache, state["exchange"], state["code"],
                                   period=params.roe_window, metric=params.roe_series_metric)
        # σ 优先由 Python 从真实数据序列计算;取不到才用 LLM 的降级假设(并告警)。
        # 序列已由 app/series.py 归一化成年度口径(口径按尺度无关的恒等式判定,不问 LLM),
        # 与回归里的年度 ROE 同口径 —— 否则 σ 项差 4 倍,预测 PB 会机械性偏高。
        try:
            series = block.get(params.roe_series_metric)
            if series is None or not series.values:
                raise F.FormulaError(
                    f"数据里没有 {params.roe_series_metric} 序列(或无有效观测)")
            if meta["n"] and meta["period"].strip().lower() != (params.roe_window or "").strip().lower():
                warnings.append(
                    f"声明的 ROE 窗口 {params.roe_window} 没能用上(没有对应抓取,或那一份缺"
                    f"参照指标判不出口径),σ 改用 {meta['source']} 的序列——窗口不同 σ 会变,"
                    "请核对"
                )
            if not series.determined:
                warnings.append(
                    f"{params.roe_series_metric} 序列口径未能锚定(pb/pe/roe 缺失或吸附不上),"
                    "σ 按原值计算,请核对是否已是年度口径"
                )
            stats = series.stats()
            if "std" not in stats:
                raise F.FormulaError(f"ROE 序列仅 {stats['n']} 个观测值,无法计算 σ")
            std = stats["std"]
            std_source = (f"数据序列({params.roe_series_metric}, {meta['source']}, "
                          f"{series.basis})")
            series_roe = stats["latest"]
        except (F.FormulaError, TypeError):
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
        "roe_series_basis": stats.get("basis"),
        "roe_series_latest": stats.get("latest"),
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
    wn, provenance = _data_checks(
        state, cache,
        ("bvps", params.bvps, "bvps"), ("eps", params.eps, "epsBasic"),
        ("pb_current", params.pb_current, "pb"), ("pe_current", params.pe_current, "pe"))
    warnings.extend(wn)
    return {
        **_write(state, "relative_valuation", params=params.model_dump(), results=results,
                 analysis=params.analysis, warnings=warnings, error=None,
                 provenance=provenance),
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
        # 综合节点不需要新数据:各方法结果与全部结论都在 ctx 里。实测它仍会去调
        # get_data_period/get_financials 各一次(重复抓一遍 5y),白烧两步与上万 token
        params, cache = await run_skill_agent("08_synthesize", ctx, schemas.SynthesisParams,
                                              allow_tools=False)
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
