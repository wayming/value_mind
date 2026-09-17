"""第九章确定性估值公式库。

设计原则(用户要求):
- 公式不是 skill:所有算术集中在本模块,纯 Python、零 IO、零 LLM 依赖,可离线单测。
- LLM 只负责提供不可量化的判断参数(β、ROE 归一化、阶段假设等),不做任何计算。
- 百分比一律以小数表示(9.6% -> 0.096);金额单位与入参口径一致(本币/元)。

书中数字勘误(已用公式复核,单测断言公式值):
- 超额回报例:书中印"超额回报现值约 582.2 亿美元、股权价值 1058.5 亿美元"为
  翻译/印刷错误,按公式应为 19.4B / 67.03B;每股 28.38 美元与公式吻合。
- PB 回归例:书中代入汤普金斯数据印"1.95",按书中系数实算为 3.2082,
  结论方向相反;以公式为准,且该回归 R² 仅 31%,仅作方向性参考。
"""

from dataclasses import dataclass, field
from statistics import stdev
from typing import Callable, Sequence


class FormulaError(ValueError):
    """公式不可计算(如 COE <= g、分母非正),携带中文 reason 供报告直接引用。"""


# ---------------------------------------------------------------------------
# 股权成本
# ---------------------------------------------------------------------------

def cost_of_equity(rf: float, beta: float, erp: float) -> float:
    """股权成本 = Rf + β × ERP。书例: cost_of_equity(0.036, 1.2, 0.05) == 0.096"""
    return rf + beta * erp


# ---------------------------------------------------------------------------
# 增长与派息
# ---------------------------------------------------------------------------

def growth_from_roe(roe: float, payout_ratio: float) -> float:
    """预期收益增长率 = ROE × (1 − 派息率)。
    书例: growth_from_roe(0.1351, 0.5463) ≈ 0.0613
    """
    return roe * (1.0 - payout_ratio)


def terminal_payout_ratio(g_terminal: float, roe_terminal: float) -> float:
    """稳定期派息率 = 1 − g/ROE(增长、派息、股权回报三者自洽的约束)。
    书例: terminal_payout_ratio(0.03, 0.086) ≈ 0.6512
    """
    if roe_terminal <= 0:
        raise FormulaError(f"稳定期 ROE 必须为正(当前 {roe_terminal})")
    if g_terminal >= roe_terminal:
        raise FormulaError(f"稳定期增长率 {g_terminal} 不能 >= 稳定期 ROE {roe_terminal}")
    return 1.0 - g_terminal / roe_terminal


def composite_payout(dividends: float, buybacks: float, net_income: float) -> float:
    """复合派息率 = (股息 + 回购) / 净收益。回购年度波动大,应使用多年平均。"""
    if net_income <= 0:
        raise FormulaError(f"净收益必须为正(当前 {net_income})")
    return (dividends + buybacks) / net_income


# ---------------------------------------------------------------------------
# 股息贴现模型(DDM)
# ---------------------------------------------------------------------------

@dataclass
class DdmResult:
    value_per_share: float = 0.0
    dividends: list[float] = field(default_factory=list)
    terminal_price: float = 0.0
    pv_dividends: float = 0.0
    pv_terminal: float = 0.0
    payout_terminal: float = 0.0
    consistency_warning: str | None = None


def ddm_gordon(dps_last: float, g: float, coe: float) -> float:
    """稳定增长(戈登)DDM:P0 = DPS0 × (1+g) / (COE − g)。"""
    if coe <= g:
        raise FormulaError(f"COE({coe}) 必须大于 g({g}),否则戈登模型发散")
    return dps_last * (1.0 + g) / (coe - g)


def ddm_multistage(
    eps0: float,
    payout_high: float,
    g_high: float,
    years_high: int,
    coe_high: float,
    g_terminal: float,
    roe_terminal: float,
    coe_terminal: float | None = None,
    roe_high: float | None = None,
) -> DdmResult:
    """多阶段 DDM(第九章富国银行式)。

    高增长期 years_high 年:EPS 按 g_high 增长,派息率 payout_high 不变,
    股息按 COE_high 贴现;期末进入稳定期:派息率 = 1 − g_terminal/roe_terminal,
    β 趋 1(coe_terminal,缺省沿用 coe_high),终值 = D_{T+1}/(coe_terminal − g_terminal)。

    若提供 roe_high,校验 |g_high − roe_high×(1−payout_high)| > 0.5% 时给出
    一致性警告(增长、派息、股权回报必须自洽,见第九章)。
    """
    if years_high < 1:
        raise FormulaError(f"高增长期年限必须 >= 1(当前 {years_high})")
    if coe_high <= g_high:
        raise FormulaError(
            f"高增长期 COE({coe_high:.2%}) 必须大于 g({g_high:.2%}),否则股息增长快于贴现率、"
            "模型发散。这通常意味着假设不自洽:高增长应配较高的 β 从而推高 COE(第九章原则),"
            "或应下调高增长期 g(此时须同时说明派息率/ROE 假设的相应调整),二者必居其一"
        )
    coe_t = coe_high if coe_terminal is None else coe_terminal
    if coe_t <= g_terminal:
        raise FormulaError(
            f"稳定期 COE({coe_t:.2%}) 必须大于 g_terminal({g_terminal:.2%})。"
            "请检查稳定期假设:永续增长率不应超过无风险利率量级,稳定期 β 趋 1 会进一步推高 COE"
        )
    payout_t = terminal_payout_ratio(g_terminal, roe_terminal)

    eps = eps0
    dividends: list[float] = []
    pv_div = 0.0
    for _ in range(1, years_high + 1):
        eps *= 1.0 + g_high
        d = eps * payout_high
        dividends.append(d)
        pv_div += d / (1.0 + coe_high) ** _

    d_next = eps * (1.0 + g_terminal) * payout_t
    terminal_price = d_next / (coe_t - g_terminal)
    pv_term = terminal_price / (1.0 + coe_high) ** years_high

    warning = None
    if roe_high is not None:
        g_implied = growth_from_roe(roe_high, payout_high)
        if abs(g_high - g_implied) > 0.005:
            warning = (
                f"增长率不自洽:g_high={g_high:.4f} 与 ROE×(1−派息率)="
                f"{g_implied:.4f} 偏差超过 0.5%,请检查输入"
            )

    return DdmResult(
        value_per_share=pv_div + pv_term,
        dividends=dividends,
        terminal_price=terminal_price,
        pv_dividends=pv_div,
        pv_terminal=pv_term,
        payout_terminal=payout_t,
        consistency_warning=warning,
    )


def pv_of_cashflows(cashflows: Sequence[float], rates: float | Sequence[float]) -> float:
    """按逐年贴现率(或统一贴现率)折现一组现金流。"""
    if isinstance(rates, (int, float)):
        rates = [rates] * len(cashflows)
    if len(rates) != len(cashflows):
        raise FormulaError("贴现率数量与现金流数量不一致")
    total = 0.0
    for t, (cf, r) in enumerate(zip(cashflows, rates), start=1):
        total += cf / (1.0 + r) ** t
    return total


def sensitivity_grid(
    fn: Callable[..., float],
    axis1: dict[str, float],
    axis2: dict[str, float],
    **fixed,
) -> list[dict]:
    """二维敏感性表:fn 形如 fn(v1, v2, **fixed);格子不可算(FormulaError)填 None。"""
    rows = []
    for name1, v1 in axis1.items():
        row: dict = {"param": name1}
        for name2, v2 in axis2.items():
            try:
                row[name2] = round(fn(v1, v2, **fixed), 4)
            except FormulaError:
                row[name2] = None
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 监管资本 FCFE(第九章"股权现金流模型")
# ---------------------------------------------------------------------------

def equity_reinvestment_reg_capital(
    assets: float,
    assets_growth: float,
    target_capital_ratio: float,
    equity_current: float,
) -> float:
    """监管资本再投资 = 目标资本比率 × 资产×(1+资产增速) − 现有股权。
    书例: equity_reinvestment_reg_capital(100e6, 0.10, 0.07, 6e6) == 1_700_000
    """
    new_assets = assets * (1.0 + assets_growth)
    return target_capital_ratio * new_assets - equity_current


def fcfe_from_net_income(net_income: float, reinvestment: float) -> float:
    """股权自由现金流 = 净收入 − 监管资本再投资。
    书例: fcfe_from_net_income(5e6, 1.7e6) == 3_300_000
    """
    return net_income - reinvestment


@dataclass
class FcfeResult:
    fcfe: list[float] = field(default_factory=list)
    equity_path: list[float] = field(default_factory=list)
    capital_ratio_path: list[float] = field(default_factory=list)
    terminal_value: float = 0.0
    pv_fcfe: float = 0.0
    pv_terminal: float = 0.0
    equity_value: float = 0.0
    value_per_share: float = 0.0


def fcfe_valuation(
    net_income0: float,
    ni_growth: float,
    years_high: int,
    assets0: float,
    assets_growth: float,
    target_capital_ratio: float,
    equity0: float,
    coe: float,
    g_terminal: float,
    coe_terminal: float | None = None,
    shares: float | None = None,
) -> FcfeResult:
    """监管资本约束下的 FCFE 折现估值。

    逐年:t 年 NI_t = NI_{t-1}×(1+ni_growth);
    再投资 reinvest_t = 目标资本比率 × assets_{t-1}×(1+assets_growth) − equity_{t-1};
    FCFE_t = NI_t − reinvest_t(即"潜在股息",第九章);
    equity_t = equity_{t-1} + NI_t − FCFE_t(留存收益增厚股权账面价值);
    终值 = FCFE_{T+1}/(coe_terminal − g_terminal),FCFE_{T+1} = FCFE_T×(1+g_terminal)。
    """
    if years_high < 1:
        raise FormulaError(f"预测年限必须 >= 1(当前 {years_high})")
    coe_t = coe if coe_terminal is None else coe_terminal
    if coe_t <= g_terminal:
        raise FormulaError(
            f"稳定期 COE({coe_t:.2%}) 必须大于 g_terminal({g_terminal:.2%}),"
            "否则终值无意义。稳定期永续增长率不应超过无风险利率量级"
        )

    ni = net_income0
    assets = assets0
    equity = equity0
    fcfe_list: list[float] = []
    equity_path = [equity0]
    ratio_path = [equity0 / assets0 if assets0 else 0.0]

    for _ in range(years_high):
        ni = ni * (1.0 + ni_growth)
        assets = assets * (1.0 + assets_growth)
        reinvest = target_capital_ratio * assets - equity
        fcf = ni - reinvest
        fcfe_list.append(fcf)
        equity = equity + reinvest  # equity_{t-1} + NI − FCFE = equity_{t-1} + reinvest
        equity_path.append(equity)
        ratio_path.append(equity / assets if assets else 0.0)

    pv_fcfe = pv_of_cashflows(fcfe_list, coe)
    if fcfe_list[-1] <= 0:
        raise FormulaError(
            f"期末 FCFE({fcfe_list[-1]}) 非正,无法计算稳定期终值,"
            "请检查资产增速与目标资本比率假设(缓冲区缺口过大时会出现)"
        )
    fcf_next = fcfe_list[-1] * (1.0 + g_terminal)
    terminal_value = fcf_next / (coe_t - g_terminal)
    pv_term = terminal_value / (1.0 + coe) ** years_high

    result = FcfeResult(
        fcfe=fcfe_list,
        equity_path=equity_path,
        capital_ratio_path=ratio_path,
        terminal_value=terminal_value,
        pv_fcfe=pv_fcfe,
        pv_terminal=pv_term,
        equity_value=pv_fcfe + pv_term,
    )
    if shares:
        result.value_per_share = result.equity_value / shares
    return result


# ---------------------------------------------------------------------------
# 超额回报模型
# ---------------------------------------------------------------------------

def excess_return_per_period(bv: float, roe: float, coe: float) -> float:
    """单期超额股权回报 = (ROE − COE) × 投入股权资本。"""
    return (roe - coe) * bv


@dataclass
class ExcessResult:
    excess_pv: float = 0.0
    equity_value: float = 0.0
    excess_by_year: list[float] = field(default_factory=list)
    bv_by_year: list[float] = field(default_factory=list)
    value_per_share: float = 0.0


def excess_returns_perpetuity(
    bv: float, roe: float, coe: float, shares: float | None = None
) -> ExcessResult:
    """超额回报永续模型(第九章富国银行例;ROE 永不回归 = 乐观上界):
    超额回报 PV = BV × (ROE − COE) / COE;股权价值 = BV + 超额回报 PV。
    书例: excess_returns_perpetuity(47.63e9, 0.1351, 0.096) -> excess_pv≈19.40e9,
    equity_value≈67.03e9;÷2.362e9 股 → 28.38(书中每股 28.38 正确)。
    ROE < COE 为合法结果(股权市场价值低于账面)。
    """
    if coe <= 0:
        raise FormulaError(f"COE 必须为正(当前 {coe})")
    excess_pv = (roe - coe) * bv / coe
    result = ExcessResult(excess_pv=excess_pv, equity_value=bv + excess_pv)
    if shares:
        result.value_per_share = result.equity_value / shares
    return result


def _linear_fade(roe_high: float, roe_terminal: float, years_high: int, years_fade: int):
    """生成 ROE 路径:years_high 年维持 roe_high,随后 years_fade 年线性衰减到 roe_terminal。"""
    path: list[float] = []
    path.extend([roe_high] * years_high)
    if years_fade > 0:
        for i in range(1, years_fade + 1):
            path.append(roe_high + (roe_terminal - roe_high) * i / (years_fade + 1))
    return path


def excess_returns_staged(
    bv0: float,
    roe_high: float,
    roe_terminal: float,
    coe: float,
    years_high: int,
    years_fade: int,
    payout: float,
    shares: float | None = None,
) -> ExcessResult:
    """有限期超额回报 + 均值回归模型(中央值):
    ROE 前 years_high 年维持高位,随后 years_fade 年线性衰减到 roe_terminal;
    bv_t = bv_{t-1} × (1 + roe_t × (1 − payout));excess_t = (roe_t − coe) × bv_{t-1};
    逐年按 COE 贴现,衰减结束后按 roe_terminal 的永续超额回报收尾
    (roe_terminal == coe 时自然归零)。
    """
    if coe <= 0:
        raise FormulaError(f"COE 必须为正(当前 {coe})")
    if years_high < 0 or years_fade < 0:
        raise FormulaError("年限不能为负")
    if not 0.0 <= payout < 1.0:
        raise FormulaError(f"派息率必须在 [0,1) 区间(当前 {payout})")

    roe_path = _linear_fade(roe_high, roe_terminal, years_high, years_fade)
    bv = bv0
    excess_pv = 0.0
    excess_by_year: list[float] = []
    bv_by_year = [bv0]
    for t, roe_t in enumerate(roe_path, start=1):
        excess_t = (roe_t - coe) * bv
        excess_by_year.append(excess_t)
        excess_pv += excess_t / (1.0 + coe) ** t
        bv = bv * (1.0 + roe_t * (1.0 - payout))
        bv_by_year.append(bv)

    # 终期:以 roe_terminal 的永续超额回报收尾
    terminal_excess = (roe_terminal - coe) * bv / coe if coe else 0.0
    excess_pv += terminal_excess / (1.0 + coe) ** (years_high + years_fade)

    result = ExcessResult(
        excess_pv=excess_pv,
        equity_value=bv0 + excess_pv,
        excess_by_year=excess_by_year,
        bv_by_year=bv_by_year,
    )
    if shares:
        result.value_per_share = result.equity_value / shares
    return result


# ---------------------------------------------------------------------------
# 相对估值
# ---------------------------------------------------------------------------

def pb_regression(roe: float, roe_std_dev: float) -> float:
    """第九章 PB 回归:PB = 1.527 + 8.63×ROE − 2.63×σ(ROE),R²=31%。
    书例: pb_regression(0.2798, 0.2789) == 3.2082
    (书中印 1.95 与自身系数矛盾,以公式为准;R² 仅 31%,结果作方向性 band)。
    """
    return 1.527 + 8.63 * roe - 2.63 * roe_std_dev


def fair_price_from_pb(predicted_pb: float, bvps: float) -> float:
    """回归公允价值 = 预测 PB × BVPS。"""
    return predicted_pb * bvps


def implied_pe(payout: float, g: float, coe: float) -> float:
    """基本面 PE(稳定 DDM 推导)= 派息率 × (1+g) / (COE − g)。
    对应第九章定性结论:增长高、派息高、股权成本低 → PE 更高。
    """
    if coe <= g:
        raise FormulaError(f"COE({coe}) 必须大于 g({g})")
    return payout * (1.0 + g) / (coe - g)


def implied_pb(value_per_share: float, bvps: float) -> float:
    """由内在估值每股价值反推的隐含 PB,与市场 PB 对照。"""
    if bvps <= 0:
        raise FormulaError(f"BVPS 必须为正(当前 {bvps})")
    return value_per_share / bvps


# ---------------------------------------------------------------------------
# 统计与工具(供相对估值等从 MCP 缓存原始序列计算,LLM 不手算)
# ---------------------------------------------------------------------------

def sample_std(values: Sequence[float | None]) -> float:
    """样本标准差(n−1 口径,对应书中 σ)。少于 2 个观测值抛 FormulaError。"""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        raise FormulaError("样本标准差至少需要 2 个观测值")
    return stdev(vals)


def infer_periods_per_year(dates: Sequence[str]) -> tuple[float, str]:
    """从日期序列的中位间隔推断"每年几期",返回 (倍数, 频度标签)。

    数据源对 ROE/ROA 这类比率指标常按单季(或单月)口径给值,而 COE、归一化 ROE、
    回归里的 ROE 都是年度口径。不年化就直接比较或代入,会把 σ 低估到 1/4,
    机械抬高回归预测 PB,并让"归一化 ROE 与数据相差过大"的警示每次都误报。
    """
    parsed = sorted({_parse_date(d) for d in dates})
    if len(parsed) < 2:
        return 1.0, "单点(无法判断频度,按年度处理)"
    gaps = [(b - a).days for a, b in zip(parsed, parsed[1:])]
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    if median_gap <= 45:
        return 12.0, "月度"
    if median_gap <= 135:
        return 4.0, "单季"
    if median_gap <= 250:
        return 2.0, "半年度"
    return 1.0, "年度"


def _parse_date(s: str):
    from datetime import date

    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError as e:
        raise FormulaError(f"无法解析日期 {s!r}(需要 YYYY-MM-DD)") from e


# 口径标签 → 每年期数
PERIODS_PER_YEAR = {"月度": 12.0, "单季": 4.0, "半年度": 2.0, "年度": 1.0}


def ratio_series_stats(
    series: dict[str, float | None], basis: str | float | None = None
) -> dict:
    """比率型指标序列(ROE/ROA 等)的**年化后**统计。

    年化口径统一在这里做,调用方拿到的 latest/mean/min/max/std 保证同一单位,
    不会再出现"ROE 年化、σ 没年化"的混用。

    basis 是口径本身(标签或倍数),**必须由调用方给出**:同一份数据源对不同公司
    的约定不一样——实测 ASX NAB 的 roe 是滚动年度值(0.099–0.124),NYSE WFC 的是
    单季值(0.014–0.038),而两者的日期间隔都是 91 天。日期只能说明"多久采一次",
    说明不了"每个点代表多长期间的比率",所以这里绝不用日期间隔去推倍数。
    basis 缺省按年度(×1)处理并在 basis_source 里说明,由调用方决定是否告警。
    """
    pts = sorted((d, v) for d, v in series.items() if v is not None)
    if not pts:
        raise FormulaError("序列为空,无法计算统计量")
    if isinstance(basis, str):
        if basis not in PERIODS_PER_YEAR:
            raise FormulaError(
                f"未知的序列口径 {basis!r},应为 {'/'.join(PERIODS_PER_YEAR)} 或一个倍数")
        factor, label = PERIODS_PER_YEAR[basis], basis
    elif basis:
        factor, label = float(basis), f"×{float(basis):g}"
    else:
        factor, label = 1.0, "年度(未声明口径,按原值处理)"
    xs = [v for _, v in pts]
    out = {
        "latest": xs[-1] * factor,
        "mean": (sum(xs) / len(xs)) * factor,
        "min": min(xs) * factor,
        "max": max(xs) * factor,
        "raw_latest": xs[-1],
        "n": len(xs),
        "latest_date": pts[-1][0],
        "periods_per_year": factor,
        "frequency": label,
        "basis_declared": basis is not None,
        # 日期间隔只作交叉核对:年度序列(间隔 > 300 天)不可能是单季值
        "date_implied": infer_periods_per_year([d for d, _ in pts])[1],
    }
    if len(xs) >= 2:
        out["std"] = stdev(xs) * factor
    return out


def basis_conflict(stats: dict) -> str | None:
    """口径与日期间隔明显矛盾时返回说明,否则 None。

    只判一种硬矛盾:采样间隔接近一年,却声称每个点是单季/月度值。
    反向不判——季度采样滚动年度值是常见做法(实测 NAB 就是这样)。

    只在调用方**自己声明**了口径时才有意义。主链路已不声明口径:序列由 app/series.py
    按尺度无关的恒等式判定并年化,`AnnualSeries.stats()` 传的是 factor=1.0。
    """
    implied = stats.get("date_implied") or ""
    if "年度" in implied and stats.get("periods_per_year", 1.0) > 2:
        return (f"序列采样间隔约一年(日期间隔判断为{implied}),"
                f"但口径声明为{stats.get('frequency')},两者矛盾")
    return None


def latest_nonnull(series: dict[str, float | None]) -> tuple[str, float] | None:
    """取日期序列中最新非空值(键为日期字符串)。"""
    for k in sorted(series.keys(), reverse=True):
        v = series.get(k)
        if v is not None:
            return k, v
    return None


def weighted_mean(values: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """加权均值;weights 缺省为等权。空输入抛 FormulaError。"""
    if not values:
        raise FormulaError("加权均值至少需要 1 个值")
    total_w = 0.0
    acc = 0.0
    for k, v in values.items():
        w = (weights or {}).get(k, 1.0 / len(values))
        acc += w * v
        total_w += w
    if total_w <= 0:
        raise FormulaError("权重之和必须为正")
    return acc / total_w


def pct(v: float, digits: int = 2) -> str:
    """小数转百分比字符串:0.096 -> '9.60%'。"""
    return f"{v * 100:.{digits}f}%"
