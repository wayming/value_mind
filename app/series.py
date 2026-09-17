"""MCP 原始序列 → **年度口径**序列的归一化。

**问题**(2026-09 实测)。MCP 的流量指标不保证同一种口径,而且同一家公司的同一条
序列里能混着两种:

    ASX:NAB  netinccmn  2021-06-30 = 1.578e9   ← 单季(= 年度值 ÷ 4)
                        2021-09-30 = 6.351e9   ← 12 个月 TTM,恰好 ×4
    ASX:NAB  dps 2022-09-30 = 1.51 正好等于 NAB FY22 两次分红 73c + 78c(自证是年报数)
    ASX:NAB  roe = 0.10–0.12(年度口径)   NYSE:WFC roe = 0.029–0.031(单季,×4 = 12%)
    尾部零星坏点:dps 2025-06-30 = 0.425(邻域的 1/4)、2026-06-30 = 3.400(×2)

分界线落在 FY21 结束那天,且 `epsBasic × sharesOutTotalCommon ≈ netinccmn` 在全部
41 个日期上成立(偏离 >10% 的为 0 个)——口径是**整块翻转**,不是随机噪声。

混着用的后果是静默的:DDM 的 eps0 取到单季值,整条预测股息序列缩水到 1/4;
PB 回归的 σ 差 4 倍,预测 PB 反向失真。两者都不会报错。

**做法**:不猜、也不去问 LLM,用**尺度无关的恒等式**把口径算出来。

    roe    ← pb / pe      两个比率都由数据源按同一收盘价和它自己的年度每股数算好,
                          pb/pe = EPS/BVPS = ROE,其绝对值就是 ROE 的年度基准
                          (实测 (pb/pe)/roe:NAB 0.99、CBA 1.00、WFC 3.5→4)
    流量   ← roe ÷ equity   roe 与 netinc/equity 的比 = 该点流量的口径倍数
                          (实测 NAB 2021-06-30 = 4.17 → 4,其余点 = 1.0)
    eps    ← 同报表流量      与 netinccmn 同基准,再用 eps×股数 ≈ 净收益复核
    dps    ← 同报表流量      用 payoutratio(无量纲)复核派息率是否落在合理区间

口径是**整块翻转**的(见上),所以按公司、按日期定一个"流量倍数",income/cash-flow
全组共用;roe 有自己的基准;存量指标(balance-sheet)是时点余额,一律不缩放;与期间
无关的无量纲比率(pe/pb/payoutratio 等)也不缩放。

**边界**:锚定不出来时**不静默按 ×1** —— `determined=False` + 告警,由调用方决定
要不要让这个数字进模型。服务器将来若保证统一 TTM,这一层退化为「断言通过」的空操作。

归一化后的序列交给 `app/formulas.py` 现成的统计函数(`AnnualSeries.stats()` 走
`ratio_series_stats`),不另起一套算术。

**用法**:`annualize(raw)` 整块归一化(要跨指标比对恒等式,单看一条序列判不出口径),
节点侧经 `mcp_tools.annual_block()` 拿缓存里的那一份,再用 `describe()` 把口径与依据
交给 LLM;消费端 `scale_mismatch()` 兜"LLM 仍填了未年化值"这最后一种情况。
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass, field
from datetime import date as _date_cls
from typing import Any, Callable, Iterable

import app.formulas as F

# 定基准的容差。WFC 实测 (pb/pe)/roe = 3.47–3.82(真值 4,数据源自身的取整与口径差
# 带来约 14% 抖动),窗口得比噪声宽;又必须比相邻候选窄(log 2 ≈ 0.69,±0.22 时
# 1/4 不会跳到 1/2)。
_LOOSE = 0.22

# 单点校正的容差:只认"**恰好**是邻域的 2 倍/4 倍"。实测坏点是精确的 0.250 与 2.000,
# 而真实的序列漂移(NAB dps 0.15→0.30、WFC 季度股息 0.20→0.40 这类逐年爬升)落在
# 0.6–0.8 区间,±0.10 的窗口把它们全部排除在外。
_TIGHT = 0.10

_CANDIDATES = (0.25, 0.5, 1.0, 2.0, 4.0)
_RECENT = 8        # 定基准只看最近 N 个点:模型消费的是近期数据,口径也以近期为准
_LOCAL = 2         # 邻域离群检测的半窗

# describe() 里 by_year 的规模上限:最近几年、每年几个观测。要够 LLM 填 schema 里的
# 按年字段(dps_by_year 一般 5 年),又要保证注入块有界(节点每步都带着它)。
_BY_YEAR_YEARS = 6
_BY_YEAR_POINTS = 4

# "这个点与基准同口径"的分界:同块内的年代噪声 ±20%,换口径至少差 2 倍(log 0.69),
# 取中间值,两边都留足余量。
_BLOCK_SEP = 0.40


def _date(s: str) -> _date_cls:
    try:
        return _date_cls.fromisoformat(str(s)[:10])
    except ValueError:
        return _date_cls(1900, 1, 1)

_STOCK, _FLOW, _RATIO = "stock", "flow", "ratio"

# metric_sources 的报表归属 → 类型。balance-sheet 全是时点余额;income/cash-flow 是流量。
_KIND_BY_STATEMENT = {
    "balance-sheet": _STOCK,
    "income-statement": _FLOW,
    "cash-flow-statement": _FLOW,
    "ratios": _RATIO,
}

# 名称优先于报表归属:同一个名字在不同公司被塞进不同组(NAB 把 payoutratio 放
# income-statement,WFC 放 ratios),而 pipeline 真正消费的那几个必须判准。
# 表是**有界**的——只列 skills 会请求的指标;其余走报表归属,判不出类型的不缩放并告警。
_BY_NAME = {
    "roe": _RATIO, "roa": _RATIO,
    "pb": _RATIO, "pe": _RATIO, "peForward": _RATIO, "pegRatio": _RATIO,
    "payoutratio": _RATIO, "dividendyield": _RATIO, "earningsyield": _RATIO,
    "buybackyield": _RATIO, "ptbvRatio": _RATIO,
    "dividendGrowth": _RATIO, "epsGrowth": _RATIO, "netIncomeGrowth": _RATIO,
    "revenueGrowth": _RATIO, "fcfGrowth": _RATIO, "ocfGrowth": _RATIO,
    "marketCapGrowth": _RATIO, "profitMargin": _RATIO, "fcfMargin": _RATIO,
    "taxrate": _RATIO, "debtequity": _RATIO, "netdebtequity": _RATIO,
    "debtfcf": _RATIO, "netdebtfcf": _RATIO, "totalreturn": _RATIO,
    "ps": _RATIO, "pfcf": _RATIO, "pocf": _RATIO, "lastCloseRatios": _RATIO,
    "marketcap": _STOCK, "bvps": _STOCK, "tangibleBookValuePerShare": _STOCK,
    "netcashpershare": _STOCK, "sharesOutTotalCommon": _STOCK,
    "sharesOutFilingDate": _STOCK, "sharesBasic": _STOCK, "sharesDiluted": _STOCK,
    "netinc": _FLOW, "netinccmn": _FLOW, "netIncomeCF": _FLOW,
    "epsBasic": _FLOW, "epsdil": _FLOW, "dps": _FLOW,
}

# 与期间长度无关的比率:分子分母同口径(流量÷流量),或本身就是按收盘价算的市价比率。
_SCALE_FREE_RATIOS = {
    "payoutratio", "pe", "peForward", "pegRatio", "pb", "ptbvRatio",
    "dividendyield", "earningsyield", "buybackyield", "profitMargin", "fcfMargin",
    "debtequity", "netdebtequity", "totalreturn", "lastCloseRatios", "taxrate",
    "dividendGrowth", "epsGrowth", "netIncomeGrowth", "revenueGrowth",
    "fcfGrowth", "ocfGrowth", "marketCapGrowth",
}


# ---------------------------------------------------------------------------
# 原始序列
# ---------------------------------------------------------------------------

def metrics_of(raw: Any) -> list[str]:
    """原始结果里出现过的指标名(按首次出现顺序)。"""
    out: list[str] = []
    for stmts in ((raw or {}).get("data") or {}).values():
        for metrics in (stmts or {}).values():
            if isinstance(metrics, dict):
                for m in metrics:
                    if m not in out:
                        out.append(m)
    return out


def points_of(raw: Any, metric: str) -> dict[str, float | None]:
    """某家公司某个指标的 日期 → 值(升序)。同一日期出现在多个报表组时后者覆盖前者。"""
    if not isinstance(raw, dict):
        raise TypeError(
            f"data_cache 的值应为 get_financials 的 structuredContent(dict),"
            f"实际是 {type(raw).__name__} —— 工具包装层没有还原结构化结果")
    points: dict[str, float | None] = {}
    for date, stmts in ((raw or {}).get("data") or {}).items():
        for metrics in (stmts or {}).values():
            if isinstance(metrics, dict) and metric in metrics:
                points[date] = metrics[metric]
    return dict(sorted(points.items()))


def classify(metric: str, statement: str | None) -> str:
    """指标类型:stock(时点余额)/ flow(期间流量)/ ratio(比率)。"""
    return _BY_NAME.get(metric) or _KIND_BY_STATEMENT.get(str(statement or ""), _RATIO)


def _snap(ratio: float | None, tol: float = _LOOSE) -> float | None:
    """把观测比值吸附到 {1/4, 1/2, 1, 2, 4};落在窗口外返回 None(不动它)。"""
    if ratio is None or not math.isfinite(ratio) or ratio <= 0:
        return None
    best = min(_CANDIDATES, key=lambda c: abs(math.log(ratio / c)))
    return best if abs(math.log(ratio / best)) <= tol else None


# ---------------------------------------------------------------------------
# 归一化结果
# ---------------------------------------------------------------------------

@dataclass
class AnnualSeries:
    """一个指标归一化到**年度口径**后的序列与判定依据。

    factor = 该点代表几分之一年的倒数:1 = 已是 12 个月口径,4 = 一个季度。
    values 是乘上 factor 的结果(存量与无量纲比率的 factor 恒为 1)。
    """

    metric: str
    kind: str
    factor: float = 1.0
    determined: bool = False
    values: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float | None] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    adjusted: dict[str, str] = field(default_factory=dict)   # 日期 → 修正说明
    warnings: list[str] = field(default_factory=list)

    @property
    def needs_normalization(self) -> bool:
        """口径是否真的需要判定 —— 存量(时点余额)与无量纲比率没有口径问题,`determined`
        对它们恒为真。挑"信息量最大的一份抓取"时按这个计数,否则一份全是资产负债表的
        抓取会靠一堆白送的 determined 赢过真正带流量指标的那份。"""
        return self.kind != _STOCK and self.metric not in _SCALE_FREE_RATIOS

    @property
    def basis(self) -> str:
        if self.kind == _STOCK:
            return "时点余额(不年化)"
        if self.kind == _RATIO and self.metric in _SCALE_FREE_RATIOS:
            return "无量纲比率(不年化)"
        return {1.0: "年度(TTM)", 2.0: "半年度(已×2)", 4.0: "单季(已×4)",
                12.0: "月度(已×12)"}.get(self.factor, f"×{self.factor:g}")

    def stats(self) -> dict:
        """复用 formulas.ratio_series_stats:序列已年化,故 basis 传 1.0。"""
        out = F.ratio_series_stats(self.values, 1.0)
        out["basis"] = self.basis
        out["frequency"] = self.basis
        out["factor"] = self.factor
        out["determined"] = self.determined
        out["evidence"] = self.evidence
        out["adjusted"] = self.adjusted
        return out

    def latest(self) -> tuple[str, float] | None:
        if not self.values:
            return None
        d = max(self.values)
        return d, self.values[d]

    def sampling_per_year(self) -> float:
        """每年的采样点数(按日期间隔推断,至少 1)。"""
        return max(1.0, F.infer_periods_per_year(list(self.values))[0])

    def growth(self, lag_quarters: int | None = None) -> float | None:
        """年度口径序列上的同比增速,取各年比值的中位数。

        比数据源的 `dividendGrowth` 可靠:实测它尾部被坏点污染成 +300%
        (2025-09-30 = 1.70/0.425 − 1,分母正是那个 ÷4 的坏点)。

        滞后点数默认按采样频度算(每期 91 天 → 4 个点 = 1 年),这样服务器哪天改成
        一年一个年度点时同比不会变成"四年比"。
        """
        lag = lag_quarters if lag_quarters is not None else max(1, round(self.sampling_per_year()))
        return growth_from_anchors(self.values, lag)


def growth_from_anchors(values: dict[str, float], lag_quarters: int = 4) -> float | None:
    """隔 lag_quarters 个采样点做同比,取比值中位数。

    中位数而非最新一年:单年会被一次性因素带偏(NAB 2020 年股息腰斩)。
    """
    dates = sorted(values)
    if len(dates) <= lag_quarters:
        return None
    ratios = [values[dates[i]] / values[dates[i - lag_quarters]]
              for i in range(lag_quarters, len(dates))
              if values[dates[i - lag_quarters]] > 0 and values[dates[i]] > 0]
    return statistics.median(ratios) - 1.0 if ratios else None


# ---------------------------------------------------------------------------
# 口径判定
# ---------------------------------------------------------------------------

@dataclass
class _Resolution:
    """全公司共用的判定结果。口径整块翻转 → 一个流量倍数就够。"""

    f_roe: float = 1.0
    f_flow: float = 1.0
    flow_votes: dict[str, float] = field(default_factory=dict)  # 日期 → 该点流量倍数
    evidence: dict[str, list[str]] = field(default_factory=dict)
    anchored: bool = False                                      # pb/pe 锚是否吸附成功


def _series(points: dict[str, dict[str, float | None]], metric: str) -> dict[str, float]:
    return {d: v for d, v in (points.get(metric) or {}).items() if v is not None}


def _ratio_of(a: dict[str, float], b: dict[str, float],
              fn: Callable[[float, float], float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in sorted(set(a) & set(b)):
        try:
            r = fn(a[d], b[d])
        except ZeroDivisionError:
            continue
        if r and math.isfinite(r) and r > 0:
            out[d] = r
    return out


def _median_recent(votes: dict[str, float], n: int = _RECENT) -> tuple[float | None, list[str]]:
    if not votes:
        return None, []
    recent = sorted(votes)[-n:]
    return statistics.median([votes[d] for d in recent]), recent


def _resolve(points: dict[str, dict[str, float | None]]) -> _Resolution:
    """定基准:pb/pe 锚 roe,再用净资产把流量倍数比出来。"""
    res = _Resolution()
    roe, pb, pe = _series(points, "roe"), _series(points, "pb"), _series(points, "pe")
    netinc, equity = _series(points, "netinccmn"), _series(points, "totalCommonEquity")

    # (1) roe 的年度基准:(pb/pe)/roe。两个比率都由数据源按同一收盘价与它自己的
    #     年度每股数算好 —— 这是唯一不依赖"另一个指标判得对不对"的绝对锚。
    anchor = _ratio_of(_ratio_of(pb, pe, lambda p, e: p / e), roe, lambda r, x: r / x)
    med, recent = _median_recent(anchor)
    snapped = _snap(med)
    if snapped:
        res.f_roe, res.anchored = snapped, True
        res.evidence["roe"] = [
            f"pb/pe ÷ roe 最近 {len(recent)} 点中位数 {med:.3f}"
            f"({recent[0]}..{recent[-1]})→ roe 为{'年度' if snapped == 1 else ' ×%g' % snapped}"
            f"口径,与存量/无量纲指标同基准"]
    elif med is not None:
        res.evidence["roe"] = [
            f"pb/pe ÷ roe 中位数 {med:.3f} 吸附不到整倍数,roe 按原值使用(未声明口径)"]

    # (2) 流量倍数 = f_roe × roe ÷ (netinc/equity)。roe 已是年度口径时这个比值就是该点
    #     流量的倍数;roe 自己也是单季时(实测 WFC),f_roe 正好把它补回来。
    #     该日期没有 roe 时用最近几个点的 roe 代表值 —— 否则像 CBA(roe 只从 2024 起
    #     才有)前半段的半年度口径就完全看不出来了。
    if netinc and equity:
        per_equity = _ratio_of(netinc, equity, lambda n, e: n / e)
        roe_ref = statistics.median(list(roe.values())[-_RECENT:]) if roe else None
        votes: dict[str, float] = {}
        for d, pe in per_equity.items():
            basis = roe.get(d) or roe_ref
            if basis and pe:
                votes[d] = res.f_roe * basis / pe
        res.flow_votes = votes
        med, recent = _median_recent(votes)
        snapped = _snap(med)
        res.f_flow = snapped or 1.0
        if med is not None:
            res.evidence["flow"] = [
                f"roe ÷ (净收益/净资产) 最近 {len(recent)} 点中位数 {med:.3f}"
                f"→ 流量 ×{res.f_flow:g}"
                f"(已含 roe 基准 ×{res.f_roe:g}"
                + (f";{len(votes) - len(set(roe) & set(per_equity))} 个点无 roe,"
                   f"用最近 roe 代表值 {roe_ref:.4g} 推算" if roe_ref else "")
                + ")"]
    if not res.evidence.get("flow"):
        res.f_flow = res.f_roe
        res.evidence["flow"] = [
            f"缺净资产或净收益:流量按 roe 的基准 ×{res.f_roe:g} 处理(无法独立验证)"]
    return res


def _block_factors(votes: dict[str, float], canonical: float) -> dict[str, float]:
    """序列中途整块翻口径时的逐点倍数。返回 {日期: 该点自己的倍数}。

    判据是**每个点自己的投票**离基准有多远:同一块的年代噪声(实测 NAB 前期
    3.9–6.5,围绕块中位数 ±20%)远小于换口径的整倍差(log 2 = 0.69),所以用一个
    0.40 的分界就能干净地分开,且不受"块边界落在哪一天"影响。

    块倍数取**全部异口径点投票的中位数**再吸附 —— 单点的年代噪声(实测有个 9.8)
    被中位数吃掉,不会把整块带偏。
    """
    other = {d: v for d, v in votes.items() if abs(math.log(v / canonical)) > _BLOCK_SEP}
    if len(other) < min(2, len(votes)):
        return {}                       # 孤点不足以判"整块翻转",宁可不改
    med = statistics.median(other.values())
    f = _snap(med)
    if f is None or f == canonical:
        return {}
    return {d: f for d in other}


def _nearest(voted: list[str], d: str) -> str | None:
    """与 d 日期最接近的已判定点(ISO 日期字符串可直接比较)。"""
    if not voted:
        return None
    i = bisect.bisect_left(voted, d)
    cands = [v for v in (voted[i - 1] if i else None, voted[i] if i < len(voted) else None) if v]
    return min(cands, key=lambda v: abs(_date(v) - _date(d))) if cands else None


def _dominant(win: list[float]) -> tuple[float, int]:
    """邻居里最大的一组"互相在 ±_TIGHT 内"的值 → (组内中位数, 组大小)。

    用最大簇而不是所有邻居的中位数:序列正在换挡时(实测 CBA 半年度→年度、
    WFC 股息 0.10→0.20)两种水平会同时出现在窗口里,中位数落在两者之间,
    谁也算不准。
    """
    best = (statistics.median(win), 1)
    for c in win:
        grp = [v for v in win if abs(math.log(v / c)) <= _TIGHT]
        if len(grp) > best[1]:
            best = (statistics.median(grp), len(grp))
    return best


def _level_of(vals: list[float], i: int) -> tuple[float, bool] | None:
    """该点所属的**水平**,以及"够不够格判它是坏点"。

    取离它最近的 2×_LOCAL 个邻居,要求其中 ≥75% 构成同一水平(否则这个窗口本身
    还在换挡,不判)。另外**换挡点的两侧邻居必然分属两个水平** —— 真实的口径/水平
    变化会被这一条挡下,只有"前后都站在同一水平上、自己孤零零差一整倍"的点才会
    被判为坏点(实测 NAB 尾部 dps 3.400 与 0.425 正是这种形状)。
    """
    n = len(vals)
    order = sorted((j for j in range(n) if j != i and vals[j] > 0), key=lambda j: (abs(j - i), j))
    win = order[: _LOCAL * 2]
    if len(win) < 2:
        return None
    level, size = _dominant([vals[j] for j in win])
    if size < max(2, math.ceil(0.75 * len(win))):
        return None
    sides = [[j for j in win if j < i][:1], [j for j in win if j > i][:1]]
    same = [abs(math.log(vals[j] / level)) <= _TIGHT for side in sides for j in side]
    return level, len(same) < 2 or all(same)     # 端点只有一侧,只要求那一侧


def _build(metric: str, kind: str, points: dict[str, float | None], res: _Resolution,
           factor: float, determined: bool) -> AnnualSeries:
    """按因子换算成年度口径,再逐点查坏点(整块翻转与尾部零星坏点都在这一步)。"""
    scalable = not (metric in _SCALE_FREE_RATIOS or kind == _STOCK)
    key = ("roe" if kind == _RATIO else "flow") if scalable else None
    s = AnnualSeries(metric=metric, kind=kind, factor=factor, determined=determined,
                     raw=dict(points), evidence=list(res.evidence.get(key or "", [])))
    scaled = {d: float(v) * factor for d, v in points.items() if v is not None}
    if not scaled:
        return s
    dates = sorted(scaled)

    # (1) 整块翻口径:逐点按自己的恒等式投票换算(点自己说了算,边界点也不会漏)。
    if scalable and kind == _FLOW and res.flow_votes:
        pf = _block_factors(res.flow_votes, factor)
        # 每个有依据的点的倍数(含基准块),供"自身无依据"的点就近推断
        known = {d: factor for d in res.flow_votes}
        known.update(pf)
        voted = sorted(known)
        inherited = 0
        for d in dates:
            own = pf.get(d)
            if own is None and d not in res.flow_votes:
                near = _nearest(voted, d)
                if near is None or known[near] == factor:
                    continue             # 邻居都是基准口径 → 无需推断
                own, inherited = known[near], inherited + 1
                s.adjusted[d] = (f"该点无自身的口径依据,按最近的 {near} 推断为 ×{own:g},"
                                 f"修正为 {float(points[d]) * own:.6g}")
            elif own is not None and own != factor:
                s.adjusted[d] = (f"该点口径倍数为 ×{own:g}(与序列基准 ×{factor:g} 不同),"
                                 f"修正为 {float(points[d]) * own:.6g}")
            else:
                continue
            scaled[d] = float(points[d]) * own
        if pf:
            med = statistics.median(res.flow_votes[d] for d in pf)
            s.evidence.append(
                f"{len(pf)} 个点({min(pf)}..{max(pf)})的恒等式比值中位数 {med:.2f}"
                f" → 这些点是 ×{next(iter(pf.values())):g} 口径,已逐点换算"
                + (f";另有 {inherited} 个点缺依据,按时间最近的同类点推断" if inherited else ""))

    # (2) 单点坏点:与邻域水平差恰好整倍的按邻域水平修正。这一类修不了"整块翻转"
    #     (块内邻居互相支持),但能抓住紧挨着的尾部坏点,以及 5 年窗口里孤零零一个
    #     前期口径点。
    #     邻域用**已修正**的值,否则相邻两个坏点会互相拖住(实测 NAB 尾部
    #     dps 0.425 与 3.400 紧挨着,各自是对方的"异常邻域")。
    if scalable:
        vals = [scaled[d] for d in dates]
        for i, d in enumerate(dates):
            probe = _level_of(vals, i)
            if not probe:
                continue
            level, ok = probe
            r = scaled[d] / level
            own = _snap(r, _TIGHT)
            if own and own != 1.0 and ok:
                fixed = scaled[d] / own
                s.adjusted[d] = (f"该点是前后邻域水平({level:.6g})的 {r:.3f} 倍,"
                                 f"按 {fixed:.6g} 修正")
                scaled[d], vals[i] = fixed, fixed

    s.values = {d: scaled[d] for d in dates}
    return s


def annualize(raw: Any) -> dict[str, AnnualSeries]:
    """一次抓取的原始结果 → 每个指标一条**年度口径**序列。

    必须整块一起走:恒等式要跨指标比对,单看一条序列判不出口径。
    """
    points = {m: points_of(raw, m) for m in metrics_of(raw)}
    sources = (raw or {}).get("metric_sources") or {}
    res = _resolve(points)

    out: dict[str, AnnualSeries] = {}
    for metric, pts in points.items():
        kind = classify(metric, sources.get(metric))
        if kind == _STOCK or metric in _SCALE_FREE_RATIOS:
            s = _build(metric, kind, pts, res, 1.0, determined=True)
        elif kind == _RATIO:                      # roe / roa:期间比率,随 roe 的基准
            s = _build(metric, kind, pts, res, res.f_roe, res.anchored)
            if not res.anchored:
                s.warnings.append(f"{metric}:口径锚定失败(pb/pe/roe 缺失或吸附不上),按原值使用")
        else:                                     # 流量:随流量基准
            s = _build(metric, kind, pts, res, res.f_flow, res.anchored)
            if not res.anchored:
                s.warnings.append(
                    f"{metric}:口径锚定失败(pb/pe/roe 或净资产缺失),按原值使用"
                    "——若该序列是单季口径,代入年度模型会系统性偏小")
        out[metric] = s
    return out


def annual_series(raw: Any, metric: str) -> AnnualSeries | None:
    """单指标便捷入口(等价于 annualize(raw)[metric])。"""
    return annualize(raw).get(metric)


def scale_mismatch(value: float | None, ref: float | None, tol: float = 0.20) -> float | None:
    """LLM 填的标量相对数据里的年度值是否**差着整倍数**(0.25/0.5/2/4)。是则返回该倍数。

    消费端的最后一道闸:上下文里已经给了归一化后的年度值,模型仍可能填进原始的单季
    值(实测 DDM 的 eps0 就是这么错的)。用整倍数而不是"相差超过 x%"来判断,是因为
    真实的归一化调整(危机年 ROE、一次性重组的 EPS)会偏离年度值十几二十个百分点,
    模糊阈值会一直误报,而口径错误一定是精确的 2 倍/4 倍。

    返回值不含 1.0(那就是同口径,不算问题)。
    """
    if not value or not ref or value <= 0 or ref <= 0:
        return None
    r = float(value) / float(ref)
    for f in _CANDIDATES:
        if f != 1.0 and abs(math.log(r / f)) <= tol:
            return f
    return None


def by_year(s: AnnualSeries, years: int = _BY_YEAR_YEARS,
            per_year: int = _BY_YEAR_POINTS) -> dict[str, dict[str, float]]:
    """按年列出**归一化后**的观测:{"2025": {"2025-06-30": 4.85, "2025-12-31": 4.95}}。

    为什么要有这一层(skill 03 实测连调 13 次 `get_financials` 的根因):schema 要求
    LLM 填**按年**的字段(`dps_by_year`),而注入块原来只给了 `latest` 与被修正的点 ——
    实测 CBA 的 dps 只给了 6 个修正点(2021–2023),2024/2025 的年度值一个都没有,
    而工具返回的原始序列又只保留首尾各 6 个点(中间年份被省略)。模型于是换着 period
    反复抓取,指望某个窗口能露出缺的那一年:5y→all→2y→1y→5y→all…,上下文涨到 31k
    token。**数据缺口只能用数据补,提示词补不上。**

    这里不做任何聚合:每格就是该年观测到的原始点(已年化),取哪一年由 LLM 按日期判断
    (CBA 财年 6 月结束,NAB 是 9 月,聚合会把两者都算错)。只限制最近 `years` 年、
    每年最近 `per_year` 个点,保证注入块有界。
    """
    out: dict[str, dict[str, float]] = {}
    for d, v in sorted(s.values.items())[-(years * per_year * 4):]:
        if v is None or not math.isfinite(float(v)):
            continue
        out.setdefault(str(d)[:4], {})[str(d)] = round(float(v), 6)
    return {y: dict(sorted(pts.items())[-per_year:]) for y, pts in sorted(out.items())[-years:]}


def describe(block: dict[str, AnnualSeries], metrics: Iterable[str] | None = None) -> dict:
    """供 ctx 注入:把**口径与依据**明确交给 LLM,替掉"靠量级猜口径"的提示词。

    金额类指标另附 Python 算好的同比 `growth_yoy`(中位数口径,且基于修过坏点的序列 ——
    数据源自己的 dividendGrowth 尾部被坏点污染成 +300%),LLM 不手算。
    每条指标还带 `by_year`(见 `by_year`):按年字段只能用这个填,否则 LLM 会去换 period
    反复抓取找缺失年份 —— 原始序列在工具层只保留首尾各 6 个点,越抓越缺。
    """
    keep = set(metrics) if metrics else None
    out: dict[str, Any] = {}
    for m, s in block.items():
        if (keep and m not in keep) or not s.values:
            continue
        d, v = s.latest()
        out[m] = {
            "basis": s.basis,
            "latest_date": d,
            "latest": round(v, 6),
            "n": len(s.values),
            "evidence": (s.evidence or ["(无)"])[0],
            "adjusted_points": dict(sorted(s.adjusted.items())[-_RECENT:]),
            "by_year": by_year(s),
        }
        if s.kind == _FLOW:
            # 金额类指标的同比由 Python 算(LLM 不手算),且基于修过坏点的序列:
            # 数据源自己的 dividendGrowth 尾部被坏点污染成 +300%
            g = s.growth()
            if g is not None:
                out[m]["growth_yoy"] = round(g, 6)
        if s.warnings:
            out[m]["warning"] = s.warnings[0]
    return out
