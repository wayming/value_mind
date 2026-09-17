"""MCP 工具包装:把 mcp_client 的静态定义变成 LangChain callable 工具。

关键设计(用户要求):
- MCP 只注册给 LLM:三个工具以 callable 形式绑定到模型,LLM 自主决定何时调用哪个;
  编排代码不硬编码调用时机。
- 原始 structuredContent 记入 cache_holder,供 Python 侧交叉校验与统计
  (如 ROE 序列的标准差,由代码计算而非 LLM 手算)。
- 调用失败不抛异常,返回错误字符串,让 LLM 自行换指标/换 period 或改用假设。
- 返回内容压缩,防止超长(每个指标序列保留首尾各 6 个点)。
"""

from typing import Any, Callable, Sequence

import json

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

import app.config as config
import app.series as S
from mcp_client import get_tool_definitions

MAX_RESULT_CHARS = 30000
MAX_METRIC_NAMES = 150
MAX_SERIES_POINTS = 12  # 每条指标序列保留**最近** 12 个点(见 _compact_financials)
_MIN_ANCHOR_POINTS = 4  # 靠"能锚定口径"胜出的抓取至少要有的观测数(见 _select_fetch)

# 账本的两句提示。都极短:被拦下的调用也要占一个 tool 结果,长提示会把
# "别再抓了"变成新的上下文负担。
_REPEAT_NOTE = (
    "【重复调用,未执行】这组参数已经调用过(第 {n} 次),结果与上面完全相同,不再重复返回。"
    "**不要再调用任何工具**,请立即输出结构化参数。"
)
_BUDGET_NOTE = (
    "【额度用完,未执行】本节点最多 {budget} 次 get_financials,已用完。"
    "**不要再调用任何工具**,立即基于已有数据输出结构化参数;缺失字段用合理假设代替,"
    "并在理由/analysis 字段里标注为假设。换 period 也没有用:序列只保留最近的点,"
    "逐年年度值在上下文的「Python 判定口径后的年度数据」里(每条指标带 by_year)。"
)

# 工具中文描述直接从 mcp_client 静态定义复制,保持单一事实来源
_DESC = {
    d["function"]["name"]: d["function"]["description"]
    for d in get_tool_definitions()
}


def _truncate(text: str) -> str:
    if len(text) > MAX_RESULT_CHARS:
        return text[:MAX_RESULT_CHARS] + f"\n...(已截断,共 {len(text)} 字符)"
    return text


def _compact_financials(raw: Any) -> str:
    """get_financials:把 date->statement->metric 重组为 metric->date 序列并压缩。

    截断保留**最近**的点(而不是首尾各一半):估值只消费近期数据,而被省略的中间年份
    正是模型的饥饿点 —— 实测它为了找 2023/2024 的年度值换着 period 反复抓取
    (`_truncated` 里因此必须写明省掉的是哪一段,以及换窗口找不回来)。
    """
    data = (raw or {}).get("data") or {}
    series: dict[str, dict[str, Any]] = {}
    for date, stmts in data.items():
        for stmt, metrics in (stmts or {}).items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                series.setdefault(metric, {})[date] = value
    truncated: dict[str, dict] = {}
    for metric, points in series.items():
        dates = sorted(points)
        if len(dates) > MAX_SERIES_POINTS:
            keep = dates[-MAX_SERIES_POINTS:]
            series[metric] = {d: points[d] for d in keep}
            truncated[metric] = {
                "total": len(dates), "kept": MAX_SERIES_POINTS,
                "omitted": f"{dates[0]}…{keep[0]}(更早的 {len(dates) - MAX_SERIES_POINTS} 个点)",
            }
    out: dict[str, Any] = {"metric_sources": (raw or {}).get("metric_sources"), "series": series}
    if truncated:
        out["_truncated"] = truncated
        out["_note"] = (
            f"每条序列只返回**最近 {MAX_SERIES_POINTS} 个**观测(更早的点被省略,见 _truncated),"
            "换 period 或换窗口只会换一批被省略的点,不会多出中间的年份。"
            "逐年年度值(已按口径年化)在上下文的「Python 判定口径后的年度数据」的 by_year 里,"
            "直接用它,不要再为了找某一年反复抓取。"
        )
    return _truncate(json.dumps(out, ensure_ascii=False))


def _compact_metrics(raw: Any) -> str:
    """list_metrics:压缩指标名列表。"""
    metrics = dict((raw or {}).get("metrics") or {})
    truncated: dict[str, dict] = {}
    for group, names in metrics.items():
        if isinstance(names, list) and len(names) > MAX_METRIC_NAMES:
            truncated[group] = {"total": len(names), "kept": MAX_METRIC_NAMES}
            metrics[group] = names[:MAX_METRIC_NAMES]
    out: dict[str, Any] = {"metrics": metrics}
    if truncated:
        out["_truncated"] = truncated
    return _truncate(json.dumps(out, ensure_ascii=False))


def _compact(tool_name: str, raw: Any) -> str:
    if isinstance(raw, dict) and set(raw.keys()) <= {"text"}:
        return raw["text"]  # 非结构化文本(通常是错误消息),原样透传
    if tool_name == "list_metrics":
        return _compact_metrics(raw)
    if tool_name == "get_financials":
        return _compact_financials(raw)
    return _truncate(json.dumps(raw, ensure_ascii=False))

class _CallLedger:
    """单节点的工具调用账本:**去重** + `get_financials` 硬额度。

    这是"数据够用即止"的确定性版本 —— 提示词挡不住的那个循环由代码来断。实测
    skill 03 拿同一个指标清单换着 period 连抓 13 次(5y→all→2y→1y→5y→all…),每次都把
    ~2k token 的结果追加进上下文,涨到 31k。拦下来之后:
      完全相同参数 → 不再返回数据(结果就在上面),只回一行提示;
      get_financials 达到 `config.MAX_FINANCIALS_CALLS` → 不执行,提示立即输出参数。

    拒绝一律返回**短字符串**而不抛异常:异常会打断 agent(structured_response=None),
    把"拦一次多余的抓取"变成"整个节点失败"。
    """

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self._seen: dict[str, dict[str, int]] = {}

    def check(self, name: str, args: dict[str, Any]) -> str | None:
        """放行返回 None;否则返回该给 LLM 的提示文本(本次调用不执行)。"""
        key = json.dumps(args, sort_keys=True, ensure_ascii=False)
        seen = self._seen.setdefault(name, {})
        if key in seen:
            seen[key] += 1
            return _REPEAT_NOTE.format(n=seen[key])
        if name == "get_financials" and len(seen) >= self.budget:
            return _BUDGET_NOTE.format(budget=self.budget)
        seen[key] = 1
        return None


async def make_mcp_tools() -> tuple[list[Callable], dict[str, Any]]:
    """创建 3 个 MCP callable 工具 + 缓存容器。

    返回 (tools, cache):tools 绑定给 LLM;cache 记录 LLM 调用过的原始结果,
    节点结束时合并进 state["data_cache"]。账本随工具一起创建 = 每个节点一份。
    """
    tools = []
    cache: dict[str, Any] = {}
    ledger = _CallLedger(config.MAX_FINANCIALS_CALLS)
    try:
        client = MultiServerMCPClient(
            {
                "sacollecotor": {
                    "url": "http://localhost:8081/mcp",
                    "transport": "streamable-http"
                }
            }
        )
        for tool in await client.get_tools():
            tools.append(_wrap_tool(tool, cache, ledger))
        return tools, cache
    except Exception as e:
        print(f"Failed to create MCP tools: {e}")
        return [], cache

def _wrap_tool(tool, cache: dict[str, Any], ledger: _CallLedger | None = None):
    ledger = ledger or _CallLedger(config.MAX_FINANCIALS_CALLS)
    async def _ainvoke(**kwargs: Any):
        # 缓存还原后的结构化结果(供 Python 侧算统计量),给 LLM 的是压缩文本:
        # 两者不能混——塞原始块进缓存会让 _points_of 拿到 list 直接炸,
        # 而把结构化结果整个丢给 LLM 又会白烧上下文(实测一次 get_financials 未压缩
        # 就上万 token,且长序列会把关键的头尾挤出视野)。
        blocked = ledger.check(tool.name, kwargs)
        if blocked is not None:
            return blocked
        raw = _structured(await tool.ainvoke(kwargs))
        cache[f"{tool.name}|{json.dumps(kwargs, sort_keys=True, ensure_ascii=False)}"] = raw
        return _compact(tool.name, raw)
    return StructuredTool.from_function(
        coroutine = _ainvoke,
        name = tool.name,
        # 中文描述仍以 mcp_client 的静态定义为准(服务器给的是英文,而 skill 提示词、
        # 指标口径说明都是中文,单一事实来源在这里)
        description = _DESC.get(tool.name, tool.description),
        args_schema=tool.args_schema
    )

def _fetch_args(key: str) -> dict[str, Any]:
    """从缓存键 `get_financials|{json}` 还原调用参数。"""
    try:
        return json.loads(key.split("|", 1)[1])
    except (IndexError, json.JSONDecodeError):
        return {}


def _structured(raw: Any) -> Any:
    """把 MCP 工具的返回值还原成 structuredContent 那个 dict。

    langchain-mcp-adapters 的工具是 `response_format="content_and_artifact"`
    (见其 tools.py 的 StructuredTool(...)),但 `ainvoke()` 只回 **content 块** ——
    `[{"type": "text", "text": "{...}"}]`,structuredContent 在 artifact 里被丢掉了。
    而下游(`_points_of` / `_compact_financials`)读的是 structuredContent 的结构,
    拿到 list 就 `AttributeError: 'list' object has no attribute 'get'`。

    文本块里的 JSON 与 structuredContent 逐字节相同(ASX NAB 三个工具实测),
    所以这里按"解析文本块"还原,与旧版 mcp_client.call_tool 的取值顺序等价:
    structuredContent(拿不到)→ content[0].text 解析 → 解析不了就 {"text": ...} 原样透传
    (错误消息走这条,由 _compact 识别并直接给 LLM)。
    """
    if isinstance(raw, tuple):          # 万一将来改回 (content, artifact)
        raw = raw[0]
    if isinstance(raw, dict):
        return raw
    blocks = raw if isinstance(raw, list) else []
    texts = [b.get("text", "") for b in blocks
             if isinstance(b, dict) and b.get("type") == "text"]
    if len(blocks) == 1 and len(texts) == 1:
        try:
            parsed = json.loads(texts[0])
        except json.JSONDecodeError:
            return {"text": texts[0]}
        return parsed if isinstance(parsed, dict) else {"text": texts[0]}
    # 多块/非文本块(图片等)不是本应用的用法:拼成文本透传,不猜结构
    return {"text": "\n".join(texts) or json.dumps(blocks, ensure_ascii=False)}


_points_of = S.points_of      # 单一实现(含"缓存值必须是结构化 dict"的报错)


def _rank(raw: Any, metric: str | None, focus: Sequence[str] | None = None) -> tuple:
    """这次抓取在"该选它吗"这件事上的排序键(越靠前越该选,空元组 = 没有优势)。

    口径归一化靠指标间的恒等式(`pb/pe ÷ roe`、`roe ÷ (净收益/净资产)`),一次抓取里缺了
    参照指标就判不出来,只能按原值用。实测 LLM 会把 `pe/pb/roe` 和 `roe/dps` 拆成两次抓取
    (skill 里也允许它这么做),于是选到哪一份决定了 σ 是否可信:NYSE:WFC 的 roe 是单季值,
    未年化直接算标准差会小 4 倍,而 PB 回归的 σ 正是预测倍数的主要输入。

    - `metric` 给定时只看这一个指标:能判定口径才算优势(1/0)。
    - `focus` 给定时(focus = 调用方真正要用的指标)按 **(判定得出的个数, 覆盖到的个数,
      全表能判定的个数)** 排序。判定个数在前:注入给 LLM 的值是让它**直接采用**的,一份
      覆盖全但一个都没判定的抓取,等于把可能差 4 倍的值喂进去 —— 宁可少给几个指标。
    - 两者都不给时按整表能判定的个数(`needs_normalization`,存量与无量纲比率的 determined
      是白送的,不数它们:一份全是资产负债表的抓取会靠这些赢过真正带流量指标的那份)。
    """
    try:
        block = S.annualize(raw)
    except Exception:
        return ()
    if metric:
        s = block.get(metric)
        return (1 if s is not None and s.determined else 0,)
    determined = covered = 0
    for m in focus or ():
        s = block.get(m)
        if s is not None and s.values:
            covered += 1
            determined += 1 if s.determined else 0
    provable = sum(1 for s in block.values() if s.determined and s.needs_normalization)
    return (determined, covered, provable)


def _select_fetch(
    cache: dict[str, Any],
    metric: str | None,
    exchange: str,
    code: str,
    period: str | None = None,
    focus: Sequence[str] | None = None,
) -> tuple[Any | None, int, str]:
    """挑出**一家公司、一次** get_financials 抓取。返回 (raw 或 None, 非空观测数, 实际 period)。

    metric 为 None 表示"不限定指标",此时按整份抓取能判定口径的指标个数打分,再比非空观测
    总数 —— 口径归一化(app/series.py)要跨指标比对恒等式,拿到的那一份越完整越判得准。

    `focus` 是多指标调用方(整块注入、报告口径表)真正要用的指标名单,参与排序(见 `_rank`)。
    取数本身只按 metric 计数。

    三条约束都是实测踩出来的:

    - **必须限定 exchange/code**。缓存里同时躺着本次运行抓过的所有公司,只按日期合并会把
      同业公司的同名指标缝进目标公司序列。实测 ASX NAB 的报告里 `roe_series_raw_latest`
      = 0.0949 其实是西太平洋银行(WBC)的值——NAB 自己是 0.09908,而 skill 07 恰好会抓
      同业做对比,两家澳洲银行的报告日又完全相同(都是 3/31、9/30),于是逐日覆盖、无痕。
      σ 也因此是两家公司的混合体,直接喂进了 PB 回归。
    - **取单次抓取,不跨抓取合并**。同一公司多次抓取(1y/2y/5y/all)在重叠日期上取值一致,
      合并本身不产生错值,但会改变样本数:`all` 比 `5y` 多出 20 年历史(含 2020 年 ROE
      腰斩那段),σ 于是随 LLM 恰好调过哪些 period 而变,同一家公司两次运行能给出不同的
      预测 PB。所以按声明的窗口选一份,选不中就取信息量最大的那份,并把实际用的 period
      回传给调用方,让它能核对是不是自己声明的那个。
    - **优先选能自己锚定口径的那份**(见 `_rank`)。缓存是跨节点累积的
      (`state["data_cache"]`),同一个问题往往有好几份抓取;多一样参照指标,口径就从
      "按原值用"变成"判定得出",这比多几个观测点值钱。整块调用时,调用方还要把
      `focus`(它真正要用的指标)传进来,否则一份全是资产负债表的抓取会靠白送的
      determined 胜出,注入里一个要用的指标都没有。
    """
    want_ex, want_code = str(exchange).strip().upper(), str(code).strip().upper()
    cands: list[tuple[Any, int, int, str, tuple]] = []
    for key, raw in cache.items():
        if not key.startswith("get_financials|"):
            continue
        if raw is None:
            continue          # 抓取没留下结果(调用失败),当没抓过;类型不对则仍要炸出来
        args = _fetch_args(key)
        if (str(args.get("exchange", "")).strip().upper(),
                str(args.get("code", "")).strip().upper()) != (want_ex, want_code):
            continue
        if metric:
            points = _points_of(raw, metric)
            n = sum(v is not None for v in points.values())
            total = len(points)
        else:
            points = {m: _points_of(raw, m) for m in S.metrics_of(raw)}
            n = sum(v is not None for p in points.values() for v in p.values())
            total = sum(len(p) for p in points.values())
        if not n:
            continue          # 该次抓取没带回可用数据(或全是空值),不参与选择
        cands.append((raw, n, total, str(args.get("period") or ""),
                      _rank(raw, metric, focus)))
    if not cands:
        return None, 0, ""

    # 口径优势优先,但只认观测数不至于太离谱的抓取:一份 1 个点的抓取不该顶掉 40 个点的
    # 历史(σ 会变成单点算出来的数)。门槛定得低是因为两个方向的代价不对称 —— 口径错的
    # 序列差 2/4 倍,样本少一半顶多差几十个百分点,而节点另有 n<3 的告警。
    def credit(c: tuple) -> tuple:
        return c[4] if c[1] >= _MIN_ANCHOR_POINTS else ()

    # 口径优势 > 信息量(并列时取总点数少的,即最紧凑的窗口)
    def best(pool: list[tuple]) -> tuple:
        return max(pool, key=lambda c: (credit(c), c[1], -c[2]))

    declared = (period or "").strip().lower()
    exact = [c for c in cands if c[3].strip().lower() == declared] if declared else []
    chosen = best(exact or cands)
    if not any(credit(chosen)):       # () = 观测太少;全 0 = 有观测但没有任何口径优势
        # 声明的窗口里没有一份判得出口径(缺 pb/pe 等参照指标)时,换一份能判定的抓取:
        # 未年化的值直接进 σ 会把标准差放大 2/4 倍,比换窗口伤得多。实测 ASX:NAB 就是
        # LLM 把 pe/pb/roe(all)与 roe/payoutratio/dps(5y)拆成两次抓取,声明 5y 却选了
        # 只有 roe 的那份。换窗口会在窗口核对里体现(节点会告警),σ 却是对的。
        usable = [c for c in cands if any(credit(c))]
        if usable:
            chosen = best(usable)
    raw, n, _, used_period, _ = chosen
    return raw, n, used_period


def cached_metrics(cache: dict[str, Any], exchange: str, code: str) -> set[str]:
    """这家公司在缓存里出现过的**指标名**集合(不看取值,只看有没有)。

    报告用它把"没判出口径"说清楚:某个指标一次都没抓到,和抓到了却没有参照指标可判定,
    是两件不同的事,读报告的人得能区分。
    """
    want_ex, want_code = str(exchange).strip().upper(), str(code).strip().upper()
    out: set[str] = set()
    for key, raw in cache.items():
        if not key.startswith("get_financials|") or not isinstance(raw, dict):
            continue
        args = _fetch_args(key)
        if (str(args.get("exchange", "")).strip().upper(),
                str(args.get("code", "")).strip().upper()) == (want_ex, want_code):
            out.update(S.metrics_of(raw))
    return out


def extract_series(
    cache: dict[str, Any],
    metric: str,
    exchange: str,
    code: str,
    period: str | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """从 data_cache 的 get_financials 原始结果中提取**一家公司**的 metric 日期->值序列(升序)。

    **这是数据源的原始值,口径混杂**(同一序列里可能前几年是年报值 ÷4、后几年是 12 个月
    TTM,不同公司各不相同)。要喂给年度复利的估值模型,请改用 `annual_block`。

    返回 (points, meta)。meta = {period, n, source},period 为实际选中的抓取窗口,
    n 为非空观测数,source 可直接写进报告供追溯。
    """
    raw, n, used_period = _select_fetch(cache, metric, exchange, code, period)
    if raw is None:
        return {}, {"period": "", "n": 0, "source": "无匹配抓取"}
    return _points_of(raw, metric), {
        "period": used_period, "n": n,
        "source": f"period={used_period or '未指定'}, n={n}"}


def annual_block(
    cache: dict[str, Any],
    exchange: str,
    code: str,
    period: str | None = None,
    metric: str | None = None,
    metrics: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """挑一次抓取,把**整块**指标归一化成年度口径(app/series.py)。返回 (block, meta)。

    block = {指标名: AnnualSeries},值已换算到 12 个月口径(DDM/FCFE/超额回报全部按年度
    复利,喂未年化的季度/半年度值会把估值系统性缩小数倍);meta 含实际 period、口径判定
    依据与被修正的点,可直接写进报告供追溯。

    metric 给定时要求选中的这次抓取确实带回了该指标(与 extract_series 同一套选择规则),
    meta 里也会带上该指标的口径、倍数与告警。整块调用(两者都不给)用于上下文注入;若把
    **要用的指标名单**经 `metrics` 传进来,选择会偏向"既有这些指标、又能判定其口径"的抓取
    (否则一份全是资产负债表的抓取会靠白送的 determined 胜出,注入里一个要用的指标都没有)。
    """
    raw, n, used_period = _select_fetch(cache, metric, exchange, code, period, focus=metrics)
    if raw is None:
        return {}, {"period": "", "n": 0, "source": "无匹配抓取"}
    block = S.annualize(raw)
    focus = block.get(metric) if metric else None
    if metric:
        source = f"period={used_period or '未指定'}, 指标 {metric}, n={n}"
    else:
        source = f"period={used_period or '未指定'}, {len(block)} 个指标"
    meta: dict[str, Any] = {"period": used_period, "n": n, "source": source}
    if focus is not None:
        meta.update({
            "basis": focus.basis, "factor": focus.factor, "determined": focus.determined,
            "evidence": list(focus.evidence), "adjusted": dict(sorted(focus.adjusted.items())),
            "warnings": list(focus.warnings),
            "source": f"{source}, 口径={focus.basis}",
        })
    return block, meta
