"""MCP 工具包装:把 mcp_client 的静态定义变成 LangChain callable 工具。

关键设计(用户要求):
- MCP 只注册给 LLM:三个工具以 callable 形式绑定到模型,LLM 自主决定何时调用哪个;
  编排代码不硬编码调用时机。
- 原始 structuredContent 记入 cache_holder,供 Python 侧交叉校验与统计
  (如 ROE 序列的标准差,由代码计算而非 LLM 手算)。
- 调用失败不抛异常,返回错误字符串,让 LLM 自行换指标/换 period 或改用假设。
- 返回内容压缩,防止超长(每个指标序列保留首尾各 6 个点)。
"""

import json
from typing import Any, Callable

from langchain_core.tools import tool as lc_tool

from mcp_client import get_mcp_client, get_tool_definitions

MAX_RESULT_CHARS = 30000
MAX_METRIC_NAMES = 150
MAX_SERIES_POINTS = 12  # 每条指标序列保留首尾各 6 个点

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
    """get_financials:把 date->statement->metric 重组为 metric->date 序列并压缩。"""
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
            keep = dates[: MAX_SERIES_POINTS // 2] + dates[-MAX_SERIES_POINTS // 2:]
            series[metric] = {d: points[d] for d in keep}
            truncated[metric] = {"total": len(dates), "kept": MAX_SERIES_POINTS}
    out: dict[str, Any] = {"metric_sources": (raw or {}).get("metric_sources"), "series": series}
    if truncated:
        out["_truncated"] = truncated
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


def make_mcp_tools() -> tuple[list[Callable], dict[str, Any]]:
    """创建 3 个 MCP callable 工具 + 缓存容器。

    返回 (tools, cache):tools 绑定给 LLM;cache 记录 LLM 调用过的原始结果,
    节点结束时合并进 state["data_cache"]。
    """
    client = get_mcp_client()
    cache: dict[str, Any] = {}

    def call(name: str, args: dict) -> str:
        if client is None:
            return ("错误:MCP 未连接,数据不可用。请基于行业常识给出假设,"
                    "并在相应理由字段中明确标注为假设。")
        raw = client.call_tool(name, args)
        if raw is None:
            return (f"错误:调用 {name} 失败或无数据(可能指标不存在、该股票无数据或服务器错误)。"
                    "可先调用 list_metrics 确认可用指标,或缩短 period 重试;"
                    "仍不行则基于行业常识假设并在理由字段标注。")
        cache[f"{name}|{json.dumps(args, sort_keys=True, ensure_ascii=False)}"] = raw
        return _compact(name, raw)

    def _list_metrics(exchange: str, code: str) -> str:
        return call("list_metrics", {"exchange": exchange, "code": code})

    def _get_data_period(exchange: str, code: str) -> str:
        return call("get_data_period", {"exchange": exchange, "code": code})

    def _get_financials(exchange: str, code: str, metrics: list[str], period: str) -> str:
        return call(
            "get_financials",
            {"exchange": exchange, "code": code, "metrics": metrics, "period": period},
        )

    # 先赋中文 docstring(langchain 在 lc_tool() 调用时读取),再包装成工具
    for fn, name in (
        (_list_metrics, "list_metrics"),
        (_get_data_period, "get_data_period"),
        (_get_financials, "get_financials"),
    ):
        fn.__doc__ = _DESC[name]

    tools = [
        lc_tool("list_metrics")(_list_metrics),
        lc_tool("get_data_period")(_get_data_period),
        lc_tool("get_financials")(_get_financials),
    ]
    return tools, cache


def _fetch_args(key: str) -> dict[str, Any]:
    """从缓存键 `get_financials|{json}` 还原调用参数。"""
    try:
        return json.loads(key.split("|", 1)[1])
    except (IndexError, json.JSONDecodeError):
        return {}


def _points_of(raw: Any, metric: str) -> dict[str, float | None]:
    points: dict[str, float | None] = {}
    for date, stmts in ((raw or {}).get("data") or {}).items():
        for metrics in (stmts or {}).values():
            if isinstance(metrics, dict) and metric in metrics:
                points[date] = metrics[metric]
    return dict(sorted(points.items()))


def extract_series(
    cache: dict[str, Any],
    metric: str,
    exchange: str,
    code: str,
    period: str | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """从 data_cache 的 get_financials 原始结果中提取**一家公司**的 metric 日期->值序列(升序)。

    供 Python 计算统计量(标准差、均值、最新值),LLM 不手算。

    两条约束都是实测踩出来的:

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

    返回 (points, meta)。meta = {period, n, source},period 为实际选中的抓取窗口,
    n 为非空观测数,source 可直接写进报告供追溯。
    """
    want_ex, want_code = str(exchange).strip().upper(), str(code).strip().upper()
    cands: list[tuple[dict[str, float | None], str]] = []
    for key, raw in cache.items():
        if not key.startswith("get_financials|"):
            continue
        args = _fetch_args(key)
        if (str(args.get("exchange", "")).strip().upper(),
                str(args.get("code", "")).strip().upper()) != (want_ex, want_code):
            continue
        points = _points_of(raw, metric)
        if not any(v is not None for v in points.values()):
            continue          # 该次抓取没带回这个指标(或全是空值),不参与选择
        cands.append((points, str(args.get("period") or "")))
    if not cands:
        return {}, {"period": "", "n": 0, "source": "无匹配抓取"}

    declared = (period or "").strip().lower()
    exact = [c for c in cands if c[1].strip().lower() == declared] if declared else []
    # 命中声明的窗口就用它;否则取非空观测最多的一份(并列时取总点数少的,即最紧凑的窗口)
    points, used_period = max(
        exact or cands,
        key=lambda c: (sum(v is not None for v in c[0].values()), -len(c[0])),
    )
    n = sum(v is not None for v in points.values())
    return points, {"period": used_period, "n": n,
                    "source": f"period={used_period or '未指定'}, n={n}"}
