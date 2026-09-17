"""MCP 工具包装测试(真实调用需 localhost:8081 的 sacollector-mcp 在线)。

验证标的:SHA:600036(招商银行,实测 2016-2026 数据齐全)。
服务器不可达时跳过——但最后两条是离线的,任何环境都会跑。
"""

import asyncio
import json

import pytest

from mcp_client import init_mcp_client, shutdown_mcp_client

TEST_EXCHANGE, TEST_CODE = "SHA", "600036"


@pytest.fixture(scope="module")
def mcp():
    ok = init_mcp_client("http://localhost:8081")
    if not ok:
        pytest.skip("MCP 服务器不可达,跳过")
    yield
    shutdown_mcp_client()


async def _make():
    from app.mcp_tools import make_mcp_tools

    return await make_mcp_tools()


def _tools():
    tools, cache = asyncio.run(_make())
    return {t.name: t for t in tools}, cache


def _call(tool, args):
    """工具只实现了 async(适配器给的 coroutine),必须走 ainvoke。"""
    return asyncio.run(tool.ainvoke(args))


def test_list_metrics_returns_groups(mcp):
    by_name, cache = _tools()
    out = _call(by_name["list_metrics"], {"exchange": TEST_EXCHANGE, "code": TEST_CODE})
    assert "metrics" in out
    assert cache, "调用结果应写入缓存"


def test_get_data_period(mcp):
    by_name, _ = _tools()
    out = _call(by_name["get_data_period"], {"exchange": TEST_EXCHANGE, "code": TEST_CODE})
    assert "earliest_date" in out


def test_get_financials_compacted_and_cached(mcp):
    from app.mcp_tools import extract_series

    by_name, cache = _tools()
    out = _call(by_name["get_financials"], {
        "exchange": TEST_EXCHANGE, "code": TEST_CODE,
        "metrics": ["roe", "pb", "epsBasic"], "period": "5y",
    })
    parsed = json.loads(out)
    assert "series" in parsed
    assert "roe" in parsed["series"]
    # 原始结果已缓存,且能被 extract_series 提取
    series, meta = extract_series(cache, "roe", TEST_EXCHANGE, TEST_CODE)
    assert len(series) >= 2
    assert all(k < k2 for k, k2 in zip(series, list(series)[1:]))  # 升序
    assert meta["period"] == "5y" and meta["n"] > 0


def test_unknown_code_returns_error_string_not_exception(mcp):
    by_name, _ = _tools()
    out = _call(by_name["get_data_period"], {"exchange": "SHA", "code": "999999"})
    assert isinstance(out, str)
    # 服务器返回的错误文本原样透传给 LLM(英文),不得抛异常
    assert "no data" in out.lower()


def test_tool_descriptions_in_chinese(mcp):
    tools, _ = asyncio.run(_make())
    for t in tools:
        assert t.description and any("一" <= c <= "鿿" for c in t.description)


# ---------------------------------------------------------------------------
# 离线:工具返回值的形状(不需要 MCP 服务器)
# ---------------------------------------------------------------------------

class _FakeAdapterTool:
    """冒充 langchain-mcp-adapters 造出来的工具:ainvoke 回 **content 块**。

    实测(2026-09-16)该适配器 0.3.2 的工具是 content_and_artifact 格式,
    ainvoke() 只回 content,structuredContent 被丢在 artifact 里拿不到。
    """

    name = "get_financials"
    description = "Fetch financial data"
    args_schema = {"type": "object", "properties": {
        "exchange": {"type": "string"}, "code": {"type": "string"}}}

    def __init__(self, result):
        self._result = result

    async def ainvoke(self, kwargs, *a, **k):
        return self._result


def _wrapped(result):
    from app.mcp_tools import _wrap_tool

    cache: dict = {}
    tool = _wrap_tool(_FakeAdapterTool(result), cache)
    out = asyncio.run(tool.ainvoke({"exchange": "ASX", "code": "NAB"}))
    return out, cache


def test_content_blocks_are_unwrapped_for_the_cache():
    """回归:文本块必须还原成 structuredContent 那个 dict。

    只把 list 原样塞进缓存时,07 节点每次运行都死在
    `'list' object has no attribute 'get'`(下游读 raw["data"])。
    """
    from app.mcp_tools import extract_series

    payload = {"code": "NAB", "exchange": "asx",
               "data": {"2026-06-30": {"ratios": {"roe": 0.09908}}}}
    out, cache = _wrapped([{"type": "text", "text": json.dumps(payload)}])

    [cached] = cache.values()
    assert isinstance(cached, dict) and cached["code"] == "NAB"
    series, meta = extract_series(cache, "roe", "ASX", "NAB")
    assert series == {"2026-06-30": 0.09908} and meta["n"] == 1
    assert json.loads(out)["series"]["roe"] == {"2026-06-30": 0.09908}   # LLM 拿到压缩后的


def test_plain_text_block_passes_through():
    """错误消息(非 JSON 文本)原样给 LLM,不猜结构、不抛。"""
    out, cache = _wrapped([{"type": "text", "text": "no data for 999999"}])

    assert out == "no data for 999999"
    assert list(cache.values()) == [{"text": "no data for 999999"}]


# ---------------------------------------------------------------------------
# 离线:取数账本(去重 + 硬额度)
# ---------------------------------------------------------------------------

class _FakeAnyTool:
    """可配置 name / 返回值 / 调用计数的假工具,用于直接验账本。"""

    description = "fake"
    args_schema = {"type": "object", "properties": {
        "exchange": {"type": "string"}, "code": {"type": "string"},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "period": {"type": "string"}}}

    def __init__(self, name: str, result, calls: list | None = None):
        self.name, self._result, self.calls = name, result, calls if calls is not None else []

    async def ainvoke(self, kwargs, *a, **k):
        self.calls.append(kwargs)
        return [{"type": "text", "text": json.dumps(self._result)}]


def _ledger_tool(name: str, budget: int = 2, result: dict | None = None):
    from app.mcp_tools import _CallLedger, _wrap_tool

    tool = _FakeAnyTool(name, result or {"data": {"2026-06-30": {"ratios": {"roe": 1.0}}}})
    cache: dict = {}
    return _wrap_tool(tool, cache, _CallLedger(budget)), tool, cache


PAYLOAD = {"exchange": "ASX", "code": "NAB", "metrics": ["roe"], "period": "5y"}


def test_repeated_call_is_deduped_without_hitting_the_server():
    """实测循环:同一组参数反复抓(5y↔all↔5y…)。第二次起不再执行、也不再回数据。

    上下文是这类循环真正的代价:每次重复都追加 ~2k token 的同一份序列(实测涨到 31k)。
    """
    tool, fake, cache = _ledger_tool("get_financials")

    first = asyncio.run(tool.ainvoke(PAYLOAD))
    second = asyncio.run(tool.ainvoke(dict(PAYLOAD)))

    assert json.loads(first)["series"]["roe"]              # 第一次正常返回数据
    assert "重复调用" in second and "立即输出结构化参数" in second
    assert len(fake.calls) == 1 and len(cache) == 1         # 服务器只被打了一次
    assert "roe" not in second                             # 不再重复塞数据进上下文


def test_financials_budget_blocks_further_distinct_fetches():
    """不同 period 的抓取不算重复:仍受硬额度约束(默认 2 次),第 3 次被拒绝。"""
    tool, fake, _cache = _ledger_tool("get_financials", budget=2)

    for period in ("5y", "all"):
        assert "series" in asyncio.run(tool.ainvoke({**PAYLOAD, "period": period}))
    third = asyncio.run(tool.ainvoke({**PAYLOAD, "period": "2y"}))

    assert "额度用完" in third and "最多 2 次" in third
    assert "by_year" in third                              # 提示它去哪里找逐年数据
    assert len(fake.calls) == 2                            # 被拒的那次没有发出去


def test_other_tools_are_deduped_but_not_budgeted():
    """额度只针对 get_financials:换参数的其他工具调用照常放行,同参数的才去重。"""
    tool, fake, _cache = _ledger_tool("list_metrics")

    assert "metrics" in asyncio.run(tool.ainvoke(dict(PAYLOAD)))
    assert "重复调用" in asyncio.run(tool.ainvoke(dict(PAYLOAD)))
    assert "metrics" in asyncio.run(tool.ainvoke({**PAYLOAD, "code": "CBA"}))

    assert len(fake.calls) == 2


def test_compaction_keeps_the_most_recent_points_and_says_what_is_omitted():
    """截断保留**最近**的点,并写明省掉的是哪一段 —— 省中间年份正是循环的诱因。"""
    from app.mcp_tools import _compact_financials

    dates = [f"20{20 + i // 4:02d}-{(i % 4) * 3 + 3:02d}-30" for i in range(20)]  # 20 个观测
    raw = {"metric_sources": {"roe": "ratios"},
           "data": {d: {"ratios": {"roe": 0.1 + i / 100}} for i, d in enumerate(dates)}}

    out = json.loads(_compact_financials(raw))
    kept = sorted(out["series"]["roe"])

    assert len(kept) == 12
    assert kept == sorted(dates)[-12:]                     # 头部的 8 个点被省略
    assert out["_truncated"]["roe"]["total"] == 20
    assert dates[0] in out["_truncated"]["roe"]["omitted"]
    assert "换 period" in out["_note"] and "by_year" in out["_note"]
