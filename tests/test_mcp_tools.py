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
