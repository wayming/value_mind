"""MCP 工具包装测试(需要 localhost:8081 的 sacollector-mcp 在线)。

验证标的:SHA:600036(招商银行,实测 2016-2026 数据齐全)。
服务器不可达时跳过。
"""

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


def test_list_metrics_returns_groups(mcp):
    from app.mcp_tools import make_mcp_tools

    tools, cache = make_mcp_tools()
    by_name = {t.name: t for t in tools}
    out = by_name["list_metrics"].invoke({"exchange": TEST_EXCHANGE, "code": TEST_CODE})
    assert "metrics" in out
    assert cache, "调用结果应写入缓存"


def test_get_data_period(mcp):
    from app.mcp_tools import make_mcp_tools

    tools, _ = make_mcp_tools()
    by_name = {t.name: t for t in tools}
    out = by_name["get_data_period"].invoke({"exchange": TEST_EXCHANGE, "code": TEST_CODE})
    assert "earliest_date" in out


def test_get_financials_compacted_and_cached(mcp):
    from app.mcp_tools import extract_series, make_mcp_tools

    tools, cache = make_mcp_tools()
    by_name = {t.name: t for t in tools}
    out = by_name["get_financials"].invoke({
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
    from app.mcp_tools import make_mcp_tools

    tools, _ = make_mcp_tools()
    by_name = {t.name: t for t in tools}
    out = by_name["get_data_period"].invoke({"exchange": "SHA", "code": "999999"})
    assert isinstance(out, str)
    # 服务器返回的错误文本原样透传给 LLM(英文),不得抛异常
    assert "no data" in out.lower()


def test_tool_descriptions_in_chinese(mcp):
    from app.mcp_tools import make_mcp_tools

    tools, _ = make_mcp_tools()
    for t in tools:
        assert t.description and any("一" <= c <= "鿿" for c in t.description)
