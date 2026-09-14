"""MCP HTTP client for the sacollector MCP server.

Communicates via the MCP streamable HTTP protocol (SSE-based JSON-RPC).
Maintains a persistent session across tool calls.
"""

import json
import logging
import re
import httpx

logger = logging.getLogger(f"backend.{__name__}")


class McpClient:
    """Persistent MCP client that initializes once and reuses the session."""

    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None
        self._request_id = 0
        self._client = httpx.Client(timeout=30.0)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialize the MCP session. Call once on startup."""
        try:
            resp = self._client.post(
                f"{self.base_url}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "stock-trading-agent", "version": "1.0.0"},
                    },
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            # Extract session ID from response headers
            sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
            if not sid:
                logger.error("MCP initialize: no Mcp-Session-Id in response headers")
                return False
            self.session_id = sid

            # Parse SSE body to confirm initialize succeeded
            body = self._parse_sse(resp.text)
            if body and "result" in body:
                server_info = body["result"].get("serverInfo", {})
                logger.info(
                    f"MCP connected: {server_info.get('name', '?')} "
                    f"v{server_info.get('version', '?')}, session={self.session_id[:12]}..."
                )
            else:
                logger.warning(f"MCP initialize response unexpected: {resp.text[:200]}")

            # Send initialized notification
            self._send_notification("notifications/initialized")
            return True

        except httpx.RequestError as e:
            logger.error(f"MCP connect failed: {e}")
            return False

    def disconnect(self):
        """Close the HTTP client."""
        self._client.close()
        self.session_id = None

    def reconnect(self) -> bool:
        """Re-establish the MCP session."""
        self.disconnect()
        self._client = httpx.Client(timeout=30.0)
        self._request_id = 0
        return self.connect()

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict) -> dict | None:
        """Call an MCP tool and return the structuredContent.

        Auto-reconnects once if the session appears dead.
        """
        if not self.session_id:
            logger.warning(f"MCP session lost — attempting reconnect for {name}")
            if not self.reconnect():
                return None

        try:
            resp = self._client.post(
                f"{self.base_url}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": self.session_id,
                },
            )
            body = self._parse_sse(resp.text)
            if body is None:
                logger.error(f"MCP tool {name}: empty or invalid response")
                return None
            if "error" in body:
                logger.error(f"MCP tool {name} error: {body['error']}")
                return None
            result = body.get("result", {})
            # structuredContent is the typed output; fall back to content[0].text
            if "structuredContent" in result:
                return result["structuredContent"]
            content = result.get("content", [])
            if content and isinstance(content[0], dict) and "text" in content[0]:
                try:
                    return json.loads(content[0]["text"])
                except json.JSONDecodeError:
                    return {"text": content[0]["text"]}
            return result
        except httpx.RequestError as e:
            logger.error(f"MCP tool {name} request failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Convenience methods for the three sacollector tools
    # ------------------------------------------------------------------

    def list_metrics(self, exchange: str, code: str) -> dict | None:
        """List available financial metrics for a stock."""
        return self.call_tool("list_metrics", {"exchange": exchange, "code": code})

    def get_data_period(self, exchange: str, code: str) -> dict | None:
        """Get earliest/latest date range for a stock."""
        return self.call_tool("get_data_period", {"exchange": exchange, "code": code})

    def get_financials(self, exchange: str, code: str, metrics: list[str], period: str = "2y") -> dict | None:
        """Fetch financial data for specified metrics over a period."""
        return self.call_tool(
            "get_financials",
            {"exchange": exchange, "code": code, "metrics": metrics, "period": period},
        )

    # ------------------------------------------------------------------
    # Tool definitions (OpenAI function-calling format)
    # ------------------------------------------------------------------

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """Return tool definitions in OpenAI function-calling format.

        These are registered with the LLM so it can decide when to call them.
        The actual execution goes through call_tool().
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_metrics",
                    "description": "列出某只股票所有可用的财务指标，按报表类型分组（income-statement, balance-sheet, cash-flow-statement, ratios）。在需要了解有哪些数据可用时调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "exchange": {
                                "type": "string",
                                "description": "交易所代码，如 ASX, HKG, NASDAQ, NYSE, SHA, SHE",
                            },
                            "code": {
                                "type": "string",
                                "description": "股票代码，如 0700 (腾讯), MGX, AAPL",
                            },
                        },
                        "required": ["exchange", "code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_data_period",
                    "description": "获取某只股票财务数据的最早和最晚日期范围。在需要了解数据覆盖的时间跨度时调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "exchange": {
                                "type": "string",
                                "description": "交易所代码，如 ASX, HKG, NASDAQ, NYSE, SHA, SHE",
                            },
                            "code": {
                                "type": "string",
                                "description": "股票代码，如 0700 (腾讯), MGX, AAPL",
                            },
                        },
                        "required": ["exchange", "code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_financials",
                    "description": "获取股票的财务数据。返回指定指标在指定时间范围内的历史数据，包括收入(revenue)、净利润(netinc)、EPS、PE、PB、利润率等。在分析新闻对股价影响时，用此工具获取实际财务数据作为参考。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "exchange": {
                                "type": "string",
                                "description": "交易所代码，如 ASX, HKG, NASDAQ, NYSE, SHA, SHE",
                            },
                            "code": {
                                "type": "string",
                                "description": "股票代码，如 0700 (腾讯), MGX, AAPL",
                            },
                            "metrics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "要获取的指标列表，如 ['revenue', 'netinc', 'epsBasic', 'pe', 'pb', 'operatingMargin', 'profitMargin']",
                            },
                            "period": {
                                "type": "string",
                                "enum": ["1y", "2y", "5y", "all"],
                                "description": "时间范围：1y(一年), 2y(两年), 5y(五年), all(全部)",
                            },
                        },
                        "required": ["exchange", "code", "metrics", "period"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_notification(self, method: str) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            if self.session_id:
                headers["Mcp-Session-Id"] = self.session_id
            self._client.post(
                f"{self.base_url}/mcp",
                json={"jsonrpc": "2.0", "method": method},
                headers=headers,
            )
        except Exception:
            pass  # Notifications are best-effort

    @staticmethod
    def _parse_sse(text: str) -> dict | None:
        """Parse an SSE 'data:' line into a JSON object."""
        if not text:
            return None
        match = re.search(r"data:\s*(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        # Fallback: try parsing the whole body as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_mcp_client: McpClient | None = None


def init_mcp_client(base_url: str = "") -> bool:
    """Initialize the global MCP client. Call once on startup."""
    global _mcp_client
    url = base_url or "http://localhost:8081"
    _mcp_client = McpClient(url)
    if _mcp_client.connect():
        return True
    _mcp_client = None
    return False


def get_mcp_client() -> McpClient | None:
    """Get the global MCP client singleton. Returns None if not initialized."""
    global _mcp_client
    return _mcp_client


def shutdown_mcp_client():
    """Disconnect the global MCP client."""
    global _mcp_client
    if _mcp_client:
        _mcp_client.disconnect()
        _mcp_client = None


def get_tool_definitions() -> list[dict]:
    """Return tool definitions in OpenAI function-calling format.

    These are static — no MCP connection needed to get the definitions.
    Only tool *execution* requires an active MCP session.
    """
    return McpClient.get_tool_definitions()
