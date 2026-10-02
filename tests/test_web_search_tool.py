"""Tests for multi-provider web search."""

import json

import httpx
import pytest

from roboclaw.agent.tools.web import WebSearchTool
from roboclaw.config.schema import WebSearchConfig


def _tool(provider: str = "brave", api_key: str = "", base_url: str = "") -> WebSearchTool:
    return WebSearchTool(config=WebSearchConfig(provider=provider, api_key=api_key, base_url=base_url))


def _response(status: int = 200, json: dict | None = None) -> httpx.Response:
    """Build a mock httpx.Response with a dummy request attached."""
    r = httpx.Response(status, json=json)
    r._request = httpx.Request("GET", "https://mock")
    return r


_YOUCOM_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2025-06-18",
        "serverInfo": {"name": "You.com", "version": "4.0.2"},
    },
}


def _youcom_result(web: list[dict]) -> httpx.Response:
    """Build the tools/call reply with a plain-JSON JSON-RPC wrapper."""
    payload = {"results": {"web": web}}
    return _response(json={
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
    })


def _sse(*messages: dict) -> httpx.Response:
    """Build a mock text/event-stream httpx.Response carrying JSON-RPC messages."""
    body = "".join(f"event: message\ndata: {json.dumps(m)}\n\n" for m in messages)
    r = httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
    r._request = httpx.Request("POST", "https://mock")
    return r


def _youcom_post(mock_search):
    """Route JSON-RPC methods to canned You.com MCP responses.

    mock_search(**kwargs) receives the httpx kwargs of each client.post call and
    returns the tools/call response.
    """

    async def post(self, url, **kw):
        body = kw["json"]
        if body["method"] == "initialize":
            return _response(json=_YOUCOM_INIT)
        if body["method"] == "notifications/initialized":
            return _response(status=202)
        assert body["method"] == "tools/call"
        assert body["params"]["name"] == "you-search"
        return mock_search(url=url, **kw)

    return post


@pytest.mark.asyncio
async def test_youcom_search_keyless(monkeypatch):
    seen: list[dict] = []

    def mock_search(url, **kw):
        seen.append({"url": str(url), "headers": kw["headers"], "args": kw["json"]["params"]["arguments"]})
        progress = {"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info"}}
        result = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"content": [{"type": "text", "text": json.dumps(
                {"results": {"web": [{"url": "https://example.com", "title": "You.com Result",
                                      "description": "A snippet", "contents": {"highlights": ["hl"]}}]}}
            )}]},
        }
        return _sse(progress, result)

    monkeypatch.setattr(httpx.AsyncClient, "post", _youcom_post(mock_search))
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    monkeypatch.delenv("YOUCOM_MCP_URL", raising=False)

    tool = _tool(provider="youcom")
    result = await tool.execute(query="embodied ai", count=3)
    assert "You.com Result" in result
    assert "https://example.com" in result
    assert "A snippet" in result
    assert all("api.you.com/mcp?profile=free" in s["url"] for s in seen)
    assert all("Authorization" not in s["headers"] for s in seen)
    assert seen[-1]["args"]["query"] == "embodied ai"
    assert seen[-1]["args"]["count"] == 3


@pytest.mark.asyncio
async def test_youcom_search_plain_json(monkeypatch):
    def mock_search(url, **kw):
        return _youcom_result([{"url": "https://example.com/plain", "title": "Plain JSON",
                                "description": "From JSON"}])

    monkeypatch.setattr(httpx.AsyncClient, "post", _youcom_post(mock_search))
    tool = _tool(provider="youcom")
    result = await tool.execute(query="test")
    assert "Plain JSON" in result
    assert "https://example.com/plain" in result


@pytest.mark.asyncio
async def test_youcom_search_with_api_key(monkeypatch):
    seen: dict = {}

    def mock_search(url, **kw):
        seen["url"] = str(url)
        seen["headers"] = kw["headers"]
        return _youcom_result([{"url": "https://example.com/authed", "title": "Authed",
                                "description": "With key"}])

    monkeypatch.setattr(httpx.AsyncClient, "post", _youcom_post(mock_search))
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    monkeypatch.delenv("YOUCOM_MCP_URL", raising=False)

    tool = _tool(provider="youcom", api_key="ydc-key")
    result = await tool.execute(query="test")
    assert "Authed" in result
    assert seen["url"] == "https://api.you.com/mcp"
    assert seen["headers"]["Authorization"] == "Bearer ydc-key"


@pytest.mark.asyncio
async def test_youcom_env_var_key_and_url_override(monkeypatch):
    seen: dict = {}

    def mock_search(url, **kw):
        seen["url"] = str(url)
        seen["headers"] = kw["headers"]
        return _youcom_result([{"url": "https://example.com/env", "title": "Env", "description": "ok"}])

    monkeypatch.setattr(httpx.AsyncClient, "post", _youcom_post(mock_search))
    monkeypatch.setenv("YDC_API_KEY", "env-key")
    monkeypatch.setenv("YOUCOM_MCP_URL", "https://mcp.example.com/you")

    tool = _tool(provider="youcom")
    result = await tool.execute(query="test")
    assert "Env" in result
    assert seen["url"] == "https://mcp.example.com/you"
    assert seen["headers"]["Authorization"] == "Bearer env-key"


@pytest.mark.asyncio
async def test_youcom_description_fallback_to_highlight(monkeypatch):
    def mock_search(url, **kw):
        return _youcom_result([{"url": "https://example.com", "title": "Highlight Result",
                                "description": "", "contents": {"highlights": ["First highlight", "Second"]}}])

    monkeypatch.setattr(httpx.AsyncClient, "post", _youcom_post(mock_search))
    tool = _tool(provider="youcom")
    result = await tool.execute(query="test")
    assert "Highlight Result" in result
    assert "First highlight" in result
    assert "Second" not in result


@pytest.mark.asyncio
async def test_youcom_jsonrpc_error_fails_safe(monkeypatch):
    def mock_search(url, **kw):
        return _response(json={"jsonrpc": "2.0", "id": 2,
                               "error": {"code": -32000, "message": "quota exceeded"}})

    monkeypatch.setattr(httpx.AsyncClient, "post", _youcom_post(mock_search))
    tool = _tool(provider="youcom")
    result = await tool.execute(query="test")
    assert result.startswith("Error: You.com search failed")
    assert "quota exceeded" in result


@pytest.mark.asyncio
async def test_youcom_http_error_fails_safe(monkeypatch):
    def mock_search(url, **kw):
        return _response(status=500)

    monkeypatch.setattr(httpx.AsyncClient, "post", _youcom_post(mock_search))
    tool = _tool(provider="youcom")
    result = await tool.execute(query="test")
    assert result.startswith("Error: You.com search failed")


@pytest.mark.asyncio
async def test_youcom_session_id_passthrough(monkeypatch):
    sessions: list[str] = []

    async def post(self, url, **kw):
        body = kw["json"]
        if body["method"] == "initialize":
            r = _response(json=_YOUCOM_INIT)
            r.headers["mcp-session-id"] = "sess-123"
            return r
        if body["method"] == "notifications/initialized":
            assert kw["headers"].get("Mcp-Session-Id") == "sess-123"
            return _response(status=202)
        sessions.append(kw["headers"].get("Mcp-Session-Id", ""))
        return _youcom_result([{"url": "https://example.com", "title": "Session", "description": "ok"}])

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    tool = _tool(provider="youcom")
    result = await tool.execute(query="test")
    assert "Session" in result
    assert sessions == ["sess-123"]


@pytest.mark.asyncio
async def test_brave_search(monkeypatch):
    async def mock_get(self, url, **kw):
        assert "brave" in url
        assert kw["headers"]["X-Subscription-Token"] == "brave-key"
        return _response(json={
            "web": {"results": [{"title": "RoboClaw", "url": "https://example.com", "description": "AI assistant"}]}
        })

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    tool = _tool(provider="brave", api_key="brave-key")
    result = await tool.execute(query="roboclaw", count=1)
    assert "RoboClaw" in result
    assert "https://example.com" in result


@pytest.mark.asyncio
async def test_tavily_search(monkeypatch):
    async def mock_post(self, url, **kw):
        assert "tavily" in url
        assert kw["headers"]["Authorization"] == "Bearer tavily-key"
        return _response(json={
            "results": [{"title": "OpenClaw", "url": "https://openclaw.io", "content": "Framework"}]
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    tool = _tool(provider="tavily", api_key="tavily-key")
    result = await tool.execute(query="openclaw")
    assert "OpenClaw" in result
    assert "https://openclaw.io" in result


@pytest.mark.asyncio
async def test_searxng_search(monkeypatch):
    async def mock_get(self, url, **kw):
        assert "searx.example" in url
        return _response(json={
            "results": [{"title": "Result", "url": "https://example.com", "content": "SearXNG result"}]
        })

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    tool = _tool(provider="searxng", base_url="https://searx.example")
    result = await tool.execute(query="test")
    assert "Result" in result


@pytest.mark.asyncio
async def test_duckduckgo_search(monkeypatch):
    class MockDDGS:
        def __init__(self, **kw):
            pass

        def text(self, query, max_results=5):
            return [{"title": "DDG Result", "href": "https://ddg.example", "body": "From DuckDuckGo"}]

    monkeypatch.setattr("roboclaw.agent.tools.web.DDGS", MockDDGS, raising=False)
    import roboclaw.agent.tools.web as web_mod
    monkeypatch.setattr(web_mod, "DDGS", MockDDGS, raising=False)

    from ddgs import DDGS
    monkeypatch.setattr("ddgs.DDGS", MockDDGS)

    tool = _tool(provider="duckduckgo")
    result = await tool.execute(query="hello")
    assert "DDG Result" in result


@pytest.mark.asyncio
async def test_brave_fallback_to_duckduckgo_when_no_key(monkeypatch):
    class MockDDGS:
        def __init__(self, **kw):
            pass

        def text(self, query, max_results=5):
            return [{"title": "Fallback", "href": "https://ddg.example", "body": "DuckDuckGo fallback"}]

    monkeypatch.setattr("ddgs.DDGS", MockDDGS)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    tool = _tool(provider="brave", api_key="")
    result = await tool.execute(query="test")
    assert "Fallback" in result


@pytest.mark.asyncio
async def test_jina_search(monkeypatch):
    async def mock_get(self, url, **kw):
        assert "s.jina.ai" in str(url)
        assert kw["headers"]["Authorization"] == "Bearer jina-key"
        return _response(json={
            "data": [{"title": "Jina Result", "url": "https://jina.ai", "content": "AI search"}]
        })

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    tool = _tool(provider="jina", api_key="jina-key")
    result = await tool.execute(query="test")
    assert "Jina Result" in result
    assert "https://jina.ai" in result


@pytest.mark.asyncio
async def test_unknown_provider():
    tool = _tool(provider="unknown")
    result = await tool.execute(query="test")
    assert "unknown" in result
    assert "Error" in result


@pytest.mark.asyncio
async def test_default_provider_is_brave(monkeypatch):
    async def mock_get(self, url, **kw):
        assert "brave" in url
        return _response(json={"web": {"results": []}})

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    tool = _tool(provider="", api_key="test-key")
    result = await tool.execute(query="test")
    assert "No results" in result


@pytest.mark.asyncio
async def test_searxng_no_base_url_falls_back(monkeypatch):
    class MockDDGS:
        def __init__(self, **kw):
            pass

        def text(self, query, max_results=5):
            return [{"title": "Fallback", "href": "https://ddg.example", "body": "fallback"}]

    monkeypatch.setattr("ddgs.DDGS", MockDDGS)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)

    tool = _tool(provider="searxng", base_url="")
    result = await tool.execute(query="test")
    assert "Fallback" in result


@pytest.mark.asyncio
async def test_searxng_invalid_url():
    tool = _tool(provider="searxng", base_url="not-a-url")
    result = await tool.execute(query="test")
    assert "Error" in result
