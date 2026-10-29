"""真实 SDK、Weather Handler 和 Runtime 的离线端到端接线验证。"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import httpcore2
import httpx2
import pytest
from openai import AsyncOpenAI

from backend.agent_runtime.bootstrap import build_agent_runtime
from backend.agent_runtime.contracts import AgentRunStatus, LLMMessage
from backend.tools.implementations import weather
from backend.tools.infrastructure.contracts import ToolContext


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("sdk_retries", [0, 1])
@pytest.mark.parametrize("tool_recover", [False, True])
@pytest.mark.parametrize("recovery_error", ["status", "connect", "connect_timeout"])
def test_sdk_runtime_weather_roundtrip_with_mocked_external_io(monkeypatch, legacy_service, recover, sdk_retries, tool_recover, recovery_error) -> None:
    raw_arguments = '{ "city" : "武汉" }'
    requests = []
    weather_payloads = [
        {"results": [{"name": "武汉", "latitude": 30.58, "longitude": 114.27,
                      "admin1": "湖北", "country": "中国", "country_code": "CN"}]},
        {"timezone": "Asia/Shanghai", "current": {"time": "2026-10-25T10:00",
         "temperature_2m": 20.5, "relative_humidity_2m": 65.0, "weather_code": 0}},
    ]
    weather_requests = []
    counts = {weather._GEOCODING_URL: 0, weather._WEATHER_URL: 0}

    def mapped_connection_error():
        from httpx2._transports.default import map_httpcore_exceptions

        core_type = httpcore2.ConnectError if recovery_error == "connect" else httpcore2.ConnectTimeout
        root = ConnectionRefusedError(10061, "temporary refusal") if recovery_error == "connect" else TimeoutError("temporary timeout")
        try:
            with map_httpcore_exceptions():
                raise core_type("transport failure") from root
        except httpx2.RequestError as mapped:
            return mapped

    async def weather_get(client, url, *, params):
        request = client.build_request("GET", url, params=params)
        weather_requests.append(request)
        counts[url] += 1
        if tool_recover and counts[url] == 1:
            if recovery_error != "status":
                raise mapped_connection_error()
            return httpx2.Response(503, request=request, json={})
        return httpx2.Response(200, request=request, json=weather_payloads[0 if url == weather._GEOCODING_URL else 1])

    # 拦截 HTTP 边界，保留真实 Weather 解析、局部重试及 Dispatcher 执行。
    monkeypatch.setattr(httpx2.AsyncClient, "get", weather_get)
    weather_io = AsyncMock(wraps=weather._get_json)
    monkeypatch.setattr(weather, "_get_json", weather_io)

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "test-deepseek"
        assert body["thinking"] == {"type": "disabled"}
        assert body["tools"][0]["function"]["name"] == "get_weather"
        assert "handler" not in body["tools"][0]["function"]
        if recover and 2 <= len(requests) <= 2 + sdk_retries:
            if recovery_error != "status":
                raise mapped_connection_error()
            return httpx2.Response(503, json={"error": {"message": "private-upstream"}})
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "weather-1", "type": "function",
                "function": {"name": "get_weather", "arguments": raw_arguments},
            }]}
            finish_reason = "tool_calls"
        else:
            assert len(requests) == (3 + sdk_retries if recover else 2)
            if recover:
                assert all(body == requests[1] for body in requests[2:])
            assistant, tool = body["messages"][-2:]
            assert assistant["tool_calls"][0]["function"]["arguments"] == raw_arguments
            assert tool["tool_call_id"] == "weather-1"
            data = json.loads(tool["content"])
            assert data["status"] == "success"
            assert data["data"]["location"]["name"] == "武汉"
            assert data["data"]["current"]["temperature_c"] == 20.5
            assert data["data"]["current"]["relative_humidity_percent"] == 65.0
            assert data["data"]["source"] == "Open-Meteo"
            message = {"role": "assistant", "content": "武汉当前气温 20.5°C，相对湿度 65%。"}
            finish_reason = "stop"
        return httpx2.Response(200, json={
            "id": f"completion-{len(requests)}", "object": "chat.completion", "created": 1,
            "model": "test-deepseek", "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        })

    async def scenario():
        async with AsyncOpenAI(
            api_key="test-not-a-secret", base_url="https://example.invalid/v1", max_retries=sdk_retries,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        ) as client:
            sdk_wait = AsyncMock()
            monkeypatch.setattr(client, "_sleep_for_retry", sdk_wait)
            translator = Mock(wraps=legacy_service._translate_external_error)
            runtime = build_agent_runtime(client=client, model="test-deepseek",
                                          overall_timeout_seconds=2, translate_error=translator)
            result = await runtime.run([LLMMessage("user", "武汉的气温和湿度是多少？")], ToolContext())
            assert result.status is AgentRunStatus.COMPLETED
            assert result.iterations == 2
            assert result.final_answer == "武汉当前气温 20.5°C，相对湿度 65%。"
            assert [message.role for message in result.messages] == ["user", "assistant", "tool", "assistant"]
            assert client.max_retries == sdk_retries
            assert translator.call_count == (1 if recover else 0)
            assert sdk_wait.await_count == (sdk_retries if recover else 0)

    asyncio.run(scenario())
    assert len(requests) == (3 + sdk_retries if recover else 2)
    assert weather_io.await_count == 2
    assert len(weather_requests) == (4 if tool_recover else 2)
    assert list(counts.values()) == ([2, 2] if tool_recover else [1, 1])
    assert all(r.extensions["timeout"]["read"] == 10 for r in weather_requests)
    assert weather_io.await_args_list[0].args[2]["name"] == "武汉"
    assert weather_io.await_args_list[1].args[2]["latitude"] == 30.58
