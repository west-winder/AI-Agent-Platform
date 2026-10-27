"""真实 SDK、Weather Handler 和 Runtime 的离线端到端接线验证。"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import httpx2
from openai import AsyncOpenAI

from backend.agent_runtime.bootstrap import build_agent_runtime
from backend.agent_runtime.contracts import AgentRunStatus, LLMMessage
from backend.tools.implementations import weather
from backend.tools.infrastructure.contracts import ToolContext


def test_sdk_runtime_weather_roundtrip_with_mocked_external_io(monkeypatch) -> None:
    raw_arguments = '{ "city" : "武汉" }'
    requests = []
    weather_io = AsyncMock(side_effect=[
        {"results": [{"name": "武汉", "latitude": 30.58, "longitude": 114.27,
                      "admin1": "湖北", "country": "中国", "country_code": "CN"}]},
        {"timezone": "Asia/Shanghai", "current": {"time": "2026-10-25T10:00",
         "temperature_2m": 20.5, "relative_humidity_2m": 65.0, "weather_code": 0}},
    ])
    monkeypatch.setattr(weather, "_get_json", weather_io)

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "test-deepseek"
        assert body["thinking"] == {"type": "disabled"}
        assert body["tools"][0]["function"]["name"] == "get_weather"
        assert "handler" not in body["tools"][0]["function"]
        if len(requests) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "weather-1", "type": "function",
                "function": {"name": "get_weather", "arguments": raw_arguments},
            }]}
            finish_reason = "tool_calls"
        else:
            assert len(requests) == 2
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
            api_key="test-not-a-secret", base_url="https://example.invalid/v1", max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        ) as client:
            translator = Mock()
            runtime = build_agent_runtime(client=client, model="test-deepseek",
                                          overall_timeout_seconds=2, translate_error=translator)
            result = await runtime.run([LLMMessage("user", "武汉的气温和湿度是多少？")], ToolContext())
            assert result.status is AgentRunStatus.COMPLETED
            assert result.iterations == 2
            assert result.final_answer == "武汉当前气温 20.5°C，相对湿度 65%。"
            assert [message.role for message in result.messages] == ["user", "assistant", "tool", "assistant"]
            translator.assert_not_called()

    asyncio.run(scenario())
    assert len(requests) == 2
    assert weather_io.await_count == 2
    assert weather_io.await_args_list[0].args[2]["name"] == "武汉"
    assert weather_io.await_args_list[1].args[2]["latitude"] == 30.58
