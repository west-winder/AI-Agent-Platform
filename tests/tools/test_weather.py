"""Weather Tool V1 确定性单元测试，所有 HTTP 请求均由 MockTransport 接管。"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError

from backend.tools.implementations import weather
from backend.tools.infrastructure.contracts import ToolContext
from backend.tools.infrastructure.errors import ToolExecutionError


_GEOCODING_HOST = "geocoding-api.open-meteo.com"
_WEATHER_HOST = "api.open-meteo.com"


@pytest.fixture
def geocoding_payload() -> dict[str, Any]:
    return {
        "results": [
            {
                "name": "武汉市",
                "admin1": "湖北省",
                "country": "中国",
                "country_code": "CN",
                "latitude": 30.59,
                "longitude": 114.3,
                "timezone": "Asia/Shanghai",
            }
        ]
    }


@pytest.fixture
def weather_payload() -> dict[str, Any]:
    return {
        "timezone": "Asia/Shanghai",
        "current": {
            "time": "2026-10-22T14:15",
            "temperature_2m": 25.5,
            "relative_humidity_2m": 70,
            "weather_code": 1,
        },
    }


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch):
    real_client = httpx2.AsyncClient
    clients: list[httpx2.AsyncClient] = []

    def install(
        respond: Callable[[httpx2.Request], httpx2.Response],
    ) -> list[httpx2.Request]:
        requests: list[httpx2.Request] = []

        def transport_handler(request: httpx2.Request) -> httpx2.Response:
            requests.append(request)
            return respond(request)

        def client_factory(**kwargs):
            client = real_client(
                transport=httpx2.MockTransport(transport_handler),
                trust_env=False,
                **kwargs,
            )
            clients.append(client)
            return client

        monkeypatch.setattr(weather.httpx2, "AsyncClient", client_factory)
        return requests

    yield install

    assert all(client.is_closed for client in clients)


def json_responses(
    geocoding: Any,
    forecast: Any,
) -> Callable[[httpx2.Request], httpx2.Response]:
    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "GET"
        if request.url.host == _GEOCODING_HOST:
            assert request.url.path == "/v1/search"
            payload = geocoding
        else:
            assert request.url.host == _WEATHER_HOST
            assert request.url.path == "/v1/forecast"
            payload = forecast
        # 显式允许构造上游 NaN / Infinity 响应，验证工具会拒绝这些无效值。
        return httpx2.Response(
            200,
            content=json.dumps(payload, allow_nan=True).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    return respond


def query_weather(city: str = "武汉") -> dict[str, Any]:
    return asyncio.run(
        weather.get_weather_handler(
            weather.WeatherArgs(city=city),
            ToolContext(user_id=42, conversation_id=7),
        )
    )


def assert_error(
    error: ToolExecutionError,
    *,
    code: str,
    message: str,
    retryable: bool,
) -> None:
    assert error.code == code
    assert error.message == message
    assert str(error) == message
    assert error.retryable is retryable


@pytest.mark.parametrize(
    ("city", "expected"),
    [("  武汉\t", "武汉"), (" London ", "London"), ("x" * 100, "x" * 100)],
)
def test_weather_args_accept_and_trim_city(city: str, expected: str) -> None:
    assert weather.WeatherArgs(city=city).city == expected


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({}, id="missing"),
        pytest.param({"city": ""}, id="empty"),
        pytest.param({"city": " \t\n\u3000"}, id="whitespace"),
        pytest.param({"city": "x" * 101}, id="too-long"),
        pytest.param({"city": None}, id="null"),
        pytest.param({"city": 123}, id="number"),
        pytest.param({"city": ["武汉"]}, id="list"),
    ],
)
def test_weather_args_reject_invalid_city(arguments: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        weather.WeatherArgs.model_validate(arguments)


@pytest.mark.parametrize("city", ["  武汉  ", "London"], ids=["chinese", "english"])
def test_city_query_returns_matched_location_and_current_weather(
    mock_http,
    geocoding_payload: dict[str, Any],
    weather_payload: dict[str, Any],
    city: str,
) -> None:
    location = geocoding_payload["results"][0]
    if city == "London":
        location.update(
            name="伦敦",
            admin1="英格兰",
            country="英国",
            country_code="GB",
            latitude=51.5,
            longitude=-0.12,
            timezone="Europe/London",
        )
        weather_payload["timezone"] = "Europe/London"
    second_match = {**location, "name": "其他同名地点", "latitude": 0.0}
    geocoding_payload["results"].append(second_match)
    requests = mock_http(json_responses(geocoding_payload, weather_payload))

    result = query_weather(city)

    assert result == {
        "requested_city": city.strip(),
        "location": location,
        "current": {
            "time": "2026-10-22T14:15",
            "temperature_c": 25.5,
            "relative_humidity_percent": 70,
            "weather_code": 1,
        },
        "source": "Open-Meteo",
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert len(requests) == 2
    assert dict(requests[0].url.params) == {
        "name": city.strip(),
        "count": "1",
        "language": "zh",
        "format": "json",
    }
    assert dict(requests[1].url.params) == {
        "latitude": str(location["latitude"]),
        "longitude": str(location["longitude"]),
        "current": "temperature_2m,relative_humidity_2m,weather_code",
        "temperature_unit": "celsius",
        "timezone": "auto",
        "timeformat": "iso8601",
    }


def test_current_weather_preserves_negative_temperature_and_zero_values(
    mock_http,
    geocoding_payload: dict[str, Any],
    weather_payload: dict[str, Any],
) -> None:
    weather_payload["current"].update(
        temperature_2m=-3.5,
        relative_humidity_2m=0,
        weather_code=0,
    )
    mock_http(json_responses(geocoding_payload, weather_payload))

    result = query_weather()

    assert result["current"] == {
        "time": "2026-10-22T14:15",
        "temperature_c": -3.5,
        "relative_humidity_percent": 0,
        "weather_code": 0,
    }
    json.dumps(result, allow_nan=False)


def test_optional_location_metadata_can_be_absent(
    mock_http,
    geocoding_payload: dict[str, Any],
    weather_payload: dict[str, Any],
) -> None:
    for field in ("admin1", "country", "country_code", "timezone"):
        del geocoding_payload["results"][0][field]
    mock_http(json_responses(geocoding_payload, weather_payload))

    result = query_weather()

    assert result["location"]["name"] == "武汉市"
    assert result["location"]["admin1"] is None
    assert result["location"]["country"] is None
    assert result["location"]["country_code"] is None
    assert result["location"]["timezone"] == "Asia/Shanghai"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("payload", [{}, {"results": []}], ids=["omitted", "empty"])
def test_no_geocoding_match_is_not_retryable(mock_http, payload: dict[str, Any]) -> None:
    requests = mock_http(json_responses(payload, None))

    with pytest.raises(ToolExecutionError) as raised:
        query_weather()

    assert_error(
        raised.value,
        code="LOCATION_NOT_FOUND",
        message="未找到匹配的城市",
        retryable=False,
    )
    assert len(requests) == 1


@pytest.mark.parametrize("field", ["name", "latitude", "longitude"])
def test_geocoding_missing_required_fields_is_invalid_response(
    mock_http,
    geocoding_payload: dict[str, Any],
    field: str,
) -> None:
    del geocoding_payload["results"][0][field]
    requests = mock_http(json_responses(geocoding_payload, None))

    with pytest.raises(ToolExecutionError) as raised:
        query_weather()

    assert_error(
        raised.value,
        code="WEATHER_INVALID_RESPONSE",
        message="天气服务返回的数据无效",
        retryable=False,
    )
    assert len(requests) == 1


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param([], id="non-object"),
        pytest.param({"results": None}, id="null-results"),
        pytest.param({"results": {}}, id="non-list-results"),
        pytest.param({"results": [None]}, id="non-object-location"),
    ],
)
def test_geocoding_malformed_structure_is_invalid_response(mock_http, payload: Any) -> None:
    requests = mock_http(json_responses(payload, None))

    with pytest.raises(ToolExecutionError) as raised:
        query_weather()

    assert_error(
        raised.value,
        code="WEATHER_INVALID_RESPONSE",
        message="天气服务返回的数据无效",
        retryable=False,
    )
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("name", " "), ("latitude", 91), ("longitude", float("inf"))],
    ids=["blank-name", "latitude-range", "non-finite-longitude"],
)
def test_geocoding_invalid_values_are_rejected(
    mock_http,
    geocoding_payload: dict[str, Any],
    field: str,
    value: Any,
) -> None:
    geocoding_payload["results"][0][field] = value
    requests = mock_http(json_responses(geocoding_payload, None))

    with pytest.raises(ToolExecutionError) as raised:
        query_weather()

    assert_error(
        raised.value,
        code="WEATHER_INVALID_RESPONSE",
        message="天气服务返回的数据无效",
        retryable=False,
    )
    assert len(requests) == 1


@pytest.mark.parametrize(
    "field",
    ["timezone", "current", "time", "temperature_2m", "relative_humidity_2m", "weather_code"],
)
def test_weather_missing_required_fields_is_invalid_response(
    mock_http,
    geocoding_payload: dict[str, Any],
    weather_payload: dict[str, Any],
    field: str,
) -> None:
    container = weather_payload if field in ("timezone", "current") else weather_payload["current"]
    del container[field]
    requests = mock_http(json_responses(geocoding_payload, weather_payload))

    with pytest.raises(ToolExecutionError) as raised:
        query_weather()

    assert_error(
        raised.value,
        code="WEATHER_INVALID_RESPONSE",
        message="天气服务返回的数据无效",
        retryable=False,
    )
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("timezone", " ", id="blank-timezone"),
        pytest.param("current", [], id="non-object-current"),
        pytest.param("time", "not-a-time", id="invalid-time"),
        pytest.param("temperature_2m", None, id="missing-temperature"),
        pytest.param("temperature_2m", "25.5", id="string-temperature"),
        pytest.param("temperature_2m", float("nan"), id="nan-temperature"),
        pytest.param("temperature_2m", float("inf"), id="infinite-temperature"),
        pytest.param("relative_humidity_2m", 101, id="humidity-range"),
        pytest.param("relative_humidity_2m", True, id="boolean-humidity"),
        pytest.param("weather_code", 1.5, id="non-integer-weather-code"),
    ],
)
def test_weather_invalid_values_are_rejected(
    mock_http,
    geocoding_payload: dict[str, Any],
    weather_payload: dict[str, Any],
    field: str,
    value: Any,
) -> None:
    container = weather_payload if field in ("timezone", "current") else weather_payload["current"]
    container[field] = value
    mock_http(json_responses(geocoding_payload, weather_payload))

    with pytest.raises(ToolExecutionError) as raised:
        query_weather()

    assert_error(
        raised.value,
        code="WEATHER_INVALID_RESPONSE",
        message="天气服务返回的数据无效",
        retryable=False,
    )


@pytest.mark.parametrize("stage", [_GEOCODING_HOST, _WEATHER_HOST], ids=["geocoding", "weather"])
@pytest.mark.parametrize("error_type", [httpx2.ReadTimeout, httpx2.ConnectError])
def test_network_failures_are_retryable_without_automatic_retry(
    mock_http,
    geocoding_payload: dict[str, Any],
    stage: str,
    error_type: type[httpx2.RequestError],
) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == stage:
            raise error_type("private transport details", request=request)
        return httpx2.Response(200, json=geocoding_payload)

    requests = mock_http(respond)

    with pytest.raises(ToolExecutionError) as raised:
        query_weather()

    assert_error(
        raised.value,
        code="WEATHER_SERVICE_UNAVAILABLE",
        message="天气服务暂时不可用，请稍后重试",
        retryable=True,
    )
    assert isinstance(raised.value.__cause__, error_type)
    assert len(requests) == (1 if stage == _GEOCODING_HOST else 2)


@pytest.mark.parametrize("stage", [_GEOCODING_HOST, _WEATHER_HOST], ids=["geocoding", "weather"])
@pytest.mark.parametrize(
    ("status", "retryable"),
    [(429, True), (503, True), (408, True), (400, False)],
)
def test_http_status_errors_have_explicit_retry_policy(
    mock_http,
    geocoding_payload: dict[str, Any],
    stage: str,
    status: int,
    retryable: bool,
) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == stage:
            return httpx2.Response(status, json={"reason": "private upstream details"})
        return httpx2.Response(200, json=geocoding_payload)

    requests = mock_http(respond)

    with pytest.raises(ToolExecutionError) as raised:
        query_weather()

    assert_error(
        raised.value,
        code="WEATHER_SERVICE_UNAVAILABLE",
        message="天气服务暂时不可用，请稍后重试" if retryable else "天气服务请求失败",
        retryable=retryable,
    )
    assert isinstance(raised.value.__cause__, httpx2.HTTPStatusError)
    assert len(requests) == (1 if stage == _GEOCODING_HOST else 2)


@pytest.mark.parametrize("stage", [_GEOCODING_HOST, _WEATHER_HOST], ids=["geocoding", "weather"])
@pytest.mark.parametrize("content", [b"not-json", b"\xff"], ids=["invalid-json", "invalid-encoding"])
def test_unparseable_json_is_invalid_response(
    mock_http,
    geocoding_payload: dict[str, Any],
    stage: str,
    content: bytes,
) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == stage:
            return httpx2.Response(200, content=content)
        return httpx2.Response(200, json=geocoding_payload)

    requests = mock_http(respond)

    with pytest.raises(ToolExecutionError) as raised:
        query_weather()

    assert_error(
        raised.value,
        code="WEATHER_INVALID_RESPONSE",
        message="天气服务返回的数据无效",
        retryable=False,
    )
    assert isinstance(raised.value.__cause__, (json.JSONDecodeError, UnicodeDecodeError))
    assert len(requests) == (1 if stage == _GEOCODING_HOST else 2)


def test_response_decoding_failure_is_invalid_response(mock_http) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.DecodingError("private decoder details", request=request)

    requests = mock_http(respond)

    with pytest.raises(ToolExecutionError) as raised:
        query_weather()

    assert_error(
        raised.value,
        code="WEATHER_INVALID_RESPONSE",
        message="天气服务返回的数据无效",
        retryable=False,
    )
    assert len(requests) == 1


def test_programming_error_propagates_unchanged(mock_http) -> None:
    error = RuntimeError("programming bug")

    def respond(request: httpx2.Request) -> httpx2.Response:
        raise error

    requests = mock_http(respond)

    with pytest.raises(RuntimeError) as raised:
        query_weather()

    assert raised.value is error
    assert len(requests) == 1
