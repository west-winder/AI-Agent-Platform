"""Weather Tool V1 确定性单元测试，所有 HTTP 请求均由 MockTransport 接管。"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpcore2
import httpx2
import pytest
from pydantic import ValidationError

from backend.tools.implementations import weather
from backend.tools.infrastructure.contracts import ToolContext
from backend.tools.infrastructure.errors import ToolExecutionError


_GEOCODING_HOST = "geocoding-api.open-meteo.com"
_WEATHER_HOST = "api.open-meteo.com"


def mapped_core_connection(core_type, root_kind):
    """使用真实依赖的异常映射，保留 core 与底层异常两级原因。"""
    import ssl
    from httpx2._transports.default import map_httpcore_exceptions

    roots = {
        "refused": ConnectionRefusedError(10061, "temporary refusal"),
        "timeout": TimeoutError("temporary timeout"),
        "tls": ssl.SSLCertVerificationError("certificate"),
        "permission": PermissionError(13, "permission"),
        "bug": AttributeError("bug"),
        "unknown": OSError("unknown"),
    }
    root = roots.get(root_kind)
    if root_kind in ("aggregate_refused", "aggregate_bug"):
        root = OSError("All connection attempts failed")
        root.__cause__ = roots["refused" if root_kind == "aggregate_refused" else "bug"]
    core = core_type("transport failure")
    core.__cause__ = core if root_kind == "cycle" else root
    try:
        with map_httpcore_exceptions():
            raise core
    except httpx2.RequestError as mapped:
        return mapped


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
def test_network_error_metadata_does_not_authorize_every_error_for_retry(
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
    attempts = 2 if error_type is httpx2.ConnectError else 1
    assert len(requests) == attempts + (0 if stage == _GEOCODING_HOST else 1)


@pytest.mark.parametrize("stage", [_GEOCODING_HOST, _WEATHER_HOST], ids=["geocoding", "weather"])
@pytest.mark.parametrize(
    ("status", "retryable"),
    [(429, True), (500, True), (502, True), (503, True), (504, True), (408, True),
     (501, True), (505, True), (599, True),
     (400, False), (401, False), (403, False), (409, False), (422, False)],
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
    attempts = 2 if status in (408, 500, 502, 503, 504) else 1
    assert len(requests) == attempts + (0 if stage == _GEOCODING_HOST else 1)


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


@pytest.mark.parametrize("stage", [_GEOCODING_HOST, _WEATHER_HOST])
@pytest.mark.parametrize("failure", ["connect", "connect_timeout", 408, 503])
def test_authorized_query_recovers_once_and_keeps_timeout(
    mock_http, geocoding_payload, weather_payload, stage, failure,
) -> None:
    failures = []
    success = json_responses(geocoding_payload, weather_payload)

    def respond(request):
        if request.url.host == stage and not failures:
            failures.append(request)
            if failure == "connect":
                raise httpx2.ConnectError("temporary", request=request)
            if failure == "connect_timeout":
                raise httpx2.ConnectTimeout("temporary", request=request)
            return httpx2.Response(failure, json={})
        return success(request)

    requests = mock_http(respond)
    result = query_weather()
    assert result["current"]["temperature_c"] == 25.5
    assert len(requests) == 3
    repeated = [r for r in requests if r.url.host == stage]
    assert len(repeated) == 2
    assert repeated[0].url == repeated[1].url
    assert all(r.method == "GET" for r in requests)
    assert all(r.extensions["timeout"] == dict.fromkeys(("connect", "read", "write", "pool"), 10.0) for r in requests)


@pytest.mark.parametrize("stage", [_GEOCODING_HOST, _WEATHER_HOST])
@pytest.mark.parametrize("error_type", [httpx2.ConnectTimeout, httpx2.ReadTimeout, httpx2.WriteTimeout, httpx2.PoolTimeout])
def test_timeouts_have_finite_attempts_and_preserve_final_cause(mock_http, geocoding_payload, stage, error_type) -> None:
    failures = []

    def respond(request):
        if request.url.host == stage:
            error = error_type(f"timeout-{len(failures)}", request=request)
            failures.append(error)
            raise error
        return httpx2.Response(200, json=geocoding_payload)

    requests = mock_http(respond)
    with pytest.raises(ToolExecutionError) as raised:
        query_weather()
    expected = 2 if error_type is httpx2.ConnectTimeout else 1
    assert len(failures) == expected
    assert raised.value.__cause__ is failures[-1]
    assert raised.value.code == "WEATHER_SERVICE_UNAVAILABLE"
    assert raised.value.retryable is True
    assert all(r.extensions["timeout"]["connect"] == 10 for r in requests)
    assert all(r.extensions["timeout"]["read"] == 10 for r in requests)


@pytest.mark.parametrize("url,authorize", [(weather._GEOCODING_URL, None), (weather._GEOCODING_URL, False),
                                         ("https://unapproved.invalid/query", True)])
def test_retry_requires_both_explicit_authorization_and_reviewed_path(url, authorize) -> None:
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(503, json={})

    async def scenario():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond), timeout=10) as client:
            kwargs = {} if authorize is None else {"allow_retry": authorize}
            with pytest.raises(ToolExecutionError) as raised:
                await weather._get_json(client, url, {}, **kwargs)
            assert raised.value.retryable is True

    asyncio.run(scenario())
    assert len(requests) == 1


def test_both_query_paths_have_independent_two_attempt_limits(mock_http, geocoding_payload, weather_payload) -> None:
    counts = {_GEOCODING_HOST: 0, _WEATHER_HOST: 0}
    success = json_responses(geocoding_payload, weather_payload)

    def respond(request):
        counts[request.url.host] += 1
        if counts[request.url.host] == 1:
            return httpx2.Response(503, json={})
        return success(request)

    requests = mock_http(respond)
    assert query_weather()["source"] == "Open-Meteo"
    assert list(counts.values()) == [2, 2]
    assert len(requests) == 4


@pytest.mark.parametrize("root", [AttributeError("bug"), RuntimeError("bug")])
def test_wrapped_unknown_connection_failure_is_not_retried(mock_http, root) -> None:
    error = httpx2.ConnectError("connection")
    error.__cause__ = root

    def respond(request):
        raise error

    requests = mock_http(respond)
    with pytest.raises(ToolExecutionError) as raised:
        query_weather()
    assert raised.value.__cause__ is error
    assert raised.value.__cause__.__cause__ is root
    assert len(requests) == 1


def test_second_failure_controls_final_error_without_third_attempt(mock_http) -> None:
    errors = []

    def respond(request):
        status = 503 if not errors else 403
        errors.append(status)
        return httpx2.Response(status, json={})

    requests = mock_http(respond)
    with pytest.raises(ToolExecutionError) as raised:
        query_weather()
    assert errors == [503, 403]
    assert len(requests) == 2
    assert raised.value.retryable is False
    assert raised.value.__cause__.response.status_code == 403


@pytest.mark.parametrize("stage", [_GEOCODING_HOST, _WEATHER_HOST])
@pytest.mark.parametrize("attempt", [1, 2])
@pytest.mark.parametrize("failure", ["status", "core"])
def test_weather_active_cancellation_is_not_translated_or_retried(monkeypatch, geocoding_payload, stage, attempt, failure) -> None:
    async def scenario():
        entered = asyncio.Event()
        requests = []
        stage_calls = []

        async def respond(request):
            requests.append(request)
            if request.url.host != stage:
                return httpx2.Response(200, json=geocoding_payload)
            stage_calls.append(request)
            if len(stage_calls) < attempt:
                if failure == "core":
                    raise mapped_core_connection(httpcore2.ConnectError, "refused")
                return httpx2.Response(503, json={})
            entered.set()
            await asyncio.Event().wait()

        real_client = httpx2.AsyncClient
        clients = []

        def factory(**kwargs):
            client = real_client(transport=httpx2.MockTransport(respond), trust_env=False, **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(weather.httpx2, "AsyncClient", factory)
        task = asyncio.create_task(weather.get_weather_handler(weather.WeatherArgs(city="武汉"), ToolContext()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(stage_calls) == attempt
        assert len(requests) == attempt + (stage == _WEATHER_HOST)
        assert all(c.is_closed for c in clients)

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", [_GEOCODING_HOST, _WEATHER_HOST])
@pytest.mark.parametrize("core_type,root", [(httpcore2.ConnectError, "refused"),
                                         (httpcore2.ConnectTimeout, "timeout"),
                                         (httpcore2.ConnectError, "aggregate_refused")])
@pytest.mark.parametrize("recover", [True, False])
def test_core_connection_chain_recovers_or_preserves_final_failure(
    mock_http, geocoding_payload, weather_payload, stage, core_type, root, recover,
) -> None:
    failures = []
    success = json_responses(geocoding_payload, weather_payload)

    def respond(request):
        if request.url.host == stage and (not recover or not failures):
            error = mapped_core_connection(core_type, root)
            failures.append(error)
            raise error
        return success(request)

    requests = mock_http(respond)
    if recover:
        assert query_weather()["current"]["temperature_c"] == 25.5
        assert len(failures) == 1
    else:
        with pytest.raises(ToolExecutionError) as raised:
            query_weather()
        assert raised.value.__cause__ is failures[-1]
        assert raised.value.retryable is True
        assert len(failures) == 2
    assert sum(r.url.host == stage for r in requests) == 2
    assert isinstance(failures[0].__cause__, core_type)
    assert all(r.extensions["timeout"]["read"] == 10 for r in requests)


@pytest.mark.parametrize("core_type", [httpcore2.ConnectError, httpcore2.ConnectTimeout])
@pytest.mark.parametrize("root", ["tls", "permission", "bug", "unknown", "missing", "cycle", "aggregate_bug"])
def test_core_connection_chain_rejects_unsafe_or_unknown_causes(mock_http, core_type, root) -> None:
    error = mapped_core_connection(core_type, root)

    def respond(request):
        raise error

    requests = mock_http(respond)
    with pytest.raises(ToolExecutionError) as raised:
        query_weather()
    assert raised.value.__cause__ is error
    assert len(requests) == 1


@pytest.mark.parametrize("core_type,root", [(httpcore2.ConnectError, "refused"), (httpcore2.ConnectTimeout, "timeout")])
@pytest.mark.parametrize("authorize", [None, False])
def test_core_connection_chain_does_not_grant_tool_authorization(core_type, root, authorize) -> None:
    requests = []

    def respond(request):
        requests.append(request)
        raise mapped_core_connection(core_type, root)

    async def scenario():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond), timeout=10) as client:
            kwargs = {} if authorize is None else {"allow_retry": authorize}
            with pytest.raises(ToolExecutionError):
                await weather._get_json(client, weather._GEOCODING_URL, {}, **kwargs)

    asyncio.run(scenario())
    assert len(requests) == 1
