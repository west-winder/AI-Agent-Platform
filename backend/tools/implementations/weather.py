"""Open-Meteo 当前天气工具；返回天气模型条件数据，不代表气象站实测。"""

import json
from datetime import datetime
from typing import Annotated, Any

import httpcore2
import httpx2
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from backend.tools.infrastructure.contracts import ToolContext
from backend.tools.infrastructure.errors import ToolExecutionError


_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_HTTP_TIMEOUT_SECONDS = 10.0


class WeatherArgs(BaseModel):
    """城市名按首尾去空白后的长度限制为 1 至 100 个字符。"""

    city: Annotated[
        str,
        StringConstraints(
            strict=True,
            strip_whitespace=True,
            min_length=1,
            max_length=100,
        ),
    ]


class _Location(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    admin1: str | None = None
    country: str | None = None
    country_code: str | None = None


class _CurrentWeather(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    time: str
    temperature_2m: float
    relative_humidity_2m: float = Field(ge=0, le=100)
    weather_code: int = Field(ge=0, le=99)

    @field_validator("time")
    @classmethod
    def validate_time(cls, value: str) -> str:
        if "T" not in value:
            raise ValueError("Expected an ISO 8601 date and time")
        datetime.fromisoformat(value)
        return value


class _WeatherResponse(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    timezone: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    current: _CurrentWeather


def _invalid_response() -> ToolExecutionError:
    return ToolExecutionError(
        code="WEATHER_INVALID_RESPONSE",
        message="天气服务返回的数据无效",
        retryable=False,
    )


def _transient_connection_error(exc: httpx2.RequestError) -> bool:
    """仅授权可确认的连接故障；协议、TLS 和未知底层异常不自动恢复。"""
    if not isinstance(exc, (httpx2.ConnectError, httpx2.ConnectTimeout)):
        return False
    source: BaseException | None = exc
    seen: set[int] = set()
    while source is not None and id(source) not in seen:
        seen.add(id(source))
        if isinstance(source, (httpcore2.ConnectError, httpcore2.ConnectTimeout)):
            # 可信传输层包装本身不是瞬时故障证据，必须继续核对原因。
            if source.__cause__ is None:
                return False
            source = source.__cause__
            continue
        if type(source) is OSError and source.errno is None and source.__cause__ is not None:
            # AnyIO 的 "All connection attempts failed" 包装；最终原因仍须获准。
            source = source.__cause__
            continue
        if not isinstance(source, (
            httpx2.ConnectError, httpx2.ConnectTimeout, ConnectionRefusedError,
            ConnectionResetError, ConnectionAbortedError, TimeoutError,
        )):
            return False
        source = source.__cause__
    return source is None


async def _get_json(
    client: httpx2.AsyncClient,
    url: str,
    params: dict[str, str | int | float],
    *,
    allow_retry: bool = False,
) -> dict[str, Any]:
    # 仅这两个已审查的只读 GET 路径获准；不依据 ToolErrorInfo.retryable 授权。
    authorized = allow_retry and url in (_GEOCODING_URL, _WEATHER_URL)
    for attempt in range(2 if authorized else 1):
        try:
            response = await client.get(url, params=params)
        except (
            httpx2.TimeoutException,
            httpx2.NetworkError,
            httpx2.RemoteProtocolError,
        ) as exc:
            if authorized and attempt == 0 and _transient_connection_error(exc):
                continue
            raise ToolExecutionError(
                code="WEATHER_SERVICE_UNAVAILABLE",
                message="天气服务暂时不可用，请稍后重试",
                retryable=True,
            ) from exc
        except httpx2.DecodingError as exc:
            raise _invalid_response() from exc

        try:
            response.raise_for_status()
        except httpx2.HTTPStatusError as exc:
            status = response.status_code
            transient = status == 408 or 500 <= status < 600
            if authorized and attempt == 0 and status in (408, 500, 502, 503, 504):
                continue
            retryable = status == 429 or transient
            raise ToolExecutionError(
                code="WEATHER_SERVICE_UNAVAILABLE",
                message="天气服务暂时不可用，请稍后重试" if retryable else "天气服务请求失败",
                retryable=retryable,
            ) from exc
        break

    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _invalid_response() from exc

    if not isinstance(payload, dict) or payload.get("error") is True:
        raise _invalid_response()
    return payload


async def get_weather_handler(
    args: WeatherArgs,
    context: ToolContext,
) -> dict[str, Any]:
    """查询当前天气；仅两个只读外部请求分别获准一次有限重试。"""

    async with httpx2.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as client:
        geocoding = await _get_json(
            client,
            _GEOCODING_URL,
            {"name": args.city, "count": 1, "language": "zh", "format": "json"},
            allow_retry=True,
        )
        # Open-Meteo 无匹配结果时可以省略 results 字段。
        matches = geocoding.get("results", [])
        if not isinstance(matches, list):
            raise _invalid_response()
        if not matches:
            raise ToolExecutionError(
                code="LOCATION_NOT_FOUND",
                message="未找到匹配的城市",
                retryable=False,
            )

        try:
            location = _Location.model_validate(matches[0])
        except ValidationError as exc:
            raise _invalid_response() from exc

        weather = await _get_json(
            client,
            _WEATHER_URL,
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": "temperature_2m,relative_humidity_2m,weather_code",
                "temperature_unit": "celsius",
                "timezone": "auto",
                "timeformat": "iso8601",
            },
            allow_retry=True,
        )
        try:
            forecast = _WeatherResponse.model_validate(weather)
        except ValidationError as exc:
            raise _invalid_response() from exc

    return {
        "requested_city": args.city,
        "location": {**location.model_dump(), "timezone": forecast.timezone},
        "current": {
            "time": forecast.current.time,
            "temperature_c": forecast.current.temperature_2m,
            "relative_humidity_percent": forecast.current.relative_humidity_2m,
            "weather_code": forecast.current.weather_code,
        },
        "source": "Open-Meteo",
    }
