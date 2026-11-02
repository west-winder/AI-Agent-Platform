"""Never construct public errors from an upstream exception or response."""

from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_URL = "INVALID_URL"
    TARGET_NOT_ALLOWED = "TARGET_NOT_ALLOWED"
    DNS_ERROR = "DNS_ERROR"
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    REDIRECT_ERROR = "REDIRECT_ERROR"
    UNSUPPORTED_CONTENT_TYPE = "UNSUPPORTED_CONTENT_TYPE"
    UNSUPPORTED_CONTENT_ENCODING = "UNSUPPORTED_CONTENT_ENCODING"
    RESPONSE_TOO_LARGE = "RESPONSE_TOO_LARGE"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_MESSAGES = {
    ErrorCode.INVALID_URL: "The URL is invalid.",
    ErrorCode.TARGET_NOT_ALLOWED: "The target is not allowed.",
    ErrorCode.DNS_ERROR: "The target could not be resolved.",
    ErrorCode.TIMEOUT: "The fetch exceeded its time limit.",
    ErrorCode.NETWORK_ERROR: "The network request failed.",
    ErrorCode.HTTP_ERROR: "The page returned an unsuccessful HTTP status.",
    ErrorCode.REDIRECT_ERROR: "The redirect is invalid or not allowed.",
    ErrorCode.UNSUPPORTED_CONTENT_TYPE: "Only text/html is supported.",
    ErrorCode.UNSUPPORTED_CONTENT_ENCODING: "Compressed responses are not supported.",
    ErrorCode.RESPONSE_TOO_LARGE: "The response exceeds the byte limit.",
    ErrorCode.INVALID_RESPONSE: "The response is invalid.",
    ErrorCode.INTERNAL_ERROR: "The fetch could not be completed.",
}


class FetchError(Exception):
    def __init__(self, code: ErrorCode):
        self.code = ErrorCode(code)
        super().__init__(f"{self.code}: {_MESSAGES[self.code]}")
