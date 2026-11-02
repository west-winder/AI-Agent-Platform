import pytest
from pydantic import ValidationError

from web_page_fetch.contracts import FetchWebPageResult
from web_page_fetch.errors import ErrorCode, FetchError


def test_result_strict_fields():
    valid = dict(title="", url="https://example.com/", content="", truncated=False, content_type="text/html")
    assert FetchWebPageResult(**valid).model_dump() == valid
    for changes in ({"extra": 1}, {"truncated": "false"}, {"title": "x" * 501},
                    {"content": "x" * 12001}, {"content_type": "application/xhtml+xml"}):
        with pytest.raises(ValidationError):
            FetchWebPageResult(**(valid | changes))


@pytest.mark.parametrize("code", list(ErrorCode))
def test_fixed_errors(code):
    assert str(FetchError(code)).startswith(f"{code}: ")
