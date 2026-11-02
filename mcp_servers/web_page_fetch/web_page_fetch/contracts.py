"""Successful tool output; FastMCP derives outputSchema from this model."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FetchWebPageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(max_length=500)
    url: str
    content: str = Field(max_length=12_000)
    truncated: bool
    content_type: Literal["text/html"]
