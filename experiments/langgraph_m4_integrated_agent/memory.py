import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class MemoryDecision(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", revalidate_instances="always")

    should_save: bool = Field(description="Whether the user explicitly stated a durable fact or preference.")
    content: str | None = Field(description="Exact excerpt of the user statement, or null when skipping.")


EXTRACTION_INSTRUCTIONS = (
    "Save only a durable fact or preference that the author explicitly states about themselves. "
    "Copy an exact excerpt. For temporary requests, small talk, quotations from other people, "
    "or inferred information, return should_save=false and content=null."
)


class MemoryService(Protocol):
    async def load_context(self, user_id: int, query: str) -> str: ...

    async def save_memory(self, user_id: int, content: str) -> None: ...


class FakeMemoryService:
    """按用户分桶，按简单词项匹配相关记忆。"""

    def __init__(self, initial: dict[int, list[str]] | None = None):
        self._records = {user_id: list(records) for user_id, records in (initial or {}).items()}

    async def load_context(self, user_id: int, query: str) -> str:
        terms = set(re.findall(r"\w{3,}", query.casefold()))
        return "\n".join(
            record for record in self._records.get(user_id, [])
            if terms.intersection(re.findall(r"\w{3,}", record.casefold()))
        )

    async def save_memory(self, user_id: int, content: str) -> None:
        self._records.setdefault(user_id, []).append(content)
