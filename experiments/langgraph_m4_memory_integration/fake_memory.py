import re


class FakeMemoryService:
    """演示存储按用户分桶，同一用户内用简单词项匹配相关记忆。"""

    def __init__(self, initial: dict[int, list[str]] | None = None):
        self._records = {user_id: list(records) for user_id, records in (initial or {}).items()}

    async def load_context(self, user_id: int, query: str) -> str:
        terms = set(re.findall(r"\w{3,}", query.casefold()))
        return "\n".join(
            record
            for record in self._records.get(user_id, [])
            if terms.intersection(re.findall(r"\w{3,}", record.casefold()))
        )

    async def save_memory(self, user_id: int, content: str) -> None:
        self._records.setdefault(user_id, []).append(content)
