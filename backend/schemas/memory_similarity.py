from pydantic import BaseModel, ConfigDict


class MemorySimilarityResult(BaseModel):
    """
    Candidate 与已有 Memory 的相似度结果。
    """

    memory_id: int | None = None

    content: str

    memory_type: str

    similarity: float

    model_config = ConfigDict(
        from_attributes=True
    )


class MemorySimilaritySearchResult(BaseModel):
    """
    Memory Similarity Search 的最终结果。
    """

    matches: list[MemorySimilarityResult]

    threshold: float

    top_k: int