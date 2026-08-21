from pydantic import BaseModel


class MemoryRelationship(BaseModel):
    """
    Candidate Memory 与 Existing Memory
    之间的关系。
    """

    memory_id: int

    relationship: str

    reason: str


class MemoryRelationshipResult(BaseModel):
    """
    Memory Relationship Judge 的最终结果。
    """

    relationships: list[MemoryRelationship]