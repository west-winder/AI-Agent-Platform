from typing import Literal

from pydantic import BaseModel, field_validator


class Document(BaseModel):
    """
    RAG 中的文件级对象。

    当前只保存已经确定的稳定元数据。
    TXT / Markdown / PDF 后续都统一抽象为 Document。
    """

    document_id: str
    file_name: str
    file_type: Literal["txt", "markdown", "pdf"]

    @field_validator("document_id", "file_name")
    @classmethod
    def validate_non_blank_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be blank")
        return value


class Chunk(BaseModel):
    """
    Document 切分后的知识单元。

    Chunk 描述知识本身以及它属于哪个 Document。
    不包含 retrieval score，也不重复保存 Document 的 source 信息。
    """

    chunk_id: str
    document_id: str
    content: str

    @field_validator("chunk_id", "document_id", "content")
    @classmethod
    def validate_non_blank_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be blank")
        return value


class SearchResult(BaseModel):
    """
    Exact Search 的轻量检索结果。

    只表示：
    - 找到了哪个 Chunk
    - 该 Chunk 的检索分数是多少
    """

    chunk_id: str
    score: float

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id must not be blank")
        return value


class RetrievedChunk(BaseModel):
    """
    SearchResult 完成 Hydration 后得到的完整检索结果。

    它已经包含下游 ContextAssembler 消费所需要的信息。
    """

    chunk_id: str
    document_id: str
    content: str
    source: str
    score: float

    @field_validator(
        "chunk_id",
        "document_id",
        "content",
        "source",
    )
    @classmethod
    def validate_non_blank_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must not be blank")
        return value


class AssembledContext(BaseModel):
    """
    ContextAssembler 的输出。

    context_text:
        真正交给 LLM 的 Document Context。

    used_chunks:
        实际参与 Context Assembly 的完整 RetrievedChunk，
        用于 Citation、Debug 和后续 Evaluation。
    """

    context_text: str
    used_chunks: list[RetrievedChunk]


class RAGGenerationOutput(BaseModel):
    """
    LLM Structured Output。

    answer:
        模型生成的最终回答。

    supporting_chunk_ids:
        模型声明用于支撑当前回答的知识库 Chunk。
        这里只允许返回 Context 中真实存在的 chunk_id。
    """

    answer: str
    supporting_chunk_ids: list[str]

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must not be blank")
        return value

    @field_validator("supporting_chunk_ids")
    @classmethod
    def validate_supporting_chunk_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        if any(
            not chunk_id.strip()
            for chunk_id in value
        ):
            raise ValueError(
                "supporting_chunk_ids must not contain blank id"
            )

        return value


class RAGAnswer(BaseModel):
    """
    RAG Backend 最终输出。

    answer:
        最终回答。

    sources:
        Backend 根据 supporting_chunk_ids
        从真实 RetrievedChunk 中解析出的来源。
    """

    answer: str
    sources: list[str]