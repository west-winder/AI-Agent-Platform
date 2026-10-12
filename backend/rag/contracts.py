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


class DenseRetrieverResult(BaseModel):
    """
    Dense Retriever 的检索与排序结果。

    只表示：
    - 命中了哪个 Chunk
    - Dense Retrieval 的相似度分数
    - Dense Retrieval 的业务排名（1-based）

    不暴露 Query Vector / Chunk Vector 等向量实现细节，
    向量计算完全封装在 Dense Retriever 内部。
    """

    chunk_id: str
    dense_score: float
    dense_rank: int

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id must not be blank")
        return value


class BM25RetrieverResult(BaseModel):
    """
    BM25 Retriever 的检索与排序结果。

    只表示：
    - 命中了哪个 Chunk
    - BM25 算法分数
    - BM25 的业务排名（1-based）

    不包含 content（Hydration 的职责），
    也不包含任何 Embedding / Vector 信息。
    """

    chunk_id: str
    bm25_score: float
    bm25_rank: int

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id must not be blank")
        return value


class RetrievedCandidate(BaseModel):
    """
    Retrieval / Ranking 阶段交给 Hydration 阶段的统一候选。

    它只承载 Hydration 真正需要的信息：

    - 命中了哪个 Chunk
    - 该 Chunk 在当前 Retrieval Ranking 中的业务排名（1-based）

    不包含任何阶段专有分数（dense_score / bm25_score / rrf_score /
    rerank_score ...），因此 Dense / Fusion / Reranker 任何一段的输出
    都可以适配成 RetrievedCandidate，Hydrator 无需感知上游是哪个阶段。
    """

    chunk_id: str
    rank: int

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id must not be blank")
        return value


class RagRetrievedChunk(BaseModel):
    """
    Retrieval + Ranking + Hydration 之后得到的完整 RAG 检索结果。

    它已经包含下游 ContextAssembler 消费所需要的信息，
    不代表 Retriever 的原始输出。

    retrieval_rank:
        该 Chunk 在 Retrieval 阶段的业务排名（1-based）。
    """

    chunk_id: str
    document_id: str
    content: str
    source: str
    retrieval_rank: int

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


class FusedRetrieverResult(BaseModel):
    """
    Hybrid Retrieval（Dense + BM25 -> RRF）融合结果。

    一个 Chunk 可能同时被 Dense 与 BM25 命中，
    也可能只被其中一个命中。

    因此 dense_* 与 bm25_* 字段允许缺失（None）：

    - 只被 BM25 命中：dense_score = None, dense_rank = None
    - 只被 Dense 命中：bm25_score = None, bm25_rank = None

    不包含 content（Hydration 的职责），
    也不包含任何 Embedding / Vector 信息。
    """

    chunk_id: str
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float
    rrf_rank: int

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk_id must not be blank")
        return value


class RerankDocument(BaseModel):
    """
    Reranker 输入文档的占位契约。

    当前只提供定义，不提供实现。
    """

    chunk_id: str
    content: str


class RerankedResult(BaseModel):
    """
    Reranker 输出结果的占位契约。

    当前只提供定义，不提供实现。
    """

    chunk_id: str
    previous_rank: int
    rerank_score: float
    rerank_rank: int


class AssembledContext(BaseModel):
    """
    ContextAssembler 的输出。

    context_text:
        真正交给 LLM 的 Document Context。

    used_chunks:
        实际参与 Context Assembly 的完整 RagRetrievedChunk，
        用于 Citation、Debug 和后续 Evaluation。
    """

    context_text: str
    used_chunks: list[RagRetrievedChunk]


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
        从真实 RagRetrievedChunk 中解析出的来源。
    """

    answer: str
    sources: list[str]