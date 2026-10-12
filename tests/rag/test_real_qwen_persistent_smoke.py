import os
import uuid

import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.database import Base, engine
from backend.embedding.embedder import Embedder

from backend.models.rag_document import DocumentORM
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_chunk_embedding import ChunkEmbeddingORM

from backend.rag.contracts import Document
from backend.rag.ingestion.rag_ingestion import (
    RAGIngestionService,
)
from backend.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)
from backend.rag.retrieval.fusion import RRFFusion
from backend.rag.retrieval.hydrator import (
    RAGHydrator,
)
from backend.rag.retrieval.postgres_vector_search import (
    PostgreSQLVectorSearch,
)
from backend.rag.retrieval.repositories import (
    ChunkRepository,
    DocumentRepository,
)
from backend.rag.retrieval.rerank_document_builder import (
    RerankDocumentBuilder,
)
from backend.rag.retrieval.reranker_adapter import (
    RerankerAdapter,
)
from backend.rag.retrieval.retriever import RAGRetriever


# ==================================================
# MANUAL KEEP
#
# 真实 Qwen Embedding
# +
# 真实 PostgreSQL
# +
# 真实 pgvector
#
# 默认不进入普通 regression
# ==================================================


# ==================================================
# Fake Reranker
# ==================================================

class FakeRerankItem:
    """
    模拟 backend.reranking.reranker.RerankResult 的字段契约。
    """

    def __init__(self, index, score):
        self.index = index
        self.score = score


class PreservingReranker:
    """
    测试专用底层 Reranker：保持 Fusion 顺序。

    本 smoke test 关注真实 Embedding + 真实 pgvector
    的召回质量，因此不叠加真实 Cross Encoder Reranker。
    """

    def rerank(self, query, texts):
        return [
            FakeRerankItem(
                index=index,
                score=1.0 / (index + 1),
            )
            for index, text in enumerate(texts)
        ]


@pytest.fixture(scope="module")
def session_factory():

    if os.getenv("RUN_REAL_QWEN_RAG_SMOKE") != "1":
        pytest.skip(
            "Real Qwen RAG smoke test requires explicit opt-in"
        )

    if engine.dialect.name != "postgresql":
        pytest.fail("PostgreSQL is required")

    schema_name = (
        f"rag_real_qwen_smoke_{uuid.uuid4().hex}"
    )

    tables = [
        DocumentORM.__table__,
        ChunkORM.__table__,
        ChunkEmbeddingORM.__table__,
    ]

    with engine.connect() as connection:

        transaction = connection.begin()

        try:
            database_name = (
                connection.exec_driver_sql(
                    "SELECT current_database()"
                ).scalar_one()
            )

            if database_name != "ai_agent_platform":
                pytest.fail(
                    "Unexpected database"
                )

            connection.exec_driver_sql(
                f'CREATE SCHEMA "{schema_name}"'
            )

            test_connection = (
                connection.execution_options(
                    schema_translate_map={
                        None: schema_name
                    }
                )
            )

            Base.metadata.create_all(
                bind=test_connection,
                tables=tables,
                checkfirst=False,
            )

            factory = sessionmaker(
                bind=test_connection,
                join_transaction_mode=(
                    "create_savepoint"
                ),
                expire_on_commit=False,
            )

            yield factory

        finally:
            if transaction.is_active:
                transaction.rollback()


@pytest.fixture(scope="module")
def embedder():

    if os.getenv("RUN_REAL_QWEN_RAG_SMOKE") != "1":
        pytest.skip(
            "Real Qwen RAG smoke test requires explicit opt-in"
        )

    # 真正加载 .env 中配置的 Qwen3-Embedding-0.6B
    return Embedder()


def test_real_qwen_persistent_rag_smoke(
    session_factory,
    embedder,
):

    # ==================================================
    # 1. Persistent Ingestion
    # ==================================================

    ingestion = RAGIngestionService(
        embedder=embedder,
        session_factory=session_factory,
    )

    documents = [
        (
            Document(
                document_id="doc_vector",
                file_name="vector_search.txt",
                file_type="txt",
            ),
            "PostgreSQL 可以通过 pgvector 扩展执行向量相似度检索。",
        ),
        (
            Document(
                document_id="doc_fastapi",
                file_name="fastapi.txt",
                file_type="txt",
            ),
            "FastAPI 是一个用于构建 Python Web API 的框架。",
        ),
        (
            Document(
                document_id="doc_cat",
                file_name="cat.txt",
                file_type="txt",
            ),
            "猫喜欢在温暖的阳光下休息和睡觉。",
        ),
    ]

    for document, raw_text in documents:

        result = ingestion.ingest(
            document=document,
            raw_text=raw_text,

            # 大于单条文本长度：
            # 每个 Document 只生成一个 Chunk
            chunk_size=200,
        )

        assert result.status == "success"
        assert result.chunk_count == 1

    # ==================================================
    # 2. Build Retrieval Pipeline
    #
    # 唯一 Orchestrator 是 RAGRetriever：
    # Dense 路径注入 PostgreSQL 实现。
    # ==================================================

    chunk_repository = ChunkRepository(
        session_factory=session_factory,
    )

    document_repository = DocumentRepository(
        session_factory=session_factory,
    )

    retriever = RAGRetriever(
        embedder=embedder,
        dense_retriever=PostgreSQLVectorSearch(
            embedding_model=embedder.model_name,
            session_factory=session_factory,
        ),
        bm25_retriever=BM25Retriever(),
        fusion=RRFFusion(),
        rerank_document_builder=RerankDocumentBuilder(),
        reranker=RerankerAdapter(PreservingReranker()),
        hydrator=RAGHydrator(
            chunk_repository=chunk_repository,
            document_repository=document_repository,
        ),
    )

    # ==================================================
    # 3. BM25 / Rerank 需要的语料文本同样来自 Repository
    # ==================================================

    chunk_store = {
        chunk.chunk_id: chunk
        for chunk in chunk_repository.get_by_ids([
            "doc_vector_chunk_0",
            "doc_fastapi_chunk_0",
            "doc_cat_chunk_0",
        ])
    }

    # ==================================================
    # 4. Real Query Embedding + pgvector Search
    # ==================================================

    results = retriever.retrieve(
        query="PostgreSQL 怎么进行向量检索？",
        chunk_store=chunk_store,
        top_k=3,
    )

    # ==================================================
    # 5. Structural Assertions
    # ==================================================

    assert len(results) == 3

    assert {
        result.source
        for result in results
    } == {
        "vector_search.txt",
        "fastapi.txt",
        "cat.txt",
    }

    # ==================================================
    # 6. Semantic Smoke Assertion
    # ==================================================

    # 这不是 Retrieval Evaluation，
    # 只要求明显相关的知识排到第一位。
    assert (
        results[0].source
        == "vector_search.txt"
    )

    # ==================================================
    # 7. Manual Inspection
    # ==================================================

    print("\n")
    print(
        f"Embedding Model: {embedder.model_name}"
    )
    print(
        f"Embedding Dimension: {embedder.dimension}"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"Rank {rank}: "
            f"source={result.source}, "
            f"retrieval_rank="
            f"{result.retrieval_rank}"
        )
        print(
            f"content={result.content}"
        )
