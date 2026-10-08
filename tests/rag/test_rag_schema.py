
import os
import uuid

import pytest
from sqlalchemy import delete, func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, configure_mappers

from backend.database.database import Base, engine
from backend.models.rag_document import DocumentORM
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_chunk_embedding import ChunkEmbeddingORM


# 只允许显式运行数据库集成测试
@pytest.fixture
def db_session():
    if os.getenv("RUN_RAG_SCHEMA_TEST") != "1":
        pytest.skip("Schema integration test requires explicit opt-in")

    if engine.dialect.name != "postgresql":
        pytest.fail("Schema integration test requires PostgreSQL")

    # 验证 ORM Mapping
    configure_mappers()

    # 每条测试使用独立 Schema
    schema_name = f"rag_test_{uuid.uuid4().hex}"

    rag_tables = [
        DocumentORM.__table__,
        ChunkORM.__table__,
        ChunkEmbeddingORM.__table__,
    ]

    with engine.connect() as connection:
        # PostgreSQL 支持事务性 DDL
        transaction = connection.begin()

        try:
            # Schema 名称由 uuid 生成，不接受外部输入
            connection.exec_driver_sql(
                f'CREATE SCHEMA "{schema_name}"'
            )

            # 将无显式 Schema 的 ORM 表映射到测试 Schema
            test_connection = connection.execution_options(
                schema_translate_map={None: schema_name}
            )

            # 只创建三张 RAG 测试表
            Base.metadata.create_all(
                bind=test_connection,
                tables=rag_tables,
                checkfirst=False,
            )

            # Session 使用 SAVEPOINT，不提交外层事务
            with Session(
                bind=test_connection,
                join_transaction_mode="create_savepoint",
            ) as session:
                yield session

        finally:
            # 回滚测试数据、建表及 CREATE SCHEMA
            if transaction.is_active:
                transaction.rollback()


def seed_rag_data(session: Session):
    """创建一组完整的 Document -> Chunk -> Embedding。"""

    document = DocumentORM(
        document_id="doc_1",
        file_name="knowledge.txt",
        file_type="txt",
    )
    session.add(document)
    session.flush()

    chunk = ChunkORM(
        chunk_id="doc_1_chunk_0",
        document_id="doc_1",
        content="RAG uses external knowledge.",
    )
    session.add(chunk)
    session.flush()

    embedding = ChunkEmbeddingORM(
        chunk_id="doc_1_chunk_0",
        embedding_model="test-model",
        embedding=[1.0, 0.0, 0.0],
    )
    session.add(embedding)
    session.flush()


# ==================================================
# Case 1: Happy Path
# ==================================================

def test_happy_path(db_session):
    seed_rag_data(db_session)

    document = db_session.get(DocumentORM, "doc_1")
    chunk = db_session.get(ChunkORM, "doc_1_chunk_0")

    embedding = db_session.get(
        ChunkEmbeddingORM,
        ("doc_1_chunk_0", "test-model"),
    )

    assert document is not None
    assert chunk is not None
    assert embedding is not None

    assert chunk.document_id == document.document_id
    assert len(embedding.embedding) == 3


# ==================================================
# Case 2: CHECK Constraint
# ==================================================

def test_invalid_file_type(db_session):
    with pytest.raises(IntegrityError):
        db_session.execute(
            insert(DocumentORM).values(
                document_id="invalid_doc",
                file_name="test.xlsx",
                file_type="excel",
            )
        )


# ==================================================
# Case 3: Foreign Key
# ==================================================

def test_orphan_chunk_rejected(db_session):
    with pytest.raises(IntegrityError):
        db_session.execute(
            insert(ChunkORM).values(
                chunk_id="orphan_chunk",
                document_id="missing_doc",
                content="orphan data",
            )
        )


# ==================================================
# Case 4: Composite Primary Key
# ==================================================

def test_duplicate_embedding_rejected(db_session):
    seed_rag_data(db_session)

    with pytest.raises(IntegrityError):
        db_session.execute(
            insert(ChunkEmbeddingORM).values(
                chunk_id="doc_1_chunk_0",
                embedding_model="test-model",
                embedding=[0.0, 1.0, 0.0],
            )
        )


# ==================================================
# Case 5: Database Cascade Delete
# ==================================================

def test_cascade_delete(db_session):
    seed_rag_data(db_session)

    # Core DELETE：验证数据库层面的 ON DELETE CASCADE
    # 不依赖 ORM relationship cascade
    db_session.execute(
        delete(DocumentORM).where(
            DocumentORM.document_id == "doc_1"
        )
    )

    chunk_count = db_session.scalar(
        select(func.count()).select_from(ChunkORM)
    )

    embedding_count = db_session.scalar(
        select(func.count()).select_from(ChunkEmbeddingORM)
    )

    assert chunk_count == 0
    assert embedding_count == 0
