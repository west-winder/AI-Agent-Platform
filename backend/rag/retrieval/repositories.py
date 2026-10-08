from sqlalchemy import select

from backend.database.database import SessionLocal
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_document import DocumentORM


class ChunkRepository:
    """
    Chunk 数据访问组件。

    只负责：
    - 根据 chunk_id 批量查询 Chunk

    不负责：
    - SearchResult 一致性检查
    - Ranking / Order Preservation
    - Hydration
    """

    def __init__(
        self,
        session_factory=SessionLocal,
    ):
        self._session_factory = session_factory

    def get_by_ids(
        self,
        chunk_ids: list[str],
    ) -> list[ChunkORM]:

        if not chunk_ids:
            return []

        statement = (
            select(ChunkORM)
            .where(
                ChunkORM.chunk_id.in_(chunk_ids)
            )
        )

        with self._session_factory() as session:
            return list(
                session.scalars(statement).all()
            )


class DocumentRepository:
    """
    Document 数据访问组件。

    只负责：
    - 根据 document_id 批量查询 Document

    不负责：
    - Chunk / Document 一致性检查
    - Hydration
    """

    def __init__(
        self,
        session_factory=SessionLocal,
    ):
        self._session_factory = session_factory

    def get_by_ids(
        self,
        document_ids: list[str],
    ) -> list[DocumentORM]:

        if not document_ids:
            return []

        statement = (
            select(DocumentORM)
            .where(
                DocumentORM.document_id.in_(
                    document_ids
                )
            )
        )

        with self._session_factory() as session:
            return list(
                session.scalars(statement).all()
            )