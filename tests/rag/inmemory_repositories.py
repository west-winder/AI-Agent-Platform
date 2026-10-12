"""
测试专用 InMemory Repository。

Hydrator 只依赖 Repository Contract（``get_by_ids``），
因此测试可以用 dict 支撑的实现替代 PostgreSQL Repository，
生产代码不需要保留第二套内存存储实现。
"""

from backend.rag.retrieval.hydrator import RAGHydrator


class InMemoryChunkRepository:
    """
    Chunk Repository 的内存实现。

    契约与 backend.rag.retrieval.repositories.ChunkRepository 一致：
    chunk_id -> Chunk
    """

    def __init__(self, chunk_store: dict):
        self._chunk_store = chunk_store

    def get_by_ids(self, chunk_ids: list[str]) -> list:

        if not chunk_ids:
            return []

        return [
            self._chunk_store[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self._chunk_store
        ]


class InMemoryDocumentRepository:
    """
    Document Repository 的内存实现。

    契约与 backend.rag.retrieval.repositories.DocumentRepository 一致：
    document_id -> Document
    """

    def __init__(self, document_store: dict):
        self._document_store = document_store

    def get_by_ids(self, document_ids: list[str]) -> list:

        if not document_ids:
            return []

        return [
            self._document_store[document_id]
            for document_id in document_ids
            if document_id in self._document_store
        ]


def build_inmemory_hydrator(
    chunk_store: dict,
    document_store: dict,
) -> RAGHydrator:
    """
    用内存语料装配一个 repository-based RAGHydrator。

    生产路径注入 PostgreSQL Repository，
    测试路径注入这里的 InMemory Repository，
    两者共享同一个 Repository Contract。
    """

    return RAGHydrator(
        chunk_repository=InMemoryChunkRepository(
            chunk_store
        ),
        document_repository=InMemoryDocumentRepository(
            document_store
        ),
    )
