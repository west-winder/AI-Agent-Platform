from backend.rag.contracts import Chunk, Document


def chunk_document(
    document: Document,
    raw_text: str,
    chunk_size: int,
) -> list[Chunk]:
    """
    将一个 Document 对应的原始文本按固定字符长度切分成 Chunk[]。

    当前 Phase 2A 只实现最小固定长度 Chunking：
    - 不负责 Parser
    - 不负责 Embedding
    - 不负责 Storage
    - 不负责 Retrieval
    """

    # --------------------------------------------------
    # 1. chunk_size 是配置 Contract
    # --------------------------------------------------

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    # --------------------------------------------------
    # 2. Empty / whitespace-only text 是合法空输入
    # --------------------------------------------------

    if not raw_text.strip():
        return []

    # --------------------------------------------------
    # 3. Fixed-size Chunking
    # --------------------------------------------------

    chunks: list[Chunk] = []

    for chunk_index, start in enumerate(
        range(0, len(raw_text), chunk_size)
    ):
        content = raw_text[
            start:start + chunk_size
        ]

        chunks.append(
            Chunk(
                chunk_id=(
                    f"{document.document_id}"
                    f"_chunk_{chunk_index}"
                ),
                document_id=document.document_id,
                content=content,
            )
        )

    return chunks