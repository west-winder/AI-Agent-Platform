from backend.rag.contracts import (
    AssembledContext,
    RetrievedChunk,
)


class ContextAssembler:
    """
    将 RetrievedChunk[] 组装成 LLM 可消费的 Document Context。

    职责：

    RetrievedChunk[]
        ↓
    保持原有 Retrieval 顺序
        ↓
    每个 Chunk 生成一个独立 Context Block
        ↓
    拼接成 document_context
        ↓
    AssembledContext

    本组件不负责：

    1. Retrieval
    2. Ranking
    3. Hydration
    4. LLM Generation
    5. 判断 Empty Retrieval 是否允许回答
    """

    def assemble(
        self,
        chunks: list[RetrievedChunk],
    ) -> AssembledContext:
        """
        将 RetrievedChunk 列表组装为结构化 Context。

        Empty Input 是合法状态：
            []
            ↓
            AssembledContext(
                context_text="",
                used_chunks=[],
            )
        """

        # --------------------------------------------------
        # 1. Empty Retrieval 是合法状态
        # --------------------------------------------------

        if not chunks:
            return AssembledContext(
                context_text="",
                used_chunks=[],
            )

        # --------------------------------------------------
        # 2. 每个 RetrievedChunk 转换成一个 Context Block
        # --------------------------------------------------

        context_blocks: list[str] = []

        for chunk in chunks:
            block = (
                f'<chunk id="{chunk.chunk_id}" '
                f'source="{chunk.source}">\n'
                f"{chunk.content}\n"
                "</chunk>"
            )

            context_blocks.append(block)

        # --------------------------------------------------
        # 3. 组合 Document Context
        # --------------------------------------------------

        context_text = (
            "<document_context>\n"
            + "\n\n".join(context_blocks)
            + "\n</document_context>"
        )

        # --------------------------------------------------
        # 4. 返回结构化 Assembly Result
        # --------------------------------------------------

        return AssembledContext(
            context_text=context_text,
            used_chunks=chunks,
        )