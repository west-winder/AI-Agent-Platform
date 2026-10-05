from backend.rag.contracts import (
    AssembledContext,
    RAGAnswer,
    RAGGenerationOutput,
)
from backend.services.llm_service import (
    call_llm_structured,
)


RAG_SYSTEM_PROMPT = """
You are answering a user question with optional document context.

Rules:

1. Prefer information from <document_context>.
2. Each <chunk> has a stable id. If a chunk actually supports your
   knowledge-base-based answer, include its id in supporting_chunk_ids.
3. supporting_chunk_ids may only contain ids that appear in
   <document_context>.
4. If the document context is empty, explicitly tell the user that the
   current knowledge base did not provide relevant information.
5. If the document context is insufficient, explicitly say that the
   knowledge base is insufficient.
6. You may supplement with your own existing knowledge when the knowledge
   base is empty or insufficient, but clearly distinguish that information
   from knowledge-base evidence.
7. Do not claim that your own model knowledge came from the document
   context.
8. If no document chunk supports the answer, supporting_chunk_ids must be [].
"""


async def generate_rag_answer(
    query: str,
    assembled_context: AssembledContext,
) -> RAGAnswer:

    user_content = (
        f"<query>\n"
        f"{query}\n"
        f"</query>\n\n"
        f"{assembled_context.context_text}"
    )

    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    generation_output = await call_llm_structured(
        messages=messages,
        output_model=RAGGenerationOutput,
    )

    # --------------------------------------------------
    # Backend Ground Truth
    # --------------------------------------------------

    chunk_by_id = {
        chunk.chunk_id: chunk
        for chunk in assembled_context.used_chunks
    }

    # --------------------------------------------------
    # Citation Referential Integrity
    # --------------------------------------------------

    invalid_chunk_ids = [
        chunk_id
        for chunk_id
        in generation_output.supporting_chunk_ids
        if chunk_id not in chunk_by_id
    ]

    if invalid_chunk_ids:
        raise ValueError(
            "LLM returned supporting_chunk_ids "
            "that were not provided in document context: "
            f"{invalid_chunk_ids}"
        )

    # --------------------------------------------------
    # supporting_chunk_ids -> real sources
    # --------------------------------------------------

    sources: list[str] = []
    seen_sources: set[str] = set()

    for chunk_id in generation_output.supporting_chunk_ids:

        source = chunk_by_id[
            chunk_id
        ].source

        if source not in seen_sources:
            sources.append(source)
            seen_sources.add(source)

    return RAGAnswer(
        answer=generation_output.answer,
        sources=sources,
    )