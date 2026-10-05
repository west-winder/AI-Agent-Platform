import asyncio

import pytest

import backend.rag.generation as generation_module

from backend.rag.contracts import (
    AssembledContext,
    RAGGenerationOutput,
    RetrievedChunk,
)


def build_retrieved_chunk(
    *,
    chunk_id: str = "doc_1_chunk_0",
    content: str = "FastAPI supports async endpoints.",
    source: str = "knowledge.txt",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc_1",
        content=content,
        source=source,
        score=0.95,
    )


def build_context(
    chunks: list[RetrievedChunk],
) -> AssembledContext:

    if not chunks:
        return AssembledContext(
            context_text="",
            used_chunks=[],
        )

    blocks = []

    for chunk in chunks:
        blocks.append(
            f'<chunk id="{chunk.chunk_id}" '
            f'source="{chunk.source}">\n'
            f"{chunk.content}\n"
            "</chunk>"
        )

    context_text = (
        "<document_context>\n"
        + "\n\n".join(blocks)
        + "\n</document_context>"
    )

    return AssembledContext(
        context_text=context_text,
        used_chunks=chunks,
    )


def test_generate_rag_answer_happy_path(
    monkeypatch,
):
    """
    合法 supporting_chunk_id
    应该由 Backend 映射成真实 source。
    """

    chunk = build_retrieved_chunk()

    assembled_context = build_context(
        [chunk]
    )

    async def fake_call_llm_structured(
        messages,
        output_model,
    ):
        assert output_model is RAGGenerationOutput

        return RAGGenerationOutput(
            answer="FastAPI 支持异步接口。",
            supporting_chunk_ids=[
                "doc_1_chunk_0"
            ],
        )

    monkeypatch.setattr(
        generation_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    result = asyncio.run(
        generation_module.generate_rag_answer(
            query="FastAPI 支持异步接口吗？",
            assembled_context=assembled_context,
        )
    )

    assert result.answer == "FastAPI 支持异步接口。"
    assert result.sources == ["knowledge.txt"]


def test_generate_rag_answer_empty_context_returns_no_sources(
    monkeypatch,
):
    """
    Empty Knowledge：

    可以正常回答，
    但不能产生知识库 Source。
    """

    assembled_context = build_context([])

    async def fake_call_llm_structured(
        messages,
        output_model,
    ):
        return RAGGenerationOutput(
            answer=(
                "当前知识库没有检索到相关内容。"
                "以下回答基于模型已有知识。"
            ),
            supporting_chunk_ids=[],
        )

    monkeypatch.setattr(
        generation_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    result = asyncio.run(
        generation_module.generate_rag_answer(
            query="FastAPI 是什么？",
            assembled_context=assembled_context,
        )
    )

    assert result.answer
    assert result.sources == []


def test_generate_rag_answer_raises_for_unknown_supporting_chunk_id(
    monkeypatch,
):
    """
    LLM 引用了没有提供给它的 Chunk：

    属于 Citation Contract Failure，
    不能静默忽略。
    """

    chunk = build_retrieved_chunk()

    assembled_context = build_context(
        [chunk]
    )

    async def fake_call_llm_structured(
        messages,
        output_model,
    ):
        return RAGGenerationOutput(
            answer="测试回答",
            supporting_chunk_ids=[
                "doc_999_chunk_7"
            ],
        )

    monkeypatch.setattr(
        generation_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    with pytest.raises(
        ValueError,
        match=(
            "LLM returned supporting_chunk_ids "
            "that were not provided"
        ),
    ):
        asyncio.run(
            generation_module.generate_rag_answer(
                query="测试问题",
                assembled_context=assembled_context,
            )
        )


def test_generate_rag_answer_context_exists_but_no_chunk_is_cited(
    monkeypatch,
):
    """
    Context 被提供给 LLM
    ≠
    LLM 声明该 Context 支撑最终 Answer。
    """

    chunk = build_retrieved_chunk()

    assembled_context = build_context(
        [chunk]
    )

    async def fake_call_llm_structured(
        messages,
        output_model,
    ):
        return RAGGenerationOutput(
            answer=(
                "当前知识库内容不足，"
                "以下基于模型已有知识补充。"
            ),
            supporting_chunk_ids=[],
        )

    monkeypatch.setattr(
        generation_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    result = asyncio.run(
        generation_module.generate_rag_answer(
            query="测试问题",
            assembled_context=assembled_context,
        )
    )

    assert result.answer
    assert result.sources == []


def test_generate_rag_answer_propagates_llm_failure(
    monkeypatch,
):
    """
    Generation Failure 不是 Empty Knowledge。
    LLM Failure 必须向上传播。
    """

    assembled_context = build_context([])

    async def fake_call_llm_structured(
        messages,
        output_model,
    ):
        raise RuntimeError("fake llm failure")

    monkeypatch.setattr(
        generation_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    with pytest.raises(
        RuntimeError,
        match="fake llm failure",
    ):
        asyncio.run(
            generation_module.generate_rag_answer(
                query="测试问题",
                assembled_context=assembled_context,
            )
        )
