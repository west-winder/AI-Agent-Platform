import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START

from backend.rag.contracts import Chunk, RagRetrievedChunk
from experiments.langgraph_m4_rag_integration.fake_retriever import FakeRetriever
from experiments.langgraph_m4_rag_integration.rag_graph import NO_RESULTS, build_rag_graph


@pytest.fixture
def documents():
    # Fake 刻意让 rank 与列表顺序不同，用于检查图不会重排或重新编号。
    return [
        RagRetrievedChunk(
            chunk_id="chunk-z",
            document_id="backup-policy",
            content="每日备份保留 30 天。\nPolicy name: \"Daily backup\".",
            source="policy.md",
            retrieval_rank=4,
        ),
        RagRetrievedChunk(
            chunk_id="chunk-a",
            document_id="backup-glossary",
            content="The retention period is measured in calendar days.",
            source="glossary.md",
            retrieval_rank=2,
        ),
    ]


@pytest.fixture
def chunk_store(documents):
    return {
        doc.chunk_id: Chunk(chunk_id=doc.chunk_id, document_id=doc.document_id, content=doc.content)
        for doc in documents
    }


def test_retrieve_inputs_field_mapping_order_and_model_context(documents, chunk_store, model_invoke):
    query = "每日备份保留多久？"
    answer = "每日备份保留 30 天。"
    retriever = FakeRetriever({query: documents})
    model = FakeListChatModel(responses=[answer], cache=False)
    graph = build_rag_graph(retriever, chunk_store, model, top_k=7)
    assert not retriever.calls
    model_invoke.assert_not_called()
    assert {(edge.source, edge.target) for edge in graph.get_graph().edges} == {
        (START, "Retrieve"), ("Retrieve", "LLM"), ("LLM", END)
    }

    result = graph.invoke({"query": query})

    assert retriever.calls == [(query, chunk_store, 7)]
    assert retriever.calls[0][1] is chunk_store
    expected_docs = [
        {
            "chunk_id": "chunk-z",
            "document_id": "backup-policy",
            "content": documents[0].content,
            "source": "policy.md",
            "rank": 4,
        },
        {
            "chunk_id": "chunk-a",
            "document_id": "backup-glossary",
            "content": documents[1].content,
            "source": "glossary.md",
            "rank": 2,
        },
    ]
    assert result == {"query": query, "retrieved_docs": expected_docs, "answer": answer}
    assert model_invoke.call_count == 1
    actual_model, prompt = model_invoke.call_args.args
    assert actual_model is model
    system, human = prompt
    assert isinstance(system, SystemMessage)
    assert "Use only the content of the retrieved passages as knowledge-base evidence" in system.content
    assert "say the retrieved evidence is insufficient" in system.content
    assert isinstance(human, HumanMessage)
    assert json.loads(human.content) == {"query": query, "retrieved_docs": expected_docs}


def test_empty_retrieval_retains_empty_list_and_skips_model(model_invoke):
    query = "What is the support number?"
    chunk_store = {}
    retriever = FakeRetriever({query: []})
    model = FakeListChatModel(responses=["Unexpected model answer"], cache=False)

    result = build_rag_graph(retriever, chunk_store, model).invoke({"query": query})

    assert retriever.calls == [(query, chunk_store, 5)]
    assert result == {"query": query, "retrieved_docs": [], "answer": NO_RESULTS}
    model_invoke.assert_not_called()


def test_retrieval_exception_propagates_without_calling_model(model_invoke):
    query = "What is the backup schedule?"
    error = RuntimeError("retriever unavailable")
    retriever = FakeRetriever({query: error})
    model = FakeListChatModel(responses=["Unexpected model answer"], cache=False)

    with pytest.raises(RuntimeError, match="retriever unavailable") as raised:
        build_rag_graph(retriever, {}, model).invoke({"query": query})

    assert raised.value is error
    assert retriever.calls == [(query, {}, 5)]
    model_invoke.assert_not_called()


def test_independent_invocations_use_only_current_retrieval(documents, chunk_store, model_invoke):
    first, empty, last = "first query", "no matching document", "last query"
    retriever = FakeRetriever({first: [documents[0]], empty: [], last: [documents[1]]})
    model = FakeListChatModel(responses=["First answer", "Last answer"], cache=False)
    graph = build_rag_graph(retriever, chunk_store, model)

    first_state = graph.invoke({"query": first})
    empty_state = graph.invoke({"query": empty})
    last_state = graph.invoke({"query": last})

    assert first_state["answer"] == "First answer"
    assert empty_state == {"query": empty, "retrieved_docs": [], "answer": NO_RESULTS}
    assert last_state["answer"] == "Last answer"
    assert [doc["chunk_id"] for doc in first_state["retrieved_docs"]] == ["chunk-z"]
    assert [doc["chunk_id"] for doc in last_state["retrieved_docs"]] == ["chunk-a"]
    assert [query for query, _, _ in retriever.calls] == [first, empty, last]
    assert model_invoke.call_count == 2
    last_prompt = json.loads(model_invoke.call_args.args[1][-1].content)
    assert last_prompt == {"query": last, "retrieved_docs": last_state["retrieved_docs"]}
