import asyncio
from collections.abc import AsyncIterator, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk


QUESTION = "In one short phrase, describe a runnable."
FAKE_TEXT = "Compose and run."


async def invoke_async(model: BaseChatModel, question: str) -> AIMessage:
    return await model.ainvoke(question)


def stream_chunks(model: BaseChatModel, question: str) -> Iterator[AIMessageChunk]:
    yield from model.stream(question)


async def stream_chunks_async(
    model: BaseChatModel, question: str
) -> AsyncIterator[AIMessageChunk]:
    async for chunk in model.astream(question):
        yield chunk


def combine_chunks(chunks: list[AIMessageChunk]) -> AIMessageChunk:
    """合并非空 Chunk 列表。"""
    # 用消息对象的加法合并内容及元数据。合并结果仍是 Chunk。
    combined, *remaining = chunks
    for chunk in remaining:
        combined = combined + chunk
    return combined


async def main() -> None:
    model = FakeListChatModel(responses=[FAKE_TEXT])
    message = await invoke_async(model, QUESTION)
    print("ainvoke", type(message).__name__, repr(message.content))
    sync_chunks = list(stream_chunks(model, QUESTION))
    async_chunks = [chunk async for chunk in stream_chunks_async(model, QUESTION)]
    for method, chunks in [("stream", sync_chunks), ("astream", async_chunks)]:
        print(method, [(type(chunk).__name__, chunk.content) for chunk in chunks])
        print("combined", repr(combine_chunks(chunks).content))


if __name__ == "__main__":
    asyncio.run(main())
