"""
[KEEP]

Basic Logging V1 Contract Regression

验证 ChatService 四个 Memory Logging Boundary：

chat()
├── Memory Read Failure
└── Memory Write Failure

stream_chat()
├── Memory Read Failure
└── Memory Write Failure

四条 Contract 统一为：

- level:     WARNING
- message:   "Memory read failed" / "Memory write failed"
- operation: memory_read / memory_write
- fallback:  empty_memory_context / skip_memory_write
- context:   user_id / conversation_id
- traceback: exc_info=True 保留

同时验证 Logging 没有改变原有 Failure Policy：

- Memory Read 失败 → Chat 降级继续，Main LLM 仍被调用
- Memory Write 失败 → 已完成的 Chat / Stream 不受影响

不调用真实 DeepSeek。
不加载真实 Embedding / Reranker。
不依赖真实网络 / PostgreSQL。

本仓库没有 pytest-asyncio：
所有 test 函数保持同步 def，
async contract 用 run() = asyncio.run() 驱动。

stream_chat() 是 Async Iterator：
Memory Write 位于最后一个 yield 之后，
因此测试通过完整消费 generator
真正驱动到 Memory Write 阶段。
"""

import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


# ==================================================
# Project Root Bootstrap
#
# 无论是：
#
#   从 tests/logging 运行
#   从项目根目录运行
#   由 pytest 导入
#
# sys.path 里都不会自动有项目根目录，
# 这里根据 __file__ 反推并显式加入，
# 保证 import backend... 三种方式都成立。
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


import backend.services.chat_service as chat_service


# ==================================================
# Async Contract → 同步测试适配
# ==================================================


def run(coro):
    """
    在同步测试中真实执行并等待
    async production contract。
    """

    return asyncio.run(coro)


def consume_stream():
    """
    完整消费 stream_chat() Async Iterator。

    Memory Write 阶段只在
    最后一次 __anext__() 时执行，
    所以必须完整消费到 StopAsyncIteration，
    测试才真正覆盖了 Memory Write Boundary。
    """

    async def _consume():

        chunks = []

        async for chunk in chat_service.stream_chat(
            db=None,
            conversation_id=1,
            user_message=USER_MESSAGE
        ):
            chunks.append(chunk)

        return chunks

    return run(_consume())


# ==================================================
# Shared Fake Data
# ==================================================

USER_MESSAGE = "我最近是不是在学习 LangGraph？"

MEMORY_CONTEXT = (
    "<memory_context>"
    "用户最近开始学习 LangGraph"
    "</memory_context>"
)

STREAM_CHUNKS = [
    "你好",
    "，",
    "这是",
    "流式",
    "回答"
]

CHAT_ANSWER = "普通 Chat 仍然正常回答。"


# ==================================================
# Shared Fakes
#
# ChatService 当前 await memory_reader.read(...)
# 和 await pipeline.process(...)，
# 因此 Fake 必须保持 async Calling Contract，
# 否则 TypeError 会被降级分支静默吞掉，
# 造成假 PASS。
# ==================================================


def fake_get_conversation(
    db,
    conversation_id
):
    """
    ChatService 只需要：
    conversation.user_id
    conversation.agent_snapshot
    """

    return SimpleNamespace(
        id=conversation_id,
        user_id=1,
        agent_snapshot={
            "system_prompt": (
                "你是一个测试 Agent。"
            )
        }
    )


def fake_create_message(
    db,
    message_data,
    conversation_id,
    role
):
    """
    Assistant Message 最后需要 .id。
    """

    if role == "assistant":
        return SimpleNamespace(
            id=200
        )

    return SimpleNamespace(
        id=100
    )


def fake_get_messages(
    db,
    conversation_id
):
    return [
        SimpleNamespace(
            role="user",
            content=USER_MESSAGE
        )
    ]


class SuccessMemoryReader:
    """
    Memory Read 成功路径的 Fake。
    """

    async def read(
        self,
        db,
        user_id,
        query,
        top_n,
        top_k
    ):
        return SimpleNamespace(
            memory_context=MEMORY_CONTEXT
        )


class FailingMemoryReader:
    """
    Memory Read 失败路径的 Fake。

    模拟真实出现过的：
    LLM Judge 返回空内容。
    """

    async def read(
        self,
        db,
        user_id,
        query,
        top_n,
        top_k
    ):
        raise ValueError(
            "LLM Judge 返回了空内容"
        )


class FakeMemoryPipeline:
    """
    Memory Write 成功路径的 Fake。
    """

    async def process(
        self,
        db,
        user_id,
        user_message
    ):
        return None


class FailingMemoryPipeline:
    """
    Memory Write 失败路径的 Fake。
    """

    async def process(
        self,
        db,
        user_id,
        user_message
    ):
        raise RuntimeError(
            "Memory pipeline 写入失败"
        )


# ==================================================
# Logging Contract Assertion Helpers
#
# 不检查"出现了一串日志文本"，
# 而是直接检查 LogRecord 本身。
# ==================================================


def find_record(caplog, message):
    """
    在 caplog 中定位恰好一条
    指定 message 的 LogRecord。
    """

    matching = [
        record
        for record in caplog.records
        if record.getMessage() == message
    ]

    assert len(matching) == 1, (
        f"期望恰好一条 {message!r} 日志，"
        f"实际捕获 {len(matching)} 条"
    )

    return matching[0]


def count_records(caplog, message):
    return sum(
        1
        for record in caplog.records
        if record.getMessage() == message
    )


def assert_memory_read_contract(
    record,
    expected_exc_type
):
    """
    Memory Read Failure Logging Contract。
    """

    assert (
        record.name
        == chat_service.__name__
    )

    assert record.levelname == "WARNING"

    assert record.operation == "memory_read"

    assert record.user_id == 1

    assert record.conversation_id == 1

    assert (
        record.fallback
        == "empty_memory_context"
    )

    # traceback 必须保留

    assert record.exc_info is not None

    assert (
        record.exc_info[0]
        is expected_exc_type
    )


def assert_memory_write_contract(
    record,
    expected_exc_type
):
    """
    Memory Write Failure Logging Contract。
    """

    assert (
        record.name
        == chat_service.__name__
    )

    assert record.levelname == "WARNING"

    assert record.operation == "memory_write"

    assert record.user_id == 1

    assert record.conversation_id == 1

    assert (
        record.fallback
        == "skip_memory_write"
    )

    # traceback 必须保留

    assert record.exc_info is not None

    assert (
        record.exc_info[0]
        is expected_exc_type
    )


# ==================================================
# Case 1
# chat() Memory Read Failure
# ==================================================


def test_chat_memory_read_failure_logging_contract(
    caplog
):

    caplog.set_level(logging.WARNING)

    captured_messages = None

    async def fake_call_llm(messages):

        nonlocal captured_messages

        captured_messages = messages

        return CHAT_ANSWER

    with (
        patch.object(
            chat_service,
            "get_conversation",
            fake_get_conversation
        ),
        patch.object(
            chat_service,
            "create_message",
            fake_create_message
        ),
        patch.object(
            chat_service,
            "get_messages_by_conversation",
            fake_get_messages
        ),
        patch.object(
            chat_service,
            "call_llm",
            fake_call_llm
        ),
        patch.object(
            chat_service,
            "memory_reader",
            FailingMemoryReader()
        ),
        patch(
            "backend.memory.memory_write."
            "memory_pipeline.MemoryPipeline",
            FakeMemoryPipeline
        )
    ):

        response = run(
            chat_service.chat(
                db=None,
                conversation_id=1,
                user_message=USER_MESSAGE
            )
        )

    # ------------------------------------------------
    # Logging Contract
    # ------------------------------------------------

    record = find_record(
        caplog,
        "Memory read failed"
    )

    assert_memory_read_contract(
        record,
        ValueError
    )

    # read 失败场景下不应出现 write 失败日志

    assert (
        count_records(
            caplog,
            "Memory write failed"
        ) == 0
    )

    # ------------------------------------------------
    # Failure Policy:
    # Chat 继续 + Main LLM 被调用
    # ------------------------------------------------

    assert captured_messages is not None

    system_message = captured_messages[0]

    assert system_message["role"] == "system"

    assert (
        "你是一个测试 Agent"
        in system_message["content"]
    )

    # 降级为普通 Chat，不注入 Memory Context

    assert (
        "<memory_context>"
        not in system_message["content"]
    )

    # ------------------------------------------------
    # Failure Policy:
    # 返回正常 ChatResponse
    # ------------------------------------------------

    assert response.answer == CHAT_ANSWER

    assert response.message_id == 200

    assert response.conversation_id == 1


# ==================================================
# Case 2
# chat() Memory Write Failure
# ==================================================


def test_chat_memory_write_failure_logging_contract(
    caplog
):

    caplog.set_level(logging.WARNING)

    captured_messages = None

    async def fake_call_llm(messages):

        nonlocal captured_messages

        captured_messages = messages

        return CHAT_ANSWER

    with (
        patch.object(
            chat_service,
            "get_conversation",
            fake_get_conversation
        ),
        patch.object(
            chat_service,
            "create_message",
            fake_create_message
        ),
        patch.object(
            chat_service,
            "get_messages_by_conversation",
            fake_get_messages
        ),
        patch.object(
            chat_service,
            "call_llm",
            fake_call_llm
        ),
        patch.object(
            chat_service,
            "memory_reader",
            SuccessMemoryReader()
        ),
        patch(
            "backend.memory.memory_write."
            "memory_pipeline.MemoryPipeline",
            FailingMemoryPipeline
        )
    ):

        response = run(
            chat_service.chat(
                db=None,
                conversation_id=1,
                user_message=USER_MESSAGE
            )
        )

    # ------------------------------------------------
    # Logging Contract
    # ------------------------------------------------

    record = find_record(
        caplog,
        "Memory write failed"
    )

    assert_memory_write_contract(
        record,
        RuntimeError
    )

    # write 失败场景下 read 未失败

    assert (
        count_records(
            caplog,
            "Memory read failed"
        ) == 0
    )

    # ------------------------------------------------
    # Failure Policy:
    # Memory Read 路径不受影响
    # ------------------------------------------------

    system_message = captured_messages[0]

    assert (
        MEMORY_CONTEXT
        in system_message["content"]
    )

    # ------------------------------------------------
    # Failure Policy:
    # ChatResponse 仍正常返回
    # ------------------------------------------------

    assert response.answer == CHAT_ANSWER

    assert response.message_id == 200


# ==================================================
# Case 3
# stream_chat() Memory Read Failure
# ==================================================


def test_stream_chat_memory_read_failure_logging_contract(
    caplog
):

    caplog.set_level(logging.WARNING)

    captured_messages = None

    async def fake_stream_llm(messages):

        nonlocal captured_messages

        captured_messages = messages

        for chunk in STREAM_CHUNKS:
            yield chunk

    with (
        patch.object(
            chat_service,
            "get_conversation",
            fake_get_conversation
        ),
        patch.object(
            chat_service,
            "create_message",
            fake_create_message
        ),
        patch.object(
            chat_service,
            "get_messages_by_conversation",
            fake_get_messages
        ),
        patch.object(
            chat_service,
            "stream_llm",
            fake_stream_llm
        ),
        patch.object(
            chat_service,
            "memory_reader",
            FailingMemoryReader()
        ),
        patch(
            "backend.memory.memory_write."
            "memory_pipeline.MemoryPipeline",
            FakeMemoryPipeline
        )
    ):

        chunks = consume_stream()

    # ------------------------------------------------
    # Logging Contract
    # ------------------------------------------------

    record = find_record(
        caplog,
        "Memory read failed"
    )

    assert_memory_read_contract(
        record,
        ValueError
    )

    assert (
        count_records(
            caplog,
            "Memory write failed"
        ) == 0
    )

    # ------------------------------------------------
    # Failure Policy:
    # Streaming 仍正常产生 chunk
    # ------------------------------------------------

    assert chunks == STREAM_CHUNKS

    assert captured_messages is not None

    system_message = captured_messages[0]

    # 降级为普通 Chat，不注入 Memory Context

    assert (
        "<memory_context>"
        not in system_message["content"]
    )


# ==================================================
# Case 4
# stream_chat() Memory Write Failure
#
# Memory Write 位于最后一个 yield 之后，
# 必须完整消费 generator 才真正执行到该阶段。
# ==================================================


def test_stream_chat_memory_write_failure_logging_contract(
    caplog
):

    caplog.set_level(logging.WARNING)

    captured_messages = None

    async def fake_stream_llm(messages):

        nonlocal captured_messages

        captured_messages = messages

        for chunk in STREAM_CHUNKS:
            yield chunk

    with (
        patch.object(
            chat_service,
            "get_conversation",
            fake_get_conversation
        ),
        patch.object(
            chat_service,
            "create_message",
            fake_create_message
        ),
        patch.object(
            chat_service,
            "get_messages_by_conversation",
            fake_get_messages
        ),
        patch.object(
            chat_service,
            "stream_llm",
            fake_stream_llm
        ),
        patch.object(
            chat_service,
            "memory_reader",
            SuccessMemoryReader()
        ),
        patch(
            "backend.memory.memory_write."
            "memory_pipeline.MemoryPipeline",
            FailingMemoryPipeline
        )
    ):

        chunks = consume_stream()

    # ------------------------------------------------
    # Logging Contract
    # ------------------------------------------------

    record = find_record(
        caplog,
        "Memory write failed"
    )

    assert_memory_write_contract(
        record,
        RuntimeError
    )

    assert (
        count_records(
            caplog,
            "Memory read failed"
        ) == 0
    )

    # ------------------------------------------------
    # Failure Policy:
    # stream chunks 正常完成，
    # Memory Write failure 不会破坏已完成的 stream
    # （如果异常传播出 generator，
    #  consume_stream 会直接抛错导致测试失败）
    # ------------------------------------------------

    assert chunks == STREAM_CHUNKS

    system_message = captured_messages[0]

    # Memory Read 路径不受影响

    assert (
        MEMORY_CONTEXT
        in system_message["content"]
    )
