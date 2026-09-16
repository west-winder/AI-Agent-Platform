import asyncio
import sys
from pathlib import Path

from types import SimpleNamespace
from unittest.mock import patch


# ============================================================
# Project Root Bootstrap
#
# 本文件位于：
#
#   <PROJECT_ROOT>/test/memory/
#
# Python 运行脚本时，
# 只会把脚本所在目录放进 sys.path[0]，
# 不会自动把当前工作目录放进 sys.path。
#
# 所以无论是：
#
#   从 test/memory 运行
#   从项目根目录运行
#   由 pytest 导入
#
# sys.path 里都不会有项目根目录，
# 于是：
#
#   import backend...
#
# 会抛出：
#
#   ModuleNotFoundError: No module named 'backend'
#
# 这里根据 __file__ 反推项目根目录，
# 显式加入模块搜索路径，
# 使三种运行方式都能正常导入 backend。
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


import backend.services.chat_service as chat_service


# ============================================================
# Async Contract → 同步测试适配
#
# ChatService.chat 已经是 async Contract。
#
# 本模块的测试保持同步（脚本式运行，
# 同时可被 pytest 原生收集），
# 因此这里只把 coroutine 驱动到底：
#
#     1. 直接 python 运行时会真的执行
#     2. pytest 不会把测试静默跳过
#     3. coroutine 一定被 await
#     4. 不会产生假 PASS
# ============================================================


def run(coro):
    """
    在同步测试中真实执行并等待
    async production contract。
    """

    return asyncio.run(coro)


# ============================================================
# Shared Fake Data
# ============================================================

USER_MESSAGE = "我最近是不是在学习 LangGraph？"

MEMORY_CONTEXT = """
<memory_context>
用户最近开始学习 LangGraph
</memory_context>
""".strip()


def fake_get_conversation(
    db,
    conversation_id
):
    """
    模拟真实 Conversation。

    ChatService 当前只需要：

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
    模拟消息持久化。

    Assistant Message 最后需要 .id，
    所以返回一个带 id 的对象即可。
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
    """
    模拟 Conversation History。
    """

    return [
        SimpleNamespace(
            role="user",
            content=USER_MESSAGE
        )
    ]


class FakeMemoryPipeline:
    """
    Memory Write 不属于本次测试范围。

    所以这里只提供合法接口，
    防止真实 Memory Write 被调用。

    注意：

    ChatService 现在 await pipeline.process(...)，
    因此本 Fake 必须保持 async Calling Contract，
    否则会在 ChatService 内部触发 TypeError，
    被 Memory Write 的降级分支静默吞掉。
    """

    async def process(
        self,
        db,
        user_id,
        user_message
    ):
        return None


# ============================================================
# Case 1
# Memory Read Success
# ============================================================


class SuccessMemoryReader:
    """
    Fake Memory Reader（成功路径）。

    注意：

    ChatService 现在 await memory_reader.read(...)，
    因此本 Fake 必须保持 async Calling Contract。
    """

    def __init__(self):
        self.received_user_id = None
        self.received_query = None
        self.received_top_n = None
        self.received_top_k = None

    async def read(
        self,
        db,
        user_id,
        query,
        top_n,
        top_k
    ):
        self.received_user_id = user_id
        self.received_query = query
        self.received_top_n = top_n
        self.received_top_k = top_k

        return SimpleNamespace(
            memory_context=MEMORY_CONTEXT
        )


def test_memory_read_success():

    print()
    print("=" * 70)
    print("CASE 1 - CHAT MEMORY READ SUCCESS")
    print("=" * 70)

    fake_reader = SuccessMemoryReader()

    captured_messages = None

    async def fake_call_llm(messages):

        nonlocal captured_messages

        captured_messages = messages

        return "是的，你最近正在学习 LangGraph。"

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
            fake_reader
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

    # --------------------------------------------------------
    # MemoryReader Contract
    # --------------------------------------------------------

    assert fake_reader.received_user_id == 1

    assert (
        fake_reader.received_query
        == USER_MESSAGE
    )

    assert fake_reader.received_top_n == 20

    assert fake_reader.received_top_k == 5

    # --------------------------------------------------------
    # Chat LLM 必须真的被调用
    # --------------------------------------------------------

    assert captured_messages is not None

    # --------------------------------------------------------
    # 第一条应该是 System Message
    # --------------------------------------------------------

    system_message = captured_messages[0]

    assert (
        system_message["role"]
        == "system"
    )

    # --------------------------------------------------------
    # Agent Prompt 必须保留
    # --------------------------------------------------------

    assert (
        "你是一个测试 Agent"
        in system_message["content"]
    )

    # --------------------------------------------------------
    # Memory Context 必须被注入
    # --------------------------------------------------------

    assert (
        MEMORY_CONTEXT
        in system_message["content"]
    )

    # --------------------------------------------------------
    # Chat 正常返回
    # --------------------------------------------------------

    assert (
        response.answer
        == "是的，你最近正在学习 LangGraph。"
    )

    assert response.message_id == 200

    print()
    print("[PASS] Memory Context injected into Chat")


# ============================================================
# Case 2
# Memory Read Failure
# ============================================================


class FailingMemoryReader:
    """
    Fake Memory Reader（失败路径）。

    注意：

    ChatService 现在 await memory_reader.read(...)，
    因此本 Fake 必须保持 async Calling Contract：
    异常仍然在 await 时抛出，
    语义与迁移前完全一致。
    """

    async def read(
        self,
        db,
        user_id,
        query,
        top_n,
        top_k
    ):
        """
        模拟真实出现过的：

        LLM Judge 返回空内容
        """

        raise ValueError(
            "LLM Judge 返回了空内容"
        )


def test_memory_read_failure_degradation():

    print()
    print("=" * 70)
    print("CASE 2 - MEMORY READ FAILURE DEGRADATION")
    print("=" * 70)

    fake_reader = FailingMemoryReader()

    captured_messages = None

    async def fake_call_llm(messages):

        nonlocal captured_messages

        captured_messages = messages

        return "普通 Chat 仍然正常回答。"

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
            fake_reader
        ),
        patch(
            "backend.memory.memory_write."
            "memory_pipeline.MemoryPipeline",
            FakeMemoryPipeline
        )
    ):

        # ----------------------------------------------------
        # 这里如果 Memory Read Failure
        # 传播出了 chat()，
        # 测试会直接失败。
        # ----------------------------------------------------

        response = run(
            chat_service.chat(
                db=None,
                conversation_id=1,
                user_message=USER_MESSAGE
            )
        )

    # --------------------------------------------------------
    # 即使 Memory Read 失败，
    # Chat LLM 仍然必须执行。
    # --------------------------------------------------------

    assert captured_messages is not None

    system_message = captured_messages[0]

    # --------------------------------------------------------
    # Agent Prompt 仍然存在
    # --------------------------------------------------------

    assert (
        "你是一个测试 Agent"
        in system_message["content"]
    )

    # --------------------------------------------------------
    # 失败后不应注入 Memory Context
    # --------------------------------------------------------

    assert (
        "<memory_context>"
        not in system_message["content"]
    )

    # --------------------------------------------------------
    # Chat 必须仍然正常返回
    # --------------------------------------------------------

    assert (
        response.answer
        == "普通 Chat 仍然正常回答。"
    )

    assert response.message_id == 200

    print()
    print(
        "[PASS] Memory Read failed, "
        "Chat degraded normally"
    )


# ============================================================
# Main
# ============================================================


def main():

    print()
    print("=" * 70)
    print(
        "CHAT SERVICE - MEMORY REGRESSION TEST"
    )
    print("=" * 70)

    test_memory_read_success()

    test_memory_read_failure_degradation()

    print()
    print("=" * 70)
    print(
        "ALL CHAT SERVICE MEMORY "
        "REGRESSION TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()