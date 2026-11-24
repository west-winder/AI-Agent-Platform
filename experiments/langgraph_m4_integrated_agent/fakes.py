from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from backend.rag.contracts import Chunk, RagRetrievedChunk

from .memory import MemoryDecision


class FakeRetriever:
    def __init__(self, results: dict[str, list[RagRetrievedChunk] | Exception]):
        self.results = results
        self.calls: list[tuple[str, dict[str, Chunk], int]] = []

    def retrieve(
        self, query: str, chunk_store: dict[str, Chunk], top_k: int = 5
    ) -> list[RagRetrievedChunk]:
        self.calls.append((query, chunk_store, top_k))
        result = self.results[query]
        if isinstance(result, Exception):
            raise result
        return result


class ScriptedChatModel(GenericFakeChatModel):
    """执行真实的工具绑定和结构化解析入口，回复由测试脚本控制。"""

    _requests: list[tuple[list[BaseMessage], list[dict]]] = PrivateAttr(default_factory=list)

    @property
    def requests(self) -> list[tuple[list[BaseMessage], list[dict]]]:
        return self._requests

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, *, tools, **kwargs):
        self._requests.append((list(messages), tools))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_request(*calls: tuple[str, str, dict]) -> AIMessage:
    """每个元组按 (调用 ID, 工具名, 参数) 排列。"""
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": call_id} for call_id, name, args in calls
    ])


def memory_reply(should_save: bool, content: str | None) -> AIMessage:
    return tool_request(("memory-1", MemoryDecision.__name__, {"should_save": should_save, "content": content}))
