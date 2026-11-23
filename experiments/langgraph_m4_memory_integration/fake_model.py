from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from memory_graph import MemoryDecision


class RecordingChatModel(GenericFakeChatModel):
    """记录真实模型输入，并支持基类的工具式结构化输出。"""

    _requests: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def requests(self) -> list[list[BaseMessage]]:
        return self._requests

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def decision_reply(should_save: bool, content: str | None) -> AIMessage:
    """构造模拟模型的原始工具调用，由 LangChain 解析为 MemoryDecision。"""
    return AIMessage(
        content="",
        tool_calls=[{
            "name": MemoryDecision.__name__,
            "args": {"should_save": should_save, "content": content},
            "id": "memory-decision",
        }],
    )
