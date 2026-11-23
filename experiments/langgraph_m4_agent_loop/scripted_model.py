from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr


class ScriptedChatModel(GenericFakeChatModel):
    """逐次读取预设回复，并记录收到的对话和工具定义。"""

    _requests: list[tuple[list[BaseMessage], list[dict]]] = PrivateAttr(default_factory=list)

    @property
    def requests(self) -> list[tuple[list[BaseMessage], list[dict]]]:
        return self._requests

    def bind_tools(self, tools, **kwargs):
        # GenericFakeChatModel 负责返回消息。本方法补上本实验需要的工具绑定。
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, *, tools, **kwargs):
        self._requests.append((list(messages), tools))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def weather_request(*calls: tuple[str, str]) -> AIMessage:
    """用 (调用 ID, 城市) 模拟模型提出的一组 Weather Tool Call。"""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "get_weather", "args": {"city": city}, "id": call_id}
            for call_id, city in calls
        ],
    )
