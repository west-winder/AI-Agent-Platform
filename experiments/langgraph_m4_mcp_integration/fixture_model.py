import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr


class FixtureChatModel(BaseChatModel):
    """第一次请求网页工具，收到 ToolMessage 后从可见文本生成确定性回答。"""

    url: str
    _requests: list[tuple[list[BaseMessage], list[dict]]] = PrivateAttr(default_factory=list)

    @property
    def requests(self) -> list[tuple[list[BaseMessage], list[dict]]]:
        return self._requests

    @property
    def _llm_type(self) -> str:
        return "mcp-web-page-fixture"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, *, tools, **kwargs):
        self._requests.append((list(messages), tools))
        last = messages[-1]
        if isinstance(last, ToolMessage):
            if last.status == "error":
                reply = AIMessage(content=f"抓取失败：{last.text}")
            else:
                reply = AIMessage(content=json.loads(last.text)["content"])
        else:
            reply = AIMessage(
                content="",
                tool_calls=[{"name": "fetch_web_page", "args": {"url": self.url}, "id": "fetch-1"}],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])
