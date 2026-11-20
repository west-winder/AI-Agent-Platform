from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You teach software engineering. Answer in {language}."),
        ("human", "Explain {topic} in one short sentence."),
    ]
)
EXAMPLE = {"language": "English", "topic": "LCEL"}
FAKE_TEXT = "LCEL connects runnable components."


def explain(model: BaseChatModel, variables: dict[str, str]) -> str:
    # LCEL 把模板生成的消息交给模型，再把模型消息交给文本解析器。
    return (PROMPT | model | StrOutputParser()).invoke(variables)


if __name__ == "__main__":
    model = FakeListChatModel(responses=[FAKE_TEXT])
    prompt_value = PROMPT.invoke(EXAMPLE)
    message = model.invoke(prompt_value)
    result = explain(model, EXAMPLE)
    print(type(EXAMPLE).__name__, "->", type(prompt_value).__name__)
    for prompt_message in prompt_value.to_messages():
        print(type(prompt_message).__name__, prompt_message.content)
    print(type(message).__name__, "->", type(result).__name__, repr(result))
    print("isinstance(result, str) =", isinstance(result, str))
