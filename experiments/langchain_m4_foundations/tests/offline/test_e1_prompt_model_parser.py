import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue

from e1_prompt_model_parser import EXAMPLE, FAKE_TEXT, PROMPT, explain


def test_prompt_variables_and_message_types():
    value = PROMPT.invoke({"language": "French", "topic": "runnables"})
    assert isinstance(value, ChatPromptValue)
    system, human = value.to_messages()
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    assert system.content == "You teach software engineering. Answer in French."
    assert human.content == "Explain runnables in one short sentence."


def test_prompt_missing_variable():
    with pytest.raises(KeyError, match="topic"):
        PROMPT.invoke({"language": "English"})


def test_chain_extracts_string_from_model_message():
    model = FakeListChatModel(responses=[FAKE_TEXT])
    message = model.invoke(PROMPT.invoke(EXAMPLE))
    assert isinstance(message, AIMessage)
    result = explain(model, EXAMPLE)
    assert isinstance(result, str)
    assert type(result).__name__ == "TextAccessor"  # 已锁定的 Core 1.8.1
    assert type(str(result)) is str
    assert result == StrOutputParser().invoke(message) == FAKE_TEXT
