import pytest
from langchain_core.runnables import RunnableSequence

from e2_runnable_lcel import build_sequence


@pytest.mark.parametrize(
    ("text", "words", "summary"),
    [
        (
            "  Prompt   Model Parser  ",
            ["Prompt", "Model", "Parser"],
            {"count": 3, "joined": "Prompt/Model/Parser"},
        ),
        ("  ", [], {"count": 0, "joined": ""}),
    ],
)
def test_runnable_sequence_passes_intermediate_data(text, words, summary):
    sequence = build_sequence()
    assert isinstance(sequence, RunnableSequence)
    first, second = sequence.steps
    intermediate = first.invoke(text)
    assert intermediate == words
    assert sequence.invoke(text) == second.invoke(intermediate) == summary
