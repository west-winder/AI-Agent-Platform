from langchain_core.runnables import RunnableLambda, RunnableSequence


def split_words(text: str) -> list[str]:
    return text.split()


def summarize_words(words: list[str]) -> dict[str, int | str]:
    return {"count": len(words), "joined": "/".join(words)}


def build_sequence() -> RunnableSequence:
    # 左边返回的 list[str] 就是右边接收的参数。
    return RunnableLambda(split_words) | RunnableLambda(summarize_words)


if __name__ == "__main__":
    text = "  Prompt   Model Parser  "
    sequence = build_sequence()
    print("str -> list[str]", split_words(text))
    print(type(sequence).__name__, "-> dict", sequence.invoke(text))
