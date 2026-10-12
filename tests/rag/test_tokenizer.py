from backend.rag.retrieval.tokenizer import (
    JiebaTokenizer,
    RegexTokenizer,
    Tokenizer,
)


# ==================================================
# RegexTokenizer
# ==================================================

def test_regex_tokenizer_lowercases_and_splits():
    tokens = RegexTokenizer().tokenize(
        "FastAPI Validation Error"
    )

    assert tokens == [
        "fastapi",
        "validation",
        "error",
    ]


def test_regex_tokenizer_ignores_punctuation():
    tokens = RegexTokenizer().tokenize(
        "FastAPI, returns validation-error!"
    )

    assert tokens == [
        "fastapi",
        "returns",
        "validation",
        "error",
    ]


def test_regex_tokenizer_empty_text_returns_empty_list():
    assert RegexTokenizer().tokenize("") == []


# ==================================================
# JiebaTokenizer
# ==================================================

def test_jieba_tokenizer_splits_chinese():
    tokens = JiebaTokenizer().tokenize(
        "用户正在学习 AI Agent"
    )

    assert "用户" in tokens
    assert "学习" in tokens


def test_jieba_tokenizer_lowercases_ascii():
    tokens = JiebaTokenizer().tokenize("FastAPI")

    assert "fastapi" in tokens


def test_jieba_tokenizer_empty_text_returns_empty_list():
    assert JiebaTokenizer().tokenize("") == []


# ==================================================
# 标点 / 空白过滤（Tokenizer 边界）
# ==================================================

def test_jieba_tokenizer_drops_pure_punctuation():
    """
    jieba 会把标点单独切成 token，
    纯标点不允许成为有效 token。
    """

    tokens = JiebaTokenizer().tokenize("用户。，！？——")

    assert tokens == ["用户"]


def test_jieba_tokenizer_drops_whitespace_only_tokens():
    tokens = JiebaTokenizer().tokenize("用户 正在 学习")

    assert tokens == [
        "用户",
        "正在",
        "学习",
    ]


def test_jieba_tokenizer_punctuation_only_text_returns_empty_list():
    assert JiebaTokenizer().tokenize("。，！？——") == []


def test_regex_tokenizer_never_produces_punctuation():
    tokens = RegexTokenizer().tokenize("AI, Agent！(测试)")

    assert tokens == [
        "ai",
        "agent",
        "测试",
    ]


# ==================================================
# 接口一致性
# ==================================================

def test_all_tokenizers_share_the_same_interface():
    for tokenizer in (
        RegexTokenizer(),
        JiebaTokenizer(),
    ):
        assert isinstance(tokenizer, Tokenizer)

        tokens = tokenizer.tokenize("test")

        assert isinstance(tokens, list)

        assert all(
            isinstance(token, str)
            for token in tokens
        )
