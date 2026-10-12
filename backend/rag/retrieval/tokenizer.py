import re


# 至少包含一个 word 字符才视为有效 token。
_WORD_CHARACTER = re.compile(r"\w", re.UNICODE)


def _is_valid_token(token) -> bool:
    """
    统一的 token 有效性判定。

    过滤：

    - 非字符串
    - whitespace-only
    - 纯标点 / 纯符号（不含任何 word 字符）

    所有 Tokenizer 共用同一判定，
    避免分词器把标点当作可检索 token 带进 Retrieval。
    """

    if not isinstance(token, str):
        return False

    if not token.strip():
        return False

    return _WORD_CHARACTER.search(token) is not None


class Tokenizer:
    """
    文本 Tokenizer 接口。

    职责：

    text
        ↓
    token[]

    本组件不负责：

    1. 打分
    2. 排序
    3. 文本存储访问
    4. Hydration

    实现可以是纯 Python，
    也可以接入中文分词等外部能力。
    """

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError


class RegexTokenizer(Tokenizer):
    """
    默认 Tokenizer：纯 Python 实现。

    先转小写，再按 Unicode 单词字符切分。

    注意：
    连续中文会被切成一个 token。
    需要中文分词时使用 JiebaTokenizer。
    """

    _PATTERN = re.compile(r"\w+", re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in self._PATTERN.findall(
                text.lower()
            )
            if _is_valid_token(token)
        ]


class JiebaTokenizer(Tokenizer):
    """
    中文分词 Tokenizer。

    jieba 会把标点与空白单独切成 token，
    因此必须经过 _is_valid_token 过滤，
    否则纯标点会变成可检索 token。

    jieba 采用延迟导入：
    只有真正构造 JiebaTokenizer 时才加载词典，
    因此 BM25Retriever 默认路径不依赖 jieba。
    """

    def __init__(self):
        import jieba

        self._jieba = jieba

    def tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in self._jieba.cut_for_search(
                text.lower()
            )
            if _is_valid_token(token)
        ]
