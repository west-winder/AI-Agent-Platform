import math

from backend.rag.contracts import BM25RetrieverResult
from backend.rag.retrieval.tokenizer import (
    RegexTokenizer,
    Tokenizer,
)


# ==================================================
# BM25 Okapi 参数
# ==================================================

BM25_K1 = 1.5
BM25_B = 0.75


class BM25Retriever:
    """
    RAG 的 BM25 Retrieval Engine。

    职责：

    Query
        ↓
    Query Tokenize
        ↓
    与候选 Chunk 文本做 BM25 打分
        ↓
    Ranking（score 降序）+ 1-based bm25_rank
        ↓
    返回 Top-K BM25RetrieverResult[]

    本组件只做：
    query + text matching + ranking

    本组件不负责：

    1. Query Embedding
    2. Vector Search
    3. Hydration（不产出 RagRetrievedChunk）
    4. Context Assembly
    5. Fusion / RRF
    6. Reranking
    7. LLM Generation

    它与 DenseRetriever 是并列的 Retrieval Engine，
    二者互不感知。

    文本切分交给注入的 Tokenizer，
    本组件只负责打分与排序。

    只返回至少命中一个 Query term 的 Chunk：
    score = 0 表示没有命中，不进入候选池。
    """

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
    ):
        """
        注入 Tokenizer。

        默认使用 RegexTokenizer（纯 Python，无额外依赖）；
        需要中文分词时可注入 JiebaTokenizer。
        """

        self._tokenizer = tokenizer or RegexTokenizer()

    def search(
        self,
        query: str,
        chunk_texts: dict[str, str],
        top_k: int = 5,
    ) -> list[BM25RetrieverResult]:
        """
        对候选 Chunk 文本执行一次 BM25 检索。

        参数：
            query:
                查询文本。

            chunk_texts:
                chunk_id -> 文本 的显式映射。
                与 DenseRetriever 的 chunk_id -> vector 对称。

            top_k:
                最多返回多少个 BM25RetrieverResult。

        返回：
            按 bm25_score 从高到低排列的 BM25RetrieverResult
            （含 1-based bm25_rank）。
        """

        # --------------------------------------------------
        # 1. 参数校验
        # --------------------------------------------------

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be int")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not isinstance(chunk_texts, dict):
            raise TypeError(
                "chunk_texts must be dict[str, str]"
            )

        if not isinstance(query, str):
            raise TypeError("query must be str")

        if not query.strip():
            raise ValueError("query must not be blank")

        # --------------------------------------------------
        # 2. Empty Corpus 是合法状态
        # --------------------------------------------------

        if not chunk_texts:
            return []

        # --------------------------------------------------
        # 3. Corpus Tokenize
        # --------------------------------------------------

        tokenized_corpus: list[tuple[str, list[str]]] = []

        for chunk_id, text in chunk_texts.items():

            if not isinstance(chunk_id, str):
                raise TypeError("chunk_id must be str")

            if not chunk_id.strip():
                raise ValueError("chunk_id must not be blank")

            if not isinstance(text, str):
                raise TypeError("chunk text must be str")

            tokenized_corpus.append(
                (chunk_id, self._tokenizer.tokenize(text))
            )

        # --------------------------------------------------
        # 4. 语料统计：文档频率 / 平均文档长度
        # --------------------------------------------------

        document_count = len(tokenized_corpus)

        document_frequency: dict[str, int] = {}

        for _, tokens in tokenized_corpus:
            for token in set(tokens):
                document_frequency[token] = (
                    document_frequency.get(token, 0) + 1
                )

        total_length = sum(
            len(tokens)
            for _, tokens in tokenized_corpus
        )

        average_length = total_length / document_count

        # 所有文档都没有任何 Token：
        # 不存在可检索内容，Empty Retrieval 是合法状态。
        if average_length == 0:
            return []

        # --------------------------------------------------
        # 5. Query Tokenize（去重后按词求和）
        # --------------------------------------------------

        query_tokens = list(
            dict.fromkeys(
                self._tokenizer.tokenize(query)
            )
        )

        if not query_tokens:
            return []

        # --------------------------------------------------
        # 6. 逐个 Chunk 计算 BM25 Score
        # --------------------------------------------------

        scored: list[tuple[str, float]] = []

        for chunk_id, tokens in tokenized_corpus:

            score = self._bm25_score(
                tokens=tokens,
                query_tokens=query_tokens,
                document_frequency=(
                    document_frequency
                ),
                document_count=document_count,
                average_length=average_length,
            )

            # 与 Query 没有任何词重合的 Chunk 不是检索结果：
            # BM25RetrieverResult 表示「命中了哪个 Chunk」，
            # score = 0 表示没有命中，不进入候选池，
            # 否则会污染下游 Fusion。
            if score <= 0.0:
                continue

            scored.append((chunk_id, score))

        # --------------------------------------------------
        # 7. Ranking
        # --------------------------------------------------

        # score 降序；list.sort 稳定，保留入参顺序作为 tie-break。
        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # --------------------------------------------------
        # 8. Top-K + 1-based bm25_rank
        # --------------------------------------------------

        return [
            BM25RetrieverResult(
                chunk_id=chunk_id,
                bm25_score=score,
                bm25_rank=rank,
            )
            for rank, (chunk_id, score) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]

    @staticmethod
    def _bm25_score(
        *,
        tokens: list[str],
        query_tokens: list[str],
        document_frequency: dict[str, int],
        document_count: int,
        average_length: float,
    ) -> float:
        """
        计算单个 Chunk 的 BM25 Okapi Score。

        IDF 使用 log( (N - n + 0.5) / (n + 0.5) + 1 )，
        保证 IDF 非负。
        """

        length = len(tokens)

        term_frequency: dict[str, int] = {}

        for token in tokens:
            term_frequency[token] = (
                term_frequency.get(token, 0) + 1
            )

        length_norm = (
            BM25_K1
            * (
                1.0
                - BM25_B
                + BM25_B * length / average_length
            )
        )

        score = 0.0

        for token in query_tokens:

            frequency = term_frequency.get(token, 0)

            if frequency == 0:
                continue

            idf = math.log(
                (
                    document_count
                    - document_frequency[token]
                    + 0.5
                )
                / (document_frequency[token] + 0.5)
                + 1.0
            )

            score += (
                idf
                * (frequency * (BM25_K1 + 1.0))
                / (frequency + length_norm)
            )

        return score
