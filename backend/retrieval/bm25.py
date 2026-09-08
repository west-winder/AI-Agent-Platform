from dataclasses import dataclass

import bm25s
import jieba

from bm25s.tokenization import Tokenizer


@dataclass
class BM25Result:
    index: int
    text: str
    score: float


def jieba_splitter(text: str) -> list[str]:
    tokens = jieba.cut_for_search(text)

    return [
        token
        for token in tokens
        if token.strip()
    ]


class BM25Retriever:

    def __init__(self):
        self._texts: list[str] = []
        self._tokenizer: Tokenizer | None = None
        self._bm25: bm25s.BM25 | None = None

    def index(
        self,
        texts: list[str]
    ) -> None:

        self._texts = list(texts)

        self._tokenizer = Tokenizer(
            splitter=jieba_splitter
        )

        corpus_tokens = self._tokenizer.tokenize(
            self._texts,
            return_as="tuple"
        )

        self._bm25 = bm25s.BM25()

        self._bm25.index(
            corpus_tokens
        )


    def search(
        self,
        query: str,
        top_n: int = 5
    ) -> list[BM25Result]:

        if self._tokenizer is None or self._bm25 is None:
            raise RuntimeError(
                "BM25Retriever 必须先调用 index()，再调用 search()"
            )

        query_tokens = self._tokenizer.tokenize(
            [query],
            update_vocab=False,
            return_as="tuple"
        )

        results = self._bm25.retrieve(
            query_tokens,
            k=top_n
        )

        doc_indexes = results.documents[0]
        scores = results.scores[0]

        bm25_results = []

        for doc_index, score in zip(
            doc_indexes,
            scores
        ):
            result = BM25Result(
                index=int(doc_index),
                text=self._texts[int(doc_index)],
                score=float(score)
            )

            bm25_results.append(result)

        return bm25_results