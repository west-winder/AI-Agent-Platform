import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    """
    单条 Reranker 排序结果。

    属性：
        index:
            文本在原始 texts 列表中的位置。

        text:
            原始文本。

        score:
            Cross Encoder 计算得到的相关性分数。
            分数越高，表示 query 和 text 越相关。
    """

    index: int
    text: str
    score: float


class CrossEncoderReranker:
    """
    通用 Cross Encoder Reranker。

    当前使用本地模型：

    BAAI/bge-reranker-v2-m3

    职责：

    Query
        +
    Text Candidates
        ↓
    Cross Encoder
        ↓
    Relevance Score
        ↓
    Ranking

    本模块只负责文本候选的精排。

    不负责：

    1. Memory
    2. Database
    3. Embedding
    4. Candidate Generation
    5. Vector Search
    6. LLM Judge
    7. Memory Injection
    """

    def __init__(self):
        """
        初始化本地 Cross Encoder 模型。
        """

        load_dotenv()

        model_path = os.getenv(
            "RERANKER_MODEL_PATH"
        )

        if not model_path:
            raise ValueError(
                "未设置 RERANKER_MODEL_PATH"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Reranker 模型路径不存在：{model_path}"
            )

        print(
            f"[CrossEncoderReranker] 正在加载模型：{model_path}"
        )

        self.model = CrossEncoder(
            model_path,
            device="cpu",
            local_files_only=True,
            activation_fn=torch.nn.Identity()
        )

        print(
            "[CrossEncoderReranker] Reranker Model 加载完成"
        )

    def rerank(
        self,
        query: str,
        texts: list[str]
    ) -> list[RerankResult]:
        """
        对候选文本进行重新排序。

        参数：
            query:
                用户查询文本。

            texts:
                Candidate Generation 阶段召回的候选文本列表。

        返回：
            list[RerankResult]

            按照 rerank score 从高到低排序后的结果。
        """

        if not isinstance(query, str):
            raise TypeError(
                "query 必须是 str 类型"
            )

        if not query.strip():
            raise ValueError(
                "query 不能为空"
            )

        if not isinstance(texts, list):
            raise TypeError(
                "texts 必须是 list[str]"
            )

        if not texts:
            return []

        for text in texts:

            if not isinstance(text, str):
                raise TypeError(
                    "texts 中的每个元素必须是 str"
                )

            if not text.strip():
                raise ValueError(
                    "texts 中不能存在空字符串"
                )

        pairs = [
            (query, text)
            for text in texts
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        results = [
            RerankResult(
                index=index,
                text=text,
                score=float(score)
            )
            for index, (text, score) in enumerate(
                zip(texts, scores)
            )
        ]

        results.sort(
            key=lambda result: result.score,
            reverse=True
        )

        return results