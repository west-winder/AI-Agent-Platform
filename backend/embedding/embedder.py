import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    通用 Embedding 模块。

    当前使用本地 SentenceTransformer 模型：

    Qwen3-Embedding-0.6B

    职责：

    文本
        ↓
    Embedding Model
        ↓
    Embedding Vector

    注意：

    本模块不负责：

    1. Memory
    2. Similarity Search
    3. Top-K
    4. Threshold
    5. LLM Judge
    6. 数据库操作
    """

    def __init__(self):
        """
        初始化 Embedding Model。
        """

        load_dotenv()

        model_path = os.getenv(
            "EMBEDDING_MODEL_PATH"
        )

        if not model_path:
            model_path = "all-MiniLM-L6-v2"
            print(
                "[Embedder] 警告：未设置 EMBEDDING_MODEL_PATH，"
                "回退到默认模型 all-MiniLM-L6-v2"
            )

        print(
            f"[Embedder] 正在加载模型：{model_path}"
        )

        self.model = SentenceTransformer(
            model_path
        )

        print(
            "[Embedder] Embedding Model 加载完成"
        )

    # ==================================================
    # 单条文本 Embedding
    # ==================================================

    def embed(
        self,
        text: str
    ):
        """
        将单条文本转换为 Embedding Vector。

        参数：
            text: 要进行 Embedding 的文本

        返回：
            numpy.ndarray
        """

        if not isinstance(text, str):
            raise TypeError(
                "text必须是str类型"
            )

        if not text.strip():
            raise ValueError(
                "text不能为空"
            )

        embedding = self.model.encode(
            text
        )

        return embedding

    # ==================================================
    # 批量文本 Embedding
    # ==================================================

    def embed_batch(
        self,
        texts: list[str]
    ):
        """
        批量生成 Embedding。

        参数：
            texts: 文本列表

        返回：
            numpy.ndarray
        """

        if not isinstance(texts, list):
            raise TypeError(
                "texts必须是list[str]"
            )

        if not texts:
            return []

        for text in texts:

            if not isinstance(text, str):
                raise TypeError(
                    "texts中的每个元素必须是str"
                )

            if not text.strip():
                raise ValueError(
                    "texts中不能存在空字符串"
                )

        embeddings = self.model.encode(
            texts
        )

        return embeddings

    # ==================================================
    # Embedding维度
    # ==================================================

    @property
    def dimension(self) -> int:
        """
        返回当前 Embedding Model 的向量维度。
        """

        return self.model.get_embedding_dimension()