
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from sqlalchemy import select

from backend.database.database import SessionLocal
from backend.models.rag_document import DocumentORM
from backend.rag.ingestion.txt_loader import TXTLoader
from backend.rag.ingestion.rag_ingestion import RAGIngestionService


@dataclass(frozen=True)
class FileIngestionResult:
    document_id: str
    file_name: str
    status: Literal["success", "empty"]
    chunk_count: int


class FileIngestionService:
    """
    文件导入上层编排 V1。

    负责：
    - 校验上传文件名
    - 顺序上传时分配未使用的文件名
    - 生成 document_id
    - 调用 Loader
    - 调用已有 RAGIngestionService

    不负责：
    - 文件内容解析算法
    - Chunking / Embedding
    - 数据库写入事务
    - 并发文件名分配
    """

    def __init__(
        self,
        ingestion_service: RAGIngestionService,
        session_factory=SessionLocal,
        txt_loader: TXTLoader | None = None,
    ):
        self._ingestion = ingestion_service
        self._session_factory = session_factory
        self._txt_loader = txt_loader or TXTLoader()

    @staticmethod
    def _validate_file_name(file_name: str) -> None:
        if (
            not isinstance(file_name, str)
            or not file_name
            or file_name != file_name.strip()
        ):
            raise ValueError("invalid file_name")

        # V1 只接收文件名，不接收客户端路径
        if any(c in file_name for c in ("/", "\\", "\x00")):
            raise ValueError("file_name must not contain a path")

        if not file_name.lower().endswith(".txt"):
            raise ValueError("only .txt files are supported")

        if len(file_name) > 255:
            raise ValueError("file_name exceeds 255 characters")

    def _allocate_file_name(self, file_name: str) -> str:
        """读取已有名称，顺序寻找可用文件名。"""

        stem, extension = file_name.rsplit(".", 1)

        index = 0

        # 名称查询使用短会话，不跨越 Embedding 阶段
        with self._session_factory() as session:
            while True:
                candidate = (
                    file_name
                    if index == 0
                    else f"{stem}({index}).{extension}"
                )

                if len(candidate) > 255:
                    raise ValueError(
                        "allocated file_name exceeds 255 characters"
                    )

                statement = (
                    select(DocumentORM.document_id)
                    .where(DocumentORM.file_name == candidate)
                    .limit(1)
                )

                exists = session.scalar(statement) is not None

                if not exists:
                    return candidate

                index += 1

    def ingest_txt(
        self,
        file_name: str,
        file_bytes: bytes,
        chunk_size: int,
    ) -> FileIngestionResult:

        # 1. 输入校验
        self._validate_file_name(file_name)

        if not isinstance(file_bytes, bytes):
            raise TypeError("file_bytes must be bytes")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        # 2. 分配名称和独立身份
        allocated_name = self._allocate_file_name(file_name)
        document_id = uuid4().hex

        # 3. Loader：Bytes -> Document + raw_text
        document, raw_text = self._txt_loader.load(
            file_name=allocated_name,
            file_bytes=file_bytes,
            document_id=document_id,
        )

        # 4. 复用现有 Persistent Ingestion
        ingestion_result = self._ingestion.ingest(
            document=document,
            raw_text=raw_text,
            chunk_size=chunk_size,
        )

        # 5. 返回上层结果
        return FileIngestionResult(
            document_id=ingestion_result.document_id,
            file_name=allocated_name,
            status=ingestion_result.status,
            chunk_count=ingestion_result.chunk_count,
        )
