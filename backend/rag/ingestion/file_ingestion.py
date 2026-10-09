from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from sqlalchemy import select

from backend.database.database import SessionLocal
from backend.models.rag_document import DocumentORM
from backend.rag.ingestion.markdown_loader import MarkdownLoader
from backend.rag.ingestion.pdf_loader import PDFLoader
from backend.rag.ingestion.rag_ingestion import RAGIngestionService
from backend.rag.ingestion.txt_loader import TXTLoader


@dataclass(frozen=True)
class FileIngestionResult:
    document_id: str
    file_name: str
    status: Literal["success", "empty"]
    chunk_count: int


class FileIngestionService:
    """
    文件导入的上层业务编排。

    负责：
    - 输入校验
    - Loader 选择
    - 文件名冲突处理
    - document_id 生成
    - 调用 RAGIngestionService
    - 返回最终结果

    不负责：
    - 文本解析算法
    - Chunking / Embedding
    - 数据库写入事务
    - 并发名称分配
    """

    def __init__(
        self,
        ingestion_service: RAGIngestionService,
        session_factory=SessionLocal,
        txt_loader: TXTLoader | None = None,
        markdown_loader: MarkdownLoader | None = None,
        pdf_loader: PDFLoader | None = None,
    ):
        self._ingestion = ingestion_service
        self._session_factory = session_factory
        self._txt_loader = txt_loader or TXTLoader()
        self._markdown_loader = (
            markdown_loader or MarkdownLoader()
        )
        self._pdf_loader = pdf_loader or PDFLoader()

    # ---------------------------------------
    # 1. 公共文件名校验
    # ---------------------------------------

    @staticmethod
    def _validate_file_name(file_name: str) -> None:
        if (
            not isinstance(file_name, str)
            or not file_name
            or file_name != file_name.strip()
        ):
            raise ValueError("invalid file_name")

        if any(c in file_name for c in ("/", "\\", "\x00")):
            raise ValueError("file_name must not contain a path")

        if len(file_name) > 255:
            raise ValueError("file_name exceeds 255 characters")

    # ---------------------------------------
    # 2. Loader 选择
    # ---------------------------------------

    def _select_loader(self, file_name: str):
        lower_name = file_name.lower()

        if lower_name.endswith(".txt"):
            return self._txt_loader

        if lower_name.endswith((".md", ".markdown")):
            return self._markdown_loader

        if lower_name.endswith(".pdf"):
            return self._pdf_loader

        raise ValueError("unsupported file type")

    # ---------------------------------------
    # 3. 文件名冲突处理（保留原有逻辑）
    # ---------------------------------------

    def _allocate_file_name(self, file_name: str) -> str:
        """顺序寻找数据库中尚未使用的文件名。"""

        stem, extension = file_name.rsplit(".", 1)
        index = 0

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

    # ---------------------------------------
    # 4. 公共 File Ingestion 入口
    # ---------------------------------------

    def ingest_file(
        self,
        file_name: str,
        file_bytes: bytes,
        chunk_size: int,
    ) -> FileIngestionResult:

        # Step 1: 输入校验
        self._validate_file_name(file_name)

        if not isinstance(file_bytes, bytes):
            raise TypeError("file_bytes must be bytes")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        # Step 2: 选择对应 Loader
        loader = self._select_loader(file_name)

        # Step 3: 文件名及身份管理
        allocated_name = self._allocate_file_name(file_name)
        document_id = uuid4().hex

        # Step 4: 文件解析
        document, raw_text = loader.load(
            file_name=allocated_name,
            file_bytes=file_bytes,
            document_id=document_id,
        )

        # Step 5: 复用已有持久化 Ingestion
        ingestion_result = self._ingestion.ingest(
            document=document,
            raw_text=raw_text,
            chunk_size=chunk_size,
        )

        # Step 6: 返回最终结果
        return FileIngestionResult(
            document_id=ingestion_result.document_id,
            file_name=allocated_name,
            status=ingestion_result.status,
            chunk_count=ingestion_result.chunk_count,
        )

    # ---------------------------------------
    # 5. TXT 旧接口：保持兼容
    # ---------------------------------------

    def ingest_txt(
        self,
        file_name: str,
        file_bytes: bytes,
        chunk_size: int,
    ) -> FileIngestionResult:

        self._validate_file_name(file_name)

        if not file_name.lower().endswith(".txt"):
            raise ValueError("only .txt files are supported")

        return self.ingest_file(
            file_name=file_name,
            file_bytes=file_bytes,
            chunk_size=chunk_size,
        )