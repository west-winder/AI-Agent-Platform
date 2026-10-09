
from backend.rag.contracts import Document


class MarkdownLoader:
    """Markdown 文件加载器：保留原文并构造 Document。"""

    def load(
        self,
        file_name: str,
        file_bytes: bytes,
        document_id: str,
    ) -> tuple[Document, str]:

        # 1. 仅接受 Markdown 格式
        if not file_name.lower().endswith((".md", ".markdown")):
            raise ValueError(
                "MarkdownLoader only accepts .md or .markdown files"
            )

        # 2. 严格 UTF-8 解码，保留 Markdown 标记
        raw_text = file_bytes.decode("utf-8-sig")

        # 3. 使用上层分配的身份构造 Document
        document = Document(
            document_id=document_id,
            file_name=file_name,
            file_type="markdown",
        )

        return document, raw_text
