
from backend.rag.contracts import Document


class TXTLoader:
    """TXT 文件加载器：解码文本并构造 Document。"""

    def load(
        self,
        file_name: str,
        file_bytes: bytes,
        document_id: str,
    ) -> tuple[Document, str]:

        # 1. 只接受 TXT 文件
        if not file_name.lower().endswith(".txt"):
            raise ValueError("TXTLoader only accepts .txt files")

        # 2. 严格解码；非法编码直接抛出异常
        raw_text = file_bytes.decode("utf-8-sig")

        # 3. 使用上层提供的身份信息构造 Document
        document = Document(
            document_id=document_id,
            file_name=file_name,
            file_type="txt",
        )

        return document, raw_text
