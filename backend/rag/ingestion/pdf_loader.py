import pymupdf

from backend.rag.contracts import Document


class NoExtractableTextError(ValueError):
    """PDF does not contain any extractable text."""


class IncompletePDFExtractionError(ValueError):
    """PDF contains visible content that could not be extracted as text."""


class EncryptedPDFError(ValueError):
    """Encrypted PDF is not supported."""


class PDFLoader:
    # V1.1 heuristic:
    # 灰度值越接近 255 越接近白色。
    _WHITE_THRESHOLD = 245

    # 非白像素超过总像素的 0.2%，认为页面存在明显视觉内容。
    _VISIBLE_PIXEL_RATIO = 0.002

    def load(
        self,
        file_name: str,
        file_bytes: bytes,
        document_id: str,
    ) -> tuple[Document, str]:

        if not file_name.lower().endswith(".pdf"):
            raise ValueError("PDFLoader only accepts .pdf files")

        page_texts: list[str] = []
        suspicious_pages: list[int] = []

        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise EncryptedPDFError(
                    f"Encrypted PDF is not supported: {file_name}"
                )

            for page_index, page in enumerate(pdf):
                page_number = page_index + 1

                text = page.get_text("text").strip()

                if text:
                    page_texts.append(text)
                    continue

                # 页面没有提取出文字：
                # 再判断它是否真的接近空白。
                if self._page_has_visible_content(page):
                    suspicious_pages.append(page_number)

        # 只要发现“无文本但明显有视觉内容”的页面，
        # 就不能把整份 PDF 当作完整解析成功。
        if suspicious_pages:
            raise IncompletePDFExtractionError(
                "PDF contains visible content without extractable text "
                f"on pages: {suspicious_pages}"
            )

        # 没有 suspicious page，但整份 PDF 仍然一段文字都没有。
        # 此时可能就是空白 PDF。
        if not page_texts:
            raise NoExtractableTextError(
                f"No extractable text found in PDF: {file_name}"
            )

        raw_text = "\n\n".join(page_texts)

        document = Document(
            document_id=document_id,
            file_name=file_name,
            file_type="pdf",
        )

        return document, raw_text

    def _page_has_visible_content(self, page: pymupdf.Page) -> bool:
        """
        Heuristic check for visible non-white page content.

        This does not prove semantic content exists.
        It only helps distinguish approximately blank pages from pages
        that visibly contain something but produced no extractable text.
        """

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(0.5, 0.5),
            colorspace=pymupdf.csGRAY,
            alpha=False,
        )

        pixels = pixmap.samples

        if not pixels:
            return False

        non_white_pixels = sum(
            pixel < self._WHITE_THRESHOLD
            for pixel in pixels
        )

        ratio = non_white_pixels / len(pixels)

        return ratio >= self._VISIBLE_PIXEL_RATIO