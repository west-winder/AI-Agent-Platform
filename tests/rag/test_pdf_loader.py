import pytest
import pymupdf


from backend.rag.ingestion.pdf_loader import (
    PDFLoader,
    NoExtractableTextError,
    IncompletePDFExtractionError,
    EncryptedPDFError,
)


def _build_pdf(page_builders) -> bytes:
    """
    Build a small in-memory PDF for deterministic tests.

    Each page_builder receives a newly created PyMuPDF page
    and can add text / drawings / nothing.
    """
    document = pymupdf.open()

    try:
        for page_builder in page_builders:
            page = document.new_page()
            page_builder(page)

        return document.tobytes()
    finally:
        document.close()


def _text_page(text: str):
    def build(page):
        page.insert_text((72, 72), text)

    return build


def _blank_page(page):
    pass


def _visible_graphic_page(page):
    # Visible content, but no extractable text.
    page.draw_rect(
        pymupdf.Rect(72, 72, 300, 300),
        color=(0, 0, 0),
        fill=(0, 0, 0),
    )


def test_pdf_loader_loads_single_text_page():
    loader = PDFLoader()

    file_bytes = _build_pdf([
        _text_page("RAG retrieval generation"),
    ])

    document, raw_text = loader.load(
        file_name="rag.pdf",
        file_bytes=file_bytes,
        document_id="doc-1",
    )

    assert document.document_id == "doc-1"
    assert document.file_name == "rag.pdf"
    assert document.file_type == "pdf"
    assert raw_text == "RAG retrieval generation"


def test_pdf_loader_preserves_multi_page_order():
    loader = PDFLoader()

    file_bytes = _build_pdf([
        _text_page("Page One"),
        _text_page("Page Two"),
        _text_page("Page Three"),
    ])

    _, raw_text = loader.load(
        file_name="multi.pdf",
        file_bytes=file_bytes,
        document_id="doc-2",
    )

    assert raw_text == "Page One\n\nPage Two\n\nPage Three"


def test_pdf_loader_allows_blank_page_between_text_pages():
    loader = PDFLoader()

    file_bytes = _build_pdf([
        _text_page("Before"),
        _blank_page,
        _text_page("After"),
    ])

    _, raw_text = loader.load(
        file_name="blank-page.pdf",
        file_bytes=file_bytes,
        document_id="doc-3",
    )

    assert raw_text == "Before\n\nAfter"


def test_pdf_loader_rejects_fully_blank_pdf():
    loader = PDFLoader()

    file_bytes = _build_pdf([
        _blank_page,
        _blank_page,
    ])

    with pytest.raises(NoExtractableTextError):
        loader.load(
            file_name="blank.pdf",
            file_bytes=file_bytes,
            document_id="doc-4",
        )


def test_pdf_loader_rejects_visible_page_without_extractable_text():
    loader = PDFLoader()

    file_bytes = _build_pdf([
        _text_page("Normal text"),
        _visible_graphic_page,
    ])

    with pytest.raises(
        IncompletePDFExtractionError,
        match=r"2",
    ):
        loader.load(
            file_name="incomplete.pdf",
            file_bytes=file_bytes,
            document_id="doc-5",
        )


def test_pdf_loader_accepts_uppercase_pdf_extension():
    loader = PDFLoader()

    file_bytes = _build_pdf([
        _text_page("Uppercase extension"),
    ])

    document, raw_text = loader.load(
        file_name="RAG.PDF",
        file_bytes=file_bytes,
        document_id="doc-6",
    )

    assert document.file_type == "pdf"
    assert raw_text == "Uppercase extension"


def test_pdf_loader_rejects_non_pdf_extension():
    loader = PDFLoader()

    with pytest.raises(
        ValueError,
        match=r"only accepts \.pdf",
    ):
        loader.load(
            file_name="rag.txt",
            file_bytes=b"anything",
            document_id="doc-7",
        )


def test_pdf_loader_rejects_corrupted_pdf():
    loader = PDFLoader()

    with pytest.raises(Exception):
        loader.load(
            file_name="broken.pdf",
            file_bytes=b"this is not a valid pdf",
            document_id="doc-8",
        )


def _build_encrypted_pdf() -> bytes:
    document = pymupdf.open()

    try:
        page = document.new_page()
        page.insert_text((72, 72), "Secret RAG content")

        return document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )
    finally:
        document.close()


def test_pdf_loader_rejects_encrypted_pdf():
    loader = PDFLoader()

    file_bytes = _build_encrypted_pdf()

    with pytest.raises(
        EncryptedPDFError,
        match=r"Encrypted PDF is not supported",
    ):
        loader.load(
            file_name="encrypted.pdf",
            file_bytes=file_bytes,
            document_id="doc-9",
        )