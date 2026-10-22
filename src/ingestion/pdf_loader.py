"""Loader implementation for PDF documents using pypdf."""

from pathlib import Path
from typing import Union
from pypdf import PdfReader
from src.domain.documents import Document, DocumentMetadata, DocumentType, ParsedPage
from src.utils.file_utils import compute_file_hash
from src.utils.text_utils import normalize_whitespace


class PDFLoader:
    """Extracts text, pages, and metadata from PDF documents."""

    def load(self, file_path: Union[str, Path]) -> Document:
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF file does not exist: {path}")

        file_hash = compute_file_hash(path)
        file_size = path.stat().st_size
        doc_id = f"doc_{file_hash[:16]}"

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)

        parsed_pages: list[ParsedPage] = []
        full_text_accumulator: list[str] = []

        for idx, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned_page_text = normalize_whitespace(raw_text)

            if cleaned_page_text:
                parsed_pages.append(
                    ParsedPage(
                        page_number=idx,
                        text=cleaned_page_text,
                        metadata={"source_page": idx, "doc_id": doc_id},
                    )
                )
                full_text_accumulator.append(cleaned_page_text)

        concatenated_text = "\n\n".join(full_text_accumulator)

        # Attempt to read PDF embedded title from metadata if available
        pdf_title = None
        if reader.metadata and reader.metadata.title:
            pdf_title = str(reader.metadata.title).strip()
        if not pdf_title:
            pdf_title = path.stem.replace("_", " ").title()

        metadata = DocumentMetadata(
            source_path=str(path),
            file_name=path.name,
            file_type=DocumentType.PDF,
            file_size_bytes=file_size,
            file_hash=file_hash,
            total_pages=total_pages,
            title=pdf_title,
        )

        return Document(
            doc_id=doc_id,
            content=concatenated_text,
            pages=parsed_pages,
            metadata=metadata,
        )