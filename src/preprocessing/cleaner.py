"""Text sanitization and boilerplate removal for HR policy documents."""

import re
from src.domain.documents import Document, ParsedPage
from src.utils.text_utils import clean_markdown_artifacts, normalize_whitespace


class DocumentCleaner:
    """Cleans document content, removes repetitive headers/footers and standardizes layout."""

    # Common HR policy boilerplate patterns (e.g. "Page 1 of 12", "Confidential", etc.)
    BOILERPLATE_PATTERNS = [
        re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
        re.compile(r"---+\s*Confidential\s*---+", re.IGNORECASE),
        re.compile(r"^\s*Internal\s+Use\s+Only\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"\[\s*Confidential\s*\]", re.IGNORECASE),
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Apply all text cleaning passes to a raw string."""
        if not text:
            return ""

        cleaned = clean_markdown_artifacts(text)

        # Remove boilerplate regex patterns
        for pattern in cls.BOILERPLATE_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        cleaned = normalize_whitespace(cleaned)
        return cleaned

    @classmethod
    def clean_document(cls, document: Document) -> Document:
        """Produce a cleaned clone of the Document with sanitized pages and text."""
        cleaned_pages: list[ParsedPage] = []
        cleaned_text_blocks: list[str] = []

        for p in document.pages:
            sanitized = cls.clean_text(p.text)
            if sanitized:
                cleaned_pages.append(
                    ParsedPage(
                        page_number=p.page_number,
                        text=sanitized,
                        metadata=p.metadata,
                    )
                )
                cleaned_text_blocks.append(sanitized)

        new_content = "\n\n".join(cleaned_text_blocks)

        return Document(
            doc_id=document.doc_id,
            content=new_content,
            pages=cleaned_pages,
            metadata=document.metadata,
        )