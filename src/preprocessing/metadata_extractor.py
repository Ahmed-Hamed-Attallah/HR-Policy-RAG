"""Extracts structural metadata, policy titles, and policy departments from text."""

import re
from typing import Dict, Optional
from src.domain.documents import Document


class MetadataExtractor:
    """Extracts contextual attributes from HR policy document headers and content."""

    # Common HR document policy header patterns
    POLICY_TITLE_PATTERNS = [
        re.compile(r"^(?:HR\s+Policy(?:\s*:\s*|\s*-\s*))([^\n]+)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^(?:Policy\s+Name(?:\s*:\s*|\s*-\s*))([^\n]+)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^#\s+([^\n]+)", re.MULTILINE),
    ]

    DEPARTMENT_PATTERNS = [
        re.compile(r"Department\s*:\s*([A-Za-z\s]+)", re.IGNORECASE),
        re.compile(r"Scope\s*:\s*([A-Za-z\s]+)", re.IGNORECASE),
    ]

    EFFECTIVE_DATE_PATTERNS = [
        re.compile(r"Effective\s+Date\s*:\s*([\w\s\d,/-]+)", re.IGNORECASE),
        re.compile(r"Last\s+Revised\s*:\s*([\w\s\d,/-]+)", re.IGNORECASE),
    ]

    @classmethod
    def extract_metadata(cls, text: str) -> Dict[str, Optional[str]]:
        """Extract title, department, and effective date from top portion of document."""
        sample_text = text[:3000]  # Examine first 3000 chars where headers reside

        extracted_title = None
        for pattern in cls.POLICY_TITLE_PATTERNS:
            match = pattern.search(sample_text)
            if match:
                extracted_title = match.group(1).strip()
                break

        department = None
        for pattern in cls.DEPARTMENT_PATTERNS:
            match = pattern.search(sample_text)
            if match:
                department = match.group(1).strip()
                break

        effective_date = None
        for pattern in cls.EFFECTIVE_DATE_PATTERNS:
            match = pattern.search(sample_text)
            if match:
                effective_date = match.group(1).strip()
                break

        return {
            "detected_title": extracted_title,
            "department": department,
            "effective_date": effective_date,
        }

    @classmethod
    def enrich_document_metadata(cls, document: Document) -> Document:
        """Enrich existing document metadata with rule-extracted contextual properties."""
        extra = cls.extract_metadata(document.content)

        if extra.get("detected_title") and not document.metadata.title:
            document.metadata.title = extra["detected_title"]

        document.metadata.extra_attributes.update({
            k: v for k, v in extra.items() if v is not None
        })

        return document