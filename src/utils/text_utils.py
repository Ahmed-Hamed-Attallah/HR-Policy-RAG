"""Text sanitization, normalization, and token estimation utilities."""

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Normalize unicode representations and clean consecutive spaces/newlines."""
    if not text:
        return ""
    # 1. NFKC decomposes ligatures (e.g. ﬁ -> fi, ﬂ -> fl)
    normalized = unicodedata.normalize("NFKC", str(text))
    
    # 2. Strip horizontal spaces surrounding newlines
    normalized = re.sub(r"[ \t]*\n[ \t]*", "\n", normalized)
    
    # 3. Collapse multiple consecutive newlines (more than 2) into 2
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    
    # 4. Collapse consecutive horizontal whitespace into a single space
    normalized = re.sub(r"[ \t]+", " ", normalized)
    
    return normalized.strip()


def clean_markdown_artifacts(text: str) -> str:
    """Remove control characters and broken markdown tags."""
    if not text:
        return ""
    # Strip null bytes and non-printable control characters (retain tabs/newlines)
    cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    return cleaned


def estimate_token_count(text: str) -> int:
    """Rough estimation of token count using character length heuristic (approx 4 chars/token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)