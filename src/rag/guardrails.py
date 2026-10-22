"""Guardrail filters to protect against prompt injection, toxic inputs, and out-of-scope abuse."""

import re
from typing import List, Optional
from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    """Validation verdict for text passing through safety barriers."""

    is_safe: bool = Field(description="True if content passes all guard checks")
    violation_reason: Optional[str] = Field(default=None, description="Explanation if rejected")
    sanitized_text: str = Field(default="", description="Sanitized version of input/output")


class PolicyGuardrails:
    """Input and output validation engine for Enterprise HR RAG."""

    # Patterns indicating prompt injection or jailbreak attempts
    INJECTION_PATTERNS: List[re.Pattern] = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"disregard\s+(?:system\s+)?(?:rules|prompts)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(?:a\s+)?unrestricted", re.IGNORECASE),
        re.compile(r"dan\s+mode|jailbreak", re.IGNORECASE),
        re.compile(r"reveal\s+your\s+(?:system\s+)?prompt", re.IGNORECASE),
        re.compile(r"bypass\s+policy", re.IGNORECASE),
    ]

    # Broad domain keywords indicative of workplace / HR inquiries
    HR_KEYWORDS = {
        "leave", "vacation", "sick", "holiday", "salary", "compensation", "bonus",
        "probation", "notice", "termination", "resignation", "promotion", "overtime",
        "attendance", "working hours", "remote", "hybrid", "insurance", "pension",
        "code of conduct", "harassment", "maternity", "paternity", "allowance", "expense",
        "travel", "performance", "appraisal", "benefit", "policy", "employee", "contract"
    }

    def __init__(self, min_query_len: int = 3, max_query_len: int = 1000):
        self.min_query_len = min_query_len
        self.max_query_len = max_query_len

    def validate_input(self, query: str) -> GuardrailResult:
        """Inspect user query for security violations, size anomalies, and prompt injections."""
        cleaned = query.strip()

        # 1. Length checks
        if len(cleaned) < self.min_query_len:
            return GuardrailResult(
                is_safe=False,
                violation_reason="Query is too short to process.",
                sanitized_text=cleaned,
            )

        if len(cleaned) > self.max_query_len:
            return GuardrailResult(
                is_safe=False,
                violation_reason=f"Query exceeds maximum character allowance ({self.max_query_len}).",
                sanitized_text=cleaned[:self.max_query_len],
            )

        # 2. Injection inspection
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(cleaned):
                return GuardrailResult(
                    is_safe=False,
                    violation_reason="Potentially malicious prompt injection pattern detected.",
                    sanitized_text="",
                )

        return GuardrailResult(is_safe=True, sanitized_text=cleaned)

    def validate_output(self, answer: str) -> GuardrailResult:
        """Inspect generated answer to prevent leakage or corrupted output."""
        cleaned = answer.strip()
        if not cleaned:
            return GuardrailResult(
                is_safe=False,
                violation_reason="Generated response is unexpectedly empty.",
                sanitized_text="",
            )

        # Ensure no system instructions leaked into output
        if "STRICT INSTRUCTION GUIDELINES" in cleaned or "CONTEXT PASSAGES:" in cleaned:
            return GuardrailResult(
                is_safe=False,
                violation_reason="System prompt leakage identified in model output.",
                sanitized_text="Based on available HR policy documentation, this request could not be fulfilled.",
            )

        return GuardrailResult(is_safe=True, sanitized_text=cleaned)

    def is_likely_hr_query(self, query: str) -> bool:
        """Heuristic check to determine if query bears relevance to workplace policies."""
        lower_q = query.lower()
        return any(kw in lower_q for kw in self.HR_KEYWORDS)