"""High-level generator orchestrating prompt formatting and LLM calls."""

import logging
from typing import Optional

from src.generation.llm_client import BaseLLMClient, LLMClientFactory
from src.generation.prompt_templates import (
    FALLBACK_EMPTY_CONTEXT_MESSAGE,
    format_rag_prompt,
)

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Synthesizes structured answers from queries and assembled context."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.client = llm_client or LLMClientFactory.create_client()

    def generate_answer(
        self,
        query: str,
        context: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Formulate prompt, call LLM client, and return validated text response."""
        stripped_context = context.strip()
        if not stripped_context or stripped_context == "No relevant HR policy passages found.":
            logger.info("Context is empty; returning standard no-context fallback.")
            return FALLBACK_EMPTY_CONTEXT_MESSAGE

        messages = format_rag_prompt(query=query, context=stripped_context)

        try:
            raw_answer = self.client.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            cleaned_answer = raw_answer.strip()
            return cleaned_answer if cleaned_answer else FALLBACK_EMPTY_CONTEXT_MESSAGE
        except Exception as e:
            logger.error(f"Failed to generate answer from LLM client: {e}")
            return (
                "An unexpected error occurred while communicating with the generation service. "
                "Please retry or contact system administrators."
            )