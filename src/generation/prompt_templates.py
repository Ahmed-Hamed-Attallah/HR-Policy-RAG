"""System and user prompt templates for HR policy generation."""

from typing import Dict, List

SYSTEM_PROMPT = """You are an authoritative, accurate, and helpful Enterprise HR Policy Assistant.
Your mission is to answer employee queries using ONLY the provided HR policy context.

STRICT INSTRUCTION GUIDELINES:
1. Rely exclusively on facts mentioned in the provided context. If the policy context does not contain sufficient information to answer the inquiry, state clearly: "Based on the available HR policy documentation, this information is not specified."
2. Cite the context sources directly in your response using the format [Source N] (e.g., [Source 1], [Source 2]) whenever stating specific rules, thresholds, allowances, or obligations.
3. Keep answers concise, professional, structured with bullet points where appropriate, and objective.
4. Do not offer legal opinions or speculate on policies not explicitly described in the context.
"""

USER_PROMPT_TEMPLATE = """CONTEXT PASSAGES:
---------------------
{context}
---------------------

EMPLOYEE INQUIRY:
{query}

ANSWER (ensure you cite relevant sources as [Source N]):"""

FALLBACK_EMPTY_CONTEXT_MESSAGE = (
    "I could not locate any relevant HR policies or company guidelines matching your query. "
    "Please reach out to the HR department directly or rephrase your question."
)


def format_rag_prompt(query: str, context: str) -> List[Dict[str, str]]:
    """Format query and retrieved context passages into standard ChatML message list."""
    formatted_user_message = USER_PROMPT_TEMPLATE.format(
        context=context.strip(),
        query=query.strip(),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": formatted_user_message},
    ]


def format_refusal_response(reason: str) -> str:
    """Format standardized refusal messages when inputs trigger guardrail violations."""
    return f"Request declined by policy guardrails: {reason}"