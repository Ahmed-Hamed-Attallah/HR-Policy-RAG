"""UI Component for rendering formatted chat conversation turns."""

from typing import Optional
import streamlit as st


def render_user_message(content: str):
    """Render an employee inquiry bubble."""
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(content)


def render_assistant_message(
    content: str,
    model_name: Optional[str] = None,
    latency_ms: Optional[float] = None,
    is_fallback: bool = False,
):
    """Render the AI response bubble with diagnostic meta tags."""
    with st.chat_message("assistant", avatar="🏢"):
        st.markdown(content)

        meta_items = []
        if model_name:
            meta_items.append(f"🤖 **Model:** `{model_name}`")
        if latency_ms is not None:
            meta_items.append(f"⏱️ **Latency:** `{latency_ms:.0f} ms`")
        if is_fallback:
            meta_items.append("⚠️ *Fallback Active (No context found)*")

        if meta_items:
            st.caption(" | ".join(meta_items))