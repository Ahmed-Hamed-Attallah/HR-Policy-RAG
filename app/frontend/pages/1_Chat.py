"""Interactive RAG Chat Interface with citations and provenance inspection."""

import streamlit as st
from app.frontend.api_client import api_client
from app.frontend.components.chat_message import render_assistant_message, render_user_message
from app.frontend.components.source_card import render_citation_cards, render_passage_expander
from app.frontend.config import frontend_settings

st.set_page_config(
    page_title=f"Chat - {frontend_settings.APP_TITLE}",
    page_icon="💬",
    layout="wide",
)

st.title("💬 HR Policy Assistant")
st.caption("Ask questions about company regulations, leave, compensation, and code of conduct.")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Query Configuration")
    top_k = st.slider("Passages Retrieved (Top-K)", min_value=1, max_value=15, value=5)
    rerank_k = st.slider("Final Passages Kept (Rerank-K)", min_value=1, max_value=10, value=3)

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Conversation History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user_message(msg["content"])
    else:
        render_assistant_message(
            content=msg["content"],
            model_name=msg.get("model_name"),
            latency_ms=msg.get("latency_ms"),
            is_fallback=msg.get("is_fallback", False),
        )
        if msg.get("citations"):
            render_citation_cards(msg["citations"])
        if msg.get("retrieved_passages"):
            render_passage_expander(msg["retrieved_passages"])

# Handle User Input
if prompt := st.chat_input("Ask a question about HR policies..."):
    render_user_message(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching policies and generating answer..."):
        try:
            result = api_client.submit_chat_query(
                query=prompt,
                top_k=top_k,
                rerank_k=rerank_k,
            )

            answer = result.get("answer", "")
            model_name = result.get("model_name")
            latency_ms = result.get("latency_ms")
            is_fallback = result.get("is_fallback", False)
            citations = result.get("citations", [])
            retrieved_passages = result.get("retrieved_passages", [])

            render_assistant_message(
                content=answer,
                model_name=model_name,
                latency_ms=latency_ms,
                is_fallback=is_fallback,
            )
            render_citation_cards(citations)
            render_passage_expander(retrieved_passages)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "model_name": model_name,
                "latency_ms": latency_ms,
                "is_fallback": is_fallback,
                "citations": citations,
                "retrieved_passages": retrieved_passages,
            })

        except Exception as e:
            st.error(f"❌ Failed to get response: {e}")