"""UI Component for displaying citation cards and retrieved diagnostic passages."""

from typing import Any, Dict, List
import streamlit as st


def render_citation_cards(citations: List[Dict[str, Any]]):
    """Render interactive cards for official citations."""
    if not citations:
        return

    st.markdown("##### 📌 Verified Policy Sources & Citations")
    cols = st.columns(min(len(citations), 3))

    for idx, cite in enumerate(citations):
        with cols[idx % 3]:
            source_name = cite.get("source_name", "Unknown Source")
            page_no = cite.get("page_number")
            snippet = cite.get("snippet", "")
            score = cite.get("score")

            page_badge = f" (Page {page_no})" if page_no else ""
            st.info(
                f"**[{idx + 1}] {source_name}{page_badge}**\n\n"
                f"*\"{snippet[:180]}...\"*"
                + (f"\n\n`Match Score: {score:.3f}`" if score else "")
            )


def render_passage_expander(passages: List[Dict[str, Any]]):
    """Collapsible diagnostic expander showing raw candidate passages."""
    if not passages:
        return

    with st.expander(f"🔍 Inspect Retrieved Context ({len(passages)} passages)"):
        for i, p in enumerate(passages):
            st.markdown(
                f"**Passage {i + 1}:** `{p.get('document_name', 'Doc')}` "
                f"(Page {p.get('page_number', 'N/A')}) | "
                f"Sim: `{p.get('similarity_score', 0):.4f}`"
                + (f" | Rerank: `{p.get('rerank_score', 0):.4f}`" if p.get("rerank_score") else "")
            )
            st.text(p.get("content", ""))
            st.divider()