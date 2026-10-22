"""Policy Library Page for document inspection and ingestion."""

import streamlit as st
from app.frontend.api_client import api_client
from app.frontend.config import frontend_settings

st.set_page_config(
    page_title=f"Policy Library - {frontend_settings.APP_TITLE}",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Policy Documents Library")
st.caption("Manage company manuals, employee handbooks, and uploaded guidelines.")

tab_list, tab_upload = st.tabs(["📑 Registered Documents", "📤 Upload New Policy"])

with tab_list:
    docs = api_client.list_documents()

    if not docs:
        st.info("No policy documents found in registry.")
    else:
        st.markdown(f"**Total Documents Registered:** `{len(docs)}`")
        for d in docs:
            with st.expander(f"📄 {d.get('file_name')} ({d.get('file_type', 'N/A').upper()})"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Document ID:** `{d.get('doc_id')}`")
                c2.write(f"**File Size:** `{d.get('file_size_bytes', 0) / 1024:.1f} KB`")
                c3.write(f"**Pages:** `{d.get('total_pages', 'N/A')}`")

                if st.button("Preview Document Details", key=f"btn_{d.get('doc_id')}"):
                    details = api_client.get_document_details(d.get("doc_id"))
                    st.text_area(
                        "Text Preview (First 500 chars)",
                        value=details.get("content_preview", ""),
                        height=150,
                    )

with tab_upload:
    st.subheader("Upload Policy Document")
    uploaded_file = st.file_uploader(
        "Select a PDF or DOCX file to add to the knowledge base",
        type=["pdf", "docx"],
    )

    if uploaded_file is not None:
        if st.button("🚀 Upload and Index Document", type="primary"):
            with st.spinner("Uploading, parsing, chunking, and embedding document..."):
                try:
                    res = api_client.upload_document(
                        file_name=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                    )
                    st.success(
                        f"✅ Successfully indexed `{uploaded_file.name}`! "
                        f"Created `{res.get('chunks_created', 0)}` vector chunks."
                    )
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")