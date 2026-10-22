"""Settings and Administrative Vector Store Controls."""

import streamlit as st
from app.frontend.api_client import api_client
from app.frontend.config import frontend_settings

st.set_page_config(
    page_title=f"Settings - {frontend_settings.APP_TITLE}",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ System Settings & Administration")

st.markdown("### 🔌 API Connectivity")
st.text_input("Backend API Base URL", value=frontend_settings.BACKEND_API_URL, disabled=True)

st.divider()

st.markdown("### 🗄️ Vector Store Administration")
st.info("Rebuilding the index will parse all processed documents in `data/processed/` and update FAISS.")

c1, c2 = st.columns(2)

with c1:
    if st.button("🔄 Rebuild FAISS Index", use_container_width=True):
        with st.spinner("Rebuilding vector store..."):
            try:
                res = api_client.rebuild_index()
                st.success(f"Index successfully rebuilt! {res.get('message', '')}")
            except Exception as e:
                st.error(f"Rebuild failed: {e}")

with c2:
    if st.button("⚠️ Clear FAISS Index", type="secondary", use_container_width=True):
        with st.spinner("Clearing vector store..."):
            try:
                res = api_client.clear_index()
                st.warning(f"Index cleared. {res.get('message', '')}")
            except Exception as e:
                st.error(f"Failed to clear index: {e}")