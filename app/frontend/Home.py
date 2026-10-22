"""Streamlit Landing Page & System Health Dashboard."""
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app.frontend.api_client import api_client
from app.frontend.config import frontend_settings

st.set_page_config(
    page_title=frontend_settings.APP_TITLE,
    page_icon=frontend_settings.APP_ICON,
    layout="wide",
)

st.title(f"{frontend_settings.APP_ICON} {frontend_settings.APP_TITLE}")
st.markdown("### Enterprise Retrieval-Augmented Generation Platform for Company Policies")

st.divider()

# System Status Cards
health_data = api_client.check_health()
is_healthy = health_data.get("status") == "healthy"

col1, col2, col3, col4 = st.columns(4)

with col1:
    if is_healthy:
        st.success("🟢 **Backend API:** Operational")
    else:
        st.error(f"🔴 **Backend API:** Offline ({health_data.get('error', 'Unreachable')})")

with col2:
    vectors = health_data.get("vectors_indexed", 0)
    st.metric("Indexed Chunks", f"{vectors:,}")

with col3:
    model = health_data.get("embedding_model", "BAAI/bge-small-en-v1.5")
    st.metric("Embedding Model", model.split("/")[-1])

with col4:
    persisted = health_data.get("index_persisted", False)
    st.metric("Index Persisted", "Yes ✅" if persisted else "No ❌")

st.divider()

st.markdown("""
#### 🚀 Quick Navigation
* **💬 Chat:** Ask natural language questions about workplace policies, working hours, benefits, and vacation rules.
* **📚 Policy Library:** View ingested HR documents, upload new policy files (`.pdf`, `.docx`), and inspect text chunks.
* **📊 Evaluation:** Run benchmark queries to verify hit rates, MRR, and grounding faithfulness.
* **⚙️ Settings:** Adjust retrieval top-k parameters, threshold sensitivity, and manage index rebuilds.
""")

if not is_healthy:
    st.warning(
        "⚠️ Backend service is not detected at `http://localhost:8000`. "
        "Please make sure `uvicorn app.backend.main:app --reload` is running in your terminal."
    )