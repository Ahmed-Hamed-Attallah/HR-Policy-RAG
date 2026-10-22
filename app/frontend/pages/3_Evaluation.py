"""RAG Evaluation and Metrics Inspection Page."""

from pathlib import Path
import json
import pandas as pd
import streamlit as st
from app.frontend.config import frontend_settings

st.set_page_config(
    page_title=f"Evaluation - {frontend_settings.APP_TITLE}",
    page_icon="📊",
    layout="wide",
)

st.title("📊 RAG Evaluation & Benchmark Metrics")
st.caption("Inspect retrieval recall, answer faithfulness, and latency statistics.")

report_path = Path("artifacts/evaluation_report.json")

if not report_path.exists():
    st.warning(
        "No evaluation report found at `artifacts/evaluation_report.json`. "
        "Run `python scripts/evaluate_rag.py` to generate one."
    )
else:
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        summary = report_data.get("summary", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Queries Evaluated", summary.get("total_queries", 0))
        col2.metric("Mean Latency", f"{summary.get('mean_latency_ms', 0):.1f} ms")
        col3.metric("Retrieval Hit Rate", f"{summary.get('retrieval_hit_rate', 0) * 100:.1f}%")
        col4.metric("Faithfulness Score", f"{summary.get('faithfulness_score', 0) * 100:.1f}%")

        st.divider()

        st.subheader("Itemized Query Results")
        details = report_data.get("details", [])
        if details:
            df = pd.DataFrame(details)
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load evaluation report: {e}")