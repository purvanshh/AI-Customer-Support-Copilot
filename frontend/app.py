"""
SupportCopilot AI — Streamlit Demo UI

Launch:
    streamlit run frontend/app.py

Expects the FastAPI backend running at API_URL (default: http://localhost:8000).
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SupportCopilot AI",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for a clean, modern look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Global overrides */
    .block-container { max-width: 960px; padding-top: 2rem; }
    .stAlert { border-radius: 10px; }

    /* Source badges */
    .source-badge {
        display: inline-block;
        background: #e8eaf6;
        color: #283593;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.82em;
        margin: 2px 4px 2px 0;
    }

    /* Analytics metric cards */
    div[data-testid="stMetric"] {
        background: #f5f5f5;
        border-radius: 12px;
        padding: 12px 16px;
    }

    /* Divider */
    hr { margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🤖 SupportCopilot AI")
st.caption("AI-powered customer support copilot — RAG + Feedback Learning")

# ---------------------------------------------------------------------------
# Sidebar — Data Upload + Analytics
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📊 Analytics")

    try:
        analytics = requests.get(f"{API_URL}/analytics", timeout=5).json()
        col1, col2 = st.columns(2)
        col1.metric("Total Queries", analytics.get("tickets_processed", 0))
        col2.metric("Feedback Count", analytics.get("feedback_count", 0))

        col3, col4 = st.columns(2)
        col3.metric("Auto-Resolved", f"{analytics.get('percent_auto_resolved', 0):.1f}%")
        col4.metric("Accuracy", f"{analytics.get('accuracy', 0):.0%}")

        avg_ms = analytics.get("avg_response_time_ms", 0)
        st.metric("Avg Latency", f"{avg_ms:.0f} ms")
    except Exception:
        st.info("Backend not reachable. Start the API server first.")

    st.divider()

    # ---------- Upload Tickets ----------
    st.header("📁 Upload Data")
    ticket_file = st.file_uploader("Upload Tickets (CSV / JSON)", type=["csv", "json", "jsonl"])
    if ticket_file is not None:
        if st.button("Ingest Tickets", use_container_width=True):
            with st.spinner("Uploading & indexing…"):
                resp = requests.post(
                    f"{API_URL}/upload-tickets",
                    files={"file": (ticket_file.name, ticket_file.getvalue(), ticket_file.type or "text/csv")},
                    timeout=120,
                )
            if resp.ok:
                data = resp.json()
                st.success(f"✅ {data['tickets_ingested']} tickets ingested, index rebuilt.")
            else:
                st.error(f"Upload failed: {resp.text}")

    # ---------- Upload Docs ----------
    doc_files = st.file_uploader(
        "Upload KB Docs (TXT / MD / PDF)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if doc_files:
        if st.button("Ingest Docs", use_container_width=True):
            files = [("docs", (d.name, d.getvalue(), d.type or "text/plain")) for d in doc_files]
            with st.spinner("Processing docs…"):
                resp = requests.post(f"{API_URL}/upload-docs", files=files, timeout=120)
            if resp.ok:
                data = resp.json()
                st.success(f"✅ {data['docs_processed']} docs → {data['chunks_added']} chunks indexed.")
            else:
                st.error(f"Upload failed: {resp.text}")

# ---------------------------------------------------------------------------
# Main area — Query & Response
# ---------------------------------------------------------------------------

st.header("💬 Ask a Question")

customer_query = st.text_area(
    "Customer query",
    placeholder="e.g. I was charged twice for my subscription…",
    height=100,
    label_visibility="collapsed",
)

col_send, col_model = st.columns([3, 1])
with col_model:
    llm_choice = st.selectbox("LLM", ["auto", "mock", "openai", "gemini"], index=0)

with col_send:
    send = st.button("🔍 Get AI Response", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Response display + feedback
# ---------------------------------------------------------------------------

if send and customer_query.strip():
    payload = {"customer_query": customer_query.strip()}
    if llm_choice != "auto":
        payload["llm_override"] = llm_choice

    with st.spinner("Thinking…"):
        t0 = time.time()
        try:
            resp = requests.post(f"{API_URL}/query", json=payload, timeout=60)
        except requests.ConnectionError:
            st.error("Cannot reach the backend. Is the API server running?")
            st.stop()
        latency = time.time() - t0

    if not resp.ok:
        st.error(f"API error: {resp.text}")
        st.stop()

    result = resp.json()
    gen_id = result["generation_id"]

    # Store in session for feedback
    st.session_state["last_result"] = result

    # ---------- Response card ----------
    st.subheader("🧠 AI Suggestion")

    st.markdown(result["suggestion_text"])

    info_cols = st.columns(3)
    info_cols[0].caption(f"**Confidence:** {result['confidence']:.0%}")
    info_cols[1].caption(f"**Model:** {result['model_used']}")
    info_cols[2].caption(f"**Latency:** {latency:.2f}s")

    # ---------- Sources ----------
    sources = result.get("sources", [])
    if sources:
        st.markdown("**Sources:**")
        badges = ""
        for s in sources:
            badges += f'<span class="source-badge">{s["source_type"]}: {s["source_ref"]} ({s["score"]:.2f})</span>'
        st.markdown(badges, unsafe_allow_html=True)

    st.divider()

    # ---------- Feedback ----------
    st.subheader("📝 Feedback")
    fb_cols = st.columns(3)

    with fb_cols[0]:
        if st.button("👍 Accept", use_container_width=True):
            requests.post(
                f"{API_URL}/feedback",
                json={"generation_id": gen_id, "user_action": "accepted"},
                timeout=10,
            )
            st.success("Feedback submitted: Accepted ✅")
            st.rerun()

    with fb_cols[1]:
        if st.button("👎 Reject", use_container_width=True):
            requests.post(
                f"{API_URL}/feedback",
                json={"generation_id": gen_id, "user_action": "rejected"},
                timeout=10,
            )
            st.warning("Feedback submitted: Rejected")
            st.rerun()

    with fb_cols[2]:
        show_edit = st.button("✏️ Edit", use_container_width=True)

    if show_edit:
        corrected = st.text_area("Edit the response:", value=result["suggestion_text"], height=150)
        if st.button("Submit Correction", type="primary"):
            requests.post(
                f"{API_URL}/feedback",
                json={
                    "generation_id": gen_id,
                    "user_action": "corrected",
                    "corrected_text": corrected,
                },
                timeout=10,
            )
            st.success("Correction submitted ✅")
            st.rerun()

elif send:
    st.warning("Please enter a customer query.")
