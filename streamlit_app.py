import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 30  # seconds

st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="📊",
    layout="wide"
)

# ---- Custom Styling ----
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.stChatMessage {
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Financial Report AI Assistant")
st.caption("Upload a financial report and ask intelligent, context-aware questions.")

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


def check_backend_status():
    """Checks backend connectivity and status."""
    try:
        res = requests.get(f"{API_BASE}/status", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


# ---- Sidebar ----
with st.sidebar:
    st.header("📂 Upload Report")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", help="Upload a financial earnings report or 10-K/10-Q filing.")

    if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_filename:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
        }

        with st.spinner("Validating and indexing PDF..."):
            try:
                response = requests.post(
                    f"{API_BASE}/upload",
                    files=files,
                    timeout=REQUEST_TIMEOUT
                )
                if response.status_code == 200:
                    data = response.json()
                    chunks = data.get("chunks_created", 0)
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.success(f"Report '{uploaded_file.name}' processed successfully ({chunks} chunks indexed).")
                else:
                    error_msg = response.json().get("error", "Upload failed.")
                    st.error(f"Upload error: {error_msg}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend server. Please make sure FastAPI is running on http://127.0.0.1:8000.")
            except requests.exceptions.Timeout:
                st.error("Upload timed out. The file may be too large to process.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

    status_data = check_backend_status()
    if status_data:
        st.markdown("---")
        st.markdown("### 📌 Active Document Status")
        if status_data.get("has_active_document"):
            st.info(f"**Active Report:** {status_data.get('active_document_name')}\n\n**Indexed Chunks:** {status_data.get('total_chunks')}")
            if st.button("🔄 Clear Active Document"):
                try:
                    requests.post(f"{API_BASE}/reset", timeout=5)
                    st.session_state.uploaded_filename = None
                    st.session_state.messages = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to reset: {e}")
        else:
            st.warning("No report currently active. Please upload a PDF.")

    st.markdown("---")
    st.markdown("### 🔍 About")
    st.write("RAG-powered financial analysis using FAISS + Groq LLM.")

# ---- Chat Interface ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- Chat Input ----
if prompt := st.chat_input("Ask a financial question..."):
    if not prompt.strip():
        st.warning("Please enter a non-empty question.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Analyzing report..."):
                try:
                    res = requests.post(
                        f"{API_BASE}/ask",
                        json={"question": prompt.strip()},
                        timeout=REQUEST_TIMEOUT
                    )

                    if res.status_code == 200:
                        data = res.json()
                        answer = data.get("answer", "No answer received.")
                    else:
                        error_detail = res.json().get("error", "Error retrieving answer.")
                        answer = f"⚠️ {error_detail}"
                except requests.exceptions.ConnectionError:
                    answer = "⚠️ Backend connection failed. Ensure FastAPI is running on http://127.0.0.1:8000."
                except requests.exceptions.Timeout:
                    answer = "⚠️ Request timed out while querying the LLM."
                except Exception as e:
                    answer = f"⚠️ An unexpected error occurred: {str(e)}"

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.caption("Built using FastAPI, FAISS, HuggingFace Embeddings, and Groq LLM.")