import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Financial Report Analyzer", layout="wide")

st.title("📊 Financial Report Analyzer (FastAPI + RAG)")
st.write("Upload a financial report PDF and ask questions.")

# Upload PDF
uploaded_file = st.file_uploader("Upload Financial Report (PDF)", type="pdf")

if uploaded_file is not None:

    files = {
        "file": (uploaded_file.name, uploaded_file, "application/pdf")
    }

    with st.spinner("Uploading and processing PDF..."):
        response = requests.post(f"{API_BASE}/upload", files=files)

    if response.status_code == 200:
        st.success("PDF processed successfully!")
    else:
        st.error("Error processing PDF")
        st.stop()

    # Ask Question
    query = st.text_input("Ask a question about the report:")

    if query:
        with st.spinner("Analyzing..."):
            res = requests.post(
                f"{API_BASE}/ask",
                json={"question": query}
            )

        if res.status_code == 200:
            answer = res.json()["answer"]
            st.subheader("📌 Answer")
            st.write(answer)
        else:
            st.error("Error fetching answer")