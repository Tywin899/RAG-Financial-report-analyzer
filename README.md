# 📊 Financial Report RAG System

A Retrieval-Augmented Generation (RAG) system that analyzes financial PDF reports using:

- FastAPI (Backend)
- Streamlit (Frontend)
- FAISS (Vector Database)
- HuggingFace Embeddings
- Groq LLM (LLaMA 3.1)

---

## 🚀 Features

- Upload financial PDF reports
- Semantic chunking with RecursiveCharacterTextSplitter
- Vector storage using FAISS
- Context-aware financial Q&A
- REST API backend with FastAPI
- Interactive frontend with Streamlit

---

## 🧠 Architecture

Streamlit UI → FastAPI Backend → FAISS Retriever → Groq LLM

---

## 🛠️ Setup

### 1. Clone repo

git clone https://github.com/Tywin899/RAG-Financial-report-analyzer.git  
cd RAG-Financial-report-analyzer  

### 2. Create virtual environment

python -m venv venv  
venv\Scripts\activate  

### 3. Install dependencies

pip install -r requirements.txt  

### 4. Add .env file

Create `.env` file:

GROQ_API_KEY=your_api_key_here  

### 5. Run backend

uvicorn main:app --reload  

### 6. Run frontend

streamlit run streamlit_app.py  

---

## 📌 Note

This project was built as a learning exercise exploring Retrieval-Augmented Generation (RAG) systems.