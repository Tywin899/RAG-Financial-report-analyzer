import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(title="Financial Report RAG API")

groq_api_key = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

vectorstore = None


class QuestionRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)

    return {"message": "PDF processed successfully"}


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global vectorstore

    if vectorstore is None:
        return {"error": "Please upload a PDF first."}

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(request.question)

    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    You are a financial analyst assistant.

    Answer using only the context below.

    Context:
    {context}

    Question:
    {request.question}
    """

    response = llm.invoke(prompt)

    return {"answer": response.content}