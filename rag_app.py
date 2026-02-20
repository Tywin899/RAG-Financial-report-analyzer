import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load API key
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# 1️⃣ Load data
loader = PyPDFLoader("financial_report.pdf")
documents = loader.load()

# 2️⃣ Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# 3️⃣ Create FREE embeddings (local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4️⃣ Store in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# 5️⃣ Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6️⃣ Create Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# 7️⃣ Ask question
query = input("Ask something: ")

# Retrieve relevant docs
relevant_docs = retriever.invoke(query)

# Combine context
context = "\n".join([doc.page_content for doc in relevant_docs])

prompt = f"""
You are a financial analyst assistant.

Answer the question using ONLY the provided financial report context.
If the answer is not present, say "The report does not specify."

Financial Report Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)