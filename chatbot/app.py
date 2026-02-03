import os
import sys
import tempfile
import numpy as np
import streamlit as st

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

# ---------------- SESSION STATE ----------------
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_file" not in st.session_state:
    st.session_state.last_file = None

# ---------------- IMPORTS ----------------
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# ---------------- ENV ----------------
load_dotenv()

# ---------------- MODELS ----------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("LLM_MODEL")
)

@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedder = load_embedder()

# ---------------- PDF EXTRACTION ----------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)

    return "\n".join(full_text)

# ---------------- CHUNKING ----------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_text(text)

# ---------------- RETRIEVAL ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_chunks(question, chunks, embeddings, top_k=4):
    q_emb = embedder.embed_query(question)

    scored = []
    for emb, chunk in zip(embeddings, chunks):
        score = cosine_similarity(q_emb, emb)
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [item[1] for item in scored[:top_k]]

# ---------------- ANSWERING ----------------
def answer_question(context_chunks, question):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a document-based assistant.

Answer ONLY using the context below.
If the answer is not present, say:
"Answer not found in the document."

Context:
{context}

Question:
{question}
"""
    return llm.invoke(prompt).content

# ---------------- UI ----------------
st.title("AI From Mars–")

with st.sidebar:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Upload a text-based PDF", type=["pdf"])

# ---------------- PROCESS PDF ----------------
if uploaded_file:
    # Reset chat history ONLY if new file uploaded
    if st.session_state.last_file != uploaded_file.name:
        st.session_state.chat_history = []
        st.session_state.last_file = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text.strip():
        st.error(" No text extracted. This PDF may be scanned.")
    else:
        chunks = chunk_text(raw_text)
        embeddings = embedder.embed_documents(chunks)

        st.session_state.doc_chunks = chunks
        st.session_state.doc_embeddings = embeddings

        st.success(f"✅ Extracted {len(chunks)} chunks from PDF")

# ---------------- RENDER PREVIOUS CHAT ----------------
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])

# ---------------- CHAT INPUT ----------------
question = st.chat_input("Ask a question from the document")

if question and st.session_state.doc_chunks:
    with st.chat_message("user"):
        st.write(question)

    top_chunks = retrieve_chunks(
        question,
        st.session_state.doc_chunks,
        st.session_state.doc_embeddings
    )

    answer = answer_question(top_chunks, question)

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat_history.append(
        {"question": question, "answer": answer}
    )

# ---------------- CHAT HISTORY (EXPANDER) ----------------
with st.expander("🕘 Chat History"):
    for chat in st.session_state.chat_history:
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")
        st.markdown("---")

