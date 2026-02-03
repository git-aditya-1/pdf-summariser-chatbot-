# 📄 AI From Mars – Document-Based RAG Chatbot

AI From Mars is a document-based conversational chatbot built using a **Retrieval-Augmented Generation (RAG)** architecture. The application allows users to upload a PDF document and interact with its content through a chat interface. Unlike generic chatbots, this system answers questions strictly based on the uploaded document, ensuring accurate, reliable, and non-hallucinated responses.

---

## 🚀 Project Overview

Large Language Models (LLMs) are powerful, but they do not inherently understand private or user-provided documents. AI From Mars addresses this limitation by combining document retrieval with language generation. The chatbot first retrieves relevant information from the document and then uses an LLM to generate context-aware answers.

This project is designed to demonstrate a **clean, modular, and industry-style RAG pipeline** suitable for small to medium-scale document question-answering use cases.

---

## 🧠 Architecture

The system follows a layered architecture:

1. **User Interface (Streamlit)**  
   Provides a chat-style interface for file upload and question answering.

2. **Document Ingestion**  
   Uses **PyPDF** to extract text from text-based PDF files.

3. **Text Processing**  
   Splits extracted text into overlapping chunks using `RecursiveCharacterTextSplitter` to handle large documents efficiently.

4. **Embedding Layer**  
   Converts document chunks into semantic vector embeddings using **HuggingFace sentence-transformer models**.

5. **Retrieval Layer**  
   Uses cosine similarity to compare the user’s query embedding with document embeddings and retrieves the most relevant chunks.

6. **Generation Layer (LLM)**  
   Uses a **Groq-powered Large Language Model** to generate natural language answers grounded strictly in the retrieved document content.

---

## 🔄 Data Flow

