# Enterprise Document Assistant (RAG Pipeline)

A robust Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, Google Gemini, and Streamlit. The application allows users to upload enterprise documents (digital or scanned PDFs) and ask context-grounded questions with real-time source citations.

## Features
- **Semantic Search:** Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and ChromaDB vector store.
- **Multimodal OCR Fallback:** Extracts text from scanned/image PDFs using Gemini Vision before indexing.
- **Source Grounding:** Direct citations and page references for auditability and zero hallucination.
- **Interactive UI:** Streamlit interface for document uploading and conversational querying.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd <repo-folder>