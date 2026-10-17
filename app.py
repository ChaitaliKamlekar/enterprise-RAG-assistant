import os
import io
import streamlit as st
from PIL import Image
import pypdfium2 as pdfium
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

st.set_page_config(page_title="Enterprise RAG Assistant", layout="centered")
st.title("Enterprise Document Assistant")

uploaded_file = st.file_uploader("Upload an enterprise PDF document:", type=["pdf"])

def extract_text_with_fallback(file_bytes):
    docs = []
    pdf = pdfium.PdfDocument(file_bytes)
    total_pages = len(pdf)

    # Fast check: text page-wise try karein
    for page_idx in range(total_pages):
        page = pdf[page_idx]
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        if text and text.strip():
            docs.append(Document(page_content=text, metadata={"page": page_idx}))

    # Agar digital text nahi mila to Vision OCR fallback
    if not docs:
        st.info("Scanned document detected. Extracting text using OCR Vision...")
        vision_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Max first 10 pages process karein demo ke liye
        process_limit = min(total_pages, 10)
        for page_idx in range(process_limit):
            page = pdf[page_idx]
            pil_image = page.render(scale=2).to_pil()
            
            response = vision_model.generate_content([
                "Extract and transcribe all readable text from this document image accurately. Do not add commentary.",
                pil_image
            ])
            if response.text and response.text.strip():
                docs.append(Document(page_content=response.text, metadata={"page": page_idx}))

    return docs

def setup_rag_pipeline(file_bytes):
    docs = extract_text_with_fallback(file_bytes)

    if not docs:
        return None, None, "No text could be extracted from this document."

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not valid_chunks:
        return None, None, "No extractable text segments found."

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(valid_chunks, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an enterprise AI assistant. Answer the user query using only the provided context.
    If the context does not contain the answer, explicitly state: "The requested information is not available in the provided document."

    Context:
    {context}

    Question: {question}
    """)

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, None

if uploaded_file is not None:
    with st.spinner("Processing document embeddings..."):
        rag_chain, retriever, error_msg = setup_rag_pipeline(uploaded_file.getvalue())

    if error_msg:
        st.error(error_msg)
    else:
        st.success("Document indexed successfully! Ask your questions below.")

        user_query = st.text_input("Ask a question about the document:")

        if user_query:
            with st.spinner("Retrieving information and generating response..."):
                response = rag_chain.invoke(user_query)
                st.markdown("### Answer:")
                st.write(response)

                source_docs = retriever.invoke(user_query)
                with st.expander("References / Citations"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 0) + 1}):**")
                        st.write(doc.page_content)
                        st.divider()
else:
    st.info("Please upload a PDF file to begin.")