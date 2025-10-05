import os
import json
import streamlit as st
from dotenv import load_dotenv
import openai
import PyPDF2
import logging
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# --- PDF RAG Pipeline ---

def load_pdf_texts(pdf_path):
    """
    Extracts text from the given PDF file and returns a list of LangChain Document objects.
    """
    docs = []
    if os.path.exists(pdf_path):
        logger.info(f"Loading PDF: {pdf_path}")
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(pdf_path)}))
        logger.info(f"Extracted {len(docs)} document(s) from PDF.")
    else:
        logger.warning(f"PDF file not found: {pdf_path}")
    return docs

@st.cache_resource
def build_vectorstore():
    """
    Builds a FAISS vectorstore from the PDF document using OpenAI embeddings.
    """
    pdf_path = os.path.join(os.path.dirname(__file__), "../sample_data/air-india-coc.pdf")
    docs = load_pdf_texts(pdf_path)
    if not docs:
        logger.error("No documents loaded from PDF. Vectorstore will be empty.")
        return None
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    logger.info(f"Split PDF into {len(split_docs)} chunks for embedding.")

    # convert text to embeddings(vectors), this vectors are numerical representation of text which is stored in faiss vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    logger.info("Vectorstore built successfully.")
    return vectorstore

def retrieve_context(question, vectorstore, k=2):
    """
    Retrieves the top-k most relevant chunks from the vectorstore for the given question.
    """
    if not vectorstore:
        logger.error("Vectorstore is not available. Returning empty context.")
        return ""
    results = vectorstore.similarity_search(question, k=k)
    logger.info(f"Retrieved {len(results)} relevant context chunks for the question.")
    return "\n\n".join([doc.page_content for doc in results])

def ask_rag_agent(question: str, info: dict, context: str):
    """
    Calls the OpenAI LLM with the provided question, airline info, and retrieved context.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
        You are an expert on airline travel rules. Use the following information and context to answer the user's question.\n\n        Airline Info: {json.dumps(info)}\n\n        Context from documents:\n        {context}\n\n        Question: {question}\n        Answer:
    """
    logger.info("Sending prompt to OpenAI LLM.")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert on airline travel rules."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content.strip()
    logger.info("Received answer from OpenAI LLM.")
    return answer

# --- Streamlit UI ---
# ...existing code...

def main():
    """Main function to run the Streamlit RAG app."""
    st.title("Air Travel Rules Agent (RAG)")

    user_question = st.text_input("Ask a question about airline rules:")

    if user_question:
        logger.info(f"User question: {user_question}")
        # No airline info logic, only RAG
        with st.spinner("Thinking..."):
            vectorstore = build_vectorstore()
            context = retrieve_context(user_question, vectorstore)
            answer = ask_rag_agent(user_question, {}, context)
        st.success(answer)
        logger.info("Displayed answer to user.")

if __name__ == "__main__":
    main()
