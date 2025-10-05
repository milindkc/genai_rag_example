import os
import json
import streamlit as st
from dotenv import load_dotenv
import openai
import PyPDF2
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Load environment variables from .env file
load_dotenv()

# Find the correct path for airline_rules.json
RULES_PATH = os.path.join(os.path.dirname(__file__), "airline_rules.json")

# Load airline rules knowledge base
with open(RULES_PATH) as f:
    airline_rules = json.load(f)


def find_airline_info(question: str):
    for rule in airline_rules:
        if rule["airline"].lower() in question.lower():
            return rule
    return None

def ask_gpt_agent(question: str, info: dict):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"))
    prompt = f"""
        You are an expert on airline travel rules. Use the following information to answer the user's question:

        Airline Info: {json.dumps(info)}

        Question: {question}
        Answer:
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert on airline travel rules."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def load_pdf_texts(pdf_path):
    docs = []
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(pdf_path)}))
    return docs

@st.cache_resource
def build_vectorstore():
    pdf_path = os.path.join(os.path.dirname(__file__), "../sample_data/air-india-coc.pdf")
    docs = load_pdf_texts(pdf_path)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

def retrieve_context(question, vectorstore, k=2):
    results = vectorstore.similarity_search(question, k=k)
    return "\n\n".join([doc.page_content for doc in results])

def ask_rag_agent(question: str, info: dict, context: str):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"))
    prompt = f"""
        You are an expert on airline travel rules. Use the following information and context to answer the user's question.\n\n        Airline Info: {json.dumps(info)}\n\n        Context from documents:\n        {context}\n\n        Question: {question}\n        Answer:
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert on airline travel rules."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

st.title("Air Travel Rules Agent (RAG)")

user_question = st.text_input("Ask a question about airline rules:")

if user_question:
    info = find_airline_info(user_question)
    with st.spinner("Thinking..."):
        vectorstore = build_vectorstore()
        context = retrieve_context(user_question, vectorstore)
        # Use empty dict if info is None
        answer = ask_rag_agent(user_question, info if info else {}, context)
    st.success(answer)
