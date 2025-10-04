import os
import json
import streamlit as st
from dotenv import load_dotenv
import openai

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

st.title("Air Travel Rules Agent")

user_question = st.text_input("Ask a question about airline rules:")

if user_question:
    info = find_airline_info(user_question)
    if not info:
        st.warning("Sorry, I couldn't find information for that airline.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_gpt_agent(user_question, info)
        st.success(answer)
