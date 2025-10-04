# Air Travel Rules Agent

This project uses LangChain to build an agent that answers questions about air travel rules and regulations for specific airlines.

## Features
- Backend using Python and LangChain
- Streamlit UI for user interaction
- Sample knowledge base (CSV/JSON)
- Easy setup and run instructions

## Setup Instructions

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv rag_venv
   source rag_venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```
4. Test the UI:
   - Open the provided local URL in your browser and ask questions about airline rules.

## Project Structure
- `app.py`: Streamlit app and agent logic
- `airline_rules.json`: Sample airline rules knowledge base
- `requirements.txt`: Python dependencies
