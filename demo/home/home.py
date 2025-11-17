import streamlit as st

st.markdown(
    """
# Whisky RAG Service Demo

Welcome to the **Whisky AI Recommendation Service** demonstration!


### Background
This AI-powered service leverages 1,000 high-quality whisky reviews from the Hippo Whisky Club database to provide personalized whisky recommendations and detailed explanations based on your preferences.

- :blue[**Market Context** :] The whisky market has experienced significant volatility, with the WhiskyStats index declining approximately 35.8% (from 357.94 to 229.77)
- :blue[**Challenge** :] Traditional offline expert consultations have scalability limitations
- :blue[**Solution** :] An AI-driven recommendation service utilizing extensive review data to deliver personalized whisky guidance

### Technology Stack

This demo is built using:

- :blue[**Framework** :] LangChain for RAG pipeline construction
- :blue[**Vector Database** :] FAISS for efficient similarity search
- :blue[**LLM** :] Google Gemini 2.5 Flash for response generation
- :blue[**Embedding** :] Google Gemini Embedding (`gemini-embedding-001`)
- :blue[**UI Framework** :] Streamlit for interactive web interface

### Features

The chatbot can help you with:

- Finding whiskies that match your taste preferences (fruity, smoky, sweet, etc.)
- Exploring flavor profiles including nose, palate, and finish characteristics
- Discovering whiskies similar to ones you already enjoy
- Getting detailed tasting notes and expert insights

---

### Available Pages

Navigate through the sidebar to explore:

- :blue[**Settings** :] Configure the RAG system parameters and vector database settings
- :blue[**RAG Application** :] Interact with the AI chatbot and get personalized whisky recommendations
""",
)
