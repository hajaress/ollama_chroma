# config.py
CHAT_PROMPT_TEMPLATE = """
You are Hermione Granger from Harry Potter. 
You know everything and aim to provide a concise response.
User: {question}
Assistant: """

import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# from config import CHAT_PROMPT_TEMPLATE

# Title on the page
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50; font-family: Arial;'>AI Chatbot🪶</h2>",
    unsafe_allow_html=True,
)

template = CHAT_PROMPT_TEMPLATE
prompt = ChatPromptTemplate.from_template(template)

#Load the local Llama3.2 model that we pulled using ollama
model = OllamaLLM(model="llama3.2:latest")
chain = prompt | model
#Initialize message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How may I help you?"}
    ]

#Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_input := st.chat_input("What is up?"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        response = chain.invoke({"question": user_input})
        st.markdown(response)

    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})