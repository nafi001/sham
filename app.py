import streamlit as st
from model import chat_with_llama3

st.set_page_config(page_title="LLaMA 3 Chatbot", layout="wide")

st.title("🤖 LLaMA 3 Chatbot")
st.info("Ask me anything!")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get LLaMA 3 response
    response = chat_with_llama3(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
