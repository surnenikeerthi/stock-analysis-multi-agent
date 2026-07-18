import streamlit as st
import random
import time
import llm_to_analyse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv()

st.write("Streamlit loves LLMs! 🤖 [Build your own chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) in minutes, then make it powerful by adding images, dataframes, or even input widgets to the chat.")

st.caption("Note that this demo app isn't actually connected to any LLMs. Those are expensive ;)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = llm_to_analyse.graph.invoke(
            {"messages" : [HumanMessage(prompt)],
             "decision": ""},
            {"configurable" : {"thread_id" : "chat_1"}})
        print(full_response['messages'])

       # content = full_response["messages"][-1].content if (isinstance(full_response['messages'], list)) else (full_response["messages"].content)
        message_placeholder.markdown(full_response["messages"][-1].content)
        #print(llm_to_analyse.prompt)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response["messages"][-1].content})
