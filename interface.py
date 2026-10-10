import streamlit as st
import random
import time
import llm_to_analyse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command
load_dotenv()

st.write("Streamlit loves LLMs! 🤖 [Build your own chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) in minutes, then make it powerful by adding images, dataframes, or even input widgets to the chat.")

st.caption("Note that this demo app isn't actually connected to any LLMs. Those are expensive ;)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

if "pending_question" not in st.session_state:          # NEW
    st.session_state.pending_question = None            # NEW

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
        if st.session_state.pending_question:  # NEW
            full_response = llm_to_analyse.graph.invoke(  # NEW
                Command(resume=prompt),  # NEW
                {"configurable": {"thread_id": "chat_1"}})  # NEW
        else:  # NEW
            full_response = llm_to_analyse.graph.invoke(  # CHANGED (indented under else)
                {"messages": [HumanMessage(prompt)],
                 "decision": ""},
                {"configurable": {"thread_id": "chat_1"}})
        if ("__interrupt__" in full_response):
            system_response = full_response["__interrupt__"][-1].value
            message_placeholder.markdown(system_response)
            st.session_state.pending_question = system_response  # CHANGED (replaces the second invoke + chat_input)
        else:
            print(full_response['messages'])
            system_response = full_response["messages"][-1].content
            message_placeholder.markdown(system_response)
            st.session_state.pending_question = None
            # content = full_response["messages"][-1].content if (isinstance(full_response['messages'], list)) else (full_response["messages"].content)
        #message_placeholder.markdown(system_response)
        #print(llm_to_analyse.prompt)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": system_response})
