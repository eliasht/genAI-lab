import sys

sys.path.append("salesmate-be/app")

import streamlit as st
from utils import init_azure_openai_models

# Set embedding model

init_azure_openai_models()

# Load index 
from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="salesmate-be\local_demo\storage")

# load index
index = load_index_from_storage(storage_context, index_id="vector_chunked_index")

# Define query engine
query_engine = index.as_query_engine(similarity_top_k=5)


def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Logo 
st.image("https://eproc.unitedtractors.com/Content/assets/images/logo-ut-baru.png", width=350)

# Title
st.title("Salesmate Chatbot")

# Side bar content

with st.sidebar:
    st.title("Salesmate Chatbot")
    st.markdown(
        """Knowledge management expert for Business Consultants """
    )


mode_descriptions = {
    "OpenAI": [
        "OpenAI LLMs.",
        30,
    ],
    "Vertex AI": [
        "Vertex AI LLMs.",
        15,
    ],
}


# User Configuration Sidebar
# with st.sidebar:
#     mode = st.radio(
#         "**LLM options for answer generation**", options=["OpenAI", "Vertex AI"], index=1
#     )
#     st.info(mode_descriptions[mode][0])

st.divider()

# Chat area
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "Hey! I am your chat assistant, help you to answer questions on Product Specification and Comparison category!"
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

# Example prompts
example_prompts = [
    "What is the difference between PC200-8 and PC200 7?",
    "What is the fuel consumption comparison between PC210 and PC400?",
    "I need to do a land clearing of 100ha, how many units do I need?",
    "How many hot leads were identified in the past week, and who are they?",
    "What is the productivity of an excavator unit?",
    "PC300 frequently has engine problems, what are the improvements?",
]

button_cols_1 = st.columns(3)
button_cols_2 = st.columns(3)
button_all = button_cols_1 + button_cols_2

button_pressed = ""

for i in range(6):
    if button_all[i].button(example_prompts[i]):
        button_pressed = example_prompts[i]
        break


if prompt := (st.chat_input("What is the question you want to answer?") or button_pressed):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if prompt != "":
        query = prompt.strip().lower()
        with st.chat_message("assistant"):
            ### will be removed
            # Create a chat completion, will be removed after talking to backend
            response = query_engine.query(prompt)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            st.rerun()
