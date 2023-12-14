import streamlit as st
import time
import logging
from utils import setup_query_pipe

###################
## Setup logging ##
###################

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)


####################
## Streamlit code ##
####################

# Setup page headers and titles
st.set_page_config(
    page_title="Tesla Model 3 Helpbot",
    page_icon="assets/icon-tesla-48x48.png"
)

col_icon, col_title = st.columns([0.1, 0.9])
with col_icon:
    st.image("assets/icon-tesla-48x48.png", use_column_width="always")
with col_title:
    st.title(f":red[Tesla] :blue[Model 3] Helpbot 🤖")

st.write(f"Powered by :orange[llama2-7b] / :blue[mistral-7b] and :violet[haystack] for Retrieval Augmented Generation")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define sidebar
st.sidebar.header("Parameters for text generation")
st.session_state.llm = st.sidebar.selectbox('Large Language Model:', ('Mistral-7B', 'Llama-2-7B'), help="Choose a language model for answering your questions.")
reset_chat = st.sidebar.button("Clear Chat")

# Reset chat history
if reset_chat:
    st.session_state.messages = []

# Initialize query pipeline
if "query_pipeline" not in st.session_state:
    with st.spinner('Loading...'):
        st.session_state.query_pipeline = {
            'Mistral-7B': setup_query_pipe('Mistral-7B'),
            'Llama-2-7B': setup_query_pipe('Llama-2-7B')
        }

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Ask a question"):

    # Display user message in chat message container
    st.chat_message("user").markdown(user_input)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner('Generating answer...'):
            try:
                user_input = user_input.strip()
                # Pass in pipeline arguments for each query run
                response = st.session_state.query_pipeline[st.session_state.llm].run({
                    "text_embedder": {"text": user_input}, 
                    "retriever": {"top_k": 3}, 
                    "prompt_builder": {"question": user_input}
                })

                llm_response = response['llm']['replies'][0]

                for chunk in llm_response.split():
                    # Break response into chunks
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            except Exception as e:
                logging.warning(f"Exception occurred while getting response from API endpoint")
                logging.warning(e)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
