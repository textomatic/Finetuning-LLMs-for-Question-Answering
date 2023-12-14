import streamlit as st
import time
import json
import logging
import boto3


###################
## Setup basics  ##
###################

# logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)

# sagemaker client
sm_client = boto3.client('sagemaker-runtime')

# sagemaker inference parameters for each model
sm_parameters = {
    "Mistral-7B-Finetuned": {
        "max_new_tokens": 1024,
        "return_full_text": False,
        "repetition_penalty": 1.2,
        "stop": ["</s>"],
    }, 
    "Llama-2-7B-Finetuned": {
        "max_new_tokens": 1024,
        "return_full_text": False,
        "stop": ["</s>"],
    }, 
}

####################
## Streamlit code ##
####################

# setup page headers and title
st.set_page_config(
    page_title="Tesla Model 3 Helpbot",
    page_icon="assets/icon-tesla-48x48.png"
)

col_icon, col_title = st.columns([0.1, 0.9])
with col_icon:
    st.image("assets/icon-tesla-48x48.png", use_column_width="always")
with col_title:
    st.title(f":red[Tesla] :blue[Model 3] Helpbot 🤖")

st.write(f"Powered by :orange[llama2-7b] fine-tuned and :blue[mistral-7b] fine-tuned")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define sidebar
st.sidebar.header("Parameters for text generation")
st.session_state.llm = st.sidebar.selectbox('Large Language Model:', ('Mistral-7B-Finetuned', 'Llama-2-7B-Finetuned'), help="Choose a language model for answering your questions.")
reset_chat = st.sidebar.button("Clear Chat")

# Reset chat history
if reset_chat:
    st.session_state.messages = []

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

        with st.spinner('Thinking hard...'):
            try:
                # Format prompt and prepare payload for Sagemaker endpoint
                user_input = "[INST] " + user_input.strip() + " [/INST]"
                sm_payload = {
                    "inputs": user_input,
                    "parameters": sm_parameters[st.session_state.llm]
                }
                # Invoke Sagemaker inference endpoint
                response = sm_client.invoke_endpoint(
                    EndpointName=st.session_state.llm.lower(), 
                    ContentType="application/json",
                    Accept="application/json",
                    Body=json.dumps(sm_payload)
                )

            except Exception as e:
                logging.warning(f"Exception occurred while getting response from API endpoint")
                logging.warning(e)
                raise e

        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            # Deserialize response body into JSON and sanitize it
            answer = json.loads(response['Body'].read().decode("utf-8"))[0]["generated_text"].replace("</s>", "").strip()
            # Simulate stream of response with milliseconds delay
            for chunk in answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        else:
            full_response = "Sorry, an error occurred in the service. Please try again later!"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
