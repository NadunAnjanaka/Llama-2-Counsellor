import streamlit as st
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline,
                          logging)

# Load model

model_name = "NadunAnjanaka/Llama-2-7b-chat-Counsellor"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

logging.set_verbosity(logging.CRITICAL)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=300)




# Chatbot UI

def get_response(prompt):
  response_raw = pipe(prompt)
  response = response_raw[0]['generated_text'].split("[/INST]")
  return response[1]


st.title("Counsellor Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display chat messages from history on app rerun
if st.session_state.messages != []:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
             st.markdown(message["content"])



# React to user input
if prompt := st.chat_input("Tell me how you feel"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})



    #code
    response = get_response(prompt)



    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



