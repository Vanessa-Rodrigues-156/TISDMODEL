import streamlit as st
import requests
from datetime import datetime
import json

# Configuration
API_URL = "https://api-inference.huggingface.co/models/savonamendes156/sthira"
# You'll need to get this from https://huggingface.co/settings/tokens
API_TOKEN = st.secrets["HUGGING_FACE_API_TOKEN"]  

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

def generate_response(prompt):
    try:
        # Get conversation history
        conversation_history = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in st.session_state.messages[-3:]
        ])
        
        # Prepare system prompt
        system_prompt = f"""You are a supportive friend having a conversation. Be natural and empathetic.
        Recent conversation:
        {conversation_history}

        Current message: {prompt}
        You are a warm, empathetic friend who listens and offers support. 
        Keep responses natural and conversational, like talking to a close friend. Listen to the user and respond with empathy. 
        Ask follow-up questions to show you care, let them speak their heart out. Share personal examples when appropriate.
        Avoid clinical language - speak from the heart."""

        # Make API request to Hugging Face
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": f"[INST]{system_prompt}[/INST]"}
        )
        
        if response.status_code == 200:
            return response.json()[0]["generated_text"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
st.title("Mental Health Support Chat 💬")
st.caption(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt + "Answer only in 2-3 sentences and form a conversation with the user by asking follow-up questions.")
            st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("Start New Chat"):
    st.session_state.messages = []
    st.rerun()