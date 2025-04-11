import streamlit as st
import requests
from datetime import datetime
import json

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral_mental_health"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to generate a response using Ollama
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
        Keep responses natural and conversational, like talking to a close friend. 
        Always Ask follow-up questions to show you care and you want to listen to their problems. Share personal examples when appropriate and extremely required and in short.
       
        Avoid clinical language - speak from the heart. If you feel the person is going towards serious mental health issues then slowly persuade the person to go to a counselor not directly."""

        # Prepare the request payload
        payload = {
            "model": MODEL_NAME,
            "prompt": f"[INST]{system_prompt}[/INST]",
            "stream": False
        }

        # Make the API request
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        # Parse the response
        response_data = response.json()
        return response_data.get("response", "").strip()
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