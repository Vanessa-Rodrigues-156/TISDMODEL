import streamlit as st
import requests
import json
from datetime import datetime

# Configuration
OLLAMA_MODEL = "mistral_mental_health"  # This should match your model name in Ollama
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Fixed API endpoint

# Streamlit UI setup
st.title("Mental Health Support Chat")
st.caption(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("User: vanessa@dev")

class MentalHealthChat:
    def __init__(self, model_name=OLLAMA_MODEL):
        self.model_name = model_name
        self.setup_done = self._check_ollama_connection()
        
    def _check_ollama_connection(self):
        try:
            # Simple test request to check if Ollama is running
            test_data = {
                "model": self.model_name,
                "prompt": "test",
                "stream": False
            }
            response = requests.post(OLLAMA_API_URL, json=test_data, timeout=5)
            if response.status_code == 200:
                return True
            else:
                st.error(f"Ollama returned status code: {response.status_code}")
                st.error(f"Response: {response.text}")
                return False
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to Ollama. Make sure it's running.")
            return False
        except requests.exceptions.Timeout:
            st.error("Connection to Ollama timed out.")
            return False
        except Exception as e:
            st.error(f"Error checking Ollama connection: {str(e)}")
            return False

    def get_response(self, prompt):
        if not self.setup_done:
            return "Error: Ollama server is not running. Please start Ollama with 'ollama serve' command."
            
        try:
            # Get conversation history
            conversation_history = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in st.session_state.messages[-3:]
            ])
            
            # Prepare the request data with parameters from Modelfile
            data = {
                "model": self.model_name,
                "prompt": f"[INST]{conversation_history}\n\nCurrent message: {prompt}[/INST]",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["[INST]", "[/INST]"]
                }
            }
            
            # Make the request
            response = requests.post(OLLAMA_API_URL, json=data)
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"

# Initialize chat in session state
if 'chat' not in st.session_state:
    st.session_state.chat = MentalHealthChat()
    st.session_state.messages = []

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
            response = st.session_state.chat.get_response(prompt)
            st.write(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("Start New Chat"):
    st.session_state.messages = []
    st.rerun()
