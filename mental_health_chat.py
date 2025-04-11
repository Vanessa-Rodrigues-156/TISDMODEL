import streamlit as st
import requests
import json
from datetime import datetime
import time

# Configuration
OLLAMA_MODEL = "mistral_mental_health"  # Use your local model name
OLLAMA_API_URL = "http://localhost:11434/api/generate"
TIMEOUT = 60  # Increased timeout to 60 seconds
MAX_RETRIES = 3  # Number of retries for failed requests

# Page configuration
st.set_page_config(
    page_title="Mental Health Support Chat",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("Mental Health Support Chat 💬")
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <p style='color: #666;'>A safe space to talk about your feelings and concerns</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This chat interface is powered by a specialized AI model trained on mental health discussions.
    
    **Features:**
    - Empathetic responses
    - Safe and supportive environment
    - Confidential conversations
    - 24/7 availability
    
    **Note:** This is not a replacement for professional help. 
    If you're in crisis, please contact a mental health professional.
    """)

class MentalHealthChat:
    def __init__(self, model_name=OLLAMA_MODEL):
        self.model_name = model_name
        self.setup_done = self._check_ollama_connection()
        
    def _check_ollama_connection(self):
        for attempt in range(MAX_RETRIES):
            try:
                test_data = {
                    "model": self.model_name,
                    "prompt": "test",
                    "stream": False
                }
                response = requests.post(OLLAMA_API_URL, json=test_data, timeout=TIMEOUT)
                if response.status_code == 200:
                    return True
                else:
                    st.warning(f"Attempt {attempt + 1}: Ollama returned status code: {response.status_code}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2)  # Wait before retrying
                        continue
                    st.error(f"Ollama returned status code: {response.status_code}")
                    st.error(f"Response: {response.text}")
                    return False
            except requests.exceptions.ConnectionError:
                st.warning(f"Attempt {attempt + 1}: Could not connect to Ollama. Retrying...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                st.error("Could not connect to Ollama. Please make sure it's running.")
                return False
            except requests.exceptions.Timeout:
                st.warning(f"Attempt {attempt + 1}: Connection timed out. Retrying...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                st.error(f"Connection to Ollama timed out after {TIMEOUT} seconds.")
                return False
            except Exception as e:
                st.error(f"Error checking Ollama connection: {str(e)}")
                return False
        return False

    def get_response(self, prompt):
        if not self.setup_done:
            return "Error: Ollama server is not running. Please start Ollama with 'ollama serve' command."
            
        for attempt in range(MAX_RETRIES):
            try:
                # Get conversation history
                conversation_history = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in st.session_state.messages[-3:]
                ])
                
                # Prepare the request data
                data = {
                    "model": self.model_name,
                    "prompt": f"[INST]{conversation_history}\n\nCurrent message: {prompt}[/INST]",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "stop": ["[INST]", "[/INST]"],
                        "num_predict": 512
                    }
                }
                
                # Make the request with retry logic
                response = requests.post(OLLAMA_API_URL, json=data, timeout=TIMEOUT)
                
                if response.status_code == 200:
                    return response.json()['response'].strip()
                else:
                    st.warning(f"Attempt {attempt + 1}: Ollama API Error: {response.status_code}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2)
                        continue
                    st.error(f"Ollama API Error: {response.status_code}")
                    st.error(f"Response: {response.text}")
                    return f"Error: {response.status_code} - {response.text}"
            except requests.exceptions.Timeout:
                st.warning(f"Attempt {attempt + 1}: Request timed out. Retrying...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                    continue
                st.error(f"Request timed out after {TIMEOUT} seconds.")
                return "I'm taking longer than expected to respond. Please try again or rephrase your question."
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return f"Error: {str(e)}"
        return "Sorry, I'm having trouble responding right now. Please try again later."

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
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Start New Chat"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 20px; color: #666;'>
        <p>Remember: You're not alone. Help is always available.</p>
    </div>
    """, unsafe_allow_html=True) 