import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import requests
import json

# Your API key
OPENROUTER_API_KEY = "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"  # Replace with your key

class ChatBot:
    def __init__(self):
        self.setup_done = False
        try:
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
            self.lora_model = PeftModel.from_pretrained(
                self.base_model,
                "./mistral_mental_health"
            )
            self.setup_done = True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

    def get_openrouter_response(self, prompt):
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8501",
            "Content-Type": "application/json"
        }
        
        # More conversational system prompt
        system_prompt = """You are a warm, empathetic friend who listens and offers support. 
        Keep responses natural and conversational, like talking to a close friend. 
        Ask follow-up questions to show you care. Share personal examples when appropriate.
        Avoid clinical language - speak from the heart."""
        
        data = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,  # Add some variability
            "presence_penalty": 0.6,  # Encourage diverse responses
            "frequency_penalty": 0.6  # Reduce repetition
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Sorry, I'm having trouble connecting right now. Can we try again? ({str(e)})"

    def get_response(self, prompt, use_lora=True):
        conversation_prompt = f"""As a supportive friend, respond to: {prompt}
        Be warm, natural, and conversational. Ask questions to show you care."""
        
        try:
            if use_lora and self.setup_done:
                inputs = self.tokenizer(f"[INST]{conversation_prompt}[/INST]", return_tensors="pt").to(self.lora_model.device)
                outputs = self.lora_model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                return self.get_openrouter_response(conversation_prompt)
        except Exception as e:
            return f"Hey, I'm having some trouble right now. Mind if we try again? ({str(e)})"

# Streamlit UI with more personal touch
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .bot-message {
        background-color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Hey there! 👋 Let's chat")
st.markdown("I'm here to listen and chat with you about anything that's on your mind.")

# Initialize bot in session state
if 'bot' not in st.session_state:
    st.session_state.bot = ChatBot()
    st.session_state.messages = []

# Model selection in sidebar with friendlier language
with st.sidebar:
    st.markdown("### Chat Settings")
    use_lora = st.checkbox("Use my specialized training", value=True)
    
    st.markdown("### Want to talk about...")
    prompts = [
        "Been feeling down lately...",
        "Everything feels overwhelming right now",
        "Having trouble with sleep",
        "Need to vent about something",
        "Just want someone to listen"
    ]
    
    for prompt in prompts:
        if st.button(prompt):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                response = st.session_state.bot.get_response(prompt, use_lora)
                st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()

    if st.button("Start Fresh"):
        st.session_state.messages = []
        st.experimental_rerun()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='chat-message {'user-message' if message['role'] == 'user' else 'bot-message'}'>{message['content']}</div>", unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Share what's on your mind..."):
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-message user-message'>{prompt}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.bot.get_response(prompt, use_lora)
            st.markdown(f"<div class='chat-message bot-message'>{response}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})