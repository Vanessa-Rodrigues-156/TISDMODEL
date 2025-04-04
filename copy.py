import streamlit as st # type: ignore
from transformers import AutoTokenizer
from peft import PeftModel # type: ignore
import torch # type: ignore
import requests # type: ignore
from datetime import datetime
from transformers import AutoModelForCausalLM
# Your settings
OPENROUTER_API_KEY = "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"
LORA_PATH = "mistral_mental_health"

class ChatBot:
    def __init__(self):
        self.setup_done = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
            base_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.3",
                device_map="auto",  # optional but recommended
                torch_dtype="auto"  # or torch.float16 if you're using it
            )
            
            # Load the LoRA model
            self.lora_model = PeftModel.from_pretrained(
                base_model,
                LORA_PATH
            )
            self.setup_done = True
        except RuntimeError as e:
            if "torch::class_" in str(e):
                st.warning("A PyTorch class registration error occurred. Ignoring this step.")
            else:
                st.error(f"Error loading models: {str(e)}")

    def get_response(self, prompt, use_lora=True):
        if use_lora and self.setup_done:
            try:
                conversation_history = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in st.session_state.messages[-3:]
                ])
                
                system_prompt = f"""You are a supportive friend having a conversation. Be natural and empathetic.
                Recent conversation:
                {conversation_history}

                Current message: {prompt}
You are a warm, empathetic friend who listens and offers support. 
        Keep responses natural and conversational, like talking to a close friend. 
        Ask follow-up questions to show you care. Share personal examples when appropriate.
        Avoid clinical language - speak from the heart.If you feel the person is going towards serious mental health issues then slowly persuade the person to go to a counselor not directly"""

                inputs = self.tokenizer(f"[INST]{system_prompt}[/INST]", return_tensors="pt").to(self.lora_model.device)
                outputs = self.lora_model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response.replace(system_prompt, "").strip()
            except Exception as e:
                return self.get_openrouter_response(prompt)
        else:
            return self.get_openrouter_response(prompt)

    def get_openrouter_response(self, prompt):
        conversation_history = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in st.session_state.messages[-3:]
        ])
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8501",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a supportive friend having a conversation. Your responses should be:
                    1. Natural and conversational - like a real friend
                    2. Empathetic and understanding
                    3. Gently perceptive of emotional state
                    4. If you notice signs of depression, gradually and naturally guide towards suggesting professional help
                    5. Focus on building trust and understanding first
                    Never start with crisis resources or hotlines unless absolutely necessary."""
                },
                {
                    "role": "user",
                    "content": f"Previous conversation:\n{conversation_history}\n\nCurrent message: {prompt}"
                }
            ],
            "max_tokens": 500,
            "temperature": 0.8
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

# Streamlit UI
st.title("Friendly Chat")
st.caption("2025-04-04 07:25:26")  # Using your specified datetime
st.caption("User: vanessa@dev")

# Initialize bot in session state
if 'bot' not in st.session_state:
    st.session_state.bot = ChatBot()
    st.session_state.messages = []

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("What's on your mind?"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            response = st.session_state.bot.get_response(prompt)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("Start New Chat"):
    st.session_state.messages = []
    st.rerun()  # Using st.rerun() instead of experimental_rerun()