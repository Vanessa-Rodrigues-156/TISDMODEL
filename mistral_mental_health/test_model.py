
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_response(prompt):
    formatted_prompt = f"[INST]{prompt}[/INST]"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test prompts
test_prompts = [
    "How can I help someone who's experiencing anxiety attacks?",
    "What are the early warning signs of depression I should watch out for?",
    "I'm feeling overwhelmed with stress at work. What coping strategies would you recommend?"
]

for prompt in test_prompts:
    print(f"Prompt: {prompt}")
    print(f"Response: {generate_response(prompt)}")
