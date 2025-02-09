from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "meta-llama/Llama-3-8B"  # Use a smaller model if needed

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

def chat_with_llama3(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
