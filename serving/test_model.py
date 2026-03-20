from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "../training/dpo_model")

prompt = "Explain what a transformer model is in machine learning."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200
)

print(tokenizer.decode(outputs[0]))