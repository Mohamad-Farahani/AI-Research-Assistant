from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "../training/dpo_model")

model = model.merge_and_unload()

model.save_pretrained("../serving/merged_model")
tokenizer.save_pretrained("../serving/merged_model")

print("Model merged successfully")