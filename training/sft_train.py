import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model


# ----------------------------
# Model
# ----------------------------

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantization config (8-bit)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ----------------------------
# LoRA Config
# ----------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# ----------------------------
# Dataset
# ----------------------------

dataset = load_dataset(
    "json",
    data_files="sft_dataset.json"
)


def format_prompt(example):

    prompt = f"""
### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}
"""

    tokens = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


dataset = dataset.map(format_prompt)

dataset = dataset["train"]


# ----------------------------
# Training Arguments
# ----------------------------

training_args = TrainingArguments(

    output_dir="./sft_model",

    per_device_train_batch_size=2,

    num_train_epochs=3,

    logging_steps=10,

    save_steps=100,

    learning_rate=2e-4,

    fp16=torch.cuda.is_available(),

    remove_unused_columns=False
)


# ----------------------------
# Trainer
# ----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)


# ----------------------------
# Train
# ----------------------------

trainer.train()

trainer.save_model("./sft_model")

print("Training finished. Model saved.")