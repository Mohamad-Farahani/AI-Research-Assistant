import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel
from trl import DPOTrainer


# ----------------------------
# Base Model
# ----------------------------

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)


# ----------------------------
# Load LoRA Adapter
# ----------------------------

model = PeftModel.from_pretrained(
    base_model,
    "../training/sft_model"
)

model.train()

# important for DPO
model.enable_input_require_grads()

# reduces GPU memory usage
model.gradient_checkpointing_enable()


# ----------------------------
# Dataset
# ----------------------------

dataset = load_dataset(
    "json",
    data_files="dpo_dataset.json"
)["train"]


# ----------------------------
# Training Arguments
# ----------------------------

training_args = TrainingArguments(

    output_dir="./dpo_model",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=2,

    num_train_epochs=1,

    logging_steps=10,

    remove_unused_columns=False,

    report_to="none"
)


# ----------------------------
# Trainer
# ----------------------------

trainer = DPOTrainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    tokenizer=tokenizer,

    max_length=512,

    max_prompt_length=128
)


# ----------------------------
# Train
# ----------------------------

trainer.train()

trainer.save_model("./dpo_model")

print("DPO training finished.")