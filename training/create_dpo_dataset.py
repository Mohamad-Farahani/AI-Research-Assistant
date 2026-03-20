import json
import random

with open("../data/processed_chunks.json") as f:
    chunks = json.load(f)

dataset = []

for c in chunks:

    prompt = "Explain the following machine learning concept."

    good_answer = c["text"]

    bad_answer = "This text describes a concept but the explanation is unclear."

    example = {
        "prompt": prompt + "\n\n" + c["text"],
        "chosen": good_answer,
        "rejected": bad_answer
    }

    dataset.append(example)

with open("dpo_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("DPO dataset created:", len(dataset))