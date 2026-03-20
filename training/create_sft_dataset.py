import json

with open("../data/processed_chunks.json") as f:
    chunks = json.load(f)

dataset = []

for c in chunks:

    example = {
        "instruction": "Explain the following research concept",
        "input": c["text"],
        "output": c["text"]
    }

    dataset.append(example)

with open("sft_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Dataset created:", len(dataset))