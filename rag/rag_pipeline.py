import json
import faiss
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# -------- Paths --------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

index_path = os.path.join(BASE_DIR, "embeddings", "vector_index.faiss")
metadata_path = os.path.join(BASE_DIR, "embeddings", "chunks_metadata.json")


# -------- Load Retriever --------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(index_path)

with open(metadata_path) as f:
    chunks = json.load(f)


def retrieve(query, k=5):

    query_embedding = embedding_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results


# -------- Load LLM --------

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


# -------- RAG Pipeline --------

def generate_answer(question):

    docs = retrieve(question)

    context = "\n\n".join([d["text"] for d in docs])

    prompt = f"""
You are a research assistant.

Use the following research excerpts to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer, docs


# -------- Run --------

if __name__ == "__main__":

    question = input("Ask a research question: ")

    answer, docs = generate_answer(question)

    print("\nANSWER:\n")
    print(answer)

    print("\nSOURCES:\n")

    for d in docs:
        print(d["paper_title"])
        print(d["source"])
        print()