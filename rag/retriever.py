import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build paths
index_path = os.path.join(BASE_DIR, "embeddings", "vector_index.faiss")
metadata_path = os.path.join(BASE_DIR, "embeddings", "chunks_metadata.json")

# Load FAISS index
index = faiss.read_index(index_path)

# Load metadata
with open(metadata_path) as f:
    chunks = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, k=5):

    print("\nUser Question:", query)

    # Convert question to embedding
    query_embedding = model.encode([query]).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results


if __name__ == "__main__":

    question = input("Ask a research question: ")

    results = retrieve(question)

    print("\nTop relevant chunks:\n")

    for r in results:
        print("TITLE:", r["paper_title"])
        print("TEXT:", r["text"])
        print("SOURCE:", r["source"])
        print("-" * 50)