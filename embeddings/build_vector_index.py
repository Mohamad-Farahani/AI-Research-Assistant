import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load processed chunks
with open("../data/processed_chunks.json") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(texts)} chunks.")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings shape:", embeddings.shape)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index built.")

# Save index
faiss.write_index(index, "vector_index.faiss")

# Save metadata
with open("chunks_metadata.json", "w") as f:
    json.dump(chunks, f)

print("Index and metadata saved.")