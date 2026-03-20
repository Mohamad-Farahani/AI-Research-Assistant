import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
# load papers
with open("../data/papers.json") as f:
    papers = json.load(f)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

processed_chunks = []

for paper in papers:
    text = paper["abstract"]

    chunks = splitter.split_text(text)

    for chunk in chunks:
        processed_chunks.append({
            "text": chunk,
            "paper_title": paper["title"],
            "source": paper["url"]
        })

with open("../data/processed_chunks.json", "w") as f:
    json.dump(processed_chunks, f, indent=2)

print(f"Created {len(processed_chunks)} chunks.")