import arxiv
import json

query = "machine learning OR reinforcement learning OR transformers"
client = arxiv.Client()
search = arxiv.Search(
    query=query,
    max_results=200,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

papers = []

for result in client.results(search):
    paper = {
        "title": result.title,
        "abstract": result.summary,
        "authors": [a.name for a in result.authors],
        "published": str(result.published),
        "url": result.entry_id
    }

    papers.append(paper)

with open("../data/papers.json", "w") as f:
    json.dump(papers, f, indent=2)

print(f"Downloaded {len(papers)} papers.")

