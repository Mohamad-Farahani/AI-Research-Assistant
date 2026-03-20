from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# allow importing rag pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from rag.rag_pipeline import generate_answer

app = FastAPI()

class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Research AI Assistant is running"}


@app.post("/ask")
def ask_question(q: Question):

    answer, docs = generate_answer(q.question)

    sources = []

    for d in docs:
        sources.append({
            "title": d["paper_title"],
            "source": d["source"]
        })

    return {
        "question": q.question,
        "answer": answer,
        "sources": sources
    }