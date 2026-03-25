# 📚 AI Research Assistant

An end-to-end LLM system for answering research questions using Retrieval-Augmented Generation (RAG), fine-tuned language models, and high-performance inference.


## 🚀 Features

- RAG pipeline using FAISS for document retrieval
- Fine-tuned LLM using LoRA (SFT + DPO)
- Optimized inference with vLLM (GPU)
- Streamlit-based interactive UI
- OpenAI-compatible API

## 🧠 Tech Stack

- PyTorch
- Hugging Face Transformers
- TRL (DPO training)
- vLLM
- FAISS
- Streamlit

## ⚙️ Architecture


![Demo of the application](structure.png)
        ┌────────────────────────────┐
        │   Research Papers (Arxiv)  │
        └─────────────┬──────────────┘
                      │
              Data Processing
                      │
          Chunking + Embeddings
                      │
        ┌─────────────▼─────────────┐
        │   Vector DB (FAISS)       │
        └─────────────┬─────────────┘
                      │
              RAG Retrieval
                      │
        ┌─────────────▼─────────────┐
        │  Fine-tuned LLM (LoRA)    │
        │   SFT + DPO Training      │
        └─────────────┬─────────────┘
                      │
           vLLM Inference Server
                      │
        ┌─────────────▼─────────────┐
        │   API (OpenAI format)     │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Streamlit UI (Frontend) │
        └───────────────────────────┘
        
Data → Embeddings → Vector DB → RAG → Fine-tuned Model → vLLM API → UI



## Tech Stack

Python

PyTorch

Hugging Face Transformers

TRL (DPO Training)

FAISS

vLLM

Streamlit


## 🔥 Highlights

Built a full LLM pipeline from scratch locally

Implemented RAG + vector search

Applied LoRA fine-tuning (SFT + DPO)

Deployed with vLLM for optimized GPU inference

Created a working AI product (UI + API)

## 🚧 Future Improvements

📄 Full PDF ingestion

📊 Evaluation metrics (BLEU, ROUGE, etc.)

☁️ Cloud deployment (AWS / GCP)

🧠 Better prompt engineering

## 🖥️ Demo

![Demo of the application](Animation.gif)
Run locally:

```bash
streamlit run app.py


