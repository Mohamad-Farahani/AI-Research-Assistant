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

## 🖥️ Demo

Run locally:

```bash
streamlit run app.py
