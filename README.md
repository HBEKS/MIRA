# 🤖 MIRA - Multilingual Intelligent Research Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-success.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-orange.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **An end-to-end multilingual Retrieval-Augmented Generation (RAG) research assistant that enables grounded question answering over PDF documents using local Large Language Models powered by Ollama.**

---

## 🚀 Overview

**MIRA (Multilingual Intelligent Research Assistant)** is an end-to-end **Generative AI application** built with **LangChain**, **LangGraph**, **ChromaDB**, and **Ollama**.

The application allows users to upload academic papers in PDF format and ask questions in either **English** or **Indonesian**. By combining semantic retrieval, query rewriting, and cross-encoder re-ranking, MIRA produces highly relevant answers while minimizing hallucinations through grounded generation.

Designed as a **fully local AI assistant**, MIRA ensures user privacy by processing all data on-device without relying on external cloud APIs.

---

# ✨ Features

* 📄 Upload and analyze academic PDF documents
* 🔍 Advanced Retrieval-Augmented Generation (RAG)
* 🧠 Query Rewriting for improved retrieval quality
* 🎯 Cross-Encoder Re-ranking for context optimization
* 🌐 Multilingual responses (English & Indonesian)
* 💬 Persistent chat history
* ⚡ Dynamic retrieval optimization based on available RAM
* 🏠 Fully local deployment with Ollama
* 🔒 Privacy-friendly architecture with offline processing
* 📚 Grounded answer generation to minimize hallucinations

---

# 🏗️ Architecture

```text
                 ┌─────────────────────┐
                 │    PDF Document     │
                 └──────────┬──────────┘
                            │
                    PDF Processing
                            │
                   Dynamic Chunking
                            │
                    Embedding Model
                            │
                    Chroma Vector DB
                            │
                     User Question
                            │
                     Query Rewriting
                            │
                     Vector Retrieval
                            │
                  Cross-Encoder Ranking
                            │
                   Context Selection
                            │
                  Llama 3.2 via Ollama
                            │
                  Translation (Optional)
                            │
                      Final Response
```

---

# 📚 Table of Contents

* Features
* Technology Stack
* System Requirements
* Installation
* Configuration
* Usage
* Project Structure
* Troubleshooting
* Performance
* Portfolio Highlights
* Future Improvements
* License

---

# 🛠️ Technology Stack

| Component            | Technology                            |
| -------------------- | ------------------------------------- |
| Programming Language | Python                                |
| LLM                  | Llama 3.2 3B (Ollama)                 |
| Embedding Model      | nomic-embed-text-v2-moe               |
| AI Framework         | LangChain                             |
| Workflow Engine      | LangGraph                             |
| Vector Database      | ChromaDB                              |
| Re-ranking           | Cross Encoder (MS MARCO MiniLM-L6-v2) |
| Frontend             | Streamlit                             |
| Package Manager      | uv                                    |
| PDF Processing       | PyPDF                                 |
| Local Deployment     | Ollama                                |

---

# 💻 System Requirements

## Minimum

* Python 3.10–3.12
* 8 GB RAM
* Windows / Linux / macOS
* 10 GB available storage

## Recommended

* 16–32 GB RAM
* NVIDIA GPU (8 GB VRAM or above)
* 20 GB storage

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/MIRA.git

cd MIRA
```

or

```bash
uv init MIRA

cd MIRA
```

---

## Install Ollama

### Windows

Download and install Ollama from:

https://ollama.com

### Linux/macOS

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

## Download Models

```bash
# Main LLM

ollama pull llama3.2:3b

# Lightweight alternative

ollama pull gemma2:2b

# Embedding model

ollama pull nomic-embed-text-v2-moe
```

---

## Install Dependencies

```bash
uv add langchain langchain-ollama langchain-chroma chromadb

uv add langchain-community langchain-text-splitters

uv add streamlit pypdf python-dotenv

uv add langdetect tenacity

uv add sentence-transformers

uv add psutil torch
```

---

## Create Required Directories

```bash
mkdir -p agents pipeline utils

mkdir -p chroma_db chat_history
```

---

# ⚙️ Configuration

## Run Ollama

```bash
ollama serve
```

Verify installed models:

```bash
ollama list
```

---

## Optional Environment Variables

Create a `.env` file:

```env
OLLAMA_HOST=http://127.0.0.1:11434
```

---

## Automatic Hardware Optimization

| RAM      | Chunk Size | Retrieval K | Batch Size |
| -------- | ---------- | ----------- | ---------- |
| ≥32 GB   | 800        | 12          | 64         |
| 16–32 GB | 600        | 10          | 48         |
| 8–16 GB  | 500        | 8           | 32         |
| <8 GB    | 400        | 6           | 16         |

---

# 🚀 Usage

Run the application:

```bash
uv run streamlit run ui/app.py
```

Open:

```
http://localhost:8501
```

---

## Workflow

1. Upload a PDF document.
2. Select the preferred output language.
3. Ask questions related to the document.
4. View persistent conversation history.

---

## Example Questions

```text
What methodology is used in this research?

Summarize the main findings.

Explain the proposed model architecture.

Compare the proposed method with baseline approaches.
```

---

# 📁 Project Structure

```text
MIRA/

├── ui/
│   └── app.py

├── agents/
│   ├── state.py
│   ├── nodes.py
│   └── graph_mira.py

├── pipeline/
│   └── pdf_processor.py

├── utils/
│   ├── system_check.py
│   └── rag_optimizer.py

├── chroma_db/

├── chat_history/

├── pyproject.toml

└── README.md
```

---

# 🔧 Troubleshooting

## Ollama Connection Error

```bash
ollama serve

ollama list

netstat -an | findstr "11434"
```

---

---

## Model Not Found

```bash
ollama pull llama3.2:3b

ollama pull nomic-embed-text-v2-moe

ollama list
```

---

## Out of Memory

Possible solutions:

* Switch to `gemma2:2b`
* Reduce `chunk_size`
* Reduce `k_retrieval`

---

# 📊 Performance

| Metric                  | Value       |
| ----------------------- | ----------- |
| PDF Indexing (30 pages) | 30–60 s     |
| Response Time           | 5–15 s      |
| RAG Accuracy            | 75–85%      |
| Maximum PDF Size        | 200 MB      |
| Deployment              | Fully Local |

---

# 💼 Portfolio Highlights

This project demonstrates expertise in:

* End-to-End Generative AI Development
* Retrieval-Augmented Generation (RAG)
* LangChain & LangGraph Workflow Orchestration
* Semantic Search Systems
* Cross-Encoder Re-ranking
* Local LLM Deployment
* Vector Database Integration
* Prompt Engineering
* AI Application Optimization
* Streamlit Full-Stack Development

Suitable for showcasing skills relevant to:

* AI Engineer
* Generative AI Engineer
* LLM Engineer
* Machine Learning Engineer
* NLP Engineer
* AI Software Engineer

---

# 🛣️ Future Improvements

* Multi-document Retrieval
* Hybrid Search (BM25 + Dense Retrieval)
* Citation Highlighting
* Agentic RAG Workflow
* Knowledge Graph Integration
* Voice-based Interaction
* Docker Support
* REST API
* Cloud Deployment
* Authentication & Multi-user Support

---

# 🤝 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more details.

---

# 🙏 Acknowledgements

* LangChain
* LangGraph
* Ollama
* ChromaDB
* Streamlit
* Sentence Transformers
* PyTorch

---

**Built with ❤️ using Python, LangChain, LangGraph, ChromaDB, Streamlit, and Ollama.**
