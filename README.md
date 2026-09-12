# 🚀 Advanced RAG (Retrieval-Augmented Generation) Learning Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-FF6F61?style=for-the-badge)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-3.6_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Cohere](https://img.shields.io/badge/Cohere-Rerank_v3.0-3949AB?style=for-the-badge)](https://cohere.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end hands-on exploration and implementation of **Retrieval-Augmented Generation (RAG)** systems. This repository documents a progressive learning journey—starting from basic vector database ingestion to advanced document chunking, multi-query expansion, Reciprocal Rank Fusion (RRF), sparse-dense hybrid search (BM25 + ChromaDB), Cohere re-ranking, and history-aware conversational RAG pipelines.

---

## 📌 Architecture Overview

```
                          ┌───────────────────────────┐
                          │   Raw Documents (docs/)   │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │            Document Chunking Strategies             │
             │   (Recursive, Semantic, & Agentic LLM Chunking)     │
             └──────────────────────────┬──────────────────────────┘
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │       Embeddings & Vector Storage (ChromaDB)        │
             │   (HuggingFace all-MiniLM-L6-v2 + Cosine Similarity) │
             └──────────────────────────┬──────────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
             ┌──────────────────────┐      ┌──────────────────────┐
             │  Dense Vector Search │      │ Sparse Keyword Search│
             │     (ChromaDB)       │      │       (BM25)         │
             └──────────┬───────────┘      └──────────┬───────────┘
                        │                             │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
             ┌─────────────────────────────────────────────────────┐
             │     Retrieval & Ranking Optimization Modules        │
             │  • Multi-Query Variations    • RRF (Reciprocal Rank)    │
             │  • MMR (Max Diversity)       • Cohere Two-Stage Rerank  │
             └─────────────────────────┬───────────────────────────┘
                                       │
                                       ▼
             ┌─────────────────────────────────────────────────────┐
             │      History-Aware Standalone Query Rewriter        │
             └─────────────────────────┬───────────────────────────┘
                                       │
                                       ▼
             ┌─────────────────────────────────────────────────────┐
             │         LLM Generation (Gemini 3.6 Flash)           │
             │         Grounded Answer Synthesis & Output          │
             └─────────────────────────┬───────────────────────────┘
```

---

## 🗺️ Step-by-Step Learning Progression

This repository is structured sequentially to reflect a step-by-step mastery of modern RAG architectures:

### 🔹 Phase 1: Basic RAG Ingestion & Storage
* **Document Ingestion (`ingestion_pipeline.py`)**: Built an automated loader reading text corpora (`docs/`) using LangChain's `DirectoryLoader` & `TextLoader`.
* **Vector Store Persistence**: Generated vector embeddings using HuggingFace's open-source `sentence-transformers/all-MiniLM-L6-v2` model and persisted vectors to **ChromaDB** with Cosine similarity space (`hnsw:space: cosine`).
* **Naive Retrieval (`retrieval_pipeline.py`)**: Implemented simple similarity retrieval matching user queries against ChromaDB embeddings.

### 🔹 Phase 2: Advanced Text Chunking Strategies
Chunk size and boundaries dictate retrieval quality. Explored four distinct chunking paradigms:
1. **Character Splitter (`recursive_char_text_splitter.py`)**: Fixed-length character splitting.
2. **Recursive Character Splitter (`recursive_char_text_splitter.py`)**: Hierarchical splitting using `["\n\n", "\n", ". ", ",", " "]` to preserve sentence structure.
3. **Semantic Chunking (`semantic_chunking.py`)**: Using distance thresholds between consecutive sentence embeddings to detect natural topic shifts (`langchain_experimental.text_splitter.SemanticChunker`).
4. **Agentic Chunking (`agentic_chunking.py`)**: Utilizing LLM intelligence (`ChatGoogleGenerativeAI`) to analyze context and insert explicit `<<<SPLIT>>>` markers based on semantic context boundaries.

### 🔹 Phase 3: Advanced Retrieval Techniques & Ranking
Basic similarity search often suffers from low recall or duplicate information. Implemented advanced retrieval algorithms:
* **Similarity Score Threshold (`retrieval_methods.py`)**: Filters out retrieved documents with similarity scores below a confidence cutoff (e.g., `score_threshold: 0.3`).
* **Maximal Marginal Relevance (MMR) (`retrieval_methods.py`)**: Balances document relevance with diversity (`lambda_mult: 0.5`) to eliminate redundant chunks.
* **Multi-Query Expansion (`multi_query_retrieval.py`)**: Uses Pydantic structured output (`QueryVariations`) to generate 3 alternate query formulations, querying the vector database from multiple semantic perspectives.
* **Reciprocal Rank Fusion (RRF) (`reciprocal_rank_fusion.py`)**: Combines and ranks documents retrieved across multiple query variations using the RRF scoring formula:
  $$RRF\_Score(d) = \sum_{q \in Q} \frac{1}{k + rank(d, q)} \quad (k=60)$$
* **Sparse + Dense Hybrid Search (`hybrid_search.ipynb`)**: Combines sparse keyword search (BM25) with dense vector search (ChromaDB) to capture exact keyword matches (e.g., financial numbers, model names) alongside semantic meanings.
* **Two-Stage Cohere Reranking (`reranker.ipynb`)**: Leverages `CohereRerank` (`rerank-english-v3.0`) to re-score candidate documents retrieved by BM25/Vector search, ensuring top context precision before passing to the LLM.

### 🔹 Phase 4: Answer Generation & Conversational RAG
* **Context-Grounded Answer Generation (`answer_generation.py`)**: Passes retrieved context to **Gemini 3.6 Flash** with system prompts enforcing zero hallucination (strictly answering from provided context).
* **History-Aware Conversational System (`history_aware_answer_generation.py`)**: Maintains multi-turn conversation memory. Uses an LLM agent to rewrite follow-up user questions into self-contained, standalone search queries before performing vector retrieval.

---

## 📊 Feature Comparison Matrix

| Module | Technique | Primary Advantage | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **Chunking** | Recursive Splitter | Fast, structure-aware | General text documents |
| **Chunking** | Semantic Chunking | Topic-coherent boundaries | Unstructured narrative text |
| **Chunking** | Agentic Chunking | Contextually optimal chunks | High-value, complex documents |
| **Retrieval** | Similarity Search | Baseline, minimal compute | Simple queries |
| **Retrieval** | MMR | High diversity, no duplicate context | Multi-topic queries |
| **Retrieval** | Multi-Query + RRF | High recall, overcomes poor user phrasing | Ambiguous queries |
| **Retrieval** | Hybrid (BM25 + Dense) | High precision & recall (Keywords + Vector) | Financial/technical datasets |
| **Ranking** | Cohere Reranker | Cross-encoder precision for top-N ranking | Production QA pipelines |

---

## 📁 Repository Structure

```
production-rag-architectures/
├── docs/                                  # Source knowledge documents (.txt, .pdf)
│   ├── Google.txt
│   ├── Microsoft.txt
│   ├── Nvidia.txt
│   ├── SpaceX.txt
│   └── Tesla.txt
│
├── db/                                    # Persisted ChromaDB vector database
│   └── chroma_db/
│
├── ingestion_pipeline.py                  # Document loading & ChromaDB vectorstore creation
├── recursive_char_text_splitter.py        # Character vs. Recursive Character chunking comparison
├── semantic_chunking.py                   # Embedding-distance-based semantic chunking
├── agentic_chunking.py                    # LLM-guided context-aware agentic chunking
│
├── retrieval_pipeline.py                  # Basic retriever setup and sample queries
├── retrieval_methods.py                   # Comparison of Similarity, Score Threshold, & MMR
├── multi_query_retrieval.py              # LLM query expansion via Pydantic structured output
├── reciprocal_rank_fusion.py              # Custom Reciprocal Rank Fusion (RRF) implementation
├── hybrid_search.ipynb                    # Sparse (BM25) + Dense (ChromaDB) Hybrid Search
├── reranker.ipynb                         # Two-stage retrieval with Cohere Cross-Encoder Reranker
│
├── answer_generation.py                   # Single-turn grounded Q&A with Gemini 3.6 Flash
├── history_aware_answer_generation.py     # Interactive multi-turn conversational RAG CLI
│
├── requirements.txt                       # Project dependencies
├── .env.example                           # Template for environment variables (No secrets exposed)
└── .gitignore                             # Ignored files (.env, venv, vector DBs)
```

---

## 🛠️ Setup & Installation

### Prerequisites
* Python 3.10+
* Virtual Environment manager (`venv` or `conda`)
* Google Gemini API Key
* Cohere API Key (optional, for reranking notebook)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/production-rag-architectures.git
cd production-rag-architectures
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory (refer to `.env.example` below). 

> ⚠️ **Security Notice**: Never commit `.env` or any secret keys to version control. Keep `.env` listed in `.gitignore`.

Create `.env`:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
CO_API_KEY=your_cohere_api_key_here
```

---

## 🚀 Quickstart & Execution Guide

### 1. Run Data Ingestion
Build the ChromaDB vector database from documents in `docs/`:
```bash
python ingestion_pipeline.py
```

### 2. Test Text Chunking Strategies
```bash
# Recursive character chunking
python recursive_char_text_splitter.py

# Semantic chunking
python semantic_chunking.py

# Agentic chunking via LLM
python agentic_chunking.py
```

### 3. Test Advanced Retrieval & Ranking
```bash
# Basic, Threshold, & MMR retrieval
python retrieval_methods.py

# Multi-query expansion
python multi_query_retrieval.py

# Reciprocal Rank Fusion (RRF)
python reciprocal_rank_fusion.py
```

### 4. Interactive Conversational RAG
Run the interactive history-aware CLI assistant:
```bash
python history_aware_answer_generation.py
```
*Example session:*
```
Welcome to question answer system
Type 'exit' to quit.

Ask a question: What was Tesla's revenue in Q3 2024?
Found 5 relevant documents...
Answer: Tesla reported record revenue of $25.2B in Q3 2024.

Ask a question: How does that compare to production costs?
--- Standalone rewritten question: How does Tesla's Q3 2024 revenue compare to its production costs? ---
...
```

---

## 🔑 Key Takeaways & Learnings

1. **Chunking is Critical**: Fixed-size chunking often splits key context across boundaries. Semantic and Agentic chunking significantly improve sentence-level topic preservation.
2. **Hybrid Search > Pure Vector Search**: Dense embeddings excel at conceptual queries, while sparse BM25 search is indispensable for specific keyword identifiers (model numbers, monetary figures, names).
3. **Reranking Elevates Precision**: Cross-encoder reranking (Cohere) acts as a high-precision filter before context injection, drastically reducing LLM hallucination and context window noise.
4. **Standalone Question Rewriting**: Multi-turn conversation history requires reformulating ambiguous follow-up questions (e.g., "How much did they pay?") into standalone queries ("How much did Microsoft pay for GitHub?") before vector search.

---

## 📜 License & Credits

Distributed under the MIT License. Built as part of an advanced AI Engineering and RAG learning journey using [LangChain](https://www.langchain.com/), [ChromaDB](https://www.trychroma.com/), [Google Gemini](https://ai.google.dev/), and [Cohere](https://cohere.com/).
