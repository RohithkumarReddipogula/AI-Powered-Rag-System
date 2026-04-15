# AI-Powered RAG System

[![CI](https://github.com/RohithkumarReddipogula/AI-Powered-Rag-System/actions/workflows/ci.yml/badge.svg)](https://github.com/RohithkumarReddipogula/AI-Powered-Rag-System/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-online-brightgreen?logo=huggingface)](https://rohith2026-hybrid-rag-demo.hf.space)
[![API Docs](https://img.shields.io/badge/API_Docs-online-blue?logo=fastapi)](https://rohith2026-hybrid-rag-api.hf.space/docs)
[![Setup Guide](https://img.shields.io/badge/Setup_Guide-Netlify-00C7B7?logo=netlify&logoColor=white)](https://sage-bunny-d1e261.netlify.app)

A **production-grade Hybrid Retrieval-Augmented Generation system** combining BM25 sparse retrieval with Microsoft E5 dense embeddings, evaluated on 8.84M MS MARCO passages. Built as my MSc Data Science thesis at the University of Europe for Applied Sciences, Potsdam (2026).

---

## Table of Contents

1. [Introduction](#introduction)
2. [Key Results](#key-results)
3. [Tech Stack](#tech-stack)
4. [System Architecture](#system-architecture)
5. [Getting Started](#getting-started)

   * [Prerequisites](#prerequisites)
   * [Clone the Repo](#clone-the-repo)
   * [Docker Setup](#docker-setup)
   * [Local Python Setup](#local-python-setup)
   * [Live Demo](#live-demo)
6. [API Endpoints](#api-endpoints)
7. [Reproducing the Experiments](#reproducing-the-experiments)
8. [Project Structure](#project-structure)
9. [Implementation Details](#implementation-details)
10. [Dependencies](#dependencies)
11. [Citation](#citation)
12. [References](#references)

---

## Introduction

This project implements a hybrid retrieval pipeline that fuses BM25 lexical matching with E5 semantic embeddings using a weighted score fusion formula. The central finding of the thesis is that the optimal fusion weight is **alpha = 0.70** (70% dense, 30% sparse), not the default alpha = 0.5 used by LangChain, LlamaIndex, and most published hybrid systems.

This result is statistically significant (t(99) = 3.14, p = 0.002) and yields a 2.9 percentage-point improvement in Recall@10 over the commonly used default. The system also reduces query latency from 84,000ms to 710ms and compresses index storage from ~27GB to 4.46MB through FAISS quantization and persistent index caching.

This work addresses six research gaps identified in the RAG survey by Oche et al. (2025) at https://arxiv.org/abs/2507.18910.

---

## Key Results

**Retrieval Performance**

| Alpha | Method | Recall@10 | MRR | vs. Baseline |
|-------|--------|-----------|-----|--------------|
| 0.0 | BM25 only | 81.6% | 0.78 | — |
| 0.5 | Hybrid (default) | 90.1% | 0.96 | +10.4% |
| **0.7** | **Hybrid (optimal)** | **93.0%** | **1.0** | **+13.9%** |
| 1.0 | E5 only | 82.5% | 0.85 | +1.1% |

**System Efficiency**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Query latency | 84,000ms | 710ms | 99.2% reduction |
| Index storage | ~27 GB | 4.46 MB | 99.8% reduction |
| LLM API cost/month | $4,500 | $2,475 | 45% reduction |
| Container startup | ~120s | under 30s | via pre-built indices |

**Ablation Study**

| Configuration | Recall@10 | Latency | Cost/day |
|---------------|-----------|---------|----------|
| Baseline (E5 only, naive) | 82.5% | 84,000ms | $150 |
| + Hybrid alpha=0.70 | 93.0% | 84,035ms | $150 |
| + Persistent indices | 93.0% | 1,150ms | $150 |
| + Score normalization | 93.0% | 1,050ms | $150 |
| + FAISS/BM25 optimization | 93.0% | 710ms | $150 |
| + Multi-level caching | 93.0% | 710ms | $82.50 |

---

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Retrieval**: BM25 (rank-bm25), FAISS (faiss-cpu), Microsoft E5 (sentence-transformers)
- **Answer Generation**: OpenAI GPT-4
- **Containerization**: Docker, Docker Compose
- **Evaluation**: SciPy (paired t-test), custom Recall@k and MRR metrics
- **Dataset**: MS MARCO (8.84M passages)

---

## System Architecture

```
+------------------------------------------------------------------+
|                     OFFLINE (Index Build)                        |
|                                                                  |
|  MS MARCO 8.84M Passages                                         |
|       |                                                          |
|       +--> BM25 Tokenize --> Inverted Index --> bm25.pkl 1.41MB  |
|       |                                                          |
|       +--> E5 Encode (8h, A100 GPU) --> FAISS Index --> 3.05MB  |
+----------------------------------------+-------------------------+
                                         | Load once at startup (~5s)
+----------------------------------------+-------------------------+
|                     ONLINE (Query Time)                          |
|                                                                  |
|  User Query                                                      |
|      |                                                           |
|      +--> [Stage 1] BM25 Lookup ------------------------- 380ms |
|      |                                                           |
|      +--> [Stage 2] E5 Encode query -------------------- 75ms  |
|      |       (cached for repeat queries)                         |
|      +--> [Stage 3] FAISS ANN Search ------------------- 220ms |
|      |                                                           |
|      +--> [Stage 4] Min-Max Normalize + Fuse (alpha=0.70) - 35ms|
|                   score = 0.70 x E5_norm + 0.30 x BM25_norm    |
|                                  |                               |
|                          Top-k Passages                          |
|                                  |                               |
|                        [Optional] GPT-4 --> Grounded Answer     |
|                                                                  |
|  Total: 710ms average                                            |
+------------------------------------------------------------------+
```

Three-tier deployment:

```
Streamlit UI  --HTTP-->  FastAPI Backend  --reads-->  FAISS + BM25 Indices
(port 8501)              (port 8000)                  (4.46 MB total)
```

---

## Getting Started

### Prerequisites

- **Python** >= 3.11
- **Docker** and **Docker Compose** (recommended)
- **OpenAI API key** (only required for answer generation, not for retrieval)

### Clone the Repo

```bash
git clone https://github.com/rohithkumarreddipogula/ai-powered-rag-system.git
cd ai-powered-rag-system
```

### Docker Setup

This is the recommended way to run the system. It starts both services in under 30 seconds.

```bash
echo "OPENAI_API_KEY=sk-..." > .env
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

### Local Python Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start the backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Start the frontend (new terminal)
streamlit run frontend/app.py
```

### Live Demo

HuggingFace Spaces may take around 30 seconds to load on first visit.

- Interactive UI: https://rohith2026-hybrid-rag-demo.hf.space
- API documentation: https://rohith2026-hybrid-rag-api.hf.space/docs
- Setup guide: https://sage-bunny-d1e261.netlify.app

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search` | Hybrid retrieval with configurable alpha and top-k |
| `GET` | `/health` | System status, latency, cache hit rates, index sizes |

### POST /search — example request

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is machine learning?",
    "method": "hybrid",
    "alpha": 0.7,
    "k": 5,
    "generate_answer": false
  }'
```

Example response:

```json
{
  "query": "what is machine learning?",
  "method": "hybrid",
  "alpha": 0.7,
  "k": 5,
  "latency_ms": 712.4,
  "results": [
    {
      "doc_id": 3245781,
      "rank": 1,
      "score": 0.9312,
      "bm25_score": 0.8421,
      "dense_score": 0.9714,
      "text": "Machine learning is a branch of artificial intelligence..."
    }
  ],
  "answer": null
}
```

### GET /health — example request

```bash
curl http://localhost:8000/health
```

Returns uptime, average latency, cache hit rates per layer, index sizes, and total queries processed.

---

## Reproducing the Experiments

### Run the alpha grid search

```bash
python scripts/parameter_optimization.py \
    --queries data/eval_queries.json \
    --output results/alpha_sweep.csv \
    --k 10
```

Expected output:

```
alpha=0.0  (sparse) -> Recall@10=0.816, MRR=0.780
alpha=0.5  (hybrid) -> Recall@10=0.901, MRR=0.960
alpha=0.7  (hybrid) -> Recall@10=0.930, MRR=1.000  <- OPTIMAL
alpha=1.0  (dense)  -> Recall@10=0.825, MRR=0.850
```

### Run unit tests

```bash
pytest tests/ -v --cov=app
```

---

## Project Structure

```
ai-powered-rag-system/
|
+-- app/
|   +-- main.py                   # FastAPI app: /search, /health endpoints
|   +-- config.py                 # Central config (alpha, k, cache sizes, paths)
|   |
|   +-- retrieval/
|   |   +-- bm25_retriever.py     # BM25 sparse retrieval
|   |   +-- dense_retriever.py    # E5 + FAISS dense retrieval
|   |   +-- hybrid_retriever.py   # Core: weighted score fusion, alpha optimization
|   |   +-- normalizer.py         # Min-max + z-score normalization
|   |   +-- cache.py              # 3-level LRU cache
|   |
|   +-- generation/
|   |   +-- generator.py          # GPT-4 answer generation with prompt engineering
|   |
|   +-- evaluation/
|       +-- metrics.py            # Recall@k, MRR, alpha_grid_search()
|
+-- frontend/
|   +-- app.py                    # Streamlit UI
|
+-- data/
|   +-- preprocessing.py          # MS MARCO cleaning pipeline
|
+-- scripts/
|   +-- parameter_optimization.py # Grid search + paired t-test
|
+-- tests/
|   +-- test_retrieval.py         # Unit tests
|
+-- indices/
|   +-- bm25_index.pkl            # 1.41 MB BM25 inverted index
|   +-- faiss_index.bin           # 3.05 MB quantized FAISS index
|
+-- RESULTS.md
+-- docker-compose.yml
+-- Dockerfile.backend
+-- Dockerfile.frontend
+-- requirements.txt
+-- pyproject.toml
```

---

## Implementation Details

### Why alpha = 0.70?

The fusion formula is:

```
score_hybrid(d) = alpha x score_E5_norm(d) + (1 - alpha) x score_BM25_norm(d)
```

Before fusion, both score sets are min-max normalized to [0, 1]:

```
score_norm = (score - min_score) / (max_score - min_score)
```

This step is necessary because BM25 scores are unbounded (0 to 50+, query-length dependent) while E5 cosine similarities are in [-1, 1]. Combining them directly without normalization produces meaningless results.

At alpha = 0.70, the system weights semantic matching at 70% and exact-term matching at 30%. This reflects the MS MARCO query distribution: most queries are natural-language questions that favor dense retrieval, while a meaningful minority require lexical precision where sparse retrieval remains useful.

### Three-Level Caching

```
Query arrives
    |
    +--> Retrieval cache hit? --> Return cached results immediately
    |        (key: "hybrid:0.7:10:<query_text>")
    |
    +--> Embedding cache hit? --> Skip E5 inference (saves 75ms)
    |        (key: query_text -> np.ndarray)
    |
    +--> Generation cache hit? --> Skip LLM API call (saves $0.15)
             (key: SHA256(query + passages))
```

Under a Zipfian query distribution (realistic search traffic), this achieves a 40 to 45% overall cache hit rate, which directly translates to the same reduction in LLM API costs.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| rank-bm25 | 0.2.2 | BM25 index construction and retrieval |
| faiss-cpu | 1.7.4 | Dense vector indexing and ANN search |
| sentence-transformers | 2.2.2 | E5 embedding generation |
| fastapi | 0.104.1 | REST API with auto-generated OpenAPI docs |
| uvicorn | 0.24.0 | ASGI server |
| streamlit | 1.28.0 | Interactive web interface |
| openai | 1.3.0 | GPT-4 answer generation |
| scipy | 1.11+ | Paired t-test for statistical significance |

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{reddipogula2026rag,
  author  = {Rohith Kumar Reddipogula},
  title   = {AI-Powered Retrieval-Augmented Assistant for Evidence-Based Question Answering},
  school  = {University of Europe for Applied Sciences},
  year    = {2026},
  month   = {February},
  address = {Potsdam, Germany}
}
```

This work addresses gaps identified in:

```bibtex
@article{oche2025rag,
  author  = {Oche, Agada Joseph and Folashade, Ademola Glory and Ghosal, Tirthankar and Biswas, Arpan},
  title   = {A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions},
  journal = {arXiv preprint arXiv:2507.18910},
  year    = {2025}
}
```

---

## References

1. Robertson, S. and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4), 333-389.
2. Wang et al. (2024). Improving text embeddings with large language models. arXiv:2401.00368.
3. Lewis et al. (2020). Retrieval-Augmented Generation for knowledge-intensive NLP tasks. NeurIPS 2020.
4. Johnson et al. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data.
5. Nguyen et al. (2016). MS MARCO: A human generated machine reading comprehension dataset. arXiv:1611.09268.
6. Oche et al. (2025). A systematic review of key RAG systems. arXiv:2507.18910.

---

Built by Rohith Kumar Reddipogula — MSc Data Science, University of Europe for Applied Sciences, Potsdam, 2026.

Open to ML Engineer, NLP Engineer, and AI Engineer roles in Berlin — [LinkedIn](https://www.linkedin.com/in/rohith-kumar-reddipogula-a6692030b/) · [Email](mailto:rohith.reddipogula@gmail.com)
