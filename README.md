<div align="center">

<h1>AI-Powered RAG System</h1>
<h3>Production-Grade Hybrid Retrieval-Augmented Generation</h3>

<p>
  <a href="https://github.com/RohithkumarReddipogula/AI-Powered-Rag-System/actions">
  <img src="https://github.com/RohithkumarReddipogula/AI-Powered-Rag-System/actions/workflows/ci.yml/badge.svg" alt="CI">
</a>
  <a href="https://www.python.org/downloads/release/python-3110/">
    <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://fastapi.tiangolo.com">
    <img src="https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  </a>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <a href="https://rohith2026-hybrid-rag-demo.hf.space">
    <img src="https://img.shields.io/badge/Live_Demo-online-brightgreen?logo=huggingface" alt="Demo">
  </a>
  <a href="https://rohith2026-hybrid-rag-api.hf.space/docs">
    <img src="https://img.shields.io/badge/API_Docs-online-blue?logo=fastapi" alt="API">
  </a>
</p>

<p>
  <a href="https://rohith2026-hybrid-rag-demo.hf.space"><strong> Live Demo</strong></a> ·
  <a href="https://rohith2026-hybrid-rag-api.hf.space/docs"><strong> API Docs</strong></a> ·
  <a href="#-quick-start"><strong> Quick Start</strong></a> ·
  <a href="RESULTS.md"><strong> Full Results</strong></a> ·
  <a href="https://arxiv.org/abs/2507.18910"><strong> Referenced Paper</strong></a>
</p>

<br>

<!-- Key metrics banner -->
<table>
  <tr>
    <td align="center"><b>93.0%</b><br><sub>Recall@10</sub></td>
    <td align="center"><b>1.0</b><br><sub>MRR (perfect)</sub></td>
    <td align="center"><b>710ms</b><br><sub>Avg Latency</sub></td>
    <td align="center"><b>4.46 MB</b><br><sub>Index Size</sub></td>
    <td align="center"><b>45%</b><br><sub>Cost Reduction</sub></td>
    <td align="center"><b>8.84M</b><br><sub>Passages (MS MARCO)</sub></td>
  </tr>
</table>

</div>

---

## What Is This?

This is the implementation of my **MSc Data Science thesis** (University of Europe for Applied Sciences, 2026). It builds a **production-ready Retrieval-Augmented Generation (RAG) system** that combines two complementary retrieval approaches:

- **BM25** (sparse retrieval) — precise lexical matching, fast, low storage
- **Microsoft E5** (dense retrieval) — semantic similarity, handles vocabulary mismatch

The **central finding**: the optimal fusion weight is **α = 0.70** (70% dense, 30% sparse), not the default α = 0.5 used by LangChain, LlamaIndex, and virtually all published hybrid systems. This is **statistically significant** (p = 0.002) and yields a 2.9 percentage-point improvement in Recall@10.

This work directly addresses six research gaps identified in the RAG survey [Oche et al., 2025 · arXiv:2507.18910](https://arxiv.org/abs/2507.18910).

---

## Key Results

<details open>
<summary><b>Retrieval Performance (click to expand)</b></summary>

| α | Method | Recall@10 | MRR | vs. Baseline |
|---|--------|-----------|-----|-------------|
| 0.0 | BM25 only | 81.6% | 0.78 | — |
| 0.5 | Hybrid (default) | 90.1% | 0.96 | +10.4% |
| **0.7** | **Hybrid (optimal)** | **93.0%** | **1.0** | **+13.9% ★** |
| 1.0 | E5 only | 82.5% | 0.85 | +1.1% |

α = 0.70 vs α = 0.50: **t(99) = 3.14, p = 0.002** 

</details>

<details>
<summary><b>System Efficiency</b></summary>

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Query latency | 84,000 ms | **710 ms** | **99.2% reduction** |
| Index storage | ~27 GB | **4.46 MB** | **99.8% reduction** |
| LLM API cost/month | $4,500 | **$2,475** | **45% reduction** |
| Container startup | ~120s | **<30s** | via pre-built indices |

</details>

<details>
<summary><b>Ablation Study</b></summary>

Each row adds one component — showing which change drives which improvement:

| Configuration | Recall@10 | Latency | Cost/day |
|---|---|---|---|
| Baseline (E5 only, naive) | 82.5% | 84,000ms | $150 |
| + Hybrid α=0.70 | **93.0%** | 84,035ms | $150 |
| + Persistent indices | 93.0% | 1,150ms | $150 |
| + Score normalization | 93.0% | 1,050ms | $150 |
| + FAISS/BM25 optimization | 93.0% | 710ms | $150 |
| + Multi-level caching | 93.0% | 710ms | **$82.50** |

Quality gains come from hybrid retrieval. Efficiency gains come from engineering.

</details>

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OFFLINE (Index Build)                       │
│                                                                 │
│  MS MARCO 8.84M Passages                                        │
│       │                                                         │
│       ├──► BM25 Tokenize ──► Inverted Index ──► bm25.pkl 1.41MB│
│       │                                                         │
│       └──► E5 Encode (8h, A100 GPU) ──► FAISS Index ──► 3.05MB │
└────────────────────────────────────────┬────────────────────────┘
                                         │ Load once at startup (~5s)
┌────────────────────────────────────────▼────────────────────────┐
│                     ONLINE (Query Time)                         │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ├── [Stage 1] BM25 Lookup ──────────────────── 380ms      │
│      │                                                          │
│      ├── [Stage 2] E5 Encode query ──────────────── 75ms       │
│      │       ↓ (cached for repeat queries)                      │
│      ├── [Stage 3] FAISS ANN Search ─────────────── 220ms      │
│      │                                                          │
│      └── [Stage 4] Min-Max Normalize + Fuse (α=0.70) ─ 35ms   │
│                   score = 0.70×E5_norm + 0.30×BM25_norm        │
│                                  │                              │
│                          Top-k Passages                         │
│                                  │                              │
│                        [Optional] GPT-4 ──► Grounded Answer    │
│                                                                 │
│  Total: 710ms average                                           │
└─────────────────────────────────────────────────────────────────┘
```

**Three-tier deployment:**
```
Streamlit UI  ──HTTP──►  FastAPI Backend  ──reads──►  FAISS + BM25 Indices
(port 8501)               (port 8000)                  (4.46 MB total)
```

---

## Quick Start

### Option 1: Docker (recommended — runs in under 30 seconds)

```bash
git clone https://github.com/rohithkumarreddipogula/ai-powered-rag-system.git
cd ai-powered-rag-system

# Add your OpenAI key (only needed for answer generation)
echo "OPENAI_API_KEY=sk-..." > .env

docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

### Option 2: Local Python

```bash
git clone https://github.com/rohithkumarreddipogula/ai-powered-rag-system.git
cd ai-powered-rag-system

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (new terminal)
streamlit run frontend/app.py
```

### Option 3: Try the live demo

👉 **[https://sage-bunny-d1e261.netlify.app](https://sage-bunny-d1e261.netlify.app)**

---

## API Reference

### `POST /search` — Hybrid Retrieval

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

<details>
<summary>Response example</summary>

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

</details>

### `GET /health` — System Monitoring

```bash
curl http://localhost:8000/health
```

Returns: uptime, avg latency, cache hit rates per layer, index sizes, queries processed.

---

## Reproducing the Experiments

### Run the alpha grid search

```bash
# Evaluate all α values on your query set
python scripts/parameter_optimization.py \
    --queries data/eval_queries.json \
    --output results/alpha_sweep.csv \
    --k 10
```

Expected output:
```
α=0.0  (sparse) → Recall@10=0.816, MRR=0.780
α=0.5  (hybrid) → Recall@10=0.901, MRR=0.960
α=0.7  (hybrid) → Recall@10=0.930, MRR=1.000  ← OPTIMAL
α=1.0  (dense)  → Recall@10=0.825, MRR=0.850
```

### Run unit tests

```bash
pytest tests/ -v --cov=app
```

---

## Project Structure

```
ai-powered-rag-system/
│
├── app/                          # Backend application
│   ├── main.py                   # FastAPI app: /search, /health endpoints
│   ├── config.py                 # Central config (α, k, cache sizes, paths)
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py     # BM25 sparse retrieval (Robertson & Zaragoza, 2009)
│   │   ├── dense_retriever.py    # E5 + FAISS dense retrieval (Wang et al., 2024)
│   │   ├── hybrid_retriever.py   # ★ Core: weighted score fusion, α optimization
│   │   ├── normalizer.py         # Min-max + z-score normalization
│   │   └── cache.py              # 3-level LRU cache (retrieval/embedding/generation)
│   │
│   ├── generation/
│   │   └── generator.py          # GPT-4 answer generation with prompt engineering
│   │
│   └── evaluation/
│       └── metrics.py            # Recall@k, MRR, alpha_grid_search()
│
├── frontend/
│   └── app.py                    # Streamlit UI (method selector, α slider, results)
│
├── data/
│   └── preprocessing.py          # MS MARCO cleaning pipeline (HTML→Unicode→dedup)
│
├── scripts/
│   └── parameter_optimization.py # Grid search + paired t-test for statistical significance
│
├── tests/
│   └── test_retrieval.py         # Unit tests: normalizer, cache, metrics, fusion logic
│
├── indices/                      # Pre-built indices (download via setup script)
│   ├── bm25_index.pkl            # 1.41 MB compressed BM25 inverted index
│   └── faiss_index.bin           # 3.05 MB quantized FAISS index
│
├── RESULTS.md                    # Full experimental results tables
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements.txt
└── pyproject.toml
```

---

## Implementation Details

### Why α = 0.70?

The fusion formula is:

```
score_hybrid(d) = α × score_E5_norm(d) + (1 - α) × score_BM25_norm(d)
```

Before fusion, both score sets are **min-max normalized** to [0, 1]:
```
score_norm = (score - min_score) / (max_score - min_score)
```

This is necessary because BM25 scores are unbounded (0–50+, query-length dependent) while E5 cosine similarities are in [-1, 1]. Direct combination without normalization would be meaningless.

At α = 0.70, the system weights semantic matching 70% and exact-term matching 30%. This reflects the MS MARCO query distribution: most are natural-language questions (favoring dense retrieval), but a significant minority require lexical precision (keeping sparse retrieval useful).

### Three-Level Caching

```
Query arrives
    │
    ├── Retrieval cache hit? ──► Return cached results immediately
    │         (key: "hybrid:0.7:10:<query_text>")
    │
    ├── Embedding cache hit? ──► Skip E5 inference (saves 75ms)
    │         (key: query_text → np.ndarray)
    │
    └── Generation cache hit? ──► Skip LLM API call (saves $0.15)
              (key: SHA256(query + passages))
```

Under Zipfian query distribution (realistic search traffic), this achieves **40–45% overall hit rate**, directly translating to the same reduction in LLM API costs.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `rank-bm25` | 0.2.2 | BM25 index construction and retrieval |
| `faiss-cpu` | 1.7.4 | Dense vector indexing and ANN search |
| `sentence-transformers` | 2.2.2 | E5 embedding generation |
| `fastapi` | 0.104.1 | REST API with auto-generated OpenAPI docs |
| `uvicorn` | 0.24.0 | ASGI server |
| `streamlit` | 1.28.0 | Interactive web interface |
| `openai` | 1.3.0 | GPT-4 answer generation |
| `scipy` | ≥1.11 | Paired t-test for statistical significance |

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

1. Robertson, S. & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
2. Wang et al. (2024). Improving text embeddings with large language models. *arXiv:2401.00368*.
3. Lewis et al. (2020). Retrieval-Augmented Generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.
4. Johnson et al. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*.
5. Nguyen et al. (2016). MS MARCO: A human generated machine reading comprehension dataset. *arXiv:1611.09268*.
6. Oche et al. (2025). A systematic review of key RAG systems. *arXiv:2507.18910*.

---

<div align="center">

Built by **Rohith Kumar Reddipogula**
MSc Data Science · University of Europe for Applied Sciences, Potsdam · 2026

[LinkedIn](https://linkedin.com/in/rohithkumarreddipogula) · [Demo](https://sage-bunny-d1e261.netlify.app) · [Email](mailto:rohith.reddipogula@gmail.com)

</div>
