# Setup Guide

Complete instructions for running the RAG system locally or with Docker.

---

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized setup)
- OpenAI API key (only needed for answer generation, not retrieval)
- 8 GB RAM minimum

---

## Option 1 — Docker (Recommended)

The fastest way to run the full system.

### Step 1 — Clone the repository

```bash
git clone https://github.com/RohithkumarReddipogula/AI-Powered-Rag-System.git
cd AI-Powered-Rag-System
```

### Step 2 — Set your OpenAI API key

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

Skip this if you only want retrieval (not answer generation).

### Step 3 — Start the system

```bash
docker-compose up --build
```

Wait approximately 30 seconds for startup:
- Backend loads BM25 + FAISS indices (~5s)
- E5 model weights load (~10s)
- FastAPI starts (~2s)
- Streamlit starts (~5s)

### Step 4 — Access the system

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| API documentation | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

---

## Option 2 — Local Python

### Step 1 — Clone and install

```bash
git clone https://github.com/RohithkumarReddipogula/AI-Powered-Rag-System.git
cd AI-Powered-Rag-System

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2 — Start the backend

```bash
export OPENAI_API_KEY=sk-your-key-here   # Windows: set OPENAI_API_KEY=...

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3 — Start the frontend (new terminal)

```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

---

## Building the Indices

The pre-built indices are included for quick start. To rebuild from scratch:

### BM25 index

```bash
python scripts/build_bm25_index.py \
    --passages data/passages.jsonl \
    --output indices/bm25_index.pkl
```

Approximate time: 10 minutes for 8.8M passages.

### FAISS index (requires GPU recommended)

```bash
python scripts/build_faiss_index.py \
    --passages data/passages.jsonl \
    --output indices/faiss_index.bin \
    --model intfloat/e5-base-v2
```

Approximate time: 8 hours on NVIDIA A100 (300 passages/second). 2-3 days on CPU.

---

## Downloading MS MARCO Data

```bash
# Download the passage collection (~3 GB)
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
tar -xzf collection.tar.gz -C data/raw/

# Preprocess
python data/preprocessing.py \
    --input data/raw/collection.tsv \
    --output data/passages.jsonl
```

---

## Running the Evaluation

```bash
# Run alpha grid search
python scripts/parameter_optimization.py \
    --queries data/eval_queries.json \
    --output results/alpha_sweep.csv \
    --k 10

# Run unit tests
pytest tests/ -v --cov=app
```

---

## Troubleshooting

**"Index not found" error on startup**
The pre-built indices must be in the `indices/` folder. Download them from the releases page or build them with the scripts above.

**High memory usage**
The system uses ~4 GB RAM at runtime (indices + E5 model). Ensure your machine has at least 8 GB available.

**Slow first query**
The first query after startup may be slower (~2-3x) as the CPU cache warms up. Subsequent queries run at the reported 710ms average.

**Docker build takes too long**
The E5 model weights (~430 MB) are downloaded during the build. This is a one-time cost — subsequent builds use the Docker layer cache.
