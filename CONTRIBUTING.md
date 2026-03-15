# Contributing

Thank you for your interest in contributing to the AI-Powered RAG System.

## Ways to Contribute

- **Bug reports** — open an issue with steps to reproduce
- **Performance improvements** — especially BM25 parallelization (current bottleneck at 380ms)
- **New retrieval methods** — ColBERT, SPLADE, or other hybrid approaches
- **Dataset evaluation** — testing on BEIR benchmark or domain-specific corpora
- **Documentation** — improving setup guides or API documentation

## Development Setup

```bash
git clone https://github.com/RohithkumarReddipogula/AI-Powered-Rag-System.git
cd AI-Powered-Rag-System

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

All tests must pass before submitting a pull request.

## Code Style

This project uses `ruff` for linting:

```bash
pip install ruff
ruff check app/ tests/
```

## Pull Request Process

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear, descriptive commits
4. Run all tests: `pytest tests/ -v`
5. Run linting: `ruff check app/ tests/`
6. Submit a pull request with a clear description of what changed and why

## Key Architecture Decisions

Before proposing changes, please read:

- **Why α=0.70?** The fusion parameter was determined via systematic grid search (see `scripts/parameter_optimization.py`). Any change to this value requires re-running the evaluation on MS MARCO.
- **Why min-max normalization?** BM25 scores are unbounded; E5 scores are cosine similarities in [-1,1]. Direct combination is meaningless without normalization. See `app/retrieval/normalizer.py`.
- **Why three cache levels?** Each layer targets a different cost bottleneck. See `app/retrieval/cache.py` for the design rationale.

## Reporting Issues

When reporting a bug, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
