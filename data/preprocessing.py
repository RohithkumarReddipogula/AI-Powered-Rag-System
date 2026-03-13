"""
MS MARCO Passage Preprocessing Pipeline
=========================================
Cleans and normalizes 8.84M raw web passages for indexing.

Steps
-----
1. HTML tag removal    (web extraction artifacts)
2. Whitespace normalization
3. Unicode NFKC normalization  (ligatures, special chars → ASCII)
4. Punctuation standardization (em-dash, smart quotes → ASCII equivalents)
5. Exact deduplication         (reduces 8.84M → 8.80M passages)

Output
------
JSONL file at data/passages.jsonl  (one {"id": int, "text": str} per line)

Usage
-----
    python data/preprocessing.py \
        --input data/raw/collection.tsv \
        --output data/passages.jsonl
"""

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Generator, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Cleaning functions ───────────────────────────────────────────────────────

def remove_html_tags(text: str) -> str:
    """Remove HTML tags while preserving text content."""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace sequences to single space, strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode(text: str) -> str:
    """NFKC normalization: converts ligatures, special chars to base forms."""
    return unicodedata.normalize("NFKC", text)


def normalize_punctuation(text: str) -> str:
    """Replace typographic punctuation with ASCII equivalents."""
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00a0": " ",   # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def clean_passage(text: str) -> Optional[str]:
    """
    Apply the full cleaning pipeline to a single passage.
    Returns None if the passage is too short after cleaning.
    """
    text = remove_html_tags(text)
    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)

    if len(text.split()) < 5:  # Discard near-empty passages
        return None
    return text


# ─── I/O ─────────────────────────────────────────────────────────────────────

def iter_tsv_passages(path: str) -> Generator:
    """
    Read MS MARCO collection.tsv  (tab-separated: passage_id \t passage_text).
    Yields (passage_id, raw_text) tuples.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                yield parts[0], parts[1]


def run_preprocessing(input_path: str, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    n_total = n_kept = n_duplicate = n_short = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for raw_id, raw_text in iter_tsv_passages(input_path):
            n_total += 1

            cleaned = clean_passage(raw_text)

            if cleaned is None:
                n_short += 1
                continue

            # Exact deduplication on cleaned text
            if cleaned in seen:
                n_duplicate += 1
                continue
            seen.add(cleaned)

            record = {"id": n_kept, "original_id": raw_id, "text": cleaned}
            out_f.write(json.dumps(record) + "\n")
            n_kept += 1

            if n_total % 500_000 == 0:
                logger.info(
                    f"Processed {n_total:,} passages | "
                    f"kept {n_kept:,} | "
                    f"dropped (short) {n_short:,} | "
                    f"dropped (dupe) {n_duplicate:,}"
                )

    logger.info(
        f"\nPreprocessing complete:\n"
        f"  Input:      {n_total:,} passages\n"
        f"  Output:     {n_kept:,} passages\n"
        f"  Duplicates: {n_duplicate:,}\n"
        f"  Too short:  {n_short:,}\n"
        f"  Saved to:   {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MS MARCO passage collection")
    parser.add_argument("--input", required=True, help="Path to raw collection.tsv")
    parser.add_argument("--output", default="data/passages.jsonl", help="Output JSONL path")
    args = parser.parse_args()
    run_preprocessing(args.input, args.output)
