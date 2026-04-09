import argparse
import json
import random
import sqlite3
import time
from pathlib import Path

import numpy as np

from src.embedder import get_embedder

DB_PATH = Path("data/processed/chunks.db")
OUTPUT_PATH = Path("kb_embeddings.json")
BATCH_SIZE = 32
PROGRESS_EVERY = 10


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]
    return [f"{part}." for part in parts]


def build_chunk_text(abstract: str) -> str:
    sentences = split_sentences(abstract)
    if not sentences:
        return abstract.strip()
    return " ".join(sentences[:2]).strip()


def normalize_source(source: str | None, chunk_type: str | None) -> str:
    if source:
        return source
    if chunk_type == "abstract":
        return "paper"
    if chunk_type and "wiki" in chunk_type:
        return "wikipedia"
    return "unknown"


def infer_domain(title: str, abstract: str, source: str) -> str:
    text = f"{title} {abstract}".lower()

    if any(term in text for term in ["diffusion", "gan", "image synthesis", "text-to-image", "imagen", "controlnet"]):
        return "vision"
    if any(term in text for term in ["language model", "large language model", "transformer", "attention"]):
        return "language"
    if source == "wikipedia":
        return "reference"
    return "general"


def load_documents() -> list[dict]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} not found.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, title, chunk, source, chunk_type
        FROM chunks
        ORDER BY id
        """
    )
    rows = cursor.fetchall()
    conn.close()

    documents = []
    for row in rows:
        title = (row["title"] or "").strip()
        abstract = (row["chunk"] or "").strip()
        chunk_type = (row["chunk_type"] or "").strip()
        source = normalize_source((row["source"] or "").strip(), chunk_type)

        documents.append(
            {
                "id": str(row["id"]),
                "title": title,
                "abstract": abstract,
                "source": source,
                "domain": infer_domain(title, abstract, source),
                "chunk_type": chunk_type or "abstract",
            }
        )

    return documents


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_documents(documents: list[dict]) -> list[dict]:
    embedder = get_embedder()
    output = []
    start = time.time()

    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch = documents[batch_start:batch_start + BATCH_SIZE]

        full_inputs = [f"{doc['title']} [SEP] {doc['abstract']}" for doc in batch]
        title_inputs = [doc["title"] for doc in batch]
        chunk_inputs = [build_chunk_text(doc["abstract"]) for doc in batch]

        full_embeddings = embedder.embed(full_inputs)
        title_embeddings = embedder.embed(title_inputs)
        chunk_embeddings = embedder.embed(chunk_inputs)

        for index, doc in enumerate(batch):
            output.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "abstract": doc["abstract"],
                    "source": doc["source"],
                    "domain": doc["domain"],
                    "chunk_type": doc["chunk_type"],
                    "embedding": full_embeddings[index],
                    "title_embedding": title_embeddings[index],
                    "chunk_embedding": chunk_embeddings[index],
                }
            )

            processed = len(output)
            if processed % PROGRESS_EVERY == 0:
                elapsed = time.time() - start
                print(f"Processed {processed} documents in {elapsed:.1f}s")

    return output


def verify_embeddings(records: list[dict]) -> bool:
    if len(records) < 3:
        sample = records
    else:
        sample = random.sample(records, 3)

    embedder = get_embedder()
    passed = True

    for record in sample:
        text = f"{record['title']} [SEP] {record['abstract']}"
        fresh = np.asarray(embedder.embed_single(text), dtype=np.float32)
        stored = np.asarray(record["embedding"], dtype=np.float32)
        score = cosine_similarity(fresh, stored)
        print(f"Verify id={record['id']}: cosine={score:.6f}")
        if score < 0.999:
            passed = False

    print("PASS" if passed else "FAIL")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    started = time.time()
    documents = load_documents()
    print(f"Loaded {len(documents)} local KB documents from {DB_PATH}")

    records = embed_documents(documents)

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle)

    elapsed = time.time() - started
    print(f"Saved {len(records)} documents to {OUTPUT_PATH} in {elapsed:.1f}s")

    if args.verify:
        verify_embeddings(records)


if __name__ == "__main__":
    main()
