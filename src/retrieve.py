import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================

DB_PATH = "data/processed/chunks.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---- LOCKED THRESHOLDS ----
SUPPORTED_THRESHOLD = 0.65
PARTIAL_THRESHOLD   = 0.50

# ---- WEIGHTS ----
SOURCE_WEIGHTS = {
    "paper": 1.0,
    "wikipedia": 0.7,
}

CHUNK_TYPE_WEIGHTS = {
    "abstract": 1.0,
    "pdf": 0.85,
    "wiki": 0.7,
}

ALPHA = 0.6   # cosine similarity
BETA  = 0.2   # source reliability
GAMMA = 0.2   # chunk reliability

ABSOLUTE_TERMS = [
    "always", "never", "completely", "entirely",
    "guarantees", "perfect", "fails", "impossible"
]

# ================= HELPERS =================

def overstatement_penalty(claim: str) -> float:
    claim = claim.lower()
    return 0.05 * sum(term in claim for term in ABSOLUTE_TERMS)

# ================= RETRIEVER =================

class Retriever:
    def __init__(self):
        print("🔄 Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print("📚 Loading corpus from DB...")
        self.texts, self.meta = self._load_corpus()
        print(f"📚 Loaded {len(self.texts)} corpus chunks")

        print("🔄 Encoding corpus...")
        self.embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        print("⚡ Building FAISS index...")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print("✅ Retriever (FAISS) ready")

    def _load_corpus(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            SELECT title, chunk, source, chunk_type
            FROM chunks
        """)
        rows = c.fetchall()
        conn.close()

        texts = []
        meta = []

        for title, chunk, source, chunk_type in rows:
            texts.append(chunk)
            meta.append({
                "title": title,
                "source": source,
                "chunk_type": chunk_type,
                "text": chunk
            })

        return texts, meta

    # ---------- WEIGHTED SCORE ----------
    def _weighted_score(self, sim, source, chunk_type, claim):
        source_w = SOURCE_WEIGHTS.get(source, 0.5)
        chunk_w  = CHUNK_TYPE_WEIGHTS.get(chunk_type, 0.7)
        penalty  = overstatement_penalty(claim)

        return (
            ALPHA * sim +
            BETA  * source_w +
            GAMMA * chunk_w -
            penalty
        )

    # ---------- SEARCH ----------
    def search(self, query, top_k=3, mode="full"):
        query_emb = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        # FAISS search
        sims, idxs = self.index.search(query_emb.reshape(1, -1), top_k * 4)

        scored = []
        for sim, i in zip(sims[0], idxs[0]):
            meta = self.meta[i]

            if mode == "cosine":
                score = sim

            elif mode == "cosine+source":
                score = (
                    ALPHA * sim +
                    BETA * SOURCE_WEIGHTS.get(meta["source"], 0.5)
                )

            elif mode == "full":
                score = self._weighted_score(
                    sim=sim,
                    source=meta["source"],
                    chunk_type=meta["chunk_type"],
                    claim=query
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            scored.append({
                "score": float(score),
                "cosine": float(sim),
                "title": meta["title"],
                "source": meta["source"],
                "chunk_type": meta["chunk_type"],
                "text": meta["text"]
            })

        # ---------- DIVERSITY PENALTY ----------
        scored.sort(key=lambda x: x["score"], reverse=True)

        seen_titles = {}
        for item in scored:
            count = seen_titles.get(item["title"], 0)
            if count > 0:
                item["score"] -= 0.10 * count
            seen_titles[item["title"]] = count + 1

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

# ================= QUICK TEST =================

if __name__ == "__main__":
    r = Retriever()
    claim = "GANs completely fail on high-resolution images."
    hits = r.search(claim, top_k=3)

    for h in hits:
        print(f"\nScore: {h['score']:.3f} | Cosine: {h['cosine']:.3f}")
        print(f"Source: {h['source']} | Type: {h['chunk_type']}")
        print(f"Title : {h['title']}")
        print(h["text"][:200], "...")