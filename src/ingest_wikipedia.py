import requests
import sqlite3
import time
import hashlib

DB_PATH = "data/processed/chunks.db"

WIKI_PAGES = [
    "Generative adversarial network",
    "Diffusion model",
    "Large language model",
    "Transformer (machine learning)",
    "Attention (machine learning)",
    "Neural network",
    "Deep learning",
    "Machine learning",
    "Image synthesis",
    "Text-to-image model",
    "Foundation model",
    "Autoregressive model",
    "Normalizing flow",
    "Language model",
    "Representation learning"
]

# ---------- Helpers ----------

def hash_text(text: str) -> str:
    text = text.lower().strip()
    text = " ".join(text.split())
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def fetch_wikipedia_sections(title: str):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "prop": "sections|wikitext",
        "format": "json",
    }

    r = requests.get(url, params=params, headers={
        "User-Agent": "HallucinationVerifier/1.0"
    })

    if r.status_code != 200:
        return None

    data = r.json()
    if "parse" not in data:
        return None

    sections = data["parse"]["sections"]
    full_text = data["parse"]["wikitext"]["*"]

    return sections, full_text

# ---------- Ingest ----------

def ingest_wikipedia():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    inserted = 0
    skipped = 0

    for page in WIKI_PAGES:
        print(f"📘 Fetching Wikipedia: {page}")
        text = fetch_wikipedia_summary(page)

        if not text:
            continue

        chunk_hash = hash_text(text)

        # Deduplication check
        c.execute(
            "SELECT 1 FROM chunks WHERE content_hash = ?",
            (chunk_hash,)
        )
        if c.fetchone():
            skipped += 1
            continue

        c.execute("""
            INSERT INTO chunks
                (title, year, chunk, chunk_type, source, content_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            page,
            None,
            text,
            "wiki",
            "wikipedia",
            chunk_hash
        ))

        inserted += 1
        time.sleep(1)  # be polite to Wikipedia

    conn.commit()
    conn.close()

    print(f"\n✅ Wikipedia ingestion complete")
    print(f"   Inserted: {inserted}")
    print(f"   Skipped (duplicates): {skipped}")

if __name__ == "__main__":
    ingest_wikipedia()