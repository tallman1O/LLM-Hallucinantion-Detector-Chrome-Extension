from dotenv import load_dotenv
load_dotenv()

import os
import time
import sqlite3
import re
from semanticscholar import SemanticScholar

# ---------------- CONFIG ----------------

S2_API_KEY = os.getenv("S2_API_KEY")
assert S2_API_KEY, "❌ S2_API_KEY not found in environment"

QUERIES = [
    "diffusion models",
    "generative models",
    "GAN",
    "transformer",
    "large language models"
]

MAX_PER_QUERY = 10          # SAFE with free tier
SLEEP_SECONDS = 1.5         # obey rate limit

DATA_PROCESSED = "data/processed"
DB_PATH = f"{DATA_PROCESSED}/chunks.db"

os.makedirs(DATA_PROCESSED, exist_ok=True)

sch = SemanticScholar(api_key=S2_API_KEY)

# ---------------- HELPERS ----------------

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

# ---------------- FETCH ABSTRACTS ----------------

def fetch_abstracts():
    print("\n📚 Fetching Semantic Scholar abstracts (SAFE MODE)...")

    records = []

    for q in QUERIES:
        print(f"  🔍 Query: {q}")

        try:
            results = sch.search_paper(
                q,
                limit=MAX_PER_QUERY,
                fields=["title", "abstract", "year"]
            )

            count = 0
            for p in results:
                if count >= MAX_PER_QUERY:
                    break  # ⛔ HARD STOP — prevents pagination

                if not p.abstract:
                    continue

                records.append({
                    "title": p.title,
                    "year": p.year,
                    "chunk": p.abstract,
                    "chunk_type": "abstract"
                })
                count += 1

            print(f"    ↳ collected {count} abstracts")

        except Exception as e:
            print(f"⚠️ Skipping query '{q}' due to error:", e)

        time.sleep(SLEEP_SECONDS)  # respect rate limit

    print(f"✔ Total abstracts collected: {len(records)}")
    return records

# ---------------- SAVE TO DB ----------------

def save_db(records):
    print("\n🗄 Saving corpus to SQLite...")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS chunks;")
    c.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            year INTEGER,
            chunk TEXT,
            chunk_type TEXT
        );
    """)

    for r in records:
        c.execute(
            "INSERT INTO chunks (title, year, chunk, chunk_type) VALUES (?, ?, ?, ?)",
            (r["title"], r["year"], r["chunk"], r["chunk_type"])
        )

    conn.commit()
    conn.close()

    print("✔ DB saved at:", DB_PATH)

# ---------------- MAIN ----------------

if __name__ == "__main__":
    records = fetch_abstracts()
    save_db(records)
    print("\n🎉 Phase-1 corpus build complete!")