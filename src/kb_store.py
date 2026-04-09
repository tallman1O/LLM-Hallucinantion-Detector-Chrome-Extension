import json
from pathlib import Path

import numpy as np

KB_PATH = Path("kb_embeddings.json")


def load_kb(path: str = "kb_embeddings.json") -> list[dict]:
    kb_path = Path(path)

    if not kb_path.exists():
        raise FileNotFoundError("kb_embeddings.json not found. Run reembed_kb.py first.")

    with kb_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    for record in records:
        record["embedding"] = np.asarray(record["embedding"], dtype=np.float32)
        record["title_embedding"] = np.asarray(record["title_embedding"], dtype=np.float32)
        record["chunk_embedding"] = np.asarray(record["chunk_embedding"], dtype=np.float32)

    return records


KB = load_kb()
