import re

CONJUNCTIONS = ["and", "but", "while", "whereas", "however", "although"]

def split_into_claims(text: str):
    """
    Naive but effective claim splitter for research text.
    """
    text = text.strip().rstrip(".")
    parts = [text]

    for conj in CONJUNCTIONS:
        new_parts = []
        for p in parts:
            if conj in p.lower():
                split = re.split(rf"\b{conj}\b", p, flags=re.IGNORECASE)
                new_parts.extend([s.strip() for s in split if len(s.strip()) > 6])
            else:
                new_parts.append(p)
        parts = new_parts

    return parts