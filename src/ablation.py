from src.retrieve import Retriever
from src.claims import split_into_claims

retriever = Retriever()

MODES = ["cosine", "cosine+source", "full"]

with open("data/claims_test.txt") as f:
    claims = [l.strip() for l in f if l.strip()]

for claim in claims:
    print("\n🧪 Claim:", claim)

    for mode in MODES:
        evidence = retriever.search(claim, top_k=1, mode=mode)
        top = evidence[0]
        score = top["score"]
        title = top["title"]

        print(f"  [{mode:15}] {round(score,3)} → {title}")
