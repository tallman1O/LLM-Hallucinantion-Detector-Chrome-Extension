from src.retriever import retrieve_relevant_docs


class Retriever:
    def __init__(self):
        print("✅ Retriever ready")

    def search(self, query, top_k=3, mode="full", domain=None):
        if mode != "full":
            raise ValueError("Retriever.search now supports only mode='full'.")

        results = retrieve_relevant_docs(
            claim=query,
            domain=domain or "",
            top_k=top_k,
        )

        return [
            {
                "id": result["id"],
                "similarity_score": float(result["similarity_score"]),
                "score": float(result["score"]),
                "cosine": float(result.get("raw_similarity_score", result["similarity_score"])),
                "title": result["title"],
                "source": result["source"],
                "domain": result.get("domain", "general"),
                "chunk_type": result.get("chunk_type", "abstract"),
                "text": result["abstract"],
                "matched_embedding": result["matched_via"],
                "entity_mismatch_penalty": float(result.get("entity_mismatch_penalty", 0.0)),
                "unmatched_claim_entities": result.get("unmatched_claim_entities", []),
                "matched_claim_entities": result.get("matched_claim_entities", []),
                "entity_match_count": int(result.get("entity_match_count", 0)),
            }
            for result in results
        ]


if __name__ == "__main__":
    retriever = Retriever()
    claim = "GANs completely fail on high-resolution images."
    hits = retriever.search(claim, top_k=3)

    for hit in hits:
        print(f"\nScore: {hit['score']:.3f} | Cosine: {hit['cosine']:.3f}")
        print(f"Source: {hit['source']} | Type: {hit['chunk_type']}")
        print(f"Title : {hit['title']}")
        print(hit["text"][:200], "...")
