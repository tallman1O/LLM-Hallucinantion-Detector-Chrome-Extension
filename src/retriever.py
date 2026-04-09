import os

import numpy as np

from src.claim_extractor import extract_entity_markers
from src.embedder import get_embedder
from src.kb_store import KB

DEBUG_RETRIEVAL = os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true"
ENTITY_MISMATCH_PENALTY = float(os.getenv("ENTITY_MISMATCH_PENALTY", "0.25"))

SOURCE_WEIGHTS = {
    "paper": 1.0,
    "wikipedia": 0.7,
}

CHUNK_TYPE_WEIGHTS = {
    "abstract": 1.0,
    "pdf": 0.85,
    "wiki": 0.7,
    "wiki_section": 0.7,
}

ALPHA = 0.6
BETA = 0.2
GAMMA = 0.2

ABSOLUTE_TERMS = [
    "always", "never", "completely", "entirely",
    "guarantees", "perfect", "fails", "impossible",
]

EMBEDDING_COLUMNS = (
    "embedding",
    "title_embedding",
    "chunk_embedding",
)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def overstatement_penalty(claim: str) -> float:
    lowered = claim.lower()
    return 0.05 * sum(term in lowered for term in ABSOLUTE_TERMS)


def _weighted_score(similarity: float, source: str, chunk_type: str, claim: str) -> float:
    source_weight = SOURCE_WEIGHTS.get(source, 0.5)
    chunk_weight = CHUNK_TYPE_WEIGHTS.get(chunk_type, 0.7)
    penalty = overstatement_penalty(claim)

    return (
        ALPHA * similarity
        + BETA * source_weight
        + GAMMA * chunk_weight
        - penalty
    )


def _search_by_embedding_column(
    claim_embedding: np.ndarray,
    domain: str,
    embedding_key: str,
    claim: str,
    top_k: int,
) -> list[dict]:
    results = []

    for doc in KB:
        if domain and doc.get("domain") != domain:
            continue

        similarity = cosine_similarity(claim_embedding, doc[embedding_key])
        source = doc.get("source") or "unknown"
        chunk_type = doc.get("chunk_type") or "abstract"

        results.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "abstract": doc["abstract"],
                "source": source,
                "domain": doc.get("domain", "general"),
                "chunk_type": chunk_type,
                "similarity_score": similarity,
                "score": _weighted_score(similarity, source, chunk_type, claim),
                "matched_via": embedding_key,
            }
        )

    results.sort(key=lambda item: item["similarity_score"], reverse=True)
    return results[:top_k]


def _apply_entity_mismatch_penalty(claim: str, results: list[dict]) -> list[dict]:
    if not results:
        return results

    claim_markers = extract_entity_markers(claim)
    penalizable_claim_markers = [
        marker for marker in claim_markers
        if marker["penalizable"]
    ]
    if not claim_markers:
        for result in results:
            result["raw_similarity_score"] = result["similarity_score"]
            result["entity_mismatch_penalty"] = 0.0
            result["unmatched_claim_entities"] = []
            result["matched_claim_entities"] = []
            result["entity_match_count"] = 0
        return results

    chunk_markers = set()
    normalized_chunk_texts = []
    for result in results:
        abstract = result.get("abstract", "")
        normalized_chunk_texts.append(abstract.lower())
        for marker in extract_entity_markers(abstract):
            chunk_markers.add(marker["normalized"])

    unmatched_entities = []
    matched_entities = []
    for marker in claim_markers:
        normalized = marker["normalized"]
        if normalized in chunk_markers or any(normalized in chunk_text for chunk_text in normalized_chunk_texts):
            matched_entities.append(marker["text"])
            continue
        unmatched_entities.append(marker["text"])

    penalizable_unmatched_entities = []
    for marker in penalizable_claim_markers:
        if marker["text"] in unmatched_entities:
            penalizable_unmatched_entities.append(marker["text"])

    penalty = ENTITY_MISMATCH_PENALTY * len(set(penalizable_unmatched_entities))

    for result in results:
        result["raw_similarity_score"] = result["similarity_score"]
        result["entity_mismatch_penalty"] = penalty
        result["unmatched_claim_entities"] = sorted(set(unmatched_entities))
        result["matched_claim_entities"] = sorted(set(matched_entities))
        result["entity_match_count"] = len(set(matched_entities))
        result["similarity_score"] = max(result["similarity_score"] - penalty, 0.0)
        result["score"] = max(result["score"] - penalty, 0.0)

    results.sort(key=lambda item: item["similarity_score"], reverse=True)
    return results


def retrieve_relevant_docs(claim: str, domain: str, top_k: int = 5) -> list[dict]:
    claim_embedding = np.asarray(get_embedder().embed_single(claim), dtype=np.float32)
    merged: dict[str, dict] = {}

    for embedding_key in EMBEDDING_COLUMNS:
        partial_results = _search_by_embedding_column(
            claim_embedding=claim_embedding,
            domain=domain,
            embedding_key=embedding_key,
            claim=claim,
            top_k=top_k,
        )

        for result in partial_results:
            existing = merged.get(result["id"])
            if existing is None or result["similarity_score"] > existing["similarity_score"]:
                merged[result["id"]] = result

    final_results = sorted(
        merged.values(),
        key=lambda item: item["similarity_score"],
        reverse=True,
    )[:top_k]
    final_results = _apply_entity_mismatch_penalty(claim, final_results)

    if DEBUG_RETRIEVAL:
        log_retrieval_diagnostics(claim, final_results)

    return final_results


def log_retrieval_diagnostics(claim: str, results: list[dict]) -> None:
    similarity_scores = [round(result["similarity_score"], 4) for result in results]
    has_045 = any(result["similarity_score"] >= 0.45 for result in results)
    has_060 = any(result["similarity_score"] >= 0.60 for result in results)
    has_070 = any(result["similarity_score"] >= 0.70 for result in results)

    print(f"[retrieval] claim: {claim}")
    print(f"[retrieval] results returned: {len(results)}")
    print(f"[retrieval] similarity scores: {similarity_scores}")
    print(f"[retrieval] any >= 0.45: {has_045}")
    print(f"[retrieval] any >= 0.60: {has_060}")
    print(f"[retrieval] any >= 0.70: {has_070}")
    if results:
        print(f"[retrieval] entity mismatch penalty: {results[0].get('entity_mismatch_penalty', 0.0):.4f}")
        print(f"[retrieval] unmatched claim entities: {results[0].get('unmatched_claim_entities', [])}")

    if not results:
        print("[retrieval] complete retrieval failure: no results returned")
    elif all(result["similarity_score"] < 0.45 for result in results):
        print("[retrieval] complete retrieval failure: all results below 0.45")
