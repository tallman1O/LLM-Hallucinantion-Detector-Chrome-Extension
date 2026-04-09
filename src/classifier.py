import os
import re

import numpy as np

from src.claim_extractor import extract_action_phrase
from src.embedder import get_embedder

THRESHOLD_LOW = float(os.getenv("THRESHOLD_LOW", "0.45"))
THRESHOLD_HIGH = float(os.getenv("THRESHOLD_HIGH", "0.75"))
SUPPORTED_MIN = float(os.getenv("SUPPORTED_MIN", "0.92"))
DEBUG_CLAIM_CLASSIFICATION = os.getenv("DEBUG_CLAIM_CLASSIFICATION", "false").lower() == "true"
ACTION_SIMILARITY_THRESHOLD = 0.40
FICTIONAL_TERMS = {
    "crystals",
    "magic",
    "time-travel",
    "time travel",
    "teleport",
}
PHYSICALLY_IMPOSSIBLE_QUALIFIERS = {
    "physically impossible",
    "violates physics",
    "faster than light",
    "perpetual motion",
    "infinite energy",
}

NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "none",
    "cannot",
    "can't",
    "fail",
    "fails",
    "failed",
    "failing",
    "lack",
    "lacks",
    "lacking",
    "unlikely",
}
PREDICTION_MARKERS = {
    "will",
    "would",
    "going to",
    "expected to",
    "forecast",
    "predict",
    "predicted",
    "prediction",
    "likely",
    "soon",
    "upcoming",
}
OPINION_MARKERS = {
    "i think",
    "i believe",
    "in my opinion",
    "should",
    "best",
    "worst",
    "better",
    "important",
    "good",
    "bad",
    "great",
    "terrible",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def _similarity_score(doc: dict) -> float:
    if "similarity_score" in doc:
        return float(doc["similarity_score"])
    if "cosine" in doc:
        return float(doc["cosine"])
    if "score" in doc:
        return float(doc["score"])
    raise KeyError("similarity_score")


def _entity_match_count(doc: dict) -> int:
    return int(doc.get("entity_match_count", 0))


def _has_negation(text: str) -> bool:
    tokens = _tokenize(text)
    return any(term in tokens for term in NEGATION_TERMS)


def has_prior_implausibility(claim: str) -> bool:
    lowered = claim.lower()

    years = [int(match) for match in re.findall(r"\b(\d{4})\b", claim)]
    if any(year < 1900 or year > 2100 for year in years):
        return True

    if any(term in lowered for term in FICTIONAL_TERMS):
        return True

    if any(term in lowered for term in PHYSICALLY_IMPOSSIBLE_QUALIFIERS):
        return True

    return False


def classify_claim_type(claim: str) -> tuple[str, str | None]:
    lowered = claim.lower()

    if has_prior_implausibility(claim):
        return "fabrication", "fabricated_or_implausible_claim"

    if any(marker in lowered for marker in PREDICTION_MARKERS):
        return "prediction", "future speculation"

    if any(marker in lowered for marker in OPINION_MARKERS):
        return "opinion", "subjective claim"

    return "factual_claim", None


def get_aggregated_support_score(retrieved_docs: list[dict]) -> float:
    if not retrieved_docs:
        return 0.0
    return max(_similarity_score(doc) for doc in retrieved_docs)


def get_threshold_bucket(score: float) -> str:
    if score < THRESHOLD_LOW:
        return f"below_low (< {THRESHOLD_LOW:.2f})"
    if score < THRESHOLD_HIGH:
        return f"between_low_high ({THRESHOLD_LOW:.2f} - {THRESHOLD_HIGH:.2f})"
    if score < SUPPORTED_MIN:
        return f"between_high_supported ({THRESHOLD_HIGH:.2f} - {SUPPORTED_MIN:.2f})"
    return f"above_supported_min (>= {SUPPORTED_MIN:.2f})"


def log_claim_debug(
    claim: str,
    retrieved_docs: list[dict],
    aggregated_support_score: float,
    claim_type: str = "factual_claim",
) -> None:
    print(f"[claim-debug] claim: {claim}")
    print(f"[claim-debug] claim_type: {claim_type}")
    print("[claim-debug] top retrieved chunks:")

    for index, doc in enumerate(retrieved_docs[:5], start=1):
        cosine = _similarity_score(doc)
        title = doc.get("title", "<untitled>")
        snippet = (doc.get("text") or doc.get("abstract") or "").replace("\n", " ")[:140]
        print(
            f"[claim-debug] {index}. cosine={cosine:.4f} "
            f"title={title} matched_via={doc.get('matched_embedding') or doc.get('matched_via')}"
        )
        if snippet:
            print(f"[claim-debug]    chunk={snippet}")
        print(
            f"[claim-debug]    entity_matches={doc.get('entity_match_count', 0)} "
            f"matched_entities={doc.get('matched_claim_entities', [])}"
        )

    print(f"[claim-debug] aggregated support score: {aggregated_support_score:.4f}")
    print(f"[claim-debug] threshold bucket: {get_threshold_bucket(aggregated_support_score)}")


def _is_contradiction(claim: str, doc: dict, claim_embedding: np.ndarray) -> bool:
    abstract = doc.get("abstract", "")
    if not abstract:
        return False

    abstract_embedding = np.asarray(
        get_embedder().embed_single(abstract),
        dtype=np.float32,
    )
    semantic_similarity = float(
        np.dot(claim_embedding, abstract_embedding)
        / (np.linalg.norm(claim_embedding) * np.linalg.norm(abstract_embedding))
    )

    if semantic_similarity < THRESHOLD_HIGH:
        return False

    claim_has_negation = _has_negation(claim)
    abstract_has_negation = _has_negation(abstract)
    if claim_has_negation == abstract_has_negation:
        return False

    claim_tokens = _tokenize(claim) - NEGATION_TERMS
    abstract_tokens = _tokenize(abstract) - NEGATION_TERMS
    shared_tokens = claim_tokens & abstract_tokens

    return len(shared_tokens) >= 3


def _compute_action_similarity(claim: str, doc: dict) -> float | None:
    chunk_text = doc.get("text") or doc.get("abstract") or ""
    claim_action = extract_action_phrase(claim)
    chunk_action = extract_action_phrase(chunk_text)

    if not claim_action or not chunk_action:
        return None

    embedder = get_embedder()
    claim_action_embedding = np.asarray(embedder.embed_single(claim_action), dtype=np.float32)
    chunk_action_embedding = np.asarray(embedder.embed_single(chunk_action), dtype=np.float32)

    similarity = float(
        np.dot(claim_action_embedding, chunk_action_embedding)
        / (np.linalg.norm(claim_action_embedding) * np.linalg.norm(chunk_action_embedding))
    )
    return similarity


def classify_claim(claim: str, retrieved_docs: list[dict]) -> dict:
    highest_score = get_aggregated_support_score(retrieved_docs)
    prior_implausibility = has_prior_implausibility(claim)

    if not retrieved_docs:
        return {
            "verdict": "unverifiable",
            "reason": "implausible_claim" if prior_implausibility else "no_relevant_sources",
            "confidence": 0.0,
            "sources": [],
            "prior_implausibility": prior_implausibility,
        }

    top_doc = max(retrieved_docs, key=_similarity_score)
    entity_match_count = _entity_match_count(top_doc)

    if highest_score < THRESHOLD_LOW and entity_match_count == 0:
        return {
            "verdict": "unverifiable",
            "reason": "implausible_claim" if prior_implausibility else "no_relevant_sources",
            "confidence": 0.0,
            "sources": [],
            "prior_implausibility": prior_implausibility,
            "entity_match_count": entity_match_count,
        }

    if highest_score < THRESHOLD_HIGH:
        if entity_match_count >= 1:
            return {
                "verdict": "hallucinated",
                "reason": "weak_evidence_with_entity_match",
                "confidence": highest_score,
                "sources": [top_doc],
                "prior_implausibility": prior_implausibility,
                "entity_match_count": entity_match_count,
            }

        return {
            "verdict": "hallucinated",
            "reason": "weak_match",
            "confidence": highest_score,
            "sources": [top_doc],
            "prior_implausibility": prior_implausibility,
            "entity_match_count": entity_match_count,
        }

    action_similarity = _compute_action_similarity(claim, top_doc)
    if action_similarity is not None and action_similarity < ACTION_SIMILARITY_THRESHOLD:
        print(
            "[classifier] action-mismatch downgrade: "
            f"action_similarity={action_similarity:.4f} < {ACTION_SIMILARITY_THRESHOLD:.2f}"
        )
        return {
            "verdict": "unverifiable",
            "reason": "action_mismatch_downgrade",
            "confidence": highest_score,
            "sources": [top_doc],
            "prior_implausibility": prior_implausibility,
            "action_similarity": action_similarity,
            "entity_match_count": entity_match_count,
        }

    strong_docs = [
        doc for doc in retrieved_docs
        if _similarity_score(doc) >= THRESHOLD_HIGH
    ]
    claim_embedding = np.asarray(get_embedder().embed_single(claim), dtype=np.float32)
    contradicting_docs = [
        doc for doc in strong_docs
        if _is_contradiction(claim, doc, claim_embedding)
    ]

    if contradicting_docs:
        return {
            "verdict": "hallucinated",
            "reason": "contradiction_detected",
            "confidence": highest_score,
            "sources": contradicting_docs,
            "prior_implausibility": prior_implausibility,
            "action_similarity": action_similarity,
            "entity_match_count": entity_match_count,
        }

    if prior_implausibility:
        return {
            "verdict": "unverifiable",
            "reason": "implausible_claim",
            "confidence": highest_score,
            "sources": strong_docs,
            "prior_implausibility": True,
            "action_similarity": action_similarity,
            "entity_match_count": entity_match_count,
        }

    if highest_score < SUPPORTED_MIN:
        return {
            "verdict": "hallucinated",
            "reason": "below_supported_min",
            "confidence": highest_score,
            "sources": strong_docs,
            "prior_implausibility": prior_implausibility,
            "action_similarity": action_similarity,
            "entity_match_count": entity_match_count,
        }

    return {
        "verdict": "supported",
        "reason": "strong_match",
        "confidence": highest_score,
        "sources": strong_docs,
        "prior_implausibility": prior_implausibility,
        "action_similarity": action_similarity,
        "entity_match_count": entity_match_count,
    }
