import time
from src.verify import HallucinationVerifier

ENGINE_VERSION = "1.0"

verifier = HallucinationVerifier()


def get_overall_status(claims):
    statuses = {claim["status"] for claim in claims}

    if "HALLUCINATED" in statuses:
        return "HALLUCINATED"

    if statuses == {"SUPPORTED"}:
        return "SUPPORTED"

    return "UNVERIFIABLE"

def run_api(text: str):
    start = time.time()
    raw = verifier.verify(text)
    latency = int((time.time() - start) * 1000)

    claims = []
    sources = set()

    for r in raw["results"]:
        for ev in r["evidence"]:
            sources.add(ev["source"])

        claims.append({
            "claim": r["claim"],
            "claim_type": r.get("claim_type", "factual_claim"),
            "status": r["status"],
            "confidence": r["confidence"],
            "top_evidence": r["evidence"][:2],
            "safe_rewrite": r["safe_rewrite"],
            "explanation": r["explanation"]
        })

    if claims:
        overall_status = get_overall_status(claims)
        overall_confidence = max(c["confidence"] for c in claims)
    else:
        overall_status = "NO_CLAIMS"
        overall_confidence = 0.0

    return {
        "input_text": text,
        "overall_status": overall_status,
        "overall_confidence": overall_confidence,
        "claims": claims,
        "meta": {
            "engine_version": ENGINE_VERSION,
            "latency_ms": latency,
            "sources_used": list(sources)
        }
    }
