import time
from src.verify import HallucinationVerifier

ENGINE_VERSION = "1.0"

verifier = HallucinationVerifier()

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
            "status": r["status"],
            "confidence": r["confidence"],
            "top_evidence": r["evidence"][:2],
            "safe_rewrite": r["safe_rewrite"],
            "explanation": r["explanation"]
        })

    return {
        "input_text": text,
        "overall_status": max(claims, key=lambda x: x["confidence"])["status"],
        "overall_confidence": max(c["confidence"] for c in claims),
        "claims": claims,
        "meta": {
            "engine_version": ENGINE_VERSION,
            "latency_ms": latency,
            "sources_used": list(sources)
        }
    }