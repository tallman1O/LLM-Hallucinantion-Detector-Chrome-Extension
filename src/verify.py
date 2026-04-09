import os

from src.retrieve import Retriever
from src.claim_extractor import extract_verifiable_claims
from src.rewrite import SafeRewriter
from src.classifier import (
    classify_claim,
    classify_claim_type,
    get_aggregated_support_score,
    log_claim_debug,
)
from llama_cpp import Llama

LLAMA_MODEL_PATH = "models/llama.gguf"
DEBUG_CLAIM_CLASSIFICATION = os.getenv("DEBUG_CLAIM_CLASSIFICATION", "false").lower() == "true"


# ---------------- LLaMA INIT ----------------
try:
    llm = Llama(
        model_path=LLAMA_MODEL_PATH,
        n_ctx=2048,
        temperature=0.2,
        verbose=False
    )
except Exception:
    llm = None

# ---------------- EXPLANATION ----------------
def generate_explanation(claim, status, evidence):
    if llm is None or not evidence:
        return None

    snippets = "\n".join(
        f"- {e['snippet']}" for e in evidence[:2]
    )

    prompt = f"""
You are a scientific fact-checking assistant.

Claim:
"{claim}"

Verification result: {status}

Relevant evidence:
{snippets}

Explain in 2–3 sentences why the claim is labeled this way.
Do not invent facts.
"""

    out = llm(prompt, max_tokens=120)
    return out["choices"][0]["text"].strip()

# ---------------- VERIFIER ----------------
class HallucinationVerifier:
    def __init__(self):
        self.retriever = Retriever()
        self.rewriter = SafeRewriter()

    def format_evidence(self, evidence, max_len=180):
        formatted = []
        for e in evidence:
            formatted.append({
                "score": round(e["score"], 3),
                "title": e["title"],
                "source": e["source"],
                "chunk_type": e["chunk_type"],
                "snippet": e["text"][:max_len].replace("\n", " ")
            })
        return formatted

    def verify(self, text, top_k=3):
        claims, skipped_claims = extract_verifiable_claims(text)
        results = []

        for claim in claims:
            claim_type, claim_type_reason = classify_claim_type(claim)

            if claim_type != "factual_claim":
                if DEBUG_CLAIM_CLASSIFICATION:
                    log_claim_debug(
                        claim,
                        [],
                        0.0,
                        claim_type=claim_type,
                    )

                results.append({
                    "claim": claim,
                    "claim_type": claim_type,
                    "status": "UNVERIFIABLE",
                    "reason": claim_type_reason or "non_factual_claim",
                    "confidence": 0.0,
                    "evidence": [],
                    "sources": [],
                    "explanation": None,
                    "safe_rewrite": None,
                })
                continue

            raw_evidence = self.retriever.search(claim, top_k=top_k)
            debug_evidence = raw_evidence
            if DEBUG_CLAIM_CLASSIFICATION and top_k < 5:
                debug_evidence = self.retriever.search(claim, top_k=5)
                log_claim_debug(
                    claim,
                    debug_evidence,
                    get_aggregated_support_score(raw_evidence),
                    claim_type=claim_type,
                )
            elif DEBUG_CLAIM_CLASSIFICATION:
                log_claim_debug(
                    claim,
                    raw_evidence,
                    get_aggregated_support_score(raw_evidence),
                    claim_type=claim_type,
                )
            evidence = self.format_evidence(raw_evidence)
            classification = classify_claim(claim, raw_evidence)
            
            # ---------- OPTIONAL LLM STEPS ----------
            explanation = generate_explanation(claim, classification["verdict"], evidence)

            safe_rewrite = None
            if classification["verdict"] != "supported":
                safe_rewrite = self.rewriter.rewrite(
                    claim,
                    [e["snippet"] for e in evidence[:2]]
                )

            results.append({
                "claim": claim,
                "claim_type": claim_type,
                "status": classification["verdict"].upper(),
                "reason": classification["reason"],
                "confidence": round(classification["confidence"], 3),
                "evidence": evidence,
                "sources": classification["sources"],
                "explanation": explanation,
                "safe_rewrite": safe_rewrite
            })

        return {
            "input_text": text,
            "results": results,
            "skipped_sentences": skipped_claims,
        }

# ---------------- QUICK TEST ----------------
if __name__ == "__main__":
    v = HallucinationVerifier()
    out = v.verify("Diffusion models guarantee photorealistic images.")
    from pprint import pprint
    pprint(out)
