from src.retrieve import Retriever
from src.claims import split_into_claims
from src.rewrite import SafeRewriter
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
SUPPORTED = 0.60
UNCERTAIN = 0.45
LLAMA_MODEL_PATH = "models/llama.gguf"

ABSOLUTE_TERMS = [
    "always", "never", "completely", "entirely",
    "guarantees", "guarantee", "perfect", "impossible"
]

def has_absolute_language(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in ABSOLUTE_TERMS)
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
        claims = split_into_claims(text)
        results = []

        for claim in claims:
            raw_evidence = self.retriever.search(claim, top_k=top_k)
            evidence = self.format_evidence(raw_evidence)

            base_score = raw_evidence[0]["score"] if raw_evidence else 0.0

            # ---------- CONTRADICTION CHECK ----------
            penalty = 0.0
            contradiction_info = None

            papers = [e for e in evidence if e["source"] == "paper"]
            wikis  = [e for e in evidence if e["source"] == "wikipedia"]

            if papers and wikis:
                emb = self.retriever.model.encode(
                    [papers[0]["snippet"], wikis[0]["snippet"]],
                    normalize_embeddings=True
                )

                sim = float(cosine_similarity(
                    emb[0].reshape(1, -1),
                    emb[1].reshape(1, -1)
                )[0][0])

                if sim < 0.45:
                    penalty = 0.20
                elif sim < 0.60:
                    penalty = 0.10

                contradiction_info = {
                    "detected": penalty > 0,
                    "paper_wiki_similarity": round(sim, 3),
                    "penalty": penalty
                }

            final_score = max(base_score - penalty, 0.0)

            # ---------- STATUS ----------
            if final_score >= SUPPORTED:
                status = "SUPPORTED"
            elif final_score >= UNCERTAIN:
                status = "PARTIALLY_SUPPORTED"
            else:
                status = "LIKELY_HALLUCINATED"
            if status == "SUPPORTED" and has_absolute_language(claim):
                status = "PARTIALLY_SUPPORTED"
            
            # ---------- OPTIONAL LLM STEPS ----------
            explanation = generate_explanation(claim, status, evidence)

            safe_rewrite = None
            if status != "SUPPORTED":
                safe_rewrite = self.rewriter.rewrite(
                    claim,
                    [e["snippet"] for e in evidence[:2]]
                )

            results.append({
                "claim": claim,
                "status": status,
                "confidence": round(final_score, 3),
                "evidence": evidence,
                "contradiction": contradiction_info,
                "explanation": explanation,
                "safe_rewrite": safe_rewrite
            })

        return {
            "input_text": text,
            "results": results
        }

# ---------------- QUICK TEST ----------------
if __name__ == "__main__":
    v = HallucinationVerifier()
    out = v.verify("Diffusion models guarantee photorealistic images.")
    from pprint import pprint
    pprint(out)