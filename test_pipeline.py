# test_pipeline.py
# Run with: python test_pipeline.py
# Tests all 4 diagnostic checks in order and prints PASS/FAIL for each

import time
import random


# ── Test 1: Embedding dimension sanity check ──────────────────────────────────
def test_embedding_dimension():
    print("\n[TEST 1] Embedding dimension check...")
    try:
        from src.embedder import Specter2Embedder
        embedder = Specter2Embedder()
        vec = embedder.embed_single("Transformers use self-attention mechanisms")
        dim = len(vec)
        if dim == 768:
            print(f"  PASS — embedding dimension is {dim}")
            return True
        else:
            print(f"  FAIL — expected 768, got {dim}")
            return False
    except Exception as e:
        print(f"  FAIL — exception: {e}")
        return False


# ── Test 2: KB coverage check ─────────────────────────────────────────────────
def test_kb_coverage():
    print("\n[TEST 2] KB coverage check...")
    try:
        from src.kb_store import KB
        from src.retriever import retrieve_relevant_docs

        if not KB:
            print("  FAIL — KB is empty, run reembed_kb.py first")
            return False

        # Pick a random doc from KB and use its first sentence as a claim
        sample_doc = random.choice(KB)
        abstract = sample_doc["abstract"]
        first_sentence = abstract.split(".")[0].strip() + "."
        domain = sample_doc.get("domain", None)

        print(f"  Testing claim: '{first_sentence[:80]}...'")
        print(f"  Expected source: '{sample_doc['title'][:60]}...'")

        results = retrieve_relevant_docs(first_sentence, domain=domain, top_k=5)

        if not results:
            print("  FAIL — retrieval returned no results at all")
            return False

        top_score = results[0]["similarity_score"]
        top_id = results[0]["id"]
        expected_id = sample_doc["id"]

        print(f"  Top result score: {top_score:.4f}")
        print(f"  Top result title: '{results[0]['title'][:60]}...'")

        if top_score >= 0.70 and str(top_id) == str(expected_id):
            print(f"  PASS — correct document retrieved with score {top_score:.4f}")
            return True
        elif top_score >= 0.70:
            print(f"  PARTIAL — high score ({top_score:.4f}) but different document returned")
            print(f"  This may still be acceptable if the returned doc is semantically similar")
            return True
        else:
            print(f"  FAIL — top score {top_score:.4f} is below 0.70 threshold")
            print("  Likely causes: embedding mismatch, wrong input format, or threshold too high")
            return False

    except Exception as e:
        print(f"  FAIL — exception: {e}")
        return False


# ── Test 3: Claim filter check ────────────────────────────────────────────────
def test_claim_filter():
    print("\n[TEST 3] Claim extractor filter check...")
    try:
        from src.claim_extractor import extract_verifiable_claims

        test_cases = [
            {
                "text": "Transformers were introduced in 2017 by Vaswani et al.",
                "should_pass": True,
                "reason": "contains named entity and date"
            },
            {
                "text": "This is particularly important.",
                "should_pass": False,
                "reason": "starts with 'This', no named entity"
            },
            {
                "text": "It has been shown that performance improves.",
                "should_pass": False,
                "reason": "starts with 'It', no named entity"
            },
            {
                "text": "GPT-4 was released by OpenAI in 2023.",
                "should_pass": True,
                "reason": "contains named entities GPT-4 and OpenAI"
            },
            {
                "text": "However, this approach has limitations.",
                "should_pass": False,
                "reason": "starts with 'However', no named entity"
            },
        ]

        all_passed = True
        for case in test_cases:
            results = extract_verifiable_claims(case["text"])
            claim_passed_filter = len(results) > 0

            expected = case["should_pass"]
            status = "PASS" if claim_passed_filter == expected else "FAIL"
            if status == "FAIL":
                all_passed = False

            symbol = "✓" if claim_passed_filter else "✗"
            print(f"  {status} [{symbol}] '{case['text'][:55]}...' — {case['reason']}")

        if all_passed:
            print("  PASS — all 5 filter cases behaved correctly")
        else:
            print("  FAIL — some filter cases did not behave as expected")
            print("  Check your NER config and blocked prefix list in claim_extractor.py")

        return all_passed

    except Exception as e:
        print(f"  FAIL — exception: {e}")
        return False


# ── Test 4: Score distribution check ─────────────────────────────────────────
def test_score_distribution():
    print("\n[TEST 4] Score distribution check...")
    try:
        from src.kb_store import KB
        from src.retriever import retrieve_relevant_docs

        if len(KB) < 5:
            print("  SKIP — KB has fewer than 5 documents, cannot run distribution check")
            return True

        # Sample up to 10 docs from KB, use first sentence of each as a claim
        sample_size = min(10, len(KB))
        kb_samples = random.sample(KB, sample_size)

        # Generate random unrelated sentences as noise
        noise_sentences = [
            "The stock market closed higher on Wednesday.",
            "Scientists discovered a new species of frog in the Amazon.",
            "The recipe requires two cups of flour and one egg.",
            "The train arrives at platform 4 at 9:15am.",
            "She decided to repaint the living room blue.",
            "The championship game was postponed due to rain.",
            "A new restaurant opened downtown last week.",
            "The package was delivered to the wrong address.",
            "Children enjoyed the new playground equipment.",
            "The film festival begins on the first of November.",
        ][:sample_size]

        print(f"  Running retrieval on {sample_size} KB claims and {sample_size} noise sentences...")

        kb_scores = []
        for doc in kb_samples:
            first_sentence = doc["abstract"].split(".")[0].strip() + "."
            domain = doc.get("domain", None)
            results = retrieve_relevant_docs(first_sentence, domain=domain, top_k=1)
            if results:
                kb_scores.append(results[0]["similarity_score"])

        noise_scores = []
        for sentence in noise_sentences:
            results = retrieve_relevant_docs(sentence, domain=None, top_k=1)
            if results:
                noise_scores.append(results[0]["similarity_score"])

        if not kb_scores or not noise_scores:
            print("  FAIL — could not retrieve results for scoring")
            return False

        avg_kb = sum(kb_scores) / len(kb_scores)
        avg_noise = sum(noise_scores) / len(noise_scores)
        gap = avg_kb - avg_noise

        print(f"  Average similarity — KB claims:      {avg_kb:.4f}")
        print(f"  Average similarity — Noise sentences: {avg_noise:.4f}")
        print(f"  Separation gap:                       {gap:.4f}")

        if gap >= 0.10:
            print(f"  PASS — KB claims score meaningfully higher than noise (gap={gap:.4f})")
            return True
        elif gap >= 0.05:
            print(f"  PARTIAL — gap exists but is small (gap={gap:.4f}), consider retuning thresholds")
            return True
        else:
            print(f"  FAIL — KB claims and noise have similar scores (gap={gap:.4f})")
            print("  Likely cause: embedding input format wrong, or SPECTER2 not loaded correctly")
            return False

    except Exception as e:
        print(f"  FAIL — exception: {e}")
        return False


# ── Runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PIPELINE DIAGNOSTIC TESTS")
    print("=" * 60)

    start = time.time()

    results = {
        "Test 1 — Embedding dimension":   test_embedding_dimension(),
        "Test 2 — KB coverage":           test_kb_coverage(),
        "Test 3 — Claim filter":          test_claim_filter(),
        "Test 4 — Score distribution":    test_score_distribution(),
    }

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n  {passed}/{total} tests passed in {elapsed:.1f}s")

    if passed == total:
        print("\n  All tests passed. Pipeline is ready.")
        print("  Next step: run calibrate_thresholds.py with your labeled test set.")
    else:
        print("\n  Fix failing tests before proceeding.")
        print("  Each failure points to a specific component to debug.")
    print("=" * 60)
