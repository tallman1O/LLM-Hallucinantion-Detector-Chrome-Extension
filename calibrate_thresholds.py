import json
import statistics
import sys
from collections import defaultdict

from src.retriever import retrieve_relevant_docs


def summarize_scores(scores: list[float]) -> str:
    if not scores:
        return "count=0"

    ordered = sorted(scores)
    median = statistics.median(ordered)
    mean = statistics.mean(ordered)
    return (
        f"count={len(ordered)} "
        f"min={ordered[0]:.4f} "
        f"median={median:.4f} "
        f"mean={mean:.4f} "
        f"max={ordered[-1]:.4f}"
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: ./venv/bin/python calibrate_thresholds.py path/to/test_set.json")

    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        test_set = json.load(handle)

    distributions: dict[str, list[float]] = defaultdict(list)

    for item in test_set:
        claim = item["claim"]
        label = item["label"]
        domain = item.get("domain", "")
        results = retrieve_relevant_docs(claim, domain, top_k=5)
        top_score = results[0]["similarity_score"] if results else 0.0
        distributions[label].append(top_score)
        print(f"{label:14} score={top_score:.4f} claim={claim}")

    print("\nScore distributions by label:")
    for label in ("supported", "hallucinated", "unverifiable"):
        print(f"- {label}: {summarize_scores(distributions[label])}")

    supported_scores = distributions["supported"]
    unverifiable_scores = distributions["unverifiable"]
    hallucinated_scores = distributions["hallucinated"]

    recommended_low = (
        (percentile(unverifiable_scores, 0.9) + percentile(supported_scores + hallucinated_scores, 0.1)) / 2
        if (unverifiable_scores and (supported_scores or hallucinated_scores))
        else 0.35
    )
    recommended_high = (
        (percentile(unverifiable_scores, 1.0) + percentile(supported_scores, 0.25)) / 2
        if (unverifiable_scores and supported_scores)
        else 0.55
    )

    print("\nRecommended thresholds:")
    print(f"- THRESHOLD_LOW={recommended_low:.4f}")
    print(f"- THRESHOLD_HIGH={recommended_high:.4f}")


if __name__ == "__main__":
    main()
