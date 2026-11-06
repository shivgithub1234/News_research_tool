"""Retrieval metrics over stable evidence IDs, not generated answers."""


def score_ranking(ranked, relevant, k=4):
    if k < 1:
        raise ValueError("k must be positive")
    ranked = list(dict.fromkeys(ranked))[:k]
    relevant = set(relevant)
    if not relevant:
        return None  # Unanswerable questions are evaluated separately.
    hits = relevant.intersection(ranked)
    first = next((i for i, item in enumerate(ranked, 1) if item in relevant), None)
    return {
        "precision": len(hits) / k,
        "recall": len(hits) / len(relevant),
        "hit": float(bool(hits)),
        "mrr": 1 / first if first else 0.0,
        "complete_evidence": float(hits == relevant),
    }
