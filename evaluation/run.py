"""Run local semantic retrieval evaluation. No paid API calls."""
import argparse
import json
import platform
import time
from pathlib import Path

from evaluation.metrics import score_ranking

ROOT = Path(__file__).resolve().parents[1]


def load_dataset():
    return json.loads((ROOT / "evaluation/dataset.json").read_text(encoding="utf-8"))


def evaluate(rank, dataset, split, k):
    rows = []
    for query in dataset["questions"]:
        if query["split"] != split:
            continue
        start = time.perf_counter()
        ranked = rank(query["question"], k)
        rows.append({**query, "retrieved": ranked,
                     "retrieval_ms": (time.perf_counter() - start) * 1000,
                     "metrics": score_ranking(ranked, query["evidence"], k)})
    scored = [r["metrics"] for r in rows if r["metrics"] is not None]
    summary = {key: sum(m[key] for m in scored) / len(scored) for key in scored[0]} if scored else {}
    return {"split": split, "k": k, "answerable_count": len(scored),
            "unanswerable_count": len(rows) - len(scored), "metrics": summary,
            "mean_retrieval_ms": sum(r["retrieval_ms"] for r in rows) / len(rows) if rows else 0,
            "generation_evaluated": False, "api_tokens": 0, "api_cost": 0,
            "questions": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output", default="evaluation/results/baseline-dev.json")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    from news_research.retrieval import create_embeddings
    import numpy as np
    dataset = load_dataset()
    model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = create_embeddings(model)
    articles = dataset["articles"]
    vectors = np.asarray(embeddings.embed_documents([a["text"] for a in articles]))
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def rank(question, k):
        query = np.asarray(embeddings.embed_query(question))
        scores = vectors @ query / max(np.linalg.norm(query), 1e-12)
        return [articles[i]["id"] for i in np.argsort(-scores)[:k]]

    report = evaluate(rank, dataset, args.split, args.k)
    report.update({"model": model, "retriever": "semantic-cosine", "python": platform.python_version(),
                   "dataset_note": dataset["description"]})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "questions"}, indent=2))


if __name__ == "__main__":
    main()
