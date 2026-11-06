"""Fixed retrieval ablation: semantic/BM25/RRF at k=2 and k=4. No generation."""
import argparse
import json
from pathlib import Path

import numpy as np
from evaluation.run import load_dataset, evaluate
from news_research.ranking import bm25_rank, reciprocal_rank_fusion
from news_research.retrieval import create_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    data = load_dataset()
    articles = data["articles"]
    texts = [a["text"] for a in articles]
    model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = create_embeddings(model)
    vectors = np.asarray(embeddings.embed_documents(texts))
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def semantic(question, k):
        query = np.asarray(embeddings.embed_query(question))
        scores = vectors @ query / max(np.linalg.norm(query), 1e-12)
        return [articles[i]["id"] for i in np.argsort(-scores)[:k]]

    def keyword(question, k):
        return [articles[i]["id"] for i in bm25_rank(texts, question, k)]

    def hybrid(question, k):
        return reciprocal_rank_fusion([semantic(question, 12), keyword(question, 12)], k)

    reports = []
    for name, rank in [("semantic", semantic), ("keyword", keyword), ("hybrid", hybrid)]:
        for k in [2, 4]:
            report = evaluate(rank, data, args.split, k)
            report.update({"retriever": name, "model": model if name != "keyword" else "BM25 k1=1.5 b=0.75",
                           "dataset_note": data["description"]})
            reports.append(report)
            print(name, k, json.dumps(report["metrics"]))
    path = Path(f"evaluation/results/comparison-{args.split}.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
