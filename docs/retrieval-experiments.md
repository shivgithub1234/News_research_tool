# Retrieval decision, fixed before inspecting the test split

The default remains semantic retrieval at k=4. BM25 keyword retrieval and equal-weight reciprocal-rank fusion (constant 60) are optional controls. Embeddings are now explicitly normalized, aligning Chroma's Euclidean ranking with the evaluation's cosine ordering. Existing collection snapshots rebuild under a new version tag.

Dev results (26 answerable synthetic questions; 4 unanswerable excluded from relevance scores): all six configurations (semantic, BM25, hybrid; k=2/4) reached recall 1.0 and complete-evidence coverage 1.0. MRR was 0.9231 for semantic/hybrid and 0.9615 for keyword search. k=2 precision was 0.5385; k=4 precision was 0.2692. More context did not improve recall on this tiny fixture. This does not establish the best k for real articles.

Why retain the baseline? The synthetic fixture contains only eight short passages with strong name/number overlap. It cannot justify replacing semantic retrieval, reducing context for long multi-article questions, or adding a reranker. Hybrid has no demonstrated improvement here. The optional modes enable future experiments on a larger independently labeled corpus without changing the current default.

Chunk sizes are unchanged: all fixture passages are below the current chunk limit, so a chunk-size ablation would be uninformative. Reranking is deferred until retrieval failures justify its extra latency. These are deliberate limitations, not unmeasured performance claims.

Reproduce with `python -m evaluation.compare --split dev`, then `--split test` after fixing the choice. Per-question results and timings are committed under `evaluation/results/`. Generation, citation support, and abstention accuracy are separate evaluations; they are not inferred from these retrieval scores.

## Held-out results (choice unchanged)

Ten answerable questions and four unanswerable questions were held out. Semantic k=2 had recall 0.95 and complete-evidence coverage 0.90, while semantic k=4 achieved 1.0 for both (MRR 0.95). Keyword and hybrid reached recall and complete-evidence coverage 1.0 at both k values. Keyword MRR was 1.0; hybrid MRR was 0.95. At k=4 all methods had precision 0.30. These results support retaining four passages for the existing default and illustrate why the apparently sufficient k=2 dev result was not enough to justify a change. The ten-question holdout is too small for general claims of superiority.
