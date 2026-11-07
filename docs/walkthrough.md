# Three-minute recruiter walkthrough

**0:00 — The problem.** Two news reports quote different profits. Was one wrong, or was the second published after a revision? Open the public demo and search for “How did Harbor Bank profit change?” Explain that the public sample is fictional and intentionally runs without a model account.

**0:30 — Inspect the evidence.** Show the dated initial and revised passages: 18 million rupees became 16 million after a provisioning adjustment. Explain why returning retrieved URLs alone cannot establish support for a claim. In a configured private workspace, expand a numbered citation to inspect the exact quote, passage, and publisher.

**1:00 — Explain the data lifecycle.** Show the collection selector or the storage diagram in the README. SQLite is the canonical record; each collection gets a rebuildable vector snapshot. Duplicate content is stored once. Deleting an article removes it from future retrieval without touching another topic. Show `tests/test_storage.py` for a reproducible demonstration.

**1:40 — Explain one measured decision.** Show the dev and held-out retrieval results. On the small synthetic dev set, k=2 looked sufficient. On held-out questions it missed part of one multi-source answer; k=4 recovered it. Hybrid did not improve recall over the default at k=4. State the sample size and why these numbers do not generalize to all news.

**2:15 — Show a failure case.** Ask an unsupported question in a configured private workspace, or show the offline evidence-validation tests. Distinguish missing information from invalid generated citations. A real quote can still be used to support a misleading interpretation; human claim-level review remains important.

**2:45 — Close with tradeoffs.** Streamlit keeps the interface simple. Local indexing avoids a hosted vector bill. A read-only public demo prevents visitors from consuming the owner's key or reading private research. The next investment should be real-world, independently labeled evaluation and stronger multi-user access control if the product scope grows.

Interview statement: “I took a URL-based RAG prototype and added isolated collections, verifiable citation references, and multi-source comparison. I benchmarked retrieval strategies, kept the simpler default where evidence did not justify extra complexity, and separated measured retrieval quality from unmeasured generation quality.”
