# Evaluation protocol

This is a small, authored synthetic finance-news fixture, not real reporting or a claim of production accuracy. Each short article is a single evidence passage (below the existing 1,000-character chunk limit). Questions include exact figures, paraphrases, multi-source comparisons, dates, and intentionally absent facts. Labels are authored expectations, not independent human annotations.

Run `python -m evaluation.run --split dev` to establish the semantic baseline using the same MiniLM model as the app. It downloads public model weights if necessary and makes **zero Groq calls**. Corpus encoding is excluded from query latency. These are evidence-passage metrics, not a full scraper-to-answer benchmark. Precision divides by k even when fewer results exist. Unanswerable questions are excluded from relevance metrics; retrieving something for them does not imply an answer is justified.

Use dev questions to choose configurations. Inspect test questions only after fixing the choice. Do not tune against test results. Dataset, model, k, per-question rankings, and timings are saved with each run. The same corpus is shared between splits; this is a query holdout, not an unseen-document benchmark.

Citation support requires reviewing claims against cited passages. Record supported/unsupported claims separately from valid citation IDs. For abstention, report decline rate on unanswerable questions and false-decline rate on answerable questions separately. Generation latency, tokens, and support scores remain unmeasured until generation is explicitly tested. No similarity score is called answer confidence.

Live evaluation is intentionally disabled by default: an API key does not establish whether its account has paid billing enabled. Free-tier-only authorization does not authorize paid API usage.
