# Comparing coverage

Select two to four articles from one collection and supply a topic. The app selects up to three 1,000-character excerpts from each article using keyword relevance, filling unused slots from the beginning of the article. All selected excerpts retain stable article and chunk IDs. The model returns structured rows labeled agreement, differing reports, or partial coverage.

Each row must represent every selected article exactly once. Reported facts need verbatim quotes from that specific article; cross-article citations and invented passages fail validation. Missing information is rendered only as **Not found in supplied excerpts**. This does not claim the full article omits it. Publication dates remain visible; differing figures can reflect revisions or reporting periods rather than contradictions.

Neither agreement nor differing reports determines truth or publisher bias. Labels and claim support still require human review. The feature is tested with controlled model responses; live generation quality has not been measured under verified free-tier billing.
