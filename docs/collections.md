# Saved collections

SQLite in `data/research.sqlite3` stores named collections, article content and metadata, content hashes, ingestion state, and extraction errors. `NEWS_RESEARCH_DATA_DIR` can point to persistent storage outside the repository. Data and vector files are ignored by Git.

Each collection has a generated ID. URL normalization removes fragments and common tracking parameters; exact text hashes catch content duplicated under different URLs. Refreshing an article replaces its stored content. A failed refresh preserves the last successful content and records the error.

Vector indexes are disposable snapshots identified by collection ID, embedding model, and chunk hashes. Queries open only the snapshot matching current SQLite content. Interrupted builds are retried with deterministic chunk IDs. A failed build never falls back to the previous collection or stale revision. Removing an article deletes its SQLite row and prunes that collection's vectors; the remaining articles are indexed again on the next query. Older snapshots are also pruned after a successful query.

This favors correctness and recovery for small personal collections over incremental indexing throughput. SQLite is the canonical copy, so back it up along with metadata. The old Stage 1 `chroma_db/` is not automatically imported: it contains no reliable collection ownership information. Re-add those URLs to a named collection.

This is a single-owner workspace, not a multi-tenant service. Collections isolate research topics, not users. Hosting must use access controls and durable storage where persistence across deployments is required.
