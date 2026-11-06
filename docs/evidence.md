# Evidence-backed answers

Retrieved chunks receive stable IDs. The model returns structured claims with evidence IDs and verbatim quotes, or an explicit insufficient-evidence response. The application rejects unknown IDs, missing citations, invalid schemas, and quotes that do not appear in the referenced passage. Source URLs and article metadata come from the stored documents, never from generated URLs.

The UI numbers claim citations and exposes their quote, full retrieved passage, publisher, publication date when available, and original URL. Unknown dates are not invented. Failed validation is a separate status from insufficient evidence; neither is counted as a supported answer.

Exact quote matching **does not verify semantic support**. A model can cite a genuine sentence while misinterpreting it. Human review of the claim/passage relationship remains necessary, and no vector similarity is displayed as a probability of correctness. Prompt injection defenses are best-effort: system instructions treat article content as data, structured output restricts the response, and source checks reject fabricated references.

Each submitted question makes at most one Groq request (no automatic retries or provider fallback), with a 30-second timeout and a 1,500-token output ceiling. Returned token usage and generation duration are shown. Offline tests use mocked model responses and test the contract; they do not measure live model answer quality. No automated paid generation is performed.
