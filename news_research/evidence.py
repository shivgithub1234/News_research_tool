"""Structured evidence validation. Exact quotes do not prove entailment."""
import hashlib
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    quote: str = Field(min_length=1, max_length=2000)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)
    citations: list[Citation] = Field(min_length=1, max_length=8)


class GroundedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_evidence"]
    claims: list[Claim] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def status_matches_claims(self):
        if (self.status == "answered") != bool(self.claims):
            raise ValueError("Answered responses need claims; abstentions must not contain claims.")
        return self


def evidence_map(documents):
    evidence = {}
    for doc in documents:
        ident = doc.metadata.get("chunk_id") or hashlib.sha256(
            (doc.metadata.get("source", "") + doc.page_content).encode()).hexdigest()
        evidence[ident] = doc
    return evidence


def normalize_quote(text):
    return " ".join(text.split())


def validate_citations(claims, evidence):
    for claim in claims:
        for citation in claim.citations:
            document = evidence.get(citation.evidence_id)
            if document is None:
                raise ValueError("Unknown evidence ID.")
            quote = normalize_quote(citation.quote)
            if not quote or quote not in normalize_quote(document.page_content):
                raise ValueError("Quoted passage does not occur in the cited evidence.")


def serialize_evidence(evidence):
    return [{"evidence_id": ident, "text": doc.page_content, "title": doc.metadata.get("title", ""),
             "publisher": doc.metadata.get("publisher", ""), "published": doc.metadata.get("published", ""),
             "source": doc.metadata.get("source", "")} for ident, doc in evidence.items()]
