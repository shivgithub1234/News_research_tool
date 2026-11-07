"""Application settings and environment validation."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = field(repr=False)
    llm_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: str = str(PROJECT_ROOT / "chroma_db")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    free_tier_confirmed: bool = field(default_factory=lambda: os.getenv("GROQ_FREE_TIER_CONFIRMED", "false").lower() == "true")
    data_directory: str = field(default_factory=lambda: os.getenv("NEWS_RESEARCH_DATA_DIR", str(PROJECT_ROOT / "data")))


def load_settings(require_api_key=True) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise ValueError("Missing GROQ_API_KEY. Add it to your environment or the project's .env file.")
    return Settings(groq_api_key=api_key)
