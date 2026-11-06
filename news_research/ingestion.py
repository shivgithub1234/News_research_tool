"""Extract article text without depending on the UI."""
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, FeatureNotFound
from langchain_core.documents import Document


@dataclass
class IngestionResult:
    documents: list[Document] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def extract_article_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(html, "html.parser")
    content = soup.find("article")
    if content is None:
        content = soup
    for tag in content(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    lines = content.get_text(separator="\n").splitlines()
    return "\n".join(line.strip() for line in lines if len(line.strip()) > 50)[:15000]


def load_articles(urls: list[str]) -> IngestionResult:
    result = IngestionResult()
    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })
        for raw_url in urls:
            url = raw_url.strip()
            try:
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    result.warnings.append(f"Invalid article URL: {url}")
                    continue
                response = session.get(url, timeout=15)
                response.raise_for_status()
                text = extract_article_text(response.text)
                if text:
                    result.documents.append(Document(page_content=text, metadata={"source": url}))
                else:
                    result.warnings.append(f"No readable article text found: {url}")
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                result.warnings.append(f"HTTP {status} while loading: {url}")
            except requests.RequestException:
                result.warnings.append(f"Could not load article (network error or timeout): {url}")
            except ValueError:
                result.warnings.append(f"Invalid article URL: {url}")
    return result
