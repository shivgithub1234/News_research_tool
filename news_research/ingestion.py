"""Extract readable article text and metadata from bounded public-web responses."""
from dataclasses import dataclass, field
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup, FeatureNotFound
from langchain_core.documents import Document
from news_research.fetching import fetch_html


@dataclass
class IngestionResult:
    documents: list[Document] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def extract_article_text(html):
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


def load_articles(urls):
    result = IngestionResult()
    for url in urls:
        url = url.strip()
        try:
            html = fetch_html(url)
            text = extract_article_text(html)
            if not text:
                result.warnings.append(f"No readable article text found: {url}")
                continue
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else url
            published = soup.find("meta", attrs={"property": "article:published_time"})
            publisher = soup.find("meta", attrs={"property": "og:site_name"})
            result.documents.append(Document(page_content=text, metadata={
                "source": url, "title": title[:200],
                "publisher": str(publisher.get("content", "") if publisher else urlparse(url).hostname)[:100],
                "published": str(published.get("content", "") if published else "")[:100],
            }))
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            result.warnings.append(f"HTTP {status} while loading: {url}")
        except requests.RequestException:
            result.warnings.append(f"Could not load article (network error or timeout): {url}")
        except ValueError as exc:
            result.warnings.append(f"Article rejected: {exc}")
    return result
