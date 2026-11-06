# My News Research Tool

A Streamlit retrieval-augmented generation (RAG) app for asking questions about news articles.

[Hosted app](https://mynewsresearchtool.streamlit.app/)

## Current capabilities

- Accept up to three article URLs and extract text using Requests and BeautifulSoup.
- Split text into 1,000-character chunks with 200-character overlap.
- Compute embeddings locally with `sentence-transformers/all-MiniLM-L6-v2`.
- Store embeddings in Chroma and retrieve four chunks per question.
- Generate answers through Groq using `openai/gpt-oss-20b` and display retrieved source URLs.

The prompt asks the model to use only the retrieved context. This does not guarantee correct answers or prevent hallucinations. Source links identify retrieved articles, not verified claim-level citations.

## Setup (Python 3.12)

```sh
git clone https://github.com/shivgithub1234/News_research_tool.git
cd News_research_tool
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS/Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` to your Groq API key. An existing environment variable takes precedence over `.env`. Do not commit credentials.

**Only the Groq API key is required.** Hugging Face authentication is not required for the public embedding model. The first indexing operation downloads model weights and needs internet access; embeddings are computed locally. Article fetching and Groq generation also require network access.

Run on Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Run on macOS/Linux:

```sh
.venv/bin/python -m streamlit run main.py
```

Enter URLs, click **Process Articles**, enter a question, and click **Ask**. The embedding model is loaded lazily and cached across Streamlit reruns. Questions are sent only when the form is submitted.

## Project structure

```text
main.py                      Streamlit entry point
news_research/
  config.py                  Environment validation and pipeline settings
  ingestion.py               HTTP fetching and article extraction
  retrieval.py               Embeddings, chunking, indexing, and retrieval
  generation.py              Prompt, Groq invocation, and answer sources
  ui.py                      Streamlit controls and session state
tests/                       Offline regression tests
.env.example                 Environment variable template
requirements.txt             Runtime dependencies
```

## Tests

After installing dependencies, run with your virtual environment's Python:

```sh
python -m unittest discover -s tests -v
```

Tests use mocked network/model boundaries and require neither credentials nor model downloads.

## Current limitations and next stages

- Extraction can fail for paywalls, JavaScript-rendered pages, or blocked requests. It drops short lines and caps each article at 15,000 characters.
- The app uses one local Chroma directory. It does not yet isolate research collections, deduplicate articles, or automatically reconnect a new UI session to an existing index. Previously indexed content can remain in the shared index.
- Basic URL syntax validation is not a complete network access policy. Harden URL fetching before treating the app as a public multi-user service.
- Retrieval quality, answer support, and abstention are not yet benchmarked.
- Planned extensions: evaluation baseline, saved collections, evidence-linked citations, retrieval experiments, and news coverage comparison.
