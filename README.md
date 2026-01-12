# 📰 My News Research Tool

My News Research Tool is a **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions directly from **real news articles** and receive **accurate, source-grounded answers**.

Deployed url: https://mynewsresearchtool.streamlit.app/

Unlike traditional summarizers, this tool:
- ❌ does **not** auto-summarize articles  
- ❌ does **not** hallucinate information  
- ✅ answers questions **only using retrieved content** 

---

## ✨ Features

- 🔗 Input multiple news article URLs  
- 🧹 Clean article extraction using **BeautifulSoup**  
- ✂️ Intelligent text chunking with overlap  
- 🧠 Semantic search using **ChromaDB**  
- 🤖 Fast LLM inference via **Groq (LLaMA-3.1-8B-Instant)**  
- 📚 Answers grounded strictly in retrieved articles  
- 🔍 Displays **source URLs** for each answer  
- 💻 Fully compatible with **Windows + Python 3.12**

---

## 🧠 How It Works

News URLs
↓
BeautifulSoup Scraper
↓
Clean Article Text
↓
Text Chunking
↓
Local Embeddings
↓
Chroma Vector DB
↓
Retriever
↓
Groq LLM
↓
Answer + Source Links

yaml
Copy code

This pipeline ensures **trustworthy and traceable** answers.

---

## 🛠 Tech Stack

- **Frontend:** Streamlit  
- **LLM:** Groq (LLaMA-3.1-8B-Instant)  
- **Framework:** LangChain (Runnable API)  
- **Embeddings:** Sentence-Transformers (local)  
- **Vector Database:** ChromaDB  
- **Web Scraping:** BeautifulSoup + Requests  
- **Language:** Python 3.12  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd my-news-research-tool
2️⃣ Create & Activate Virtual Environment
Git Bash

bash
Copy code
python -m venv .venv
source .venv/Scripts/activate
PowerShell

powershell
Copy code
python -m venv .venv
.venv\Scripts\Activate.ps1
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Set Environment Variables
Create a .env file:

env
Copy code
GROQ_API_KEY=your_groq_api_key_here
Hugging Face token is not required since embeddings are computed locally.

▶️ Run the Application
bash
Copy code
python -m streamlit run main.py
The application will open automatically in your browser.

🧪 Example Questions
How will interest rate changes affect banks?

What risks are highlighted for equity markets?

What are analysts saying about inflation?

Each answer includes clickable source URLs.

🔐 Why This Tool Is Reliable
✔ Retrieval happens before generation

✔ LLM answers are restricted to retrieved context

✔ Sources come from actual documents

❌ No auto-summarization

❌ No fabricated citations

This follows industry best practices for RAG systems.

📁 Project Structure
bash
Copy code
my-news-research-tool/
│
├── main.py
├── requirements.txt
├── README.md
├── .env
├── chroma_db/
└── .venv/
🌱 Future Improvements
Persist & reload ChromaDB automatically

Highlight supporting text snippets

RSS-based live news ingestion

Confidence scoring for answers

Deployment on Streamlit Cloud
