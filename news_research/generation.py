"""Generate answers from retrieved article passages."""
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from news_research.config import Settings


@dataclass
class Answer:
    text: str
    sources: list[str]


def generate_answer(question: str, documents: list[Document], settings: Settings) -> Answer:
    if not documents:
        return Answer("No relevant information found in the articles.", [])
    prompt = ChatPromptTemplate.from_template(
        """Answer the question ONLY using the context below.
If the answer is not present, say so clearly.

Context:
{context}

Question:
{question}
"""
    )
    llm = ChatGroq(api_key=settings.groq_api_key, model_name=settings.llm_model, temperature=0.2)
    chain = prompt | llm | StrOutputParser()
    text = chain.invoke({
        "context": "\n\n".join(doc.page_content for doc in documents),
        "question": question,
    })
    sources = list(dict.fromkeys(
        doc.metadata["source"] for doc in documents if doc.metadata.get("source")
    ))
    return Answer(text, sources)
