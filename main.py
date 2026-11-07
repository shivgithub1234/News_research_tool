"""Run with: python -m streamlit run main.py."""
import logging
from news_research.app import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
