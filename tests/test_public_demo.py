"""Runs in CI with only the lightweight public requirements installed."""
import os
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


class PublicDemoTests(unittest.TestCase):
    def test_recruiter_can_search_without_a_key_or_models(self):
        with patch.dict(os.environ, {"NEWS_RESEARCH_MODE": "demo", "GROQ_API_KEY": ""}):
            app = AppTest.from_file("main.py").run(timeout=30)
            self.assertFalse(app.exception)
            next(b for b in app.button if b.label == "Find evidence").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Harbor Bank" in h.value for h in app.subheader))
            self.assertFalse(any(b.label == "Process Articles" for b in app.button))
