import os
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from news_research.config import Settings
from news_research.hosted import hosted_settings


class HostedTests(unittest.TestCase):
    def test_keys_isolate_collections_and_replace_server_key(self):
        base = Settings("server-test-key")
        first = hosted_settings(base, "visitor-one")
        second = hosted_settings(base, "visitor-two")
        self.assertNotEqual(first.data_directory, second.data_directory)
        self.assertEqual(first.data_directory, hosted_settings(base, "visitor-one").data_directory)
        self.assertEqual(first.groq_api_key, "visitor-one")
        self.assertNotIn("visitor-one", first.data_directory)

    def test_connect_opens_full_workspace_and_disconnect_clears_session(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "NEWS_RESEARCH_MODE": "hosted", "NEWS_RESEARCH_DATA_DIR": directory, "GROQ_API_KEY": "server-test-key"
        }), patch("news_research.hosted.Groq") as provider:
            app = AppTest.from_file("main.py").run(timeout=45)
            app.text_input[0].set_value("visitor-test-key")
            app.checkbox[0].check()
            app.button[0].click().run(timeout=45)
            self.assertFalse(app.exception)
            self.assertEqual(provider.call_args.kwargs["api_key"], "visitor-test-key")
            for label in ("Process Articles", "Ask", "Compare coverage"):
                self.assertTrue(any(b.label == label for b in app.button))
            next(b for b in app.button if b.label == "Disconnect workspace").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(b.label == "Open workspace" for b in app.button))
            self.assertNotIn("hosted_groq_key", app.session_state)
