import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_retrieval import format_context


class TestKnowledgeRetrieval(unittest.TestCase):
    def test_formats_numbered_passages_with_sources(self):
        text = format_context([{
            "title": "Unification (computer science)",
            "url": "https://example.test/unification",
            "text": "Unification solves equations between symbolic expressions.",
        }])
        self.assertIn("[1] Unification (computer science)", text)
        self.assertIn("https://example.test/unification", text)

    def test_empty_search_is_explicit(self):
        self.assertEqual(
            format_context([]), "No relevant reference material was found."
        )


if __name__ == "__main__":
    unittest.main()
