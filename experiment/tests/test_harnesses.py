import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harnesses import H0, H1, H2_MCP, HARNESSES, H_ANSWER_FIRST


class TestAnswerContract(unittest.TestCase):
    def test_h0_retains_original_answer_last_contract(self):
        self.assertIn("End your reply", H0.answer_contract)
        self.assertLess(H0.system.index("reasoning"),
                        H0.system.index("final answer"))

    def test_h1_limits_reasoning_and_keeps_answer_last(self):
        self.assertIn("eight concise sentences", H1.answer_contract)
        self.assertIn("Then end your reply", H1.answer_contract)
        self.assertLess(H1.system.index("brief reasoning"),
                        H1.system.index("final answer"))

    def test_answer_first_experiment_is_preserved_separately(self):
        self.assertIn("Begin your reply", H_ANSWER_FIRST.answer_contract)
        self.assertIn("H-answer-first", HARNESSES)

    def test_all_variants_are_registered(self):
        self.assertEqual(
            set(HARNESSES), {"H0", "H1", "H2-MCP", "H-answer-first"}
        )

    def test_mcp_variant_is_clearly_open_book(self):
        self.assertIn("Open-book", H2_MCP.notes)


if __name__ == "__main__":
    unittest.main()
