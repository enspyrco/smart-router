import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harnesses import ANSWER_CONTRACT, H0


class TestAnswerContract(unittest.TestCase):
    def test_requires_answer_before_reasoning(self):
        self.assertIn("Begin your reply", ANSWER_CONTRACT)
        self.assertLess(ANSWER_CONTRACT.index("Answer: X"),
                        ANSWER_CONTRACT.index("reasoning"))

    def test_limits_reasoning_length(self):
        self.assertIn("eight concise sentences", ANSWER_CONTRACT)

    def test_h0_requests_answer_first(self):
        self.assertLess(H0.system.index("final answer"),
                        H0.system.index("brief reasoning"))


if __name__ == "__main__":
    unittest.main()
