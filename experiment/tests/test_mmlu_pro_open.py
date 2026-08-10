"""Unit tests for the open-ended (options-stripped) MMLU-Pro arm.

No model calls. The grader is stubbed everywhere so these tests assert the
*deterministic* layer's behaviour — which is the layer that decides whether the
grader is even consulted, and therefore how much grader error can reach the
dependent variable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.mmlu_pro_open import (
    Verdict,
    extract_free_answer,
    format_prompt_open_ended,
    gold_form,
    normalize_free_text,
    numeric_value,
    row_to_task_open_ended,
    score_open_ended,
    sequence_tokens,
)


class TestPromptHasNoOptions(unittest.TestCase):
    """The whole experiment rests on the options being genuinely absent."""

    def test_prompt_omits_choices(self) -> None:
        prompt = format_prompt_open_ended("What is the capital of France?")
        self.assertIn("What is the capital of France?", prompt)
        self.assertNotIn("Choices:", prompt)
        self.assertNotIn("(A)", prompt)

    def test_prompt_requests_a_final_answer_line(self) -> None:
        prompt = format_prompt_open_ended("Q?")
        self.assertIn("Answer:", prompt)

    def test_task_carries_gold_text_not_letter(self) -> None:
        row = {
            "question_id": 7,
            "question": "2 + 2 = ?",
            "options": ["3", "4", "5"],
            "answer": "B",
        }
        task = row_to_task_open_ended("math", row)
        self.assertEqual(task["gold"], "4")
        self.assertEqual(task["gold_letter"], "B")
        self.assertEqual(task["format"], "open_ended")
        self.assertNotIn("Choices:", task["prompt"])

    def test_task_id_marks_the_format(self) -> None:
        """MCQ and open-ended rows must not collide in a merged JSONL."""
        row = {"question_id": 7, "question": "q", "options": ["a", "b"], "answer": "A"}
        task = row_to_task_open_ended("math", row)
        self.assertIn("open", task["task_id"])


class TestExtractFreeAnswer(unittest.TestCase):
    def test_plain_answer_line(self) -> None:
        self.assertEqual(extract_free_answer("reasoning\nAnswer: Paris"), "Paris")

    def test_latest_answer_wins(self) -> None:
        text = "Answer: London\nwait, reconsidering\nAnswer: Paris"
        self.assertEqual(extract_free_answer(text), "Paris")

    def test_boxed_form(self) -> None:
        """Math-tuned models emit \\boxed{} instead of an Answer: line."""
        self.assertEqual(extract_free_answer(r"so \boxed{42}"), "42")

    def test_falls_back_to_last_nonempty_line(self) -> None:
        self.assertEqual(extract_free_answer("some reasoning\n\n42"), "42")

    def test_empty_output_is_none(self) -> None:
        self.assertIsNone(extract_free_answer("   "))

    def test_strips_trailing_punctuation(self) -> None:
        self.assertEqual(extract_free_answer("Answer: Paris."), "Paris")


class TestNormalizeFreeText(unittest.TestCase):
    def test_case_and_whitespace(self) -> None:
        self.assertEqual(normalize_free_text("  The   Answer "), "answer")

    def test_strips_leading_articles(self) -> None:
        self.assertEqual(normalize_free_text("a dog"), "dog")

    def test_strips_punctuation(self) -> None:
        self.assertEqual(normalize_free_text("Paris!"), "paris")


class TestNumericValue(unittest.TestCase):
    def test_plain_int(self) -> None:
        self.assertEqual(numeric_value("42"), 42.0)

    def test_with_units_is_not_numeric(self) -> None:
        self.assertIsNone(numeric_value("42 metres"))

    def test_comma_thousands(self) -> None:
        self.assertEqual(numeric_value("1,024"), 1024.0)

    def test_scientific(self) -> None:
        self.assertEqual(numeric_value("1.5e3"), 1500.0)

    def test_non_numeric(self) -> None:
        self.assertIsNone(numeric_value("Paris"))


def _explode(*_a, **_k):  # pragma: no cover - guard
    raise AssertionError("grader must not be called for a deterministic case")


class TestScoringTiers(unittest.TestCase):
    """Tier 1 is deterministic; only genuine ambiguity may reach the grader."""

    def _task(self, gold: str) -> dict:
        return {"gold": gold, "question": "q", "format": "open_ended"}

    def test_exact_match_never_calls_grader(self) -> None:
        v = score_open_ended("Answer: Paris", self._task("Paris"), grader=_explode)
        self.assertTrue(v.passed)
        self.assertEqual(v.tier, "exact")

    def test_normalized_match_never_calls_grader(self) -> None:
        v = score_open_ended("Answer: the  PARIS.", self._task("Paris"), grader=_explode)
        self.assertTrue(v.passed)
        self.assertEqual(v.tier, "exact")

    def test_numeric_match_never_calls_grader(self) -> None:
        v = score_open_ended("Answer: 1,024", self._task("1024"), grader=_explode)
        self.assertTrue(v.passed)
        self.assertEqual(v.tier, "numeric")

    def test_numeric_mismatch_never_calls_grader(self) -> None:
        """Two well-formed numbers that differ are decidable without a model."""
        v = score_open_ended("Answer: 41", self._task("42"), grader=_explode)
        self.assertFalse(v.passed)
        self.assertEqual(v.tier, "numeric")

    def test_unparseable_never_calls_grader(self) -> None:
        v = score_open_ended("", self._task("42"), grader=_explode)
        self.assertFalse(v.passed)
        self.assertEqual(v.tier, "unparseable")

    def test_ambiguous_text_defers_to_grader(self) -> None:
        calls: list[tuple[str, str]] = []

        def grader(candidate: str, gold: str, question: str) -> bool:
            calls.append((candidate, gold))
            return True

        v = score_open_ended(
            "Answer: the city of Paris, France",
            self._task("Paris"),
            grader=grader,
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.tier, "grader")
        self.assertEqual(len(calls), 1)

    def test_grader_refusal_is_a_fail_not_a_crash(self) -> None:
        def grader(*_a, **_k) -> bool:
            raise RuntimeError("grader exploded")

        v = score_open_ended("Answer: something", self._task("Paris"), grader=grader)
        self.assertFalse(v.passed)
        self.assertEqual(v.tier, "grader_error")

    def test_absent_grader_marks_undecided_rather_than_guessing(self) -> None:
        """No grader must not silently score as a fail — that biases the result."""
        v = score_open_ended("Answer: the city of Paris", self._task("Paris"), grader=None)
        self.assertEqual(v.tier, "undecided")
        self.assertIsNone(v.passed)


class TestOrderedTupleRegression(unittest.TestCase):
    """Pins a live grader failure found on the first two smoke-test calls.

    Gold "False, True" vs candidate "True, False" was graded EQUIVALENT — the
    exact opposite answer scored as a pass, inflating the arm. Ordered tuples
    must be decided deterministically, never by the grader.
    """

    def _task(self, gold: str) -> dict:
        return {"gold": gold, "question": "q", "format": "open_ended"}

    def test_reversed_tuple_is_a_fail_without_the_grader(self) -> None:
        v = score_open_ended("Answer: True, False", self._task("False, True"), grader=_explode)
        self.assertFalse(v.passed)
        self.assertEqual(v.tier, "sequence")

    def test_matching_tuple_passes_without_the_grader(self) -> None:
        """An identical tuple resolves at the `exact` tier, before `sequence`.

        The invariant under test is deterministic resolution, not which of the
        two deterministic tiers claims it.
        """
        v = score_open_ended("Answer: False, True", self._task("False, True"), grader=_explode)
        self.assertTrue(v.passed)
        self.assertIn(v.tier, ("exact", "sequence"))

    def test_tuple_match_is_case_insensitive(self) -> None:
        v = score_open_ended("Answer: false, TRUE", self._task("False, True"), grader=_explode)
        self.assertTrue(v.passed)
        self.assertIn(v.tier, ("exact", "sequence"))

    def test_spacing_variation_still_deterministic(self) -> None:
        """'False,True' (no space) must not fall through to the grader."""
        v = score_open_ended("Answer: False,True", self._task("False, True"), grader=_explode)
        self.assertTrue(v.passed)
        self.assertIn(v.tier, ("exact", "sequence"))

    def test_prose_answer_still_reaches_the_grader(self) -> None:
        """'Statement 1 is true...' is prose, not a tuple — genuinely ambiguous."""
        seen: list[str] = []

        def grader(candidate: str, gold: str, question: str) -> bool:
            seen.append(candidate)
            return False

        v = score_open_ended(
            "Answer: Statement 1 is true, Statement 2 is false",
            self._task("False, True"),
            grader=grader,
        )
        self.assertEqual(v.tier, "grader")
        self.assertEqual(len(seen), 1)

    def test_long_lists_are_not_treated_as_tuples(self) -> None:
        self.assertIsNone(sequence_tokens("a, b, c, d, e"))

    def test_multiword_parts_are_not_tuples(self) -> None:
        self.assertIsNone(sequence_tokens("the first one, some much longer phrase here"))


class TestGoldForm(unittest.TestCase):
    """Stratification key — tuple golds lose their answer-form convention when
    the options are stripped, so they must stay separable in the analysis."""

    def test_tuple(self) -> None:
        self.assertEqual(gold_form("False, True"), "ordered_tuple")

    def test_numeric(self) -> None:
        self.assertEqual(gold_form("42"), "numeric")

    def test_single_value(self) -> None:
        self.assertEqual(gold_form("Paris"), "single_value")


class TestVerdictAccounting(unittest.TestCase):
    """Every row must self-report which layer decided it."""

    def test_verdict_exposes_tier_and_detail(self) -> None:
        v = score_open_ended("Answer: Paris", {"gold": "Paris", "question": "q"}, grader=None)
        self.assertIsInstance(v, Verdict)
        self.assertTrue(v.passed)
        self.assertTrue(v.detail)


if __name__ == "__main__":
    unittest.main()
