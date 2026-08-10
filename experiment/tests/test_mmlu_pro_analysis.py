"""Tests for MMLU-Pro advanced sweep analysis."""

from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "analyze_mmlu_pro",
    ROOT / "scripts" / "analyze_mmlu_pro.py",
)
assert spec is not None and spec.loader is not None
analyze_mmlu_pro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_mmlu_pro)

by_arm = analyze_mmlu_pro.by_arm
by_category_arm = analyze_mmlu_pro.by_category_arm
build_markdown_report = analyze_mmlu_pro.build_markdown_report
category_for = analyze_mmlu_pro.category_for
oracle_diagnostics = analyze_mmlu_pro.oracle_diagnostics
sonnet_gaps = analyze_mmlu_pro.sonnet_gaps
specialist_recommendations = analyze_mmlu_pro.specialist_recommendations


def row(
    task_id: str,
    arm: str,
    passed: bool,
    sub_calls: int,
    *,
    category: str | None = None,
    detail: str = "passed",
) -> dict:
    data = {
        "task_id": task_id,
        "arm": arm,
        "passed": passed,
        "detail": detail,
        "wall_seconds": 1.0,
        "sub_calls": sub_calls,
        "benchmark": "mmlu_pro",
    }
    if category is not None:
        data["category"] = category
    return data


class TestMmluProAnalysis(unittest.TestCase):
    def test_category_for_prefers_row_category(self) -> None:
        self.assertEqual(
            category_for(row("mmlu_pro/math/1", "haiku-only", True, 1, category="math")),
            "math",
        )

    def test_category_for_falls_back_to_task_id(self) -> None:
        self.assertEqual(
            category_for(row("mmlu_pro/computer_science/1", "haiku-only", True, 1)),
            "computer science",
        )

    def test_by_arm_summary(self) -> None:
        rows = [
            row("mmlu_pro/math/1", "haiku-only", True, 1),
            row("mmlu_pro/math/2", "haiku-only", False, 1, detail="expected B got A"),
        ]
        summary = by_arm(rows)["haiku-only"]
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["failures"], 1)
        self.assertIn("expected B got A", summary["top_failure_details"])

    def test_sonnet_gaps_by_category(self) -> None:
        rows = [
            row("mmlu_pro/math/1", "sonnet-only", True, 1, category="math"),
            row("mmlu_pro/math/2", "sonnet-only", True, 1, category="math"),
            row("mmlu_pro/math/1", "haiku-only", True, 1, category="math"),
            row("mmlu_pro/math/2", "haiku-only", False, 1, category="math"),
        ]
        summaries = by_category_arm(rows)
        self.assertEqual(summaries[("math", "haiku-only")]["pass_rate"], 0.5)
        self.assertEqual(sonnet_gaps(rows)[("math", "haiku-only")], -0.5)

    def test_oracle_diagnostics_false_accept_and_escalate(self) -> None:
        rows = [
            # Task 1: oracle escalates, judge accepts -> false accept.
            row("mmlu_pro/math/1", "echo-oracle", True, 3),
            row("mmlu_pro/math/1", "echo-judge", False, 3),
            # Task 2: oracle accepts, judge escalates -> false escalation.
            row("mmlu_pro/math/2", "echo-oracle", True, 2),
            row("mmlu_pro/math/2", "echo-judge", True, 4),
            # Task 3: both accept -> aligned.
            row("mmlu_pro/math/3", "echo-oracle", True, 2),
            row("mmlu_pro/math/3", "echo-judge", True, 3),
        ]
        diag = oracle_diagnostics(rows)["echo-judge"]
        self.assertEqual(diag["n"], 3)
        self.assertAlmostEqual(diag["oracle_alignment"], 1 / 3)
        self.assertAlmostEqual(diag["false_accept_rate"], 1 / 3)
        self.assertAlmostEqual(diag["false_escalation_rate"], 1 / 3)
        self.assertEqual(diag["false_accept_tasks"], ["mmlu_pro/math/1"])
        self.assertEqual(diag["false_escalation_tasks"], ["mmlu_pro/math/2"])

    def test_specialist_recommendations_flag_weak_category(self) -> None:
        rows = [
            row("mmlu_pro/math/1", "sonnet-only", True, 1, category="math"),
            row("mmlu_pro/math/2", "sonnet-only", True, 1, category="math"),
            row("mmlu_pro/math/1", "haiku-only", True, 1, category="math"),
            row("mmlu_pro/math/2", "haiku-only", False, 1, category="math"),
            row("mmlu_pro/math/1", "echo-judge", True, 3, category="math"),
            row("mmlu_pro/math/2", "echo-judge", False, 3, category="math"),
        ]

        rec = specialist_recommendations(rows)["math"]

        self.assertTrue(rec["needs_specialist_probe"])
        self.assertEqual(rec["specialist_type"], "math-reasoning specialist")
        self.assertEqual(rec["best_echo_arm"], "echo-judge")
        self.assertIn("below sonnet", rec["reason"])

    def test_markdown_report_includes_specialist_section(self) -> None:
        rows = [
            row("mmlu_pro/math/1", "sonnet-only", True, 1, category="math"),
            row("mmlu_pro/math/1", "haiku-only", False, 1, category="math"),
            row("mmlu_pro/math/1", "echo-judge", False, 3, category="math"),
        ]
        category = by_category_arm(rows)
        gaps = sonnet_gaps(rows)
        report = build_markdown_report(
            source=Path("results/example_mmlu_pro_n1.jsonl"),
            meta={"_meta": True, "benchmark": "mmlu_pro"},
            overall=by_arm(rows),
            category=category,
            gaps=gaps,
            oracle={},
            specialists=specialist_recommendations(rows),
        )

        self.assertIn("# MMLU-Pro Echo Analysis", report)
        self.assertIn("## Specialist Routing Probes", report)
        self.assertIn("math-reasoning specialist", report)
        self.assertIn("## Interpretation Guide", report)


if __name__ == "__main__":
    unittest.main()
