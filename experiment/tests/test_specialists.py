"""Specialist registry + cost accounting (no model calls)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.specialists import (  # noqa: E402
    DEFAULT_SPECIALIST_MAP,
    has_specialist,
    specialist_map,
    specialist_tag_for,
)
from cost_units import HAIKU_PERSONA, SONNET_PERSONA, cost_units, escalated  # noqa: E402


class TestSpecialistRegistry(unittest.TestCase):
    def test_registered_categories_resolve(self):
        for category in ("law", "math", "computer science"):
            self.assertIsNotNone(specialist_tag_for(category))
            self.assertTrue(has_specialist(category))

    def test_categories_without_specialists(self):
        # Deliberately absent — no credible open specialist exists.
        for category in ("chemistry", "philosophy", "history"):
            self.assertIsNone(specialist_tag_for(category))
            self.assertFalse(has_specialist(category))

    def test_none_and_blank_category(self):
        self.assertIsNone(specialist_tag_for(None))
        self.assertIsNone(specialist_tag_for(""))

    def test_lookup_is_case_insensitive(self):
        self.assertEqual(specialist_tag_for("LAW"), DEFAULT_SPECIALIST_MAP["law"])
        self.assertEqual(specialist_tag_for("  Law  "), DEFAULT_SPECIALIST_MAP["law"])

    def test_env_override_merges_over_defaults(self):
        with mock.patch.dict(
            os.environ, {"ECHO_SPECIALIST_MAP": '{"law": "custom:tag", "health": "med:7b"}'}
        ):
            mapping = specialist_map()
            self.assertEqual(mapping["law"], "custom:tag")
            self.assertEqual(mapping["health"], "med:7b")
            # untouched default survives
            self.assertEqual(mapping["math"], DEFAULT_SPECIALIST_MAP["math"])

    def test_env_override_rejects_bad_json(self):
        with mock.patch.dict(os.environ, {"ECHO_SPECIALIST_MAP": "not json"}):
            with self.assertRaises(ValueError):
                specialist_map()

    def test_env_override_rejects_non_object(self):
        with mock.patch.dict(os.environ, {"ECHO_SPECIALIST_MAP": '["law"]'}):
            with self.assertRaises(ValueError):
                specialist_map()


class TestSpecialistCostUnits(unittest.TestCase):
    def test_specialist_only_is_free(self):
        self.assertEqual(cost_units("specialist-only", 1), 0.0)

    def test_specialist_only_haiku_fallback_costs_one(self):
        self.assertAlmostEqual(cost_units("specialist-only", 2), HAIKU_PERSONA)

    def test_echo_specialist_accept_path(self):
        self.assertAlmostEqual(cost_units("echo-specialist", 2), 2 * HAIKU_PERSONA)

    def test_echo_specialist_tiebreak_is_free(self):
        # Specialist resolved the disagreement locally — no Sonnet spend.
        self.assertAlmostEqual(cost_units("echo-specialist", 3), 2 * HAIKU_PERSONA)

    def test_echo_specialist_escalation_pays_sonnet(self):
        self.assertAlmostEqual(
            cost_units("echo-specialist", 4), 2 * HAIKU_PERSONA + SONNET_PERSONA
        )

    def test_tiebreak_is_not_counted_as_escalation(self):
        self.assertFalse(escalated("echo-specialist", 2))
        self.assertFalse(escalated("echo-specialist", 3))
        self.assertTrue(escalated("echo-specialist", 4))

    def test_tiebreak_beats_plain_echo_on_cost(self):
        # The whole point: a resolved tie costs 2.0 instead of echo-lexical's 5.0.
        self.assertLess(
            cost_units("echo-specialist", 3), cost_units("echo-lexical", 3)
        )

    def test_echo_specialist_never_dearer_than_sonnet_on_accept(self):
        self.assertLess(cost_units("echo-specialist", 3), SONNET_PERSONA)


class TestSpecialistArmsRegistered(unittest.TestCase):
    def test_arms_present(self):
        from benchmarks.bbh_arms import BBH_ARMS

        self.assertIn("specialist-only", BBH_ARMS)
        self.assertIn("echo-specialist", BBH_ARMS)

    def test_arms_available_to_mmlu_pro(self):
        import importlib.util

        runner = Path(__file__).resolve().parent.parent / "scripts" / "run_mmlu_pro_pilot.py"
        spec = importlib.util.spec_from_file_location("_mmlu_runner", runner)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertIn("specialist-only", module.MMLU_PRO_ARMS)
        self.assertIn("echo-specialist", module.MMLU_PRO_ARMS)


if __name__ == "__main__":
    unittest.main()
