#!/usr/bin/env python3
"""Echo as an error detector — triage demo.

Echo's routing value is negligible (see ECHO_ROUTER_DESIGN.md §4). Its
detection value is not: persona disagreement is a usable "check this answer"
signal. This reproduces that queue from a committed sweep.

    python scripts/echo_triage_demo.py [results/*.jsonl]

The "haiku was WRONG" column uses the answer key, which exists because this is
a benchmark. In deployment that column is what the human reviewer determines —
the system only produces the queue.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_SWEEP = RESULTS_DIR / "20260803T064926Z_mmlu_pro_n125.jsonl"

# echo-judge sub_calls: 3 = personas agreed, 4 = disagreed and escalated.
DISAGREED = 4
RANDOM_DRAWS = 1000
SEED = 7


def load_tasks(path: Path) -> list[tuple[str, dict]]:
    """task_id -> {arm: row}, dropping rows where the harness itself failed."""
    by: dict[str, dict] = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("_meta"):
            continue
        by[row["task_id"]][row["arm"]] = row
    return [
        (t, v)
        for t, v in by.items()
        if v.get("echo-judge", {}).get("sub_calls", 0) != 0
        and "haiku-only" in v
    ]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SWEEP
    if not path.exists():
        sys.exit(f"no such sweep: {path}")

    tasks = load_tasks(path)
    if not tasks:
        sys.exit(f"{path.name} has no task with both echo-judge and haiku-only")

    flagged = [(t, v) for t, v in tasks if v["echo-judge"]["sub_calls"] == DISAGREED]
    errors = {t for t, v in tasks if not v["haiku-only"]["passed"]}
    if not flagged:
        sys.exit(f"{path.name}: personas never disagreed — nothing to triage")

    print("=" * 66)
    print("ECHO REVIEW QUEUE — the answers Echo says to check")
    print("=" * 66)
    for t, v in flagged:
        wrong = not v["haiku-only"]["passed"]
        print(
            f"  {t:28s}  haiku was {'WRONG' if wrong else 'fine '}"
            f"   {'<-- caught a real error' if wrong else ''}"
        )

    hits = sum(1 for t, _ in flagged if t in errors)
    print(f"\n  queue length : {len(flagged)} of {len(tasks)} answers "
          f"({100 * len(flagged) / len(tasks):.0f}% of traffic)")
    print(f"  real errors  : {hits} of {len(flagged)} flagged were genuinely wrong "
          f"({100 * hits / len(flagged):.0f}%)")
    print(f"  coverage     : caught {hits} of {len(errors)} total errors "
          f"({100 * hits / len(errors):.0f}%)")

    # Same review budget, spent at random. The comparison that matters: a
    # detector is only worth building if it beats spot-checking.
    print("\n" + "=" * 66)
    print(f"IF YOU SPENT THE SAME REVIEW BUDGET AT RANDOM ({len(flagged)} answers)")
    print("=" * 66)
    rng = random.Random(SEED)
    ids = [t for t, _ in tasks]
    draws = [
        len(errors.intersection(rng.sample(ids, len(flagged))))
        for _ in range(RANDOM_DRAWS)
    ]
    mean = sum(draws) / len(draws)
    print(f"  errors found, average over {RANDOM_DRAWS} random draws : {mean:.1f}")
    print(f"  best random draw out of {RANDOM_DRAWS}                 : {max(draws)}")
    print(f"  Echo's queue                                 : {hits}")
    if mean:
        print(f"\n  -> Echo finds {hits / mean:.1f}x more errors for the same effort")

    # Precision varies sharply by subject — see ECHO_ROUTER_DESIGN.md §6.
    print("\n" + "=" * 66)
    print("PRECISION BY SUBJECT")
    print("=" * 66)
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for t, _ in flagged:
        cat = t.split("/")[1]
        per[cat][0] += 1
        per[cat][1] += t in errors
    for cat in sorted(per, key=lambda c: -per[c][1] / per[c][0]):
        n, real = per[cat]
        print(f"  {cat:12s} {n} flags, {real} real errors   {100 * real / n:3.0f}% precision")


if __name__ == "__main__":
    main()
