"""Audit Echo JSONL runs for routing behaviour.

The pilot harness writes one JSON object per task/arm. This script summarizes
the fields that matter for the core routing assumption:

  - how often Echo escalates
  - how often it accepts the cheap pair
  - how often that cheap accept is wrong
  - how often an escalation was unnecessary because a cheap candidate passed

It also supports older result files that only logged ``sub_calls`` by inferring
Echo decisions from the call count convention used by run_pilot.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULTS_DIR = Path(__file__).parent / "results"


def _is_echo(row: dict[str, Any]) -> bool:
    return str(row.get("arm", "")).startswith("echo-")


def _escalated(row: dict[str, Any]) -> bool:
    if "escalated" in row:
        return bool(row["escalated"])
    if not _is_echo(row):
        return False
    threshold = 3 if row.get("arm") == "echo-judge" else 2
    return int(row.get("sub_calls", 0)) > threshold


def _accepted_cheap(row: dict[str, Any]) -> bool:
    if "accepted_cheap" in row:
        return bool(row["accepted_cheap"])
    if not _is_echo(row):
        return False
    if int(row.get("sub_calls", 0)) <= 0:
        return False
    return not _escalated(row)


def _unnecessary_escalation(row: dict[str, Any]) -> bool:
    if "unnecessary_escalation" in row:
        return bool(row["unnecessary_escalation"])
    if not _escalated(row):
        return False
    return bool(row.get("cheap_a_passed") or row.get("cheap_b_passed"))


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source"] = path.name
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        n = len(arm_rows)
        passed = sum(1 for row in arm_rows if row.get("passed"))
        escalated = sum(1 for row in arm_rows if _escalated(row))
        accepted = sum(1 for row in arm_rows if _accepted_cheap(row))
        false_accepts = sum(
            1
            for row in arm_rows
            if _accepted_cheap(row) and not row.get("passed")
        )
        unnecessary_escalations = sum(
            1 for row in arm_rows if _unnecessary_escalation(row)
        )
        summary[arm] = {
            "n": n,
            "pass_rate": round(passed / n, 3) if n else None,
            "escalation_rate": round(escalated / n, 3) if n else None,
            "accepted_cheap_rate": round(accepted / n, 3) if n else None,
            "false_accept_rate": round(false_accepts / n, 3) if n else None,
            "unnecessary_escalation_rate": round(unnecessary_escalations / n, 3) if n else None,
            "mean_sub_calls": round(
                sum(int(row.get("sub_calls", 0)) for row in arm_rows) / n, 2
            ) if n else None,
        }
    return summary


def false_accept_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _accepted_cheap(row) and not row.get("passed")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Echo result JSONL files.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Result JSONL files. Defaults to experiment/results/*.jsonl.",
    )
    args = parser.parse_args()

    paths = args.paths or sorted(RESULTS_DIR.glob("*.jsonl"))
    rows = load_rows(paths)
    print(json.dumps(summarize(rows), indent=2))

    misses = false_accept_rows(rows)
    if misses:
        print("\nFalse accepts:")
        for row in misses:
            print(
                f"  {row['_source']} {row['task_id']} {row['arm']} "
                f"calls={row.get('sub_calls')} detail={row.get('detail')}"
            )


if __name__ == "__main__":
    main()
