#!/usr/bin/env python3
"""Validate the open-ended equivalence grader before its verdicts are trusted.

The grader's error lands directly on the dependent variable, so an unvalidated
grader makes the whole format experiment unfalsifiable. Two modes:

``--sample N``
    Emit N grader decisions as a human-labelling worksheet. A human marks each
    row agree/disagree; ``--score`` then reports the grader's accuracy.
    This is the only mode that produces GROUND TRUTH.

``--cross-check MODEL``
    Re-grade the same decisions with a second model and report agreement.
    Cheap, but agreement is NOT correctness — two models can be wrong together
    (the same popularity trap that motivates the diversity-aggregator work).
    Treat a disagreement as a flag for human review, never treat agreement as
    validation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_decisions(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_worksheet(rows: list[dict], out: Path) -> None:
    """Emit a markdown worksheet for a human to mark up."""
    lines = [
        "# Grader validation worksheet",
        "",
        "For each item: is the grader's verdict CORRECT? Replace the `[ ]` with",
        "`[y]` if the grader got it right, `[n]` if it got it wrong.",
        "",
        "Then score with:",
        "```",
        f"python scripts/validate_grader.py {out} --score",
        "```",
        "",
        "---",
        "",
    ]
    for i, row in enumerate(rows, 1):
        verdict = row.get("verdict")
        verdict_str = {True: "EQUIVALENT", False: "DIFFERENT", None: "ERROR"}[verdict]
        lines += [
            f"## {i}. grader said: **{verdict_str}**",
            "",
            f"- **Question:** {str(row.get('question',''))[:400]}",
            f"- **Reference (gold):** `{row.get('gold','')}`",
            f"- **Candidate:** `{row.get('candidate','')}`",
            "",
            "grader correct? [ ]",
            "",
        ]
    out.write_text("\n".join(lines))


def score_worksheet(path: Path) -> dict:
    text = path.read_text()
    marks = []
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("grader correct?"):
            if "[y]" in stripped:
                marks.append(True)
            elif "[n]" in stripped:
                marks.append(False)
    if not marks:
        raise SystemExit("No completed [y]/[n] marks found — worksheet not filled in.")
    correct = sum(marks)
    return {
        "labelled": len(marks),
        "grader_correct": correct,
        "grader_accuracy": round(correct / len(marks), 4),
        "grader_error_rate": round(1 - correct / len(marks), 4),
    }


def cross_check(rows: list[dict], model_alias: str) -> dict:
    from benchmarks.mmlu_pro_open_arms import make_grader

    grader = make_grader(model_alias)
    agree = 0
    disagreements = []
    for row in rows:
        if row.get("verdict") is None:
            continue
        try:
            second = grader(row["candidate"], row["gold"], row["question"])
        except Exception as exc:  # noqa: BLE001
            disagreements.append({**row, "second": f"ERROR: {exc}"})
            continue
        if second == row["verdict"]:
            agree += 1
        else:
            disagreements.append({**row, "second": second})
    total = sum(1 for r in rows if r.get("verdict") is not None)
    return {
        "compared": total,
        "agreement": round(agree / total, 4) if total else None,
        "disagreements": disagreements,
        "note": "agreement is NOT correctness — use --sample for ground truth",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Validate the equivalence grader.")
    p.add_argument("decisions", type=Path,
                   help="*_grader_decisions.jsonl, or a worksheet .md with --score")
    p.add_argument("--sample", type=int, default=None,
                   help="Emit N decisions as a human-labelling worksheet.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score", action="store_true",
                   help="Score a filled-in worksheet.")
    p.add_argument("--cross-check", type=str, default=None,
                   help="Second grader model alias (weak signal, see module doc).")
    args = p.parse_args()

    if args.score:
        print(json.dumps(score_worksheet(args.decisions), indent=2))
        return

    rows = load_decisions(args.decisions)
    if not rows:
        raise SystemExit(f"No grader decisions in {args.decisions}")

    if args.sample:
        random.seed(args.seed)
        picked = random.sample(rows, min(args.sample, len(rows)))
        out = args.decisions.with_name(args.decisions.stem + "_worksheet.md")
        write_worksheet(picked, out)
        print(f"Wrote {len(picked)} items to {out}")
        print("Fill in the [ ] marks, then:")
        print(f"  python scripts/validate_grader.py {out} --score")
        return

    if args.cross_check:
        print(json.dumps(cross_check(rows, args.cross_check), indent=2))
        return

    tiers: dict[str, int] = {}
    for row in rows:
        key = str(row.get("verdict"))
        tiers[key] = tiers.get(key, 0) + 1
    print(json.dumps({"total_decisions": len(rows), "verdicts": tiers}, indent=2))


if __name__ == "__main__":
    main()
