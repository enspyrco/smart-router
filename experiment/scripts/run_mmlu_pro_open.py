#!/usr/bin/env python3
"""MMLU-Pro OPEN-ENDED sweep — the format arm.

Same questions, same subjects, options removed. Everything else held constant,
so a difference between this sweep and the MCQ sweep isolates FORMAT.

Pair it with an MCQ run over the identical slice:

    python scripts/run_mmlu_pro_pilot.py --n-per-category 25 \\
        --arms haiku-only,sonnet-only
    python scripts/run_mmlu_pro_open.py  --n-per-category 25

Each row records which scoring TIER decided it. Rows decided by the grader are
the only model-dependent ones; ``--exclude-grader-tier`` in the analyzer
recomputes the table without them, which is the sensitivity check that says
whether the headline is signal or grader artefact.

Results stream to disk with flush-after-each-row; ``--resume`` skips completed
(task_id, arm) pairs. Max usage exhaustion aborts with exit code 2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# run_mmlu_pro_pilot lives beside this file, not in the package root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.mmlu_pro import PILOT_CATEGORIES
from benchmarks.mmlu_pro_open import load_mmlu_pro_open, score_open_ended
from benchmarks.mmlu_pro_open_arms import OPEN_ARMS, make_recording_grader
from run_mmlu_pro_pilot import UsageLimitReached, _is_usage_exhaustion
from run_pilot import RESULTS_DIR

BENCHMARK = "mmlu_pro_open"


def run_one_open(task: dict, arm_name: str, arm_fn, grader) -> dict:
    t0 = time.perf_counter()
    try:
        output, sub_calls = arm_fn(task)
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        if _is_usage_exhaustion(str(e)):
            raise UsageLimitReached(msg) from e
        return {
            "task_id": task["task_id"], "arm": arm_name, "passed": False,
            "detail": msg, "wall_seconds": time.perf_counter() - t0,
            "sub_calls": 0, "tier": "arm_error", "candidate": None,
            "category": task["category"], "gold": task["gold"],
        }

    verdict = score_open_ended(output, task, grader=grader)
    return {
        "task_id": task["task_id"], "arm": arm_name, "passed": verdict.passed,
        "detail": verdict.detail, "wall_seconds": time.perf_counter() - t0,
        "sub_calls": sub_calls, "tier": verdict.tier,
        "candidate": verdict.candidate, "category": task["category"],
        "gold": task["gold"],
    }


def summarize_open(rows: list[dict]) -> dict:
    """Per-arm and per-category summary that keeps the tiers visible.

    ``pass_rate`` counts only DECIDED rows (undecided excluded from both
    numerator and denominator) — an undecided row is not evidence either way,
    and folding it in as a fail would penalise categories by their own
    ambiguity rate.
    """
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("_meta"):
            by_arm[row["arm"]].append(row)

    out: dict = {}
    for arm, arm_rows in sorted(by_arm.items()):
        decided = [r for r in arm_rows if r["passed"] is not None]
        passed = [r for r in decided if r["passed"]]
        tiers: dict[str, int] = defaultdict(int)
        for r in arm_rows:
            tiers[r["tier"]] += 1

        by_cat: dict[str, dict] = {}
        for cat in sorted({r["category"] for r in arm_rows}):
            cat_rows = [r for r in arm_rows if r["category"] == cat]
            cat_decided = [r for r in cat_rows if r["passed"] is not None]
            by_cat[cat] = {
                "n": len(cat_rows),
                "decided": len(cat_decided),
                "pass_rate": (
                    round(sum(1 for r in cat_decided if r["passed"]) / len(cat_decided), 4)
                    if cat_decided else None
                ),
                "grader_decided": sum(1 for r in cat_rows if r["tier"] == "grader"),
            }

        grader_rows = [r for r in decided if r["tier"] == "grader"]
        det_rows = [r for r in decided if r["tier"] != "grader"]
        out[arm] = {
            "n": len(arm_rows),
            "decided": len(decided),
            "undecided": len(arm_rows) - len(decided),
            "pass_rate": round(len(passed) / len(decided), 4) if decided else None,
            # The sensitivity check, precomputed: if these two diverge, the
            # headline depends on the grader.
            "pass_rate_deterministic_only": (
                round(sum(1 for r in det_rows if r["passed"]) / len(det_rows), 4)
                if det_rows else None
            ),
            "grader_share": round(len(grader_rows) / len(decided), 4) if decided else None,
            "mean_wall_seconds": (
                round(sum(r["wall_seconds"] for r in arm_rows) / len(arm_rows), 2)
                if arm_rows else None
            ),
            "tiers": dict(tiers),
            "by_category": by_cat,
        }
    return out


def _load_prior(path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    prior: list[dict] = []
    done: set[tuple[str, str]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_meta"):
                continue
            prior.append(row)
            done.add((row["task_id"], row["arm"]))
    return prior, done


def main() -> None:
    parser = argparse.ArgumentParser(description="MMLU-Pro open-ended (format arm) sweep.")
    parser.add_argument("--categories", type=str, default=",".join(PILOT_CATEGORIES))
    parser.add_argument("--split", type=str, default="test", choices=("test", "validation"))
    parser.add_argument("--n-per-category", type=int, default=25)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--arms", type=str, default=",".join(OPEN_ARMS))
    parser.add_argument("--grader-model", type=str, default="sonnet",
                        help="Claude alias used for equivalence grading.")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    selected = {n: OPEN_ARMS[n] for n in args.arms.split(",") if n in OPEN_ARMS}
    unknown = [n for n in args.arms.split(",") if n.strip() and n.strip() not in OPEN_ARMS]
    if unknown:
        print(f"Unknown arms (skipped): {unknown}", file=sys.stderr)
    if not selected:
        print("No valid arms selected.", file=sys.stderr)
        sys.exit(1)

    tasks = load_mmlu_pro_open(
        categories, split=args.split,
        n_per_category=args.n_per_category, start=args.start,
    )

    # Every grader decision is recorded so it can be human-labelled later
    # without re-running the sweep.
    grader_log: list[dict] = []
    grader = make_recording_grader(args.grader_model, grader_log)

    results: list[dict] = []
    done: set[tuple[str, str]] = set()
    RESULTS_DIR.mkdir(exist_ok=True)
    if args.resume:
        out_path = Path(args.resume)
        if not out_path.exists():
            print(f"--resume target does not exist: {out_path}", file=sys.stderr)
            sys.exit(1)
        results, done = _load_prior(out_path)
        print(f"Resuming {out_path}: {len(done)} pairs already done.")
        out_f = out_path.open("a")
        new_file = False
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"{ts}_mmlu_pro_open_n{len(tasks)}.jsonl"
        out_f = out_path.open("w")
        new_file = True

    grader_path = out_path.with_name(out_path.stem + "_grader_decisions.jsonl")

    total = len(tasks) * len(selected)
    print(f"MMLU-Pro OPEN-ENDED sweep: {len(tasks)} tasks × {len(selected)} arms "
          f"= {total} runs ({len(done)} skipped via resume)")
    print(f"Grader: {args.grader_model} (decisions -> {grader_path.name})\n")

    def _write(row: dict) -> None:
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()

    def _flush_grader_log() -> None:
        if grader_log:
            with grader_path.open("a") as gf:
                for entry in grader_log:
                    gf.write(json.dumps(entry) + "\n")
            grader_log.clear()

    try:
        if new_file:
            _write({
                "_meta": True, "benchmark": BENCHMARK, "categories": categories,
                "split": args.split, "n_per_category": args.n_per_category,
                "start": args.start, "arms": list(selected),
                "grader_model": args.grader_model,
                "format": "open_ended",
            })

        for task in tasks:
            printed = False
            for arm_name, arm_fn in selected.items():
                if (task["task_id"], arm_name) in done:
                    continue
                if not printed:
                    print(f"=== {task['task_id']} (gold={task['gold'][:60]}) ===")
                    printed = True
                try:
                    row = run_one_open(task, arm_name, arm_fn, grader)
                except UsageLimitReached as e:
                    _flush_grader_log()
                    out_f.flush()
                    out_f.close()
                    print(f"\n!!! Max usage exhausted on {task['task_id']} / "
                          f"{arm_name}: {e}", file=sys.stderr)
                    print(f"Partial results saved to {out_path} "
                          f"({len(results)} rows completed).", file=sys.stderr)
                    print(f"Resume when usage resets with:\n"
                          f"  python scripts/run_mmlu_pro_open.py "
                          f"--categories {args.categories} "
                          f"--n-per-category {args.n_per_category} "
                          f"--arms {args.arms} --resume {out_path}", file=sys.stderr)
                    sys.exit(2)
                results.append(row)
                row_out = dict(row)
                row_out["benchmark"] = BENCHMARK
                _write(row_out)
                _flush_grader_log()
                marker = {True: "✓", False: "✗", None: "?"}[row["passed"]]
                print(f"  {marker} {arm_name:<18} {row['wall_seconds']:>5.1f}s  "
                      f"[{row['tier']}]  {row['detail'][:46]}")
    finally:
        _flush_grader_log()
        if not out_f.closed:
            out_f.close()

    print("\n=== aggregate ===")
    print(json.dumps(summarize_open(results), indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Grader decisions: {grader_path}")
    print(f"Validate the grader BEFORE trusting any cell:\n"
          f"  python scripts/validate_grader.py {grader_path} --sample 40")


if __name__ == "__main__":
    main()
