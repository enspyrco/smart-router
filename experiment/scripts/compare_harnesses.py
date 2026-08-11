#!/usr/bin/env python3
"""Paired comparison of two harness runs.

Comparing two accuracy percentages is the wrong test. The two runs saw the
*same* questions, so most of the variance is shared and cancels — the evidence
lives entirely in the questions whose outcome changed. This runs McNemar's
exact test on those, which needs far less data than comparing two independent
proportions.

That matters here: MMLU-Pro Computer Science has 410 questions, so a split half
carries roughly +/-7 points of sampling error on its own, while the harness
gains being chased are 3-5 points. Unpaired, those gains are unmeasurable.
Paired, they are often decidable.

    python scripts/compare_harnesses.py results/..._H0_..._dev_r1.jsonl \
                                        results/..._H1_..._dev_r1.jsonl

Multiple runs of the same harness can be passed with --a/--b repeated, in which
case each question is scored by majority across runs before pairing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_run(paths: list[Path]) -> tuple[dict, dict]:
    """question_id -> correct(bool), plus meta. Majority vote across repeats."""
    votes: dict = defaultdict(list)
    meta: dict = {}
    for path in paths:
        if not path.exists():
            sys.exit(f"no such file: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("_meta"):
                meta = meta or row
                continue
            if row.get("error"):
                continue
            votes[row["question_id"]].append(bool(row["correct"]))
    return {q: sum(v) * 2 > len(v) for q, v in votes.items()}, meta


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b, c are the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="exactly two run files")
    ap.add_argument("--a", action="append", type=Path, default=[], help="repeat of run A")
    ap.add_argument("--b", action="append", type=Path, default=[], help="repeat of run B")
    args = ap.parse_args()

    a_paths, b_paths = list(args.a), list(args.b)
    if args.files:
        if len(args.files) != 2 or a_paths or b_paths:
            sys.exit("pass either two positional files, or --a/--b repeatedly")
        a_paths, b_paths = [args.files[0]], [args.files[1]]
    if not a_paths or not b_paths:
        sys.exit("need runs for both sides: two files, or --a/--b")

    a, meta_a = load_run(a_paths)
    b, meta_b = load_run(b_paths)
    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("the two runs share no questions — different split or category?")

    name_a = meta_a.get("harness", a_paths[0].stem)
    name_b = meta_b.get("harness", b_paths[0].stem)
    for key in ("category", "split", "model"):
        if meta_a.get(key) and meta_b.get(key) and meta_a[key] != meta_b[key]:
            print(f"WARNING: runs differ on {key}: "
                  f"{meta_a[key]!r} vs {meta_b[key]!r} — not a clean comparison\n")

    acc_a = sum(a[q] for q in shared) / len(shared)
    acc_b = sum(b[q] for q in shared) / len(shared)
    only_a = [q for q in shared if a[q] and not b[q]]   # A right, B wrong
    only_b = [q for q in shared if b[q] and not a[q]]   # B right, A wrong
    both = sum(1 for q in shared if a[q] and b[q])
    neither = sum(1 for q in shared if not a[q] and not b[q])

    print(f"paired on {len(shared)} questions "
          f"({meta_a.get('category', '?')}, split={meta_a.get('split', '?')})")
    if len(a_paths) > 1 or len(b_paths) > 1:
        print(f"majority vote over {len(a_paths)} run(s) of A, {len(b_paths)} of B")
    print()
    print(f"  {name_a:8s} {acc_a:6.1%}")
    print(f"  {name_b:8s} {acc_b:6.1%}")
    print(f"  {'delta':8s} {acc_b - acc_a:+6.1%}")
    print()
    print("  agreement table")
    print(f"    both correct            {both:4d}")
    print(f"    both wrong              {neither:4d}")
    print(f"    only {name_a:8s} correct {len(only_a):4d}   <- the evidence")
    print(f"    only {name_b:8s} correct {len(only_b):4d}   <- the evidence")

    p = mcnemar_exact(len(only_a), len(only_b))
    print(f"\n  McNemar exact two-sided p = {p:.4f}")
    if p < 0.05:
        better = name_b if len(only_b) > len(only_a) else name_a
        print(f"  -> the difference is unlikely to be chance; {better} is ahead")
    else:
        print("  -> NOT distinguishable from chance on this data")
        need = max(len(only_a), len(only_b))
        print(f"     only {len(only_a) + len(only_b)} questions changed outcome; "
              f"{need} in one direction is not enough")
        print("     more questions or more repeats would be needed to decide")

    if only_b:
        print(f"\n  fixed by {name_b} (first 10): "
              f"{', '.join(str(q) for q in only_b[:10])}")
    if only_a:
        print(f"  broken by {name_b} (first 10): "
              f"{', '.join(str(q) for q in only_a[:10])}")


if __name__ == "__main__":
    main()
