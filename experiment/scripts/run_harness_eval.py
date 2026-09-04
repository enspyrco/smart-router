#!/usr/bin/env python3
"""Harness evaluation — frozen model, varying prompt.

Runs one harness over one split of one MMLU-Pro category and writes a
trajectory per question, including the model's reasoning text. The reasoning is
the point: it is what a failure taxonomy is built from later.

    # does this category have room to improve? (25 questions, no output file)
    python scripts/run_harness_eval.py --headroom --backend ollama --model qwen3.5:4b

    # baseline on the development split
    python scripts/run_harness_eval.py --harness H0 --split dev \
        --backend ollama --model qwen3.5:4b

    # final measurement, once, at the end
    python scripts/run_harness_eval.py --harness H0 --split hidden --confirm-hidden ...

The dev/hidden split is a deterministic hash of question_id, so it is identical
on every machine and every run, and does not correlate with question ordering.
The hidden split refuses to run without --confirm-hidden: a harness tuned
against questions it was measured on reports an accuracy nobody can defend, and
this project has already lost one dataset that way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.mmlu_pro import ALL_CATEGORIES, load_mmlu_pro
from harnesses import HARNESSES, get_harness
from model_backends import build_backend

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Changing this re-shuffles dev/hidden and invalidates every prior comparison.
SPLIT_SALT = "mmlu-pro-harness-v1"
HEADROOM_N = 25
# A dead backend shouldn't burn a whole sweep before saying so.
ABORT_AFTER_CONSECUTIVE_ERRORS = 3
# Outside this band the category cannot show a harness effect: too high and
# there is no room to improve, too low and the model is guessing.
HEADROOM_FLOOR, HEADROOM_CEILING = 0.35, 0.85


def split_of(question_id, dev_pct: int) -> str:
    """Deterministic dev/hidden assignment, stable across machines and runs."""
    digest = hashlib.sha256(f"{SPLIT_SALT}:{question_id}".encode()).hexdigest()
    return "dev" if int(digest[:8], 16) % 100 < dev_pct else "hidden"


def run_one(task: dict, harness, backend) -> dict:
    """One question through one harness. Never raises — failures are data."""
    record = {
        "question_id": task["question_id"],
        "task_id": task["task_id"],
        "question": task["question"],
        "category": task["category"],
        "model": backend.model,
        "backend": backend.name,
        "harness": harness.name,
        "correct_answer": task["gold"],
    }
    try:
        user_prompt = harness.render(task)
        if harness.name == "H2-MCP":
            from mcp_retrieval import retrieve_cs_context
            context = retrieve_cs_context(task["question"])
            record["retrieved_context"] = context
            user_prompt = (
                "Reference material retrieved through MCP:\n\n"
                f"{context}\n\n---\n\n{user_prompt}"
            )
        reply = backend.chat(harness.system, user_prompt)
    except Exception as exc:
        return {**record, "reasoning": "", "answer": None, "correct": False,
                "latency": 0.0, "tokens": 0, "error": f"{type(exc).__name__}: {exc}"}

    answer = harness.parse(reply.text)
    return {
        **record,
        "reasoning": reply.text,
        "answer": answer,
        # Unparseable counts as wrong. The answer contract is identical across
        # harnesses, so this cannot advantage one over another.
        "correct": answer is not None and answer == task["gold"],
        "unparseable": answer is None,
        "latency": round(reply.latency, 3),
        "tokens": reply.tokens,
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "tokens_estimated": reply.tokens_estimated,
    }


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("_meta") and not row.get("error"):
            done.add(row["question_id"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="computer science", choices=ALL_CATEGORIES)
    ap.add_argument("--harness", default="H0", choices=sorted(HARNESSES))
    ap.add_argument("--split", default="dev", choices=["dev", "hidden", "all"])
    ap.add_argument("--dev-pct", type=int, default=50,
                    help="percent of questions assigned to dev (default 50)")
    ap.add_argument("--confirm-hidden", action="store_true",
                    help="required to evaluate on the hidden split")
    ap.add_argument("--headroom", action="store_true",
                    help=f"quick {HEADROOM_N}-question check; no trajectory file")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai", "transformers"])
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, help="cap questions (after splitting)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run index; use 1,2,3 for repeated runs to measure spread")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.split == "hidden" and not args.confirm_hidden:
        sys.exit(
            "refusing to touch the hidden split without --confirm-hidden.\n"
            "The hidden split is for final measurement only. If you are still\n"
            "designing or comparing harnesses, use --split dev."
        )

    harness = get_harness(args.harness)
    tasks = load_mmlu_pro([args.category])
    if args.split != "all":
        tasks = [t for t in tasks if split_of(t["question_id"], args.dev_pct) == args.split]
    if args.headroom:
        tasks = tasks[:HEADROOM_N]
    elif args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        sys.exit(f"no tasks for category={args.category!r} split={args.split!r}")

    backend = build_backend(args.backend, args.model, temperature=args.temperature)

    out_path = None
    done: set = set()
    if not args.headroom:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = args.category.replace(" ", "_")
        name = f"{stamp}_harness_{args.harness}_{slug}_{args.split}_r{args.repeat}.jsonl"
        out_path = args.out or (RESULTS_DIR / name)
        if args.resume and out_path.exists():
            done = load_done(out_path)
            print(f"resuming: {len(done)} already complete")
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{args.harness} | {args.model} via {args.backend} | "
          f"{args.category} | split={args.split} | {len(tasks)} questions")
    if out_path:
        print(f"-> {out_path}")

    rows = []
    mode = "a" if (args.resume and out_path and out_path.exists()) else "w"
    fh = out_path.open(mode, encoding="utf-8") if out_path else None
    try:
        if fh and mode == "w":
            fh.write(json.dumps({
                "_meta": True, "harness": harness.name, "harness_system": harness.system,
                "harness_notes": harness.notes, "model": args.model, "backend": args.backend,
                "category": args.category, "split": args.split, "dev_pct": args.dev_pct,
                "split_salt": SPLIT_SALT, "temperature": args.temperature,
                "repeat": args.repeat, "n": len(tasks),
            }) + "\n")
            fh.flush()

        consecutive_errors = 0
        for i, task in enumerate(tasks, 1):
            if task["question_id"] in done:
                continue
            row = run_one(task, harness, backend)
            rows.append(row)
            if fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            mark = "ok " if row["correct"] else ("ERR" if row.get("error") else "   ")
            print(f"  [{i:4d}/{len(tasks)}] {mark} "
                  f"got={row['answer'] or '?'} want={row['correct_answer']} "
                  f"{row['latency']:6.1f}s", flush=True)

            # A dead backend should cost seconds, not a full sweep. Sporadic
            # failures are tolerated; a run of them means nothing is listening.
            consecutive_errors = consecutive_errors + 1 if row.get("error") else 0
            if consecutive_errors >= ABORT_AFTER_CONSECUTIVE_ERRORS:
                print(f"\naborting: {consecutive_errors} calls failed in a row")
                print(f"  last error: {row['error']}")
                if out_path:
                    print(f"  partial results kept; re-run with --resume {out_path}")
                break
    finally:
        if fh:
            fh.close()

    scored = [r for r in rows if not r.get("error")]
    if not scored:
        sys.exit("every call failed — check the backend is reachable")
    acc = sum(r["correct"] for r in scored) / len(scored)
    errs = len(rows) - len(scored)
    unparse = sum(r.get("unparseable", False) for r in scored)
    print(f"\naccuracy   {acc:.1%}  ({sum(r['correct'] for r in scored)}/{len(scored)})")
    print(f"unparseable {unparse}   call failures {errs}")
    print(f"mean latency {sum(r['latency'] for r in scored)/len(scored):.1f}s   "
          f"mean tokens {sum(r['tokens'] for r in scored)//len(scored)}")

    if args.headroom:
        print()
        if acc > HEADROOM_CEILING:
            print(f"NO HEADROOM — {acc:.0%} is above {HEADROOM_CEILING:.0%}. A harness")
            print("cannot show a meaningful gain here. Pick a harder category.")
        elif acc < HEADROOM_FLOOR:
            print(f"TOO HARD — {acc:.0%} is below {HEADROOM_FLOOR:.0%}. Near-guessing")
            print("leaves nothing for a harness to fix. Pick an easier category or model.")
        else:
            print(f"USABLE — {acc:.0%} sits in the {HEADROOM_FLOOR:.0%}-{HEADROOM_CEILING:.0%} band.")
        print(f"(n={len(scored)}; this is a smoke test, not a baseline)")


if __name__ == "__main__":
    main()
