#!/usr/bin/env python3
"""Type-routing decorrelation probe — the cheap falsifier for hyperspecialist routing.

Motivation: deep-research report wehmojvpc. Now that MCQ benchmarks are saturated
for frontier Claude (Haiku ties Sonnet), the difficulty-gap thesis is dead. Connor's
pivot is to route by task TYPE to small domain specialists. Before building any
router, run the ONE table that can kill the idea cheaply:

  For each MMLU-Pro category, run a general 7B and domain-specialist 7B models,
  score each item, and measure ITEM-LEVEL DECORRELATION — the fraction where a
  specialist is right and the generalist wrong (and vice versa).

Kill-conditions (report): the thesis dies cheap if, in-domain, either
  (a) the specialist does NOT beat the general 7B, OR
  (b) errors are CORRELATED (oracle best-of ~= the best single model, i.e. the
      specialist rescues nothing the generalist already gets).

Local only — zero Claude quota. The frontier arm ("does a 7B beat *saturated*
frontier Claude?") is a separate, later step, gated on this probe living.

Usage:
  uv run python scripts/run_type_routing_probe.py --n 100 \
      --categories math "computer science"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from benchmarks.mmlu_pro import category_slug, load_mmlu_pro, score_mmlu_pro
from run_pilot import RESULTS_DIR

# Same instruction the Echo arms use, so scoring is apples-to-apples with the sweeps.
PERSONA = (
    "You are a careful, methodical reasoner. Work through the problem step by step.\n"
    "End with exactly one line: Answer: X\n"
    "where X is the letter of your chosen option (A, B, C, ...). No text after that line."
)

# role -> Ollama model tag. "general" is the type-router's fallback / baseline;
# the rest are the domain specialists whose in-domain lift we're testing.
DEFAULT_MODELS = {
    "general": "qwen2.5:7b-instruct-q4_K_M",
    "math_specialist": "hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M",
    "coder_specialist": "qwen2.5-coder:7b-instruct",
}

BASE_URL = "http://localhost:11434"


def make_arm(model_tag: str, num_predict: int):
    llm = ChatOllama(
        model=model_tag, base_url=BASE_URL, temperature=0.0, num_predict=num_predict
    )

    def arm(task: dict) -> str:
        resp = llm.invoke(
            [SystemMessage(content=PERSONA), HumanMessage(content=task["prompt"])]
        )
        return resp.content

    return arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["math", "computer science"])
    ap.add_argument("--n", type=int, default=100, help="items per category")
    ap.add_argument("--num-predict", type=int, default=1536, help="max gen tokens/call")
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="role=tag pairs; default = general + math + coder specialists",
    )
    args = ap.parse_args()

    models = dict(DEFAULT_MODELS)
    if args.models:
        models = {}
        for pair in args.models:
            role, _, tag = pair.partition("=")
            models[role] = tag
    arms = {role: make_arm(tag, args.num_predict) for role, tag in models.items()}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(RESULTS_DIR) / f"{stamp}_type_routing_probe.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"models: {json.dumps(models, indent=2)}")
    print(f"categories: {args.categories}  n/cat: {args.n}")
    print(f"streaming -> {out_path}\n")

    # per_item[(cat, task_id)][role] = bool passed
    per_item: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)

    with out_path.open("w") as fh:
        for cat in args.categories:
            tasks = load_mmlu_pro([cat], split="test", n_per_category=args.n)
            print(f"[{cat}] {len(tasks)} tasks x {len(arms)} models")
            for i, task in enumerate(tasks):
                for role, arm in arms.items():
                    t0 = time.perf_counter()
                    try:
                        output = arm(task)
                        passed, detail = score_mmlu_pro(output, task)
                    except Exception as e:  # never let one call kill the run
                        passed, detail = False, f"{type(e).__name__}: {str(e)[:160]}"
                        output = ""
                    dt = time.perf_counter() - t0
                    per_item[(cat, task["task_id"])][role] = passed
                    fh.write(
                        json.dumps(
                            {
                                "category": cat,
                                "task_id": task["task_id"],
                                "role": role,
                                "model": models[role],
                                "gold": task["gold"],
                                "passed": passed,
                                "detail": detail,
                                "seconds": round(dt, 2),
                            }
                        )
                        + "\n"
                    )
                    fh.flush()
                if (i + 1) % 10 == 0:
                    print(f"  [{cat}] {i + 1}/{len(tasks)} done")

    summarize(per_item, args.categories, list(models.keys()), stamp)
    return 0


def summarize(per_item, categories, roles, stamp):
    summary = {"stamp": stamp, "categories": {}}
    print("\n" + "=" * 72)
    print("CONFUSION TABLE — the falsifier")
    print("=" * 72)
    for cat in categories:
        items = {tid: r for (c, tid), r in per_item.items() if c == cat}
        n = len(items)
        if not n:
            continue
        acc = {role: sum(r.get(role, False) for r in items.values()) / n for role in roles}
        # oracle = any model right on the item
        oracle = sum(any(r.values()) for r in items.values()) / n
        best_single = max(acc.values())
        print(f"\n### {cat}  (n={n})")
        for role in roles:
            print(f"  {role:20s} acc = {acc[role]:.3f}")
        print(f"  {'ORACLE (best-of)':20s} acc = {oracle:.3f}   "
              f"(+{oracle - best_single:+.3f} over best single)")

        # pairwise decorrelation vs the general baseline
        pairs = {}
        gen = "general"
        if gen in roles:
            for role in roles:
                if role == gen:
                    continue
                s_right_g_wrong = sum(
                    r.get(role) and not r.get(gen) for r in items.values()
                )
                g_right_s_wrong = sum(
                    r.get(gen) and not r.get(role) for r in items.values()
                )
                both_right = sum(r.get(role) and r.get(gen) for r in items.values())
                both_wrong = sum(
                    (not r.get(role)) and (not r.get(gen)) for r in items.values()
                )
                pairs[role] = {
                    "specialist_right_general_wrong": s_right_g_wrong,
                    "general_right_specialist_wrong": g_right_s_wrong,
                    "both_right": both_right,
                    "both_wrong": both_wrong,
                }
                print(
                    f"  vs general | {role}: "
                    f"spec-rescues={s_right_g_wrong}  "
                    f"gen-rescues={g_right_s_wrong}  "
                    f"both-right={both_right}  both-wrong={both_wrong}"
                )
        summary["categories"][cat] = {
            "n": n,
            "accuracy": acc,
            "oracle": oracle,
            "oracle_lift_over_best_single": oracle - best_single,
            "decorrelation_vs_general": pairs,
        }

    out = Path(RESULTS_DIR) / f"{stamp}_type_routing_probe_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out}")
    print("\nREAD: spec-rescues > 0 AND oracle-lift > 0  => decorrelation is real,")
    print("      the specialist covers items the generalist misses. Thesis lives.")
    print("      spec-rescues ~ 0  => errors correlated, thesis dies cheap.")


if __name__ == "__main__":
    raise SystemExit(main())
