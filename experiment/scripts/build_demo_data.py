#!/usr/bin/env python3
"""Join the type-routing probe JSONL with the MMLU-Pro dataset to build a
curated set of demo questions for the live routing theatre.

Selects questions that tell the decorrelation story: specialist-rescues-general,
general-rescues-specialist, both-right, and a both-wrong control.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.mmlu_pro import load_mmlu_pro

RESULTS = sorted(Path("results").glob("*_type_routing_probe.jsonl"))[-1]
ROLES = ["general", "math_specialist", "coder_specialist"]


def parsed_answer(passed: bool, detail: str, gold: str) -> str | None:
    if passed:
        return gold
    m = re.search(r"got\s+([A-Z])", detail)
    if m:
        return m.group(1)
    return None  # unparseable


def main() -> int:
    # logged results: task_id -> role -> {passed, answer}
    logs: dict[str, dict] = {}
    for line in RESULTS.open():
        r = json.loads(line)
        if r["category"] != "math":
            continue
        logs.setdefault(r["task_id"], {})[r["role"]] = r

    tasks = {t["task_id"]: t for t in load_mmlu_pro(["math"], n_per_category=100)}

    rows = []
    for tid, byrole in logs.items():
        if set(byrole) != set(ROLES) or tid not in tasks:
            continue
        t = tasks[tid]
        res = {}
        for role in ROLES:
            r = byrole[role]
            res[role] = {
                "answer": parsed_answer(r["passed"], r["detail"], t["gold"]),
                "correct": bool(r["passed"]),
            }
        oracle = any(res[role]["correct"] for role in ROLES)
        rows.append({
            "task_id": tid,
            "question": t["question"],
            "labels": t["choices"]["label"],
            "options": t["choices"]["text"],
            "gold": t["gold"],
            "results": res,
            "oracle_correct": oracle,
            # story tag
            "spec_rescue": res["math_specialist"]["correct"] and not res["general"]["correct"],
            "gen_rescue": res["general"]["correct"] and not res["math_specialist"]["correct"],
            "both_right": res["general"]["correct"] and res["math_specialist"]["correct"],
        })

    def short(r):  # prefer concise questions for on-stage readability
        return len(r["question"])

    spec = sorted([r for r in rows if r["spec_rescue"]], key=short)[:4]
    gen = sorted([r for r in rows if r["gen_rescue"]], key=short)[:3]
    both = sorted([r for r in rows if r["both_right"]], key=short)[:2]
    picked, seen = [], set()
    for r in spec + gen + both:
        if r["task_id"] not in seen:
            picked.append(r)
            seen.add(r["task_id"])

    out = Path("../demo/demo_data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"questions": picked}, indent=2))
    print(f"wrote {len(picked)} demo questions -> {out.resolve()}")
    for r in picked:
        tag = "SPEC-rescue" if r["spec_rescue"] else "GEN-rescue" if r["gen_rescue"] else "both-right"
        print(f"  [{tag:11}] gold={r['gold']} "
              f"gen={r['results']['general']['answer']} "
              f"math={r['results']['math_specialist']['answer']} "
              f"code={r['results']['coder_specialist']['answer']}  "
              f"{r['question'][:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
