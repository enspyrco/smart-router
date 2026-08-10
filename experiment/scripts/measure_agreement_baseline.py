#!/usr/bin/env python3
"""Measure the numbers Echo's economics actually rest on.

Echo accepts the cheap answer when two cheap calls agree and escalates when they
disagree. Three quantities decide whether that is a good idea:

1. ESCALATION RATE r -- how often the two cheap calls disagree.
   Echo's cost is 2*cheap + r*expensive. At the README's prices (haiku x2 = $2/M,
   sonnet x1 = $3/M) Echo is cheaper than sonnet-only iff

       2 + 3r < 3   =>   r < 1/3

   A hard break-even at 33%. The verdict uses the Wilson UPPER BOUND on r, not the
   point estimate: near the break-even, "r=30% at n=60" has an interval that
   swallows 1/3, and printing PROFITABLE off that is a sample-size gate pretending
   to be a confidence statement (Wu, cage-match #6).

2. THE ASYMMETRY -- P(agree | correct) vs P(agree | wrong).
   Echo's premise is that agreement predicts correctness, and the mechanism is
   that there is ONE way to be right and MANY ways to be wrong. NOTE the exact
   conditioning: `wrong` means THE FIRST CALL was wrong, which is the answer Echo
   would accept -- not the free-floating property "the model reproduces its
   errors". Do not blur the two in prose.

3. THE PERSONA DELTA -- does persona perturbation beat plain resampling?
   Models diverge from themselves on an IDENTICAL prompt, so the no-perturbation
   baseline is not zero. Tested with McNemar on the DISCORDANT PAIRS, not by
   eyeballing a net difference: a net delta of 1/150 can sit on ~10 discordant
   tasks in each direction, and "personas add nothing" is an EQUIVALENCE claim,
   which a small observed difference does not establish (Carnot/Tesla/Wu).

FOUR ARMS, and the third is the point. Three calls per task -- a1, a2 (both
PERSONA_A), b1 (PERSONA_B) -- yield:

    persona    a1 vs b1     (A vs B, shares a1 with the control)
    control    a1 vs a2     (A vs A, plain resampling)
    unshared   a2 vs b1     (A vs B sharing NOTHING with the control)

Sharing a1 between persona and control is a PAIRED design: it correlates the two
estimates, which REDUCES the variance of their difference. That is a strength,
not a confound, and the marginal expectations are unbiased either way. But the
narration "two independent statistics" was wrong, and the honest check is free
with data already collected: `unshared` is an A-vs-B estimate sharing no call
with `control`. If persona and unshared agree, the shared-a1 objection is
answered with evidence rather than argument.

STILL MISSING (needs a run, not a code change): agree(B, B). The control measures
A-vs-A only, so persona B's self-agreement is ASSUMED to mirror A's. Wu is right
that this is the untested symmetry.

ABSTENTION IS NOT DISAGREEMENT, BUT EXCLUDING IT IS NOT NEUTRAL EITHER.
An earlier version of this docstring claimed exclusion was the conservative
choice. That was backwards. Errors, truncation and parse failures correlate with
HARD tasks, and hard tasks are where disagreement lives -- so excluding them
biases r DOWNWARD, which FLATTERS Echo. Production Echo also cannot accept an
unparseable cheap answer; it must escalate. So both rates are reported:
`r` (abstentions dropped) and `r (abstain escalates)`, the production policy.

TOOL-FREE BY DEFAULT via ChatOAuth -- `claude --print` was measured reading files
on both tiers, and opportunistic tool use injects variance into a dependent
variable that IS agreement.

Usage:
    python3 scripts/measure_agreement_baseline.py --benchmark mmlu_pro --n 200
    python3 scripts/measure_agreement_baseline.py --from-json results/<run>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from benchmarks.bbh import load_bbh, score_bbh  # noqa: E402
from benchmarks.bbh_arms import PERSONA_A, PERSONA_B  # noqa: E402
from benchmarks.mmlu_pro import load_mmlu_pro, score_mmlu_pro  # noqa: E402
from chat_oauth import ChatOAuth  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

BREAK_EVEN = 1 / 3
MIN_CELL = 30        # minimum tasks in the correct AND wrong cells
MIN_SCORED = 60      # minimum scored tasks before quoting r at all


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """(lo, hi) Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    r = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - r) / d, (c + r) / d)


def mcnemar(b: int, c: int) -> float:
    """Two-sided McNemar p-value on discordant counts b and c.

    The decisive test for "personas add nothing": a net delta of one task can sit
    on ten discordant tasks in each direction, and only the discordant pairs carry
    information about a difference. Exact binomial for small n, normal
    approximation with continuity correction above it.
    """
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        from math import comb
        tail = sum(comb(n, i) for i in range(0, min(b, c) + 1))
        return min(1.0, 2.0 * tail / (2 ** n))
    from math import erfc, sqrt
    chi = (abs(b - c) - 1) ** 2 / n
    return erfc(sqrt(chi / 2))


def call(model: ChatOAuth, persona: str, prompt: str) -> str:
    return model.invoke([SystemMessage(content=persona),
                         HumanMessage(content=prompt)]).content


def load_tasks(benchmark: str, n: int, stratified: bool):
    """Load tasks, STRATIFIED across categories by default.

    load_mmlu_pro(n=150) returns the FIRST 150 rows, which in MMLU-Pro are all one
    category -- the n=150 run in this repo was 150/150 physics while every artifact
    said "mmlu_pro" (Wu's catch, cage-match #6). Since the whole point of Echo's
    per-category table is that categories DIFFER, a single-category sample labelled
    as the benchmark is a mislabel that propagates straight into it.
    """
    if benchmark == "bbh":
        return load_bbh(n=n), score_bbh
    if stratified:
        tasks = load_mmlu_pro(n_per_category=max(1, n // 14))
        return tasks[:n], score_mmlu_pro
    return load_mmlu_pro(n=n), score_mmlu_pro


def agreement(rows: dict, k1: str, k2: str) -> dict:
    """Per-arm stats. Abstentions counted, never silently folded into either side."""
    st: Counter = Counter()
    per_task: dict[str, bool] = {}
    for tid, r in rows.items():
        x, y = r.get(k1) or {}, r.get(k2) or {}
        if x.get("answer") is None or y.get("answer") is None:
            st["abstained"] += 1
            continue
        st["scored"] += 1
        agree = x["answer"] == y["answer"]
        per_task[tid] = bool(agree)
        st["agree"] += agree
        if x["correct"]:
            st["correct"] += 1
            st["agree_given_correct"] += agree
        else:
            st["wrong"] += 1
            st["agree_given_wrong"] += agree
    return {"counts": st, "per_task": per_task}


def summarise(label: str, res: dict) -> dict:
    st = res["counts"]
    n_tot = st["scored"]
    n = n_tot or 1
    disagree = n_tot - st["agree"]
    r = disagree / n
    denom_esc = n_tot + st["abstained"]
    r_escalate = (disagree + st["abstained"]) / denom_esc if denom_esc else 0.0
    r_lo, r_hi = wilson(disagree, n_tot)

    pac = st["agree_given_correct"] / (st["correct"] or 1)
    paw = st["agree_given_wrong"] / (st["wrong"] or 1)

    if st["scored"] < MIN_SCORED:
        cost = f"INSUFFICIENT — only {st['scored']} scored (need {MIN_SCORED})"
    elif r_hi < BREAK_EVEN:
        cost = f"PROFITABLE — 95% upper bound {r_hi*100:.0f}% < {BREAK_EVEN*100:.0f}%"
    elif r_lo > BREAK_EVEN:
        cost = f"NOT PROFITABLE — 95% lower bound {r_lo*100:.0f}% > {BREAK_EVEN*100:.0f}%"
    else:
        cost = (f"INDETERMINATE — 95% CI [{r_lo*100:.0f}%, {r_hi*100:.0f}%] "
                f"straddles the {BREAK_EVEN*100:.0f}% break-even")

    if st["correct"] >= MIN_CELL and st["wrong"] >= MIN_CELL:
        mech = ("mechanism holds" if (pac - paw) > 0.10
                else "agreement barely predicts correctness")
    else:
        short = []
        if st["correct"] < MIN_CELL:
            short.append(f"{st['correct']} correct")
        if st["wrong"] < MIN_CELL:
            short.append(f"{st['wrong']} wrong")
        mech = f"INSUFFICIENT — {', '.join(short)} (need {MIN_CELL} each)"

    return dict(label=label, scored=st["scored"], abstained=st["abstained"],
                correct=st["correct"], wrong=st["wrong"],
                escalation_rate=r, escalation_rate_ci=[r_lo, r_hi],
                escalation_rate_abstain_escalates=r_escalate,
                p_agree_given_correct=pac, p_agree_given_wrong=paw,
                separation=pac - paw, verdict=cost, mechanism_verdict=mech)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["mmlu_pro", "bbh"], default="mmlu_pro")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--single-category", action="store_true",
                    help="take the first n rows (ONE category); default is stratified")
    ap.add_argument("--from-json", default=None,
                    help="re-analyse a saved run; spends no calls and needs no other flags")
    args = ap.parse_args()

    if args.from_json:
        # Identity comes from the DATUM, not the CLI. Previously the script
        # re-loaded `--n` tasks (default 40) and reported len(tasks) as n, so a
        # replay of a 150-row file printed n=40; and a BBH-labelled re-analysis of
        # MMLU-Pro rows would be produced without complaint (Carnot/Tesla/Wu).
        saved = json.loads(Path(args.from_json).read_text())
        rows = saved["rows"]
        benchmark = saved.get("benchmark", "unknown")
        model_name = saved.get("model", "unknown")
        gen_cfg = saved.get("generation_config", "NOT RECORDED (pre-cage-match run)")
        categories = saved.get("categories", "NOT RECORDED (pre-cage-match run)")
        personas_sha = saved.get("personas_sha", "NOT RECORDED")
        print(f"re-analysing {len(rows)} saved rows from {args.from_json}")
        print(f"  identity from datum: benchmark={benchmark} model={model_name} "
              f"categories={categories}\n")
    else:
        tasks, score = load_tasks(args.benchmark, args.n, not args.single_category)
        model = ChatOAuth(model=args.model)
        benchmark, model_name = args.benchmark, args.model
        gen_cfg = model.generation_config()
        personas_sha = hashlib.sha256((PERSONA_A + PERSONA_B).encode()).hexdigest()[:12]
        categories = dict(Counter(t.get("category", "n/a") for t in tasks))
        print(f"{len(tasks)} tasks | {gen_cfg} | categories={categories}\n")

        def work(task):
            out = {}
            for key, persona in (("a1", PERSONA_A), ("a2", PERSONA_A), ("b1", PERSONA_B)):
                try:
                    raw = call(model, persona, task["prompt"])
                    ok, parsed = score(raw, task)
                    # RAW is persisted. The old rows kept only the scorer's parsed
                    # artifact, so "raw output is the datum" was false and a replay
                    # could only ever reproduce that day's parser (Carnot + Wu).
                    out[key] = {"correct": ok, "answer": parsed, "raw": raw}
                except Exception as exc:                      # noqa: BLE001
                    out[key] = {"correct": None, "answer": None, "raw": None,
                                "error": repr(exc)[:200]}
            return task["task_id"], out

        rows = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(work, t) for t in tasks]):
                tid, out = fut.result()
                rows[tid] = out
                print(".", end="", flush=True)
        print("\n")

    arms = {
        "persona (a1 vs b1)": agreement(rows, "a1", "b1"),
        "control (a1 vs a2)": agreement(rows, "a1", "a2"),
        "unshared (a2 vs b1)": agreement(rows, "a2", "b1"),
    }
    summaries = {k: summarise(k, v) for k, v in arms.items()}

    pa = arms["persona (a1 vs b1)"]["per_task"]
    pc = arms["control (a1 vs a2)"]["per_task"]
    both = set(pa) & set(pc)
    b = sum(1 for t in both if pa[t] and not pc[t])
    c = sum(1 for t in both if pc[t] and not pa[t])
    p = mcnemar(b, c)

    L = [f"# Agreement baseline — {benchmark}, model={model_name}, n={len(rows)}", "",
         f"Generation config: `{gen_cfg}`",
         f"Personas sha256[:12]: `{personas_sha}`",
         f"Categories: `{categories}`", "",
         "Tool-free raw endpoint (ChatOAuth). `claude --print` was measured reading "
         "files on both tiers, so it cannot be used for a measurement whose dependent "
         "variable is agreement.", "",
         "| arm | scored | correct | wrong | abstained | r | r 95% CI | r (abstain escalates) | "
         "P(agree\\|correct) | P(agree\\|wrong) | separation | economics | mechanism |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in summaries.values():
        lo, hi = a["escalation_rate_ci"]
        L.append(f"| {a['label']} | {a['scored']} | {a['correct']} | {a['wrong']} | "
                 f"{a['abstained']} | **{a['escalation_rate']*100:.0f}%** | "
                 f"[{lo*100:.0f}%, {hi*100:.0f}%] | "
                 f"{a['escalation_rate_abstain_escalates']*100:.0f}% | "
                 f"{a['p_agree_given_correct']*100:.0f}% | {a['p_agree_given_wrong']*100:.0f}% | "
                 f"{a['separation']*100:+.0f}pp | {a['verdict']} | {a['mechanism_verdict']} |")

    L += ["", "## Do personas beat plain resampling?", "",
          f"McNemar on persona-vs-control discordant pairs: b={b}, c={c}, **p={p:.3f}**.", ""]
    if b + c == 0:
        L.append("No discordant pairs — the arms made identical decisions on every task.")
    elif p < 0.05:
        L.append(f"**Personas DO change the escalation decision** (p={p:.3f} over {b+c} "
                 "discordant tasks). A small net delta hides real churn in both directions.")
    else:
        L.append(f"**No detectable persona effect** (p={p:.3f} over {b+c} discordant tasks). "
                 "This is a failure to reject, NOT proof of equivalence — for an "
                 "equivalence claim, pre-specify a margin and run TOST.")
    L += ["",
          "The `unshared (a2 vs b1)` arm is an A-vs-B comparison sharing NO call with the "
          "control, so it answers the shared-a1 objection with data rather than argument: "
          "if it tracks the persona arm, the pairing is not manufacturing the similarity.",
          "",
          "## Reading the numbers", "",
          "- `r` gates on the Wilson upper bound vs the break-even, not a point estimate.",
          "- `r (abstain escalates)` is the PRODUCTION policy: Echo cannot accept an "
          "unparseable cheap answer. Excluding abstentions biases r DOWNWARD and flatters "
          "Echo; both are shown so neither convention hides.",
          "- `wrong` means the FIRST call was wrong — the answer Echo would accept. It is "
          "not the free-floating claim that the model reproduces its own errors.",
          "- agree(B,B) is NOT measured: persona B's self-agreement is assumed to mirror "
          "A's. That symmetry is untested."]

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = RESULTS / f"{stamp}_agreement_baseline_{benchmark}_{model_name}_n{len(rows)}"
    base.with_suffix(".json").write_text(json.dumps(
        dict(benchmark=benchmark, model=model_name, n=len(rows),
             generation_config=gen_cfg, categories=categories, personas_sha=personas_sha,
             source_json=args.from_json, arms=summaries,
             mcnemar=dict(b=b, c=c, p=p), rows=rows), indent=2))
    base.with_suffix(".md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {base.with_suffix('.md').name}")


if __name__ == "__main__":
    main()
