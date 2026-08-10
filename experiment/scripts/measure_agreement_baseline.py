#!/usr/bin/env python3
"""Measure the numbers Echo's economics actually rest on.

Echo accepts the cheap answer when two cheap calls agree and escalates when they
disagree. Three quantities decide whether that is a good idea, and none of them
are currently measured:

1. ESCALATION RATE r -- how often the two cheap calls disagree.
   Echo's cost is 2*cheap + r*expensive. At the README's prices (haiku x2 = $2/M,
   sonnet x1 = $3/M) Echo is cheaper than sonnet-only iff

       2 + 3r < 3   =>   r < 1/3

   So there is a hard break-even at 33% escalation. Above it Echo costs MORE than
   just calling the expensive model. r is the single number that decides
   profitability and it is measurable in an afternoon.

2. THE ASYMMETRY -- P(agree | correct) vs P(agree | wrong).
   Echo's whole premise is that agreement predicts correctness. That premise has
   a real mechanism behind it: there is ONE way to be right and MANY ways to be
   wrong, so correct answers are stable attractors and wrong answers are not.
   Measured on a hard maths battery in sibling work (~/git/research/llm-reproducibility):
   models reproduced nearly every answer they got right, and 4-16% of the answers
   they got wrong. That asymmetry IS the mechanism -- but it has never been
   measured on Echo's own benchmarks, and its strength sets the ceiling on how
   well agreement can possibly predict correctness.

3. THE PERSONA DELTA -- does persona perturbation beat plain resampling?
   Echo uses two different persona prompts. But models diverge from themselves at
   temperature 0 with an IDENTICAL prompt, so the no-perturbation baseline is not
   zero. If PERSONA_A-twice produces the same escalation signal as PERSONA_A vs
   PERSONA_B, the persona machinery is doing nothing and Echo gets simpler --
   a stronger result, not a weaker one. This is the control the design is missing.

WHY A CONTROL AT ALL. Sibling work spent a morning comparing models across
families before measuring within-model variance, and every number was
uninterpretable until the baseline existed: cross-model divergence cannot exceed
within-model divergence. Echo compares across personas without a
no-perturbation baseline. Same shape.

ABSTENTION IS NOT DISAGREEMENT. A refusal or an unparseable answer is MISSING
DATA. Counting it as disagreement inflates the escalation rate (and so
understates Echo's profitability); counting it as agreement hides a real
escalation trigger. Both are reported separately and never pooled.

TOOL-FREE BY DEFAULT. Uses ChatOAuth, not ChatClaudeCode: `claude --print` is an
agent with a shell and was measured reading files on both tiers. Opportunistic
tool use injects variance into the agreement signal that has nothing to do with
task difficulty -- fatal for a measurement whose dependent variable IS agreement.

Usage:
    python3 scripts/measure_agreement_baseline.py --benchmark mmlu_pro --n 40
    python3 scripts/measure_agreement_baseline.py --benchmark bbh --n 40 --model haiku
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

BREAK_EVEN = 1 / 3   # see docstring: 2 + 3r < 3

# P(agree|wrong) is a conditional rate over ONLY the tasks the model got wrong.
# At n=12 with a ~75%-accurate model that cell holds ~3 samples, and the pilot
# duly produced 0% for one arm and 100% for the other -- a coin flip printed as a
# verdict. Sibling work (~/git/research/llm-reproducibility) already learned this
# the expensive way: a report there announced "genuine cross-family independence"
# off 0 collisions in 5 pairs. Below these thresholds the script reports
# INSUFFICIENT and makes no claim.
MIN_CELL = 30        # minimum tasks in the correct AND wrong cells
MIN_SCORED = 60      # minimum scored tasks overall before quoting r


def call(model: ChatOAuth, persona: str, prompt: str) -> str:
    return model.invoke([SystemMessage(content=persona),
                         HumanMessage(content=prompt)]).content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["mmlu_pro", "bbh"], default="mmlu_pro")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--from-json", default=None,
                    help="re-analyse a saved run's rows; spends no calls")
    args = ap.parse_args()

    if args.benchmark == "mmlu_pro":
        tasks, score = load_mmlu_pro(n=args.n), score_mmlu_pro
    else:
        tasks, score = load_bbh(n=args.n), score_bbh

    model = ChatOAuth(model=args.model)
    print(f"{len(tasks)} tasks | model={args.model} | tool-free raw endpoint\n")

    # Three calls per task:
    #   a1, a2 -> PERSONA_A twice          (plain-resampling control)
    #   b1     -> PERSONA_B                (paired with a1 for the persona arm)
    def work(task):
        out = {}
        for key, persona in (("a1", PERSONA_A), ("a2", PERSONA_A), ("b1", PERSONA_B)):
            try:
                raw = call(model, persona, task["prompt"])
                ok, parsed = score(raw, task)
                out[key] = {"correct": ok, "answer": parsed}
            except Exception as exc:                      # noqa: BLE001
                out[key] = {"correct": None, "answer": None, "error": repr(exc)[:120]}
        return task["task_id"], out

    if args.from_json:
        # Raw model output is the datum; everything downstream is interpretation
        # and must be re-runnable without re-spending the calls to fix a gate.
        saved = json.loads(Path(args.from_json).read_text())
        rows = saved["rows"]
        print(f"re-analysing {len(rows)} saved rows from {args.from_json}\n")
        tasks = tasks[:len(rows)]
    else:
        rows = {}
        _pool = ThreadPoolExecutor(max_workers=args.workers)
    if not args.from_json:
      with _pool as ex:
        for fut in as_completed([ex.submit(work, t) for t in tasks]):
            tid, out = fut.result()
            rows[tid] = out
            print(".", end="", flush=True)
      print("\n")

    def agreement(rows, k1, k2):
        """Returns per-arm stats. Abstentions are excluded, never counted as either."""
        st = defaultdict(int)
        for r in rows.values():
            x, y = r[k1], r[k2]
            if x["answer"] is None or y["answer"] is None:
                st["abstained"] += 1
                continue
            st["scored"] += 1
            agree = x["answer"] == y["answer"]
            st["agree"] += agree
            # correctness of the FIRST call, which is what Echo would accept
            if x["correct"]:
                st["correct"] += 1
                st["agree_given_correct"] += agree
            else:
                st["wrong"] += 1
                st["agree_given_wrong"] += agree
        return st

    def summarise(label, st):
        n = st["scored"] or 1
        r = 1 - st["agree"] / n                       # escalation rate
        pac = st["agree_given_correct"] / (st["correct"] or 1)
        paw = st["agree_given_wrong"] / (st["wrong"] or 1)

        # GATE EACH STATISTIC ON ITS OWN DENOMINATOR. r is computed over every
        # scored task; the separation is computed over the correct and wrong
        # cells separately. A single blanket gate blocked a well-powered r=13%
        # (n=150) because the wrong cell held 22 — the same wrong-denominator
        # mistake this script exists to avoid, made by the guard itself.
        r_ok = st["scored"] >= MIN_SCORED
        sep_ok = st["correct"] >= MIN_CELL and st["wrong"] >= MIN_CELL

        if not r_ok:
            cost_verdict = f"INSUFFICIENT — only {st['scored']} scored (need {MIN_SCORED})"
        else:
            cost_verdict = ("PROFITABLE" if r < BREAK_EVEN else
                            "NOT PROFITABLE — costs more than expensive-only")
        if sep_ok:
            mech_verdict = ("mechanism holds" if (pac - paw) > 0.10 else
                            "agreement barely predicts correctness")
        else:
            short = [] 
            if st["correct"] < MIN_CELL:
                short.append(f"{st['correct']} correct")
            if st["wrong"] < MIN_CELL:
                short.append(f"{st['wrong']} wrong")
            mech_verdict = f"INSUFFICIENT — {', '.join(short)} (need {MIN_CELL} each)"

        return dict(label=label, scored=st["scored"], abstained=st["abstained"],
                    correct=st["correct"], wrong=st["wrong"],
                    escalation_rate=r, p_agree_given_correct=pac,
                    p_agree_given_wrong=paw, separation=pac - paw,
                    r_powered=r_ok, sep_powered=sep_ok,
                    verdict=cost_verdict, mechanism_verdict=mech_verdict,
                    underpowered=not (r_ok and sep_ok))

    persona_arm = summarise("persona A vs B", agreement(rows, "a1", "b1"))
    control_arm = summarise("plain resample (A vs A)", agreement(rows, "a1", "a2"))

    L = [f"# Agreement baseline — {args.benchmark}, model={args.model}, n={len(tasks)}",
         "",
         "Tool-free raw endpoint (ChatOAuth). `claude --print` was measured reading "
         "files on both tiers, so it cannot be used for a measurement whose dependent "
         "variable is agreement.",
         "",
         "| arm | scored | correct | wrong | escalation r | P(agree\\|correct) | P(agree\\|wrong) | separation | Echo economics | mechanism |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a in (persona_arm, control_arm):
        L.append(f"| {a['label']} | {a['scored']} | {a['correct']} | {a['wrong']} | "
                 f"**{a['escalation_rate']*100:.0f}%** | {a['p_agree_given_correct']*100:.0f}% | "
                 f"{a['p_agree_given_wrong']*100:.0f}% | {a['separation']*100:+.0f}pp | "
                 f"{a['verdict']} | {a['mechanism_verdict']} |")

    delta = persona_arm["escalation_rate"] - control_arm["escalation_rate"]
    powered = persona_arm["r_powered"] and control_arm["r_powered"]
    L += ["",
          f"**Break-even is r < {BREAK_EVEN*100:.0f}%** (Echo = 2 + 3r vs expensive-only = 3).",
          "",
          (f"**Persona delta: {delta*100:+.0f}pp escalation vs plain resampling.** "
           + ("Personas move the signal materially — the perturbation is doing work."
              if abs(delta) >= 0.05 else
              "Personas add ~nothing over resampling the same prompt. Echo could drop "
              "the persona machinery entirely and get simpler, which is a stronger "
              "claim.")) if powered else
          "**Persona delta: NOT REPORTABLE at this sample size.** Both arms are "
          "underpowered; the delta is noise. Re-run with enough tasks that the "
          f"wrong-answer cell alone holds >= {MIN_CELL}.",
          "",
          "`separation` = P(agree|correct) - P(agree|wrong). This is the mechanism's "
          "strength: it is how much agreement actually tells you about correctness. "
          "Near zero means agreement is not a difficulty signal on this benchmark, "
          "whatever the escalation rate is.",
          "",
          "Abstentions are excluded, not counted as disagreement — an unparseable or "
          "refused answer is missing data, and pooling it with disagreement would "
          "inflate r and understate Echo's profitability."]

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = RESULTS / f"{stamp}_agreement_baseline_{args.benchmark}_{args.model}_n{len(tasks)}"
    base.with_suffix(".json").write_text(json.dumps(
        dict(benchmark=args.benchmark, model=args.model, n=len(tasks),
             persona_arm=persona_arm, control_arm=control_arm,
             persona_delta=delta, rows=rows), indent=2))
    base.with_suffix(".md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {base.with_suffix('.md').name}")


if __name__ == "__main__":
    main()
