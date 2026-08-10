#!/usr/bin/env python3
"""Measure the numbers Echo's economics actually rest on.

Echo accepts the cheap answer when two cheap calls agree and escalates when they
disagree. Three quantities decide whether that is a good idea:

1. ESCALATION RATE r -- how often the two cheap calls disagree.
   Echo's cost is 2*cheap + r*expensive, so Echo wins while r < (exp - 2*cheap)/exp.

   THE THRESHOLD IS NOT A CONSTANT AND NOT EVEN A CONSTANT OVER TIME. At list
   prices haiku->sonnet is $1 vs $3, giving 2 + 3r < 3 => r < 1/3. But Sonnet 5
   is in an introductory window at $2/MTok through 2026-08-31, and at $1 vs $2
   the threshold is (2 - 2)/2 = 0: two haiku calls already cost a whole sonnet
   call, so Echo CANNOT pay for itself at that tier until the intro price
   lapses. See `break_even`, which resolves the regime from the run date.

   The verdict uses the Wilson UPPER BOUND on r, not the point estimate: near the
   break-even, "r=30% at n=60" has an interval that swallows the threshold, and
   printing PROFITABLE off that is a sample-size gate pretending to be a
   confidence statement (Wu, cage-match #6).

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

FOUR CALLS, FOUR ARMS. Per task: a1, a2 (PERSONA_A), b1, b2 (PERSONA_B).

    persona         a1 vs b1    A vs B
    control         a1 vs a2    A vs A -- plain resampling
    cross           a2 vs b1    A vs B; shares a2 with control, b1 with persona
    persona-B self  b1 vs b2    B vs B -- DISJOINT from control: {b1,b2} n {a1,a2} = {}

An earlier revision called `a2 vs b1` "unshared" and claimed it shared nothing
with the control. That was mathematically FALSE -- the control is a1-vs-a2, so
a2-vs-b1 shares a2. With only THREE calls no pair can be disjoint from the
control; every arm is common-mode coupled, and renaming the pipe does not stop
the leak (Carnot + Tesla, cage-match #6 round 2). The fourth call is what buys a
genuinely disjoint arm, and it simultaneously answers Wu's round-1 point that
agree(B,B) was never measured.

Sharing a1 between persona and control remains a PAIRED design: it reduces the
variance of their difference and does not bias the null of no persona effect,
which is why McNemar on the discordant pairs is the right test. What it does NOT
do is make equal marginal rates across arms evidence of independence.

MECHANISM IS ONLY MEANINGFUL ON ECHO'S ACCEPT PATH. P(agree|correct) conditions
on the FIRST call of a pair, which is Echo's accepted answer for the persona and
control arms only. For cross and persona-B-self it is not, so their mechanism
verdict is suppressed as n/a rather than quietly cited (Tesla).

ECONOMICS GATES ON THE PRODUCTION RATE. Verdicts use r-with-abstention-escalating
and its Wilson interval, not the abstentions-dropped rate. Computing the honest
number and then stamping PROFITABLE off the flattering one was internally
inconsistent (Carnot + Tesla).

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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from benchmarks.bbh import load_bbh, score_bbh  # noqa: E402
from benchmarks.bbh_arms import PERSONA_A, PERSONA_B  # noqa: E402
from benchmarks.mmlu_pro import ALL_CATEGORIES, load_mmlu_pro, score_mmlu_pro  # noqa: E402
from chat_oauth import ChatOAuth  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Break-even follows the PRICE PAIR, not a constant. Echo costs 2*cheap + r*exp,
# so it wins while r < (exp - 2*cheap)/exp. Hardwiring 1/3 meant `--model sonnet`
# still stamped PROFITABLE against haiku->sonnet economics -- and the PR body
# itself notes sonnet->opus is UNSATISFIABLE (2*3 > 15... i.e. negative
# threshold), which the script would happily have blessed (Tesla, round 3).
# $/MTok, VERIFIED against Anthropic's published pricing on 2026-08-11. The
# earlier table was UNVERIFIED and wrong, and the README was right:
#
#   opus was 15.0 — that is SONNET'S OUTPUT price, mislabelled as OPUS INPUT.
#   Opus 5 input is $5.00. So sonnet->opus is 1.67x, exactly the README's "~1.7x,
#   not 3x", and the sonnet row's permissive r < 60% was an artefact of a
#   transcription error, not a real economics finding.
#
# Two authoritative sources disagreed and the table lost. Recorded because the
# round-3 handling (surface the conflict, mark the row provisional) was correct
# procedure but is NOT a substitute for going and reading the price list.
#
# SONNET 5 IS IN AN INTRODUCTORY WINDOW: $2.00/$10.00 per MTok through
# 2026-08-31, reverting to $3.00/$15.00. That is not a footnote — it moves the
# haiku->sonnet break-even from r < 33% to r < 0%, i.e. Echo CANNOT pay for
# itself at that tier while the intro price is live, because two haiku calls
# ($2) already cost a whole sonnet call ($2). The verdict must therefore depend
# on the RUN DATE, and the run date is recorded in the artifact (Carnot,
# cage-match #6 round 4).
INTRO_PRICES_END = date(2026, 8, 31)
LIST_PRICES = {"haiku": 1.0, "sonnet": 3.0, "opus": 5.0}
INTRO_PRICES = {"haiku": 1.0, "sonnet": 2.0, "opus": 5.0}
ESCALATE_TO = {"haiku": "sonnet", "sonnet": "opus"}

# INPUT prices alone are sufficient here, which is a claim worth justifying
# rather than assuming. Total cost is in_tok*p_in + out_tok*p_out, and every
# tier — including intro sonnet — prices output at exactly 5x input. So p_out
# factors out and the break-even ratio is unchanged, PROVIDED the token mix is
# comparable across tiers. It is, since all arms answer the same benchmark item.
# If a future tier breaks the 5x ratio this shortcut dies with it.


def prices_on(day: date) -> tuple[dict[str, float], str]:
    """(price table, regime label) in effect on `day`."""
    if day <= INTRO_PRICES_END:
        return (INTRO_PRICES, f"sonnet introductory pricing (through {INTRO_PRICES_END})")
    return (LIST_PRICES, "list pricing")


def break_even(cheap: str, day: date | None = None) -> tuple[float, str]:
    """(threshold, explanation) for the cheap->expensive pair. May be <= 0."""
    day = day or datetime.now(timezone.utc).date()
    table, regime = prices_on(day)
    exp = ESCALATE_TO.get(cheap)
    if exp is None or cheap not in table:
        return (float("nan"), f"no price pair known for {cheap!r}")
    c, e = table[cheap], table[exp]
    thr = (e - 2 * c) / e
    # Both regimes are always printed. A threshold that silently flips on
    # 2026-09-01 is exactly the kind of stale-artifact trap this file keeps
    # finding in itself, so the reader gets to see the flip coming.
    other_day = INTRO_PRICES_END + timedelta(days=1) if day <= INTRO_PRICES_END else INTRO_PRICES_END
    o_table, o_regime = prices_on(other_day)
    o_thr = (o_table[exp] - 2 * o_table[cheap]) / o_table[exp]
    alt = f" [under {o_regime}: {'UNSATISFIABLE' if o_thr <= 0 else f'r < {o_thr*100:.0f}%'}]"
    if thr <= 0:
        return (thr, f"{cheap}->{exp} @ {regime}: 2x{cheap} (${2*c}/MTok) already costs "
                     f">= {exp} (${e}/MTok) — Echo CANNOT be profitable at this tier{alt}")
    return (thr, f"{cheap}->{exp} @ {regime}: r < {thr*100:.0f}% "
                 f"(2x${c} + r*${e} < ${e}){alt}")
# The last two unexamined constants in the statistics path, and this PR's history
# is a series of unexamined defaults biting (Maxwell, round 4). They no longer
# DECIDE anything — economics gates on a Wilson bound and mechanism on a
# conservative interval, both of which widen honestly at small n — so these only
# decide whether a verdict is quoted AT ALL. That makes them a legibility gate,
# not an inference gate, and they are set accordingly:
#
#   MIN_CELL = 30    Below ~30 the Wilson interval on a cell proportion is wide
#                    enough that a separation bound is nearly always inconclusive.
#                    Printing "inconclusive" is fine; printing a pp figure readers
#                    will quote out of context is not. So suppress rather than emit.
#   MIN_SCORED = 60  Two cells of 30. Not independently motivated — it is MIN_CELL
#                    doubled, and named separately only so the two can diverge if a
#                    future benchmark has a lopsided correct/wrong split.
#
# Neither is a power calculation. If a claim ever turns on one of these numbers,
# that is the signal to do the power analysis, not to tune the constant.
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
        # Pass ALL_CATEGORIES explicitly and derive the per-category quota from
        # its LENGTH. The first version of this fix hardcoded `n // 14` while
        # letting `categories` default -- and that default is PILOT_CATEGORIES,
        # which is FIVE categories. So the "stratified" run silently produced 70
        # tasks over 5 categories while asking for 200 over 14: the same class of
        # bug as the physics-only sample it was written to fix, one layer down,
        # caught only because the run printed its category counts. Never infer a
        # loader's population from a constant you typed yourself.
        per_cat = max(1, n // len(ALL_CATEGORIES))
        tasks = load_mmlu_pro(categories=ALL_CATEGORIES, n_per_category=per_cat)
        return tasks, score_mmlu_pro
    return load_mmlu_pro(n=n), score_mmlu_pro


def agreement(rows: dict, k1: str, k2: str) -> dict:
    """Per-arm stats. Abstentions counted, never silently folded into either side."""
    st: Counter = Counter()
    per_task: dict[str, bool] = {}          # agreement, scored tasks only
    escalate: dict[str, bool] = {}          # PRODUCTION decision, every task
    for tid, r in rows.items():
        x, y = r.get(k1) or {}, r.get(k2) or {}
        if x.get("answer") is None or y.get("answer") is None:
            st["abstained"] += 1
            escalate[tid] = True            # unparseable cheap answer => escalate
            continue
        st["scored"] += 1
        agree = x["answer"] == y["answer"]
        per_task[tid] = bool(agree)
        escalate[tid] = not agree
        st["agree"] += agree
        if x["correct"]:
            st["correct"] += 1
            st["agree_given_correct"] += agree
        else:
            st["wrong"] += 1
            st["agree_given_wrong"] += agree
    return {"counts": st, "per_task": per_task, "escalate": escalate}


def summarise(label: str, res: dict, echo_accept: bool, thr: float, thr_note: str) -> dict:
    st = res["counts"]
    n_tot = st["scored"]
    n = n_tot or 1
    disagree = n_tot - st["agree"]
    r = disagree / n
    denom_esc = n_tot + st["abstained"]
    r_escalate = (disagree + st["abstained"]) / denom_esc if denom_esc else 0.0
    r_lo, r_hi = wilson(disagree, n_tot)
    # PRODUCTION policy gets the interval and the verdict. Gating on the
    # abstentions-dropped rate while the docstring admits that rate flatters Echo
    # was internally inconsistent -- computing the honest number then stamping the
    # flattering one (Carnot + Tesla, round 2).
    e_lo, e_hi = wilson(disagree + st["abstained"], denom_esc)

    pac = st["agree_given_correct"] / (st["correct"] or 1)
    paw = st["agree_given_wrong"] / (st["wrong"] or 1)

    if st["scored"] < MIN_SCORED:
        cost = f"INSUFFICIENT — only {st['scored']} scored (need {MIN_SCORED})"
    elif thr != thr or thr <= 0:            # NaN (unknown pair) or unsatisfiable
        cost = f"NOT PROFITABLE AT ANY r — {thr_note}"
    elif e_hi < thr:
        cost = f"PROFITABLE — production 95% upper bound {e_hi*100:.0f}% < {thr*100:.0f}%"
    elif e_lo > thr:
        cost = f"NOT PROFITABLE — production 95% lower bound {e_lo*100:.0f}% > {thr*100:.0f}%"
    else:
        cost = (f"INDETERMINATE — production 95% CI [{e_lo*100:.0f}%, {e_hi*100:.0f}%] "
                f"straddles the {thr*100:.0f}% break-even")

    # Separation gets an INTERVAL, not a threshold. Round 2 replaced the economics
    # sample-size gate with a Wilson bound on exactly this argument, then left the
    # mechanism verdict firing off `(pac - paw) > 0.10` with a cell-size guard —
    # the same asymmetry of rigor, one verdict over (Carnot rounds 2 AND 3).
    #
    # THIS IS NOT A 95% CI ON THE DIFFERENCE, AND CALLING IT ONE WAS THE BUG.
    # Wilson intervals are asymmetric, so summing half-widths about each point
    # estimate is neither Newcombe nor score nor bootstrap — it is a CONSERVATIVE
    # BOUND that over-covers (Tesla + Maxwell, round 4). Conservative is the right
    # direction to err for a claim we want to be hard to make, and erring this way
    # is what correctly downgraded "mechanism holds" to "positive but weak". But
    # the label has to match the estimator: it is reported as a conservative
    # bound, not quoted as exact coverage. A proper Newcombe/bootstrap difference
    # interval is the follow-up, and it can only WIDEN the set of claims we make,
    # never narrow it — so no current verdict depends on getting it.
    c_lo, c_hi = wilson(st["agree_given_correct"], st["correct"])
    w_lo, w_hi = wilson(st["agree_given_wrong"], st["wrong"])
    sep_lo, sep_hi = pac - paw - (c_hi - c_lo) / 2 - (w_hi - w_lo) / 2, \
                     pac - paw + (c_hi - c_lo) / 2 + (w_hi - w_lo) / 2
    band = f"[{sep_lo*100:.0f}pp, {sep_hi*100:.0f}pp]"
    if not echo_accept:
        mech = "n/a — first call is not Echo's accept path"
    elif st["correct"] < MIN_CELL or st["wrong"] < MIN_CELL:
        mech = None
    elif sep_lo > 0.10:
        mech = (f"mechanism holds — separation conservative lower bound "
                f"{sep_lo*100:.0f}pp > 10pp (not an exact 95% difference CI)")
    elif sep_lo > 0:
        mech = (f"separation positive but weak — conservative bound {band} "
                "(over-covers; not an exact 95% difference CI)")
    else:
        mech = ("agreement does NOT reliably predict correctness — "
                f"conservative bound {band} includes 0")
    if mech is None:
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
                escalation_rate_production_ci=[e_lo, e_hi], echo_accept_path=echo_accept,
                p_agree_given_correct=pac, p_agree_given_wrong=paw,
                separation=pac - paw, separation_ci=[sep_lo, sep_hi],
                verdict=cost, mechanism_verdict=mech)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["mmlu_pro", "bbh"], default="mmlu_pro")
    ap.add_argument("--n", type=int, default=210,
                    help="target total; stratified mode rounds DOWN to a whole "
                         "number per category (n // 14 each across 14 categories)")
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
            for key, persona in (("a1", PERSONA_A), ("a2", PERSONA_A),
                                 ("b1", PERSONA_B), ("b2", PERSONA_B)):
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

    # ECHO_ACCEPT marks arms whose FIRST key is the answer production Echo would
    # accept. P(agree|correct) on any other ordering is not the Echo premise, so
    # its mechanism verdict is suppressed rather than quietly cited (Tesla).
    ARMS = [
        ("persona (a1 vs b1)", "a1", "b1", True),
        ("control (a1 vs a2)", "a1", "a2", True),
        # NOT "unshared", and the name was mathematically false (Carnot + Tesla,
        # round 2). It shares a2 with the control AND b1 with the persona arm, so
        # it is an independent falsifier of NEITHER — the earlier comment named
        # only the a2 overlap, which understated the coupling by half (Kelvin,
        # round 4). With three calls no pair could be disjoint from the control at
        # all; the fourth call is what buys one.
        ("cross (a2 vs b1)", "a2", "b1", False),
        # Disjoint from the CONTROL: {b1,b2} n {a1,a2} = {}. Still shares b1 with
        # the persona arm — "disjoint" is a relation between two named arms, not a
        # property this row owns. It also answers Wu's round-1 point that
        # agree(B,B) was never measured, and supplies the only correctness split
        # not keyed on a1.
        ("persona-B self (b1 vs b2)", "b1", "b2", False),
    ]
    arms = {label: agreement(rows, k1, k2) for label, k1, k2, _ in ARMS}
    ECHO_ACCEPT = {label: ok for label, _, _, ok in ARMS}
    thr, thr_note = break_even(model_name)
    summaries = {k: summarise(k, v, ECHO_ACCEPT[k], thr, thr_note) for k, v in arms.items()}

    # Compare ESCALATION decisions (abstention counts as escalate), not agreement
    # over the both-parsed subset — otherwise differential parse failure silently
    # removes tasks from the test and pulls the arms together (Carnot round 3).
    def mcnemar_arms(x_label: str, y_label: str) -> tuple[int, int, float]:
        px, py = arms[x_label]["escalate"], arms[y_label]["escalate"]
        both = set(px) & set(py)
        b_ = sum(1 for t in both if px[t] and not py[t])
        c_ = sum(1 for t in both if py[t] and not px[t])
        return b_, c_, mcnemar(b_, c_)

    # ONE TEST WAS NOT ENOUGH, AND THE REASON IS STRUCTURAL (Tesla + Carnot, round 4).
    # The headline persona test compares a1-vs-b1 against a1-vs-a2. Both decisions
    # are functions of the same a1 draw, so they are positively correlated: good for
    # the VARIANCE of their difference (that is what pairing buys) but it THINS the
    # discordant set, and McNemar reads only discordant pairs. "No detectable effect"
    # is then partly the expected hum of a coupled circuit rather than a finding.
    #
    # With four calls NO pair-vs-pair comparison is fully disjoint: persona(a1,b1)
    # shares a1 with the control and b1 with the B-self arm. So "use the disjoint
    # arm" is not on the menu. What IS on the menu is running the persona test twice
    # with DIFFERENT shared anchors — if the conclusion survives both, it is not an
    # artefact of which call supplies the common mode.
    PERSONA_TESTS = [
        ("persona vs control (shared anchor: a1)",
         "persona (a1 vs b1)", "control (a1 vs a2)",
         "cross-persona vs A-self resampling"),
        ("persona vs B-self (shared anchor: b1)",
         "persona (a1 vs b1)", "persona-B self (b1 vs b2)",
         "same question, anchored on the OTHER call — a robustness replicate"),
        ("control vs B-self (DISJOINT: {a1,a2} n {b1,b2} = {})",
         "control (a1 vs a2)", "persona-B self (b1 vs b2)",
         "not a persona-effect test — asks whether the resampling BASELINE itself "
         "differs by persona, i.e. whether 'the control' is one baseline or two. "
         "This is Wu's symmetry question, and it is the only fully uncoupled pair."),
    ]
    persona_results = [(name, *mcnemar_arms(x, y), note)
                       for name, x, y, note in PERSONA_TESTS]
    # The headline pair stays first so the primary claim is unambiguous.
    b, c, p = persona_results[0][1], persona_results[0][2], persona_results[0][3]

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
          "Three McNemar tests, not one. Every pair-vs-pair comparison available from "
          "four calls shares at least one call, so a single test cannot distinguish "
          "\"no persona effect\" from \"the shared call thinned the discordant set\". "
          "Running the same question against two different anchors is the check that "
          "the conclusion is not an artefact of the coupling (Tesla + Carnot, round 4).",
          "",
          "| test | b | c | discordant | p | reads as |",
          "|---|---|---|---|---|---|"]
    for name, bb, cc, pp, note in persona_results:
        reads = ("no discordant pairs" if bb + cc == 0
                 else "**difference detected**" if pp < 0.05
                 else "no detectable difference")
        L.append(f"| {name} | {bb} | {cc} | {bb+cc} | {pp:.3f} | {reads} — {note} |")

    headline_name, hb, hc, hp, _ = persona_results[0]
    replicate_p = persona_results[1][3]
    L += ["", f"**Headline ({headline_name}):** b={hb}, c={hc}, **p={hp:.3f}**.", ""]
    if hb + hc == 0:
        L.append("No discordant pairs — the arms made identical decisions on every task.")
    elif hp < 0.05:
        L.append(f"**Personas DO change the escalation decision** (p={hp:.3f} over {hb+hc} "
                 "discordant tasks). A small net delta hides real churn in both directions.")
    else:
        L.append(f"**No detectable persona effect** (p={hp:.3f} over {hb+hc} discordant "
                 "tasks). This is a failure to reject, NOT proof of equivalence — for an "
                 "equivalence claim, pre-specify a margin and run TOST. Note the "
                 "discordant set is thinned by the shared `a1`, so power is lower than "
                 f"{hb+hc} paired tasks would suggest; see the replicate row.")
    agree_word = ("AGREES with" if (hp < 0.05) == (replicate_p < 0.05)
                  else "**DISAGREES with**")
    L += ["",
          f"The b1-anchored replicate {agree_word} the headline "
          f"(p={replicate_p:.3f} vs p={hp:.3f}). Agreement across anchors is the "
          "evidence that the result is about personas rather than about which call "
          "the two arms happen to share; disagreement would mean the common mode is "
          "driving the answer and neither number should be quoted.",
          "",
          "`cross (a2 vs b1)` shares a2 with the control AND b1 with the persona arm — "
          "it is NOT an independent falsifier on either side, and an earlier revision "
          "wrongly claimed it shared nothing. `persona-B self (b1 vs b2)` is disjoint "
          "from the control ({b1,b2} ∩ {a1,a2} = ∅) but still shares b1 with the "
          "persona arm; it is the agree(B,B) measurement that was previously missing.",
          "",
          "## Reading the numbers", "",
          "- **Economics gates on `r (abstain escalates)`** — the production rate — and "
          "specifically on the Wilson UPPER bound of that rate vs the break-even, never "
          "on a point estimate and never on the flattering abstentions-dropped `r`. An "
          "earlier version of this bullet named the wrong meter while the code used the "
          "right one, which is worse than either being wrong alone (Tesla, round 4).",
          "- `r` (abstentions dropped) is reported for comparison only. Excluding "
          "abstentions biases r DOWNWARD and flatters Echo, because parse failures "
          "correlate with hard tasks and hard tasks are where disagreement lives.",
          "- **The break-even is date-dependent.** Sonnet 5's introductory price runs "
          f"through {INTRO_PRICES_END}; the threshold in force is stated in the "
          "`economics` cell along with the threshold on the other side of that date.",
          "- `wrong` means the FIRST call was wrong — the answer Echo would accept. It is "
          "not the free-floating claim that the model reproduces its own errors.",
          "- **`separation` for `persona` and `control` is NOT two independent readings.** "
          "Both condition on `a1`, so they partition the SAME correctness split; quoting "
          "both as parallel mechanism evidence double-counts one first-call correctness "
          "frequency (Tesla, round 4). `persona-B self` conditions on `b1` and is the "
          "only row carrying an independent split.",
          "- The separation interval is a CONSERVATIVE bound (summed Wilson half-widths), "
          "not an exact 95% CI on the difference. It over-covers, so it makes claims "
          "harder rather than easier; a Newcombe/bootstrap difference interval is the "
          "follow-up and can only widen what we claim.",
          "- agree(B,B) IS now measured, as the `persona-B self` arm. Wu's round-1 point "
          "was that B's self-agreement was merely ASSUMED to mirror A's; the "
          "`control vs B-self` row above tests that symmetry directly rather than "
          "leaving it to eyeball."]

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
