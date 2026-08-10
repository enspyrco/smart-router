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
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from benchmarks.bbh import load_bbh, score_bbh  # noqa: E402
from benchmarks.bbh_arms import PERSONA_A, PERSONA_B  # noqa: E402
from benchmarks.mmlu_pro import ALL_CATEGORIES, load_mmlu_pro, score_mmlu_pro  # noqa: E402
from chat_oauth import MODEL_IDS, ChatOAuth, ChatOAuthError, ModelAlias  # noqa: E402

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
# ($2) already cost a whole sonnet call ($2). The verdict therefore depends on
# a DATE (Carnot, cage-match #6 round 4).
#
# WHICH date is the whole ballgame, and an earlier revision of this comment
# claimed "the run date is recorded in the artifact" while recording no such
# thing (Tesla, round 5) — the same overclaim-in-a-comment as the "every alias
# is PINNED" line killed two rounds earlier. Made true rather than deleted:
# `pricing_as_of` and the resolved table/threshold are now written into every
# artifact, and `--from-json` resolves economics from the RECORDED date, not
# from the wall clock. Without that, replaying the same 210 rows on 2026-09-01
# would stamp PROFITABLE where the committed artifact says NOT PROFITABLE AT
# ANY r — and CANONICAL.md is the document telling readers to replay. The thesis
# is "profitability is a function of the intro window"; the wall-clock version
# implemented "profitability is a function of when you opened the file".
INTRO_PRICES_START = date(2026, 8, 1)
INTRO_PRICES_END = date(2026, 8, 31)

# THE MODEL TIER IS A CLOSED SET, SO IT GETS THE CLOSED TYPE (Carnot, round 7).
# `ModelAlias` already existed in chat_oauth and the pricing path — the code the
# PR's entire headline claim rests on — was still keyed by bare `str`. That is not
# a style nit; it produced a real defect. With a `str` domain, "not in ESCALATE_TO"
# means two unrelated things: a TYPO, or the TOP OF THE LADDER (opus has nothing
# above it). Unable to tell them apart, the old code returned one `float("nan")`
# for both — and NaN compares False against everything, so `e_hi < thr` and
# `e_lo > thr` both failed and the verdict fell through to
# "INDETERMINATE — production 95% CI [8%, 12%] straddles the nan% break-even".
# A confident sentence with a NaN in it, from an unrepresented state.
PRICES_BY_REGIME: dict[str, dict[ModelAlias, float]] = {
    "list pricing": {"haiku": 1.0, "sonnet": 3.0, "opus": 5.0},
    f"sonnet introductory pricing ({INTRO_PRICES_START} to {INTRO_PRICES_END})":
        {"haiku": 1.0, "sonnet": 2.0, "opus": 5.0},
}
LIST_PRICES = PRICES_BY_REGIME["list pricing"]
INTRO_PRICES = PRICES_BY_REGIME[
    f"sonnet introductory pricing ({INTRO_PRICES_START} to {INTRO_PRICES_END})"]

# The escalation ladder. Absence from this map is MEANINGFUL — it is the top tier.
ESCALATE_TO: dict[ModelAlias, ModelAlias] = {"haiku": "sonnet", "sonnet": "opus"}

# STARTUP INVARIANT: every alias the transport can produce must have a price in
# every regime, and every escalation target must itself be priced. Without this,
# adding a model to MODEL_IDS and forgetting its price is discovered at NaN time —
# i.e. inside a verdict — instead of at import.
for _regime, _table in PRICES_BY_REGIME.items():
    _unpriced = sorted(set(MODEL_IDS) - set(_table))
    if _unpriced:
        raise RuntimeError(f"{_regime}: no price for {_unpriced}; every alias in "
                           "MODEL_IDS must be priced in every regime")
_bad_targets = sorted({v for v in ESCALATE_TO.values()} - set(MODEL_IDS))
if _bad_targets:
    raise RuntimeError(f"ESCALATE_TO points at unknown model(s) {_bad_targets}")


@dataclass(frozen=True)
class Escalation:
    """`cheap` has a tier above it, so escalation economics are defined."""
    cheap: ModelAlias
    expensive: ModelAlias
    threshold: float          # Echo wins while r < threshold. May be <= 0.
    regime: str
    note: str

    @property
    def satisfiable(self) -> bool:
        return self.threshold > 0


@dataclass(frozen=True)
class TopTier:
    """`cheap` is the most expensive tier — there is nothing to escalate TO.

    A legitimate state, not an error, and emphatically not the same state as an
    unknown model. Unknown models are now unrepresentable: `--model` takes
    `choices`, `ChatOAuth.model` is a `ModelAlias`, and `break_even` rejects
    anything outside the closed set rather than inventing a float for it.
    """
    cheap: ModelAlias
    regime: str

    @property
    def note(self) -> str:
        return (f"{self.cheap} is the top tier @ {self.regime} — there is nothing "
                "to escalate to, so Echo's escalation economics do not apply")


BreakEven = Escalation | TopTier

# INPUT prices alone are sufficient here, which is a claim worth justifying
# rather than assuming. Total cost is in_tok*p_in + out_tok*p_out, and every
# tier — including intro sonnet — prices output at exactly 5x input. So p_out
# factors out and the break-even ratio is unchanged, PROVIDED the token mix is
# comparable across tiers. It is, since all arms answer the same benchmark item.
# If a future tier breaks the 5x ratio this shortcut dies with it.


def prices_on(day: date) -> tuple[dict[ModelAlias, float], str]:
    """(price table, regime label) in effect on `day`.

    BOUNDED AT BOTH ENDS. An earlier version returned intro pricing for any date
    <= the end of the window, including dates before the window opened — so
    replaying a pre-August datum would silently price it under a regime that did
    not yet exist (Maxwell, round 5). If a date is going to govern the verdict,
    the window needs both edges.
    """
    if INTRO_PRICES_START <= day <= INTRO_PRICES_END:
        return (INTRO_PRICES,
                f"sonnet introductory pricing ({INTRO_PRICES_START} to {INTRO_PRICES_END})")
    return (LIST_PRICES, "list pricing")


def break_even(cheap: ModelAlias, day: date) -> BreakEven:
    """Escalation economics for `cheap` on `day` — an `Escalation` or a `TopTier`.

    Returns a SEALED RESULT, not a float that might be NaN. The two states this
    function can be in are genuinely different — there is a tier above, or there
    isn't — and collapsing them into a sentinel is what let a NaN threshold reach
    a verdict string (Carnot, round 7). An unknown alias is no longer one of the
    states: it is rejected here, and made unreachable upstream by `--model`'s
    `choices` and `ChatOAuth.model: ModelAlias`.

    `day` is REQUIRED and has no wall-clock default. That is deliberate: the
    default was `datetime.now()`, which made a replay's verdict a function of
    when someone opened the file rather than of the data (Maxwell + Tesla,
    round 5 — found independently by two families). Callers must resolve the
    pricing date explicitly, from the artifact for a replay or from today for a
    fresh measurement, so the choice is always visible at the call site.
    """
    table, regime = prices_on(day)
    if cheap not in table:
        # Unrepresentable by construction; if it happens, the closed set has been
        # widened somewhere without widening the price tables. Fail loudly rather
        # than inventing a float, which is exactly what the old NaN did.
        raise ValueError(f"{cheap!r} is not a priced model alias — expected one of "
                         f"{sorted(table)}. The startup invariant should have caught "
                         "this; a price table and MODEL_IDS have drifted apart.")
    exp = ESCALATE_TO.get(cheap)
    if exp is None:
        return TopTier(cheap=cheap, regime=regime)

    c, e = table[cheap], table[exp]
    thr = (e - 2 * c) / e
    # Both regimes are always printed. A threshold that silently flips on
    # 2026-09-01 is exactly the kind of stale-artifact trap this file keeps
    # finding in itself, so the reader gets to see the flip coming.
    in_intro = INTRO_PRICES_START <= day <= INTRO_PRICES_END
    other_day = INTRO_PRICES_END + timedelta(days=1) if in_intro else INTRO_PRICES_START
    o_table, o_regime = prices_on(other_day)
    o_thr = (o_table[exp] - 2 * o_table[cheap]) / o_table[exp]
    alt = f" [under {o_regime}: {'UNSATISFIABLE' if o_thr <= 0 else f'r < {o_thr*100:.0f}%'}]"
    note = (f"{cheap}->{exp} @ {regime}: 2x{cheap} (${2*c}/MTok) already costs "
            f">= {exp} (${e}/MTok) — Echo CANNOT be profitable at this tier{alt}"
            if thr <= 0 else
            f"{cheap}->{exp} @ {regime}: r < {thr*100:.0f}% "
            f"(2x${c} + r*${e} < ${e}){alt}")
    return Escalation(cheap=cheap, expensive=exp, threshold=thr, regime=regime, note=note)
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
# A run is a measurement or it is a failure; it is not a measurement with a
# broken connection folded into the numerator.
#
# THIS IS A JUDGMENT CALL, NOT A MEASUREMENT, and it decides whether a run exists
# at all — so both failure directions are named rather than left to the reader
# (Maxwell, round 8). 429s are already retried inside the transport, so a call
# that still fails has exhausted backoff.
#   TOO LOW  -> a genuine transient burst aborts a good cohort; cost is a re-run.
#   TOO HIGH -> a partial outage survives as "excluded" rows, and while they no
#               longer enter r or McNemar, a large excluded set means the sample
#               is no longer the stratified one that was designed.
# 2% of 840 calls is ~17. Revisit against measured transient rates rather than
# tuning it to make a particular run pass.
TRANSPORT_ERROR_ABORT = 0.02
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
        # max(1, ...) silently turned --n 1..13 into 14 tasks (Carnot, round 8) —
        # a run that quietly delivers MORE than asked is as bad as one that delivers
        # less, because the artifact records the delivered n while the operator
        # remembers the requested one.
        if n < len(ALL_CATEGORIES):
            raise SystemExit(
                f"--n {n} is below the {len(ALL_CATEGORIES)} MMLU-Pro categories, so a "
                "stratified sample cannot give even one task per category. Use --n >= "
                f"{len(ALL_CATEGORIES)}, or --single-category to sample one category.")
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

        # A DEAD CONNECTION IS NOT A MODEL BEHAVIOUR (Tesla, round 8) — the last
        # and deepest layer of the laundering this file kept half-closing.
        #
        # Rounds 5-7 typed the transport error, recorded `error_kind`, and gated
        # the run at 2%. But `agreement()` never READ `error_kind`, so beneath the
        # fuse a transport failure still entered the statistic as an abstention —
        # and ASYMMETRICALLY, which is the part that makes it a real bug rather
        # than a rounding concern:
        #
        #     a1 dies  -> persona AND control both escalate   (concordant, harmless)
        #     b1 dies  -> ONLY persona escalates              (FALSE DISCORDANT PAIR)
        #
        # McNemar reads only discordant pairs, so a b1 outage manufactures
        # evidence in exactly the test the persona claim rests on. Verified: a task
        # where the model agreed everywhere but b1 died in transit yields b=1, c=0.
        #
        # Production policy ("escalate when the cheap answer is unusable") and
        # measurement ("do these models disagree?") are DIFFERENT QUANTITIES, and
        # this file was reading both off one meter. A task whose call died is not
        # evidence either way: we do not know whether the models would have agreed.
        # It is excluded from the arm and tallied separately.
        if (x.get("error_kind") == "transport") or (y.get("error_kind") == "transport"):
            st["transport_excluded"] += 1
            continue

        if x.get("answer") is None or y.get("answer") is None:
            # A genuine abstention: the call SUCCEEDED and its answer was
            # unparseable. Production Echo cannot accept that, so it escalates —
            # this one belongs in r.
            st["abstained"] += 1
            escalate[tid] = True
            continue
        st["scored"] += 1
        agree = x["answer"] == y["answer"]
        per_task[tid] = bool(agree)
        escalate[tid] = not agree
        st["agree"] += agree
        # `is True` / `is False`, not truthiness (Tesla, round 7). A `correct`
        # of None — a scorer that sets `answer` but leaves `correct` unset —
        # is falsy, so it silently landed in the WRONG cell and invented
        # mechanism mass. Neither cell is right for "unknown"; count it as an
        # abstention, which is the honest reading and the production policy.
        if x["correct"] is True:
            st["correct"] += 1
            st["agree_given_correct"] += agree
        elif x["correct"] is False:
            st["wrong"] += 1
            st["agree_given_wrong"] += agree
        else:
            st["indeterminate"] += 1
    return {"counts": st, "per_task": per_task, "escalate": escalate}


def summarise(label: str, res: dict, echo_accept: bool, be: BreakEven) -> dict:
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

    # MATCH ON THE SEALED RESULT. The old chain opened with `thr != thr` — a
    # hand-rolled isnan whose own comment read "NaN (unknown pair) or
    # unsatisfiable", i.e. it knew it was conflating two unrelated states and
    # printed NOT PROFITABLE AT ANY r for both. A top tier is not unprofitable;
    # it has no escalation economics at all. Pattern matching also means a third
    # BreakEven case added later cannot be silently swallowed by an `else`.
    if st["scored"] < MIN_SCORED:
        cost = f"INSUFFICIENT — only {st['scored']} scored (need {MIN_SCORED})"
    else:
        match be:
            case TopTier():
                cost = f"N/A — {be.note}"
            case Escalation(threshold=thr) if thr <= 0:
                cost = f"NOT PROFITABLE AT ANY r — {be.note}"
            case Escalation(threshold=thr) if e_hi < thr:
                cost = (f"PROFITABLE — production 95% upper bound "
                        f"{e_hi*100:.0f}% < {thr*100:.0f}%")
            case Escalation(threshold=thr) if e_lo > thr:
                cost = (f"NOT PROFITABLE — production 95% lower bound "
                        f"{e_lo*100:.0f}% > {thr*100:.0f}%")
            case Escalation(threshold=thr):
                cost = (f"INDETERMINATE — production 95% CI "
                        f"[{e_lo*100:.0f}%, {e_hi*100:.0f}%] "
                        f"straddles the {thr*100:.0f}% break-even")
            case _:
                raise AssertionError(f"unhandled BreakEven case: {type(be).__name__}")

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
    # ENDPOINT BOUNDS, because the half-width version was not guaranteed to be
    # what its own label claimed (Carnot + Tesla, round 6). Wilson intervals are
    # ASYMMETRIC — centred toward 1/2, not on the point estimate — so a
    # half-width about `pac` can be SMALLER than the true lower-side drop when
    # the cell sits near 1, which is exactly where the correct-cell lives here
    # (151/156: drop 4.08pp vs half-width 2.95pp). On this dataset the wrong-cell
    # term over-corrects in the other direction and the net stayed conservative
    # by 0.84pp — but that is two errors cancelling, not a property. "Over-covers"
    # was a claim the estimator did not carry.
    #
    # c_lo - w_hi IS guaranteed: the smallest plausible correct-cell rate minus
    # the largest plausible wrong-cell rate. Still not Newcombe and still not
    # exact coverage — but genuinely hard to claim rather than branded as such,
    # and on this data it is also TIGHTER (+4.49pp vs +3.65pp). Strictly better
    # on both axes, which is the tell that the old form was simply wrong.
    sep_lo, sep_hi = c_lo - w_hi, c_hi - w_lo
    band = f"[{sep_lo*100:.0f}pp, {sep_hi*100:.0f}pp]"
    if not echo_accept:
        mech = "n/a — first call is not Echo's accept path"
    elif st["correct"] < MIN_CELL or st["wrong"] < MIN_CELL:
        mech = None
    elif sep_lo > 0.10:
        mech = (f"mechanism holds — separation endpoint lower bound "
                f"{sep_lo*100:.0f}pp > 10pp (c_lo - w_hi; not an exact difference CI)")
    elif sep_lo > 0:
        mech = (f"separation positive but weak — endpoint bound {band} "
                "(c_lo - w_hi; conservative by construction, not an exact difference CI)")
    else:
        mech = ("agreement does NOT reliably predict correctness — "
                f"endpoint bound {band} includes 0")
    if mech is None:
        short = []
        if st["correct"] < MIN_CELL:
            short.append(f"{st['correct']} correct")
        if st["wrong"] < MIN_CELL:
            short.append(f"{st['wrong']} wrong")
        mech = f"INSUFFICIENT — {', '.join(short)} (need {MIN_CELL} each)"

    # COUNTED-BUT-INVISIBLE IS STILL A SILENT DROP (Maxwell + Tesla, round 8).
    # Round 7 stopped an indeterminate `correct` landing in the WRONG cell and
    # routed it to a counter that nothing ever read — trading a wrong number for
    # an invisible one, in a file whose whole ethic is that nothing vanishes
    # quietly. Both exclusions are now reported: `correct + wrong < scored` is
    # visible rather than a silently thinned stratum.
    return dict(label=label, scored=st["scored"], abstained=st["abstained"],
                indeterminate=st["indeterminate"],
                transport_excluded=st["transport_excluded"],
                correct=st["correct"], wrong=st["wrong"],
                escalation_rate=r, escalation_rate_ci=[r_lo, r_hi],
                escalation_rate_abstain_escalates=r_escalate,
                escalation_rate_production_ci=[e_lo, e_hi], echo_accept_path=echo_accept,
                p_agree_given_correct=pac, p_agree_given_wrong=paw,
                separation=pac - paw, separation_ci=[sep_lo, sep_hi],
                verdict=cost, mechanism_verdict=mech)


# ECHO_ACCEPT (the 4th field) marks arms whose FIRST key is the answer production
# Echo would accept. P(agree|correct) on any other ordering is not the Echo
# premise, so its mechanism verdict is suppressed rather than quietly cited
# (Tesla). Module scope so the replay guard in main() can derive the set of calls
# a datum must contain from the same definition the analysis uses — deriving it
# from one source keeps the guard from drifting out of step with the arms.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["mmlu_pro", "bbh"], default="mmlu_pro")
    ap.add_argument("--n", type=int, default=210,
                    help="target total; stratified mode rounds DOWN to a whole "
                         "number per category (n // 14 each across 14 categories)")
    # choices, not a free string: an unknown --model previously sailed past
    # argparse, made break_even() return NaN, and only failed later inside
    # pydantic — so the transport and the pricing logic could disagree about
    # which model was being measured before anything complained (Carnot, round 5).
    ap.add_argument("--model", default="haiku", choices=sorted(MODEL_IDS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--single-category", action="store_true",
                    help="take the first n rows (ONE category); default is stratified")
    ap.add_argument("--from-json", default=None,
                    help="re-analyse a saved run; spends no calls and needs no other flags")
    # The stable canonical name is now WRITTEN BY THE SCRIPT, not produced by a
    # manual `mv`. Round 5 gave the canonical analysis an untimestamped filename so
    # it would have no mtime story to tell — then left the rename as a hand step,
    # so following CANONICAL.md's own regenerate instruction produced a timestamped
    # file and quietly aged the canonical copy. That is the same prose-gate-vs-
    # enforced-gate defect this PR keeps closing, reintroduced by the fix for it
    # (Maxwell + Carnot, round 6).
    ap.add_argument("--write-canonical", action="store_true",
                    help="write results/CANONICAL_ANALYSIS.md instead of a timestamped "
                         "analysis. Use when regenerating the analysis CANONICAL.md points at.")
    ap.add_argument("--pricing-as-of", default=None, metavar="YYYY-MM-DD",
                    help="override the pricing date used for the economics verdict. "
                         "Default: the artifact's recorded pricing_as_of on --from-json, "
                         "else today (UTC). Set this to ask 'what would this datum say "
                         "under a different pricing regime' — explicitly, in the artifact.")
    args = ap.parse_args()

    # A flag that is ACCEPTED AND IGNORED is worse than one that does not exist:
    # the operator has positive evidence they asked for the canonical name and no
    # evidence they did not get it. --write-canonical only means anything on the
    # replay path, so demand the combination rather than silently dropping it
    # (Maxwell, round 7 — the same read-as-active-while-doing-nothing class as the
    # fail-open guard).
    if args.write_canonical and not args.from_json:
        raise SystemExit(
            "--write-canonical applies to --from-json only. A fresh measurement mints a\n"
            "new datum and its own analysis; promoting one to canonical is a deliberate\n"
            "act — record the run, then replay it with --write-canonical.")

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

        # FAIL CLOSED ON A SCHEMA-INCOMPLETE DATUM (Carnot, round 5).
        # The retained n=150 artifacts predate the fourth call and have no `b2`.
        # Replaying one used to SUCCEED: `agreement()` counted every missing b2 as
        # an abstention, so the `persona-B self` arm was manufactured out of
        # nothing (scored=0, abstained=150, and a meaningless "r (abstain
        # escalates) = 100%") — and the run exited 0 while REPRINTING the +48pp
        # physics-only separation that this whole review exists to have retracted,
        # into a freshly-timestamped file that looks like a current analysis.
        # CANONICAL.md says those files must not be cited for any number; that was
        # a PROSE gate against a script that cheerfully regenerated them. This is
        # the enforced one.
        # THE QUANTIFIER IS THE WHOLE GUARD (Maxwell + Carnot + Tesla, round 6 —
        # all three families independently). The first version asked
        # `required - set().union(...)`, i.e. "does ANY row have b2" — the dual of
        # what is needed. The n=150 files were blocked only because ZERO rows have
        # b2; total absence is a lucky special case, not proof the seal holds. One
        # complete row among 209 incomplete ones would have passed the guard and
        # let `agreement()` fabricate the arm from abstentions exactly as before.
        # Partial coverage is not exotic — this transport raises per call, so a
        # mid-run quota lapse or transport fault produces precisely that shape.
        # The invariant is per-row: EVERY row carries EVERY required call.
        required = {k for _, k1, k2, _ in ARMS for k in (k1, k2)}
        if not rows:
            raise SystemExit(
                f"REFUSING TO REPLAY {args.from_json}: the datum contains NO rows. "
                "An empty datum is a different fault from an incomplete one, and the "
                "guard used to render it as \"0/0 rows are missing required call(s)\" "
                "with a dangling empty sample (Maxwell, round 7).")
        # PRESENCE OF A KEY IS NOT PRESENCE OF A CALL (Tesla, round 7). Round 6
        # fixed "any row has b2"; the dual left open was "the key exists but the
        # payload is a tombstone" — a row of {"a1": {}, "a2": {}, ...} satisfied
        # `required - set(r)` and then fabricated the arm from abstentions exactly
        # as before. Sealed door, latched window. A call counts only if it carries
        # a scoring outcome or an explicit recorded error.
        def _hollow(r, k):
            # ANY-ONE-OF was too permissive (Carnot, round 8): {"answer": "A"} passed
            # validation and then `agreement()` raised KeyError on x["correct"]. A call
            # is either a SCORED result (both answer and correct present) or a RECORDED
            # FAILURE (an error). Nothing else is a call.
            call = r.get(k)
            if not isinstance(call, dict):
                return True
            scored = {"answer", "correct"} <= set(call)
            failed = bool({"error", "error_kind"} & set(call))
            return not (scored or failed)
        short = {tid: sorted(k for k in required if k not in r or _hollow(r, k))
                 for tid, r in rows.items()
                 if any(k not in r or _hollow(r, k) for k in required)}
        if short:
            sample = list(short.items())[:3]
            raise SystemExit(
                f"REFUSING TO REPLAY {args.from_json}: {len(short)}/{len(rows)} rows are "
                f"missing required call(s). This analysis needs {sorted(required)} on "
                f"EVERY row. Sample: " + "; ".join(f"{t}->missing {m}" for t, m in sample) +
                ". A datum from an older arm design (see results/CANONICAL.md), or a "
                "partially-failed run. Re-analysing it would fabricate the missing arm "
                "from abstentions and reprint superseded numbers under a fresh timestamp.")

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

        # THE TRANSPORT'S TYPED ERRORS WERE BEING UNTYPED HERE (Tesla, round 7).
        # chat_oauth spent five rounds making transport faults raise
        # ChatOAuthError specifically so they could NOT be laundered into the
        # dependent variable — and then this blanket `except Exception` wrote
        # {answer: None}, which `agreement()` reads as an abstention. Auth death,
        # endpoint shape faults, a mid-cohort outage: all became soft parse
        # misses, inflating production `r (abstain escalates)`, with the process
        # still exiting 0. All that fail-closed work, undone at the one boundary
        # that mints the datum.
        #
        # Transport faults and scoring faults are different failures and are now
        # recorded as such (`error_kind`), then counted and GATED below.
        def work(task):
            out = {}
            for key, persona in (("a1", PERSONA_A), ("a2", PERSONA_A),
                                 ("b1", PERSONA_B), ("b2", PERSONA_B)):
                try:
                    raw = call(model, persona, task["prompt"])
                    ok, parsed = score(raw, task)
                    # RAW is persisted so the datum outlives this run's parser.
                    #
                    # HONEST SCOPE (Tesla, round 8): persisting it is necessary but
                    # not sufficient, and an earlier version of this comment implied
                    # the benefit was already realised. It is not — `--from-json`
                    # re-AGGREGATES the frozen `answer`/`correct`; it does not
                    # re-invoke `score()` on `raw`. Re-scoring needs the benchmark
                    # items (prompt + gold answer), which the datum does not carry,
                    # so it is a real feature and not a one-liner. What persisting
                    # raw buys TODAY is auditability — you can read what the model
                    # actually said. What it does NOT yet buy is parser-drift
                    # replay. Do not claim the second from the first.
                    out[key] = {"correct": ok, "answer": parsed, "raw": raw}
                except ChatOAuthError as exc:
                    out[key] = {"correct": None, "answer": None, "raw": None,
                                "error": repr(exc)[:200], "error_kind": "transport"}
                except Exception as exc:                      # noqa: BLE001
                    out[key] = {"correct": None, "answer": None, "raw": None,
                                "error": repr(exc)[:200], "error_kind": "scoring"}
            return task["task_id"], out

        rows = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(work, t) for t in tasks]):
                tid, out = fut.result()
                rows[tid] = out
                print(".", end="", flush=True)
        print("\n")

        # EXIT CODE IS PART OF THE INSTRUMENT (Tesla, round 7). Previously a run
        # in which every single call failed still printed dots, wrote markdown,
        # stamped a live-looking artifact and exited 0. One bad task is noise; a
        # permanent fault mode singing on every task is not a measurement, and a
        # measurement harness that cannot say "this did not work" will eventually
        # be believed when it shouldn't be.
        calls = [c for r in rows.values() for c in r.values()]
        n_transport = sum(1 for c in calls if c.get("error_kind") == "transport")
        n_scoring = sum(1 for c in calls if c.get("error_kind") == "scoring")
        if calls and n_transport / len(calls) > TRANSPORT_ERROR_ABORT:
            raise SystemExit(
                f"ABORTING: {n_transport}/{len(calls)} calls ({n_transport/len(calls)*100:.0f}%) "
                f"failed at the TRANSPORT layer, above the {TRANSPORT_ERROR_ABORT*100:.0f}% "
                "ceiling. These are not abstentions — folding them into r would report a "
                "broken connection as model disagreement. Nothing was written; fix the "
                "transport and re-run.")
        if n_transport or n_scoring:
            print(f"  NOTE: {n_transport} transport / {n_scoring} scoring failures "
                  f"across {len(calls)} calls — recorded with `error_kind`, and "
                  "escalated by production policy rather than silently dropped.\n")

    # ---- Resolve the pricing date. Precedence, most explicit first. ----
    # The economics verdict must be a pure function of (rows, pricing_as_of).
    # `break_even` has no wall-clock default any more, so this is the only place
    # the date is chosen and it is always recorded into the artifact below.
    pricing_src = ""
    if args.pricing_as_of:
        pricing_as_of = date.fromisoformat(args.pricing_as_of)
        # Bounded. An unbounded override silently priced 1999-01-01 under "list
        # pricing" and recorded a confident threshold for a regime that never
        # existed — in a file whose entire thesis is that recorded provenance is
        # what makes a number citable later (Maxwell, round 6).
        if not (date(2025, 1, 1) <= pricing_as_of <= date(2030, 1, 1)):
            raise SystemExit(
                f"--pricing-as-of {pricing_as_of} is outside the range these price "
                "tables describe (2025-01-01 to 2030-01-01). The tables encode one "
                "known intro window and one list regime; a date outside that span "
                "would produce a confident threshold for pricing nobody verified.")
        pricing_src = "--pricing-as-of (operator override)"
    elif args.from_json and saved.get("pricing_as_of"):
        pricing_as_of = date.fromisoformat(saved["pricing_as_of"])
        pricing_src = f"recorded in {Path(args.from_json).name}"
    elif args.from_json:
        # Legacy datum written before this field existed (including the current
        # canonical one). Infer from the filename's UTC stamp rather than the wall
        # clock: the stamp is when the artifact was written, which is the closest
        # honest proxy for when its prices were in force. Say so out loud — an
        # inferred date must never look like a recorded one.
        stem = Path(args.from_json).name[:8]
        try:
            pricing_as_of = datetime.strptime(stem, "%Y%m%d").date()
        except ValueError:
            raise SystemExit(
                f"{args.from_json} records no `pricing_as_of` and its filename has no "
                "YYYYMMDD stamp to infer one from. Pass --pricing-as-of explicitly; "
                "economics must not fall back to the wall clock.")
        pricing_src = f"INFERRED from filename stamp (artifact predates `pricing_as_of`)"
        print(f"  NOTE: pricing date {pricing_as_of} {pricing_src}\n")
    else:
        pricing_as_of = datetime.now(timezone.utc).date()
        pricing_src = "measurement date (fresh run, UTC)"

    price_table, price_regime = prices_on(pricing_as_of)

    arms = {label: agreement(rows, k1, k2) for label, k1, k2, _ in ARMS}
    ECHO_ACCEPT = {label: ok for label, _, _, ok in ARMS}
    be = break_even(model_name, pricing_as_of)
    summaries = {k: summarise(k, v, ECHO_ACCEPT[k], be) for k, v in arms.items()}

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

    # SURFACE A TWO-SOURCE CONFLICT, DO NOT SILENTLY TIE-BREAK. Legacy artifacts
    # carry a flat `mcnemar` field whose b/c can be INVERTED relative to what this
    # code computes — the canonical datum stores {b:15, c:7} against a recomputed
    # b=7, c=15 (Tesla, round 6). It went unnoticed for rounds because McNemar's p
    # is symmetric: p=0.134 agrees while the effect direction, 68% vs 32%, does
    # not. The field is no longer written, but a datum that has one must say so out
    # loud rather than let the next reader quote whichever surface they opened.
    if args.from_json:
        legacy = saved.get("mcnemar")
        if legacy and (legacy.get("b"), legacy.get("c")) != (b, c):
            print(f"  WARNING: {Path(args.from_json).name} carries a legacy `mcnemar` "
                  f"field b={legacy.get('b')} c={legacy.get('c')}, but the headline pair "
                  f"recomputes to b={b} c={c} — INVERTED. McNemar's p is symmetric so "
                  f"this hides in the p-value; the EFFECT direction differs. The datum's "
                  f"field is stale; this analysis is authoritative. Cite `mcnemar_tests`, "
                  f"never the flat field.\n")

    L = [f"# Agreement baseline — {benchmark}, model={model_name}, n={len(rows)}", "",
         f"Generation config: `{gen_cfg}`",
         f"Personas sha256[:12]: `{personas_sha}`",
         f"Categories: `{categories}`", "",
         "Tool-free raw endpoint (ChatOAuth). `claude --print` was measured reading "
         "files on both tiers, so it cannot be used for a measurement whose dependent "
         "variable is agreement.", "",
         "| arm | scored | correct | wrong | indet | abstained | txp-excl | r | r 95% CI | "
         "r (abstain escalates) | P(agree\\|correct) | P(agree\\|wrong) | separation | "
         "economics | mechanism |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in summaries.values():
        lo, hi = a["escalation_rate_ci"]
        L.append(f"| {a['label']} | {a['scored']} | {a['correct']} | {a['wrong']} | "
                 f"{a['indeterminate']} | {a['abstained']} | {a['transport_excluded']} | "
                 f"**{a['escalation_rate']*100:.0f}%** | "
                 f"[{lo*100:.0f}%, {hi*100:.0f}%] | "
                 f"{a['escalation_rate_abstain_escalates']*100:.0f}% | "
                 f"{a['p_agree_given_correct']*100:.0f}% | {a['p_agree_given_wrong']*100:.0f}% | "
                 f"{a['separation']*100:+.0f}pp | {a['verdict']} | {a['mechanism_verdict']} |")

    L += ["", "## Do personas beat plain resampling?", "",
          "Three McNemar tests, not one. Every comparison INVOLVING THE PERSONA ARM shares "
          "a call with whatever it is compared against, so a single such test cannot distinguish "
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
    # COMPARE THE EFFECTS, NOT THE SIGNIFICANCE DECISIONS. An earlier version
    # decided concordance with `(hp < 0.05) == (replicate_p < 0.05)` — which
    # would call p=0.049 and p=0.051 a DISAGREEMENT and p=0.134 and p=0.361 an
    # AGREEMENT despite a 2.7x spread. That is a threshold gate wearing an
    # agreement claim: exactly the sin removed from economics (now a Wilson
    # bound) and mechanism (now an interval), reintroduced one verdict over, in
    # the same commit that claimed to have fixed the class (Maxwell, round 5).
    # McNemar's effect measure is the discordant split b/(b+c); no difference is
    # b/(b+c) = 0.5. Two tests concur when their intervals on that split overlap.
    _, hb_, hc_, _, _ = persona_results[0]
    _, rb_, rc_, _, _ = persona_results[1]
    # A test with no discordant pairs has no effect estimate — b/(b+c) is 0/0.
    # The headline prose already handled that case; this block, added in the same
    # round, divided straight through (Carnot, round 6). An all-concordant replay
    # is unlikely but perfectly possible, and crashing on it violates the
    # fail-closed doctrine the rest of the file is built on.
    if not (hb_ + hc_) or not (rb_ + rc_):
        which = "headline" if not (hb_ + hc_) else "b1-anchored replicate"
        concordance = (
            f"**Do the two anchors agree?** Not comparable — the {which} test has no "
            "discordant pairs, so its effect (the discordant split b/(b+c)) is 0/0 and "
            "undefined. With no disagreement to measure, neither concordance nor "
            "tension can be claimed in either direction.")
    else:
        h_lo, h_hi = wilson(hb_, hb_ + hc_)
        r_lo, r_hi = wilson(rb_, rb_ + rc_)
        overlap = (h_lo <= r_hi) and (r_lo <= h_hi)
        both_include_null = (h_lo <= 0.5 <= h_hi) and (r_lo <= 0.5 <= r_hi)
        concordance = (
            "**Do the two anchors agree?** Compare the effects, not the p-values: "
            "McNemar's effect is the discordant split b/(b+c), with 0.5 meaning no "
            f"difference. Headline {hb_}/{hb_+hc_} = "
            f"{hb_/(hb_+hc_)*100:.0f}% [{h_lo*100:.0f}%, {h_hi*100:.0f}%]; "
            f"b1-anchored replicate {rb_}/{rb_+rc_} = "
            f"{rb_/(rb_+rc_)*100:.0f}% [{r_lo*100:.0f}%, {r_hi*100:.0f}%]. "
            + ("The intervals OVERLAP" if overlap else "The intervals DO NOT overlap")
            + (" and both contain 0.5" if both_include_null else "")
            + ", so the two anchors are "
            + ("consistent with each other" if overlap else "in tension")
            + ". Consistency across anchors is what licenses reading this as a "
              "statement about personas rather than about which call the two arms "
              "happen to share; tension would mean the common mode is driving the "
              "answer and neither number should be quoted.")
    L += ["", concordance,
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
          "- The separation interval is an ENDPOINT bound: `c_lo - w_hi` to `c_hi - w_lo`, "
          "the smallest plausible correct-cell rate minus the largest plausible wrong-cell "
          "rate. It is conservative BY CONSTRUCTION, not by assertion. An earlier version "
          "summed Wilson half-widths and called that over-covering; Wilson intervals are "
          "asymmetric, so that form could under-cover near 0 or 1 — it held here only "
          "because two errors cancelled. Still not Newcombe and still not exact coverage.",
          "- agree(B,B) IS now measured, as the `persona-B self` arm. Wu's round-1 point "
          "was that B's self-agreement was merely ASSUMED to mirror A's; the "
          "`control vs B-self` row above tests that symmetry directly rather than "
          "leaving it to eyeball."]

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # A REPLAY EMITS AN ANALYSIS, NOT A COPY OF THE DATUM (Maxwell, round 6).
    # `--from-json` used to write a fresh JSON carrying the entire `rows` payload —
    # 1.5 MB of data already on disk, byte-identical. That is precisely how the
    # "three siblings that were never siblings" chain grew (each a replay of the
    # last, all sharing sha256 97a56d9f44df3b44), and leaving the mechanism in
    # place meant the chain could grow again the same way. A re-analysis now
    # writes only the markdown plus a pointer to the datum it read.
    if args.from_json:
        out_md = (RESULTS / "CANONICAL_ANALYSIS.md" if args.write_canonical
                  else RESULTS / f"{stamp}_analysis_{benchmark}_{model_name}_n{len(rows)}.md")
        out_md.write_text("\n".join(L) + f"\n\n---\n\n*Analysis of `{Path(args.from_json).name}` "
                          f"(datum unchanged; this file is derived). Pricing as of "
                          f"{pricing_as_of} — {pricing_src}.*\n")
        print("\n".join(L))
        print(f"\nWrote {out_md.name} (analysis only — the datum was not copied)")
        return

    base = RESULTS / f"{stamp}_agreement_baseline_{benchmark}_{model_name}_n{len(rows)}"
    base.with_suffix(".json").write_text(json.dumps(
        dict(benchmark=benchmark, model=model_name, n=len(rows),
             generation_config=gen_cfg, categories=categories, personas_sha=personas_sha,
             source_json=args.from_json, arms=summaries,
             # ECONOMICS INPUTS ARE PART OF THE DATUM (Tesla + Maxwell, round 5).
             # Everything the verdict depends on is written down, so a replay
             # reproduces the verdict instead of recomputing it against whatever
             # day it happens to be run. `pricing_as_of_source` records HOW the
             # date was chosen, so an inferred date can never be mistaken for a
             # recorded one on the next hop of a replay chain.
             pricing_as_of=pricing_as_of.isoformat(),
             pricing_as_of_source=pricing_src,
             price_regime=price_regime, price_table=price_table,
             # The sealed result, flattened for JSON. `break_even_kind` names WHICH
             # case, so a consumer can tell 'top tier, economics N/A' from
             # 'unsatisfiable threshold' — the distinction the NaN sentinel destroyed.
             break_even_kind=type(be).__name__,
             break_even_threshold=(be.threshold if isinstance(be, Escalation) else None),
             break_even_note=be.note,
             # ALL THREE McNemar rows, not just the headline. The markdown is a
             # projection; the rows are the experiment, so the tests that justify
             # the persona claim belong in the machine record too — a consumer of
             # the JSON could previously see only the coupled pair and had no way
             # to know the replicate or the disjoint symmetry test existed (Tesla).
             # The legacy flat `mcnemar` field is GONE, not kept for compatibility.
             # Keeping it was the bug: the canonical datum stores {b:15, c:7} while
             # the analysis computes b=7, c=15 — INVERTED (Tesla, round 6). It hid
             # because McNemar's p is symmetric, so p=0.134 matched on both sides
             # while the effect direction was opposite (68% vs 32%). A compatibility
             # field that can silently disagree with the analysis is a second source
             # of truth, which is the thing this whole PR keeps deleting. One field,
             # named, with its arm ordering explicit in `name`.
             mcnemar_tests=[dict(name=n_, b=b_, c=c_, p=p_, note=note_)
                            for n_, b_, c_, p_, note_ in persona_results],
             rows=rows), indent=2))
    base.with_suffix(".md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {base.with_suffix('.md').name}")


if __name__ == "__main__":
    main()
