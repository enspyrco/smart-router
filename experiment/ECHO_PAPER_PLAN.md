# Echo Paper Plan: When Does Cheap-Model Self-Agreement Predict Correctness?

Date: August 5, 2026

Status: design document. Supersedes the routing direction in
`ECHO_ROUTER_LEARNING_LOOP.md` §9–§10 and closes out the specialist question
raised in `SPECIALIST_MODEL_RESEARCH.md`.

All numbers below come from `results/20260803T064926Z_mmlu_pro_n125.jsonl`
(125 MMLU-Pro tasks, 5 categories, arms `haiku-only` / `sonnet-only` /
`echo-judge`) plus a cross-run reproducibility check over the full results
directory.

---

## 1. The Paper

> **When does cheap-model self-agreement predict correctness, and what is the
> cheapest routing policy built on it?**

Not "we built a router that saves money." That framing puts us in a crowded
field (FrugalGPT, RouteLLM, Expert Orchestration, Cost-Aware Contrastive
Routing) with a weaker instrument than any of them. The framing above puts us in
a narrower space where our data is the point.

The claim we intend to defend:

```text
Self-consistency works as a correctness signal in reasoning-bound domains
and fails in knowledge-bound ones — and any cascade built on it only pays
for itself if the verifier is free.
```

Two components, both supported by pilot data, both needing held-out
confirmation at larger n.

---

## 2. What The Experiment Does

For every task, collect three raw model outputs:

```text
Haiku persona A
Haiku persona B     <- the pair gives the agreement signal
Sonnet
```

Log the **answer text**, not just pass/fail. Then evaluate every routing policy
offline, in analysis, with zero further API calls.

This is the central design decision and it is what makes the study affordable.
A router is a decision function over facts we already have. We do not need to
run a `router` arm to score a router.

```text
old: one sweep per policy      -> 5 calls/task, one policy answered
new: one sweep, many policies  -> 3 calls/task, every policy answered,
                                  including policies not yet invented
```

---

## 3. Why This, And Why The Alternatives Were Dropped

### 3.1 What the pilot actually showed

Arms as run (n=125):

| arm         | pass  | cost/task | escalation |
| ----------- | ----- | --------- | ---------- |
| haiku-only  | 78.4% | 1.00      | —          |
| echo-judge  | 80.8% | 2.99      | 11.2%      |
| sonnet-only | 86.4% | 2.98      | —          |

Echo costs what Sonnet costs and is 5.6 points worse. It is strictly dominated.

### 3.2 Why — the verifier is not free

A Haiku judge call prices at 0.66 units under `cost_units.py`:

```text
Sonnet alone                          3.00
Echo, personas agreed  (1+1+0.66)     2.66
Echo, personas differed (2.66+3.00)   5.66
```

Even at a 0% escalation rate the accept path costs 2.66 against Sonnet's 3.00 —
an 11% ceiling on the entire idea. At the observed 11.2% escalation the average
lands at 2.99.

This is structural, not a tuning failure. It generalises: **a cascade whose
verifier costs a meaningful fraction of the expensive tier cannot pay for
itself.** That is a small, clean, defensible contribution on its own.

### 3.3 But the signal underneath is good

Which better predicts "will Haiku get this right"?

```text
Do the two Haiku personas agree?
  agreed    -> Haiku correct 85.5%
  disagreed -> Haiku correct 28.6%
  separation: 56.9 points

What subject is it?
  physics 100%  math 92%  chemistry 88%  law 56%  philosophy 56%
  separation: 44.0 points
```

Agreement is the stronger signal. Its weakness is coverage, not accuracy:

```text
agreement flag as a "Haiku will fail" detector
  precision 10/14 = 71%
  recall    10/26 = 38%
```

It is right when it fires and it rarely fires. That makes it a routing
_feature_, not a routing _strategy_ — which is exactly what the paper argues.

### 3.4 The domain dependence

Echo's failures, split by whether the specialist-style tiebreaker could even see
them (accept-path failures are agreements, so a tiebreaker never sees them):

| category   | echo failures | via accept path | via escalation |
| ---------- | ------------- | --------------- | -------------- |
| law        | 10            | 8               | 2              |
| philosophy | 10            | 6               | 4              |
| chemistry  | 2             | 2               | 0              |
| math       | 1             | 1               | 0              |
| physics    | 0             | 0               | 0              |

False accepts concentrate in law and philosophy and vanish in physics. This was
**predicted in writing** by `SPECIALIST_MODEL_RESEARCH.md` §4 before the data
landed, which makes it a confirmed prediction rather than a post-hoc pattern.

Mechanism: personas vary in reasoning path, so they diverge on reasoning slips
but agree confidently on shared knowledge gaps. Law and philosophy are
knowledge- and interpretation-bound; physics and maths are reasoning-bound.

### 3.5 Policies simulated on the pilot data

Evaluated offline on the 124 tasks with complete rows:

| policy                     | pass      | cost/task |
| -------------------------- | --------- | --------- |
| always Haiku               | 79.0%     | 1.00      |
| always Sonnet              | 87.1%     | 3.00      |
| **C: route by subject**    | **87.9%** | **1.81**  |
| D: Echo alone, free judge  | 83.1%     | 2.34      |
| E: subject, then agreement | 88.7%     | 2.43      |
| oracle (upper bound)       | 90.3%     | 1.42      |

C and E both beat always-Sonnet on **both** axes. D — Echo on its own, even with
the judge cost removed — is still dominated by C. C vs E is one task apart,
which is inside the noise floor (§3.6), so C is the choice.

### 3.6 The noise floor

786 (task, arm) pairs across the results directory were measured more than once:

```text
outcome disagreed between runs: 53/786 = 6.7%
```

One in fifteen measurements flips on a rerun. At 25 questions per category one
question is 4 points, so **no 4-point category difference in the pilot is
real**. This number gates every claim in the paper and is reported in methods.

### 3.7 What was dropped, and why

| dropped                                      | reason                                                                                                                                                                                                                          |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Paid Haiku judge                             | Costs 0.66 units; is the whole reason Echo lands at 2.99 vs 2.98. Replace with `echo-lexical` (free, deterministic on MCQ).                                                                                                     |
| LLM difficulty classifier (loop doc Level 3) | Costs ~1 unit/task. Total available prize over the subject rule is oracle − C = 2.4 points and 0.39 units. It cannot pay for itself. Also: reliably telling a hard question from an easy one is nearly as hard as answering it. |
| Chemistry specialist                         | No gap to close (Haiku 88% vs Sonnet 84%) and no credible open model exists.                                                                                                                                                    |
| Philosophy specialist                        | Largest real gap (28 points) but no specialist model exists at all. Fix by routing to Sonnet.                                                                                                                                   |
| Law specialist as a route                    | Only domain with a candidate (Saul-7B). Retained as a one-off 25-call probe, disconnected from the main experiment. Registry inversion noted in §6.                                                                             |

---

## 4. Architecture

### 4.1 Harness changes

**Raw answer capture.** `TaskResult` (`run_pilot.py:457`) stores only `passed`,
`detail`, `wall_seconds`, `sub_calls`. Add an `extras` dict and widen the arm
contract from `(output, sub_calls)` to allow an optional third element —
backwards compatible, existing arms keep returning 2-tuples. `_load_prior`
(`scripts/run_mmlu_pro_pilot.py:89`) must round-trip the new field so `--resume`
does not silently drop it.

**A single `probe` arm.** Three calls — Haiku persona A, Haiku persona B, Sonnet
— logging all three raw answers and each one's independent pass/fail. Replaces
`haiku-only` + `sonnet-only` + `echo-judge` at 3 calls instead of 5.

**Token accounting.** `cost_units.py` currently _assumes_ 600 input / 250 output
per persona call and 1200 / 3 per judge call. Nothing is measured. Since the
paper's central claim is about cost, the measurement runs must record real token
counts — which means the API rather than `claude --print`, whose subprocess
interface does not surface usage.

### 4.2 Offline simulator

New `scripts/simulate_routers.py`. Reads probe rows; scores any policy; takes
`--fit` / `--holdout` so the subject table is learned on one half and reported
on the other. Ships with the policies in §3.5 plus the fixed reference points
(always-Haiku, always-Sonnet, oracle).

Validated against the existing n=125 file using `sub_calls` as the agreement
proxy before any new data is collected.

### 4.3 Data collection

```text
14 MMLU-Pro categories x 50 questions = 700 tasks
700 tasks x 3 calls                   = ~2,100 calls (~12 hours)
split 25 fit / 25 holdout per category
```

Per-category held-out numbers will be noisy (+/-10 points at n=25), but the
router's overall held-out score is measured over 350 tasks (+/-2.5 points), and
that is the number the claim rests on.

Run via `scripts/run_mmlu_pro_resumable.sh`. **Smoke-test the usage-limit abort
path on 30 tasks first** — the 3 August sweep hit a weekly subscription cap and
wrote 215 failure rows before anyone noticed. The detection strings were patched
afterwards (`run_mmlu_pro_pilot.py:31-46`) but have not been exercised at this
scale.

### 4.4 Transfer benchmark

BBH as currently configured is unusable:

```text
20260701T000922Z_bbh_n99   haiku 84.8%   sonnet 84.8%
20260701T050041Z_bbh_n99   haiku 100%    sonnet 100%
```

Both slices are saturated — Sonnet buys nothing over Haiku, so there is no
routing decision to make. Harder suites must be selected before BBH can serve as
the transfer test.

---

## 5. Outcomes

### 5.1 What we expect to find

1. The subject rule beats always-Sonnet on both axes **on held-out data**, at a
   smaller margin than the in-sample 87.9% / 1.81.
2. Agreement separation stays large on MMLU-Pro and varies by domain, high in
   physics/maths, low in law/philosophy.
3. Adding the agreement signal on top of the subject rule buys little — the
   oracle gap is only 2.4 points — so the honest recommendation is the simplest
   policy.

### 5.2 What counts as success

Success is **not** "our router wins." Success is a defensible answer to the
title question. Three of the four possible outcomes are publishable:

```text
agreement helps on held-out data      -> positive result, routing feature
agreement adds nothing over subject   -> negative result worth reporting;
                                         self-consistency is domain prior
                                         in disguise
separation is domain-dependent        -> the strongest outcome; refines the
                                         self-consistency literature
everything is inside the noise floor  -> the study was underpowered; report
                                         the power analysis and stop
```

### 5.3 Deliverables

```text
1. probe-based harness with raw-answer logging and measured tokens
2. simulate_routers.py + held-out evaluation
3. accuracy-vs-cost frontier plot, with always-Haiku / always-Sonnet /
   oracle as fixed reference points
4. reproducibility floor (6.7%) reported in methods
5. writeup
```

### 5.4 Honest positioning

Most individual components are known. Cascades for cost are FrugalGPT
(Chen et al. 2023); routing on a cost-accuracy frontier is RouteLLM and
successors; disagreement-under-resampling as a correctness signal is
self-consistency (Wang et al. 2022) and semantic entropy (Farquhar et al. 2024).

What is not well covered is **where the self-consistency signal breaks down by
domain**, and the practical corollary that a cascade's verifier must be free.
That is the contribution, and it is a workshop paper, not a main-track one.

Known weaknesses a reviewer will find, listed so they are not discovered
for us:

```text
- cost currently modelled from assumed token counts, not measured (4.1 fixes)
- single benchmark until BBH suites are re-selected (4.4)
- n=125 pilot is in-sample; the subject rule was read off the same data
  it was scored on, so that file is permanently training data
- separation is unstable across older runs: +28.6, +21.1, +11.1, -1.9,
  -14.3, +47.2, +53.0, +56.9. Strong on MMLU-Pro, unproven elsewhere.
  The weak cases have 2-12 tasks in the disagree group, so they are not
  strong counter-evidence — but they are not nothing either.
```

---

## 6. Note On The Specialist Registry

`benchmarks/specialists.py` registers law, math, and computer science.
Maths is where Haiku already scores 92% — no headroom to buy. Computer science
was not in the sweep. The two domains that need help (law, philosophy) have
either a poor candidate or none at all.

That inversion — **the domains with the best available specialists are the
domains with nothing to fix** — is itself the §10 result, and should be written
up as such rather than treated as an unfinished task.
