# Echo Literature Audit And Dead-End Log

Date: August 6, 2026

Purpose: record which papers were found, what each one already covers, and which
of our candidate research questions and design elements were **removed because
the work already exists**. Written so that a question killed here does not get
re-proposed in three weeks.

Companion to `ECHO_PAPER_PLAN.md`. Where the two disagree, this document is
newer.

---

## 1. Verification Levels

Not all citations below were read to the same depth. Each entry is tagged:

```text
[FULL]     full text fetched and interrogated against specific questions
[ABSTRACT] abstract / search summary only
[SEEN]     appeared in search results, title only, NOT read
[INHERITED] carried over from SPECIALIST_MODEL_RESEARCH.md, not re-verified here
```

Treat `[SEEN]` and `[INHERITED]` entries as leads, not evidence.

---

## 2. Papers Found

### 2.1 Consistency-gated cascades — the closest prior art

**Large Language Model Cascades with Mixture of Thought Representations for
Cost-Efficient Reasoning** — Yue et al., ICLR 2024, arXiv 2310.03094 `[FULL]`
https://arxiv.org/abs/2310.03094

This is Echo, published two years earlier. Uses "answer consistency of the
weaker LLM as a signal of question difficulty", proposes sampling and
consistency-checking methods, reports comparable accuracy to the strong model at
40% of cost. Code at https://github.com/MurongYue/LLM_MoT_cascade

What it does **not** do, verified against the full text:

```text
- no systematic sweep of sample count K
  (fixed K=20 for GPT-3.5, K=3 for GPT-4; one limited 20->40 robustness check)
- six datasets, ALL reasoning-bound:
  GSM8k, ASDIV, TabMWP (math), DATE, Navigate (symbolic), CREPE (causal)
- zero knowledge-heavy domains: no law, philosophy, history, MMLU
- no systematic analysis of how consistency reliability varies by domain
```

Consequence: the method's positive result has only ever been demonstrated in the
domains where our hypothesis predicts the signal works. Nobody has checked
whether it survives outside them.

**Resample or Reroute? Budget-Aware Test-Time Model Selection** —
arXiv 2607.08665 `[FULL]`
https://arxiv.org/html/2607.08665

The single most relevant paper. Given a budget and an imperfect verifier: resample
the committed model, or reroute to a stronger one? Bayesian marginal-correctness-
per-cost allocation over 11 open-weight models, 30 draws per (query, model) cell,
on GSM8K / MATH-500 / GPQA-Diamond / HumanEval+.

Their agreement verifier is our Echo: *"a drawn sample counts as verified when its
extracted answer matches another drawn sample, the query stops once any answer
reaches a consensus of A=2 draws."*

Critically, they observe the failure we observed, and explain it by **answer
format**:

> "On GPQA, a four-option multiple-choice benchmark, agreement is nearly
> uninformative — two wrong draws easily agree on the same letter."

What it does **not** do, verified against the full text:

```text
- "spurious consensus" is ASSERTED, never measured
  no false-consensus rates, no per-category breakdown, no quantification
- format and domain are CONFOUNDED and the confound is not acknowledged
  GSM8K = open-ended + math ; GPQA = multiple-choice + knowledge-heavy
  they never test open-ended-hard or multiple-choice-easy
- k is never swept (30 draws used as a fixed replay pool)
- no per-subject breakdown within any benchmark; GPQA is aggregate only
```

**AutoMix: Automatically Mixing Language Models** — Aggarwal, Madaan et al.,
NeurIPS 2024, arXiv 2310.12963 `[ABSTRACT]`
Few-shot self-verification of the small model's output, POMDP router over answer
confidence, >50% cost reduction. Adjacent to Echo's escalation logic.

### 2.2 Routing and cascades — the baselines

| Work | ID | Level | Why it matters |
|---|---|---|---|
| FrugalGPT | 2305.05176 | `[ABSTRACT]` | Origin of cheap-to-expensive cascading under a confidence gate |
| RouteLLM | 2406.18665 | `[ABSTRACT]` | Learned per-query router from preference data |
| Is Escalation Worth It? | 2605.06350 | `[SEEN]` | Decision-theoretic characterisation of cascades |
| Cluster, Route, Escalate | 2606.27457 | `[SEEN]` | Cost-aware cascaded serving |
| UCCI | 2605.18796 | `[SEEN]` | Calibrated uncertainty for cascade routing |
| RerouteGuard | 2601.21380 | `[SEEN]` | Adversarial risks in LLM routing — **check before building RQ3** |

### 2.3 Self-consistency and agreement as a correctness signal

| Work | ID | Level | Why it matters |
|---|---|---|---|
| Self-Consistency Improves CoT | 2203.11171 | `[ABSTRACT]` | Origin of sample-and-vote |
| Detecting hallucinations using semantic entropy | Nature 630:625–630 (2024) | `[ABSTRACT]` | Resample, measure semantic disagreement, predict confabulation. Their framing supports ours: *"there is no reason to expect that the same mechanism lies behind different ways to be wrong."* |
| Semantic Entropy Probes | 2406.15927 | `[SEEN]` | Cheap approximation of the above |
| Two Failures of Self-Consistency | 2305.14279 | `[ABSTRACT]` | Consistency and correctness are distinct properties |

### 2.4 The domain axis

**To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic
reasoning** — Sprague et al., arXiv 2409.12183 `[ABSTRACT]`

Meta-analysis of 100+ papers plus 20 datasets across 14 models. CoT helps mainly
on math and logic, much less elsewhere. Reported finding directly useful to us:

```text
On MMLU, direct answering ~= CoT accuracy UNLESS the question or response
contains an equals sign (i.e. symbolic operations).
```

This gives us a validated, externally-sourced, near-free instrument for the
reasoning-bound vs knowledge-bound axis. Use it rather than deriving our own.

### 2.5 Evaluation variance

| Work | ID | Level | Why it matters |
|---|---|---|---|
| Adding Error Bars to Evals | 2411.00640 | `[ABSTRACT]` | Standard errors, CIs, paired comparison, cluster adjustment, **power analysis for sample size**. Notes a 2-point gap on a few-thousand-item benchmark is frequently within noise. |
| Quantifying Variance in Evaluation Benchmarks | 2406.10229 | `[ABSTRACT]` | Defines and measures benchmark variance including seed variance |

### 2.6 Inherited from the specialist notes, not re-verified

`[INHERITED]` — Medprompt (2311.16452), Fully Open Meditron (2605.16215),
SaulLM-7B (2403.03883), MMLU-Pro (2406.01574), Expert Orchestration (2506.00051),
Cost-Aware Contrastive Routing (2508.12491), IR3DE (2606.06098),
PLawBench (2601.16669). See `SPECIALIST_MODEL_RESEARCH.md` §7.

---

## 3. Removed: Questions And Designs

Each row records what was proposed, what removed it, and whether the removal was
caused by **prior art** or by **our own data**.

| # | Proposal | Removed by | Cause |
|---|---|---|---|
| 1 | "Echo is a cost-saving cascade" as the headline contribution | Own data: 2.99 vs Sonnet's 2.98 units. Plus Yue et al. 2310.03094, AutoMix, FrugalGPT all did consistency- or confidence-gated cascades first. | both |
| 2 | Paid Haiku judge | Own data: judge costs 0.66 units, capping best-case saving at 11% before any escalation. Replace with `echo-lexical` (free, deterministic on MCQ). | data |
| 3 | LLM difficulty classifier (loop doc Level 3) | Own data: ~1 unit/task against a total available prize of 2.4 points and 0.39 units over the free subject rule. | data |
| 4 | Chemistry specialist | Own data: no gap to close (Haiku 88% vs Sonnet 84%), and no credible open model exists. | both |
| 5 | Philosophy specialist | Prior art: no philosophy specialist model exists. Largest real gap (28 pts) but nothing to route to. | prior art |
| 6 | Law specialist as a router tier | Own data: never measured standalone. Demoted to a 25-call side probe. | data |
| 7 | Echo as an option inside a router | Own data: a router with perfect knowledge picks Echo 3 times out of 124. Its price band (£2 against £1 and £3) is too thin. | data |
| 8 | "Is agreement a better signal than subject?" | Own data: separation 57 vs 44 points, but the combined policy beats the subject rule by 1 task. Answer is "combine them, gain is negligible" — true and boring. | data |
| 9 | **"How much of a routing win is real?" — run-to-run variance and power analysis** | **Prior art: Adding Error Bars to Evals (2411.00640) does the statistics and the sample-size framework; Quantifying Variance (2406.10229) does the measurement.** | prior art |
| 10 | Domain-dependence, v1: *"does self-consistency detect reasoning errors rather than knowledge gaps?"* | Narrowed, not killed. Sprague et al. established the domain axis; Resample-or-Reroute observed the failure. Survives only in the sharpened form in §4. | prior art |
| 11 | Per-query routing instability (aggregate stability hiding per-query churn) | Not removed — demoted. Real, evidenced (`echo-lexical` escalation 84.4% vs 82.8% while 19/64 tasks changed path), but it is one observation. Keep as a secondary result or short paper. | — |

### 3.1 Partially rehabilitated

**Domain classification by a local model.** Removed as #3 above when framed as
*difficulty* classification. It returns when framed as *subject* classification,
because those are different jobs — subject is vocabulary, difficulty is
comprehension — and because a local model costs 0 units. Simulation on the pilot
shows the subject rule degrades gently under classifier error:

```text
guesser wrong   pass    cost
        0%     87.9%   1.81
       10%     86.9%   1.85
       20%     86.0%   1.88
       50%     83.1%   2.00
reference: always-Sonnet 87.1% @ 3.00
```

The router only needs the binary cheap-vs-expensive call, not the subject name,
so confusing physics with chemistry is free.

---

## 4. What Survived

> **When self-consistency fails as a correctness signal, is the cause the answer
> FORMAT, the DOMAIN, or the DIFFICULTY?**

Three named hypotheses, one of them from published work:

```text
H1 FORMAT      few options -> two wrong draws collide by chance
               (Resample-or-Reroute 2607.08665, asserted not measured)
H2 DOMAIN      knowledge-bound tasks produce shared wrong beliefs
               (ours; predicted in SPECIALIST_MODEL_RESEARCH.md §4 before the data)
H3 DIFFICULTY  hard questions produce shared wrong answers regardless of type
               (nobody's; the confound neither we nor they controlled)
```

Why we are positioned to answer it:

**Against H1.** MMLU-Pro has ten options, not GPQA's four, so chance collision is
roughly 1-in-9 rather than 1-in-3 — the mechanism they blame is ~3x weaker in our
data. And every MMLU-Pro category shares the same ten-option format, yet false
accepts vary eightfold across categories:

```text
law 8   philosophy 6   chemistry 2   math 1   physics 0
```

A pure chance-collision mechanism cannot produce that spread at constant option
count. That is a direct counter-example from data already collected.

**Against H3 — not yet possible, and this is the critical design flaw.** In the
current five categories, difficulty and domain type move together perfectly:

```text
physics  Haiku 100%   easy   reasoning-bound
math     Haiku  92%   easy   reasoning-bound
chem     Haiku  88%   easy   reasoning-bound
law      Haiku  56%   hard   knowledge-bound
phil     Haiku  56%   hard   knowledge-bound
```

There is no overlap, so H2 and H3 make identical predictions on this data.
**Category selection is now the most important decision in the design** — the
extension to 14 categories must deliberately include a hard reasoning-bound
domain and an easier knowledge-bound domain to break the correlation.

The format arm (same questions with and without options) and the sample-count
sweep (k = 2, 3, 5, 10) are both untouched by either paper.

---

## 5. Open Checks

```text
1. RerouteGuard (2601.21380) — read before building any routing-robustness claim
2. Which MMLU-Pro categories break the difficulty/domain correlation
3. Whether anyone has run a format x domain factorial on agreement reliability
   (not found so far, but searched only in English and only via web search)
4. Open-ended scoring: needs a grader model, which adds cost and noise
   directly onto the dependent variable. Validate the grader before trusting
   the open-ended cells.
```

---

## 6. Standing Rule

Three proposals in this project were removed by prior art that a single search
would have found — #5, #9, and most of #10. Before any further design work:
**search first, design second.**
