# Smart Router

**Routing LLM requests across price tiers without training a router.**

*(This project was called **Echo** until August 2026. "Echo" now names one
mechanism inside it — see below — not the programme as a whole.)*

## The problem

Most production LLM apps overpay because they send every request to the same
model. The standard fix is a *router*: a learned classifier that decides "easy
task, cheap model; hard task, expensive model." Trained routers work but need
labelled training data per domain, which is the bottleneck for adoption.

We are looking for routing signals that need little or no training data.

## Three routing signals

| Signal | Question it answers | Status |
|---|---|---|
| **Echo** (self-consistency) | Is the cheap model unsure here? | Built, measured |
| **Domain** | Which specialist covers this? | Designed, being measured |
| **Diversity** | Which of several answers is right? | Open — the current crux |

**Echo** is the original primitive: call the cheap model twice with two
different persona prompts. If the answers agree, accept the cheap one. If they
disagree, escalate. No classifier, no training data, no calibration. Self-
consistency exists in the literature for accuracy gains (Wang et al. 2022) but
has not been studied as a *cost* tool.

**Domain** routing dispatches by subject to a hyper-specialised small model
rather than escalating by difficulty. The idea is Connor's.

**Diversity** is where the measurements keep pointing, and where the work now
is.

## What we have measured

**BBH is saturated for this model pair.** Clean n=99, three subtasks:

```
haiku-only    0.848
sonnet-only   0.848      <- identical, and identical per-subtask
echo-judge    0.869
echo-oracle   0.899
```

Haiku and Sonnet tie exactly, yet the oracle beats both. That is only possible
if they fail on *different* items. So on this slice routing's value is not
"escalate to a smarter model" — it is **decorrelated errors**: two diverse
cheap attempts covering disjoint failures.

An earlier round of BBH numbers was invalid. The harness shelled out to
`claude --print` from inside the repo, so every model call inherited the
project's `CLAUDE.md` and hooks and answered as the project agent. Fixed with
`--setting-sources ""`. Any result predating that fix is void.

**MMLU-Pro does separate by domain.** From the n=125 run (Meghana):

```
category    haiku   sonnet   -> tier
physics     100%     96%        cheap
math         92%     96%        cheap
chemistry    88%     84%        cheap
law          56%     72%        expensive
philosophy   56%     84%        expensive
```

**But domain explains less than you would hope.** Variance in "will Haiku get
this right?":

```
subject/domain label     22.2%
persona agreement        19.6%
-> 78% lives INSIDE categories, invisible to any domain classifier
```

Which caps domain routing: oracle 90.3% @ 1.42 cost units versus a trivial
subject-lookup rule at 87.9% @ 1.81. About 2.4 accuracy points of headroom
above fourteen hard-coded numbers.

## Open questions

1. **Is the category table an artefact of multiple choice?** Every figure above
   comes from ten-option MCQ, and BBH went saturated under the same format.
   The open-ended arm (`scripts/run_mmlu_pro_open.py`) strips the options and
   re-runs the same questions to isolate format from domain. Both router
   designs depend on the answer.
2. **Can a diversity signal reach the oracle without ground truth?**
   Self-consistency structurally cannot — models agree on identical wrong
   answers (the "popularity trap"). This is the crux.
3. **What does a router look like that uses domain *and* agreement?** They
   explain roughly equal, probably complementary, shares of the variance.

## Design documents

- [`docs/semantic-router-design.md`](docs/semantic-router-design.md) — classify
  before the LLM: taxonomy, linear probe on frozen embeddings, conformal
  abstention, two-tier cascade.
- `experiment/ECHO_ROUTER_DESIGN.md` (branch `feat/mmlu-pro-advanced`) —
  classifier-based routing grounded in the n=125 data, with the ceiling
  analysis above.

These two were written independently and converge on the same architecture:
the expensive semantic work happens **offline** to build a policy; the runtime
path is a cheap classifier with no LLM in it. They are being reconciled into
one document.

## Honest risks

1. **Cheap models may agree confidently while wrong.** If Haiku has consistent
   blind spots, persona perturbation will not surface them and Echo collapses
   to "Haiku with extra steps." The popularity-trap literature says this is the
   default, not the exception.
2. **Benchmark saturation.** MCQ reasoning benchmarks are at ceiling for
   current cheap models, so they cannot show a difficulty gap even where one
   exists on real work.
3. **Grader dependence.** Open-ended evaluation needs a model to judge
   correctness, and its error lands on the dependent variable. Scoring is
   tiered so grader-decided rows stay separable and auditable.
4. **Tier pricing can move.** The contribution is a technique whose economics
   depend on the tier gap, not a specific price claim.

## Where the work lives

- **Run locally:** [`experiment/README.md`](experiment/README.md)
- **Results & sweep history:** [`experiment/results/README.md`](experiment/results/README.md)
- **Blog:** [enspyr.co/blog/echo-cheap-routing-without-a-router](https://enspyr.co/blog/echo-cheap-routing-without-a-router)
- **Compute:** OCI free-tier ARM (`nick-mel`), behind Caddy. The harness is
  I/O-bound, so ARM is fine.

## Collaborators

Author profiles: **https://enspyr.co/about**

## Background reading

- Wang et al. 2022, *Self-Consistency Improves Chain of Thought Reasoning*
- Ong et al. 2024, *RouteLLM*
- Chen et al. 2023, *FrugalGPT*
- Ding et al. 2024, *Hybrid LLM*
- *Wisdom and Delusion of LLM Ensembles* (arXiv 2510.21513) — the popularity trap
- *Resample-or-Reroute* (arXiv 2607.08665) — agreement failure as a format effect
