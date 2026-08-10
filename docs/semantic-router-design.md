# Semantic Router — Design Spec

> Status: design agreed, not yet implemented. Section 8 is the build order.

---

## 1. Goal and constraints

Classify an incoming question/request **before** it reaches an LLM, so that
topic-specific queries can be dispatched to small hyper-specialised models
instead of a large general one.

**Objectives**
- Specialists should be faster, cheaper, and more accurate than the generalist *in their domain*.
- Anything the specialists can't handle must fall through to the generalist without damage.

**Hard constraints**
- **No LLM-as-judge in the routing path.** (Offline is fine — see §7.)
- **Essentially real-time.** Router overhead should be a small fraction of TTFT.

---

## 2. Original hypothesis (to be partly revised)

The starting idea was:

1. Vectorise the incoming question.
2. Compare against a vector DB of Wikipedia embeddings.
3. Find clusters; use cluster fit as a confidence measure for topic assignment.
4. Optionally follow links back to disambiguation pages to label cluster members.
5. Optionally use papers/books of known topic to build a cosine-similarity heat
   map that "draws borders" across a map of wiki pages.

The core intuition — *topic regions exist in embedding space and can be
bounded* — is sound. Four of the mechanisms need replacing.

---

## 3. What breaks, and what replaces it

### 3.1 Three routing decisions are being conflated

| Decision | Question | Machinery |
|---|---|---|
| **Domain** | Which specialist covers this? | Topic classifier (the original plan) |
| **Difficulty** | Does *any* specialist beat the generalist here? | Preference-trained / distilled scorer |
| **Coverage** | Is this outside every specialist's competence? | OOD detection + abstention |

Most published routers attack **difficulty**, not domain. Since we have genuine
per-domain specialists, domain is our primary axis — but **coverage** is where
naive semantic routers fail, and it is not optional.

### 3.2 Disambiguation pages are the wrong structure

They resolve *name collisions* ("Mercury"), not topical membership. There is no
upward path from them to a taxonomy.

**Use instead:** Wikimedia's `articletopic` taxonomy — 64 topics derived from
WikiProject tags, under four high-level branches (Culture, Geography,
History & Society, STEM), each with mid-level children. Human-curated,
multi-label, covers ~6M articles, snapshotted monthly, and reports macro
ROC-AUC around 95%.

**Do not** try to rebuild this from the raw Wikipedia category graph. It has
cycles, is non-taxonomic, and lacks stable hierarchy — the Wikimedia research
community flags it as the blocker themselves.

### 3.3 Query–document asymmetry

A user question and a Wikipedia article do not occupy the same region of
embedding space. `"why does my sourdough not rise"` is far in cosine terms from
the *Sourdough* article despite identical topic. Nearest-article lookup degrades
silently.

**Fixes (pick one or both):**
- Asymmetric embedder with query/passage prefixes (E5, BGE, GTE families).
- **Preferred:** generate synthetic questions from articles (doc2query style)
  and embed *those* as reference points. Then it's question-to-question.

### 3.4 Cosine distance is not a confidence measure

In high dimensions similarities compress into a narrow band, and *hubness*
means a few vectors are nearest-neighbour to almost everything. A raw `0.82`
means nothing in isolation.

**Mitigations:** whitening/centering (all-but-the-top), CSLS-style local
scaling, or per-class Mahalanobis distance instead of raw cosine.

### 3.5 Topic ≠ capability boundary

The deepest issue. "Write me a SQL migration", "summarise this email",
"continue", "act as a therapist" have **no Wikipedia topic**. Wikipedia's
ontology carves *knowledge*; our specialists carve *tasks*.

**Therefore:** hybrid taxonomy — Wikipedia topics for the knowledge slice,
hand-defined task classes for everything else, plus an explicit
`none-of-the-above` class.

---

## 4. The key reframe

**Wikipedia is training data, not runtime infrastructure.**

There is no vector DB in the hot path. Once you have topic-labelled text, the
router is a **linear probe on frozen embeddings** — a `d × K` matrix. At 256
dims and 64 topics that's ~16K parameters: microseconds, no ANN index, no
recall risk, no drift between index and model.

The "draw borders" intuition is right; the correction is that borders should be
**learned supervised**, not constructed geometrically. Centroid/Voronoi carving
fails because topic regions are non-convex and anisotropic — manifolds, not
balls. If staying non-parametric, use **k centroids per topic** (a mixture)
rather than one; that recovers most of the gap for free.

**Multi-label sigmoid head, not softmax.** "Model protein folding with a
transformer" is biology *and* ML; softmax forces a lie.

---

## 5. Latency budget

Dot products are free. The **encoder** is the entire budget.

| Tier | Option | Notes |
|---|---|---|
| 1 | **Static embeddings** (Model2Vec / `potion-base-8M`) | Up to ~50× smaller, up to ~500× faster on CPU than the source sentence transformer; numpy is essentially the only dependency; self-distillation takes ~30s on CPU with no dataset. Cost: real quality drop — no contextualisation, it's a weighted bag of token vectors. Usually fine for topic, because topic signal is largely lexical. |
| 1b | **Hybrid sparse+dense** | Underrated. Rare technical terms (`HNSW`, `tacrolimus`, `kubelet`) are the highest-signal features and dense embeddings smear exactly those. TF-IDF / naive-Bayes channel concatenated with the dense probe adds points at ~zero latency. |
| 2 | **Small encoder** (MiniLM / ModernBERT) | Only for escalated cases. vLLM Semantic Router uses ModernBERT with minimal latency overhead. |

### Cascade

```
query
  └─> tier 1: static embedding + linear probe
        ├─ high margin  ──────────────> specialist            (~80% of traffic)
        └─ ambiguous ─> tier 2: encoder + probe
                            ├─ resolved ──> specialist
                            └─ still ambiguous ──> generalist
```

Cascading pays a latency tax but removes the need for an accurate upfront
classifier; pure routing has zero latency overhead but demands a good
predictor. The cascade lets us tune where we sit on that trade.

---

## 6. The confidence layer (highest-effort component)

This is how we get a principled abstention guarantee **without** an LLM judge
and without hand-tuned thresholds.

Use **conformal prediction** with a held-out calibration set. Distribution-free
coverage at a chosen level, and — critically — *set-valued* output. Routing
rule becomes trivially interpretable:

- prediction set = **exactly one topic** → route to that specialist
- prediction set = **2+ topics** → ambiguous → escalate cascade, else generalist
- prediction set = **empty** → out of distribution → generalist

The standard construction pairs an OOD screen with the conformal predictor:
flagged inputs abstain with an empty set; in-distribution inputs get a valid
prediction set. See the selective-classification-with-OOD literature — it
addresses exactly this (rejecting both outliers and hard in-distribution
samples near the boundary).

**Implementation notes**
- `top1 − top2` **margin** is usually a better routing signal than `top1`
  probability.
- Score on **logits / embedding distance**, not post-softmax — softmax alone is
  a known-weak basis for adaptive conformal sets.
- OOD scoring options: energy score, Mahalanobis, kNN-distance.

---

## 7. The loophole in the constraint

LLM-as-judge is banned **at inference**, not at training time.

1. Take real production queries.
2. Label them offline with a strong model into our specialist classes + `none`.
3. **Distil that into the linear probe.**

This is the highest-leverage move available. RouteLLM's own results show
LLM-judge data augmentation is what got them to ~95% quality at 14%
strong-model calls (~85% cost reduction). It also sidesteps the Wikipedia
ontology mismatch entirely — labels become *our* task boundaries, not an
encyclopedia's.

**Related alternative:** drop topics as the definition altogether. Define each
specialist by its own data distribution and make the router a **density-ratio
estimator** — "whose training distribution does this query resemble?" Cleaner
conceptually, and it's what "confidence" *should* mean here.

---

## 8. Build order

### Week 1 — validate the axis before building infrastructure

Skip Wikipedia entirely at first.

1. Collect **5,000 real queries**.
2. Label offline with an LLM into specialist classes + `none`.
3. Distil `potion-base-8M` (Model2Vec, ~30s CPU).
4. Fit a **logistic probe** (multi-label sigmoid) on the frozen embeddings.
5. **Conformalise** on a held-out 1,000-query calibration split.
6. Measure **coverage vs. precision**.

Outcome: tells you whether the topic axis is even the right axis. Two days'
work.

### Then

7. Add the hybrid sparse channel; re-measure.
8. Add tier-2 encoder escalation for ambiguous sets; measure the latency/quality trade.
9. Bring in WikiProject taxonomy as **augmentation** — synthetic questions
   generated from articles in the domains where the probe is thin.
10. Ship behind a shadow-mode flag; compare routed vs. generalist outputs offline.

---

## 9. Evaluation

Accuracy is the wrong metric. Track:

- **Cost/quality Pareto curve** (the actual objective).
- **Coverage @ precision** — what fraction can we route at ≥ our precision floor.
- Per-specialist win-rate vs. generalist, in-domain and out.

---

## 10. Failure modes to instrument from day one

- **Silent drift.** Routers degrade over weeks as query patterns shift. Sliding-window KS test on the score distribution is a cheap alarm.
- **Multi-turn.** Route on the last message or the whole thread? Topic drifts mid-conversation; specialist switching mid-thread breaks context. Decide explicitly.
- **Adversarial routing.** If a cheap specialist is reachable by prefixing "in the context of cooking…", someone will find it.
- **Length dilution.** Long pasted inputs wash out topic signal in mean-pooled embeddings. Weight the instruction span, not the payload.
- **The killer:** hard queries often look topically trivial. Domain and difficulty are near-orthogonal — a correct domain decision never rescues you from a wrong difficulty decision.
- **Non-questions.** "thanks", "continue", bare code blocks. Needs an explicit class.

---

## 11. Open questions

- [ ] How many specialists, and what are their actual domain boundaries?
- [ ] Do we route per-message or per-conversation?
- [ ] What's the acceptable router latency ceiling in ms?
- [ ] What's the cost of a wrong route — silent quality loss, or user-visible failure?
- [ ] Is there a retry/escalate path after a specialist answers badly, or is the route final?

---

## 12. References

- Wikimedia language-agnostic article topic model — https://meta.wikimedia.org/wiki/Machine_learning_models/Production/Language_agnostic_link-based_article_topic
- Johnson et al., *Language-agnostic Topic Classification for Wikipedia* — https://arxiv.org/pdf/2103.00068
- WikiProject taxonomy mapping code — https://github.com/geohci/wikipedia-language-agnostic-topic-classification
- Model2Vec — https://github.com/MinishLab/model2vec
- RouteLLM overview / numbers — https://neuraltrust.ai/blog/llm-model-routing
- Guarded Query Routing benchmark — https://arxiv.org/pdf/2505.14524
- Plugin estimators for selective classification with OOD detection — https://arxiv.org/pdf/2301.12386
- Softmax is not Enough (for Adaptive Conformal Classification) — https://arxiv.org/pdf/2602.19498
- LLM routing survey (2024–2026) — https://zylos.ai/research/2026-03-02-ai-agent-model-routing/
