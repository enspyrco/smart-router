# Specialist Models vs Frontier Generalists: Research Notes For Echo §10

Date: August 3, 2026

Context: `ECHO_PROJECT_SUMMARY_COST_CONFIDENCE.md` §10 proposes testing
"hyper-specialized" domain models, starting with `law -> legal specialist`.
These notes check whether that direction is supported by current evidence.

Short answer: it is weaker than it looks. The strongest recent results argue
that a **generalist plus an inference-time technique** beats a **tuned
specialist** — which is exactly what Echo already is.

---

## 1. What "Frontier Generalist" Means

A frontier model is a general-purpose model at the leading edge of current
capability. The important property is that the term is **relative, not a fixed
spec**. It means "whatever sits at the leading edge right now," so the set
changes every few months as new models ship.

Two things follow from that definition, and both matter for Echo:

```text
1. The frontier moves continuously.
2. A specialist model is frozen to whatever base it was built on.
```

A specialist is built by taking some base model and continuing to train it on
domain data. That base is fixed at build time. The frontier is not. So the
comparison "specialist vs generalist" silently becomes "2024 base + domain
data vs 2026 frontier" unless somebody rebuilds the specialist.

In Echo's terms:

```text
Haiku   = cheap tier
Sonnet  = strong tier, near-frontier quality baseline
Specialist = a third axis, NOT simply "between" Haiku and Sonnet
```

A specialist is not a cheaper Sonnet. It is a different model with a different
skill profile, and it may be worse than Haiku on your benchmark while being
better than Sonnet on its own.

---

## 2. Why Specialists Are Weakening

Five mechanisms, strongest evidence first.

### 2.1 Generalist + prompting already beat specialist tuning

This is the canonical result and it is directly relevant to Echo.

Microsoft's **Medprompt** work took GPT-4 — a generalist, with no medical
tuning at all — added a systematic inference-time prompting strategy, and beat
Med-PaLM 2, the state-of-the-art medical specialist:

```text
MedQA (USMLE):
  27% reduction in error rate vs best specialist methods
  first score above 90%
  state of the art on all 9 MultiMedQA datasets
  an order of magnitude fewer model calls
```

Critically, the authors state the methods are **general purpose and make no
specific use of domain expertise** — and that the technique generalised beyond
medicine to electrical engineering, machine learning, philosophy, accounting,
**law**, nursing, and clinical psychology.

Why this matters for Echo: Echo is an inference-time technique applied to a
generalist. Medprompt is the precedent that this class of approach outcompetes
domain tuning. Echo's twist is that it spends that inference-time budget to
reduce **cost**, where Medprompt spent it to raise **accuracy**. That is a
clean, defensible position in the literature — and it argues against §10.

### 2.2 Reported specialist gains may be contamination, not capability

The 2026 Fully Open Meditron replication is the most direct recent evidence.
Their finding, in their words:

> biomedical specialists frequently fail to outperform their generalist bases
> on unseen medical data, suggesting reported gains may reflect contamination
> or benchmark adaptation rather than clinical capability

Numbers from that work:

```text
Qwen3-30B-A3B-Instruct-2507 (generalist):  59.41
Apertus-70B-MeditronFO (best fully open specialist):  53.77
MedGemma-27B (specialist):  60.67
```

A 30B generalist beats a 70B open medical specialist. Note the authors do
**not** attribute this to catastrophic forgetting — they attribute it to corpus
construction and benchmark contamination. The claim is sharper than "specialists
degrade": it is that some specialist wins were never real, because the specialist
had seen the benchmark.

This is the single most important caution for §10. If we test a specialist on
MMLU-Pro and it wins, we have to ask whether MMLU-Pro law was in its training
corpus before we believe it.

### 2.3 Domain-adaptive pretraining has diminishing returns

Continued pretraining on domain data pays off, but the payoff curve is steep
then flat, and it depends on how far the domain is from the base model's
pretraining:

```text
largest gains: first ~200M tokens, then diminishing
gains largest in domains FURTHEST from original pretraining
gains shrink as that distance shrinks
```

Law and medicine were exotic for a 2022 base model. They are not exotic for a
2026 frontier model — case law, statutes, and PubMed are all well represented in
modern pretraining corpora. The distance has shrunk, so the available gain has
shrunk with it.

One practical rule from this literature: if a base model already reaches 90%+
of target performance with good prompting, continued pretraining offers little.

### 2.4 Domain tuning trades away general capability

The stability-plasticity dilemma. Fine-tuning for a vertical domain causes
catastrophic forgetting of general ability. Observed across 1B–7B models, and
the reported direction is uncomfortable:

```text
as scale increases, forgetting gets STRONGER in:
  domain knowledge
  reasoning
  reading comprehension
```

For a multiple-choice benchmark this matters more than it sounds. MMLU-Pro
scoring depends on the model following the answer format and reasoning over 10
options. A specialist that has drifted toward free-text legal prose may lose
exactly the instruction-following and MCQ-reasoning behaviour the harness needs —
and that shows up as unparseable output, not as a wrong answer.

### 2.5 Refresh cost (inference, not a cited result)

Maintaining a specialist means redoing continued pretraining every time the base
moves. Almost nobody does this. SaulLM is a Mistral-7B derivative; the medical
specialists above are Llama-2 and Gemma derivatives. Meanwhile the frontier
ships every few months. Specialists decay by standing still.

I have not found a paper that measures this directly — flagging it as reasoning,
not evidence.

---

## 3. The Specialist Landscape, By Domain

Feasibility is judged against our infra: OCI ARM free tier + Ollama, which
already runs `echo-small-judge` on local Qwen 7B. A local specialist costs ~0
marginal cost units, which is what the routing economics need.

| Domain | Candidate | Ollama-ready | Assessment |
|---|---|---|---|
| math | Qwen2.5-Math (1.5B/7B/72B) | yes | **Strongest case.** 85.3 on MATH at 7B with tool-integrated reasoning. Verifiable domain. |
| code | Qwen2.5-Coder, DeepSeek-Coder | yes | Well established. Verifiable domain. |
| law | Saul-7B-Instruct-v1 (`adrienbrault/saul-instruct-v1`) | yes | MIT licence, Mistral-7B base, 2024 vintage. See §4. |
| biomedical | MedGemma 1.5 (Jan 2026) | yes | 4B = 69.1 MedQA; 27B tier = 85.3. Most actively maintained specialist. |
| chemistry | — | — | No credible current open specialist. |
| philosophy | — | — | No specialist exists. **Drop from §10.** |

Pattern worth noticing: the two domains where specialists clearly still win —
math and code — are the two with **machine-checkable answers**. That is not a
coincidence. Verifiable domains support RL and rejection sampling against ground
truth; law and philosophy do not.

That reframes §10's table. The question is not "which domains are weak?" but
"which domains have a verifiable training signal?"

---

## 4. Why Law Is The Weakest Place To Start

§10 picks law first because it showed the largest gap in the pilot. Gap size
and specialist availability are different things, and law is bad on the second.

**The baseline SaulLM beat is not our baseline.** SaulLM-7B is Mistral-7B
continued-pretrained on 30B legal tokens. Its reported wins are ~4 absolute
points over Mistral-7B-Instruct-v0.1 and parity-to-better against Llama2-7B-chat.
Nobody has shown it beating a Haiku-class model. Our law gap is Haiku 20% vs
Sonnet 60%.

**Task shape mismatch.** SaulLM targets LegalBench-style work: issue spotting,
rule recall, classification over legal documents. MMLU-Pro law is different:

```text
1,101 questions
100% inherited from original MMLU (zero new questions added)
expanded to 10 options
```

The MMLU-Pro authors kept all law questions from the original MMLU precisely
because they were already high-quality professional exam questions. So MMLU-Pro
law is bar-exam-style knowledge recall under 3x more distractors — not legal
text processing. Law is also noted as less amenable to added reasoning steps
than domains like math and chemistry.

**This has a direct consequence for Echo.** If law is knowledge-bound rather
than reasoning-bound, persona-perturbation self-consistency has nothing to bite
on. Two Haiku personas will confidently agree on the same wrong recall, because
the failure is missing knowledge, not a reasoning slip that varies with framing.

That is risk #1 from the top-level README — "Haiku might confidently agree with
itself when wrong" — and law is exactly where we should expect it. The pilot's
12% false accept rate is consistent with this.

Testable prediction, worth stating before the n=125 data lands:

```text
Echo's false accept rate should be HIGHEST in law,
and lower in physics/chemistry/math where errors are reasoning slips.
```

If the n=125 sweep shows that, it is a real finding about *where*
self-consistency works, which is more publishable than "Echo helps a bit."

---

## 5. Recommendation

**Do not build the specialist router yet.** Run one cheap probe first.

Once the n=125 sweep confirms whether the law gap is real, run a three-way
law-only comparison on the same 25 law questions:

```text
haiku-only  vs  saul-7b-local  vs  sonnet-only
~75 calls
```

This answers the transfer question directly and costs almost nothing. Based on
§2, the likely outcome is that Saul lands at or below Haiku.

That is a good result, not a failed experiment. It converts §10 from a plan into
a finding:

```text
Specialist routing needs current specialists.
Outside math and code, they mostly do not exist.
So inference-time routing over generalists is the practical option
 — which is what Echo is.
```

Two contamination controls if the specialist does win:

1. Check whether MMLU-Pro or original MMLU is in the specialist's training data.
2. Compare the specialist against **its own base model** (Mistral-7B-Instruct),
   not only against Haiku. Meditron's finding was that specialists fail to beat
   *their own bases* on unseen data. That comparison is the honest one.

If we want a specialist arm that is likely to *succeed*, use math with
Qwen2.5-Math, not law with Saul.

---

## 6. Related Work To Cite

§9's hybrid router is a crowded space now. These postdate the README's reading
list and a reviewer will expect them:

| Work | Why it matters |
|---|---|
| Beyond Monoliths: Expert Orchestration (arXiv 2506.00051) | Routers + judges directing queries to specialists. Closest published thing to §9. |
| Cost-Aware Contrastive Routing (arXiv 2508.12491) | Cost-aware routing; directly adjacent to our cost-per-task framing. |
| IR3DE: A Linear Router (arXiv 2606.06098) | 2026 linear router. |
| Medprompt / Nori et al. (arXiv 2311.16452) | The generalist-beats-specialist precedent. Should anchor our positioning. |
| Fully Open Meditron (arXiv 2605.16215) | The contamination caution. |
| PLawBench (arXiv 2601.16669) | 2026 legal benchmark, 850 questions w/ ~12,500 rubric items; no frontier model does well. |

Echo's calibration-free claim survives all of these — they all train something.
Worth stating that delta explicitly rather than assuming a reviewer spots it.

---

## 7. Sources

- Nori et al., *Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine* — https://arxiv.org/abs/2311.16452
- *Fully Open Meditron: An Auditable Pipeline for Clinical LLMs* — https://arxiv.org/html/2605.16215v2
- Colombo et al., *SaulLM-7B: A pioneering Large Language Model for Law* — https://arxiv.org/pdf/2403.03883
- Equall/Saul-7B-Instruct-v1 — https://huggingface.co/Equall/Saul-7B-Instruct-v1
- Wang et al., *MMLU-Pro* — https://arxiv.org/html/2406.01574v4
- *An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning* — https://arxiv.org/abs/2308.08747
- *Domain-Adaptive Continued Pre-Training of Small Language Models* — https://arxiv.org/abs/2504.09687
- *The Data Efficiency Frontier of Financial Foundation Models* — https://arxiv.org/abs/2512.12384
- *Beyond Monoliths: Expert Orchestration* — https://arxiv.org/pdf/2506.00051
- *Cost-Aware Contrastive Routing for LLMs* — https://arxiv.org/pdf/2508.12491
- MedGemma — https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/

**Not cited, deliberately:** several 2026 vendor blog posts claim specialists beat
generalists by 23–37%, attributed to NIST. I could not trace that to a primary
NIST document and it runs opposite to the peer-reviewed Meditron result.
