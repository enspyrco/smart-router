# Echo Router Design: Classifier-Based Routing With Echo As Detector

Date: August 6, 2026

Status: design document for the router itself. Complements
`ECHO_PAPER_PLAN.md` (the research question) and `ECHO_LITERATURE_AUDIT.md`
(what was ruled out and why).

All figures come from `results/20260803T064926Z_mmlu_pro_n125.jsonl` — 125
MMLU-Pro tasks across physics, maths, chemistry, law, philosophy; arms
`haiku-only`, `sonnet-only`, `echo-judge`. 124 tasks have complete rows
(one Sonnet timeout). Cost is in units where one Haiku call = 1.0.

---

## 1. The Proposal Under Review

```text
1. Echo is used to know which domain a specific question is —
   i.e. to know which model works best for which category
2. Then use a classifier (BERT text embeddings -> multi-layer perceptron)
3. Then if the domain is uncertain, move it to Echo to determine
```

Verdict up front: **component 2 is right and better-motivated than it first
appears. Component 1 works offline but not at runtime. Component 3 is
contradicted by the data.** Details below, then the revised architecture in §5.

---

## 2. Component 1 — Echo For Domain Identification

### 2.1 What does not work

Echo cannot identify a domain. Its mechanism is: ask the cheap model the same
question twice under different personas, compare the two answers. The output is
**one bit — agree or disagree.** Nothing in that pipeline inspects subject
matter, so there is no mechanism by which it could return "this is law."

What Echo reports is *"the cheap model is unsure here"*, which is a different
statement from *"this question is about law."*

### 2.2 What does work

There is a valid version of this idea, and it is the one already exercised:
**use Echo offline, as a measurement instrument, to build the category table.**

Running Haiku, Sonnet and Echo across categories is how we learned which
categories the cheap tier can handle:

```text
category    haiku   sonnet   -> tier
physics     100%     96%        cheap
maths        92%     96%        cheap
chemistry    88%     84%        cheap
law          56%     72%        expensive
philosophy   56%     84%        expensive
```

That table is the routing policy. Echo helped produce it. Echo is not consulted
again when a question arrives.

**Rule: Echo builds the table once. It does not read the question at runtime.**

### 2.3 Open item — the table must be re-checked on open-ended questions

The table in §2.2 was built **entirely from ten-option multiple-choice
questions.** Every figure in this document inherits that. Before the table is
trusted at runtime it has to be re-measured on open-ended, domain-specific
questions, for two independent reasons.

**Format may be the cause, not domain.** Resample-or-Reroute (arXiv 2607.08665)
attributes agreement failure to answer format rather than subject:

> "On GPQA, a four-option multiple-choice benchmark, agreement is nearly
> uninformative — two wrong draws easily agree on the same letter."

If that is right, the whole table is an artefact of MCQ and does not describe
what happens on real questions. Our data argues against a pure format
explanation — MMLU-Pro has ten options, not four, and false accepts still vary
eightfold across categories at constant format (law 8, philosophy 6, chemistry 2,
maths 1, physics 0) — but that is an argument, not a measurement. The
measurement is the open-ended arm.

**The tier ranking itself may move.** Haiku's standing relative to Sonnet is
measured on "pick one of ten letters". Open-ended answering is a different task:
no distractors to eliminate, no partial-credit-by-elimination, and the failure
mode shifts from wrong-letter to incomplete-or-unverifiable. A category that is
cheap-tier under MCQ is not guaranteed to stay cheap-tier when the options are
removed.

**What it costs.** Agreement comparison is free and deterministic on MCQ —
`lexical_agree` extracts a letter and compares. Open-ended answers have no letter,
so agreement needs a grader model, and correctness needs one too. That grader's
own error lands directly on the dependent variable. **Validate the grader against
human labels on a sample before trusting any open-ended cell.**

**Cheapest test.** Take the same MMLU-Pro questions, strip the options, re-run.
Same questions, same subjects, only the format changes — which isolates format
from domain and difficulty in one step. This is also the format arm of the 2×2 in
`ECHO_LITERATURE_AUDIT.md` §4, so the router work and the research question are
served by the same run.

Until this is done, treat §2.2 as **"the table for multiple-choice questions"**,
not as the table.

---

## 3. Component 2 — The Classifier

Endorsed, with adjustments. The argument for it is stronger than the one
originally given against it.

### 3.1 Why the two-step design is statistically correct

Earlier guidance in this project was to train directly on the outcome ("will
Haiku get this right?") rather than on domain, because domain is only a proxy.
That guidance ignored a data constraint that decides the matter:

```text
domain labels   ~12,000 MMLU-Pro questions   FREE (ship with the dataset)
outcome labels        125 questions          one API call each
```

BERT + MLP cannot be trained on 125 examples; it will memorise them. It can be
trained on 12,000 free domain labels. The domain -> tier mapping then costs
almost no data — it is 14 numbers, and we already have five of them.

So the two-step design puts the data-hungry component on free labels and the
data-poor component on a handful of parameters. That is the right split, and it
is the reason to prefer this architecture over direct outcome prediction until
outcome labels are cheap.

### 3.2 The classifier only needs a binary decision

It does not need to name the subject. It needs **cheap or expensive**.

Confusing physics with chemistry costs nothing — both are cheap-tier. Confusing
law with philosophy costs nothing — both are expensive-tier. Only cross-tier
confusion matters, which is a substantially easier problem than 14-way
classification.

### 3.3 Error tolerance is generous

Simulated on the pilot, with the classifier getting the cheap-vs-expensive call
wrong p% of the time:

```text
p       pass    cost
 0%    87.9%    1.81
 5%    87.4%    1.83
10%    86.9%    1.85
20%    86.0%    1.88
30%    85.0%    1.92
50%    83.1%    2.00

reference: always-Sonnet 87.1% @ 3.00 | always-Haiku 79.0% @ 1.00
```

At 20% classification error the router still returns 86.0% at 1.88 units —
roughly one accuracy point below always-Sonnet for 37% less cost. The policy
degrades gently rather than collapsing.

### 3.4 Two adjustments

**Start below BERT.** Subject identification is largely vocabulary
("plaintiff" -> law, "enthalpy" -> chemistry). TF-IDF plus logistic regression
trains in seconds, is fully interpretable, and has no inference dependency.
Establish that baseline first; adopt BERT + MLP only if it beats it, and report
the delta. Without the baseline there is no way to say what BERT bought.

**Know the ceiling.** Domain explains only part of the target:

```text
variance in "will Haiku be right?" explained by:
  subject/domain label     22.2%
  persona agreement        19.6%
  -> 78% lives INSIDE categories, invisible to any domain classifier
```

Concretely, in law Haiku answers 14 of 25 correctly, but the router sends all 25
to Sonnet because the only thing it knows is the subject. The whole branch is
capped at oracle (90.3% @ 1.42) minus subject rule (87.9% @ 1.81) — about 2.4
points and 0.39 units.

---

## 4. Component 3 — Uncertain Routed To Echo

Measured directly. It loses.

```text
router: confident categories -> Haiku, uncertain categories -> ?

  uncertain -> Sonnet   (subject rule)    87.9%    1.81
  uncertain -> Echo     (proposal)        80.6%    1.72
  uncertain -> Haiku    (reference)       79.0%    1.00
```

Echo saves 0.09 units and costs **7.3 accuracy points**.

Inside the uncertain categories:

```text
            haiku    echo   sonnet   echo cost
law         56.0%   60.0%    72.0%       2.84
philosophy  56.0%   60.0%    84.0%       2.72
```

Echo costs nearly what Sonnet costs and delivers 12 to 24 points less. Its
apparent 4-point gain over Haiku is one question out of 25 in each category —
inside the 6.7% run-to-run noise floor, so not a real gain.

### 4.1 Why this fails — the same cause, twice

**The router is uncertain exactly where Echo does not work.**

The router is unsure about law because Haiku scores 56% there. Haiku scores 56%
because law is knowledge-bound. Knowledge-bound is precisely where two personas
agree confidently on the same wrong answer. The router's uncertainty and Echo's
failure share one underlying cause, so the fallback is guaranteed to be applied
where it is least effective.

### 4.2 A conflation to avoid

"If the domain is uncertain" mixes two different uncertainties:

```text
A. the classifier is unsure WHICH SUBJECT this is
B. we are unsure WHETHER THE CHEAP MODEL WILL COPE
```

Only B affects the routing decision. And A mostly does not matter — a classifier
torn between law and philosophy routes to Sonnet either way. Classifier
uncertainty is only relevant when it straddles the cheap/expensive boundary.

### 4.3 What to do instead

Route cross-tier-uncertain questions to **Sonnet**. It is the safe default, and
on the pilot only a handful of questions would be affected.

### 4.4 The condition under which Echo would win

Echo needs a region that is simultaneously:

```text
uncertain        (Haiku roughly 50-70%, so the router genuinely cannot decide)
AND
reasoning-bound  (errors are slips, so the two personas actually diverge)
```

No such cell exists in the current five categories — the confident categories
are confident because Haiku is at 92-100%, and the uncertain ones are
knowledge-bound. Filling that cell requires hard-but-reasoning-bound tasks:
MMLU-Pro engineering or economics, or the 24 unrun BBH suites
(`dyck_languages`, `multistep_arithmetic_two`, `word_sorting`,
`geometric_shapes`, `formal_fallacies`) listed in `benchmarks/bbh.py:22`.

Until that is measured, component 3 stays out of the router.

---

## 5. Revised Architecture

```text
OFFLINE — run once, rerun when models change
  sweep haiku-only + sonnet-only across categories
  -> category -> tier table
  (this is where Echo contributes: as a measurement instrument)

RUNTIME — per question
  question
    -> classifier (TF-IDF + logistic regression; BERT + MLP if it beats that)
    -> predicted subject
    -> tier from the table
    -> answer

  cross-tier-uncertain -> Sonnet (safe default, NOT Echo)

AFTER THE ANSWER — optional, where human review exists
  ask the cheap model a second time, compare
  disagreement -> add to review queue
```

Echo remains in the system. It moves from the front, where it cannot see
anything useful, to the back, where it measurably works.

---

## 6. Echo's Actual Role: Error Detector

Repositioned from cost-saver to detector, and scored against detection
baselines rather than cost baselines.

```text
Echo flags 14 of 124 answers      (11% of traffic)
  10 of the 14 are genuinely wrong (71% precision)
  10 of the 26 total errors caught (38% recall)

same review budget spent at random, 1000 simulated draws:
  average           2.9 errors found
  best single draw  8 errors found
  Echo's queue     10 errors found     -> 3.4x better than random
```

Echo's queue beats the best of a thousand random draws, so the advantage is not
sampling luck.

Per-subject, the detector behaves very differently:

```text
philosophy   6 flags, 6 real errors   100% precision — but silent on 6 others
maths        1 flag,  1 real error    100% precision
law          7 flags, 3 real errors    43% precision — and missed 8 more
```

Every false alarm came from law. Law is unreliable in both directions;
philosophy is high-precision and low-recall.

Why detection succeeds where routing failed: routing asks Echo to **decide**, and
71% accuracy means a third of its decisions are wrong and cost money. Detection
asks Echo only to **rank**, and a human decides. 71% precision is mediocre for an
automatic decision and excellent for a to-do list.

Cost: one extra cheap call per question (1.0 unit). Justified wherever an
undetected wrong answer costs more than that.

Demo script: `scripts/echo_triage_demo.py` (if kept) reproduces the queue above
from the committed results file.

---

## 7. Build Order

```text
1. TF-IDF + logistic regression subject classifier on MMLU-Pro's free labels
   report cross-tier error rate; look it up in §3.3
2. Wire it to the category->tier table; evaluate on held-out questions
3. Only then try BERT + MLP; report the delta over step 1
4. Echo triage as a separate, optional post-answer stage
```

Steps 1 and 2 require no new API calls — MMLU-Pro labels are free and the tier
table already exists for five categories.

---

## 8. What Is Not Yet Measured

```text
- whether the category->tier table survives on OPEN-ENDED questions (§2.3);
  every number in this document comes from ten-option multiple choice
- classifier accuracy on real (non-benchmark) questions; MMLU-Pro law is
  bar-exam MCQ and does not resemble ordinary legal queries
- behaviour on questions belonging to NO category, which is most real traffic
- whether any hard-but-reasoning-bound cell exists where component 3 wins (§4.4)
- the 9 MMLU-Pro categories not yet swept
- whether the 22.2% domain ceiling holds outside these five categories
- grader reliability for open-ended scoring, validated against human labels
```

The first three decide whether this router survives contact with anything other
than a benchmark. The open-ended check (§2.3) is the highest-value single run
available, because it serves the router and the research question at once.
