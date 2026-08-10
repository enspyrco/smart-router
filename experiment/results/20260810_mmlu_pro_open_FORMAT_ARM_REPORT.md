# Format arm: is the category table an MCQ artefact?

**Run:** `20260810T043012Z_mmlu_pro_open_n125.jsonl` (250 rows, 125 questions × 2 arms, clean exit)
**Paired against:** `20260803T064926Z_mmlu_pro_n125.jsonl` (Meghana's MCQ run)
**Design:** identical slice — `start=0`, `n_per_category=25`, same five categories. All 125 question IDs shared, so every comparison below is **paired**.

## Headline

**The category table cannot support the weight the router designs put on it — but the reason is not the one we were testing for.**

Running a paired McNemar test on Sonnet-vs-Haiku, per category, per format:

| category | format | S>H | H>S | p | verdict |
|---|---|---|---|---|---|
| physics | MCQ | 0 | 1 | 1.000 | no difference |
| physics | OPEN | 1 | 2 | 1.000 | no difference |
| math | MCQ | 1 | 0 | 1.000 | no difference |
| math | OPEN | 0 | 2 | 0.500 | no difference |
| chemistry | MCQ | 0 | 1 | 1.000 | no difference |
| chemistry | OPEN | 4 | 6 | 0.754 | no difference |
| law | MCQ | 6 | 2 | 0.289 | **no difference** |
| law | OPEN | 4 | 4 | 1.000 | no difference |
| philosophy | MCQ | 7 | 0 | **0.016** | **sonnet better** |
| philosophy | OPEN | 2 | 3 | 1.000 | no difference |

Two findings, and the first is the bigger one:

1. **Four of the five MCQ cells were never statistically significant.** Only philosophy separates Haiku from Sonnet at n=25. Law — the table's other "expensive tier" category, and the one named as the clearest first specialist candidate — is 6-vs-2 discordant, p=0.289. That is not a tier assignment; it is noise that happens to point in a plausible direction.
2. **The one real effect is format-dependent.** Philosophy's significant Sonnet advantage (7-0 discordant under MCQ) vanishes when the options are removed (2-vs-3, p=1.000).

## What this does and does not license

**Does not** license "domain routing is dead." n=25 per cell is underpowered; "no difference" here means *not detectable at this n*, not *no effect exists*. Law's 6-vs-2 may well be real and simply under-powered.

**Does** license: the current table is not a sound basis for a routing policy. A per-category tier assignment built from cells that are individually non-significant will encode sampling noise as policy. The error-tolerance analysis in `ECHO_ROUTER_DESIGN.md` §3.3 shows the *policy* degrades gently with classifier error — but that analysis assumes the tier labels themselves are correct. If a tier label is noise, gentle degradation around a wrong target does not help.

**Required before the table is used:** more n per category. The cheapest decisive move is to raise n on law and philosophy specifically, in both formats, until each cell is powered.

## Raw rates, for reference

| category | MCQ h | MCQ s | OPEN h | OPEN s |
|---|---|---|---|---|
| physics | 1.00 | 0.96 | 0.80 | 0.76 |
| math | 0.92 | 0.96 | 0.92 | 0.84 |
| chemistry | 0.88 | 0.84 | 0.68 | 0.60 |
| law | 0.56 | 0.72 | 0.44 | 0.44 |
| philosophy | 0.56 | 0.84 | 0.64 | 0.60 |

Open-ended is uniformly harder except philosophy-Haiku. Read these as descriptive only — the paired tests above are the inferential result.

## Threats to validity

**1. Grader dependence — the big one. UNRESOLVED.**
207 of 250 rows (83%) were decided by the model grader, because MMLU-Pro gold answers are long descriptive MCQ options that a freely-answering model never reproduces verbatim. Only 43 rows resolved deterministically.

On those 43 grader-free rows, **Haiku and Sonnet tie exactly at 0.95** — weak but genuinely grader-independent support for "no difference." Weak because the sample is small and skewed toward short/numeric answers.

**2. Grader may penalise verbosity — live confound.**
Sonnet's answers average 102 characters against Haiku's 82, and Sonnet's grader-tier pass rate is lower (0.58 vs 0.64). That is exactly the pattern a length-biased grader would produce, and it would manufacture the disappearance of Sonnet's advantage. Not proof of bias — Sonnet may simply be wrong more often here — but it is unresolved and it points the same way as the headline, which is the dangerous direction.

**Gate:** `..._worksheet.md` holds 40 sampled grader decisions for human labelling. Until those are scored, every open-ended number here is provisional.

**3. The manipulation is not purely format.**
Stripping options also removes the *answer-form convention*. Nothing tells the model that a multi-statement item expects "False, True" in question order. Tasks are tagged with `gold_form` (`single_value` / `numeric` / `ordered_tuple`) so this stratum stays separable.

**4. A grader bug was found and fixed mid-build.**
Gold `False, True` vs candidate `True, False` was graded EQUIVALENT — the opposite answer scored as a pass. Fixed with a deterministic `sequence` tier plus an order-sensitivity clause in the grader prompt. It was caught on the first two smoke-test calls, which is a reminder that the grader is the weakest link and one unvalidated grader can invert a result.

## Recommended next steps

1. **Score the grader worksheet.** Blocking. 40 human labels.
2. **Raise n on law and philosophy** in both formats until each cell is powered. This is the cheapest thing that converts the table from suggestive to usable.
3. **Re-grade with a cross-family grader** (OpenAI/Gemini) and check the collapse reproduces. Same-family agreement is a weak signal, but a *disagreement* would be informative.
4. **Hold the router build** until 1 and 2 land. Both designs treat the category table as given.
