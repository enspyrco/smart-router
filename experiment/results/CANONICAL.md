# Which agreement-baseline artifact is the real one

`results/` is committed on purpose. Replayability has paid for itself repeatedly
here: every re-derivation across four cage-match rounds was free because the raw
rows were in the repo, and each round's fix was verified by replaying the same
data rather than re-running the experiment (Maxwell, rounds 2–4).

What it must NOT do is let the next citation pick the flattering number by
mtime (Tesla, round 4). Those two are only in tension if you assume the sibling
files are sibling *experiments*. They are not — see below. So the fix is to
delete the ambiguity, not the data.

## Canonical

| | |
|---|---|
| **Datum** | `20260810T133611Z_agreement_baseline_mmlu_pro_haiku_n210.json` |
| **Analysis** | `CANONICAL_ANALYSIS.md` — deliberately *untimestamped* |
| n | 210 tasks, 15 per category across all 14 MMLU-Pro categories |
| Calls | 4 per task (`a1`, `a2`, `b1`, `b2`), raw output persisted |
| Model | `claude-haiku-4-5-20251001`, temperature 1.0, max_tokens 4096 |
| Personas | sha256[:12] `9e1c3c3a871e` |
| Pricing as of | **2026-08-10** — sonnet introductory window; `haiku->sonnet` break-even is `r < 0%`, i.e. unsatisfiable |

The analysis has a **stable filename on purpose.** Every other artifact here is
timestamped, which is what made "pick the newest `.md`" a plausible-looking way
to choose a number — the failure mode this file exists to prevent. A fixed name
has no mtime story to tell.

**Cite nothing else.** Regenerate rather than trusting a committed `.md` blindly
— `.md` files are *derived*, the `.json` is the datum, and this repo has shipped
analyses the code had since outgrown:

```
python3 scripts/measure_agreement_baseline.py \
  --from-json results/20260810T133611Z_agreement_baseline_mmlu_pro_haiku_n210.json \
  --write-canonical
```

`--write-canonical` is what keeps this file honest. Round 5 gave the canonical
analysis a fixed name so it would have no mtime story; round 6 caught that the
rename was a manual `mv`, so following the instruction above produced a
*timestamped* file and silently aged the canonical copy — the same
prose-gate-vs-enforced-gate defect, reintroduced by the fix for it. The flag
makes the script write the stable name. Omit it and you get a timestamped
analysis, which is the right default for an exploratory re-run.

**A replay no longer copies the datum.** `--from-json` writes only the analysis
plus a pointer to the file it read. It used to emit a fresh JSON carrying the
entire `rows` payload — which is exactly how the three-artifact replay chain
below grew, and leaving that in place meant it could grow again the same way.

### ⚠️ The datum's flat `mcnemar` field is inverted — do not cite it

`20260810T133611Z…json` carries a legacy top-level `mcnemar: {b: 15, c: 7}`.
The headline pair recomputes to **b=7, c=15** — the *opposite direction*.

It hid for several rounds because **McNemar's p is symmetric**: `p=0.134` agrees
across both orderings while the effect, `b/(b+c)`, does not — 68% versus 32%.
Two authoritative surfaces disagreeing about direction, with the number that
would have exposed it being the one number that matches.

The field is no longer written; `mcnemar_tests` (a list, each entry naming its
arm ordering explicitly) replaces it. Replaying an artifact that still has the
old field prints a loud WARNING rather than quietly preferring one surface.
**Cite `mcnemar_tests`. Never the flat field.**

### What the canonical datum does NOT contain, and why we are not adding it

The datum predates `pricing_as_of`, `price_table`, `break_even_*`, `error_kind`
and `mcnemar_tests`. **Those fields exist only in artifacts from fresh runs.**
This datum has none of them, and it still carries the inverted flat `mcnemar`.

That is stated rather than fixed, deliberately. Back-filling derived fields into
an existing measurement would make the file assert things it did not record —
the precise move that produced the missing-root and inverted-`mcnemar` problems
in the first place. A datum is what was measured; everything else is derived and
belongs in the analysis.

So for this datum: **`CANONICAL_ANALYSIS.md` is authoritative for the McNemar
tests, the pricing regime and the economics verdict**, and the JSON is
authoritative only for `rows` (the actual measurement) and the run identity.
The next fresh measurement will carry all of it in the machine record; this one
cannot without being rewritten, and rewriting it is worse than saying so.

### Regeneration is now safe, which it previously was not

Two things made "just regenerate it" dangerous until round 5:

**Economics used to read the wall clock.** `break_even()` defaulted the pricing
date to `datetime.now()`, so replaying this datum on 2026-09-01 would have
printed `PROFITABLE` where the committed analysis says `NOT PROFITABLE AT ANY r`
— identical rows, opposite verdict, no code change, and this very document
instructing the reader to regenerate. Found independently by two model families.
Fixed: `break_even()` has no wall-clock default, every artifact records
`pricing_as_of` / `price_regime` / `price_table` / `break_even_threshold`, and a
replay resolves economics from the recorded date. Asking what the datum says
under a different regime is still possible, but only *explicitly*:

```
… --from-json <datum> --pricing-as-of 2026-09-01     # → PROFITABLE, as an ASKED question
```

This datum predates the field, so its date is **inferred from the filename stamp**
and the run prints a `NOTE:` saying so. An inferred date must never be mistaken
for a recorded one; `pricing_as_of_source` carries that distinction forward into
any artifact derived from it.

**The n=150 files used to replay happily.** See below — that is now enforced,
not merely requested.

## Why the n=210 "siblings" were never siblings

Three n=210 artifacts were tracked. All three carry **byte-identical row data**
(`sha256(rows)[:16] = 97a56d9f44df3b44`) because each was produced by
`--from-json` re-analysis of the one before it:

```
20260810T133508Z  (the actual measurement — SEE BELOW)
   └─ 133611Z  ─ replay ─▶ 135511Z  ─ replay ─▶ 135735Z
```

They are three *analyses* of one experiment, not three experiments. So pruning
the two redundant replays costs zero replayability — the objection that kept
them and the objection that wanted them gone were both reasoning about a set of
independent runs that does not exist. Nobody had checked whether the siblings
were siblings.

Untracked as redundant: `135511Z` and `135735Z` (`.json` + `.md`), and
`133611Z.md` (an analysis superseded by a scoring change).

## The provenance gap, stated rather than carried

**The root of that chain, `20260810T133508Z`, is not in the repo and not on
disk.** Every artifact we kept is a replay of a file that no longer exists.

No data was lost — each replay copies the rows forward verbatim, which is why
all three hash identically, so `133611Z` holds the measured rows exactly as
measured. But the filename asserts a measurement time that is not the
measurement time, and that is worth writing down rather than leaving for
someone to rediscover. The `source_json` field is what makes the chain
recoverable at all; it is the reason this gap is a footnote instead of an
unknown.

## Kept but NOT usable as data

`20260810T071311Z` and `20260810T071425Z` (`n150`) are retained as evidence of a
documented defect, not as results. Both are:

- **150/150 physics** while every artifact labelled them `mmlu_pro` — a
  single-category sample presented as the benchmark (Wu, round 1). This is the
  defect that moved the headline from +48pp to +17–25pp.
- **3 calls per task** (`a1`, `a2`, `b1`), so no disjoint arm and no `agree(B,B)`.
- **No raw output persisted**, so they can only ever reproduce that day's parser.

They must not be cited for any number, and the script now **enforces** that
rather than asking for it.

Until round 5 this section was a prose gate against a script that cheerfully
ignored it. Replaying one of these files *succeeded*: `agreement()` counted every
missing `b2` as an abstention, so the `persona-B self` arm was fabricated out of
nothing (`scored=0, abstained=150`, and a meaningless `r (abstain escalates) =
100%`), the process exited **0**, and it reprinted the discredited **+48pp**
physics-only separation into a freshly-timestamped file that looked exactly like
a current analysis. A document saying "don't cite this" is worth very little
against a tool that regenerates it on request.

The replay path now derives the required call set from the same `ARMS` definition
the analysis uses, and refuses a datum missing any of them:

```
REFUSING TO REPLAY …_n150.json: rows are missing call(s) ['b2'], which this
analysis requires. This datum predates the current arm design — see
results/CANONICAL.md. Re-analysing it would fabricate the missing arm from
abstentions and reprint superseded numbers under a fresh timestamp.
```
