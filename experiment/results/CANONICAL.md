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
  --from-json results/20260810T133611Z_agreement_baseline_mmlu_pro_haiku_n210.json
```

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
