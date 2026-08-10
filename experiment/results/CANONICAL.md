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
| **Analysis** | `20260810T212604Z_agreement_baseline_mmlu_pro_haiku_n210.md` |
| n | 210 tasks, 15 per category across all 14 MMLU-Pro categories |
| Calls | 4 per task (`a1`, `a2`, `b1`, `b2`), raw output persisted |
| Model | `claude-haiku-4-5-20251001`, temperature 1.0, max_tokens 4096 |
| Personas | sha256[:12] `9e1c3c3a871e` |

**Cite nothing else.** Regenerate the analysis rather than reading a committed
`.md` that predates a scoring change: `.md` files are *derived*, the `.json` is
the datum, and this repo has already shipped analyses that the code had since
outgrown.

```
python3 scripts/measure_agreement_baseline.py \
  --from-json results/20260810T133611Z_agreement_baseline_mmlu_pro_haiku_n210.json
```

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

They cannot be replayed by the current script (it needs `b2`) and must not be
cited for any number.
