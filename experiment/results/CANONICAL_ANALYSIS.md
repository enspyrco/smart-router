# Agreement baseline — mmlu_pro, model=haiku, n=210

Generation config: `{'endpoint': 'https://api.anthropic.com/v1/messages', 'model_alias': 'haiku', 'model_id': 'claude-haiku-4-5-20251001', 'temperature': 1.0, 'max_tokens': 4096}`
Personas sha256[:12]: `9e1c3c3a871e`
Categories: `{'biology': 15, 'business': 15, 'chemistry': 15, 'computer science': 15, 'economics': 15, 'engineering': 15, 'health': 15, 'history': 15, 'law': 15, 'math': 15, 'other': 15, 'philosophy': 15, 'physics': 15, 'psychology': 15}`

Tool-free raw endpoint (ChatOAuth). `claude --print` was measured reading files on both tiers, so it cannot be used for a measurement whose dependent variable is agreement.

| arm | scored | correct | wrong | abstained | r | r 95% CI | r (abstain escalates) | P(agree\|correct) | P(agree\|wrong) | separation | economics | mechanism |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persona (a1 vs b1) | 210 | 156 | 54 | 0 | **8%** | [5%, 12%] | 8% | 97% | 80% | +17pp | NOT PROFITABLE AT ANY r — haiku->sonnet @ sonnet introductory pricing (2026-08-01 to 2026-08-31): 2xhaiku ($2.0/MTok) already costs >= sonnet ($2.0/MTok) — Echo CANNOT be profitable at this tier [under list pricing: r < 33%] | separation positive but weak — conservative bound [4pp, 31pp] (over-covers; not an exact 95% difference CI) |
| control (a1 vs a2) | 210 | 156 | 54 | 0 | **11%** | [8%, 16%] | 11% | 95% | 70% | +25pp | NOT PROFITABLE AT ANY r — haiku->sonnet @ sonnet introductory pricing (2026-08-01 to 2026-08-31): 2xhaiku ($2.0/MTok) already costs >= sonnet ($2.0/MTok) — Echo CANNOT be profitable at this tier [under list pricing: r < 33%] | separation positive but weak — conservative bound [9pp, 40pp] (over-covers; not an exact 95% difference CI) |
| cross (a2 vs b1) | 210 | 155 | 55 | 0 | **11%** | [7%, 16%] | 11% | 95% | 71% | +25pp | NOT PROFITABLE AT ANY r — haiku->sonnet @ sonnet introductory pricing (2026-08-01 to 2026-08-31): 2xhaiku ($2.0/MTok) already costs >= sonnet ($2.0/MTok) — Echo CANNOT be profitable at this tier [under list pricing: r < 33%] | n/a — first call is not Echo's accept path |
| persona-B self (b1 vs b2) | 210 | 156 | 54 | 0 | **10%** | [7%, 15%] | 10% | 94% | 78% | +16pp | NOT PROFITABLE AT ANY r — haiku->sonnet @ sonnet introductory pricing (2026-08-01 to 2026-08-31): 2xhaiku ($2.0/MTok) already costs >= sonnet ($2.0/MTok) — Echo CANNOT be profitable at this tier [under list pricing: r < 33%] | n/a — first call is not Echo's accept path |

## Do personas beat plain resampling?

Three McNemar tests, not one. Every pair-vs-pair comparison available from four calls shares at least one call, so a single test cannot distinguish "no persona effect" from "the shared call thinned the discordant set". Running the same question against two different anchors is the check that the conclusion is not an artefact of the coupling (Tesla + Carnot, round 4).

| test | b | c | discordant | p | reads as |
|---|---|---|---|---|---|
| persona vs control (shared anchor: a1) | 7 | 15 | 22 | 0.134 | no detectable difference — cross-persona vs A-self resampling |
| persona vs B-self (shared anchor: b1) | 12 | 18 | 30 | 0.361 | no detectable difference — same question, anchored on the OTHER call — a robustness replicate |
| control vs B-self (DISJOINT: {a1,a2} n {b1,b2} = {}) | 13 | 11 | 24 | 0.839 | no detectable difference — not a persona-effect test — asks whether the resampling BASELINE itself differs by persona, i.e. whether 'the control' is one baseline or two. This is Wu's symmetry question, and it is the only fully uncoupled pair. |

**Headline (persona vs control (shared anchor: a1)):** b=7, c=15, **p=0.134**.

**No detectable persona effect** (p=0.134 over 22 discordant tasks). This is a failure to reject, NOT proof of equivalence — for an equivalence claim, pre-specify a margin and run TOST. Note the discordant set is thinned by the shared `a1`, so power is lower than 22 paired tasks would suggest; see the replicate row.

**Do the two anchors agree?** Compare the effects, not the p-values: McNemar's effect is the discordant split b/(b+c), with 0.5 meaning no difference. Headline 7/22 = 32% [16%, 53%]; b1-anchored replicate 12/30 = 40% [25%, 58%]. The intervals OVERLAP and both contain 0.5, so the two anchors are consistent with each other. Consistency across anchors is what licenses reading this as a statement about personas rather than about which call the two arms happen to share; tension would mean the common mode is driving the answer and neither number should be quoted.

`cross (a2 vs b1)` shares a2 with the control AND b1 with the persona arm — it is NOT an independent falsifier on either side, and an earlier revision wrongly claimed it shared nothing. `persona-B self (b1 vs b2)` is disjoint from the control ({b1,b2} ∩ {a1,a2} = ∅) but still shares b1 with the persona arm; it is the agree(B,B) measurement that was previously missing.

## Reading the numbers

- **Economics gates on `r (abstain escalates)`** — the production rate — and specifically on the Wilson UPPER bound of that rate vs the break-even, never on a point estimate and never on the flattering abstentions-dropped `r`. An earlier version of this bullet named the wrong meter while the code used the right one, which is worse than either being wrong alone (Tesla, round 4).
- `r` (abstentions dropped) is reported for comparison only. Excluding abstentions biases r DOWNWARD and flatters Echo, because parse failures correlate with hard tasks and hard tasks are where disagreement lives.
- **The break-even is date-dependent.** Sonnet 5's introductory price runs through 2026-08-31; the threshold in force is stated in the `economics` cell along with the threshold on the other side of that date.
- `wrong` means the FIRST call was wrong — the answer Echo would accept. It is not the free-floating claim that the model reproduces its own errors.
- **`separation` for `persona` and `control` is NOT two independent readings.** Both condition on `a1`, so they partition the SAME correctness split; quoting both as parallel mechanism evidence double-counts one first-call correctness frequency (Tesla, round 4). `persona-B self` conditions on `b1` and is the only row carrying an independent split.
- The separation interval is a CONSERVATIVE bound (summed Wilson half-widths), not an exact 95% CI on the difference. It over-covers, so it makes claims harder rather than easier; a Newcombe/bootstrap difference interval is the follow-up and can only widen what we claim.
- agree(B,B) IS now measured, as the `persona-B self` arm. Wu's round-1 point was that B's self-agreement was merely ASSUMED to mirror A's; the `control vs B-self` row above tests that symmetry directly rather than leaving it to eyeball.