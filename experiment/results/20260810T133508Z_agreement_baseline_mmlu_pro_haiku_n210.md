# Agreement baseline — mmlu_pro, model=haiku, n=210

Generation config: `{'endpoint': 'https://api.anthropic.com/v1/messages', 'model_alias': 'haiku', 'model_id': 'claude-haiku-4-5-20251001', 'temperature': 1.0, 'max_tokens': 4096}`
Personas sha256[:12]: `9e1c3c3a871e`
Categories: `{'biology': 15, 'business': 15, 'chemistry': 15, 'computer science': 15, 'economics': 15, 'engineering': 15, 'health': 15, 'history': 15, 'law': 15, 'math': 15, 'other': 15, 'philosophy': 15, 'physics': 15, 'psychology': 15}`

Tool-free raw endpoint (ChatOAuth). `claude --print` was measured reading files on both tiers, so it cannot be used for a measurement whose dependent variable is agreement.

| arm | scored | correct | wrong | abstained | r | r 95% CI | r (abstain escalates) | P(agree\|correct) | P(agree\|wrong) | separation | economics | mechanism |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persona (a1 vs b1) | 210 | 156 | 54 | 0 | **8%** | [5%, 12%] | 8% | 97% | 80% | +17pp | PROFITABLE — production 95% upper bound 12% < 33% | mechanism holds |
| control (a1 vs a2) | 210 | 156 | 54 | 0 | **11%** | [8%, 16%] | 11% | 95% | 70% | +25pp | PROFITABLE — production 95% upper bound 16% < 33% | mechanism holds |
| cross (a2 vs b1) | 210 | 155 | 55 | 0 | **11%** | [7%, 16%] | 11% | 95% | 71% | +25pp | PROFITABLE — production 95% upper bound 16% < 33% | n/a — first call is not Echo's accept path |
| persona-B self (b1 vs b2) | 210 | 156 | 54 | 0 | **10%** | [7%, 15%] | 10% | 94% | 78% | +16pp | PROFITABLE — production 95% upper bound 15% < 33% | n/a — first call is not Echo's accept path |

## Do personas beat plain resampling?

McNemar on persona-vs-control discordant pairs: b=15, c=7, **p=0.134**.

**No detectable persona effect** (p=0.134 over 22 discordant tasks). This is a failure to reject, NOT proof of equivalence — for an equivalence claim, pre-specify a margin and run TOST.

`cross (a2 vs b1)` shares a2 with the control and b1 with the persona arm — it is NOT an independent falsifier, and an earlier revision wrongly claimed it shared nothing. `persona-B self (b1 vs b2)` IS disjoint from the control ({b1,b2} ∩ {a1,a2} = ∅), and is also the agree(B,B) measurement that was previously missing.

## Reading the numbers

- `r` gates on the Wilson upper bound vs the break-even, not a point estimate.
- `r (abstain escalates)` is the PRODUCTION policy: Echo cannot accept an unparseable cheap answer. Excluding abstentions biases r DOWNWARD and flatters Echo; both are shown so neither convention hides.
- `wrong` means the FIRST call was wrong — the answer Echo would accept. It is not the free-floating claim that the model reproduces its own errors.
- agree(B,B) is NOT measured: persona B's self-agreement is assumed to mirror A's. That symmetry is untested.