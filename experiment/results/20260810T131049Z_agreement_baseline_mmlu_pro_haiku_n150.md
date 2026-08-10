# Agreement baseline — mmlu_pro, model=haiku, n=150

Generation config: `NOT RECORDED (pre-cage-match run)`
Personas sha256[:12]: `NOT RECORDED`
Categories: `NOT RECORDED (pre-cage-match run)`

Tool-free raw endpoint (ChatOAuth). `claude --print` was measured reading files on both tiers, so it cannot be used for a measurement whose dependent variable is agreement.

| arm | scored | correct | wrong | abstained | r | r 95% CI | r (abstain escalates) | P(agree\|correct) | P(agree\|wrong) | separation | economics | mechanism |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persona (a1 vs b1) | 150 | 128 | 22 | 0 | **13%** | [9%, 20%] | 13% | 94% | 45% | +48pp | PROFITABLE — 95% upper bound 20% < 33% | INSUFFICIENT — 22 wrong (need 30 each) |
| control (a1 vs a2) | 150 | 128 | 22 | 0 | **13%** | [8%, 19%] | 13% | 95% | 45% | +49pp | PROFITABLE — 95% upper bound 19% < 33% | INSUFFICIENT — 22 wrong (need 30 each) |
| unshared (a2 vs b1) | 150 | 130 | 20 | 0 | **13%** | [8%, 19%] | 13% | 93% | 50% | +43pp | PROFITABLE — 95% upper bound 19% < 33% | INSUFFICIENT — 20 wrong (need 30 each) |

## Do personas beat plain resampling?

McNemar on persona-vs-control discordant pairs: b=7, c=8, **p=1.000**.

**No detectable persona effect** (p=1.000 over 15 discordant tasks). This is a failure to reject, NOT proof of equivalence — for an equivalence claim, pre-specify a margin and run TOST.

The `unshared (a2 vs b1)` arm is an A-vs-B comparison sharing NO call with the control, so it answers the shared-a1 objection with data rather than argument: if it tracks the persona arm, the pairing is not manufacturing the similarity.

## Reading the numbers

- `r` gates on the Wilson upper bound vs the break-even, not a point estimate.
- `r (abstain escalates)` is the PRODUCTION policy: Echo cannot accept an unparseable cheap answer. Excluding abstentions biases r DOWNWARD and flatters Echo; both are shown so neither convention hides.
- `wrong` means the FIRST call was wrong — the answer Echo would accept. It is not the free-floating claim that the model reproduces its own errors.
- agree(B,B) is NOT measured: persona B's self-agreement is assumed to mirror A's. That symmetry is untested.