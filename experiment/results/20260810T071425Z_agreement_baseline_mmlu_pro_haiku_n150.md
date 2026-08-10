# Agreement baseline — mmlu_pro, model=haiku, n=150

Tool-free raw endpoint (ChatOAuth). `claude --print` was measured reading files on both tiers, so it cannot be used for a measurement whose dependent variable is agreement.

| arm | scored | correct | wrong | escalation r | P(agree\|correct) | P(agree\|wrong) | separation | Echo economics | mechanism |
|---|---|---|---|---|---|---|---|---|---|
| persona A vs B | 150 | 128 | 22 | **13%** | 94% | 45% | +48pp | PROFITABLE | INSUFFICIENT — 22 wrong (need 30 each) |
| plain resample (A vs A) | 150 | 128 | 22 | **13%** | 95% | 45% | +49pp | PROFITABLE | INSUFFICIENT — 22 wrong (need 30 each) |

**Break-even is r < 33%** (Echo = 2 + 3r vs expensive-only = 3).

**Persona delta: +1pp escalation vs plain resampling.** Personas add ~nothing over resampling the same prompt. Echo could drop the persona machinery entirely and get simpler, which is a stronger claim.

`separation` = P(agree|correct) - P(agree|wrong). This is the mechanism's strength: it is how much agreement actually tells you about correctness. Near zero means agreement is not a difficulty signal on this benchmark, whatever the escalation rate is.

Abstentions are excluded, not counted as disagreement — an unparseable or refused answer is missing data, and pooling it with disagreement would inflate r and understate Echo's profitability.