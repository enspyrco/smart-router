# Agreement baseline — mmlu_pro, model=haiku, n=12

Tool-free raw endpoint (ChatOAuth). `claude --print` was measured reading files on both tiers, so it cannot be used for a measurement whose dependent variable is agreement.

| arm | scored | abstained | escalation r | P(agree\|correct) | P(agree\|wrong) | separation | Echo economics |
|---|---|---|---|---|---|---|---|
| persona A vs B | 12 | 0 | **33%** | 80% | 0% | +80pp | NOT PROFITABLE — costs more than expensive-only |
| plain resample (A vs A) | 12 | 0 | **33%** | 60% | 100% | -40pp | NOT PROFITABLE — costs more than expensive-only |

**Break-even is r < 33%** (Echo = 2 + 3r vs expensive-only = 3).

**Persona delta: +0pp escalation vs plain resampling.** Personas add ~nothing over resampling the same prompt. Echo could drop the persona machinery entirely and get simpler, which is a stronger claim.

`separation` = P(agree|correct) - P(agree|wrong). This is the mechanism's strength: it is how much agreement actually tells you about correctness. Near zero means agreement is not a difficulty signal on this benchmark, whatever the escalation rate is.

Abstentions are excluded, not counted as disagreement — an unparseable or refused answer is missing data, and pooling it with disagreement would inflate r and understate Echo's profitability.