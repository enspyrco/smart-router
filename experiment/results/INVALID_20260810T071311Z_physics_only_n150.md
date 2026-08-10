# ⛔ DEFECT SPECIMEN — DO NOT CITE ANY NUMBER FROM THIS FILE

This is not a result. It is evidence of a documented defect, retained so the
mistake stays visible. See `results/CANONICAL.md`.

**Why it is invalid:**
- The sample is **150/150 physics**, labelled `mmlu_pro`. A single-category slice
  presented as the whole benchmark. This is the defect that moved the headline
  from +48pp to +17-25pp.
- Only **3 calls per task** (`a1`,`a2`,`b1`) — no disjoint arm, no agree(B,B).
- **No raw output persisted**, so it can only ever reproduce that day's parser.
- Its economics predate the corrected price table AND the introductory-pricing
  window, so any PROFITABLE verdict below is doubly wrong.

The script now **refuses** to replay it. The original text is preserved below,
struck through in intent if not in markup, purely so the record is complete.

---

# Agreement baseline — mmlu_pro, model=haiku, n=150

Tool-free raw endpoint (ChatOAuth). `claude --print` was measured reading files on both tiers, so it cannot be used for a measurement whose dependent variable is agreement.

| arm | scored | abstained | escalation r | P(agree\|correct) | P(agree\|wrong) | separation | Echo economics |
|---|---|---|---|---|---|---|---|
| persona A vs B | 150 | 0 | **13%** | 94% | 45% | +48pp | INSUFFICIENT DATA — only 22 wrong (need 30). No claim. |
| plain resample (A vs A) | 150 | 0 | **13%** | 95% | 45% | +49pp | INSUFFICIENT DATA — only 22 wrong (need 30). No claim. |

**Break-even is r < 33%** (Echo = 2 + 3r vs expensive-only = 3).

**Persona delta: NOT REPORTABLE at this sample size.** Both arms are underpowered; the delta is noise. Re-run with enough tasks that the wrong-answer cell alone holds >= 30.

`separation` = P(agree|correct) - P(agree|wrong). This is the mechanism's strength: it is how much agreement actually tells you about correctness. Near zero means agreement is not a difficulty signal on this benchmark, whatever the escalation rate is.

Abstentions are excluded, not counted as disagreement — an unparseable or refused answer is missing data, and pooling it with disagreement would inflate r and understate Echo's profitability.