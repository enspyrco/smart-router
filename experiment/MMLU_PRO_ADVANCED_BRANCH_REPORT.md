# MMLU-Pro Advanced Harness Branch Report

Branch: `feat/mmlu-pro-advanced`

## What Was Added

1. **MMLU-Pro harness as an extension of the BBH harness**
   - Added MMLU-Pro support using the same multiple-choice task shape as BBH.
   - MMLU-Pro questions are loaded from `TIGER-Lab/MMLU-Pro`, converted into the same prompt format, and scored by extracting the final answer letter.
   - This lets the same Echo routing arms run on BBH and MMLU-Pro without rewriting every arm.

2. **Resume support after Claude/API credit limits**
   - The MMLU-Pro runner writes each result row immediately to JSONL.
   - If Claude usage limits, rate limits, or quota errors happen mid-run, the partial file is still saved.
   - You can restart with `--resume <results-file>` and it skips completed `(task_id, arm)` pairs.

3. **LLM-as-judge arms across model families**
   - The harness reuses BBH judge arms for MMLU-Pro.
   - Supported judge families include:
     - Claude judge: `echo-judge`
     - OpenAI judges: `echo-judge-openai`, `echo-judge-openai-gpt-5.4`, `echo-judge-openai-gpt-5.4-mini`, `echo-judge-openai-gpt-5.4-nano`
     - Gemini judges: `echo-judge-gemini-pro`, `echo-judge-gemini-flash`, `echo-judge-gemini-flash-lite`
   - These judges compare two cheap Haiku answers and decide whether they agree. If they disagree, Echo escalates to Sonnet.
   - Important: this compares different **judge model families**, while the candidate answer generators are still Haiku/Sonnet.

4. **Advanced analysis for research decisions**
   - Added `scripts/analyze_mmlu_pro.py`.
   - Reports:
     - pass rate by arm
     - escalation rate
     - cost units and cost per task
     - per-category pass rate
     - gap versus `sonnet-only`
     - oracle routing diagnostics when `echo-oracle` is included

5. **Category-specialist probes**
   - The analyzer now identifies categories where a specialist model may help.
   - Example specialist probes:
     - math -> math-reasoning specialist
     - law -> legal reasoning specialist
     - health/biology -> biomedical specialist
     - computer science -> code/CS specialist
     - economics/business -> finance/economics specialist
   - These are not yet separate specialist model arms. They are analysis signals that tell us where to test specialist models next.

6. **Markdown report output**
   - `--report` generates a clean Markdown report next to the JSONL result file.
   - This is useful for sharing results with supervisors without manually copying terminal output.

## How To Run On OCI

```bash
ssh -i ~/.ssh/id_ed25519 meghana@158.179.17.233
cd ~/echo
git fetch origin
git switch feat/mmlu-pro-advanced || git switch -c feat/mmlu-pro-advanced origin/feat/mmlu-pro-advanced
git pull --ff-only
cd experiment
source .venv/bin/activate
```

If the virtual environment does not exist:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "datasets>=2.14" "langchain-core>=0.3,<0.4" "langchain>=0.3,<0.4" "langchain-openai>=0.3" "langchain-google-genai>=2.0"
```

Run tests:

```bash
python -m unittest discover -s tests -p "test_mmlu_pro*.py" -v
```

Run a small MMLU-Pro sweep:

```bash
python scripts/run_mmlu_pro_pilot.py \
  --categories physics,math,law,chemistry,philosophy \
  --n-per-category 5 \
  --arms haiku-only,sonnet-only,echo-judge-openai-gpt-5.4-mini,echo-oracle
```

Run a judge-family comparison:

```bash
python scripts/run_mmlu_pro_pilot.py \
  --categories physics,math,law,chemistry,philosophy \
  --n-per-category 5 \
  --arms haiku-only,sonnet-only,echo-judge,echo-judge-openai-gpt-5.4-mini,echo-judge-openai-gpt-5.4-nano,echo-judge-gemini-flash,echo-judge-gemini-flash-lite,echo-oracle
```

Analyze results and generate a Markdown report:

```bash
python scripts/analyze_mmlu_pro.py results/<timestamp>_mmlu_pro_n25.jsonl --report
```

Resume after limits:

```bash
python scripts/run_mmlu_pro_pilot.py \
  --categories physics,math,law,chemistry,philosophy \
  --n-per-category 5 \
  --arms haiku-only,sonnet-only,echo-judge-openai-gpt-5.4-mini,echo-oracle \
  --resume results/<partial_file>.jsonl
```

For long OCI runs, use `tmux`:

```bash
tmux new -s mmlu
```

Detach with `Ctrl-b`, then `d`.

Reconnect:

```bash
tmux attach -t mmlu
```

## How To Read The Metrics

- `pass_rate`: how often the final selected answer was correct.
- `escalation_rate`: how often Echo had to call Sonnet.
- `cost_per_task`: normalized cost estimate per question.
- `sonnet_gap`: how far an arm is behind or ahead of `sonnet-only` in a category.
- `false_accept_rate`: dangerous routing error where Echo accepts cheap output but oracle would escalate.
- `false_escalation_rate`: cost error where Echo escalates even though cheap output was enough.

The best judge is not simply the one with the lowest escalation rate. A good judge keeps pass rate high, avoids false accepts, and only reduces escalation when it is safe.

