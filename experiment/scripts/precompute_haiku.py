#!/usr/bin/env python3
"""Precompute cloud-Haiku answers for the demo questions (temp 0, deterministic —
identical to a live call). Writes demo/haiku_answers.json keyed by task_id so the
theatre can show 'cloud Haiku (paid)' beside the live local route without a
33s claude -p stall on stage. The point is cost parity, not a speed race."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from langchain_core.messages import HumanMessage, SystemMessage
from chat_claude_code import ChatClaudeCode
from benchmarks.bbh import format_prompt, extract_choice
from benchmarks.hyperspecialist import ANSWER_PERSONA

DEMO = Path(__file__).resolve().parent.parent.parent / "demo"
data = json.loads((DEMO / "demo_data.json").read_text())
out = {}
haiku = ChatClaudeCode(model="haiku")
for q in data["questions"]:
    prompt = format_prompt(q["question"], {"label": q["labels"], "text": q["options"]})
    t = time.time()
    try:
        raw = haiku.invoke([SystemMessage(content=ANSWER_PERSONA),
                            HumanMessage(content=prompt)]).content
        letter = extract_choice(raw) or "?"
    except Exception as e:
        letter, raw = "?", f"err: {e}"
    rec = {"answer": letter, "correct": letter == q["gold"], "secs": round(time.time()-t, 1)}
    out[q["task_id"]] = rec
    (DEMO / "haiku_answers.json").write_text(json.dumps(out, indent=2))  # flush each
    print(f"{q['task_id']}: {letter} (gold {q['gold']}) {rec['secs']}s", flush=True)
print("done")
