"""Open-ended arms + the equivalence grader.

Scope is deliberately narrow. The claim under test is the per-category
cheap/expensive table, and that table is *haiku vs sonnet per category*. So
this module ships exactly those two arms in open-ended form.

Echo/agreement arms are NOT included. Agreement on open-ended answers needs a
grader too (two free-text answers, are they the same?), which would put a
second model-dependent layer underneath the dependent variable before we have
validated the first one. That is a separate experiment and should be run after
the grader's error rate is known.

Grader independence
-------------------
The grader defaults to Sonnet, which is the same family as the models being
graded. For *correctness* judging that would be a real problem — a same-family
judge is blind to its own family's failure modes. Here the grader's job is much
narrower: string-level equivalence ("is 'the city of Paris' the same answer as
'Paris'?"), with the gold answer supplied. That is closer to normalization than
to judgement, so the blindness risk is lower.

It is not zero, which is why ``scripts/validate_grader.py`` exists and why
every row records which tier decided it.
"""

from __future__ import annotations

import re
import textwrap
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from benchmarks.mmlu_pro_open import Grader
from chat_claude_code import ChatClaudeCode

# Mirrors the MCQ personas' shape, minus the letter contract.
OPEN_SYSTEM = textwrap.dedent("""\
    Answer the question directly. Work through it if you need to, then end with
    exactly one line:
    Answer: <your answer>
    Give the answer itself, as briefly as possible. No text after that line.
""").strip()

GRADER_SYSTEM = textwrap.dedent("""\
    You judge whether a candidate answer means the same thing as a reference
    answer to a given question.

    Reply with exactly one word: EQUIVALENT or DIFFERENT.

    Judge MEANING, not wording. These are EQUIVALENT:
      - different phrasing or word order for the same fact
      - a value with units vs the same value without them
      - a more specific but still correct form of the same answer
      - different but equal notation for the same quantity (1/2 vs 0.5)

    These are DIFFERENT:
      - a different value, entity, or claim
      - a partial answer missing something the reference states
      - a hedge or refusal that never commits to an answer
      - the SAME parts in a DIFFERENT ORDER, when the reference is a sequence.
        "True, False" is NOT equivalent to "False, True". Order carries meaning
        in multi-part answers; check each position against its counterpart.

    When genuinely unsure, reply DIFFERENT.
""").strip()

_VERDICT = re.compile(r"\b(EQUIVALENT|DIFFERENT)\b", re.IGNORECASE)


def _answer_open(model_alias: str, task: dict) -> tuple[str, int]:
    model = ChatClaudeCode(model=model_alias)
    response = model.invoke([
        SystemMessage(content=OPEN_SYSTEM),
        HumanMessage(content=task["prompt"]),
    ])
    return response.content, 1


def arm_haiku_only_open(task: dict) -> tuple[str, int]:
    return _answer_open("haiku", task)


def arm_sonnet_only_open(task: dict) -> tuple[str, int]:
    return _answer_open("sonnet", task)


OPEN_ARMS = {
    "haiku-only-open": arm_haiku_only_open,
    "sonnet-only-open": arm_sonnet_only_open,
}


def make_grader(model_alias: str = "sonnet") -> Grader:
    """Build an equivalence grader backed by one Claude alias.

    Fails CLOSED: an unparseable grader reply raises rather than defaulting to
    'equivalent'. A silent default would inflate pass rates in exactly the
    ambiguous cases the grader exists to adjudicate.
    """
    model = ChatClaudeCode(model=model_alias)

    def grade(candidate: str, gold: str, question: str) -> bool:
        prompt = (
            f"Question:\n{question.strip()}\n\n"
            f"Reference answer:\n{gold.strip()}\n\n"
            f"Candidate answer:\n{candidate.strip()}\n\n"
            "EQUIVALENT or DIFFERENT?"
        )
        response = model.invoke([
            SystemMessage(content=GRADER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        match = _VERDICT.search(str(response.content))
        if match is None:
            raise ValueError(f"grader gave no verdict: {str(response.content)[:200]!r}")
        return match.group(1).upper() == "EQUIVALENT"

    return grade


def make_recording_grader(model_alias: str, sink: list[dict]) -> Grader:
    """Grader that appends every decision to ``sink`` for later audit.

    Used by the run so that grader-tier decisions can be re-examined (and
    human-labelled) without re-running the sweep.
    """
    inner = make_grader(model_alias)

    def grade(candidate: str, gold: str, question: str) -> bool:
        try:
            verdict = inner(candidate, gold, question)
        except Exception as exc:  # noqa: BLE001 - record the fault, then re-raise
            sink.append({
                "question": question, "gold": gold, "candidate": candidate,
                "grader_model": model_alias, "verdict": None, "error": str(exc),
            })
            raise
        sink.append({
            "question": question, "gold": gold, "candidate": candidate,
            "grader_model": model_alias, "verdict": verdict, "error": None,
        })
        return verdict

    return grade
