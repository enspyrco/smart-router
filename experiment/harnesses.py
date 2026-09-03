"""Inference harnesses — the experimental variable.

A harness is everything wrapped around a frozen model: system prompt, user
prompt, and answer parser. Changing a harness must not change the model, the
questions, or the scoring, so that an accuracy delta is attributable to the
harness alone.

Harnesses are *data*, not code paths. Adding H1 is one entry in ``HARNESSES``;
the runner and the analysis need no changes.

H0 is deliberately boring. It is the control, not a first attempt at a good
prompt — its only job is to be a defensible floor. Resist tuning it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from benchmarks.bbh import extract_choice

# Every harness includes this instruction. The answer line is a *scoring*
# requirement, not a prompting technique — without a parseable answer an
# otherwise-correct response scores zero, which would confound harness quality
# with output-format compliance. It comes first so a slow local model cannot
# lose an otherwise-correct answer when its reasoning hits the output limit.
# Keep it identical across harnesses.
ANSWER_CONTRACT = (
    "Begin your reply with exactly one line:\n"
    "Answer: X\n"
    "where X is the letter of the correct choice. Then give no more than "
    "eight concise sentences of reasoning."
)


def format_choices(choices: dict) -> str:
    labels = choices["label"]
    texts = choices["text"]
    return "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))


@dataclass(frozen=True)
class Harness:
    """A named prompt strategy over a frozen model."""

    name: str
    system: str
    template: str
    notes: str = ""
    parse: Callable[[str], str | None] = field(default=extract_choice)

    def render(self, task: dict) -> str:
        return self.template.format(
            question=task["question"].strip(),
            choices=format_choices(task["choices"]),
            answer_contract=ANSWER_CONTRACT,
        )


H0 = Harness(
    name="H0",
    notes="Control. Minimal instruction; no decomposition, no elimination, "
    "no verification. Every other harness is measured against this.",
    system=(
        "Solve the following multiple-choice question.\n\n"
        "Think through the problem carefully.\n"
        "Return:\n"
        "1. final answer\n"
        "2. brief reasoning"
    ),
    template=("{question}\n\nChoices:\n{choices}\n\n{answer_contract}"),
)


HARNESSES: dict[str, Harness] = {
    "H0": H0,
}


def get_harness(name: str) -> Harness:
    try:
        return HARNESSES[name]
    except KeyError:
        raise SystemExit(
            f"unknown harness {name!r}; available: {', '.join(sorted(HARNESSES))}"
        ) from None
