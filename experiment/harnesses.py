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

# The answer line is a scoring requirement. H0 retains the original contract;
# variants may change ordering or verbosity only under a distinct harness name.
ANSWER_CONTRACT = (
    "End your reply with exactly one line:\n"
    "Answer: X\n"
    "where X is the letter of the correct choice."
)

CONCISE_ANSWER_CONTRACT = (
    "Give no more than eight concise sentences of reasoning. Then end your "
    "reply with exactly one line:\n"
    "Answer: X\n"
    "where X is the letter of the correct choice."
)

ANSWER_FIRST_CONTRACT = (
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
    answer_contract: str = ANSWER_CONTRACT

    def render(self, task: dict) -> str:
        return self.template.format(
            question=task["question"].strip(),
            choices=format_choices(task["choices"]),
            answer_contract=self.answer_contract,
        )


H0 = Harness(
    name="H0",
    notes="Control. Minimal instruction; no decomposition, no elimination, "
    "no verification. Every other harness is measured against this.",
    system=(
        "Solve the following multiple-choice question.\n\n"
        "Think through the problem carefully.\n"
        "Return:\n"
        "1. reasoning\n"
        "2. final answer"
    ),
    template=("{question}\n\nChoices:\n{choices}\n\n{answer_contract}"),
)

H1 = Harness(
    name="H1",
    notes="Concise reasoning followed by the answer. Prevents long local-model "
    "responses from exhausting their output budget while preserving reasoning-first.",
    system=(
        "Solve the following multiple-choice question.\n\n"
        "Think through the problem carefully but briefly.\n"
        "Return:\n"
        "1. brief reasoning\n"
        "2. final answer"
    ),
    template=("{question}\n\nChoices:\n{choices}\n\n{answer_contract}"),
    answer_contract=CONCISE_ANSWER_CONTRACT,
)

H_ANSWER_FIRST = Harness(
    name="H-answer-first",
    notes="Answer-first diagnostic. Fast and parseable, but reduced accuracy in "
    "the initial five-question Qwen computer-science run.",
    system=(
        "Solve the following multiple-choice question.\n\n"
        "Return:\n"
        "1. final answer\n"
        "2. brief reasoning"
    ),
    template=("{question}\n\nChoices:\n{choices}\n\n{answer_contract}"),
    answer_contract=ANSWER_FIRST_CONTRACT,
)


HARNESSES: dict[str, Harness] = {
    "H0": H0,
    "H1": H1,
    "H-answer-first": H_ANSWER_FIRST,
}


def get_harness(name: str) -> Harness:
    try:
        return HARNESSES[name]
    except KeyError:
        raise SystemExit(
            f"unknown harness {name!r}; available: {', '.join(sorted(HARNESSES))}"
        ) from None
