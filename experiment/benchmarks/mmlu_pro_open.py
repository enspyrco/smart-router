"""Open-ended (options-stripped) MMLU-Pro variant.

Why this exists
---------------
The per-category cheap/expensive table that both router designs rest on was
built *entirely* from ten-option multiple choice. Resample-or-Reroute
(arXiv 2607.08665) attributes agreement failure to answer **format** rather
than subject. If that is right for us, the table is an MCQ artefact and
describes nothing about real questions.

This module isolates format from domain in one step: the *same* questions, the
*same* subjects, options removed. Nothing else changes.

The grader problem
------------------
MCQ scoring is free and deterministic — extract a letter, compare. Open-ended
answers have no letter, so correctness needs a grader model, and that grader's
own error lands directly on the dependent variable.

The mitigation here is a **tiered** scorer. A verdict records which layer
decided it:

    exact       normalized string equality      deterministic
    numeric     both sides parse as numbers     deterministic
    unparseable no answer could be extracted    deterministic
    grader      genuine ambiguity, model asked  MODEL-DEPENDENT
    undecided   ambiguous and no grader wired   not counted either way

Only the ``grader`` tier can be wrong for grader reasons, so the analysis can
report exactly how much of the result rests on the model's judgement. If the
headline flips when grader-tier rows are excluded, the finding is grader
artefact, not signal.

``undecided`` exists so that a missing grader cannot silently score as a fail —
that would bias every category by its own ambiguity rate, which is precisely
the confound we are trying to measure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from benchmarks.bbh import _clean_label
from benchmarks.mmlu_pro import ALL_CATEGORIES, category_slug, options_to_choices

# A grader takes (candidate, gold, question) and returns True if equivalent.
Grader = Callable[[str, str, str], bool]

_LEADING_ARTICLES = ("the ", "a ", "an ")

_ANSWER_LINE = re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")


@dataclass(frozen=True)
class Verdict:
    """Outcome of grading one open-ended response.

    ``passed`` is Optional because ``undecided`` is a real third state: an
    ambiguous answer with no grader wired is not evidence either way.
    """

    passed: Optional[bool]
    tier: str
    detail: str
    candidate: Optional[str] = None


def format_prompt_open_ended(question: str) -> str:
    """Model-facing prompt with NO options.

    Deliberately mirrors the MCQ prompt's shape (reasoning, then one final
    answer line) so that prompt *structure* is not an extra moving part
    between the two arms. The only difference is the absence of choices.
    """
    return (
        f"{question.strip()}\n\n"
        "Reply with your reasoning, then end with exactly one line:\n"
        "Answer: <your answer>\n"
        "Give the answer itself, as briefly as possible — no explanation on "
        "that line, and no restatement of the question."
    )


def row_to_task_open_ended(category: str, row: dict) -> dict:
    """Build an open-ended task from a raw MMLU-Pro row.

    Gold becomes the *text* of the correct option rather than its letter. The
    letter is retained as ``gold_letter`` so a row can still be joined against
    its MCQ counterpart.
    """
    options = list(row["options"])
    if not options:
        raise ValueError(f"MMLU-Pro question {row.get('question_id')} has no options")

    choices = options_to_choices(options)
    labels = [_clean_label(label) for label in choices["label"]]
    gold_letter = _clean_label(str(row["answer"]))
    if gold_letter not in labels:
        raise ValueError(f"Gold {row['answer']!r} not in choice labels {labels}")
    gold_text = options[labels.index(gold_letter)]

    slug = category_slug(category)
    return {
        # "open" in the id keeps MCQ and open-ended rows distinct if the two
        # sweeps are ever concatenated into one JSONL.
        "task_id": f"mmlu_pro_open/{slug}/{row['question_id']}",
        "prompt": format_prompt_open_ended(row["question"]),
        "gold": gold_text,
        "gold_letter": gold_letter,
        # Stratification key — see gold_form().
        "gold_form": gold_form(gold_text),
        "benchmark": "mmlu_pro",
        "format": "open_ended",
        "category": category,
        "question_id": row["question_id"],
        "question": row["question"],
        # Distractors are kept for analysis only — they are NOT shown to the
        # model. They let us ask afterwards whether a wrong open-ended answer
        # happened to land on a distractor.
        "distractors": [o for i, o in enumerate(options) if labels[i] != gold_letter],
    }


def extract_free_answer(text: str) -> Optional[str]:
    """Pull the model's final answer out of a free-text response.

    Order matters. An explicit ``Answer:`` line is the contract we asked for,
    so it wins; ``\\boxed{}`` is next because math-tuned models emit it
    regardless of instructions; the last non-empty line is a weak fallback.
    Within the explicit forms the LAST occurrence wins, so a chain of thought
    that revises itself resolves to the revision.
    """
    if not text or not text.strip():
        return None

    matches = list(_ANSWER_LINE.finditer(text))
    if matches:
        return _tidy(matches[-1].group(1))

    boxed = list(_BOXED.finditer(text))
    if boxed:
        return _tidy(boxed[-1].group(1))

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return _tidy(lines[-1])
    return None


def _tidy(value: str) -> Optional[str]:
    cleaned = value.strip().strip("*` ").rstrip(".,;:!").strip()
    return cleaned or None


def normalize_free_text(text: str) -> str:
    """Aggressive normalization for the deterministic equality tier."""
    lowered = str(text).strip().lower()
    lowered = re.sub(r"[^\w\s.\-/]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    for article in _LEADING_ARTICLES:
        if lowered.startswith(article):
            lowered = lowered[len(article):]
            break
    return lowered.strip(" .")


def sequence_tokens(text: str) -> Optional[list[str]]:
    """Split an ordered short-token answer like ``"False, True"`` into parts.

    Exists because of a live grader failure: gold ``False, True`` vs candidate
    ``True, False`` was graded EQUIVALENT. Order-sensitive tuple answers are
    common in MMLU-Pro's multi-statement items, and they are exactly the case a
    text-similarity judgement gets wrong — while being trivially decidable
    deterministically.

    Conservative on purpose: only 2-4 comma/semicolon separated parts, each at
    most two words. Anything longer is prose and belongs to the grader.
    """
    raw = str(text).strip()
    if not raw:
        return None
    parts = [p.strip() for p in re.split(r"[,;]", raw) if p.strip()]
    if not 2 <= len(parts) <= 4:
        return None
    if any(len(p.split()) > 2 for p in parts):
        return None
    return [normalize_free_text(p) for p in parts]


def gold_form(gold: str) -> str:
    """Classify the shape of a gold answer.

    Used to stratify the results. Stripping options removes the answer-FORM
    convention for tuple-shaped golds (nothing tells the model to answer
    "False, True" in question order), so those items measure something slightly
    different from the single-value ones. Reporting them separately keeps that
    visible instead of letting it ride inside the headline.
    """
    if sequence_tokens(gold) is not None:
        return "ordered_tuple"
    if numeric_value(gold) is not None:
        return "numeric"
    return "single_value"


def numeric_value(text: str) -> Optional[float]:
    """Return a float if the whole string is a number, else None.

    Deliberately strict: ``42 metres`` is NOT numeric, because unit handling is
    exactly the kind of judgement the grader exists for. Being permissive here
    would move errors from the visible grader tier into the invisible
    deterministic tier.
    """
    candidate = str(text).strip().replace(",", "")
    if not candidate:
        return None
    try:
        return float(candidate)
    except ValueError:
        return None


def score_open_ended(
    model_output: str,
    task: dict[str, Any],
    grader: Optional[Grader] = None,
) -> Verdict:
    """Grade one open-ended response, deferring to the grader only when needed."""
    gold = str(task.get("gold", ""))
    question = str(task.get("question", ""))

    candidate = extract_free_answer(model_output)
    if candidate is None:
        return Verdict(False, "unparseable", "no answer could be extracted", None)

    norm_candidate = normalize_free_text(candidate)
    norm_gold = normalize_free_text(gold)
    if norm_candidate == norm_gold:
        return Verdict(True, "exact", "normalized string match", candidate)

    cand_num = numeric_value(candidate)
    gold_num = numeric_value(gold)
    if cand_num is not None and gold_num is not None:
        if cand_num == gold_num:
            return Verdict(True, "numeric", "numeric equality", candidate)
        return Verdict(False, "numeric", f"expected {gold} got {candidate}", candidate)

    # Ordered tuples are decidable without a model, and the grader demonstrably
    # gets them wrong (it called "True, False" equivalent to "False, True").
    gold_seq = sequence_tokens(gold)
    cand_seq = sequence_tokens(candidate)
    if gold_seq is not None and cand_seq is not None and len(gold_seq) == len(cand_seq):
        if gold_seq == cand_seq:
            return Verdict(True, "sequence", "ordered token match", candidate)
        return Verdict(False, "sequence", f"expected {gold} got {candidate}", candidate)

    if grader is None:
        return Verdict(None, "undecided", "ambiguous, no grader wired", candidate)

    try:
        equivalent = bool(grader(candidate, gold, question))
    except Exception as exc:  # noqa: BLE001 - a grader fault must not pass silently
        return Verdict(False, "grader_error", f"grader failed: {exc}", candidate)

    detail = "grader: equivalent" if equivalent else f"grader: expected {gold} got {candidate}"
    return Verdict(equivalent, "grader", detail, candidate)


def load_mmlu_pro_open(
    categories: list[str] | None = None,
    *,
    split: str = "test",
    n_per_category: int | None = None,
    start: int = 0,
    n: int | None = None,
) -> list[dict]:
    """Load MMLU-Pro rows as open-ended tasks.

    Slicing semantics are identical to ``load_mmlu_pro`` on purpose: the same
    ``start``/``n_per_category`` must select the same underlying questions, or
    the two arms are not comparable and the experiment is meaningless.
    """
    from collections import defaultdict

    from datasets import load_dataset

    names = categories or list(_PILOT)
    unknown = [name for name in names if name not in ALL_CATEGORIES]
    if unknown:
        raise ValueError(f"Unknown MMLU-Pro categories: {unknown}")

    ds = load_dataset("TIGER-Lab/MMLU-Pro")[split]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        cat = row["category"]
        if cat in names:
            by_category[cat].append(row)

    tasks: list[dict] = []
    for cat in names:
        rows = sorted(by_category[cat], key=lambda r: int(r["question_id"]))
        rows = rows[start:]
        if n_per_category is not None:
            rows = rows[:n_per_category]
        for row in rows:
            tasks.append(row_to_task_open_ended(cat, row))
        if n is not None and len(tasks) >= n:
            return tasks[:n]
    return tasks[:n] if n is not None else tasks


from benchmarks.mmlu_pro import PILOT_CATEGORIES as _PILOT  # noqa: E402
