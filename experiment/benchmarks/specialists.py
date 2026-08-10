"""Domain-specialist model registry for Echo routing (§10).

Specialists run locally via Ollama, so they cost **0 cost units** — same
accounting as ``echo-small-judge``'s local Qwen judge. That is what makes a
specialist tier economically viable: it is latency, not spend.

Model tags must exist locally. Pull before use, e.g.::

    ollama pull adrienbrault/saul-instruct-v1:Q5_K_S
    ollama pull qwen2-math:7b
    ollama pull qwen2.5-coder:7b

Override the map without editing code via ``ECHO_SPECIALIST_MAP`` (JSON)::

    export ECHO_SPECIALIST_MAP='{"law": "adrienbrault/saul-instruct-v1:Q2_K"}'

Categories with no credible open specialist (chemistry, philosophy, history)
are deliberately absent — tasks in those categories fall through to the normal
Haiku/Sonnet path. See ``SPECIALIST_MODEL_RESEARCH.md`` for why.
"""

from __future__ import annotations

import json
import os

from run_pilot import SMALL_JUDGE_BASE_URL, _HAS_OLLAMA

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None  # type: ignore[misc, assignment]

SPECIALIST_BASE_URL = os.environ.get("ECHO_SPECIALIST_BASE_URL", SMALL_JUDGE_BASE_URL)

# MMLU-Pro category -> local Ollama tag.
DEFAULT_SPECIALIST_MAP: dict[str, str] = {
    "law": "adrienbrault/saul-instruct-v1:Q5_K_S",
    "math": "qwen2-math:7b",
    "computer science": "qwen2.5-coder:7b",
}


def specialist_map() -> dict[str, str]:
    """Category -> model tag, with ``ECHO_SPECIALIST_MAP`` merged over defaults."""
    mapping = dict(DEFAULT_SPECIALIST_MAP)
    raw = os.environ.get("ECHO_SPECIALIST_MAP")
    if raw:
        try:
            override = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ECHO_SPECIALIST_MAP is not valid JSON: {exc}") from exc
        if not isinstance(override, dict):
            raise ValueError("ECHO_SPECIALIST_MAP must be a JSON object")
        mapping.update({str(k): str(v) for k, v in override.items()})
    return mapping


def specialist_tag_for(category: str | None) -> str | None:
    """Model tag for a category, or None when no specialist is registered."""
    if not category:
        return None
    return specialist_map().get(category.strip().lower())


def has_specialist(category: str | None) -> bool:
    return specialist_tag_for(category) is not None


def load_specialist(category: str | None):
    """Instantiate the specialist chat model for ``category``.

    Returns None when no specialist is registered for the category. Raises when
    one *is* registered but Ollama is unavailable — a silent fallback there would
    quietly turn a specialist arm back into plain Echo and corrupt the result.
    """
    tag = specialist_tag_for(category)
    if tag is None:
        return None
    if not _HAS_OLLAMA or ChatOllama is None:
        raise RuntimeError(
            f"Specialist {tag!r} for category {category!r} requires langchain-ollama "
            "and a running Ollama. Run: pip install langchain-ollama && "
            f"ollama pull {tag}"
        )
    return ChatOllama(model=tag, base_url=SPECIALIST_BASE_URL, temperature=0)
