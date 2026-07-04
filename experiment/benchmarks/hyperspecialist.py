"""Hyperspecialist routing — a deployable type-router that shells out to local
domain-specialist models via Ollama.

Unlike the oracle (which peeks at ground truth), this is a REAL router:
  1. a cheap classifier (qwen2.5:0.5b, ~400MB) reads the question and names its domain
  2. the router shells out to the domain SPECIALIST for that type
  3. falls back to the generalist when the type is unclear ("other")

This is the deployable counterpart to the oracle ceiling measured by the
type-routing probe: how much of the 0.90 best-of ceiling can a cheap,
ground-truth-free router actually capture?
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

BASE_URL = "http://localhost:11434"

CLASSIFIER_MODEL = "qwen2.5:0.5b"
GENERAL_MODEL = "qwen2.5:7b-instruct-q4_K_M"
SPECIALISTS = {
    "math": "hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M",
    "code": "qwen2.5-coder:7b-instruct",
}

ANSWER_PERSONA = (
    "You are a careful, methodical reasoner. Work through the problem step by step.\n"
    "End with exactly one line: Answer: X\n"
    "where X is the letter of your chosen option (A, B, C, ...). No text after that line."
)

CLASSIFIER_PERSONA = (
    "You are a fast query classifier. Read the question and output its single "
    "best domain from this exact list: math, code, other.\n"
    "- math: arithmetic, algebra, calculus, probability, formal logic, proofs.\n"
    "- code: programming, algorithms, data structures, software.\n"
    "- other: anything else (law, biology, history, business, ...).\n"
    "Reply with ONE word only: math, code, or other."
)

_VALID = {"math", "code", "other"}


def classify_type(question: str, *, base_url: str = BASE_URL) -> str:
    """One cheap 0.5B call -> a domain label in {math, code, other}."""
    clf = ChatOllama(model=CLASSIFIER_MODEL, base_url=base_url, temperature=0.0,
                     num_predict=8)
    raw = clf.invoke(
        [SystemMessage(content=CLASSIFIER_PERSONA), HumanMessage(content=question)]
    ).content
    token = re.sub(r"[^a-z]", "", str(raw).strip().lower()[:8])
    for label in _VALID:
        if token.startswith(label):
            return label
    return "other"


def pick_model(domain: str) -> tuple[str, str]:
    """Map a domain -> (role, ollama_tag). 'other' falls back to the generalist."""
    if domain in SPECIALISTS:
        return f"{domain}_specialist", SPECIALISTS[domain]
    return "general", GENERAL_MODEL


def route(task: dict, *, base_url: str = BASE_URL, num_predict: int = 1536) -> dict:
    """Classify the task's type, shell out to the chosen model, return a record.

    Returns {domain, role, model, output} — the caller scores `output`.
    """
    domain = classify_type(task["question"], base_url=base_url)
    role, tag = pick_model(domain)
    llm = ChatOllama(model=tag, base_url=base_url, temperature=0.0,
                     num_predict=num_predict)
    output = llm.invoke(
        [SystemMessage(content=ANSWER_PERSONA), HumanMessage(content=task["prompt"])]
    ).content
    return {"domain": domain, "role": role, "model": tag, "output": output}


def arm_hyperspecialist(task: dict) -> tuple[str, int]:
    """Echo arm: type-route to a local specialist. 2 sub-calls (classify + answer)."""
    rec = route(task)
    return rec["output"], 2
