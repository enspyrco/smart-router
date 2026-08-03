"""BBH routing arms — same Echo logic as run_pilot, MCQ prompts and scoring."""

from __future__ import annotations

import re
import textwrap
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel

from benchmarks.bbh import extract_choice, extract_choice_text, score_bbh
from benchmarks.specialists import load_specialist
from chat_claude_code import ChatClaudeCode
from run_pilot import (
    GEMINI_JUDGE_MODELS,
    OPENAI_JUDGE_MODELS,
    SMALL_JUDGE_BASE_URL,
    SMALL_JUDGE_MODEL,
    _HAS_OLLAMA,
)

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None  # type: ignore[misc, assignment]

PERSONA_A = textwrap.dedent("""\
    You are a careful, methodical reasoner. Work through the problem step by step.
    End with exactly one line: Answer: X
    where X is the letter of your chosen option (A, B, C, ...). No text after that line.
""").strip()

PERSONA_B = textwrap.dedent("""\
    You are a pragmatic problem solver. Use the simplest reasoning that fits the evidence.
    End with exactly one line: Answer: X
    where X is the letter of your chosen option (A, B, C, ...). No text after that line.
""").strip()


def call_with_persona(model: ChatClaudeCode, persona: str, task_prompt: str) -> str:
    response = model.invoke([
        SystemMessage(content=persona),
        HumanMessage(content=task_prompt),
    ])
    return response.content


def _haiku_pair(task_prompt: str) -> dict[str, str]:
    haiku = ChatClaudeCode(model="haiku")
    parallel = RunnableParallel(
        a=RunnableLambda(lambda p: call_with_persona(haiku, PERSONA_A, p)),
        b=RunnableLambda(lambda p: call_with_persona(haiku, PERSONA_B, p)),
    )
    return parallel.invoke(task_prompt)


def _normalize_answer_text(text: str) -> str:
    """Lexical signal for BBH: collapse whitespace on full response."""
    return re.sub(r"\s+", " ", text.strip())


def _answer_label(text: str, task: dict | None = None) -> str | None:
    """Extract a choice label from model text, using task context when available."""
    letter = extract_choice(text)
    if task is None:
        return letter
    valid = set(task.get("choice_labels") or [])
    if letter is not None and letter in valid:
        return letter
    text_label = extract_choice_text(text, task)
    if text_label is not None:
        return text_label
    return letter if letter in valid else None


def lexical_agree(a: str, b: str, task: dict | None = None) -> bool:
    """Same final choice label, or identical normalized text."""
    ca, cb = _answer_label(a, task), _answer_label(b, task)
    if ca is not None and cb is not None:
        return ca == cb
    return _normalize_answer_text(a) == _normalize_answer_text(b)


def judge_agree(a: str, b: str, task: dict, judge: BaseChatModel | None = None) -> bool:
    if judge is None:
        judge = ChatClaudeCode(model="haiku")
    resp = judge.invoke([
        SystemMessage(content=(
            "You compare two answers to the same multiple-choice question. "
            "Reply ONLY with the single word YES or NO. Nothing else."
        )),
        HumanMessage(content=(
            f"Question and choices:\n{task['prompt']}\n\n"
            f"--- Candidate A ---\n{a}\n\n"
            f"--- Candidate B ---\n{b}\n\n"
            "Do A and B select the same choice letter (same meaning)? Answer YES or NO."
        )),
    ])
    return resp.content.strip().upper().startswith("YES")


def arm_haiku_only(task: dict) -> tuple[str, int]:
    haiku = ChatClaudeCode(model="haiku")
    return call_with_persona(haiku, PERSONA_A, task["prompt"]), 1


def arm_sonnet_only(task: dict) -> tuple[str, int]:
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 1


def arm_echo_lexical(task: dict) -> tuple[str, int]:
    pair = _haiku_pair(task["prompt"])
    if lexical_agree(pair["a"], pair["b"], task):
        return pair["a"], 2
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 3


def arm_echo_judge(task: dict) -> tuple[str, int]:
    pair = _haiku_pair(task["prompt"])
    if judge_agree(pair["a"], pair["b"], task):
        return pair["a"], 3
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 4


def arm_echo_small_judge(task: dict) -> tuple[str, int]:
    if not _HAS_OLLAMA or ChatOllama is None:
        raise RuntimeError(
            "echo-small-judge requires langchain-ollama and Ollama with "
            f"{SMALL_JUDGE_MODEL}"
        )
    pair = _haiku_pair(task["prompt"])
    small_judge = ChatOllama(model=SMALL_JUDGE_MODEL, base_url=SMALL_JUDGE_BASE_URL)
    if judge_agree(pair["a"], pair["b"], task, judge=small_judge):
        return pair["a"], 2
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 3


def arm_echo_judge_openai_model(task: dict, model_name: str) -> tuple[str, int]:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "echo-judge-openai requires langchain-openai. "
            "Run: pip install langchain-openai"
        ) from exc
    pair = _haiku_pair(task["prompt"])
    openai_judge = ChatOpenAI(model=model_name, temperature=0)
    if judge_agree(pair["a"], pair["b"], task, judge=openai_judge):
        return pair["a"], 3
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 4


def arm_echo_judge_openai(task: dict) -> tuple[str, int]:
    return arm_echo_judge_openai_model(task, OPENAI_JUDGE_MODELS["gpt-5.5"])


def arm_echo_judge_openai_gpt_5_4(task: dict) -> tuple[str, int]:
    return arm_echo_judge_openai_model(task, OPENAI_JUDGE_MODELS["gpt-5.4"])


def arm_echo_judge_openai_gpt_5_4_mini(task: dict) -> tuple[str, int]:
    return arm_echo_judge_openai_model(task, OPENAI_JUDGE_MODELS["gpt-5.4-mini"])


def arm_echo_judge_openai_gpt_5_4_nano(task: dict) -> tuple[str, int]:
    return arm_echo_judge_openai_model(task, OPENAI_JUDGE_MODELS["gpt-5.4-nano"])


def arm_echo_judge_gemini_model(task: dict, model_name: str) -> tuple[str, int]:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "echo-judge-gemini requires langchain-google-genai. "
            "Run: pip install langchain-google-genai"
        ) from exc
    pair = _haiku_pair(task["prompt"])
    gemini_judge = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    if judge_agree(pair["a"], pair["b"], task, judge=gemini_judge):
        return pair["a"], 3
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 4


def arm_echo_judge_gemini_pro(task: dict) -> tuple[str, int]:
    return arm_echo_judge_gemini_model(task, GEMINI_JUDGE_MODELS["gemini-2.5-pro"])


def arm_echo_judge_gemini_flash(task: dict) -> tuple[str, int]:
    return arm_echo_judge_gemini_model(task, GEMINI_JUDGE_MODELS["gemini-2.5-flash"])


def arm_echo_judge_gemini_flash_lite(task: dict) -> tuple[str, int]:
    return arm_echo_judge_gemini_model(task, GEMINI_JUDGE_MODELS["gemini-2.5-flash-lite"])


def arm_specialist_only(task: dict) -> tuple[str, int]:
    """Domain specialist answers directly; Haiku when no specialist is registered.

    Diagnostic baseline for §10 — without it you cannot tell whether a specialist
    is competent at all, only whether it helps inside a router.

    sub_calls encodes cost (house convention, cf. echo-judge 3-vs-4):
      1 = specialist answered (local, 0 units)
      2 = no specialist for this category, Haiku answered (1 unit)
    """
    specialist = load_specialist(task.get("category"))
    if specialist is None:
        haiku = ChatClaudeCode(model="haiku")
        return call_with_persona(haiku, PERSONA_A, task["prompt"]), 2
    return call_with_persona(specialist, PERSONA_A, task["prompt"]), 1


def arm_echo_specialist(task: dict) -> tuple[str, int]:
    """Echo with a free local specialist as tiebreaker before paying for Sonnet.

    Both Haiku personas agreeing is unchanged, so a weak specialist cannot damage
    the accept path — it is only consulted where the current router would already
    be spending 3.0 units on Sonnet.

    sub_calls encodes cost:
      2 = personas agreed, accepted                       (2.0 units)
      3 = personas disagreed, specialist broke the tie    (2.0 units, specialist free)
      4 = personas disagreed, escalated to Sonnet         (5.0 units)
    """
    pair = _haiku_pair(task["prompt"])
    if lexical_agree(pair["a"], pair["b"], task):
        return pair["a"], 2

    specialist = load_specialist(task.get("category"))
    if specialist is not None:
        verdict = call_with_persona(specialist, PERSONA_A, task["prompt"])
        choice = _answer_label(verdict, task)
        # Only a parseable vote matching one candidate counts as a tiebreak;
        # anything else (unparseable, or a third answer) escalates.
        if choice is not None:
            if choice == _answer_label(pair["a"], task):
                return pair["a"], 3
            if choice == _answer_label(pair["b"], task):
                return pair["b"], 3

    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 4


def arm_echo_oracle(task: dict) -> tuple[str, int]:
    """Escalate only when both cheap answers are wrong vs gold label."""
    pair = _haiku_pair(task["prompt"])
    a_ok, _ = score_bbh(pair["a"], task)
    b_ok, _ = score_bbh(pair["b"], task)
    if a_ok and b_ok:
        return pair["a"], 2
    if a_ok:
        return pair["a"], 2
    if b_ok:
        return pair["b"], 2
    sonnet = ChatClaudeCode(model="sonnet")
    return call_with_persona(sonnet, PERSONA_A, task["prompt"]), 3


BBH_ARMS: dict[str, Callable[[dict], tuple[str, int]]] = {
    "haiku-only": arm_haiku_only,
    "sonnet-only": arm_sonnet_only,
    "echo-lexical": arm_echo_lexical,
    "echo-judge": arm_echo_judge,
    "echo-small-judge": arm_echo_small_judge,
    "echo-judge-openai": arm_echo_judge_openai,
    "echo-judge-openai-gpt-5.4": arm_echo_judge_openai_gpt_5_4,
    "echo-judge-openai-gpt-5.4-mini": arm_echo_judge_openai_gpt_5_4_mini,
    "echo-judge-openai-gpt-5.4-nano": arm_echo_judge_openai_gpt_5_4_nano,
    "echo-judge-gemini-pro": arm_echo_judge_gemini_pro,
    "echo-judge-gemini-flash": arm_echo_judge_gemini_flash,
    "echo-judge-gemini-flash-lite": arm_echo_judge_gemini_flash_lite,
    "specialist-only": arm_specialist_only,
    "echo-specialist": arm_echo_specialist,
    "echo-oracle": arm_echo_oracle,
}
