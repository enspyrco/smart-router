"""LangChain ``BaseChatModel`` backed by the Anthropic API over direct-Bearer OAuth.

WHY THIS EXISTS: ``ChatClaudeCode`` shells out to ``claude --print``, which is an
AGENT WITH TOOLS. Probed 2026-08-10 against the harness's exact invocation --
``claude --print --setting-sources "" --model <alias>`` -- with an unguessable
probe (return the contents of a file holding fresh random bytes; nothing but file
access explains returning it):

    haiku    TOOLS ACTIVE — read the secret
    sonnet   TOOLS ACTIVE — read the secret

``--setting-sources ""`` strips project CLAUDE.md, hooks and task-restore
injection (the #1305 contamination) but does NOT remove tools. So every benchmark
row produced through that path was answered by a model that could execute code
and read files.

WHY THAT MATTERS SPECIFICALLY FOR ECHO. Tool use is opportunistic, not uniform.
Two calls to the same model on the same task can differ in whether a tool was
reached for, so the agreement signal picks up tool-luck as well as task
difficulty. Echo escalates on disagreement; a disagreement caused by "run A shelled
out and run B reasoned" is noise in the dependent variable.

WHY NOT JUST PASS ``--disallowed-tools``. It is a mitigation, not a fix: it blocked
3/3 on haiku here but leaked 2/3 on a different model in sibling work, and a
behavioural probe can only prove tools ARE reachable -- "blocked in N trials" is
absence of evidence.

THIS MODULE IS STRUCTURALLY TOOL-FREE. A raw HTTPS completions endpoint has no
harness that could execute anything.

NOT A SEMANTIC DROP-IN -- READ THIS BEFORE SWAPPING (cage-match #6, Maxwell/Wu/Tesla).
``ChatClaudeCode`` never sends temperature and imposes no caller-side token cap;
this transport sends both. On a measurement whose dependent variable IS
run-to-run agreement, temperature is not a detail, it is the independent
variable. So:
  * ``temperature`` and ``max_tokens`` are EXPLICIT fields with documented
    defaults, and
  * ``generation_config()`` exposes the resolved settings so callers can RECORD
    them in their artifacts. A result that does not state its temperature is not
    reproducible.

COST IS UNCHANGED. Direct-Bearer OAuth bills the Max subscription exactly like
the CLI. Zero marginal API spend, and it drops the ~3s CLI startup per call.
Credential: ``CLAUDE_CODE_OAUTH_TOKEN`` (or ``CLAUDE_OAUTH_TOKEN``). Never
``ANTHROPIC_API_KEY`` -- that is metered.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any, List, Literal, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

ENDPOINT = "https://api.anthropic.com/v1/messages"

# Aliases kept identical to ChatClaudeCode so this is a drop-in swap at the call site.
#
# PINNING IS PARTIAL, AND CLAIMING OTHERWISE WAS THE BUG. Only `haiku` resolves to
# a dated snapshot; `sonnet` and `opus` are FLOATING and can silently resolve to a
# different model between runs. An earlier revision of this comment asserted
# "every alias is PINNED" — precisely the replicate-this-run overclaim Wu killed
# once already, restated one line below the fix (Tesla, cage-match #6 round 2).
# The mitigation that actually works is downstream: generation_config() records
# the RESOLVED model_id into every artifact, so a mid-cohort swap is at least
# visible after the fact. Pin the dated ids here as soon as they are published.
ModelAlias = Literal["haiku", "sonnet", "opus"]

MODEL_IDS: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
}

# 429 on this path is a SHARED Max-plan window: transient, and it carries
# x-should-retry: true. Raising immediately converts a rate limit into MISSING
# DATA in any caller that catches exceptions per-task -- and 429s cluster in
# time, so the dropped set is not random. Retry with jittered backoff instead.
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 5


def _token() -> str:
    tok = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("CLAUDE_OAUTH_TOKEN")
    if not tok:
        raise RuntimeError(
            "No OAuth token. Run `claude setup-token`, put it in ~/.claude/.env as "
            "CLAUDE_CODE_OAUTH_TOKEN, and `source ~/.claude/.env` before running. "
            "Do NOT fall back to ANTHROPIC_API_KEY — that is metered."
        )
    return tok


class ChatOAuthError(RuntimeError):
    """Any transport failure, so callers can distinguish it from a scoring error."""


class ChatOAuth(BaseChatModel):
    """Chat model over the raw Anthropic endpoint. Structurally tool-free."""

    model: ModelAlias = Field(default="sonnet", description="haiku | sonnet | opus")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=1.0)
    timeout: int = Field(default=300)

    @property
    def _llm_type(self) -> str:
        return "chat-oauth"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return self.generation_config()

    def generation_config(self) -> dict[str, Any]:
        """The resolved settings a caller MUST record alongside any result.

        r (agreement rate) is temperature-dependent, so a measured escalation rate
        is meaningless without the temperature that produced it. Exposing this makes
        the artifact self-describing rather than pinned to a config nobody wrote down.
        """
        return {
            "endpoint": ENDPOINT,
            "model_alias": self.model,
            "model_id": MODEL_IDS.get(self.model, self.model),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _split(self, messages: List[BaseMessage]) -> tuple[Optional[str], list[dict]]:
        """Anthropic takes the system prompt as a top-level field, not a message."""
        system, turns = None, []
        for m in messages:
            if m.type == "system":
                system = m.content if system is None else f"{system}\n\n{m.content}"
            elif m.type == "ai":
                turns.append({"role": "assistant", "content": m.content})
            elif m.type == "human":
                turns.append({"role": "user", "content": m.content})
            else:
                # Previously every non-system, non-ai message silently became a
                # user turn. Fine for this harness's System+Human calls, a
                # semantic collapse if reused generally (Tesla + Wu).
                raise ChatOAuthError(
                    f"unsupported message type {m.type!r}; ChatOAuth handles "
                    "system / human / ai only"
                )
        if not turns:
            # An empty content block is a 400 from the API; surface it as the
            # programming error it is rather than a generic transport failure.
            raise ChatOAuthError("no human/ai turns — a system prompt alone is not a request")
        return system, turns

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        req_headers = {
            "Authorization": f"Bearer {_token()}",
            "anthropic-beta": "oauth-2025-04-20",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            req = urllib.request.Request(
                ENDPOINT, data=json.dumps(body).encode(), headers=req_headers
            )
            try:
                return json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:300].decode(errors="replace")
                last = ChatOAuthError(f"anthropic {exc.code} (model={self.model}): {detail}")
                if exc.code not in RETRY_STATUSES or attempt == MAX_RETRIES - 1:
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                # Previously only HTTPError was wrapped, so timeouts and DNS
                # failures escaped with a different type and no model context.
                last = ChatOAuthError(f"anthropic transport error (model={self.model}): {exc!r}")
                if attempt == MAX_RETRIES - 1:
                    raise last from exc
            time.sleep(min(2 ** attempt, 30) + random.random())
        raise last or ChatOAuthError("unreachable")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, turns = self._split(messages)
        body: dict[str, Any] = {
            "model": MODEL_IDS.get(self.model, self.model),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": turns,
        }
        if system:
            body["system"] = system
        if stop:
            # Previously accepted and silently discarded — a BaseChatModel
            # contract violation that fails quietly rather than loudly.
            body["stop_sequences"] = list(stop)

        payload = self._post(body)

        parts = [b.get("text", "") for b in payload.get("content", [])
                 if b.get("type") == "text"]
        text = "".join(parts).rstrip()
        if not text:
            # A 200 with no text block used to return a cheerful empty ChatResult,
            # which the scorer then read as an abstention — a transport failure
            # laundered into missing data (Tesla + Wu).
            raise ChatOAuthError(
                f"empty completion (model={self.model}, "
                f"stop_reason={payload.get('stop_reason')!r}) — "
                "possible max_tokens truncation before any text was emitted"
            )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
