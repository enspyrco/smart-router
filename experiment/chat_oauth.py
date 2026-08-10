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
out and run B reasoned" is noise in the dependent variable, not signal about
whether the task is hard. It also inflates per-tier accuracy and can distort the
haiku/sonnet gap unevenly, which is exactly what the per-category cheap/expensive
table is trying to measure.

WHY NOT JUST PASS ``--disallowed-tools``. It is a mitigation, not a fix. On haiku
it blocked 3/3 trials here, but the same flag leaked 2/3 on a different model in
sibling work (see ~/git/research/eval-hygiene). More fundamentally a behavioural
probe can only prove tools ARE reachable; "blocked in N trials" is absence of
evidence. Do not build a measurement on it.

THIS MODULE IS STRUCTURALLY TOOL-FREE. A raw HTTPS completions endpoint has no
harness that could execute anything -- the isolation is a property of the
transport, not of a flag someone has to remember.

COST IS UNCHANGED. Direct-Bearer OAuth bills the Max subscription, exactly like
the CLI: pass the ``claude setup-token`` credential as ``Authorization: Bearer``
plus ``anthropic-beta: oauth-2025-04-20``. Zero marginal API spend. It is also
markedly faster, since it drops the ~3s CLI startup per call.

Credential: ``CLAUDE_CODE_OAUTH_TOKEN`` (or ``CLAUDE_OAUTH_TOKEN``) from
``~/.claude/.env``. Never ``ANTHROPIC_API_KEY`` -- that is metered.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

ENDPOINT = "https://api.anthropic.com/v1/messages"

# Aliases kept identical to ChatClaudeCode so this is a drop-in swap.
MODEL_IDS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
}


def _token() -> str:
    tok = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("CLAUDE_OAUTH_TOKEN")
    if not tok:
        raise RuntimeError(
            "No OAuth token. Run `claude setup-token`, put it in ~/.claude/.env as "
            "CLAUDE_CODE_OAUTH_TOKEN, and `source ~/.claude/.env` before running. "
            "Do NOT fall back to ANTHROPIC_API_KEY — that is metered."
        )
    return tok


class ChatOAuth(BaseChatModel):
    """Chat model over the raw Anthropic endpoint. Structurally tool-free."""

    model: str = Field(default="sonnet", description="Alias: haiku | sonnet | opus")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=1.0)
    timeout: int = Field(default=300)

    @property
    def _llm_type(self) -> str:
        return "chat-oauth"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "endpoint": ENDPOINT}

    def _split(self, messages: List[BaseMessage]) -> tuple[Optional[str], list[dict]]:
        """Anthropic takes the system prompt as a top-level field, not a message."""
        system, turns = None, []
        for m in messages:
            if m.type == "system":
                system = m.content if system is None else f"{system}\n\n{m.content}"
            else:
                turns.append({"role": "assistant" if m.type == "ai" else "user",
                              "content": m.content})
        if not turns:
            turns = [{"role": "user", "content": ""}]
        return system, turns

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

        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {_token()}",
                "anthropic-beta": "oauth-2025-04-20",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        try:
            payload = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode(errors="replace")
            # 429 here is a SHARED Max-plan window, not a broken credential or a
            # permanently unavailable path. It carries x-should-retry: true and
            # clears on its own. Do not re-architect around it; retry later.
            raise RuntimeError(
                f"anthropic {exc.code} (model={self.model}): {detail}"
            ) from exc

        parts = [b.get("text", "") for b in payload.get("content", [])
                 if b.get("type") == "text"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="".join(parts).rstrip()))])
