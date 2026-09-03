"""Model backends for harness evaluation.

The harness experiment holds the model frozen and varies the prompt, so the
backend only has to do one thing: take a system + user message and return text
plus a token count. Keeping that behind one interface means the evaluator is
not welded to whichever server happens to be running.

Backends:
  ollama       HTTP to a local/remote Ollama. Zero extra dependencies.
  openai       Any OpenAI-compatible endpoint (vLLM, llama.cpp server, LM Studio).
  transformers Local torch. Correct but slow on CPU — prototyping only.

Token counts are reported when the backend supplies them and estimated
otherwise; ``Reply.tokens_estimated`` records which, so a latency/cost table
never silently mixes the two.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_OLLAMA_URL = os.environ.get("HARNESS_OLLAMA_URL", "http://localhost:11434")
DEFAULT_OPENAI_URL = os.environ.get("HARNESS_OPENAI_URL", "http://localhost:8000/v1")
REQUEST_TIMEOUT = int(os.environ.get("HARNESS_TIMEOUT", "600"))


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency: float
    tokens_estimated: bool = False

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _estimate_tokens(text: str) -> int:
    """~4 chars/token. Only used when the backend reports nothing."""
    return max(1, len(text) // 4)


class Backend:
    name = "base"

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def chat(self, system: str, user: str) -> Reply:  # pragma: no cover
        raise NotImplementedError


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: str, temperature: float = 0.0, base_url: str | None = None,
                 think: bool = False):
        super().__init__(model, temperature)
        self.base_url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        # Reasoning models can spend the entire generation budget in the
        # hidden thinking channel and return an empty answer. Harness runs
        # measure the visible answer, so thinking is opt-in and frozen across
        # every question in a run.
        self.think = think

    def chat(self, system: str, user: str) -> Reply:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama unreachable at {self.base_url}: {exc}. "
                f"Start it, or pull the model: ollama pull {self.model}"
            ) from exc
        latency = time.perf_counter() - t0
        text = body.get("message", {}).get("content", "")
        # Ollama reports real counts; absent only on very old versions.
        prompt_tok = body.get("prompt_eval_count")
        completion_tok = body.get("eval_count")
        if prompt_tok is None or completion_tok is None:
            return Reply(text, _estimate_tokens(system + user), _estimate_tokens(text),
                         latency, tokens_estimated=True)
        return Reply(text, int(prompt_tok), int(completion_tok), latency)


class OpenAIBackend(Backend):
    """Any OpenAI-compatible server. Also reaches real OpenAI if pointed there."""

    name = "openai"

    def __init__(self, model: str, temperature: float = 0.0, base_url: str | None = None):
        super().__init__(model, temperature)
        self.base_url = base_url or DEFAULT_OPENAI_URL
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package not installed: pip install openai") from exc
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-local"),
            timeout=REQUEST_TIMEOUT,
        )

    def chat(self, system: str, user: str) -> Reply:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        latency = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        if usage is None:
            return Reply(text, _estimate_tokens(system + user), _estimate_tokens(text),
                         latency, tokens_estimated=True)
        return Reply(text, usage.prompt_tokens, usage.completion_tokens, latency)


def _free_ram_gb() -> float | None:
    """Physical RAM currently free, or None if it cannot be determined."""
    try:  # Linux / most POSIX
        return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 1e9
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / 1e9
    except Exception:
        return None


class TransformersBackend(Backend):
    """Local torch. Prototyping only — on a CPU expect a few tokens/sec.

    Memory is the binding constraint, not speed. ``from_pretrained`` defaults to
    float32 on CPU, which doubles the weight footprint for no accuracy benefit
    at inference, so this pins bfloat16 and refuses to start if the machine
    clearly cannot hold the model. Swapping a language model to disk does not
    "run slowly" — it makes the whole machine unusable.

    For anything beyond a smoke test, use a quantised model through the Ollama
    backend instead: 4-bit weights are roughly a quarter the size and llama.cpp
    is far better at CPU inference than torch.
    """

    name = "transformers"
    # Activations, KV cache, tokenizer and framework overhead on top of weights.
    OVERHEAD_GB = 1.5

    def __init__(self, model: str, temperature: float = 0.0, max_new_tokens: int = 1024,
                 dtype: str = "bfloat16", skip_memory_check: bool = False):
        super().__init__(model, temperature)
        self.max_new_tokens = max_new_tokens
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers/torch not installed") from exc

        if not skip_memory_check:
            self._preflight(model)

        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=getattr(torch, dtype),
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    def _preflight(self, model: str) -> None:
        """Refuse to load a model that plainly will not fit in free RAM."""
        free = _free_ram_gb()
        if free is None:
            return
        try:
            from huggingface_hub import model_info
            total = sum(s.size or 0 for s in model_info(model, files_metadata=True).siblings
                        if s.rfilename.endswith(".safetensors"))
        except Exception:
            return
        if not total:
            return
        # Repo files are usually bf16 already; assume the on-disk size is the
        # in-memory weight size and add headroom for everything else.
        need = total / 1e9 + self.OVERHEAD_GB
        if need > free:
            raise RuntimeError(
                f"{model} needs about {need:.1f} GB but only {free:.1f} GB of RAM is free.\n"
                f"Loading it would swap to disk and stall the machine.\n"
                f"Options: close other applications; use a smaller model; or run a\n"
                f"quantised build through Ollama (--backend ollama), which needs "
                f"roughly a quarter of this.\n"
                f"Override with skip_memory_check=True only if you know the figure is wrong."
            )

    def chat(self, system: str, user: str) -> Reply:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        t0 = time.perf_counter()
        out = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            pad_token_id=self._tok.eos_token_id,
        )
        latency = time.perf_counter() - t0
        generated = out[0][inputs["input_ids"].shape[1]:]
        text = self._tok.decode(generated, skip_special_tokens=True)
        return Reply(text, int(inputs["input_ids"].shape[1]), int(generated.shape[0]), latency)


BACKENDS = {
    "ollama": OllamaBackend,
    "openai": OpenAIBackend,
    "transformers": TransformersBackend,
}


def build_backend(kind: str, model: str, temperature: float = 0.0, **kwargs) -> Backend:
    if kind not in BACKENDS:
        raise SystemExit(f"unknown backend {kind!r}; available: {', '.join(BACKENDS)}")
    return BACKENDS[kind](model, temperature=temperature, **kwargs)
