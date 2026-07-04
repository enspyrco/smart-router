#!/usr/bin/env python3
"""Live routing-theatre server for the Echo demo.

Serves the theatre page and exposes a genuinely LIVE route endpoint that shells
out to local Ollama: a 0.5B classifier names the domain, then the 7B specialist
answers. Temperature 0, so the live answer reproduces the logged sweep answer.

Run:  uv run python scripts/demo_server.py   ->  http://localhost:8765
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.bbh import extract_choice, format_prompt
from benchmarks.hyperspecialist import (
    CLASSIFIER_MODEL, GENERAL_MODEL, SPECIALISTS, classify_type, pick_model,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from benchmarks.hyperspecialist import ANSWER_PERSONA, BASE_URL
from chat_claude_code import ChatClaudeCode

PORT = 8765
DEMO_DIR = Path(__file__).resolve().parent.parent.parent / "demo"
DATA = json.loads((DEMO_DIR / "demo_data.json").read_text())
BY_ID = {q["task_id"]: q for q in DATA["questions"]}


def _answer(tag: str, prompt: str) -> tuple[str, str]:
    llm = ChatOllama(model=tag, base_url=BASE_URL, temperature=0.0, num_predict=1536)
    raw = llm.invoke(
        [SystemMessage(content=ANSWER_PERSONA), HumanMessage(content=prompt)]
    ).content
    return extract_choice(raw) or "?", raw


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (DEMO_DIR / "unified.html").read_bytes(), "text/html")
        elif self.path == "/theatre":
            self._send(200, (DEMO_DIR / "theatre.html").read_bytes(), "text/html")
        elif self.path == "/data":
            # Merge precomputed cloud-Haiku answers (written live by precompute_haiku.py)
            hpath = DEMO_DIR / "haiku_answers.json"
            haiku = json.loads(hpath.read_text()) if hpath.exists() else {}
            payload = {"questions": [{**q, "haiku": haiku.get(q["task_id"])}
                                     for q in DATA["questions"]]}
            self._send(200, json.dumps(payload))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/warmup":
            for tag in [CLASSIFIER_MODEL, GENERAL_MODEL, *SPECIALISTS.values()]:
                try:
                    ChatOllama(model=tag, base_url=BASE_URL, num_predict=1).invoke(
                        [HumanMessage(content="hi")]
                    )
                except Exception:
                    pass
            self._send(200, json.dumps({"ok": True}))
            return
        if self.path == "/classify":
            q = BY_ID.get(req.get("task_id"))
            if not q:
                self._send(404, json.dumps({"error": "unknown task_id"}))
                return
            domain = classify_type(q["question"])  # live 0.5B call, ~1-2s
            role, tag = pick_model(domain)
            self._send(200, json.dumps({"domain": domain, "role": role, "model": tag}))
            return
        if self.path == "/haiku":
            q = BY_ID.get(req.get("task_id"))
            if not q:
                self._send(404, json.dumps({"error": "unknown task_id"}))
                return
            prompt = format_prompt(q["question"], {"label": q["labels"], "text": q["options"]})
            import time as _t
            t0 = _t.perf_counter()
            try:
                raw = ChatClaudeCode(model="haiku").invoke(
                    [SystemMessage(content=ANSWER_PERSONA), HumanMessage(content=prompt)]
                ).content
                letter = extract_choice(raw) or "?"
            except Exception as e:
                letter, raw = "?", f"err: {str(e)[:120]}"
            self._send(200, json.dumps({
                "answer": letter, "correct": letter == q["gold"],
                "secs": round(_t.perf_counter() - t0, 1),
            }))
            return
        if self.path == "/answer":
            q = BY_ID.get(req.get("task_id"))
            if not q:
                self._send(404, json.dumps({"error": "unknown task_id"}))
                return
            tag = req.get("model") or GENERAL_MODEL
            prompt = format_prompt(q["question"], {"label": q["labels"], "text": q["options"]})
            letter, raw = _answer(tag, prompt)  # live 7B specialist call, ~10-15s
            self._send(200, json.dumps({
                "answer": letter, "correct": letter == q["gold"], "raw_tail": raw[-320:],
            }))
            return
        self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    print(f"Echo routing theatre -> http://localhost:{PORT}  ({len(BY_ID)} questions)")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
