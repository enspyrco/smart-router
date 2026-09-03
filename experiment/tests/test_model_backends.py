import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_backends import OllamaBackend


class _Response:
    def __init__(self, body):
        self._body = BytesIO(json.dumps(body).encode())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body.read()


class TestOllamaBackend(unittest.TestCase):
    @patch("model_backends.urllib.request.urlopen")
    def test_disables_thinking_by_default(self, urlopen):
        urlopen.return_value = _Response({
            "message": {"content": "Answer: A"},
            "prompt_eval_count": 10,
            "eval_count": 3,
        })

        reply = OllamaBackend("qwen3.5:4b").chat("system", "question")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertIs(payload["think"], False)
        self.assertEqual(reply.text, "Answer: A")
        self.assertEqual(reply.tokens, 13)

    @patch("model_backends.urllib.request.urlopen")
    def test_thinking_can_be_enabled_explicitly(self, urlopen):
        urlopen.return_value = _Response({
            "message": {"content": "Answer: B"},
            "prompt_eval_count": 1,
            "eval_count": 1,
        })

        OllamaBackend("qwen3.5:4b", think=True).chat("system", "question")

        request = urlopen.call_args.args[0]
        self.assertIs(json.loads(request.data)["think"], True)


if __name__ == "__main__":
    unittest.main()
