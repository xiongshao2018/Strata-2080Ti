"""pad_prefix.py 的回归测试；需要本机 pack tokenizer 与 chat template。

    STRATA_TOKENIZER=/path/to/tokenizer python -m unittest tools.test_pad_prefix -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import pad_prefix as PP  # noqa: E402

TOKENIZER_DIR = Path(os.environ.get(
    "STRATA_TOKENIZER", "/home/lingdeng/AI/models/Strata-SC117/sc117_pack/tokenizer"))
TEMPLATE = TOKENIZER_DIR / "chat_template.jinja"
if not TEMPLATE.exists():
    TEMPLATE = ROOT / "serve" / "chat_template.jinja"


@unittest.skipUnless((TOKENIZER_DIR / "vocab.json").is_file(), "需要 pack tokenizer/")
class PadPrefixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = PP.Tokenizer(TOKENIZER_DIR)
        cls.renderer = PP.Renderer(TEMPLATE)

    def request(self, prefix_tokens: int, placeholder: bool = False) -> dict:
        req = {"model": "qwen3.8-flash-next-iq3_s",
               "messages": [{"role": "system", "content": ""},
                            {"role": "user", "content": "hello"}],
               "tools": [{"type": "function", "function": {
                   "name": "f", "description": "d",
                   "parameters": {"type": "object", "properties": {}}}}]}
        system = req["messages"][0]
        if placeholder:
            system["content"] = PP.DEFAULT_PLACEHOLDER
        base = self.tokenizer.count(self.renderer.fixed_text(req))
        head = system["content"] + ("\n" if system["content"] else "") + \
            PP.PAD_UNIT * max(0, prefix_tokens - base)
        for _ in range(24):
            system["content"] = head
            actual = self.tokenizer.count(self.renderer.fixed_text(req))
            if actual == prefix_tokens:
                break
            delta = prefix_tokens - actual
            if delta > 0:
                head += PP.PAD_UNIT * delta
            else:
                head = head[:max(0, len(head) + delta * 2)]
        return req

    def align(self, req, **kw):
        return PP.align_request(req, tokenizer=self.tokenizer, renderer=self.renderer, **kw)

    def test_30022_to_32768_and_json_valid(self):
        _, report = self.align(self.request(30022))
        self.assertEqual(report["final_tokens"], 32768)
        self.assertEqual(report["inserted_tokens"], 2746)
        self.assertEqual(report["status"], "padded_to_32768")

    def test_idempotent(self):
        req, _ = self.align(self.request(30022))
        _, report = self.align(req)
        self.assertEqual(report["inserted_tokens"], 0)
        self.assertEqual(report["final_tokens"], 32768)

    def test_overflow_to_40960(self):
        _, report = self.align(self.request(32769))
        self.assertEqual(report["final_tokens"], 40960)
        self.assertTrue(report["alarm"])

    def test_exact_boundary_untouched(self):
        _, report = self.align(self.request(32768))
        self.assertEqual(report["inserted_tokens"], 0)

    def test_below_window_untouched(self):
        _, report = self.align(self.request(1000))
        self.assertEqual(report["status"], "below_window")
        self.assertEqual(report["inserted_tokens"], 0)

    def test_force(self):
        _, report = self.align(self.request(1000), force=True)
        self.assertEqual(report["final_tokens"], 32768)

    def test_placeholder_and_json(self):
        req, report = self.align(self.request(30022, True), placeholder=PP.DEFAULT_PLACEHOLDER)
        self.assertEqual(report["insert_mode"], "replace")
        self.assertNotIn(PP.DEFAULT_PLACEHOLDER, req["messages"][0]["content"])
        self.assertIsInstance(json.loads(json.dumps(req)), dict)

    def test_responses_api_instructions(self):
        base = {"model": "m", "instructions": "", "input": "hello", "max_output_tokens": 4}
        head = PP.PAD_UNIT * 30022
        for _ in range(24):
            candidate = dict(base, instructions=head)
            actual = self.tokenizer.count(self.renderer.fixed_text(candidate))
            if actual == 30022:
                break
            delta = 30022 - actual
            if delta > 0:
                head += PP.PAD_UNIT * delta
            else:
                head = head[:max(0, len(head) + delta * 2)]
        req, report = self.align(dict(base, instructions=head))
        self.assertEqual(report["final_tokens"], 32768)
        self.assertIn("STRATA_PREFIX_PAD", req["instructions"])


if __name__ == "__main__":
    unittest.main()
