"""Unit tests for ruler64k/128k prompt splitting (query-agnostic parity).

No network / no model: feeds synthetic ruler-full-format prompts through
`_split_prompt` and asserts the chat template is stripped, the question is
separated out of `context` (so KV compression stays query-agnostic), and
context+question+answer_prefix reconstructs the body exactly.
"""
from __future__ import annotations

import unittest

from eval_harness.benchmarks.ruler64k import _split_prompt, QUESTION_ANCHORS

HEAD = ("<|im_start|>system\nYou are a precise assistant evaluating RULER "
        "long-context prompts.<|im_end|>\n<|im_start|>user\n")
TAIL = "<|im_end|>\n<|im_start|>assistant\n"


def _wrap(context_body: str, q_block: str) -> str:
    return HEAD + context_body + q_block + TAIL


class TestRulerSplit(unittest.TestCase):
    def _check(self, prompt, task, expect_anchor_in_ctx=False):
        res = _split_prompt(prompt, task)
        self.assertIsNotNone(res, f"split returned None for {task}")
        ctx, q, ap = res
        # body = everything between the last user-head and the assistant tail
        body = prompt[prompt.rfind("<|im_start|>user\n") + len("<|im_start|>user\n"):]
        body = body[: body.rfind(TAIL)]
        self.assertEqual(ctx + q + ap, body, f"reconstruction failed for {task}")
        self.assertNotIn("<|im_start|>", ctx)
        self.assertNotIn("<|im_end|>", ctx)
        self.assertIn(QUESTION_ANCHORS[task], q)   # the (last) question lives in `question`
        self.assertNotEqual(ap.strip(), "")        # answer_prefix non-empty
        if expect_anchor_in_ctx:
            self.assertIn(QUESTION_ANCHORS[task], ctx)  # preamble occurrence stayed in context

    def test_niah_number(self):
        ctx = "A special magic number is hidden in the text.\n<haystack...>\n\n"
        qb = ("What is the special magic number for foo-bar mentioned in the provided text? "
              "The special magic number for foo-bar mentioned in the provided text is")
        self._check(_wrap(ctx, qb), "niah_single_2")

    def test_niah_uuid(self):
        ctx = "<haystack...>\n\n"
        qb = ("What is the special magic uuid for ab-cd mentioned in the provided text? "
              "The special magic uuid for ab-cd mentioned in the provided text is")
        self._check(_wrap(ctx, qb), "niah_single_3")

    def test_niah_multiquery(self):
        ctx = "<haystack...>\n"
        qb = ("What are all the special magic numbers for a, b, and c mentioned in the provided text? "
              "The special magic numbers for a, b, and c mentioned in the provided text are")
        self._check(_wrap(ctx, qb), "niah_multiquery")

    def test_qa_double_anchor_uses_last(self):
        # qa repeats the instruction as a preamble (must stay in context).
        preamble = "Answer the question based on the given documents. (preamble)\n\n<docs...>\n\n"
        qb = ("Answer the question based on the given documents. Only give me the answer.\n\n"
              "Question: In what country is Normandy located? Answer:")
        self._check(_wrap(preamble, qb), "qa_1", expect_anchor_in_ctx=True)

    def test_vt_no_questionmark(self):
        ctx = "<assignments...>\n\n"
        qb = ("Question: Find all variables that are assigned the value 15311 in the text above."
              "Answer: According to the chain(s) of variable assignment, they are:")
        self._check(_wrap(ctx, qb), "vt")

    def test_cwe_double_anchor(self):
        preamble = "Question: What are the 10 most common words ... (preamble)\n\n<word list>\n\n"
        qb = ("Question: What are the 10 most common words in the above list? "
              "Answer: The top 10 words that appear most often in the list are:")
        self._check(_wrap(preamble, qb), "cwe", expect_anchor_in_ctx=True)

    def test_fwe(self):
        ctx = "<coded text ...>\n\n"
        qb = ("Question: Do not provide any explanation. What are the three most frequent words? "
              "Answer: According to the coded text above, the three most frequently appeared words are:")
        self._check(_wrap(ctx, qb), "fwe")

    def test_unknown_task_returns_none(self):
        self.assertIsNone(_split_prompt(_wrap("ctx\n\n", "What is the special magic number for x? y"),
                                        "some_future_task"))


if __name__ == "__main__":
    unittest.main()
