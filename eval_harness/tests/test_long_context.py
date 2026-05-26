import unittest

from eval_harness.long_context import (
    CompressionBudget,
    LongContextCompressionConfig,
    compress_token_ids,
    merge_budgeted_indices,
    select_topk_indices_from_scores,
)


class LongContextTests(unittest.TestCase):
    def test_select_topk_indices_from_scores(self):
        candidates = [10, 11, 12, 13]
        scores = [0.1, 0.9, 0.4, 0.8]
        picked = select_topk_indices_from_scores(candidates, scores, top_k=2)
        self.assertEqual(picked, [11, 13])

    def test_merge_budgeted_indices_respects_budget(self):
        budget = CompressionBudget(
            sink_indices=[0, 1],
            local_indices=[18, 19],
            candidate_indices=list(range(2, 18)),
            top_k_budget=2,
        )
        kept = merge_budgeted_indices(
            token_count=20,
            budget=budget,
            topk_indices=[6, 12],
            span_tokens=5,
        )
        self.assertLessEqual(len(kept), 6)
        self.assertIn(0, kept)
        self.assertIn(1, kept)
        self.assertIn(18, kept)
        self.assertIn(19, kept)

    def test_proxy_compression_keeps_within_context_len(self):
        cfg = LongContextCompressionConfig(
            enabled=True,
            max_context_len=8,
            sink_tokens=2,
            local_tokens=3,
            top_k_tokens=3,
            span_tokens=0,
        )
        token_ids = list(range(20))
        out = compress_token_ids(token_ids, cfg)
        self.assertTrue(out.was_compressed)
        self.assertLessEqual(out.compressed_length, 8)
        self.assertEqual(out.token_ids[:2], [0, 1])
        self.assertEqual(out.token_ids[-3:], [17, 18, 19])


if __name__ == "__main__":
    unittest.main()
