"""Data windows: tokenisation, seeded selection, suffix modes, disjointness (CPU, stub tokenizer)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from eval_harness.kv_recovery.config import DataCfg, RecoveryConfig
from eval_harness.kv_recovery.data import (
    assert_disjoint,
    bos_id_for,
    build_examples,
    load_split,
    read_jsonl,
    split_window,
)


class _WordTokenizer:
    """One token per word (ids = hash of the word); deterministic and dependency-free."""
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [2 + (hash(w) % 1000) for w in text.split()]


def _rows(n, words_per_row):
    return [{"id": f"r{i}", "text": " ".join(f"w{i}_{j}" for j in range(words_per_row))} for i in range(n)]


class TestWindows(unittest.TestCase):
    def test_split_modes(self):
        ids = list(range(100))
        import random
        ctx, suf = split_window(ids, 40, 8, suffix_mode="continuation", rng=random.Random(0))
        self.assertEqual(ctx, ids[:32]); self.assertEqual(suf, ids[32:40])
        ctx2, suf2 = split_window(ids, 40, 8, suffix_mode="recall", rng=random.Random(0))
        self.assertEqual(ctx2, ids[:32]); self.assertEqual(len(suf2), 8)
        self.assertTrue(all(x < 32 for x in suf2))          # a span copied from the context
        with self.assertRaises(ValueError):
            split_window(ids[:10], 40, 8, suffix_mode="continuation", rng=random.Random(0))

    def test_build_examples_bos_shapes_and_skips(self):
        tok = _WordTokenizer()
        dcfg = DataCfg(max_length=16, suffix_length=4, num_train_examples=3)
        rows = _rows(4, 30) + [{"id": "short", "text": "only a few words"}, {"id": "empty", "text": "  "}]
        ex, stats = build_examples(rows, tok, dcfg, n_examples=3, seed=0)
        self.assertEqual(len(ex), 3)
        for e in ex:
            self.assertEqual(tuple(e.ctx_ids.shape), (1, 12))
            self.assertEqual(tuple(e.suffix_ids.shape), (1, 4))
            self.assertEqual(int(e.ctx_ids[0, 0]), 1)       # bos prepended
        self.assertEqual(stats.n_used, 3)
        # seeded selection order is reproducible and seed-dependent
        ex2, _ = build_examples(rows, tok, dcfg, n_examples=3, seed=0)
        self.assertEqual([e.id for e in ex], [e.id for e in ex2])
        ex3, _ = build_examples(rows, tok, dcfg, n_examples=4, seed=1)
        self.assertEqual(len(ex3), 4)
        with self.assertRaises(ValueError):
            build_examples(rows, tok, dcfg, n_examples=5, seed=0)   # only 4 long enough

    def test_bos_fallbacks(self):
        class NoBos:
            bos_token_id = None
        self.assertIsNone(bos_id_for(NoBos()))
        import types
        model = types.SimpleNamespace(generation_config=types.SimpleNamespace(bos_token_id=7))
        self.assertEqual(bos_id_for(NoBos(), model), 7)
        self.assertEqual(bos_id_for(_WordTokenizer()), 1)

    def test_load_split_and_disjointness(self):
        tok = _WordTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp) / "train.jsonl"; va = Path(tmp) / "val.jsonl"
            tr.write_text("\n".join(json.dumps(r) for r in _rows(3, 40)) + "\n")
            va.write_text("\n".join(json.dumps(r) for r in [{"id": "v0", "text": " ".join(f"x{j}" for j in range(40))}]) + "\n")
            cfg = RecoveryConfig.from_dict({"data": {"path": str(tr), "val_path": str(va), "max_length": 16, "suffix_length": 4,
                                                      "num_train_examples": 2, "num_val_examples": 1}})
            train, _ = load_split(cfg, tok, "train")
            val, _ = load_split(cfg, tok, "val")
            self.assertEqual(len(train), 2); self.assertEqual(len(val), 1)
            assert_disjoint(train, val)
            with self.assertRaises(AssertionError):
                assert_disjoint(train, train)
            self.assertEqual(len(read_jsonl(tr)), 3)


if __name__ == "__main__":
    unittest.main()
