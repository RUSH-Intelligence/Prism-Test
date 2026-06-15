"""Tests for KVzipSketch — port of kvpress KVzipPress (kvzip_press.py).

No model loading, no network: the tokenizer is injected via the constructor
(char-level stub) and models are 4-level fakes. The kvpress ``score_kvzip`` and
``compress_post`` math is transcribed in-test as reference oracles (bitwise for
the tensor-op transcription, explicit loops as an independent secondary check),
and the globally patched sdpa entry of ``ALL_ATTENTION_FUNCTIONS`` is pinned
end-to-end against masked-key-pruned attention.
"""

import logging
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

import eval_harness.kv_compression.compressors.kvzip_sketch as kvzip_module
from eval_harness.kv_compression.cache_adapter import StandardCacheAdapter
from eval_harness.kv_compression.compressors.kvzip_sketch import KVzipSketch, _get_prerope_query_states
from eval_harness.kv_compression.registry import available_kv_compressors, get_kv_compressor, get_kv_compressor_class

logging.getLogger("eval_harness.kv_compression.compressors.kvzip_sketch").setLevel(logging.ERROR)


def _rope_pos_emb(positions, dim, base=10000.0):
    """Real RoPE trig of shape [1, S, D] (transformers 5.x kwargs convention)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    angles = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


class _StubTokenizer:
    """Deterministic char-level tokenizer: one token per character, id = ord(char)."""

    def __init__(self, chat_template=None):
        self.chat_template = chat_template

    def encode(self, text, return_tensors="pt", add_special_tokens=False):
        if not text:
            return torch.zeros(1, 0, dtype=torch.long)
        return torch.tensor([[ord(c) for c in text]], dtype=torch.long)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, enable_thinking=False):
        return "<PRE>" + messages[0]["content"] + "<SUF>"


class _FakeAttnModule(nn.Module):
    def __init__(
        self,
        layer_idx=0,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        hidden_size=None,
        attn_implementation="sdpa",
        seed=0,
        with_o_proj=False,
        with_kv_proj=False,
    ):
        super().__init__()
        hidden_size = hidden_size or num_heads * head_dim
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
            _attn_implementation=attn_implementation,
        )
        torch.manual_seed(seed)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        if with_o_proj:
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        if with_kv_proj:
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)

    def forward(self, hidden_states=None, **kwargs):
        return (hidden_states, None)


class _FakeDecoderLayer(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


class _FakeInnerModel(nn.Module):
    """Minimal decoder stack: embeds ids, updates a real DynamicCache per layer,
    then calls each self_attn with hook-compatible kwargs."""

    def __init__(self, n_layers=2, num_heads=2, num_kv_heads=2, head_dim=4, hidden_size=8, vocab=1200, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, hidden_size)
        self.layers = nn.ModuleList(
            _FakeDecoderLayer(
                _FakeAttnModule(
                    layer_idx=i,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    hidden_size=hidden_size,
                    seed=seed + i,
                    with_kv_proj=True,
                )
            )
            for i in range(n_layers)
        )
        self.rotary_emb = nn.Module()
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads

    def forward(self, input_ids=None, past_key_values=None, **kwargs):
        bsz, q_len = input_ids.shape
        past = past_key_values.get_seq_length()
        positions = torch.arange(past, past + q_len)
        cos, sin = _rope_pos_emb(positions, self.head_dim)
        hidden = self.embed(input_ids)
        for layer in self.layers:
            attn = layer.self_attn
            k = attn.k_proj(hidden).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(hidden).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            past_key_values.update(k, v, attn.layer_idx)
            attn(
                hidden_states=hidden,
                past_key_values=past_key_values,
                position_embeddings=(cos, sin),
                cache_position=positions,
            )
        return None


class _FakeOuterModel:
    def __init__(self, inner, num_heads=2, num_kv_heads=2, hidden_size=8):
        self.model = inner
        self.config = SimpleNamespace(
            num_hidden_layers=len(inner.layers),
            num_key_value_heads=num_kv_heads,
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
            _attn_implementation="sdpa",
            name_or_path="fake/model",
        )
        self.calls = []
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def __call__(self, input_ids=None, past_key_values=None, **kwargs):
        self.calls.append({"input_ids": input_ids, "past_key_values": past_key_values, **kwargs})
        return self.model(input_ids=input_ids, past_key_values=past_key_values)


def _vendored_attn_weights(module, hidden_states, keys, position_embeddings, start_idx, end_idx, n_sink):
    """Verbatim transcription of kvpress KVzipPress.score_kvzip attention math
    (kvzip_press.py lines 299-328), up to (and including) the softmax."""
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    num_heads_kv = module.config.num_key_value_heads
    head_dim = module.head_dim
    num_key_value_groups = num_heads // num_heads_kv

    queries = module.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    queries = (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))
    queries = queries.view(bsz, num_heads_kv, num_key_value_groups, q_len, head_dim)

    sink = min(n_sink, start_idx)
    ctx_len = end_idx - start_idx
    keys_subsampled = torch.cat(
        [keys[:, :, :sink], keys[:, :, start_idx:end_idx], keys[:, :, -q_len:]], dim=2
    )
    keys_subsampled = keys_subsampled.unsqueeze(2).transpose(-2, -1).contiguous()

    attn_weights = torch.matmul(queries, keys_subsampled) / math.sqrt(head_dim)
    mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min)
    mask_cond = torch.arange(q_len)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(q_len, 1), 0)
    attn_weights[..., -q_len:, -q_len:] += mask[None, None, None, :, :]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    return attn_weights, sink, ctx_len


def _vendored_score_reference(module, hidden_states, keys, values, position_embeddings,
                              start_idx, end_idx, context_length, n_sink):
    """Verbatim transcription of kvpress KVzipPress.score_kvzip (lines 299-354)."""
    attn_weights, sink, ctx_len = _vendored_attn_weights(
        module, hidden_states, keys, position_embeddings, start_idx, end_idx, n_sink
    )
    attn_weights = attn_weights[..., sink : sink + ctx_len]
    scores = attn_weights.amax(dim=(-3, -2))
    return scores, keys[:, :, :context_length], values[:, :, :context_length]


def _kvpress_compress_post_reference(score_val, compression_ratio, layerwise):
    """Verbatim transcription of kvpress KVzipPress.compress_post (lines 361-390)."""
    n_layer, bsz, num_key_value_heads, ctx_len = score_val.shape

    if layerwise:
        nl = int(bsz * num_key_value_heads * ctx_len * compression_ratio)
        n_pruned_layers = nl * torch.ones(n_layer, device=score_val.device, dtype=torch.int)
    else:
        n_pruned_indices = int(score_val.numel() * compression_ratio)
        pruned_indices = torch.topk(-score_val.reshape(-1), n_pruned_indices).indices
        n_tokens_per_layer = bsz * num_key_value_heads * ctx_len
        n_pruned_layers = torch.bincount(pruned_indices // n_tokens_per_layer, minlength=n_layer).int()

    per_layer = []
    for layer_idx in range(n_layer):
        scores = score_val[layer_idx]
        n_pruned = n_pruned_layers[layer_idx].cpu()
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten().cpu()
        batch_indices = torch.arange(bsz, device=n_pruned.device).repeat_interleave(n_pruned)
        head_indices = indices // ctx_len
        seq_indices = indices % ctx_len
        per_layer.append((batch_indices, head_indices, seq_indices))
    return per_layer


def _post_model(n_layers, attn_implementation="sdpa"):
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                layer_idx=i, config=SimpleNamespace(_attn_implementation=attn_implementation)
            )
        )
        for i in range(n_layers)
    ]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


class TestKVzipRegistry(unittest.TestCase):
    def test_registered_name_resolves(self):
        self.assertIn("kvzip", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("kvzip"), KVzipSketch)

    def test_distinct_from_kvzap(self):
        self.assertIsNot(get_kv_compressor_class("kvzip"), get_kv_compressor_class("kvzap"))

    def test_get_kv_compressor_instantiates_with_kwargs(self):
        sketch = get_kv_compressor("kvzip", compression_ratio=0.4, layerwise=True, n_sink=2)
        self.assertIsInstance(sketch, KVzipSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.4)
        self.assertTrue(sketch.layerwise)
        self.assertEqual(sketch.n_sink, 2)
        self.assertFalse(sketch.kvzip_plus_normalization)
        self.assertEqual(sketch.chunk_size, 2048)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            get_kv_compressor_class("kvzip_does_not_exist")

    def test_ratio_validation(self):
        with self.assertRaises(AssertionError):
            KVzipSketch(compression_ratio=1.0)
        with self.assertRaises(AssertionError):
            KVzipSketch(compression_ratio=-0.1)


class TestPreropeQueryStates(unittest.TestCase):
    def test_q_proj_path(self):
        module = _FakeAttnModule(seed=1)
        torch.manual_seed(2)
        hidden = torch.randn(1, 3, 16)
        actual = _get_prerope_query_states(module, hidden)
        expected = module.q_proj(hidden).view(1, 3, 4, 4).transpose(1, 2)
        self.assertTrue(torch.equal(actual, expected))

    def test_fused_qkv_and_q_norm_paths(self):
        class _Doubler(nn.Module):
            def forward(self, x):
                return x * 2

        module = nn.Module()
        module.head_dim = 4
        module.layer_idx = 0
        module.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, hidden_size=16)
        torch.manual_seed(3)
        module.qkv_proj = nn.Linear(16, (4 + 2 + 2) * 4, bias=False)
        module.q_norm = _Doubler()

        torch.manual_seed(4)
        hidden = torch.randn(1, 3, 16)
        actual = _get_prerope_query_states(module, hidden)
        expected = module.qkv_proj(hidden)[..., :16].view(1, 3, 4, 4).transpose(1, 2) * 2
        self.assertTrue(torch.equal(actual, expected))

    def test_unsupported_module_raises(self):
        module = nn.Module()
        module.head_dim = 4
        module.config = SimpleNamespace(num_attention_heads=4)
        with self.assertRaises(NotImplementedError):
            _get_prerope_query_states(module, torch.randn(1, 2, 16))


class TestKVzipChunkingAndPrepare(unittest.TestCase):
    def test_chunk_boundaries(self):
        sketch = KVzipSketch()
        for length, expected in [(100, [100]), (2048, [2048]), (2049, [2048, 1]), (4096, [2048, 2048])]:
            chunks = sketch._chunk_fn(torch.zeros(1, length, dtype=torch.long), 2048)
            self.assertEqual([c.shape[1] for c in chunks], expected, msg=f"ctx_len={length}")

    def _prepared(self, n_sink=2):
        tok = _StubTokenizer()
        sketch = KVzipSketch(compression_ratio=0.5, chunk_size=4, n_sink=n_sink, tokenizer=tok)
        ctx = torch.arange(100, 112).unsqueeze(0)  # 12 context tokens
        sketch._context_ids = ctx
        sketch.context_length = 12
        sketch.prefix_length = 3
        sketch._suffix_ids = tok.encode("<SUF>")
        model = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=3, num_key_value_heads=2),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        pairs = sketch.prepare(model, tok)
        return sketch, tok, ctx, pairs

    def test_prepare_repeat_inputs_pinned(self):
        sketch, tok, ctx, pairs = self._prepared()
        post = ctx[:, 3:]
        chunks = [post[:, 0:4], post[:, 4:8], post[:, 8:9]]
        suffix = tok.encode("<SUF>")
        prompt0 = tok.encode("\n\nRepeat the previous context exactly.")
        prompt_i = tok.encode("\n\nRepeat the part of the previous context exactly, starting with")

        self.assertEqual(len(pairs), 3)
        for pair, chunk in zip(pairs, chunks):
            self.assertTrue(torch.equal(pair[0], chunk))
        self.assertTrue(torch.equal(pairs[0][1], torch.cat([prompt0, suffix, chunks[0]], dim=1)))
        # prev_postfix_size=8 > chunk length 4: the whole previous chunk is prepended.
        self.assertTrue(
            torch.equal(pairs[1][1], torch.cat([prompt_i, chunks[0][:, -8:], suffix, chunks[1]], dim=1))
        )
        self.assertTrue(
            torch.equal(pairs[2][1], torch.cat([prompt_i, chunks[1][:, -8:], suffix, chunks[2]], dim=1))
        )

    def test_prepare_score_val_init(self):
        sketch, _, _, _ = self._prepared(n_sink=2)
        self.assertEqual(tuple(sketch.score_val.shape), (3, 1, 2, 12))
        self.assertEqual(sketch.score_val.dtype, torch.float32)
        self.assertTrue((sketch.score_val[..., :2] == 1.0).all())
        self.assertTrue((sketch.score_val[..., 2:] == 0.0).all())


class TestKVzipMask(unittest.TestCase):
    def test_make_mask_values(self):
        sketch = KVzipSketch()
        attn = torch.zeros(1, 2, 2, 4, 9)
        sketch._make_mask(attn, 4)
        self.assertEqual(tuple(sketch.causal_mask_score.shape), (1, 1, 1, 4, 4))
        mask = sketch.causal_mask_score[0, 0, 0]
        dtype_min = torch.finfo(torch.float32).min
        for i in range(4):
            for j in range(4):
                self.assertEqual(mask[i, j].item(), 0.0 if j <= i else dtype_min)

    def test_mask_applied_only_to_trailing_block(self):
        sketch = KVzipSketch()
        attn = torch.zeros(1, 2, 2, 3, 10)
        sketch._mask_causal(attn, 3)
        self.assertTrue((attn[..., :7] == 0).all())
        dtype_min = torch.finfo(torch.float32).min
        block = attn[..., -3:]
        for i in range(3):
            for j in range(3):
                expected = 0.0 if j <= i else dtype_min
                self.assertTrue((block[..., i, j] == expected).all())

    def test_mask_cached_then_rebuilt_on_window_change(self):
        sketch = KVzipSketch()
        sketch._mask_causal(torch.zeros(1, 1, 1, 3, 6), 3)
        first = sketch.causal_mask_score
        sketch._mask_causal(torch.zeros(1, 1, 1, 3, 8), 3)
        self.assertIs(sketch.causal_mask_score, first)
        sketch._mask_causal(torch.zeros(1, 1, 1, 5, 9), 5)
        self.assertEqual(sketch.causal_mask_score.size(-1), 5)


class TestKVzipScoreOracle(unittest.TestCase):
    CTX, Q_LEN, D, H_KV = 10, 5, 4, 2

    def _setup(self, seed=0, with_o_proj=False):
        module = _FakeAttnModule(num_heads=4, num_kv_heads=2, head_dim=4, seed=seed, with_o_proj=with_o_proj)
        torch.manual_seed(seed + 100)
        hidden = torch.randn(1, self.Q_LEN, 16)
        keys = torch.randn(1, 2, self.CTX + self.Q_LEN, 4)
        values = torch.randn(1, 2, self.CTX + self.Q_LEN, 4)
        pos_emb = _rope_pos_emb(torch.arange(self.CTX, self.CTX + self.Q_LEN), 4)
        return module, hidden, keys, values, pos_emb

    def _run(self, sketch, module, hidden, keys, values, pos_emb, start, end):
        sketch.score_val = torch.zeros(1, 1, self.H_KV, self.CTX)
        sketch.context_length = self.CTX
        sketch.start_idx = start
        sketch.end_idx = end
        return sketch.score_kvzip(module, hidden, keys, values, None, {"position_embeddings": pos_emb})

    def test_matches_vendored_kvpress_math_bitwise(self):
        module, hidden, keys, values, pos_emb = self._setup()
        sketch = KVzipSketch(compression_ratio=0.5, n_sink=2)
        out_keys, out_values = self._run(sketch, module, hidden, keys, values, pos_emb, 2, 7)

        ref_scores, ref_keys, ref_values = _vendored_score_reference(
            module, hidden, keys, values, pos_emb, 2, 7, self.CTX, 2
        )
        self.assertEqual(tuple(ref_scores.shape), (1, self.H_KV, 5))
        self.assertTrue(torch.equal(sketch.score_val[0][..., 2:7], ref_scores))
        # untouched slots keep their previous values
        self.assertTrue((sketch.score_val[0][..., :2] == 0).all())
        self.assertTrue((sketch.score_val[0][..., 7:] == 0).all())
        # trim invariant: only the originally prefilled KV pairs are returned
        self.assertTrue(torch.equal(out_keys, keys[:, :, : self.CTX]))
        self.assertTrue(torch.equal(out_values, values[:, :, : self.CTX]))
        self.assertTrue(torch.equal(ref_keys, out_keys))
        self.assertTrue(torch.equal(ref_values, out_values))

    def test_matches_explicit_loop_reference(self):
        module, hidden, keys, values, pos_emb = self._setup(seed=3)
        sketch = KVzipSketch(compression_ratio=0.5, n_sink=2)
        self._run(sketch, module, hidden, keys, values, pos_emb, 2, 7)

        cos, sin = pos_emb
        q = module.q_proj(hidden).view(1, self.Q_LEN, 4, self.D).transpose(1, 2)
        q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
        q = q.view(1, 2, 2, self.Q_LEN, self.D)
        sink = 2
        k_sub = torch.cat([keys[:, :, :sink], keys[:, :, 2:7], keys[:, :, -self.Q_LEN :]], dim=2)
        total = k_sub.shape[2]  # sink + chunk + repeat = 12
        dtype_min = torch.finfo(torch.float32).min

        expected = torch.zeros(self.H_KV, 5)
        for h in range(self.H_KV):
            best = torch.full((5,), -float("inf"))
            for g in range(2):
                for t in range(self.Q_LEN):
                    logits = torch.stack(
                        [(q[0, h, g, t] @ k_sub[0, h, j]) / math.sqrt(self.D) for j in range(total)]
                    )
                    for j in range(total - self.Q_LEN, total):
                        if (j - (total - self.Q_LEN)) > t:
                            logits[j] = logits[j] + dtype_min
                    row = torch.softmax(logits, dim=-1)
                    best = torch.maximum(best, row[sink : sink + 5])
            expected[h] = best

        torch.testing.assert_close(sketch.score_val[0][0, :, 2:7], expected, atol=1e-5, rtol=1e-5)

    def test_sink_clamped_to_start_idx(self):
        # start_idx=0 (no prefix): sink = min(n_sink, 0) = 0, no sink keys prepended.
        module, hidden, keys, values, pos_emb = self._setup(seed=7)
        sketch = KVzipSketch(compression_ratio=0.5, n_sink=4)
        self._run(sketch, module, hidden, keys, values, pos_emb, 0, 5)
        ref_scores, _, _ = _vendored_score_reference(module, hidden, keys, values, pos_emb, 0, 5, self.CTX, 4)
        self.assertTrue(torch.equal(sketch.score_val[0][..., 0:5], ref_scores))

    def test_gqa_duplicate_group_equals_single_head_amax(self):
        module, hidden, keys, values, pos_emb = self._setup(seed=5)
        with torch.no_grad():
            # query heads 0 and 1 form kv-group 0; make them identical
            module.q_proj.weight[4:8] = module.q_proj.weight[0:4]
        sketch = KVzipSketch(compression_ratio=0.5, n_sink=2)
        self._run(sketch, module, hidden, keys, values, pos_emb, 2, 7)

        attn_weights, sink, ctx_len = _vendored_attn_weights(module, hidden, keys, pos_emb, 2, 7, 2)
        # scores are per KV head (never H_q)
        self.assertEqual(tuple(sketch.score_val[0].shape), (1, self.H_KV, self.CTX))
        single_head = attn_weights[0, 0, 0][..., sink : sink + ctx_len].amax(dim=0)
        torch.testing.assert_close(sketch.score_val[0][0, 0, 2:7], single_head, atol=1e-6, rtol=1e-6)


class TestKVzipPlusNormalization(unittest.TestCase):
    def _setup(self):
        module = _FakeAttnModule(num_heads=4, num_kv_heads=2, head_dim=4, seed=11, with_o_proj=True)
        torch.manual_seed(12)
        ctx, q_len = 8, 4
        hidden = torch.randn(1, q_len, 16)
        keys = torch.randn(1, 2, ctx + q_len, 4)
        values = torch.randn(1, 2, ctx + q_len, 4)
        pos_emb = _rope_pos_emb(torch.arange(ctx, ctx + q_len), 4)
        return module, hidden, keys, values, pos_emb, ctx, q_len

    def test_matches_double_loop_reference(self):
        module, hidden, keys, values, pos_emb, ctx, q_len = self._setup()
        start, end, n_sink = 2, 6, 2
        sketch = KVzipSketch(compression_ratio=0.5, n_sink=n_sink, kvzip_plus_normalization=True)
        sketch.score_val = torch.zeros(1, 1, 2, ctx)
        sketch.context_length = ctx
        sketch.start_idx, sketch.end_idx = start, end
        sketch.score_kvzip(module, hidden, keys, values, None, {"position_embeddings": pos_emb})

        attn, sink, ctx_len = _vendored_attn_weights(module, hidden, keys, pos_emb, start, end, n_sink)
        ref = attn.clone()
        for t in range(q_len):
            ref[0, :, :, t, :] = ref[0, :, :, t, :] / hidden[0, t].norm()
        Wo = module.o_proj.weight.transpose(0, 1).view(2, 2, 4, 16)
        v_sub = torch.cat([values[:, :, :sink], values[:, :, start:end], values[:, :, -q_len:]], dim=2)
        for h in range(2):
            for g in range(2):
                for col in range(v_sub.shape[2]):
                    wov = v_sub[0, h, col] @ Wo[h, g]  # [hidden_size]
                    ref[0, h, g, :, col] = ref[0, h, g, :, col] * wov.norm()
        ref_scores = ref[..., sink : sink + ctx_len].amax(dim=(-3, -2))

        torch.testing.assert_close(sketch.score_val[0][..., start:end], ref_scores, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(sketch.score_val).all())
        self.assertEqual(tuple(sketch.score_val[0].shape), (1, 2, ctx))

    def test_wov_einsum_matches_loop(self):
        module, hidden, keys, values, pos_emb, ctx, q_len = self._setup()
        sink, start, end = 2, 2, 6
        Wo = module.o_proj.weight.transpose(0, 1).view(2, 2, 4, 16)
        v_sub = torch.cat([values[:, :, :sink], values[:, :, start:end], values[:, :, -q_len:]], dim=2)
        V = v_sub.unsqueeze(2).transpose(-2, -1).contiguous().repeat_interleave(2, dim=2)
        prod = torch.einsum("h g i j, b h g i t -> b h g t j", Wo, V).norm(dim=-1)

        total = v_sub.shape[2]
        loop = torch.zeros(1, 2, 2, total)
        for h in range(2):
            for g in range(2):
                for t in range(total):
                    acc = torch.zeros(16)
                    for i in range(4):
                        acc = acc + Wo[h, g, i] * v_sub[0, h, t, i]
                    loop[0, h, g, t] = acc.norm()
        torch.testing.assert_close(prod, loop, atol=1e-5, rtol=1e-5)


class TestKVzipCompressPost(unittest.TestCase):
    def test_layerwise_pinned_selection(self):
        sketch = KVzipSketch(compression_ratio=0.25, layerwise=True)
        sketch.score_val = torch.tensor(
            [
                [[[0.16, 0.01, 0.15, 0.02, 0.14, 0.03, 0.13, 0.04],
                  [0.12, 0.11, 0.10, 0.09, 0.50, 0.51, 0.52, 0.53]]],
                [[[0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98],
                  [0.20, 0.19, 0.18, 0.17, 0.50, 0.51, 0.52, 0.53]]],
            ]
        )
        model = _post_model(2)
        sketch.compress_post(model)

        m0 = model.model.layers[0].self_attn
        b0, h0, s0 = m0.masked_key_indices
        self.assertEqual(s0.numel(), int(1 * 2 * 8 * 0.25))
        self.assertTrue(torch.equal(b0, torch.tensor([0, 0, 0, 0])))
        self.assertTrue(torch.equal(h0, torch.tensor([0, 0, 0, 0])))
        self.assertTrue(torch.equal(s0, torch.tensor([1, 3, 5, 7])))

        m1 = model.model.layers[1].self_attn
        b1, h1, s1 = m1.masked_key_indices
        self.assertTrue(torch.equal(h1, torch.tensor([1, 1, 1, 1])))
        self.assertTrue(torch.equal(s1, torch.tensor([3, 2, 1, 0])))

    def test_global_budget_uneven_across_layers(self):
        # numel = 32, ratio = 6/32: the 6 smallest scores all live in layer 1.
        sketch = KVzipSketch(compression_ratio=0.1875, layerwise=False)
        sketch.score_val = torch.tensor(
            [
                [[[0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57],
                  [0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67]]],
                [[[1.00, 0.01, 0.02, 1.00, 0.03, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 0.04, 0.05, 1.00, 1.00, 0.06, 1.00]]],
            ]
        )
        model = _post_model(2)
        sketch.compress_post(model)

        b0, h0, s0 = model.model.layers[0].self_attn.masked_key_indices
        self.assertEqual(b0.numel(), 0)
        self.assertEqual(h0.numel(), 0)
        self.assertEqual(s0.numel(), 0)

        b1, h1, s1 = model.model.layers[1].self_attn.masked_key_indices
        self.assertTrue(torch.equal(b1, torch.zeros(6, dtype=torch.long)))
        self.assertTrue(torch.equal(h1, torch.tensor([0, 0, 0, 1, 1, 1])))
        self.assertTrue(torch.equal(s1, torch.tensor([1, 2, 4, 2, 3, 6])))

    def test_sink_scores_never_pruned(self):
        sketch = KVzipSketch(compression_ratio=0.5, layerwise=True, n_sink=2)
        sketch.score_val = torch.tensor(
            [
                [[[1.00, 1.00, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45],
                  [1.00, 1.00, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35]]],
            ]
        )
        model = _post_model(1)
        sketch.compress_post(model)
        _, _, seq_indices = model.model.layers[0].self_attn.masked_key_indices
        self.assertEqual(seq_indices.numel(), 8)
        self.assertEqual(set(seq_indices.tolist()) & {0, 1}, set())

    def test_randomized_oracle_against_kvpress_transcription(self):
        torch.manual_seed(42)
        score_val = torch.rand(3, 1, 2, 7)
        for layerwise in (True, False):
            sketch = KVzipSketch(compression_ratio=0.43, layerwise=layerwise)
            sketch.score_val = score_val.clone()
            model = _post_model(3)
            sketch.compress_post(model)
            expected = _kvpress_compress_post_reference(score_val, 0.43, layerwise)
            for layer, (ref_b, ref_h, ref_s) in zip(model.model.layers, expected):
                act_b, act_h, act_s = layer.self_attn.masked_key_indices
                self.assertTrue(torch.equal(act_b, ref_b), msg=f"layerwise={layerwise}")
                self.assertTrue(torch.equal(act_h, ref_h), msg=f"layerwise={layerwise}")
                self.assertTrue(torch.equal(act_s, ref_s), msg=f"layerwise={layerwise}")

    def test_eager_attention_rejected(self):
        sketch = KVzipSketch(compression_ratio=0.5, layerwise=True)
        sketch.score_val = torch.rand(1, 1, 2, 8)
        with self.assertRaisesRegex(AssertionError, "eager mode not supported"):
            sketch.compress_post(_post_model(1, attn_implementation="eager"))


class TestAttentionPatchEndToEnd(unittest.TestCase):
    """First-consumer pin of the globally patched ALL_ATTENTION_FUNCTIONS sdpa entry."""

    def _patched_sdpa(self):
        import eval_harness.kv_compression  # noqa: F401  (applies patch_attention_functions at import)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        return ALL_ATTENTION_FUNCTIONS["sdpa"]

    def test_masked_entries_excluded_through_patched_sdpa(self):
        sdpa = self._patched_sdpa()
        torch.manual_seed(21)
        B, H_q, H_kv, S, D = 1, 4, 2, 8, 8
        module = SimpleNamespace(
            num_key_value_groups=2,
            is_causal=True,
            masked_key_indices=(torch.tensor([0, 0, 0]), torch.tensor([0, 0, 1]), torch.tensor([2, 5, 0])),
        )
        q = torch.randn(B, H_q, 1, D)
        k = torch.randn(B, H_kv, S, D)
        v = torch.randn(B, H_kv, S, D)
        k_orig = k.clone()
        bias = torch.zeros(B, 1, 1, S)  # no-op float mask (forces the repeat_kv path)

        out, _ = sdpa(module, q, k, v, bias, 0.0, scaling=None, is_causal=False)
        self.assertEqual(tuple(out.shape), (1, 1, H_q, D))

        masked = {0: {2, 5}, 1: {0}}
        for h in range(H_q):
            g = h // 2
            kept = [s for s in range(S) if s not in masked[g]]
            logits = (q[0, h, 0] @ k_orig[0, g, kept].T) / math.sqrt(D)
            ref = torch.softmax(logits, dim=-1) @ v[0, g, kept]
            torch.testing.assert_close(out[0, 0, h], ref, atol=1e-5, rtol=1e-5)

            # the substituted fake keys leave ~zero attention mass on masked slots
            full_logits = (q[0, h, 0] @ k[0, g].T) / math.sqrt(D)
            weights = torch.softmax(full_logits, dim=-1)
            for s in masked[g]:
                self.assertLess(weights[s].item(), 1e-12)

        # masked rows were replaced in place; all other rows are bitwise unchanged
        for g in range(H_kv):
            for s in range(S):
                if s in masked[g]:
                    self.assertFalse(torch.equal(k[0, g, s], k_orig[0, g, s]))
                else:
                    self.assertTrue(torch.equal(k[0, g, s], k_orig[0, g, s]))

    def test_full_prefill_resets_masked_key_indices(self):
        sdpa = self._patched_sdpa()
        torch.manual_seed(22)
        module = SimpleNamespace(
            is_causal=True,
            masked_key_indices=(torch.tensor([0]), torch.tensor([0]), torch.tensor([3])),
        )
        q = torch.randn(1, 2, 8, 8)
        k = torch.randn(1, 2, 8, 8)
        k_orig = k.clone()
        sdpa(module, q, k, k.clone(), torch.zeros(1, 1, 8, 8), 0.0, scaling=None, is_causal=False)
        self.assertIsNone(module.masked_key_indices)
        self.assertTrue(torch.equal(k, k_orig))


class TestKVzipLifecycle(unittest.TestCase):
    def test_zero_ratio_noop_lifecycle(self):
        inner = _FakeInnerModel()
        model = _FakeOuterModel(inner)
        sketch = KVzipSketch(compression_ratio=0.0, tokenizer=_StubTokenizer())
        cache = DynamicCache()
        ctx_ids = torch.randint(0, 1000, (1, 8))

        with sketch(model):
            self.assertIn("forward", model.model.__dict__)  # capture wrapper installed
            model.model(input_ids=ctx_ids, past_key_values=cache)
            self.assertIs(sketch._context_ids, ctx_ids)
            self.assertIs(sketch._cache, cache)

        self.assertEqual(model.calls, [])  # no reconstruction passes
        self.assertIs(model.model.forward.__func__, _FakeInnerModel.forward)
        for layer in inner.layers:
            self.assertEqual(len(layer.self_attn._forward_hooks), 0)
            self.assertIsNone(getattr(layer.self_attn, "masked_key_indices", None))
        self.assertIsNone(sketch._context_ids)
        self.assertIsNone(sketch._cache)
        self.assertIsNone(sketch.score_val)
        self.assertEqual(sketch.context_length, 0)
        self.assertEqual(sketch.start_idx, 0)
        self.assertEqual(sketch.end_idx, 0)

    def test_forward_restored_when_prefill_raises(self):
        inner = _FakeInnerModel()
        model = _FakeOuterModel(inner)
        sketch = KVzipSketch(compression_ratio=0.5, tokenizer=_StubTokenizer())
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with sketch(model):
                raise RuntimeError("boom")
        self.assertIs(model.model.forward.__func__, _FakeInnerModel.forward)
        self.assertIsNone(sketch._context_ids)

    def test_full_lifecycle_integration(self):
        S, n_layers, h_kv, ratio = 32, 2, 2, 0.5
        inner = _FakeInnerModel(n_layers=n_layers, num_heads=2, num_kv_heads=h_kv, head_dim=4, hidden_size=8)
        model = _FakeOuterModel(inner)
        tok = _StubTokenizer()  # chat_template=None: prefix_length=0, suffix="\n"
        sketch = KVzipSketch(compression_ratio=ratio, chunk_size=8, tokenizer=tok)
        cache = DynamicCache()
        torch.manual_seed(7)
        ctx_ids = torch.randint(0, 1000, (1, S))

        with sketch(model):
            model.model(input_ids=ctx_ids, past_key_values=cache)
            self.assertIs(sketch._context_ids, ctx_ids)
            self.assertIs(sketch._cache, cache)
            prefill_keys = [cache.layers[i].keys.clone() for i in range(n_layers)]
            prefill_values = [cache.layers[i].values.clone() for i in range(n_layers)]

        # ceil(32/8) = 4 reconstruction passes, each logits_to_keep=1 on the SAME cache
        self.assertEqual(len(model.calls), 4)
        for call in model.calls:
            self.assertEqual(call["logits_to_keep"], 1)
            self.assertIs(call["past_key_values"], cache)

        suffix = tok.encode("\n")
        prompt0 = tok.encode("\n\nRepeat the previous context exactly.")
        prompt_i = tok.encode("\n\nRepeat the part of the previous context exactly, starting with")
        expected0 = torch.cat([prompt0, suffix, ctx_ids[:, 0:8]], dim=1)
        self.assertTrue(torch.equal(model.calls[0]["input_ids"], expected0))
        expected1 = torch.cat([prompt_i, ctx_ids[:, 0:8][:, -8:], suffix, ctx_ids[:, 8:16]], dim=1)
        self.assertTrue(torch.equal(model.calls[1]["input_ids"], expected1))

        # fake compression: rectangular, untrimmed, unmodified context cache
        total_masked = 0
        for i in range(n_layers):
            self.assertEqual(cache.layers[i].keys.shape[2], S)
            self.assertTrue(torch.equal(cache.layers[i].keys, prefill_keys[i]))
            self.assertTrue(torch.equal(cache.layers[i].values, prefill_values[i]))
            attn = inner.layers[i].self_attn
            self.assertEqual(len(attn._forward_hooks), 0)
            b, h, s = attn.masked_key_indices
            self.assertTrue((b == 0).all())
            self.assertTrue((h < h_kv).all())
            self.assertTrue((s < S).all())
            total_masked += s.numel()
        self.assertEqual(total_masked, int(n_layers * 1 * h_kv * S * ratio))

        # internal state reset; wrapper restored
        self.assertIsNone(sketch._context_ids)
        self.assertIsNone(sketch._cache)
        self.assertIsNone(sketch.score_val)
        self.assertEqual(sketch.context_length, 0)
        self.assertIs(model.model.forward.__func__, _FakeInnerModel.forward)

        # multi-question interplay: checkpoint -> append question KV -> restore
        snapshot = [
            tuple(t.clone() for t in inner.layers[i].self_attn.masked_key_indices) for i in range(n_layers)
        ]
        adapter = StandardCacheAdapter(model=None)
        checkpoint = adapter.clone_or_checkpoint_for_multi_question(cache)
        for i in range(n_layers):
            cache.update(torch.randn(1, h_kv, 5, 4), torch.randn(1, h_kv, 5, 4), i)
        self.assertEqual(cache.layers[0].keys.shape[2], S + 5)
        adapter.restore_after_question(cache, checkpoint)
        for i in range(n_layers):
            self.assertEqual(cache.layers[i].keys.shape[2], S)
            for actual, expected in zip(inner.layers[i].self_attn.masked_key_indices, snapshot[i]):
                self.assertTrue(torch.equal(actual, expected))
            self.assertTrue((inner.layers[i].self_attn.masked_key_indices[2] < S).all())


class TestKVzipGuards(unittest.TestCase):
    def test_mixed_attention_layer_raises(self):
        sliding = SimpleNamespace(self_attn=SimpleNamespace(is_sliding=True))
        full = SimpleNamespace(self_attn=SimpleNamespace(is_sliding=False))
        model = SimpleNamespace(model=SimpleNamespace(layers=[full, sliding]))
        sketch = KVzipSketch(compression_ratio=0.5, tokenizer=_StubTokenizer())
        with self.assertRaisesRegex(ValueError, "full-attention"):
            with sketch(model):
                pass

    def test_gemma3_raises(self):
        class _FakeGemmaBase:
            pass

        class _FakeGemmaModel(_FakeGemmaBase):
            pass

        sketch = KVzipSketch(compression_ratio=0.5, tokenizer=_StubTokenizer())
        with mock.patch.object(kvzip_module, "Gemma3PreTrainedModel", _FakeGemmaBase):
            with self.assertRaisesRegex(ValueError, "Gemma3"):
                with sketch(_FakeGemmaModel()):
                    pass


class TestKVzipPrefixSuffixExtraction(unittest.TestCase):
    def test_no_chat_template_defaults(self):
        sketch = KVzipSketch()
        sketch._extract_prefix_suffix(_StubTokenizer(chat_template=None))
        self.assertEqual(sketch.prefix_length, 0)
        self.assertTrue(torch.equal(sketch._suffix_ids, torch.tensor([[10]])))  # "\n"

    def test_chat_template_split(self):
        tok = _StubTokenizer(chat_template="present")
        sketch = KVzipSketch()
        sketch._extract_prefix_suffix(tok)
        # template renders "<PRE>" + content + "<SUF>"; prefix is "<PRE>" (5 chars),
        # suffix is "<SUF>" under the char-level stub.
        self.assertEqual(sketch.prefix_length, 5)
        self.assertTrue(torch.equal(sketch._suffix_ids, tok.encode("<SUF>")))


if __name__ == "__main__":
    unittest.main()
