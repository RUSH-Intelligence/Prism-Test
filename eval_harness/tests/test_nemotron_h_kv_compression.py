"""KV-compression on a NemotronH Mamba-attention hybrid: full-attention layers only.

NemotronH (``nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16``) interleaves Mamba2, MLP
and a *handful* of full-attention blocks (the 4B model is 42 layers with only 4
attention layers). Unlike Llama-style models:

* the attention module lives under ``block.mixer`` (there is **no**
  ``block.self_attn``), with sibling mamba/mlp mixers that hold no K/V cache;
* blocks are typed by ``block.block_type`` / ``config.layers_block_type``
  (``hybrid_override_pattern``), not ``layer_types``;
* attention applies **no RoPE** (no ``position_embeddings``).

Only the attention blocks carry a K/V cache, so a KV compressor must hook *only*
those and leave the mamba/mlp cache slots untouched. These tests build a faithful
fake NemotronH (mirroring the native ``transformers`` module/attr layout) — no
weights, no GPU — and assert exactly that for knorm, ridge, snapkv and pyramidkv.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eval_harness.kv_compression.base import (
    _get_language_model,
    _is_non_full_attention_layer,
    _resolve_attention_module,
)
from eval_harness.kv_compression.cache_adapter import (
    HybridCacheAdapter,
    _is_hybrid_model,
    create_cache_adapter,
)
from eval_harness.kv_compression.compressors.compactor_sketch import CompactorSketch
from eval_harness.kv_compression.compressors.expected_attention_sketch import (
    ExpectedAttentionSketch,
)
from eval_harness.kv_compression.compressors.keydiff_sketch import KeyDiffSketch
from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.compressors.pyramidkv_sketch import PyramidKVSketch
from eval_harness.kv_compression.compressors.ridge_sketch import RidgeSketch
from eval_harness.kv_compression.compressors.snapkv_sketch import SnapKVSketch

# The real NVIDIA-Nemotron-3-Nano-4B-BF16 pattern (42 layers; M=mamba, *=attention,
# -=mlp). Only positions 13, 18, 25, 33 are full attention.
NEMOTRON_3_NANO_4B_PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
EXPECTED_ATTENTION_INDICES = [12, 17, 24, 32]


# ---------------------------------------------------------------------------
# Faithful fake NemotronH (mirrors native transformers nemotron_h module names)
# ---------------------------------------------------------------------------

def _make_nemotron_config(pattern=NEMOTRON_3_NANO_4B_PATTERN, *, hidden_size=32,
                          num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, attn_implementation="sdpa"):
    block_for = {"M": "mamba", "*": "attention", "-": "mlp"}
    layers_block_type = [block_for[c] for c in pattern]
    cfg = SimpleNamespace(
        hybrid_override_pattern=pattern,
        layers_block_type=layers_block_type,
        # Native transformers aliases ``layer_types`` -> ``layers_block_type``.
        layer_types=layers_block_type,
        num_hidden_layers=len(pattern),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        sliding_window=None,
    )
    cfg._attn_implementation = attn_implementation
    cfg.get_text_config = lambda decoder=True: cfg
    return cfg


class _FakeNemotronHAttention(nn.Module):
    """Mirrors native ``NemotronHAttention`` attrs used by the compressors."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        h = config.hidden_size
        nh, nkv, d = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.q_proj = nn.Linear(h, nh * d, bias=False)
        self.k_proj = nn.Linear(h, nkv * d, bias=False)
        self.v_proj = nn.Linear(h, nkv * d, bias=False)
        self.o_proj = nn.Linear(nh * d, h, bias=False)

    def forward(self, hidden_states, past_key_values=None, cache_position=None, **kwargs):
        # The cache is pre-populated by the test; the post-attention hook reads
        # and rewrites it. Return the native (attn_output, attn_weights) shape so
        # the hook's ``output[1]`` is the (None) attention-weights slot.
        return hidden_states, None


class _FakeNonAttentionMixer(nn.Module):
    """Stand-in for ``NemotronHMamba2Mixer`` / ``NemotronHMLP`` (no K/V, no hook).

    Mirrors the native mamba mixer's ``layer_idx`` (native sets it), so a
    *wrongly* installed hook would fail at the cache read (keys is None) — the
    real model's failure mode — rather than at ``module.layer_idx``.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _FakeNemotronHBlock(nn.Module):
    def __init__(self, config, layer_idx, block_type):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = block_type
        if block_type == "attention":
            self.mixer = _FakeNemotronHAttention(config, layer_idx)
        else:
            self.mixer = _FakeNonAttentionMixer(config, layer_idx)


class _FakeNemotronHModel(nn.Module):
    """The decoder backbone (native exposes this at ``ForCausalLM.model``).

    Deliberately has NO ``rotary_emb`` — NemotronH attention uses no RoPE.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            _FakeNemotronHBlock(config, i, bt)
            for i, bt in enumerate(config.layers_block_type)
        )


class _FakeNemotronHForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = _FakeNemotronHModel(config)


# ---------------------------------------------------------------------------
# Fake hybrid cache (mirrors a DynamicCache: ``.layers[idx].keys/.values``).
# Mamba/MLP slots have no keys/values, exactly like the real hybrid cache.
# ---------------------------------------------------------------------------

class _AttnCacheLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def get_seq_length(self):
        return int(self.keys.shape[2])


class _NonAttnCacheSlot:
    """A mamba/mlp cache slot, faithful to the transformers 5.x unified cache.

    Every per-layer slot is a CacheLayerMixin exposing ``keys``/``values`` — for
    mamba/mlp layers they stay ``None`` (no attention populates them) while the
    recurrent state lives elsewhere. ``get_seq_length`` returns 0. (A naive
    ``hasattr`` attention-layer check would wrongly treat this as sliceable and
    crash on ``None[:, :, :n]`` — see _can_slice_attention_kv.)
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.conv_states = torch.zeros(1, 1, 1)
        self.recurrent_states = torch.zeros(1, 1, 1)

    def get_seq_length(self):
        return 0


class _FakeHybridCache:
    def __init__(self, layers):
        self.layers = layers

    def __len__(self):
        return len(self.layers)


def _build_model_and_cache(*, T=80, batch=1, seed=0, attn_implementation="sdpa",
                           pattern=NEMOTRON_3_NANO_4B_PATTERN):
    """Build a fake NemotronH + a populated hybrid cache.

    Returns (model, cache, attention_indices, sentinels) where ``sentinels`` maps
    non-attention layer index -> its (untouched) cache slot object.
    """
    torch.manual_seed(seed)
    config = _make_nemotron_config(pattern, attn_implementation=attn_implementation)
    model = _FakeNemotronHForCausalLM(config)
    nkv, D = config.num_key_value_heads, config.head_dim

    layers = []
    sentinels = {}
    attention_indices = []
    for i, block in enumerate(model.model.layers):
        if block.block_type == "attention":
            attention_indices.append(i)
            layers.append(
                _AttnCacheLayer(
                    torch.randn(batch, nkv, T, D),
                    torch.randn(batch, nkv, T, D),
                )
            )
        else:
            slot = _NonAttnCacheSlot()
            sentinels[i] = slot
            layers.append(slot)
    cache = _FakeHybridCache(layers)
    return model, cache, attention_indices, sentinels


def _run_prefill_through_blocks(compressor, model, cache, T, hidden_size):
    """Install the compressor and drive a prefill forward through EVERY block.

    Mamba/MLP mixers are called with the cache too: if a hook were wrongly
    installed on one, the hook would read a slot with no ``.keys`` and raise —
    so a clean run is itself proof the install loop skipped them.
    """
    hidden = torch.randn(1, T, hidden_size)
    cache_position = torch.arange(T)
    with compressor(model):
        for block in model.model.layers:
            block.mixer(
                hidden_states=hidden,
                past_key_values=cache,
                cache_position=cache_position,
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNemotronHDetection(unittest.TestCase):
    def test_is_hybrid_model_true(self):
        model = _FakeNemotronHForCausalLM(_make_nemotron_config())
        self.assertTrue(_is_hybrid_model(model))

    def test_is_hybrid_via_pattern_only(self):
        # A config carrying only the raw pattern string still resolves as hybrid.
        cfg = SimpleNamespace(hybrid_override_pattern=NEMOTRON_3_NANO_4B_PATTERN)
        cfg.get_text_config = lambda decoder=True: cfg
        model = SimpleNamespace(config=cfg)
        self.assertTrue(_is_hybrid_model(model))

    def test_create_cache_adapter_is_hybrid(self):
        model = _FakeNemotronHForCausalLM(_make_nemotron_config())
        self.assertIsInstance(create_cache_adapter(model), HybridCacheAdapter)

    def test_dense_model_not_hybrid(self):
        # Sanity guard: a non-hybrid (pure attention) config must stay Standard.
        cfg = SimpleNamespace(layer_types=["full_attention"] * 4)
        cfg.get_text_config = lambda decoder=True: cfg
        self.assertFalse(_is_hybrid_model(SimpleNamespace(config=cfg)))


class TestNemotronHLayerTargeting(unittest.TestCase):
    def setUp(self):
        self.model = _FakeNemotronHForCausalLM(_make_nemotron_config())

    def test_pattern_has_four_attention_layers(self):
        idx = [i for i, c in enumerate(NEMOTRON_3_NANO_4B_PATTERN) if c == "*"]
        self.assertEqual(idx, EXPECTED_ATTENTION_INDICES)
        self.assertEqual(len(NEMOTRON_3_NANO_4B_PATTERN), 42)

    def test_only_attention_blocks_are_full_attention(self):
        full = [
            i for i, b in enumerate(self.model.model.layers)
            if not _is_non_full_attention_layer(b)
        ]
        self.assertEqual(full, EXPECTED_ATTENTION_INDICES)

    def test_resolve_attention_module_targets_mixer(self):
        for i, block in enumerate(self.model.model.layers):
            resolved = _resolve_attention_module(block)
            if block.block_type == "attention":
                self.assertIs(resolved, block.mixer)
                self.assertTrue(hasattr(resolved, "q_proj"))
            else:
                self.assertIsNone(resolved)

    def test_get_language_model_resolves_backbone(self):
        self.assertIs(_get_language_model(self.model), self.model.model)

    def test_install_hooks_exactly_on_attention_mixers(self):
        # The central targeting invariant, pinned DIRECTLY (not merely inferred
        # from a clean prefill run): the install loop registers a forward hook on
        # EXACTLY the 4 attention mixers and on NONE of the mamba/mlp mixers, and
        # removes them all on context exit. A regression that hooked mamba/mlp
        # blocks (or skipped an attention block) is caught here even if no
        # forward is ever run.
        sketch = KnormSketch(compression_ratio=0.5)
        with sketch(self.model):
            hooked = [
                i for i, block in enumerate(self.model.model.layers)
                if len(block.mixer._forward_hooks) > 0
            ]
            self.assertEqual(hooked, EXPECTED_ATTENTION_INDICES)
            self.assertEqual(len(hooked), 4)
            # mamba/mlp mixers must carry no hook.
            for i, block in enumerate(self.model.model.layers):
                if i not in EXPECTED_ATTENTION_INDICES:
                    self.assertEqual(len(block.mixer._forward_hooks), 0)
        # All hooks are removed once the context manager exits.
        self.assertTrue(
            all(len(b.mixer._forward_hooks) == 0 for b in self.model.model.layers)
        )


class TestNemotronHCompressionAttentionOnly(unittest.TestCase):
    """Each compressor must shrink only the 4 attention slots; mamba/mlp intact."""

    T = 80
    RATIO = 0.5

    def _assert_non_attention_untouched(self, cache, sentinels):
        for idx, slot in sentinels.items():
            self.assertIs(cache.layers[idx], slot)
            self.assertIsNone(cache.layers[idx].keys)
            self.assertIsNone(cache.layers[idx].values)

    def test_knorm_attention_only(self):
        model, cache, attn_idx, sentinels = _build_model_and_cache(T=self.T, seed=1)
        sketch = KnormSketch(compression_ratio=self.RATIO)
        _run_prefill_through_blocks(sketch, model, cache, self.T, model.config.hidden_size)

        expected = int(self.T * (1 - self.RATIO))
        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], expected)
            self.assertEqual(cache.layers[i].values.shape[2], expected)
        self._assert_non_attention_untouched(cache, sentinels)

    def test_ridge_attention_only(self):
        model, cache, attn_idx, sentinels = _build_model_and_cache(T=self.T, seed=2)
        sketch = RidgeSketch(compression_ratio=self.RATIO, sink_size=4, local_size=28)
        _run_prefill_through_blocks(sketch, model, cache, self.T, model.config.hidden_size)

        keep_total = int(self.T * (1 - self.RATIO))
        keep_mid = min(max(keep_total - 4 - 28, 0), self.T - 4 - 28)
        expected = 4 + keep_mid + 28
        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], expected)
            self.assertEqual(cache.layers[i].values.shape[2], expected)
        self._assert_non_attention_untouched(cache, sentinels)

    def test_snapkv_attention_only_no_rope(self):
        model, cache, attn_idx, sentinels = _build_model_and_cache(T=self.T, seed=3)
        sketch = SnapKVSketch(compression_ratio=self.RATIO, window_size=8, kernel_size=3)
        _run_prefill_through_blocks(sketch, model, cache, self.T, model.config.hidden_size)

        expected = int(self.T * (1 - self.RATIO))
        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], expected)
            self.assertEqual(cache.layers[i].values.shape[2], expected)
        self._assert_non_attention_untouched(cache, sentinels)

    def test_keydiff_attention_only(self):
        model, cache, attn_idx, sentinels = _build_model_and_cache(T=self.T, seed=11)
        sketch = KeyDiffSketch(compression_ratio=self.RATIO)
        _run_prefill_through_blocks(sketch, model, cache, self.T, model.config.hidden_size)

        expected = int(self.T * (1 - self.RATIO))
        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], expected)
            self.assertEqual(cache.layers[i].values.shape[2], expected)
        self._assert_non_attention_untouched(cache, sentinels)

    def test_expected_attention_attention_only_no_rope(self):
        # expected_attention rotates query stats by an averaged RoPE matrix; on
        # NemotronH (no rotary_emb) that reduces to identity. Must not crash.
        model, cache, attn_idx, sentinels = _build_model_and_cache(T=self.T, seed=12)
        sketch = ExpectedAttentionSketch(compression_ratio=self.RATIO, n_sink=4)
        _run_prefill_through_blocks(sketch, model, cache, self.T, model.config.hidden_size)

        expected = int(self.T * (1 - self.RATIO))
        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], expected)
            self.assertEqual(cache.layers[i].values.shape[2], expected)
        self._assert_non_attention_untouched(cache, sentinels)

    def test_compactor_attention_only_no_rope(self):
        # compactor's non-causal scorer rotates re-projected queries; on NemotronH
        # (no rotary_emb / no position_embeddings) it uses an identity rotation.
        # sketch_dimension < head_dim so the leverage Gram is full-rank on the fake.
        model, cache, attn_idx, sentinels = _build_model_and_cache(T=self.T, seed=13)
        sketch = CompactorSketch(
            compression_ratio=self.RATIO, sketch_dimension=4,
            sink_size_start=8, sink_size_end=4, chunk_size=256,
        )
        _run_prefill_through_blocks(sketch, model, cache, self.T, model.config.hidden_size)

        expected = int(self.T * (1 - self.RATIO))
        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], expected)
            self.assertEqual(cache.layers[i].values.shape[2], expected)
        self._assert_non_attention_untouched(cache, sentinels)

    def test_pyramidkv_attention_only_per_layer_budget(self):
        # PyramidKV's ragged cache is gated on flash_attention_2; mark it so the
        # real per-layer pyramid budget path runs (no actual flash kernel needed).
        model, cache, attn_idx, sentinels = _build_model_and_cache(
            T=self.T, seed=4, attn_implementation="flash_attention_2"
        )
        sketch = PyramidKVSketch(
            compression_ratio=self.RATIO, window_size=8, kernel_size=3, beta=20
        )
        budgets = {
            i: sketch.get_layer_budget(model.model.layers[i].mixer, self.T)
            for i in attn_idx
        }
        # Hard-pin one budget so a get_layer_budget formula regression is caught
        # (not just compared against the function's own output). T=80, ratio=0.5,
        # window=8, beta=20, 42 layers, layer_idx=12 -> round(72 - 12*64/41) = 53.
        self.assertEqual(budgets[12], 53)
        _run_prefill_through_blocks(sketch, model, cache, self.T, model.config.hidden_size)

        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], budgets[i])
            self.assertLess(cache.layers[i].keys.shape[2], self.T)
        # Pyramid: deeper attention layers keep fewer tokens (strictly decreasing).
        ordered = [budgets[i] for i in sorted(attn_idx)]
        self.assertEqual(ordered, sorted(ordered, reverse=True))
        self.assertGreater(len(set(ordered)), 1)
        self._assert_non_attention_untouched(cache, sentinels)


    def test_decode_step_is_noop(self):
        # A single-token decode forward (cache_position past q_len) must NOT prune
        # — these compressors fire on prefill only (default POST_PREFILL schedule).
        model, cache, attn_idx, sentinels = _build_model_and_cache(T=self.T, seed=9)
        sketch = KnormSketch(compression_ratio=self.RATIO)
        hidden = torch.randn(1, 1, model.config.hidden_size)
        with sketch(model):
            for block in model.model.layers:
                if block.block_type == "attention":
                    block.mixer(
                        hidden_states=hidden,
                        past_key_values=cache,
                        cache_position=torch.tensor([self.T]),
                    )
        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], self.T)  # untouched
        self._assert_non_attention_untouched(cache, sentinels)

    def test_explicit_decode_phase_overrides_prefill_heuristic(self):
        # Production drives the phase explicitly via set_phase("decode") rather
        # than relying on the cache_position heuristic. Feed a PREFILL-SHAPED
        # forward (cache_position 0..T-1, q_len=T) — which _is_decoding_step
        # would classify as PREFILL and prune — and assert the explicit decode
        # phase wins and suppresses compression. If set_phase were ignored this
        # forward would prune, so the test genuinely pins the explicit-phase path.
        model, cache, attn_idx, sentinels = _build_model_and_cache(T=self.T, seed=10)
        sketch = KnormSketch(compression_ratio=self.RATIO)
        sketch.set_phase("decode")
        hidden = torch.randn(1, self.T, model.config.hidden_size)
        with sketch(model):
            for block in model.model.layers:
                if block.block_type == "attention":
                    block.mixer(
                        hidden_states=hidden,
                        past_key_values=cache,
                        cache_position=torch.arange(self.T),
                    )
        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], self.T)  # untouched
        self._assert_non_attention_untouched(cache, sentinels)

    def test_ridge_rotate_queries_true_does_not_crash_on_no_rope(self):
        # NemotronH has no RoPE / rotary_emb. rotate_queries=True must fall back
        # gracefully (no position_embeddings, no rotary_emb) rather than raise.
        model, cache, attn_idx, sentinels = _build_model_and_cache(T=self.T, seed=5)
        sketch = RidgeSketch(
            compression_ratio=self.RATIO, sink_size=4, local_size=28,
            rotate_queries=True,
        )
        _run_prefill_through_blocks(sketch, model, cache, self.T, model.config.hidden_size)
        for i in attn_idx:
            self.assertLess(cache.layers[i].keys.shape[2], self.T)
            self.assertGreater(cache.layers[i].keys.shape[2], 0)
        self._assert_non_attention_untouched(cache, sentinels)

    def test_nemotron_isinstance_branch_compresses(self):
        # Exercise base.__call__'s isinstance(model, _NemotronH) gate (the symbol
        # is unimportable in this transformers, so patch it to the fake class).
        import eval_harness.kv_compression.base as base_mod
        from unittest.mock import patch

        model, cache, attn_idx, sentinels = _build_model_and_cache(T=self.T, seed=6)
        sketch = KnormSketch(compression_ratio=self.RATIO)
        with patch.object(base_mod, "_NemotronH", _FakeNemotronHForCausalLM):
            _run_prefill_through_blocks(sketch, model, cache, self.T, model.config.hidden_size)
        expected = int(self.T * (1 - self.RATIO))
        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], expected)
        self._assert_non_attention_untouched(cache, sentinels)


class TestNemotronHHybridCacheCheckpointRestore(unittest.TestCase):
    """HybridCacheAdapter must checkpoint/restore ONLY attention layers, and must
    NOT crash on the mamba/mlp slots whose ``keys``/``values`` are ``None`` (the
    real transformers 5.x unified-cache layout — pins the _can_slice_attention_kv
    None-guard)."""

    def test_checkpoint_restore_attention_only_no_crash_on_mamba(self):
        T = 40
        model, cache, attn_idx, sentinels = _build_model_and_cache(T=T, seed=7)
        adapter = create_cache_adapter(model)
        self.assertIsInstance(adapter, HybridCacheAdapter)

        checkpoint = adapter.clone_or_checkpoint_for_multi_question(cache)
        # Only the four attention layers are recorded; mamba/mlp (keys=None) skipped.
        self.assertEqual(sorted(checkpoint), attn_idx)
        self.assertTrue(all(v == T for v in checkpoint.values()))
        self.assertEqual(adapter.get_seq_length(cache), T)

        # Simulate a second question appending tokens to the attention caches.
        nkv, D = model.config.num_key_value_heads, model.config.head_dim
        for i in attn_idx:
            cache.layers[i].keys = torch.randn(1, nkv, 55, D)
            cache.layers[i].values = torch.randn(1, nkv, 55, D)

        # Must truncate attention layers back to T and NOT touch the None mamba slots.
        adapter.restore_after_question(cache, checkpoint)
        for i in attn_idx:
            self.assertEqual(cache.layers[i].keys.shape[2], T)
            self.assertEqual(cache.layers[i].values.shape[2], T)
        for idx, slot in sentinels.items():
            self.assertIs(cache.layers[idx], slot)
            self.assertIsNone(cache.layers[idx].keys)


class TestNemotronHKnormKeepsLowestNorm(unittest.TestCase):
    """Pin correctness of the no-self_attn path: knorm (score = -‖k‖) keeps the
    lowest-norm keys, on the attention layer reached through ``block.mixer``."""

    def test_knorm_keeps_lowest_norm_positions(self):
        torch.manual_seed(0)
        config = _make_nemotron_config()
        model = _FakeNemotronHForCausalLM(config)
        nkv, D = config.num_key_value_heads, config.head_dim
        T, ratio = 16, 0.5
        attn_idx = [i for i, b in enumerate(model.model.layers) if b.block_type == "attention"]
        first = attn_idx[0]

        # Key norm strictly increases with position; values encode position.
        scale = torch.arange(1, T + 1, dtype=torch.float32).view(1, 1, T, 1)
        keys = torch.ones(1, nkv, T, D) * scale
        values = torch.arange(T, dtype=torch.float32).view(1, 1, T, 1).expand(1, nkv, T, D).contiguous()

        layers = []
        for i, b in enumerate(model.model.layers):
            layers.append(_AttnCacheLayer(keys.clone(), values.clone())
                          if b.block_type == "attention" else _NonAttnCacheSlot())
        cache = _FakeHybridCache(layers)

        sketch = KnormSketch(compression_ratio=ratio)
        _run_prefill_through_blocks(sketch, model, cache, T, config.hidden_size)

        n_kept = int(T * (1 - ratio))
        kept = set(cache.layers[first].values[0, 0, :, 0].tolist())
        self.assertEqual(kept, set(range(0, n_kept)))


class TestNemotronHQuestionPrefill(unittest.TestCase):
    """generate_answer must feed the question token-by-token for Mamba models
    (their cached forward assumes q_len==1) and as one block for attention models."""

    def _capture_forward_qlens(self, layer_types, q_len):
        from eval_harness.research_pipeline import ResearchGenerationPipeline

        calls = []

        class _FakeModel:
            device = "cpu"
            config = SimpleNamespace(layers_block_type=layer_types)
            generation_config = SimpleNamespace(eos_token_id=999)

            def __call__(self, input_ids, past_key_values=None, position_ids=None, **kw):
                calls.append(int(input_ids.shape[1]))
                return SimpleNamespace(logits=torch.zeros(1, 1, 8))

        pipe = object.__new__(ResearchGenerationPipeline)
        pipe.model = _FakeModel()
        pipe.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "x")
        q = torch.zeros(1, q_len, dtype=torch.long)
        pipe.generate_answer(q, cache=None, context_length=100, max_new_tokens=1)
        return calls

    def test_mamba_feeds_question_token_by_token(self):
        calls = self._capture_forward_qlens(["mamba", "attention", "mlp"], q_len=5)
        self.assertEqual(calls, [1, 1, 1, 1, 1])  # each question token a separate q_len=1 forward

    def test_attention_feeds_question_as_single_block(self):
        calls = self._capture_forward_qlens(["full_attention", "full_attention"], q_len=5)
        self.assertEqual(calls, [5])  # one block forward of the whole question


if __name__ == "__main__":
    unittest.main()
