"""RareKV: rarity-based KV eviction via locality-sensitive hashing."""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
from torch import nn

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor
from eval_harness.kernels.rarekv_lsh import (
    KERNEL_MAX_PLANES, collision_sums_lmajor, lsh_buckets, planes_to_planes_t,
    resolve_mode,
)

logger = logging.getLogger(__name__)


@register_kv_compressor("rarekv", aliases=["rare_kv"])
@dataclass
class RareKVSketch(ScorerKVCompressor):
    r"""RareKV — keep the keys that are *rare* in LSH bucket space.

    Same family as ``keydiff`` (key-similarity, query-free, value-free scoring),
    but instead of measuring similarity to a single global mean-key anchor it
    measures how crowded each key's neighbourhood is, estimated with random
    hyperplane LSH.

    Every key is hashed into ``L`` independent tables of ``R = 2**P`` buckets by
    ``P`` signed random projections per table. A key that repeatedly shares a
    bucket with many other keys is *redundant* — the cache already holds its
    direction — while a key that lands alone is *rare* and carries information
    no surviving key would preserve. Scores are the **inverse collision
    density**:

    .. math::
        s_i^{\mathrm{ICD}} = \left(\epsilon + \frac{1}{L}\sum_{\ell=1}^{L}
        \frac{C_{\ell,h_\ell(\mathbf{k}_i)} - 1}{N - 1}\right)^{-\alpha}

    where :math:`C_{\ell,b}` is the number of keys in bucket ``b`` of table
    ``\ell`` and ``N`` is the sequence length. The ``-1`` removes the key's own
    contribution and the ``N-1`` normalises to a fraction of the sequence, so
    the density is in ``[0, 1]``: a key alone in its bucket in every table
    scores :math:`\epsilon^{-\alpha}` (maximal), a key colliding with the entire
    sequence scores :math:`(\epsilon + 1)^{-\alpha}` (minimal).

    The final score multiplies by the value norm, so a rare key whose value
    vector carries little magnitude is not preferred over a common key with a
    large one:

    .. math:: s_i = s_i^{\mathrm{ICD}} \cdot \lVert \mathbf{v}_i \rVert^{\gamma}

    The base class then keeps the top ``int(T * (1 - compression_ratio))`` per
    KV head.

    Implementation notes
    --------------------
    - **Fully vectorised, no Python loops.** The ``L`` tables are folded into a
      single ``[D, L*P]`` GEMM; the sign test, the bit-pack and the transpose into
      the collision kernel's L-major layout are ONE fused CUDA kernel
      (``lsh_mode="pack"``, the default; ``kernels/csrc/rarekv_pack.cu``); the
      collision histogram over all ``(batch, head, table)`` groups is a *single*
      privatised histogram (or, on the reference path, a single ``scatter_add_``
      into a flattened ``(group, bucket)`` index space).
    - **Reproducible and sync-free.** Counting uses integer ``scatter_add_``:
      CUDA atomics reorder additions, but integer addition is exact, so the
      histogram is bit-identical run to run. (For *this* kernel an fp32
      accumulator would also be order-independent — the src is all ones, and
      sums below 2**24 are exact — so int32 is chosen for half the traffic and
      no 2**24 ceiling, not because float would be wrong.) Nothing calls
      ``.item()``/``.max()``, so the scorer never forces a host sync mid-prefill,
      unlike ``snapkv`` (``scores.max().item()`` at ``snapkv_sketch.py:197``).

      **The one op that is not pinned is the projection GEMM.** ``torch.mm`` is on
      PyTorch's CUDA-nondeterministic list without ``CUBLAS_WORKSPACE_CONFIG``,
      and ``EvalConfig.deterministic`` defaults to False. Measured impact is nil
      (fp32 sign-flip rate <= 8e-8; the retained set never changed across 5
      configs). That is exactly why ``lsh_mode="pack"`` leaves the GEMM alone and
      fuses only what comes after it: everything downstream of the sign test is
      integer, so Tier 1 is **unconditionally bit-identical** to the pure-torch
      path -- ``torch.equal``, not ``allclose``.

      **The live float exposure is the reduced-precision-reduction flag**, not
      TF32. TF32 cannot touch the production bf16 GEMM at all (bf16's 8
      significand bits sit strictly inside tf32's 10); the "~1000x flip rate"
      figure describes an **fp32** K cache, which only the fp32 fallback path
      produces. ``NVIDIA_TF32_OVERRIDE=0`` is still pinned in every launcher --
      it guards that fallback and costs nothing -- but the flag that can move the
      shipped bf16 path is
      ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction``,
      which defaults True and is unpinned: were cuBLAS to select a split-k
      algorithm, today's own path would flip ~4.4e-04 of signs.
      ``profiling/environment.py`` records all of these; nothing pins them, and
      pinning them would silently move every already-published rarekv number.

      **``lsh_mode="fused"`` (Tier 2) is a different contract.** It removes the
      projection entirely by doing the GEMM on tensor cores inside the packing
      kernel, so it cannot be bit-identical to cuBLAS. It is bit-identical to a
      *declared reference* (fp32 accumulation over strictly ascending 16-wide k
      chunks on sm_90a -- see ``kernels/csrc/rarekv_fused.cu``), and its
      divergence from the shipped path is measured and published by
      ``scripts/bench_rarekv_kernel.py``, not asserted away. It is opt-in.
    - **Planes never touch the global RNG.** They are drawn from an explicitly
      seeded CPU generator and moved to device, so they are identical on any GPU
      or driver, and reproducible across processes. They are cached per
      ``(layer, head_dim)`` so the GEMM operand is built once per layer.
    - **No RoPE, no queries, no attention weights**, so this composes with
      mixed-attention hybrids (NemotronH applies no RoPE) out of the box and has
      no ``attn_implementation`` requirement.

    Peak transient memory, per element of ``B * H_kv * T * L * P``:

    ==================  ==========  ================================
    ``lsh_mode``        bytes/elem  ``T=128K, H_kv=8, L=60, P=10``
    ==================  ==========  ================================
    ``"torch"``         5           2.93 GiB
    ``"pack"``          2 + b/P     1.29 GiB
    ``"fused"``         b/P         0.12 GiB
    ==================  ==========  ================================

    where ``b`` is the bucket-id width (1 byte for ``P <= 8``, 2 for
    ``9 <= P <= 16``). The torch figure is the projection in the cache dtype
    (2 B/elem) plus the int32 bit-pack (4 B/elem) with the projection freed
    first, so the two never peak together; ``"pack"`` keeps only the projection
    plus the packed ids, and ``"fused"`` keeps only the ids. Writing the torch
    form the obvious way — keeping the projection alive across an int64 cast and
    an int64 product — costs **20 B/elem, i.e. ~11.7 GiB**, which is why the
    ordering in :func:`~eval_harness.kernels.rarekv_lsh.buckets_from_proj_torch`
    is deliberate and should not be "simplified". ``gemm_chunk_rows`` bounds the
    projection term of ``"pack"`` independently of ``T``.

    None of this is an OOM story: the prefill peak is set in the MLP
    (~120 kB/token measured), and ``rarekv`` p6l80 raised ``peak_alloc_prefill``
    by only 7.87 MB over ``keydiff`` despite a 2.51 GB transient. What the
    reduction buys is **headroom returned** (a few percent to ~18% of the prefill
    peak, more on MHA models where ``H_kv=32``), plus the removal of the ceiling
    that kept ``L`` and ``P`` small.

    ``__post_init__`` rejects configurations whose bucket table would exceed
    ``max_bucket_slots``, and configurations whose ``eps**-alpha`` would overflow
    fp32.

    Caveats worth knowing before reading results
    --------------------------------------------
    - **The selection is strongly seed-dependent.** Two seeds at ratio 0.9 agree
      on only ~26-36% of retained tokens at ``(P,L) = (10,60)/(8,50)`` on
      iid-Gaussian keys (chance is 10%), and with ``value_norm_power=0`` the
      rarity term alone lands *at* the chance floor. Most of the agreement at the
      default ``gamma=1`` comes from the value norm. **Report any accuracy number
      from this method over a seed sweep with a variance band**; a single-seed
      result is not distinguishable from draw luck. LSH only finds structure when
      the keys genuinely cluster — on anisotropic keys the rarity term does carry
      signal (up to 68% overlap at high P).
    - **Ties are a large-P hazard, not a small-P one.** Density is a function of
      an integer collision count, so small ``P`` gives large counts with a wide
      integer spread (many distinct scores) and large ``P`` gives tiny counts
      with few. At ``gamma=0``, ``(P,L) = (10,60)`` leaves 99.7% of tokens tied
      and a 94-way tie at the cutoff, decided by ``topk``'s unspecified order.
      The default ``gamma=1`` breaks ties: cutoff multiplicity is ~1.0.
    - **``L`` matters more than ``P`` for quality.** ``L=1`` is near-random;
      ``L>=8`` is the usable region. ``P`` degrades at both ends — ``P<=2`` makes
      the density nearly constant, ``P>=16`` saturates into ties.
    - **Scores are computed on RoPE-rotated keys** (that is what the cache holds).
      Random-hyperplane LSH is not rotation-invariant, so this measures
      redundancy of *content-at-similar-position*, not content alone: on keys
      built from 8 distinct contents the measured density under-estimates true
      redundancy ~10x. ``keydiff`` has the same exposure.
    - Reproducibility is exact for a fixed seed, torch build and CPU ISA. CPU
      ``randn`` differs by ~1 ulp across ATen's vectorised dispatch, which
      perturbs scores but left the retained set identical in all configs tested.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove.
    n_planes : int, default=8
        ``P`` — signed random projections per table; each table has
        ``R = 2**P`` buckets. Larger ``P`` means finer buckets, so fewer
        collisions and a sparser, higher-variance density estimate.
    n_tables : int, default=50
        ``L`` — independent hash tables averaged over. Larger ``L`` reduces the
        variance of the density estimate, linearly in cost. This is the knob that
        drives quality: ``L=1`` selects near-randomly, ``L>=8`` is usable.
    alpha : float, default=1.0
        Exponent on the inverse density. Larger values sharpen the preference
        for rare keys; ``alpha=0`` reduces the ICD term to a constant, leaving
        pure value-norm scoring.
    eps : float, default=1e-6
        Numerical floor, and the cap on the score of a collision-free key
        (``eps**-alpha``). **It rarely binds in practice**: at T >= 8K the mean
        density is ~1e-2..1e-1, thousands of times larger than the default eps,
        and no key was ever alone in all L tables. So the useful dynamic-range
        knob is ``alpha`` (the ICD spread scales as spread**alpha), not eps.
        Concretely, at the default config on unstructured keys the ICD term
        spans only ~1.6x while value norms span ~7x, so scoring is dominated by
        ``||v||`` unless the keys actually cluster. ``__post_init__`` rejects
        combinations whose ``eps**-alpha`` overflows fp32.
    value_norm_power : float, default=1.0
        ``gamma``. Set to ``0.0`` to score by rarity alone.
    seed : int, default=42
        Seeds the hyperplanes. With ``per_layer_planes`` the layer index is
        added, so layers hash independently.
    per_layer_planes : bool, default=True
        Draw a fresh plane set per layer (seeded ``seed + layer_idx``). Set
        ``False`` to share one plane set across all layers.
    max_bucket_slots : int, default=2**26
        Guard on ``B * H_kv * L * 2**P``, the size of the collision histogram.
    use_triton : bool, default=True
        Allow the compiled collision paths (CUDA extension, then Triton). False
        forces ``collision_sums_torch``. Kept under its historical name; the
        CUDA extension is preferred over Triton wherever it builds.
    lsh_mode : {"pack", "fused", "torch"}, default="pack"
        How bucket ids are produced. ``"pack"`` (Tier 1) keeps the cuBLAS GEMM
        and fuses the sign/pack/transpose into one kernel — **bit-identical** to
        ``"torch"``, which is the reference. ``"fused"`` (Tier 2) also fuses the
        GEMM onto tensor cores, removing the projection entirely; it is
        bit-identical to a declared reference but NOT to cuBLAS, so it is opt-in
        and its divergence must be published alongside any result that used it.
        ``"torch"`` is the ATen sequence, always available. The env var
        ``PRISM_RAREKV_LSH`` overrides this per run in BOTH directions (``score``
        resolves it before deciding whether to build the fused operand), and
        ``PRISM_RAREKV_CUDA=0`` forces ``"torch"``. ``"fused"`` silently degrades to ``"pack"`` (never
        straight to ``"torch"``) when its preconditions do not hold; the resolved
        path is logged once per layer and recorded in ``lsh_paths``.
    block_m : int, default=0
        Rows per CTA in the packing kernel; ``0`` uses the kernel default (128).
        An autotune knob — 128 or 256. Cannot change the answer, but it does
        double the kernel's shared-memory request (``block_m * rk_nbs(L*P)``
        bytes): at ``L*P >= 1289`` only ``block_m=128`` fits a 48 KB device, and
        past the device's opt-in limit ``lsh_buckets`` routes to the (identical)
        torch sequence rather than failing the launch. Irrelevant on the target
        grid, where ``L*P <= 900``.
    gemm_chunk_rows : int, default=0
        ``0`` (default) issues ONE cuBLAS call for the projection. A positive
        value chunks it along M so the projection never materialises in full,
        bounding the ``"pack"`` transient at ``gemm_chunk_rows * L * P * 2``
        bytes. Chunking along M cannot change any output element's reduction
        order over ``D``; the only exposure is cuBLAS re-selecting an algorithm
        at the smaller M, which is an empirical question — hence this ships off
        and is gated by the benchmark's ``gemm_chunk_exact`` column. No effect on
        ``"fused"`` (which has no projection) or ``"torch"``.
    """

    n_planes: int = 8
    n_tables: int = 50
    alpha: float = 1.0
    eps: float = 1e-6
    value_norm_power: float = 1.0
    seed: int = 42
    per_layer_planes: bool = True
    max_bucket_slots: int = 1 << 26
    use_triton: bool = True
    lsh_mode: str = "pack"          # "pack" (Tier 1, bit-identical) | "fused" | "torch"
    block_m: int = 0                # 0 = kernel default (128); autotune knob, 128 | 256
    gemm_chunk_rows: int = 0        # 0 = single cuBLAS call (Tier 1c is opt-in)
    _plane_cache: Dict[Tuple, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _lsh_paths: set = field(default_factory=set, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_planes < 1 or self.n_planes > 30:
            raise ValueError(f"n_planes (P) must be in [1, 30], got {self.n_planes}")
        if self.n_tables < 1:
            raise ValueError(f"n_tables (L) must be >= 1, got {self.n_tables}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        # A collision-free key scores eps**-alpha. If that overflows fp32 the score
        # becomes +inf, and inf * 0 (a zero-norm value row) becomes NaN -- which
        # torch.topk ranks ABOVE +inf, so the least informative token would be
        # retained first. Refuse the configuration instead of producing that.
        if self.alpha > 0 and -self.alpha * math.log10(self.eps) > 38.0:
            raise ValueError(
                f"eps**-alpha = {self.eps}**-{self.alpha} overflows float32 "
                f"(10**{-self.alpha * math.log10(self.eps):.1f} > 10**38.5). Raise eps or "
                f"lower alpha; scores would saturate to +inf and NaN-rank first in topk.")
        # The B/H-independent half of the histogram guard, checked at construction
        # rather than on layer 0's first prefill hook.
        if self.n_tables * (1 << self.n_planes) > self.max_bucket_slots:
            raise ValueError(
                f"n_tables * 2**n_planes = {self.n_tables * (1 << self.n_planes):,} already "
                f"exceeds max_bucket_slots={self.max_bucket_slots:,} before the "
                f"batch/head factor. Lower n_planes (P) or n_tables (L).")
        if self.lsh_mode not in ("pack", "fused", "torch"):
            raise ValueError(
                f"lsh_mode must be one of pack|fused|torch, got {self.lsh_mode!r}")
        if self.block_m not in (0, 128, 256):
            raise ValueError(f"block_m must be 0, 128 or 256, got {self.block_m}")
        if self.gemm_chunk_rows < 0:
            raise ValueError(f"gemm_chunk_rows must be >= 0, got {self.gemm_chunk_rows}")
        # The kernel bucket ids are uint8/int16, so the packing kernels cap at
        # P=16 while the validator above allows P<=30. `pack` degrades to the
        # (identical) torch sequence there, which is a speed loss and nothing
        # else -- but `fused` changes float behaviour, so asking for it at P>16
        # is a configuration error, not something to silently reinterpret.
        if self.lsh_mode == "fused" and self.n_planes > KERNEL_MAX_PLANES:
            raise ValueError(
                f"lsh_mode='fused' needs n_planes <= {KERNEL_MAX_PLANES} (the kernel emits "
                f"uint8/int16 bucket ids); got n_planes={self.n_planes}. Use lsh_mode='pack'.")
        self._plane_cache = {}
        self._lsh_paths = set()

    @property
    def n_buckets(self) -> int:
        """``R = 2**P`` buckets per table."""
        return 1 << self.n_planes

    def post_init_from_model(self, model) -> None:
        """Draw every layer's planes ONCE, here, off the prefill path.

        ``planes.to(device)`` is a blocking H2D copy from pageable memory. Drawn
        lazily inside the hook with ``per_layer_planes`` that is one full stream
        sync *per layer*, draining the GPU queue mid-prefill, plus ~0.3 ms of CPU
        normals each. Building them all on CPU and moving them in one transfer
        costs a single sync at install time. Falls back to lazy creation if the
        config cannot be read (the result is identical either way).
        """
        try:
            cfg = (model.config.get_text_config()
                   if hasattr(model.config, "get_text_config") else model.config)
            n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
            head_dim = int(getattr(cfg, "head_dim", 0) or 0) or (
                int(cfg.hidden_size) // int(cfg.num_attention_heads))
            param = next(model.parameters())
            device, dtype = param.device, param.dtype
        except Exception:                                          # noqa: BLE001
            return
        if not n_layers or not head_dim:
            return
        tags = list(range(n_layers)) if self.per_layer_planes else [0]
        stacked = torch.stack([self._draw(t, head_dim) for t in tags]).to(device, dtype)
        for i, tag in enumerate(tags):                             # setup only, not the hot path
            self._plane_cache[self._cache_key(tag, head_dim, device, dtype)] = stacked[i]

    def _cache_key(self, tag, head_dim, device, dtype):
        # (L, P) are in the key: the plane matrix is [head_dim, L*P] and the
        # consumer reshapes it as (L, P), so a hit across a different (L, P)
        # would silently regroup columns into the wrong tables. dtype is in the
        # key because the GEMM runs in the cache dtype.
        # `seed` is in the key too: a sweep driver that mutates seed on a warm
        # instance would otherwise silently get the previous seed's planes back
        # and report identical results for every seed.
        return (tag if self.per_layer_planes else -1, head_dim, self.n_tables,
                self.n_planes, self.seed, str(device), str(dtype))

    def _draw(self, layer_idx: int, head_dim: int) -> torch.Tensor:
        """One CPU-generated plane matrix; never advances the global RNG."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + (layer_idx if self.per_layer_planes else 0))
        # device="cpu" explicitly: torch.randn otherwise obeys
        # torch.set_default_device, which would break the CPU-generator contract.
        return torch.randn(head_dim, self.n_tables * self.n_planes, generator=gen,
                           dtype=torch.float32, device="cpu")

    def _planes(self, module: nn.Module, head_dim: int, device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:
        """``[head_dim, L*P]`` hyperplanes, cached per (layer, head_dim, L, P, device, dtype)."""
        layer_idx = int(getattr(module, "layer_idx", 0) or 0)
        cache_key = self._cache_key(layer_idx, head_dim, device, dtype)
        cached = self._plane_cache.get(cache_key)
        if cached is None:
            cached = self._draw(layer_idx, head_dim).to(device, dtype)
            self._plane_cache[cache_key] = cached
        return cached

    def _powers(self, device: torch.device) -> torch.Tensor:
        """``[1, 2, 4, ...]`` for bit packing, cached.

        No longer used by :meth:`score` -- the packing kernel ORs the sign bits
        directly, and the torch reference builds its own powers vector inside
        :func:`~eval_harness.kernels.rarekv_lsh.buckets_from_proj_torch`. Kept
        because ``scripts/bench_rarekv_dense.py`` and
        ``scripts/bench_blockrare_dense.py`` reproduce the pre-kernel sequence
        by hand and need the identical operand.
        """
        key = ("powers", self.n_planes, str(device))
        cached = self._plane_cache.get(key)
        if cached is None:
            cached = (2 ** torch.arange(self.n_planes, dtype=torch.int64)).to(
                device=device, dtype=torch.int32)
            self._plane_cache[key] = cached
        return cached

    def _planes_t(self, module: nn.Module, head_dim: int, device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
        """``[ceil(L/8)*8*P, head_dim]`` transposed, zero-padded planes for Tier 2.

        The fused kernel's B operand needs ``B[k][n] = planesT[col][k]`` with ``k``
        contiguous, and it sweeps 8 tables per column tile, so the row count is
        rounded up to a whole number of tiles. Built lazily on the first ``fused``
        call and cached under the same identity tuple as :meth:`_planes`, so a
        seed or (L, P) change cannot silently reuse the previous operand.
        """
        layer_idx = int(getattr(module, "layer_idx", 0) or 0)
        key = ("planes_t",) + self._cache_key(layer_idx, head_dim, device, dtype)
        cached = self._plane_cache.get(key)
        if cached is None:
            cached = planes_to_planes_t(
                self._planes(module, head_dim, device, dtype), self.n_tables, self.n_planes)
            self._plane_cache[key] = cached
        return cached

    def _log_path_once(self, module: nn.Module, path: str) -> None:
        """Record and log the resolved LSH path the first time each one is seen.

        A run that silently degraded from ``fused`` to ``pack`` is otherwise
        indistinguishable from one that did not, and its divergence table would be
        meaningless. ``lsh_paths`` is the machine-readable form.
        """
        if path in self._lsh_paths:
            return
        self._lsh_paths.add(path)
        logger.info("rarekv LSH path %r (requested %r, layer %s)", path, self.lsh_mode,
                    getattr(module, "layer_idx", None))

    @property
    def lsh_paths(self) -> Tuple[str, ...]:
        """Every LSH path this instance actually took, sorted. Empty before the first hook."""
        return tuple(sorted(self._lsh_paths))

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        del hidden_states, attentions, kwargs
        B, H, T, D = keys.shape
        L, P, R = self.n_tables, self.n_planes, self.n_buckets
        device = keys.device

        slots = B * H * L * R
        if slots > self.max_bucket_slots:
            raise ValueError(
                f"rarekv collision table would need B*H*L*2**P = {slots:,} slots "
                f"(> max_bucket_slots={self.max_bucket_slots:,}). Lower n_planes (P={P}) "
                f"or n_tables (L={L})."
            )

        # 1-3. Signed random projections, then the bucket ids, in the layout the
        #    collision kernel wants ([B*H, L, T]).
        #
        #    On the default `pack` path the GEMM is UNCHANGED -- one call for all L
        #    tables at once, in the cache dtype. bf16 in / fp32 accumulate (cuBLAS)
        #    flips the sign of a near-zero projection for ~5e-4 of entries versus an
        #    fp32 GEMM, which changes the retained set by ~2%: roughly 25x LESS than
        #    changing `seed` does (~56%). An fp32 cast would cost a full fp32 copy of
        #    K and a ~10x slower GEMM for a perturbation far below the method's own
        #    hash variance. Leaving the one unpinnable op alone is precisely what
        #    makes `pack` bit-identical to `torch`.
        #
        #    The packing kernel then reads `proj` ONCE and writes uint8/int16 ids
        #    straight into the L-major layout: the bool `sign`, the int32 `packed`,
        #    its in-place multiply, the int32 `bucket` and the transposing clone are
        #    all gone (11.8 -> 2.8 GB/layer at T=128K, L=70, P=6; transient
        #    5 -> 2 + b/P bytes per element). `fused` removes `proj` too -- see the
        #    class docstring for why that is a different exactness contract.
        planes = self._planes(module, D, device, keys.dtype)
        # Resolve the env override HERE, not inside `lsh_buckets`: the fused path
        # needs the transposed plane operand, and building it from `self.lsh_mode`
        # alone made `PRISM_RAREKV_LSH=fused` unable to UPGRADE a config -- it
        # resolved to "fused", failed the `planes_t is not None` precondition and
        # fell back to `pack` with nothing above INFO to say so. A divergence study
        # driven by the documented env var would then have reported a Tier-2 flip
        # rate of exactly zero for a path that never ran.
        want = resolve_mode(self.lsh_mode)
        planes_t = (self._planes_t(module, D, device, keys.dtype)
                    if want == "fused" else None)
        bucket, path = lsh_buckets(keys, planes, planes_t, L, P, mode=self.lsh_mode,
                                   block_m=self.block_m,
                                   gemm_chunk_rows=self.gemm_chunk_rows)
        self._log_path_once(module, path)

        # 4. Collision counts, then sum over L. Both are integer, so both are
        #    order-independent -- which is what lets every collision path be
        #    BIT-IDENTICAL rather than merely close.
        #
        #    CUDA path: a shared-memory privatised histogram plus a thread-per-key
        #    gather on the L-major layout, so both bucket reads are fully coalesced.
        #    torch path: one 2-D scatter_add_ with the table offset folded into the
        #    [B*H, T, L] layout (no loop over tables). Triton path: a block-privatised
        #    histogram, which cuts global atomics per (b,h,l) from T to (T/BLOCK)*R.
        #    That reduction factor is BLOCK/R, so it is taken only for small R -- see
        #    kernels/rarekv_lsh.should_use_triton. Measured on H200: 7.8x at R=8,
        #    1.3x at R=256, 0.8x (slower) at R=1024.
        csum = collision_sums_lmajor(bucket, R, prefer_kernels=self.use_triton)
        del bucket

        # mean_l[(C-1)/(N-1)] == (mean_l C - 1)/(N-1), so reduce over L FIRST and
        # scale the small result: the naive order allocates three full
        # [B*H*T*L] fp32 temporaries.
        #
        # `csum` is int32 all the way here and is converted LAST, which matters: the
        # cast exceeds fp32's exact-integer ceiling once csum > 2**24, i.e. at
        # T > 2**24/L (167,772 at L=100). That is pre-existing and identical on every
        # path (so it cannot break bit-identity), but it is a trap for any future
        # fusion -- accumulate the collision counts in int32 and convert once.
        density = (csum.float() / L - 1.0) / float(max(T - 1, 1))
        scores = (self.eps + density).pow(-self.alpha).view(B, H, T)

        # 5. Weight by value norm. NOTE: passing dtype=float32 here would NOT give
        #    "fp32 accumulation for free" -- ATen's make_reduction does a full
        #    `values.to(fp32)` copy first. Without it the accumulator is already
        #    fp32 (opmath_type<bf16> == float); only the output is rounded.
        if self.value_norm_power != 0.0:
            v_norm = torch.linalg.vector_norm(values, dim=-1).float()
            if self.value_norm_power != 1.0:
                v_norm = v_norm.pow(self.value_norm_power)
            scores = scores * v_norm
        return scores
