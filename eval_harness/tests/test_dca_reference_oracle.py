"""DCA reference-transcription oracle (permanent in-suite parity proof).

A verbatim transcription of the ChunkLlama reference ``chunkllama_attn_replace.py``
(https://github.com/HKUNLP/ChunkLlama)
(ChunkLlamaRotaryEmbedding tables, prefill chunk loop, decode branch) is run
against ``DCAMethod._make_dca_forward`` end-to-end (q/k/v/o projections, cache,
cyclic key storage).  Two documented substitutions (both deliberate port
deviations, see dca.py):

* ``do_flash_attn`` -> ``attention_with_lse`` (CPU; equal-length causal
  slices, so bottom-right == top-left; flash is the reference's de facto
  semantics),
* LSE-merge weights kept fp32 (reference downcasts to bf16).

Pins, with mscale ACTIVE (S > pretraining_length) and scaling_factor=2.0 (PI):
multi-chunk prefill outputs, bitwise cyclic cached keys, q_len==1 decode
steps, and multi-token decode blocks straddling chunk boundaries.

Adapted from the verification workflow's independent audit script.
"""
import math
import unittest

import torch
from torch import nn

from eval_harness.kernels.dca_flash import attention_with_lse
from eval_harness.prefill_methods.dca import DCAMethod

torch.manual_seed(0)

# ---------------- config ----------------
CHUNK_SIZE = 12
LOCAL_WINDOW = 4
CHUNK_LEN = CHUNK_SIZE - LOCAL_WINDOW  # 8
PRETRAIN = 16            # small so mscale is ACTIVE for S=30
SCALING_FACTOR = 2.0     # PI active
MSCALE_COEFF = 0.05
MAX_NEW_TOKENS = 512
D, NH, NKV, HID = 8, 4, 2, 32
B = 1


def ref_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return MSCALE_COEFF * math.log(scale) + 1.0


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RefRotary(nn.Module):
    """Verbatim transcription of ChunkLlamaRotaryEmbedding (attn_replace)."""

    def __init__(self, dim, max_position_embeddings, base=10000, scaling_factor=1.0):
        super().__init__()
        self.max_seq_len = max_position_embeddings
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._set_cos_sin_cache(seq_len=self.max_seq_len, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, dtype):
        scale = seq_len / self.max_position_embeddings
        mscale = ref_get_mscale(scale)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.inv_freq = inv_freq
        chunk_len = CHUNK_SIZE - LOCAL_WINDOW
        q_t = torch.arange(chunk_len, dtype=inv_freq.dtype) / self.scaling_factor
        qc_t = (torch.arange(chunk_len, dtype=inv_freq.dtype) + chunk_len).clamp(
            max=CHUNK_SIZE) / self.scaling_factor
        k_t = (torch.arange(seq_len + MAX_NEW_TOKENS, dtype=inv_freq.dtype) % chunk_len
               ) / self.scaling_factor
        q_freqs = torch.outer(q_t, inv_freq)
        qc_freqs = torch.outer(qc_t, inv_freq)
        k_freqs = torch.outer(k_t, inv_freq)
        q_emb = torch.cat((q_freqs, q_freqs), dim=-1)
        qc_emb = torch.cat((qc_freqs, qc_freqs), dim=-1)
        k_emb = torch.cat((k_freqs, k_freqs), dim=-1)
        self.q_cos_cached = q_emb.cos().to(dtype) * mscale
        self.q_sin_cached = q_emb.sin().to(dtype) * mscale
        self.qc_cos_cached = qc_emb.cos().to(dtype) * mscale
        self.qc_sin_cached = qc_emb.sin().to(dtype) * mscale
        self.k_cos_cached = k_emb.cos().to(dtype) * mscale
        self.k_sin_cached = k_emb.sin().to(dtype) * mscale

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=torch.float32)
            self.max_seq_len = seq_len
        return (self.q_cos_cached[:seq_len].to(dtype=x.dtype),
                self.q_sin_cached[:seq_len].to(dtype=x.dtype),
                self.qc_cos_cached[:seq_len].to(dtype=x.dtype),
                self.qc_sin_cached[:seq_len].to(dtype=x.dtype),
                self.k_cos_cached[:seq_len].to(dtype=x.dtype),
                self.k_sin_cached[:seq_len].to(dtype=x.dtype))


def ref_apply_rope(x, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def ref_merge_single(softmax_lse, attn_outputs):
    softmax_lse = softmax_lse.to(torch.float32)
    m = torch.max(softmax_lse, dim=0).values
    s = torch.exp(softmax_lse - m.unsqueeze(0))
    s = s / s.sum(dim=0)            # fp32 (documented port deviation from bf16)
    return (attn_outputs * s.unsqueeze(-1)).sum(dim=0)


def ref_merge(flash_results):
    out_all = [flash_results[0][0]]
    for per_chunk in flash_results[1:]:
        outs = torch.stack([t[0] for t in per_chunk])
        lses = torch.stack([t[1] for t in per_chunk])
        out_all.append(ref_merge_single(lses, outs))
    return torch.cat(out_all, dim=2)


def do_attn(q, k, v, causal=True):
    return attention_with_lse(q, k, v, causal=causal)  # scale defaults 1/sqrt(D)


class FakeAttn(nn.Module):
    def __init__(self, seed=7):
        super().__init__()
        torch.manual_seed(seed)
        self.num_heads, self.num_key_value_heads, self.head_dim = NH, NKV, D
        self.num_key_value_groups = NH // NKV
        self.hidden_size = NH * D
        self.scaling = 1.0 / math.sqrt(D)
        self.layer_idx = 0
        self.config = None
        self.q_proj = nn.Linear(HID, NH * D, bias=False)
        self.k_proj = nn.Linear(HID, NKV * D, bias=False)
        self.v_proj = nn.Linear(HID, NKV * D, bias=False)
        self.o_proj = nn.Linear(NH * D, HID, bias=False)


def ref_forward(attn, rotary, hidden_states, position_ids, kv_cache, attention_mask=None):
    """Verbatim transcription of chunkllama_attn_replace.forward."""
    bsz, q_len, _ = hidden_states.size()
    chunk_len = CHUNK_SIZE - LOCAL_WINDOW

    query_states = attn.q_proj(hidden_states).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = attn.k_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2] + kv_cache["len"]
    q_seq_len = query_states.shape[-2]
    has_kv_cache = q_seq_len != kv_seq_len

    q_cos, q_sin, qc_cos, qc_sin, k_cos, k_sin = rotary(value_states, seq_len=kv_seq_len)
    key_states = ref_apply_rope(key_states, k_cos, k_sin, position_ids)
    position_ids = position_ids % chunk_len

    # cache update (DynamicCache-style concat)
    if kv_cache["k"] is None:
        kv_cache["k"], kv_cache["v"] = key_states, value_states
    else:
        kv_cache["k"] = torch.cat([kv_cache["k"], key_states], dim=2)
        kv_cache["v"] = torch.cat([kv_cache["v"], value_states], dim=2)
    kv_cache["len"] = kv_cache["k"].shape[2]
    key_states = repeat_kv(kv_cache["k"], attn.num_key_value_groups)
    value_states = repeat_kv(kv_cache["v"], attn.num_key_value_groups)

    if not has_kv_cache:
        flash_results = []
        q_states_intra = ref_apply_rope(query_states[:, :, :chunk_len, :], q_cos, q_sin,
                                        position_ids[:, :chunk_len])
        k_states_prev = key_states[:, :, :chunk_len, :]
        v_states_prev = value_states[:, :, :chunk_len, :]
        flash_results.append(do_attn(q_states_intra, k_states_prev, v_states_prev))
        remain_len = kv_seq_len - chunk_len
        while remain_len > 0:
            flash_per_chunk = []
            begin = kv_seq_len - remain_len
            curr_chunk_len = min(chunk_len, remain_len)
            end = begin + curr_chunk_len
            q_states_intra = ref_apply_rope(query_states[:, :, begin:end, :], q_cos, q_sin,
                                            position_ids[:, begin:end])
            k_states_intra = key_states[:, :, begin:end, :]
            v_states_intra = value_states[:, :, begin:end, :]
            flash_per_chunk.append(do_attn(q_states_intra, k_states_intra, v_states_intra))
            q_states_succ = ref_apply_rope(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                           position_ids[:, begin:end])
            flash_per_chunk.append(do_attn(q_states_succ, k_states_prev, v_states_prev, False))
            if begin - (k_states_prev.size(-2)) > 0:
                prev_len = k_states_prev.size(-2)
                q_states_inter = ref_apply_rope(
                    query_states[:, :, begin:end, :], qc_cos, qc_sin,
                    position_ids[:, chunk_len - 1][:, None].repeat(1, curr_chunk_len))
                flash_per_chunk.append(do_attn(q_states_inter,
                                               key_states[:, :, :begin - prev_len, :],
                                               value_states[:, :, :begin - prev_len, :], False))
            flash_results.append(flash_per_chunk)
            k_states_prev = k_states_intra
            v_states_prev = v_states_intra
            remain_len = remain_len - chunk_len
        attn_output = ref_merge(flash_results)
    else:
        chunk_num_curr = (kv_seq_len - 1) // chunk_len
        q_states_intra = ref_apply_rope(query_states, q_cos, q_sin, position_ids)
        k_states_intra = key_states[:, :, chunk_len * chunk_num_curr:kv_seq_len, :]
        attn_weights = torch.matmul(q_states_intra, k_states_intra.transpose(2, 3)) / math.sqrt(attn.head_dim)
        attn_scores = [attn_weights]
        if chunk_num_curr >= 1:
            q_states_succ = ref_apply_rope(query_states, qc_cos, qc_sin, position_ids)
            k_states_succ = key_states[:, :, chunk_len * (chunk_num_curr - 1):chunk_len * chunk_num_curr, :]
            attn_scores = [torch.matmul(q_states_succ, k_states_succ.transpose(2, 3)) / math.sqrt(attn.head_dim)] + attn_scores
        if chunk_num_curr >= 2:
            q_states_inter = ref_apply_rope(query_states, qc_cos, qc_sin,
                                            torch.tensor([[chunk_len - 1]]))
            k_states_inter = key_states[:, :, :chunk_len * (chunk_num_curr - 1), :]
            attn_scores = [torch.matmul(q_states_inter, k_states_inter.transpose(2, 3)) / math.sqrt(attn.head_dim)] + attn_scores
        attn_weights = torch.cat(attn_scores, dim=-1)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, attn.hidden_size)
    return attn.o_proj(attn_output)


# ---------------- port side ----------------
class PortCache:
    def __init__(self):
        self.k, self.v = {}, {}

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.k:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], key], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], value], dim=2)
        else:
            self.k[layer_idx], self.v[layer_idx] = key, value
        return self.k[layer_idx], self.v[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.k[layer_idx].shape[2] if layer_idx in self.k else 0


def run_case(S_pre, decode_steps, block_len, tag):
    attn = FakeAttn(seed=7)
    rotary = RefRotary(D, PRETRAIN, scaling_factor=SCALING_FACTOR)

    m = DCAMethod(chunk_size=CHUNK_SIZE, local_window=LOCAL_WINDOW,
                  pretraining_length=PRETRAIN, scaling_factor=SCALING_FACTOR,
                  mscale_coeff=MSCALE_COEFF, use_flash_attn="off")
    m._inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
    port_fwd = m._make_dca_forward(attn, layer_idx=0)
    port_cache = PortCache()
    ref_cache = {"k": None, "v": None, "len": 0}

    torch.manual_seed(123)
    hidden = torch.randn(B, S_pre, HID)

    # 1) prefill
    out_ref = ref_forward(attn, rotary, hidden, torch.arange(S_pre).unsqueeze(0), ref_cache)
    out_port, _ = port_fwd(hidden_states=hidden, past_key_values=port_cache,
                           cache_position=torch.arange(S_pre))
    d = (out_ref - out_port).abs().max().item()
    assert d < 5e-6, "prefill mismatch"

    assert torch.equal(ref_cache["k"], port_cache.k[0]), "cached keys differ"

    # 2) q_len==1 decode steps
    pos = S_pre
    for step in range(decode_steps):
        torch.manual_seed(1000 + step)
        h1 = torch.randn(B, 1, HID)
        r = ref_forward(attn, rotary, h1, torch.tensor([[pos]]), ref_cache)
        p, _ = port_fwd(hidden_states=h1, past_key_values=port_cache,
                        cache_position=torch.tensor([pos]))
        d = (r - p).abs().max().item()
        assert d < 5e-6, "decode mismatch"
        pos += 1

    # 3) multi-token block (question pass), straddling if boundary inside
    if block_len:
        torch.manual_seed(2000)
        hb = torch.randn(B, block_len, HID)
        pid = torch.arange(pos, pos + block_len).unsqueeze(0)
        kv_after = pos + block_len
        # HF-style additive causal mask [B,1,q,kv]
        key_idx = torch.arange(kv_after)
        qabs = torch.arange(pos, pos + block_len)
        amask = torch.zeros(B, 1, block_len, kv_after)
        amask.masked_fill_(key_idx[None, None, None, :] > qabs[None, None, :, None],
                           torch.finfo(torch.float32).min)
        r = ref_forward(attn, rotary, hb, pid, ref_cache, attention_mask=amask)
        p, _ = port_fwd(hidden_states=hb, past_key_values=port_cache,
                        cache_position=torch.arange(pos, pos + block_len))
        d = (r - p).abs().max().item()
        assert d < 5e-6, "multitoken decode mismatch"



class TestDCAReferenceOracle(unittest.TestCase):
    """Each case: prefill parity + bitwise cyclic keys + decode parity."""

    def test_case_a_multichunk_prefill_mscale_pi_then_decode_and_straddle(self):
        # 4 chunks (inter active), mscale active (30 > 16), 3 decode steps,
        # then a 5-token block straddling a chunk boundary.
        run_case(S_pre=30, decode_steps=3, block_len=5, tag="A")

    def test_case_b_block_straddle_without_single_steps(self):
        run_case(S_pre=27, decode_steps=0, block_len=6, tag="B")

    def test_case_c_boundary_at_prefill_end(self):
        run_case(S_pre=17, decode_steps=4, block_len=4, tag="C")

    def test_case_d_single_chunk_prefill_decode_crosses_chunks(self):
        run_case(S_pre=6, decode_steps=4, block_len=3, tag="D")


if __name__ == "__main__":
    unittest.main()
