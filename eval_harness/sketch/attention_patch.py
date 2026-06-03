import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def search_hyperplane(X, max_iter: int = 1000):
    Y = X.mean(1)
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            return -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")


def attention_patch(func):
    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        if query.shape[2] == key.shape[2]:
            module.masked_key_indices = None
        elif getattr(module, "masked_key_indices", None) is not None:
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            k = search_hyperplane(q)
            k = k.view(bsz, num_key_value_heads, head_dim)

            batch_indices, head_indices, seq_indices = module.masked_key_indices
            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]

        if "cu_seq_lens_k" in kwargs:
            kwargs["cu_seq_lens_k"][-1] = key.shape[-2]
        return func(module, query, key, value, attention_mask, dropout, **kwargs)

    return wrapper


def patch_attention_functions():
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)
