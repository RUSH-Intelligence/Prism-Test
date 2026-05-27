from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from eval_harness.long_context import select_topk_indices_from_scores


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ReAttentionTopKSelector:
    """Naive ReAttention-style top-k token selector for HF models.

    This baseline computes per-candidate scores by averaging layer-wise QK dot products,
    where queries are taken from the final query window and keys are candidate context tokens.
    """

    def __init__(
        self,
        query_tokens: int = 128,
        query_chunk_tokens: int = 32,
        vote_top_k: int = 4,
        streaming: bool = True,
        streaming_chunk_tokens: int = 4096,
    ) -> None:
        self._query_tokens = max(int(query_tokens), 1)
        self._query_chunk_tokens = max(int(query_chunk_tokens), 1)
        self._vote_top_k = max(int(vote_top_k), 1)
        self._streaming = bool(streaming)
        self._streaming_chunk_tokens = max(int(streaming_chunk_tokens), 1)

    @staticmethod
    def _model_device(model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _build_query_projections(
        self,
        model: torch.nn.Module,
        token_ids: Sequence[int],
        device: torch.device,
    ) -> tuple[List[torch.Tensor], object, int]:
        query_start = max(0, len(token_ids) - self._query_tokens)
        query_token_ids = list(token_ids[query_start:])
        if not query_token_ids:
            raise RuntimeError("Cannot build query projections from empty token sequence")

        query_input_ids = torch.tensor([query_token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=query_input_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states for query projection")

        layers_container = getattr(model, "model", None)
        if layers_container is None or not hasattr(layers_container, "layers"):
            raise RuntimeError(
                "Unsupported architecture for ReAttention top-k baseline: missing model.layers"
            )

        query_by_layer: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(layers_container.layers):
            hs = hidden_states[layer_idx]
            attn = layer.self_attn
            q_proj = attn.q_proj

            q = q_proj(hs)
            num_heads = getattr(attn, "num_heads")
            head_dim = getattr(attn, "head_dim")
            q = q.view(1, -1, num_heads, head_dim).transpose(1, 2)
            query_by_layer.append(q.to(torch.float32))

        return query_by_layer, layers_container, len(layers_container.layers)

    def score_candidates(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        candidate_indices: Sequence[int],
    ) -> List[float]:
        if not candidate_indices:
            return []

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "Model did not return hidden_states; cannot compute ReAttention top-k scores"
            )

        layers_container = getattr(model, "model", None)
        if layers_container is None or not hasattr(layers_container, "layers"):
            raise RuntimeError(
                "Unsupported architecture for ReAttention top-k baseline: missing model.layers"
            )

        candidate_tensor = torch.tensor(candidate_indices, device=input_ids.device, dtype=torch.long)
        per_layer_scores: List[torch.Tensor] = []

        for layer_idx, layer in enumerate(layers_container.layers):
            hs = hidden_states[layer_idx]
            attn = layer.self_attn

            q_proj = attn.q_proj
            k_proj = attn.k_proj

            q_start = max(0, hs.shape[1] - self._query_tokens)
            query_hs = hs[:, q_start:, :]
            key_hs = hs.index_select(1, candidate_tensor)

            q = q_proj(query_hs)
            k = k_proj(key_hs)

            num_heads = getattr(attn, "num_heads")
            num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
            head_dim = getattr(attn, "head_dim")

            q = q.view(1, -1, num_heads, head_dim).transpose(1, 2)
            k = k.view(1, -1, num_kv_heads, head_dim).transpose(1, 2)
            k = _repeat_kv(k, max(num_heads // num_kv_heads, 1))

            layer_scores = torch.einsum("bhqd,bhkd->bhqk", q, k).mean(dim=(1, 2)).squeeze(0)
            per_layer_scores.append(layer_scores.to(torch.float32))

        stacked = torch.stack(per_layer_scores, dim=0).mean(dim=0)
        return stacked.detach().cpu().tolist()

    def score_candidates_streaming(
        self,
        model: torch.nn.Module,
        token_ids: Sequence[int],
        candidate_indices: Sequence[int],
    ) -> List[float]:
        if not candidate_indices:
            return []

        device = self._model_device(model)
        query_by_layer, layers_container, num_layers = self._build_query_projections(
            model=model,
            token_ids=token_ids,
            device=device,
        )

        sorted_candidates = sorted(candidate_indices)
        score_map: Dict[int, float] = {}
        ptr = 0
        seq_len = len(token_ids)

        for chunk_start in range(0, seq_len, self._streaming_chunk_tokens):
            chunk_end = min(chunk_start + self._streaming_chunk_tokens, seq_len)

            local_candidates: List[int] = []
            while ptr < len(sorted_candidates) and sorted_candidates[ptr] < chunk_start:
                ptr += 1
            tmp = ptr
            while tmp < len(sorted_candidates) and sorted_candidates[tmp] < chunk_end:
                local_candidates.append(sorted_candidates[tmp])
                tmp += 1
            ptr = tmp

            if not local_candidates:
                continue

            chunk_ids = list(token_ids[chunk_start:chunk_end])
            chunk_input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                outputs = model(
                    input_ids=chunk_input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states in streaming scorer")

            rel_positions = torch.tensor(
                [idx - chunk_start for idx in local_candidates],
                device=device,
                dtype=torch.long,
            )
            local_scores = torch.zeros(len(local_candidates), device=device, dtype=torch.float32)

            for layer_idx, layer in enumerate(layers_container.layers):
                hs = hidden_states[layer_idx]
                attn = layer.self_attn
                k_proj = attn.k_proj

                key_hs = hs.index_select(1, rel_positions)
                k = k_proj(key_hs)

                num_heads = getattr(attn, "num_heads")
                num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
                head_dim = getattr(attn, "head_dim")

                k = k.view(1, -1, num_kv_heads, head_dim).transpose(1, 2)
                k = _repeat_kv(k, max(num_heads // num_kv_heads, 1))

                q = query_by_layer[layer_idx]
                layer_scores = torch.einsum("bhqd,bhkd->bhqk", q, k).mean(dim=(1, 2)).squeeze(0)
                local_scores += layer_scores.to(torch.float32)

            local_scores /= float(max(num_layers, 1))
            local_scores_list = local_scores.detach().cpu().tolist()
            for idx, score in zip(local_candidates, local_scores_list):
                score_map[idx] = float(score)

            del outputs
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()

        return [score_map.get(idx, float("-inf")) for idx in candidate_indices]

    def select_indices(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        candidate_indices: Sequence[int],
        top_k: int,
        token_ids: Sequence[int] | None = None,
    ) -> List[int]:
        if top_k <= 0 or not candidate_indices:
            return []

        if self._streaming:
            if token_ids is None:
                token_ids = input_ids.squeeze(0).tolist()
            scores = self.score_candidates_streaming(
                model=model,
                token_ids=token_ids,
                candidate_indices=candidate_indices,
            )
        else:
            scores = self.score_candidates(
                model=model,
                input_ids=input_ids,
                candidate_indices=candidate_indices,
            )
            return select_topk_indices_from_scores(
                candidate_indices=candidate_indices,
                scores=scores,
                top_k=top_k,
            )

        # Non-streaming baseline: per-layer/head/query-chunk voting over middle tokens,
        # with score fallback for tie-breaking.
        return self.select_indices_with_voting(
            model=model,
            input_ids=input_ids,
            candidate_indices=candidate_indices,
            top_k=top_k,
        )

    def select_indices_with_voting(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        candidate_indices: Sequence[int],
        top_k: int,
    ) -> List[int]:
        if top_k <= 0 or not candidate_indices:
            return []

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "Model did not return hidden_states; cannot compute ReAttention top-k scores"
            )

        layers_container = getattr(model, "model", None)
        if layers_container is None or not hasattr(layers_container, "layers"):
            raise RuntimeError(
                "Unsupported architecture for ReAttention top-k baseline: missing model.layers"
            )

        candidate_tensor = torch.tensor(candidate_indices, device=input_ids.device, dtype=torch.long)
        num_candidates = len(candidate_indices)
        votes = torch.zeros(num_candidates, dtype=torch.float32, device=input_ids.device)
        score_sums = torch.zeros(num_candidates, dtype=torch.float32, device=input_ids.device)

        for layer_idx, layer in enumerate(layers_container.layers):
            hs = hidden_states[layer_idx]
            attn = layer.self_attn

            q_proj = attn.q_proj
            k_proj = attn.k_proj

            q_start = max(0, hs.shape[1] - self._query_tokens)
            query_hs = hs[:, q_start:, :]
            key_hs = hs.index_select(1, candidate_tensor)

            q = q_proj(query_hs)
            k = k_proj(key_hs)

            num_heads = getattr(attn, "num_heads")
            num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
            head_dim = getattr(attn, "head_dim")

            q = q.view(1, -1, num_heads, head_dim).transpose(1, 2)
            k = k.view(1, -1, num_kv_heads, head_dim).transpose(1, 2)
            k = _repeat_kv(k, max(num_heads // num_kv_heads, 1))

            scores = torch.einsum("bhqd,bhkd->bhqk", q, k).to(torch.float32)
            score_sums += scores.mean(dim=(1, 2)).squeeze(0)

            q_len = scores.shape[2]
            proposal_k = min(self._vote_top_k, num_candidates)
            for qs in range(0, q_len, self._query_chunk_tokens):
                qe = min(qs + self._query_chunk_tokens, q_len)
                chunk_scores = scores[:, :, qs:qe, :].max(dim=2).values  # [1, heads, candidates]
                top_idx = torch.topk(chunk_scores, k=proposal_k, dim=-1).indices.reshape(-1)
                vote_increments = torch.bincount(top_idx, minlength=num_candidates).to(torch.float32)
                votes += vote_increments

        # Rank by votes first, then by mean score as tie-breaker.
        # Use Python sort for deterministic behavior across devices.
        score_list = score_sums.detach().cpu().tolist()
        vote_list = votes.detach().cpu().tolist()

        ranked = sorted(
            zip(candidate_indices, vote_list, score_list),
            key=lambda x: (x[1], x[2], x[0]),
            reverse=True,
        )
        chosen = sorted(idx for idx, _, _ in ranked[:top_k])
        return chosen
