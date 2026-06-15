from dataclasses import dataclass, field
from typing import Optional

import torch
from huggingface_hub import PyTorchModelHubMixin, get_collection
from torch import nn
from transformers import PreTrainedModel

from eval_harness.kv_compression.compressors.expected_attention_sketch import ExpectedAttentionSketch
from eval_harness.kv_compression.registry import register_kv_compressor


class ExpectedAttentionStats(torch.nn.Module, PyTorchModelHubMixin):
    """
    Module that stores the mean and covariance matrix of the queries, possibly uploaded to the HF hub.

    Port of kvpress ``ExpectedAttentionStats``
    (kvpress/presses/expected_attention_with_stats.py). The on-disk format
    (``config.json`` + ``model.safetensors`` with ``query_mean``/``query_cov``)
    is identical, so folders produced by kvpress's calibration script load
    unchanged via ``from_pretrained``.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dataset_name: str,
        model_name: str,
        num_samples: int,
        sample_seq_len: int,
        n_sink: int,
    ):
        super().__init__()
        self.query_mean = torch.nn.Parameter(torch.zeros(num_layers, num_heads, head_dim))
        self.query_cov = torch.nn.Parameter(torch.zeros(num_layers, num_heads, head_dim, head_dim))
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_samples = num_samples
        self.sample_seq_len = sample_seq_len
        self.n_sink = n_sink

    def stats_id(self) -> str:
        """Generate the statistics ID for the model and configuration."""
        return f"alessiodevoto/exp_att_stats_{self.model_name.replace('/', '_')}_{self.dataset_name.replace('/', '_')}_{self.num_samples}_{self.sample_seq_len}_{self.n_sink}"  # noqa: E501


@register_kv_compressor("expected_attention_stats")
@dataclass
class ExpectedAttentionStatsSketch(ExpectedAttentionSketch):
    """
    Expected attention sketch that automatically loads pre-computed query statistics.

    Port of kvpress ``ExpectedAttentionStatsPress``
    (kvpress/presses/expected_attention_with_stats.py). Replaces the parent's
    on-the-fly query statistics with per-layer/per-query-head statistics of
    pre-RoPE queries computed offline on a calibration dataset, loaded once in
    ``post_init_from_model`` from ``stats_folder`` or from the HF hub repo
    named by ``ExpectedAttentionStats.stats_id()``.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    n_future_positions : int, default=512
        Number of future positions to consider when computing expected attention.
    n_sink : int, default=4
        Number of initial tokens to exclude from compression (sink tokens).
    use_covariance : bool, default=True
        Whether to include covariance information in expected attention computation.
    use_vnorm : bool, default=True
        Whether to rescale scores using value vector norms.
    epsilon : float, default=0.0
        Small constant added to scores before value norm rescaling.
    dataset_name : str, default="kmfoda/booksum"
        Dataset used to compute the statistics.
    num_samples : int, default=100
        Number of samples used to compute the statistics.
    sample_seq_len : int, default=1000
        Sequence length used to compute the statistics.
    stats_folder : Optional[str], default=None
        Local path to a saved ``ExpectedAttentionStats``; if None, the hub repo
        id is derived from the model config and downloaded.

    Deviations from kvpress
    -----------------------
    - The offline calibration script (``patch_rotary_embedding`` /
      ``collect_queries`` / ``main``) is not ported (it needs ``fire``,
      ``datasets`` and live model forwards). Stats for models missing from the
      hub collection must be produced with kvpress's own script; the saved
      folder is format-compatible with ``stats_folder``.

    Notes
    -----
    - ``mu``/``cov`` are ``field(init=False)`` exactly as upstream; tests and
      offline runs can inject pre-built tensors by assigning both attributes
      after construction — the ``mu is None and cov is None`` guard then makes
      ``post_init_from_model`` a no-op (load-once semantics).
    - As upstream, stats are loaded even when ``compression_ratio == 0``
      (``compress`` short-circuits before ``score``, so they are unused).
    - ``mu`` has shape ``(num_layers, H_q, head_dim)`` and ``cov``
      ``(num_layers, H_q, head_dim, head_dim)``, indexed by the absolute
      ``module.layer_idx`` and broadcast over the batch via a size-1 batch dim.
    - Supported combo is ``prefill_method: none`` (DCA's cyclic key positions
      and ReAttention's pre-pruned cache break the absolute-position frame).
    """

    # Override parent defaults to enable stats by default
    sample_seq_len: int = 1000
    num_samples: int = 100
    dataset_name: str = "kmfoda/booksum"
    stats_folder: Optional[str] = None

    mu: torch.Tensor = field(init=False, default=None)  # initialized in post_init_from_model
    cov: torch.Tensor = field(init=False, default=None)  # initialized in post_init_from_model

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        """
        Override the parent method to use the pre-computed query statistics.
        """
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx
        mu, cov = self.apply_avg_rope(module, self.mu[layer_idx], self.cov[layer_idx], q_len)  # type: ignore
        return mu.unsqueeze(0), cov.unsqueeze(0)

    @staticmethod
    def available_stats():
        collection = get_collection("alessiodevoto/expectedattentionstats-68b0248d519303713320e2cf")
        return [x.item_id for x in collection.items]

    def post_init_from_model(self, model: PreTrainedModel):
        """
        Automatically load or compute query statistics for the model.
        """
        if self.mu is None and self.cov is None:
            if self.stats_folder is not None:
                stats = ExpectedAttentionStats.from_pretrained(self.stats_folder)
            else:
                stats = self._maybe_load_stats_from_hub(model)
            self.mu = stats.query_mean.data.to(model.device, dtype=model.dtype)
            self.cov = stats.query_cov.data.to(model.device, dtype=model.dtype)

    def _maybe_load_stats_from_hub(self, model: PreTrainedModel):
        """Load statistics from the Hugging Face Hub."""
        stats_id = ExpectedAttentionStats(
            model_name=model.config.name_or_path,
            num_layers=model.config.num_hidden_layers,
            num_heads=model.config.num_attention_heads,
            head_dim=model.config.head_dim,
            dataset_name=self.dataset_name,
            num_samples=self.num_samples,
            sample_seq_len=self.sample_seq_len,
            n_sink=self.n_sink,
        ).stats_id()
        try:
            return ExpectedAttentionStats.from_pretrained(stats_id)
        except ValueError:
            raise ValueError(
                f"No statistics found for model {stats_id} on the Hub. Please compute them first. "
                "You can do so by running the following code: "
                "```"
                "python expected_attention_with_stats.py --model_name <model_name>"
                "```"
            )
