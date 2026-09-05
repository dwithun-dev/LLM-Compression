"""Magnitude thresholding and weight sparsification module."""

from __future__ import annotations

from typing import Tuple
import torch


def apply_magnitude_sparsity(
    weights: torch.Tensor,
    quantile_threshold: float = 0.60,
) -> Tuple[torch.Tensor, float]:
    """Zero-out weights below a given absolute magnitude quantile threshold.

    Args:
        weights: Tensor of model weights to sparsify.
        quantile_threshold: Fraction of smallest weights to zero-mask (e.g. 0.60 = 60th percentile).

    Returns:
        Tuple of (sparsified_tensor, achieved_sparsity_percentage)
    """
    epsilon = torch.quantile(torch.abs(weights), quantile_threshold)
    sparse_weights = weights.clone()
    sparse_weights[torch.abs(sparse_weights) < epsilon] = 0.0

    total_elements = sparse_weights.numel()
    zero_elements = (sparse_weights == 0.0).sum().item()
    sparsity_pct = (zero_elements / total_elements) * 100.0

    return sparse_weights, float(sparsity_pct)
