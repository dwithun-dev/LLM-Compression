"""Activation Profiling module.

Utilities for dead-neuron detection, outlier identification, and activation variance analysis.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import torch


def compute_dead_neuron_ratio(activations: np.ndarray, threshold: float = 0.05) -> float:
    """Calculate the percentage of activations whose absolute magnitude is below a threshold."""
    return float(np.mean(np.abs(activations) < threshold) * 100.0)


def compute_outlier_ratio(activations: np.ndarray, threshold: float = 3.0) -> float:
    """Calculate the percentage of activations whose absolute value exceeds an outlier cutoff."""
    return float(np.mean(np.abs(activations) > threshold) * 100.0)


def summarize_activations(
    activations: np.ndarray,
    dead_threshold: float = 0.05,
    outlier_threshold: float = 3.0,
) -> Dict[str, float]:
    """Compute comprehensive distribution metrics for an activation tensor."""
    return {
        "mean": float(np.mean(activations)),
        "std": float(np.std(activations)),
        "min": float(np.min(activations)),
        "max": float(np.max(activations)),
        "dead_ratio_pct": compute_dead_neuron_ratio(activations, threshold=dead_threshold),
        "outlier_ratio_pct": compute_outlier_ratio(activations, threshold=outlier_threshold),
    }


def partition_inactive_neurons(
    activations: np.ndarray,
    num_inactive: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank neurons by activation variance and partition into inactive vs active indices.

    Args:
        activations: 2D array of shape [samples, hidden_dim].
        num_inactive: Count of least-active neurons to isolate.

    Returns:
        Tuple of (inactive_indices, active_indices, neuron_variances)
    """
    variance = np.var(activations, axis=0)
    sorted_indices = np.argsort(variance)
    inactive_indices = sorted_indices[:num_inactive]
    active_indices = sorted_indices[num_inactive:]
    return inactive_indices, active_indices, variance
