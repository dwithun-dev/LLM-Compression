"""Singular Value Decomposition (SVD) and Spectral Analysis module."""

from __future__ import annotations

from typing import Dict, Tuple
import torch


def compute_spectral_energy(singular_values: torch.Tensor) -> Dict[str, int]:
    """Compute cumulative energy thresholds (80%, 90%, 95%, 99%) across singular values."""
    energy = torch.cumsum(singular_values**2, dim=0) / torch.sum(singular_values**2)
    ranks = {
        "rank_80": int(torch.searchsorted(energy, 0.80).item()) + 1,
        "rank_90": int(torch.searchsorted(energy, 0.90).item()) + 1,
        "rank_95": int(torch.searchsorted(energy, 0.95).item()) + 1,
        "rank_99": int(torch.searchsorted(energy, 0.99).item()) + 1,
    }
    return ranks


def decompose_svd(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute full or reduced SVD of a 2D weight matrix."""
    U, S, Vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return U, S, Vh


def truncate_svd(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, rank: int) -> torch.Tensor:
    """Reconstruct a low-rank approximation given singular components and target rank."""
    return U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
