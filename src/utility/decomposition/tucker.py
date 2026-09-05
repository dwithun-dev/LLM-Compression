"""Tensorization and Tucker Decomposition module.

Implements multi-way tensor folding, Tucker decomposition via TensorLy,
reconstruction, and compression accounting.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import torch


def compute_tucker_params(core_shape: List[int], factor_shapes: List[Tuple[int, int]]) -> Dict[str, int]:
    """Calculate the exact parameter count of a decomposed Tucker tensor representation."""
    import math
    core_params = math.prod(core_shape)
    factors_params = sum(shape[0] * shape[1] for shape in factor_shapes)
    total_compressed = core_params + factors_params
    return {
        "core_params": core_params,
        "factors_params": factors_params,
        "total_compressed": total_compressed,
    }


def decompose_tucker(
    tensor: torch.Tensor,
    rank: List[int],
    init: str = "svd",
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, Dict[str, float]]:
    """Decompose a high-order tensor into a core tensor and factor matrices using Tucker decomposition.

    Args:
        tensor: PyTorch tensor to decompose.
        rank: Target ranks for each mode of the tensor.
        init: Initialization method ('svd' or 'random').
        tol: Convergence tolerance for Tucker decomposition.

    Returns:
        Tuple of (core, factors, reconstructed_tensor, metrics_dict)
    """
    import tensorly as tl
    from tensorly.decomposition import tucker
    from tensorly.tucker_tensor import tucker_to_tensor

    tl.set_backend("pytorch")
    orig_shape = list(tensor.shape)
    orig_params = tensor.numel()

    # Perform Tucker decomposition
    core, factors = tucker(tensor, rank=rank, init=init, tol=tol)

    # Reconstruct tensor from core and factor matrices
    reconstructed = tucker_to_tensor((core, factors))

    # Accounting
    factor_shapes = [(f.shape[0], f.shape[1]) for f in factors]
    param_counts = compute_tucker_params(list(core.shape), factor_shapes)
    total_compressed = param_counts["total_compressed"]

    compression_ratio = orig_params / total_compressed if total_compressed > 0 else 0.0
    eliminated_pct = (1.0 - (total_compressed / orig_params)) * 100.0

    # Reconstruction error
    recon_error = torch.norm(tensor - reconstructed) / torch.norm(tensor)

    metrics = {
        "original_params": float(orig_params),
        "core_params": float(param_counts["core_params"]),
        "factors_params": float(param_counts["factors_params"]),
        "total_compressed_params": float(total_compressed),
        "compression_ratio": float(compression_ratio),
        "eliminated_pct": float(eliminated_pct),
        "reconstruction_error": float(recon_error.item()),
    }

    return core, factors, reconstructed, metrics
