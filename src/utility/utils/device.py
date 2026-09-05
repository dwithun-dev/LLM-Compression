"""Hardware and device utility functions."""

from __future__ import annotations

from typing import Dict
import torch


def get_device_info() -> Dict[str, object]:
    """Inspect CUDA hardware availability and memory statistics."""
    cuda_avail = torch.cuda.is_available()
    info = {
        "cuda_available": cuda_avail,
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "current_device": torch.cuda.current_device() if cuda_avail else None,
        "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
    }
    if cuda_avail:
        info["allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 3)
        info["reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 3)
    return info
