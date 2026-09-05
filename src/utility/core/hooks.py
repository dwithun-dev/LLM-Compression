"""Activation Hook Management module.

Provides reusable hook registration, storage, pooling, and context-managed cleanup
for PyTorch module forward passes.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


class ActivationHookManager:
    """Context manager and registry for capturing submodule forward activations.

    Example:
        >>> hook_mgr = ActivationHookManager(target_layer)
        >>> with hook_mgr:
        ...     outputs = model(**inputs)
        ...     step_acts = hook_mgr.get_step_activations()
    """

    def __init__(
        self,
        module: nn.Module,
        submodule_names: Optional[List[str]] = None,
        pooling_fn: Optional[Callable[[torch.Tensor], np.ndarray]] = None,
    ) -> None:
        self.module = module
        self.handles: List[RemovableHandle] = []
        self.current_step_acts: Dict[str, torch.Tensor] = {}
        self.aggregated_activations: Dict[str, List[np.ndarray]] = {}

        # If submodule_names not provided, register on all named submodules
        if submodule_names is None:
            self.target_submodules = [
                (name, submod) for name, submod in module.named_modules() if name != ""
            ]
        else:
            self.target_submodules = [
                (name, submod)
                for name, submod in module.named_modules()
                if name in submodule_names
            ]

        self.submodule_names = [name for name, _ in self.target_submodules]
        for name in self.submodule_names:
            self.aggregated_activations[name] = []

        self.pooling_fn = pooling_fn or self._default_pooling

    def _default_pooling(self, tensor: torch.Tensor) -> np.ndarray:
        """Default pooling across sequence length (mean pooling)."""
        arr = tensor.squeeze(0).mean(dim=0).cpu().numpy()
        return arr

    def _create_hook(self, submodule_name: str):
        def hook(module: nn.Module, input_tensor, output_tensor):
            act = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            self.current_step_acts[submodule_name] = act.detach().cpu()
        return hook

    def register(self) -> None:
        """Register forward hooks on target submodules."""
        self.clear_handles()
        for name, submod in self.target_submodules:
            handle = submod.register_forward_hook(self._create_hook(name))
            self.handles.append(handle)

    def remove(self) -> None:
        """Remove all active forward hooks."""
        self.clear_handles()

    def clear_handles(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def step_completed(self) -> None:
        """Pool and append current step activations to the aggregated store."""
        for name, act_tensor in self.current_step_acts.items():
            pooled = self.pooling_fn(act_tensor)
            self.aggregated_activations[name].append(pooled)
        self.current_step_acts.clear()

    def get_stacked_activations(self) -> Dict[str, np.ndarray]:
        """Return aggregated activations stacked into a NumPy array per submodule."""
        result = {}
        for name, items in self.aggregated_activations.items():
            if items:
                result[name] = np.stack(items)
            else:
                result[name] = np.array([])
        return result

    def __enter__(self) -> ActivationHookManager:
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()
