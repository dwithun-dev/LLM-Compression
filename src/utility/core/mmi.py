"""Model Management Interface (MMI) module.

Provides a unified interface for model instantiation, precision management,
and device placement for PyTorch causal language models.
"""

from __future__ import annotations

from enum import Enum
from typing import Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class DeviceMapOptions(str, Enum):
    """Device placement strategies for model loading."""

    AUTO = "auto"
    """Automatically calculates and dispatches layers across available GPUs and CPU/disk memory."""

    BALANCED = "balanced"
    """Splits layers equally across GPUs."""

    BALANCED_LOW_0 = "balanced_low_0"
    """Splits layers equally across GPUs, keeping fewer layers on GPU 0 to reserve memory for activations."""

    SEQUENTIAL = "sequential"
    """Fills GPU 0 completely before moving sequentially to GPU 1, GPU 2, then CPU/disk."""

    NONE = "None"
    """Leaves all weights unassigned or on meta device until manually moved."""


class ModelManagementInterface:
    """Manages model loading, tokenizer creation, and precision configuration."""

    def __init__(
        self,
        model_id: str,
        precision: torch.dtype = torch.float32,
        device_map: Union[DeviceMapOptions, str] = DeviceMapOptions.AUTO,
    ) -> None:
        self.model_id: str = model_id
        self.precision: torch.dtype = precision
        self.device_map: str = device_map.value if isinstance(device_map, DeviceMapOptions) else str(device_map)

    def change_precision(self, precision: torch.dtype) -> None:
        """Update target precision for model loading."""
        self.precision = precision

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Load and return tokenizer for the configured model."""
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_id)

    def get_model(self, **kwargs) -> PreTrainedModel:
        """Instantiate and return the causal language model with configured settings."""
        device_map_arg = None if self.device_map == "None" else self.device_map
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            device_map=device_map_arg,
            dtype=self.precision,
            **kwargs,
        )
