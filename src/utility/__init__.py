"""Neural Decomposition (`neural_decomp`) Library.

A framework for neural network weight decomposition, activation profiling,
and structured tensor compression.
"""

from utility.core.mmi import ModelManagementInterface, DeviceMapOptions
from utility.core.hooks import ActivationHookManager

__version__ = "0.1.0"
__all__ = [
    "ModelManagementInterface",
    "DeviceMapOptions",
    "ActivationHookManager",
    "__version__",
]
