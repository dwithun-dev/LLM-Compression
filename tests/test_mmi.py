"""Unit tests for ModelManagementInterface and DeviceMapOptions."""

import unittest
import torch
from neural_decomp.core.mmi import ModelManagementInterface, DeviceMapOptions


class TestMMI(unittest.TestCase):
    def test_device_map_options(self):
        self.assertEqual(DeviceMapOptions.AUTO.value, "auto")
        self.assertEqual(DeviceMapOptions.BALANCED.value, "balanced")
        self.assertEqual(DeviceMapOptions.BALANCED_LOW_0.value, "balanced_low_0")
        self.assertEqual(DeviceMapOptions.SEQUENTIAL.value, "sequential")
        self.assertEqual(DeviceMapOptions.NONE.value, "None")

    def test_mmi_initialization(self):
        mmi = ModelManagementInterface(
            model_id="test-model",
            precision=torch.float32,
            device_map=DeviceMapOptions.AUTO,
        )
        self.assertEqual(mmi.model_id, "test-model")
        self.assertEqual(mmi.precision, torch.float32)
        self.assertEqual(mmi.device_map, "auto")

        mmi.change_precision(torch.bfloat16)
        self.assertEqual(mmi.precision, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
