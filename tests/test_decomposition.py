"""Unit tests for SVD and Sparsity decomposition utilities."""

import unittest
import torch
from neural_decomp.decomposition.svd import decompose_svd, compute_spectral_energy, truncate_svd
from neural_decomp.decomposition.sparsity import apply_magnitude_sparsity


class TestDecomposition(unittest.TestCase):
    def test_svd_spectral_energy(self):
        # Construct a matrix with rank 2
        A = torch.randn(50, 2) @ torch.randn(2, 20)
        U, S, Vh = decompose_svd(A)
        self.assertEqual(len(S), 20)

        energy = compute_spectral_energy(S)
        self.assertLessEqual(energy["rank_80"], 3)
        self.assertLessEqual(energy["rank_95"], 3)

        recon = truncate_svd(U, S, Vh, rank=2)
        self.assertTrue(torch.allclose(A, recon, atol=1e-4))

    def test_magnitude_sparsity(self):
        weights = torch.linspace(-1.0, 1.0, 1000)
        sparse_weights, pct = apply_magnitude_sparsity(weights, quantile_threshold=0.50)
        self.assertAlmostEqual(pct, 50.0, delta=2.0)
        self.assertGreaterEqual((sparse_weights == 0.0).sum().item(), 490)


if __name__ == "__main__":
    unittest.main()
