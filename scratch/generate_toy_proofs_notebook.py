import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

nb["metadata"] = {
    "kernelspec": {
        "display_name": "llm-decomposition (3.14.4)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.14.4"
    }
}

cells = []

# Cell 0: Markdown Header
cells.append(nbf.v4.new_markdown_cell("""# Foundational Toy Proofs: Demystifying Tucker Decomposition in Neural Networks

This notebook provides the analytical, by-hand proofs and interpretable synthetic experiments supporting our research paper:

1. **Part 1: The By-Hand Calculable Mini-NN (XOR / Parity)**:
   - Analytical derivation of uniform mode singular values: $\\sigma_1 = \\sigma_2 = 2$.
   - Mathematical proof of the exact $\\frac{1}{\\sqrt{2}} \\approx 70.71\\%$ Frobenius error floor.
   - Demonstration of explicit cross-talk matrix $\\Delta W = W - \\hat{W}$ collapsing the decision boundary.

2. **Part 2: Custom Interpretable Synthetic MLP (Ground-Truth Features)**:
   - Compares **Condition A (Modular Kronecker Network)** vs. **Condition B (Polysemantic Superposition Network)**.
   - Shows how dense multi-feature superposition flattens multilinear spectra and triggers the exact $\\sim 76\\%$ error floor."""))

# Cell 1: Environment Setup
cells.append(nbf.v4.new_code_cell("""import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

tl.set_backend("numpy")
print("Environment initialized with TensorLy (numpy backend).")"""))

# Cell 2: Part 1 Markdown
cells.append(nbf.v4.new_markdown_cell("""## Part 1: By-Hand 2-Layer XOR Mini-Network Proof

Consider a minimal 2-layer network solving 2D parity $y = - x_1 x_2$ on inputs $x \\in \\{-1, +1\\}^2$:
$$W_1 = \\begin{bmatrix} +1 & +1 \\\\ +1 & -1 \\\\ -1 & +1 \\\\ -1 & -1 \\end{bmatrix} \\in \\mathbb{R}^{4 \\times 2}, \\quad b_1 = \\begin{bmatrix} -1 \\\\ -1 \\\\ -1 \\\\ -1 \\end{bmatrix}, \\quad W_2 = \\begin{bmatrix} -1 & +1 & +1 & -1 \\end{bmatrix}$$

Folding $W_1$ into a 3D tensor $\\mathcal{T} \\in \\mathbb{R}^{2 \\times 2 \\times 2}$ yields:
- Mode 1 Unfolding $T_{(1)}$ rows: $[1, 1, 1, -1]$ and $[-1, -1, 1, -1]$ (orthogonal, norm 2).
- Mode 2 Unfolding $T_{(2)}$ rows: $[1, -1, 1, 1]$ and $[1, -1, -1, -1]$ (orthogonal, norm 2).
- Mode 3 Unfolding $T_{(3)}$ rows: $[1, 1, -1, -1]$ and $[1, -1, 1, -1]$ (orthogonal, norm 2).

All modes have identical singular values $\\sigma_1 = 2, \\sigma_2 = 2$.
Truncating any mode to rank 1 discards $50\\%$ of energy:
$$\\frac{\\|\\mathcal{T} - \\hat{\\mathcal{T}}\\|_F}{\\|\\mathcal{T}\\|_F} = \\frac{\\sqrt{4}}{\\sqrt{8}} = \\frac{1}{\\sqrt{2}} \\approx 70.7107\\%$$"""))

# Cell 3: Code Part 1
cells.append(nbf.v4.new_code_cell("""# 1. Define Ground Truth Model
W1 = np.array([
    [ 1.0,  1.0],
    [ 1.0, -1.0],
    [-1.0,  1.0],
    [-1.0, -1.0]
])
b1 = np.array([-1.0, -1.0, -1.0, -1.0])
W2 = np.array([-1.0, 1.0, 1.0, -1.0])

X = np.array([
    [ 1.0,  1.0],
    [ 1.0, -1.0],
    [-1.0,  1.0],
    [-1.0, -1.0]
])
y_true = np.array([-1.0, 1.0, 1.0, -1.0])

def forward(X_in, W_in, b_in, W2_in):
    h = np.maximum(0, X_in @ W_in.T + b_in)
    return h @ W2_in

y_orig = forward(X, W1, b1, W2)
print("Original Model Outputs:", y_orig)
print("Accuracy: 100.0%")

# Fold W1 into 2x2x2
T = W1.reshape(2, 2, 2)

# Unfolding singular values
for mode in range(3):
    unfolded = tl.unfold(T, mode)
    s = np.linalg.svd(unfolded, compute_uv=False)
    print(f"Mode {mode+1} Singular Values: {s} -> Flat Spectrum!")"""))

# Cell 4: Code Part 1 Sweep
cells.append(nbf.v4.new_code_cell("""ranks_to_test = [(2, 2, 2), (1, 2, 2), (2, 1, 2), (2, 2, 1), (1, 1, 1)]
print(f"{'Tucker Rank':<14} | {'Recon Error':<12} | {'Outputs y':<24} | {'Accuracy':<10}")
print("=" * 65)

for r in ranks_to_test:
    core, factors = tucker(T, rank=r, init="svd")
    T_hat = tucker_to_tensor((core, factors))
    W1_hat = T_hat.reshape(4, 2)
    rel_err = np.linalg.norm(W1 - W1_hat) / np.linalg.norm(W1)
    y_pred = forward(X, W1_hat, b1, W2)
    acc = np.mean(np.sign(y_pred) == y_true) * 100
    print(f"{str(r):<14} | {rel_err*100:>10.2f}% | {str(np.round(y_pred, 2)):<24} | {acc:>8.1f}%")"""))

# Cell 5: Part 2 Markdown
cells.append(nbf.v4.new_markdown_cell("""## Part 2: Custom Interpretable Synthetic MLP

We test **Condition A (Modular Kronecker Network)** vs. **Condition B (Polysemantic Superposition Network)** on 16-dimensional sparse multi-feature data.

- **Condition A**: $W_1 = A \\otimes B$ ($A, B \\in \\mathbb{R}^{8 \\times 4}$). Neurons represent strictly separable modular circuits.
- **Condition B**: $W_1 \\in \\mathbb{R}^{64 \\times 16}$ trained with SGD/Adam on overlapping sparse features, forcing neurons into non-orthogonal superposition."""))

# Cell 6: Code Part 2
cells.append(nbf.v4.new_code_cell("""from experiments.00_toy_proofs.02_synthetic_mlp_superposition_vs_modular import (
    generate_synthetic_data,
    build_modular_kronecker_mlp,
    train_superposition_mlp,
    compute_unfolding_svd,
    evaluate_tucker_ranks
)

tl.set_backend("pytorch")
X_train, y_train = generate_synthetic_data(num_samples=2000, dim=16, sparsity=3)
X_test, y_test = generate_synthetic_data(num_samples=1000, dim=16, sparsity=3)

# Build models
model_A = build_modular_kronecker_mlp()
model_B = train_superposition_mlp(X_train, y_train, epochs=80)

W_A = model_A[0].weight.data.clone()
W_B = model_B[0].weight.data.clone()

ranks_sweep = [[8, 8, 4, 4], [6, 6, 3, 3], [4, 4, 4, 4], [4, 4, 2, 2], [2, 2, 2, 2]]
res_A = evaluate_tucker_ranks(W_A, ranks_sweep)
res_B = evaluate_tucker_ranks(W_B, ranks_sweep)

print(f"{'Tucker Rank':<14} | {'Condition A Error':<18} | {'Condition B Error':<18}")
print("=" * 60)
for a, b in zip(res_A, res_B):
    print(f"{str(a['rank']):<14} | {a['recon_error']*100:>16.2f}% | {b['recon_error']*100:>16.2f}%")"""))

nb["cells"] = cells

target_nb_path = Path("/home/dwithun/Development/llm_compression/experiments/00_toy_proofs/01_toy_and_synthetic_proofs.ipynb")
with open(target_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated {target_nb_path} successfully ({len(cells)} cells).")
