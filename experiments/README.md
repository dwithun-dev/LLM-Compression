# Experiments Directory & Protocol

This folder houses all research experiments. Each experiment is isolated in its own numbered subdirectory to ensure reproducibility and modular plug-and-play exploration.

---

## Directory Naming Convention

Format: `NN_descriptive_slug/`
- `01_layer_based_bench/`: Initial Layer 0 activation profiling, SVD, and Tucker compression.
- `02_cross_layer_svd/`: (Example future) Multi-layer or full-model low-rank decomposition.
- `03_tensor_train_moe/`: (Example future) Tensor-train factorization on feed-forward blocks.

---

## Standard Experiment Template

Every experiment folder must contain:

1. **`takeaways.md`** (Required):
   - **Hypothesis**: What theoretical property or compression behavior is being tested.
   - **Methodology**: Target layers, parameters, tensor shapes, thresholds, and datasets.
   - **Key Metrics Table**: Baseline vs Compressed accuracy, parameter reduction, reconstruction loss, and speedup.
   - **Scientific Conclusions**: What worked, what failed, and what to test next.

2. **`notebooks/` or Numbered Notebooks**:
   - For interactive exploration and plotting (e.g. `01_activation_profiling.ipynb`).
   - Cleanly imports from `neural_decomp` without `sys.path.append` or custom startup scripts.

3. **`scripts/`**:
   - Contains headless Python scripts (e.g. `run_eval.py`) to reproduce findings from the command line without Jupyter.

4. **`artifacts/`**:
   - Stores generated charts, HTML summaries, confusion matrices, and metrics JSON files.
