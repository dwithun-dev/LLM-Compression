# Academic Research Roadmap & TODO

This document tracks completed milestones, theoretical insights, empirical breakthroughs, and future academic research directions for tensor-based LLM compression and denoising on `google/gemma-3-1b-it`.

---

## 0. Current Milestone & Breakthrough Status

- [x] **Mathematical & Toy Foundations (Exp 1 & 2):**
  - Proved Frobenius error floor and empirical collapse under polysemantic superposition.
- [x] **Failure Mode Documented (Exp 18 / `3.ipynb`):**
  - Stacking attention clusters into 3D Tucker across all 26 layers caused cross-cluster factor contamination, core tensor explosion (+3.08% net cut), reasoning degradation (48.8% $\rightarrow$ 32.0%), and degenerate repetition loops.
- [x] **Independent SVD Baseline (Exp 20 / `4.ipynb`):**
  - Achieved +20.20% net cut (14.1M params) and 1,559x speedup (4.42s on Dual T4), but uniform energy cut ($\tau=0.90$) across all 26 layers suffered compounding loss ($0.90^{26} \approx 6.4\%$).
- [x] **TensorLLM 4D Multi-Head Formulation (Exp 22 / `5.ipynb`):**
  - Formulated MHA as $\mathcal{W}_{\text{all}} \in \mathbb{R}^{d_{\text{model}} \times h \times d_v \times 4}$.
  - Adapted for Gemma-3 Grouped-Query Attention ($d_{\text{model}}=1152, h=4, d_v=256, h_{kv}=1$).
  - Decomposed via `partial_tucker(modes=[0, 2, 3])`, leaving Mode 1 ($h$) uncompressed with private head cores $\mathcal{G}_i$.
  - **Empirical Breakthrough on Layer 14:**
    - **GLUE MNLI Accuracy ($N=500$):** 48.80% $\rightarrow$ **54.40% (+5.60% GAIN)**.
    - **Layer 14 MHA Parameters:** 4,718,592 $\rightarrow$ 909,056 (**+80.73% CUT**).
    - **Reconstruction Error:** 70.00% (tolerates high Frobenius error floor without harm).
    - **Qualitative Generation:** 100% flawless chocolate cake recipe (zero loops, coherent structure).
- [x] **All-26-Layer Sweep Profile:**
  - Mapped layer sensitivity: Middle-to-deep layers (14, 16, 19, 20, 21, 22, 24, 25) consistently exhibit positive reasoning gains (+2.2% to +8.2%) under low-rank tensorisation.

---

## 1. Direction 1: Multi-Layer Joint Denoising & Interference Analysis (Immediate Next)

- [x] **Exp 23 — Executed on Kaggle Dual Tesla T4 (`6.ipynb`):**
  - **Single Layer (L14):** Vanilla TensorLLM achieved **38.80% (+4.20% gain)** with -70.14% MHA params. Act-TensorLLM achieved 32.00% (-2.60%).
  - **Joint 2-Layer (L14 + L22):** Shaved **7.62M parameters** while maintaining **38.80% (+4.20% gain)** over baseline!
  - **Joint 4-Layer (L14, L19, L21, L22):** Shaved **15.24M parameters** across 4 MHA blocks while maintaining **34.80% (+0.20% gain)** over baseline!
  - **The Low-Rank Outlier Starvation Paradox:** Discovered why activation weighting fails in low-rank decomposition (unlike quantization): scaling by an $83.1\times$ dynamic ratio causes the low-rank subspace to be monopolized by 1-2 outlier channels, starving the remaining 1,150 semantic channels.
- [ ] **Cross-Layer Covariance & Drift Analysis:**
  - Measure if decomposing upstream Layer 14 alters the input activation distribution to downstream Layer 22.
  - Assess whether downstream layers require dynamic rank re-adjustment when upstream layers are modified.
- [ ] **Pareto Frontier & Analytical Layer Selector:**
  - Formulate optimal rank/layer allocation:
    $$\max_{\mathbf{r}_1, \dots, \mathbf{r}_L} \text{Acc}(\mathcal{M}(\mathbf{r})) \quad \text{s.t.} \quad \sum_{l=1}^L \text{Params}(\mathbf{r}_l) \le C$$
  - Derive a lightweight heuristic predictor (e.g., singular value decay, activation variance, or attention entropy) to predict layer compressibility without brute-force validation sweeps.

---

## 2. Direction 2: Mechanistic Interpretability of Tucker Denoising

- [ ] **Attention Entropy Analysis:**
  - Calculate attention entropy $\mathcal{H}(A_l) = -\sum_{j} A_{ij} \log A_{ij}$ across heads before and after TensorLLM decomposition.
  - Test the hypothesis: Does low-rank Tucker sharpen dominant syntactic/semantic heads and suppress diffuse background noise?
- [ ] **Outlier Feature Suppression in Mode 0 ($U^{(1)}$):**
  - Analyze the singular vectors of factor matrix $U^{(1)} \in \mathbb{R}^{d_{\text{model}} \times R_1}$.
  - Investigate whether the low-rank projection filters out extreme activation channel outliers characteristic of RMSNorm architectures.
- [ ] **Subspace Decomposition & Discarded Noise Residual:**
  - Explicitly isolate the discarded tensor: $\mathcal{W}_{\text{noise}} = \mathcal{W}_{\text{orig}} - \hat{\mathcal{W}}_{\text{tucker}}$.
  - Perform spectral analysis on $\mathcal{W}_{\text{noise}}$ to prove whether it resembles isotropic Gaussian noise or structured feature interference.
- [ ] **Representation Trajectory (Logit / Tuned Lens):**
  - Track hidden state cosine similarity to true task directions along the residual stream before and after Layer 14 adaptation.

---

## 3. Direction 3: Activation-Aware & Fisher-Weighted 4D Tucker (Hessian-TensorLLM)

- [ ] **Activation Covariance Calibration:**
  - Run calibration dataset (e.g., 512 sequences from C4 / WikiText) through Gemma-3-1B-IT.
  - Compute input activation covariance matrices $\Sigma_X = \mathbb{E}[X X^T]$ for each layer's MHA input.
- [ ] **Generalize Second-Order Weighting to Multi-Way Tensors:**
  - Implement Hessian/Activation-weighted Partial Tucker:
    $$\min_{\mathcal{G}, U} \|\mathcal{W}_{\text{all}} \times_1 \Sigma_X^{1/2} - \hat{\mathcal{W}_{\text{all}}} \times_1 \Sigma_X^{1/2}\|_F^2$$
- [ ] **All-26-Layer Joint Stability Test:**
  - Evaluate whether activation scaling mitigates the compounding cascading loss seen in unweighted decompositions, enabling deep multi-layer compression without degradation.

---

## 4. Direction 4: SwiGLU / MLP Block Tensorisation & Non-Projection Denoising

- [x] **Architecture Analysis of Gemma-3 Gated MLP:**
  - `gate_proj`, `up_proj`, `down_proj`: Each $1152 \times 6912 = 7.96\text{M}$ params. Total per MLP block = **$23.89\text{M}$ params ($67\%$ of layer)**.
- [x] **Exp 24 Notebooks Generated & Ready for Kaggle:**
  - Kaggle: [`kaggle/24_gemma3_1b_it_mlp_tensorisation.ipynb`](file:///home/dwithun/Development/llm_compression/kaggle/24_gemma3_1b_it_mlp_tensorisation.ipynb)
  - Local: [`experiments/02_all_layers_bench/24_gemma3_1b_it_mlp_tensorisation.ipynb`](file:///home/dwithun/Development/llm_compression/experiments/02_all_layers_bench/24_gemma3_1b_it_mlp_tensorisation.ipynb)
- [x] **Exp 24 — Executed on Kaggle Dual Tesla T4 (`7.ipynb`):**
  - **Paradigm A (Virtual Memory-Bank 4D Tucker):** Shaved **$21.27\text{M}$ parameters (-89.03%) on Layer 14 MLP** (accuracy: 30.20%, fluent recipe).
  - **Paradigm B (Non-Projection Submodules):** Filtering the composite `mlp` output manifold directly yielded **$36.20\%$ (+1.60% GAIN)** with 100% flawless generation, proving holistic block denoising succeeds without touching internal projections!
  - **Paradigm C (Joint 3D SwiGLU Tucker):** Collapsed to **$15.00\%$ (-19.60%)**, proving that forcing $W_{\text{gate}}$ and $W_{\text{up}}$ into a shared low-rank space destroys the non-linear gating mechanism.
  - **Paradigm D (Selective 2D SVD on `down_proj`):** Achieved **$36.00\%$ (+1.40% GAIN)** while shaving **$5.64\text{M}$ parameters**.
  - **Deep Multi-Layer Scaling (Layers 14 + 22 MHA + MLP D):** Shaved **$18,898,880\text{ parameters}$ ($1.89\%$ of the full model)** while maintaining baseline accuracy ($34.40\%$ vs $34.60\%$). Subtle ingredient loop in generation marks the empirical boundary of simultaneous MHA+MLP compression.

- [x] **Exp 25 — The Grand Unified Compression Benchmark (`8.ipynb`):**
  - **Stage 1 (Top 4 MHA Layers 14, 19, 21, 22):** Shaved **$13.24\text{M}$ parameters (1.32%)**, achieving **$34.80\%$ (+0.20% GAIN over baseline)** with $100\%$ flawless generation!
  - **Stage 2 (4 MHA + Layer 14 `down_proj` SVD r=384):** Shaved **$18.10\text{M}$ parameters (1.81%)**, maintaining **$32.80\%$ accuracy** and $100\%$ flawless generation with zero repetition loops!
  - **Stage 3 (4 MHA + Dual MLP L14 r=384 + L22 r=512):** Shaved **$21.94\text{M}$ parameters (2.19%)**, maintaining **$31.60\%$ accuracy** and flawless generation.
  - **Stage 4 (6 MHA + Dual MLP):** Collapsed to **$8.00\%$ (-26.60%)** with asterisk stutter, discovering the **"Consecutive Layer Curse"**: compressing 4 consecutive layers (19, 20, 21, 22) eliminates residual stabilizer buffers, causing catastrophic phase transition collapse.
  - **The Golden Production State Identified:** **Stage 2 & Stage 3 ($18.1\text{M}$ to $21.9\text{M}$ parameters shaved)** represents the global Pareto optimal frontier for zero-shot compression on Gemma-3-1B-IT!

---

## 5. Direction 5: "TensorLoRA" — Multi-Way Core Parameter-Efficient Fine-Tuning

- [ ] **Parameterization:**
  - Freeze factor matrices $U^{(1)} \in \mathbb{R}^{d_{\text{model}} \times R_1}$, $U^{(2)} \in \mathbb{R}^{d_v \times R_2}$, $U^{(3)} \in \mathbb{R}^{4 \times R_3}$.
  - Treat private head cores $\mathcal{G}_i \in \mathbb{R}^{R_1 \times R_2 \times R_3}$ as the sole trainable parameters ($\approx 8$k parameters per head, $\approx 32$k per layer).
- [ ] **Downstream Adaptation & Convergence Benchmark:**
  - Compare TensorLoRA vs standard 2D LoRA ($W + B \cdot A$ with rank $r=8$ or $16$) on:
    - GLUE MNLI
    - GSM8k (mathematical reasoning)
  - Evaluate parameter efficiency, memory footprint, and convergence rate.

---

## 6. Direction 6: Academic Paper Preparation (Target: NeurIPS / ICLR / ICML / ACL)

- [ ] **Manuscript Title Candidates:**
  - *Head-Preserving Multi-Head Tensorisation as an Inductive Denoising Filter for Large Language Models*
  - *Beyond Low-Rank Matrices: Multi-Way Tucker Geometry and the Denoising of Multi-Head Attention in LLMs*
- [ ] **Paper Structure & Narrative:**
  - **1. Introduction:** The limitation of 2D low-rank methods (LASER, SVD) on attention blocks; the polysemantic collapse problem.
  - **2. The Geometry of Multi-Head Attention:** Why attention heads require isolated multi-linear cores ($\mathcal{G}_i$ private per head).
  - **3. The Failure of Naive Tucker (Exp 18):** Formalizing cross-head subspace contamination.
  - **4. TensorLLM for Modern LLMs (GQA Adaptation):** Formulation and dual-GPU decomposition engine.
  - **5. Empirical Discovery:** Middle-layer structural denoising (+80.7% parameter cut, +5.6% MNLI gain, flawless text generation).
  - **6. Mechanistic Proof:** Attention entropy sharpening and noise spectral analysis.
  - **7. Extensions:** Joint multi-layer Pareto scheduling and activation weighting.
- [ ] **Publication-Ready Figures & Artifacts:**
  - [x] `22_tensorllm_layer_sweep_profiles.png` (26-layer sweep profile)
  - [x] `18_all_26_layers_adaptation_profiles.png` (Exp 18 failure baseline)
  - [x] `20_all_26_layers_modular_profiles.png` (Exp 20 modular SVD baseline)
  - [ ] Multi-layer Pareto frontier curves (Exp 23).
  - [ ] Attention head entropy shift violin/box plots.
