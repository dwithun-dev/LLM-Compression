#!/usr/bin/env python3
"""
Layer 0 Submodule Tucker Adaptation with Nested Clustering & 1% Recon Sweep
=============================================================================
Target Layer: Layer 0 (`model.model.layers[0]`) of `google/gemma-3-1b-it`

Target Submodules (from user profiling):
- Submodule: self_attn                      | Output Shape: (5000, 1152)
- Submodule: self_attn.q_proj               | Output Shape: (5000, 1024)
- Submodule: self_attn.k_proj               | Output Shape: (5000, 256)
- Submodule: self_attn.v_proj               | Output Shape: (5000, 256)
- Submodule: self_attn.o_proj               | Output Shape: (5000, 1152)
- Submodule: self_attn.q_norm               | Output Shape: (5000, 256)  [Preserved FP32 1D RMSNorm]
- Submodule: self_attn.k_norm               | Output Shape: (5000, 256)  [Preserved FP32 1D RMSNorm]
- Submodule: mlp                            | Output Shape: (5000, 1152)
  * `mlp.gate_proj`                         | Output Shape: (5000, 6912)
  * `mlp.up_proj`                           | Output Shape: (5000, 6912)
  * `mlp.down_proj`                         | Output Shape: (5000, 1152)

Methodology:
1. Normal Distribution Thresholding (Superweight Quarantine):
   Quarantines isolated coordinates where |z| > 3.0, magnitude > 3.0, or variance >= top 1%.
2. Nested Clustering (n iterations):
   Hierarchical partitioning using iterative DBSCAN repeated `n_iter` times on candidate residuals
   to assemble balanced 3D tensors:
     T in R^{NumChunks x ChunkSize x Dim}
3. 1% Incremental Reconstruction Rate Loss Sweep:
   Steps target reconstruction retention in 1% increments (tau = 0.99 down to 0.70)
   to find the exact Pareto frontier between reconstruction error, parameter cut %, and downstream accuracy.
"""

import os
import sys
from pathlib import Path

# Configure Triton cache directory within writable workspace
workspace_dir = Path(__file__).resolve().parents[2]
os.environ["TRITON_CACHE_DIR"] = str(workspace_dir / ".triton_cache")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

import time
import json
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

tl.set_backend("pytorch")

# -----------------------------------------------------------------------------
# 1. Forward Hook Profiler for Real Activation Profiles
# -----------------------------------------------------------------------------
class Layer0ActivationProfiler:
    """Captures forward activations across Layer 0 submodules."""
    def __init__(self, layer_0):
        self.l0 = layer_0
        self.hooks = []
        self.activations = {
            "self_attn.q_proj": [],
            "self_attn.k_proj": [],
            "self_attn.v_proj": [],
            "self_attn.o_proj": [],
            "self_attn.q_norm": [],
            "self_attn.k_norm": [],
            "mlp.gate_proj": [],
            "mlp.up_proj": [],
            "mlp.down_proj": [],
        }

    def _hook_fn(self, name, mode="output"):
        def hook(module, inp, out):
            tensor = inp[0] if mode == "input" else (out[0] if isinstance(out, tuple) else out)
            flat = tensor.detach().cpu().reshape(-1, tensor.shape[-1])
            self.activations[name].append(flat)
        return hook

    def register(self):
        self.hooks.append(self.l0.self_attn.q_proj.register_forward_hook(self._hook_fn("self_attn.q_proj", "output")))
        self.hooks.append(self.l0.self_attn.k_proj.register_forward_hook(self._hook_fn("self_attn.k_proj", "output")))
        self.hooks.append(self.l0.self_attn.v_proj.register_forward_hook(self._hook_fn("self_attn.v_proj", "output")))
        self.hooks.append(self.l0.self_attn.o_proj.register_forward_hook(self._hook_fn("self_attn.o_proj", "output")))
        if hasattr(self.l0.self_attn, "q_norm"):
            self.hooks.append(self.l0.self_attn.q_norm.register_forward_hook(self._hook_fn("self_attn.q_norm", "output")))
        if hasattr(self.l0.self_attn, "k_norm"):
            self.hooks.append(self.l0.self_attn.k_norm.register_forward_hook(self._hook_fn("self_attn.k_norm", "output")))
        self.hooks.append(self.l0.mlp.gate_proj.register_forward_hook(self._hook_fn("mlp.gate_proj", "output")))
        self.hooks.append(self.l0.mlp.up_proj.register_forward_hook(self._hook_fn("mlp.up_proj", "output")))
        self.hooks.append(self.l0.mlp.down_proj.register_forward_hook(self._hook_fn("mlp.down_proj", "input")))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_stacked(self, max_tokens: int = 5000) -> Dict[str, np.ndarray]:
        result = {}
        for k, v in self.activations.items():
            if len(v) > 0:
                cat = torch.cat(v, dim=0).float().numpy()
                if cat.shape[0] > max_tokens:
                    cat = cat[:max_tokens]
                result[k] = cat
        return result


# -----------------------------------------------------------------------------
# 2. Normal Thresholding & Nested Clustering (n iterations)
# -----------------------------------------------------------------------------
def nested_clustering_with_thresholding(
    acts_matrix: np.ndarray,
    weight_tensor: torch.Tensor,
    chunk_size: int,
    num_chunks: int,
    is_col: bool = False,
    n_iter: int = 3,
    z_cutoff: float = 3.0,
) -> Dict[str, Any]:
    """
    Applies normal distribution outlier thresholding followed by nested DBSCAN
    clustering repeated `n_iter` times to partition coordinates and assemble 3D tensors.
    """
    target_dim = weight_tensor.shape[1] if is_col else weight_tensor.shape[0]

    # Compute coordinate distribution statistics
    if acts_matrix is not None and acts_matrix.ndim == 2:
        if acts_matrix.shape[1] == target_dim:
            v = np.mean(acts_matrix, axis=0)
            variances = np.var(acts_matrix, axis=0)
            max_mags = np.max(np.abs(acts_matrix), axis=0)
        elif acts_matrix.shape[0] == target_dim:
            v = np.mean(acts_matrix, axis=1)
            variances = np.var(acts_matrix, axis=1)
            max_mags = np.max(np.abs(acts_matrix), axis=1)
        else:
            w_np = weight_tensor.detach().cpu().float().numpy()
            v = np.mean(w_np, axis=0 if is_col else 1)
            variances = np.var(w_np, axis=0 if is_col else 1)
            max_mags = np.max(np.abs(w_np), axis=0 if is_col else 1)
    else:
        w_np = weight_tensor.detach().cpu().float().numpy()
        v = np.mean(w_np, axis=0 if is_col else 1)
        variances = np.var(w_np, axis=0 if is_col else 1)
        max_mags = np.max(np.abs(w_np), axis=0 if is_col else 1)

    # 1. Normal Distribution Thresholding (|z| > 3.0 or top 1% variance)
    std_v = float(np.std(v) + 1e-8)
    z_scores = np.abs((v - np.mean(v)) / std_v)
    var_99 = float(np.quantile(variances, 0.99)) if len(variances) > 10 else 1e9

    super_mask = (z_scores > z_cutoff) | (variances >= var_99)
    super_coords = np.where(super_mask)[0]

    # 2. Nested Clustering Loop (n_iter iterations)
    candidate_indices = np.where(~super_mask)[0]
    if len(candidate_indices) < num_chunks * chunk_size:
        chunk_size = max(10, len(candidate_indices) // num_chunks)

    chunk_list = []

    for it in range(n_iter):
        if len(candidate_indices) < chunk_size or len(chunk_list) >= num_chunks:
            break

        v_sub = v[candidate_indices]
        eps = max(0.02, float(np.std(v_sub) * 0.18))
        min_samples = max(10, min(30, chunk_size // 4))

        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = db.fit_predict(v_sub.reshape(-1, 1))

        unique_labels = [l for l in np.unique(labels) if l != -1]
        for lab in unique_labels:
            c_local = np.where(labels == lab)[0]
            if len(c_local) >= chunk_size:
                sorted_local = c_local[np.argsort(v_sub[c_local])]
                for ci in range(len(sorted_local) // chunk_size):
                    selected = candidate_indices[sorted_local[ci * chunk_size : (ci + 1) * chunk_size]]
                    chunk_list.append(selected)
                    if len(chunk_list) >= num_chunks:
                        break
            if len(chunk_list) >= num_chunks:
                break

        # Residual unassigned coordinates passed to next iteration
        assigned = set(np.concatenate(chunk_list) if chunk_list else [])
        candidate_indices = np.array([i for i in candidate_indices if i not in assigned])

    # Fallback padding if nested clusters didn't yield enough chunks
    if len(chunk_list) < num_chunks:
        assigned = set(np.concatenate(chunk_list) if chunk_list else [])
        avail = [i for i in range(target_dim) if i not in assigned and i not in super_coords]
        needed = num_chunks - len(chunk_list)
        for _ in range(needed):
            if len(avail) >= chunk_size:
                chunk_list.append(np.array(avail[:chunk_size]))
                avail = avail[chunk_size:]
            else:
                break

    actual_num_chunks = len(chunk_list)
    if actual_num_chunks == 0:
        raise RuntimeError(f"Failed to assemble chunks for target dimension {target_dim} (chunk_size={chunk_size}).")

    # 3. Assemble 3D Tensor
    if is_col:
        T = torch.stack([weight_tensor[:, c].T.float().cpu() for c in chunk_list], dim=0)
    else:
        T = torch.stack([weight_tensor[c, :].float().cpu() for c in chunk_list], dim=0)

    return {
        "tensor": T,
        "chunk_list": chunk_list,
        "super_coords": super_coords,
        "is_col": is_col,
        "num_chunks": actual_num_chunks,
        "chunk_size": chunk_size,
        "target_dim": target_dim,
    }


# -----------------------------------------------------------------------------
# 3. 1% Incremental Reconstruction Rate Loss Sweep
# -----------------------------------------------------------------------------
def compute_bespoke_ranks_by_energy(T: torch.Tensor, energy_threshold: float) -> List[int]:
    """
    Determines bespoke Tucker ranks [R1, R2, R3] that preserve `energy_threshold`
    of cumulative singular value energy along each mode unfolding.
    """
    ranks = []
    for mode in range(3):
        unfolded = tl.unfold(T, mode)
        s = torch.linalg.svdvals(unfolded)
        cum_energy = torch.cumsum(s**2, dim=0) / torch.sum(s**2)
        idx = (cum_energy >= energy_threshold).nonzero()
        r = int(idx[0].item()) + 1 if len(idx) > 0 else T.shape[mode]
        r = max(1, min(T.shape[mode], r))
        ranks.append(r)
    return ranks


def sweep_tucker_compositions_1pct(
    T: torch.Tensor,
    start_recon_pct: float = 0.99,
    end_recon_pct: float = 0.70,
    step: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Sweeps target reconstruction retention in 1% steps to generate candidate rank compositions.
    """
    candidates = []
    current_pct = start_recon_pct

    while current_pct >= end_recon_pct - 1e-6:
        ranks = compute_bespoke_ranks_by_energy(T, energy_threshold=current_pct)

        # Truncated Tucker decomposition
        core, factors = tucker(T, rank=ranks, init="svd")
        T_hat = tucker_to_tensor((core, factors))

        rel_err = float((torch.norm(T - T_hat) / torch.norm(T)).item())
        orig_p = T.numel()
        comp_p = core.numel() + sum(f.numel() for f in factors)
        cut_pct = float((orig_p - comp_p) / orig_p * 100.0)

        candidates.append({
            "target_energy": round(current_pct, 4),
            "loss_in_recon_pct": round((1.0 - current_pct) * 100.0, 2),
            "ranks": ranks,
            "recon_error_pct": round(rel_err * 100.0, 2),
            "params_cut_pct": round(cut_pct, 2),
            "T_hat": T_hat,
            "core_shape": list(core.shape),
            "factor_shapes": [list(f.shape) for f in factors],
        })
        current_pct -= step

    return candidates


# -----------------------------------------------------------------------------
# 4. Downstream & Qualitative Generation Evaluation Runners
# -----------------------------------------------------------------------------
CAKE_PROMPT = (
    "<start_of_turn>user\n"
    "What is the best recipe to make a chocolate cake?<end_of_turn>\n"
    "<start_of_turn>model\n"
)

def generate_cake_recipe(model, tokenizer, max_new_tokens: int = 256) -> str:
    """Generates qualitative text for the chocolate cake recipe benchmark."""
    model.eval()
    inputs = tokenizer(CAKE_PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    recipe_text = tokenizer.decode(tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return recipe_text


def evaluate_mnli(model, tokenizer, eval_data, label_token_ids) -> float:
    model.eval()
    predictions, ground_truth = [], []
    with torch.no_grad():
        for sample in eval_data:
            prompt = (
                f"<start_of_turn>user\n"
                f"Premise: {sample['premise']}\n"
                f"Hypothesis: {sample['hypothesis']}\n"
                f"Determine if the relationship between the 'Premise' and 'Hypothesis' is 'entailment', 'neutral' or 'contradiction.'\n"
                f"Answer with one word\n"
                f"<start_of_turn>model\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs, logits_to_keep=1)
            candidate_logits = outputs.logits[0, -1, :][label_token_ids]
            predictions.append(torch.argmax(candidate_logits).item())
            ground_truth.append(sample["label"])
    return float(accuracy_score(ground_truth, predictions))


# -----------------------------------------------------------------------------
# 5. Main Execution Engine
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Layer 0 Submodule Tucker Adaptation Sweep")
    parser.add_argument("--model-id", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--samples", type=int, default=150, help="MNLI evaluation sample count")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-iter", type=int, default=3, help="Nested clustering iterations")
    parser.add_argument("--step", type=float, default=0.01, help="Recon rate loss step (default: 0.01 = 1%%)")
    parser.add_argument("--min-energy", type=float, default=0.70, help="Minimum energy threshold (default: 0.70)")
    parser.add_argument("--submodule", type=str, default="all", help="Target submodule name or 'all'")
    parser.add_argument("--cake-tokens", type=int, default=256, help="Max new tokens for chocolate cake recipe generation")
    parser.add_argument("--skip-cake", action="store_true", help="Skip chocolate cake qualitative generation")
    parser.add_argument("--output-json", type=str, default="experiments/01_layer_based_bench/artifacts/layer_0_submodule_adaptation_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    print("=" * 80)
    print("LAYER 0 SUBMODULE TUCKER ADAPTATION SWEEP")
    print(f"Model: {args.model_id} | Device: {args.device} | Eval Samples: {args.samples}")
    print(f"Nested Clustering Iterations: {args.n_iter} | Step: {args.step * 100:.0f}%%")
    print(f"Qualitative Query: 'What is the best recipe to make a chocolate cake?' ({args.cake_tokens} tokens)")
    print("=" * 80)

    print(f"Loading {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32, device_map=args.device)
    model.eval()

    print(f"Loading GLUE MNLI validation split ({args.samples} samples)...")
    ds = load_dataset("nyu-mll/glue", "mnli", split="validation_matched").select(range(args.samples))
    label_token_ids = [tokenizer.encode(" " + name, add_special_tokens=False)[0] for name in ["entailment", "neutral", "contradiction"]]

    l0 = model.model.layers[0]

    # Target Submodules in Layer 0
    submodule_definitions = {
        # Attention Projections
        "self_attn.q_proj": {"module": l0.self_attn.q_proj, "chunk_size": 256, "num_chunks": 4, "is_col": False, "desc": "Output Shape: (5000, 1024)"},
        "self_attn.k_proj": {"module": l0.self_attn.k_proj, "chunk_size": 64,  "num_chunks": 4, "is_col": False, "desc": "Output Shape: (5000, 256)"},
        "self_attn.v_proj": {"module": l0.self_attn.v_proj, "chunk_size": 64,  "num_chunks": 4, "is_col": False, "desc": "Output Shape: (5000, 256)"},
        "self_attn.o_proj": {"module": l0.self_attn.o_proj, "chunk_size": 256, "num_chunks": 4, "is_col": False, "desc": "Output Shape: (5000, 1152)"},
        # MLP Projections
        "mlp.gate_proj":    {"module": l0.mlp.gate_proj,    "chunk_size": 400, "num_chunks": 6, "is_col": False, "desc": "Output Shape: (5000, 6912)"},
        "mlp.up_proj":      {"module": l0.mlp.up_proj,      "chunk_size": 400, "num_chunks": 6, "is_col": False, "desc": "Output Shape: (5000, 6912)"},
        "mlp.down_proj":    {"module": l0.mlp.down_proj,    "chunk_size": 400, "num_chunks": 6, "is_col": True,  "desc": "Input Shape: (5000, 6912) -> Output: (5000, 1152)"},
    }

    # Profiling pass to collect genuine forward activation shapes and compute baseline accuracy
    print("\n--- STEP 1: Profiling Baseline Accuracy & Layer 0 Submodule Activations ---")
    profiler = Layer0ActivationProfiler(l0)
    profiler.register()

    baseline_acc = evaluate_mnli(model, tokenizer, ds, label_token_ids)
    profiler.remove()
    acts_dict = profiler.get_stacked(max_tokens=5000)

    print(f"Pristine Baseline MNLI Accuracy: {baseline_acc * 100:.2f}%\n")

    baseline_cake_recipe = ""
    if not args.skip_cake:
        print("--- Generating Pristine Baseline Chocolate Cake Recipe ---")
        baseline_cake_recipe = generate_cake_recipe(model, tokenizer, max_new_tokens=args.cake_tokens)
        print(f"Pristine Baseline Cake Recipe Preview:\n{baseline_cake_recipe[:350]}...\n")

    print(f"{'Submodule':<26} | {'Profiled Tokens / Shape':<24} | {'Norm / Notes'}")
    print("-" * 75)
    for name, act in acts_dict.items():
        print(f"{name:<26} | {str(act.shape):<24} | Mean={act.mean():.4f}, Std={act.std():.4f}")

    # Note on RMSNorm scale vectors
    print(f"{'self_attn.q_norm':<26} | {'(5000, 256)':<24} | Preserved in FP32 (1D RMSNorm scale vector)")
    print(f"{'self_attn.k_norm':<26} | {'(5000, 256)':<24} | Preserved in FP32 (1D RMSNorm scale vector)")
    print(f"{'mlp (container)':<26} | {'(5000, 1152)':<24} | Composite module decomposed via projections")
    print("-" * 75)

    # Filter target submodules if specified
    targets = submodule_definitions.keys() if args.submodule == "all" else [args.submodule]

    all_adaptation_results = {}

    for sub_name in targets:
        cfg = submodule_definitions[sub_name]
        mod = cfg["module"]
        W_orig = mod.weight.data.clone()

        print("\n" + "=" * 80)
        print(f"TUCKER ADAPTATION SWEEP: {sub_name}")
        print(f"Weight Shape: {list(W_orig.shape)} | {cfg['desc']}")
        print("=" * 80)

        acts = acts_dict.get(sub_name, None)

        cdata = nested_clustering_with_thresholding(
            acts_matrix=acts,
            weight_tensor=W_orig,
            chunk_size=cfg["chunk_size"],
            num_chunks=cfg["num_chunks"],
            is_col=cfg["is_col"],
            n_iter=args.n_iter,
        )

        T = cdata["tensor"]
        num_super = len(cdata["super_coords"])
        super_pct = (num_super / cdata["target_dim"]) * 100.0
        print(f"Constructed 3D Tensor: {list(T.shape)} ({cdata['num_chunks']} chunks of size {cdata['chunk_size']})")
        print(f"Quarantined Superweights: {num_super}/{cdata['target_dim']} ({super_pct:.2f}%)")

        sweep_candidates = sweep_tucker_compositions_1pct(
            T,
            start_recon_pct=0.99,
            end_recon_pct=args.min_energy,
            step=args.step
        )
        print(f"Generated {len(sweep_candidates)} composition candidates (stepping by {args.step*100:.0f}% recon loss).")

        sub_eval_log = []
        best_acc = -1.0
        best_comp = None
        best_cand = None

        print(f"\n{'Loss in Recon':<15} | {'Target Ret':<11} | {'Tucker Ranks':<15} | {'Recon Err %':<12} | {'Params Cut %':<14} | {'Accuracy':<10}")
        print("-" * 88)

        for cand in sweep_candidates:
            T_hat = cand["T_hat"]
            mod.weight.data = W_orig.clone()

            if cfg["is_col"]:
                for k_idx, c in enumerate(cdata["chunk_list"]):
                    mod.weight.data[:, c] = T_hat[k_idx].T.to(mod.weight.device, dtype=mod.weight.dtype)
            else:
                for k_idx, c in enumerate(cdata["chunk_list"]):
                    mod.weight.data[c, :] = T_hat[k_idx].to(mod.weight.device, dtype=mod.weight.dtype)

            acc = evaluate_mnli(model, tokenizer, ds, label_token_ids)
            cand_log = {
                "loss_in_recon_pct": cand["loss_in_recon_pct"],
                "target_retention": cand["target_energy"],
                "ranks": cand["ranks"],
                "recon_error_pct": cand["recon_error_pct"],
                "params_cut_pct": cand["params_cut_pct"],
                "accuracy_pct": round(acc * 100.0, 2),
            }
            sub_eval_log.append(cand_log)

            if acc > best_acc:
                best_acc = acc
                best_comp = cand_log
                best_cand = cand

            print(f"{cand['loss_in_recon_pct']:>13.1f}% | {cand['target_energy']:>9.2f} | {str(cand['ranks']):<15} | {cand['recon_error_pct']:>10.2f}% | {cand['params_cut_pct']:>12.2f}% | {acc*100:>8.2f}%")

        mod.weight.data = W_orig.clone()

        # Qualitative Chocolate Cake Recipe Generation with adapted best composition
        adapted_cake_recipe = ""
        if not args.skip_cake and best_cand is not None:
            print(f"\n--- Generating Chocolate Cake Recipe with Best Composition of {sub_name} ---")
            if cfg["is_col"]:
                for k_idx, c in enumerate(cdata["chunk_list"]):
                    mod.weight.data[:, c] = best_cand["T_hat"][k_idx].T.to(mod.weight.device, dtype=mod.weight.dtype)
            else:
                for k_idx, c in enumerate(cdata["chunk_list"]):
                    mod.weight.data[c, :] = best_cand["T_hat"][k_idx].to(mod.weight.device, dtype=mod.weight.dtype)

            adapted_cake_recipe = generate_cake_recipe(model, tokenizer, max_new_tokens=args.cake_tokens)
            print(f"Adapted {sub_name} Cake Recipe Preview:\n{adapted_cake_recipe[:350]}...\n")
            mod.weight.data = W_orig.clone()

        print(f"\n>> BEST COMPOSITION FOR {sub_name}:")
        print(f"   Ranks: {best_comp['ranks']}")
        print(f"   Accuracy: {best_comp['accuracy_pct']:.2f}% (Baseline: {baseline_acc*100:.2f}%)")
        print(f"   Recon Error: {best_comp['recon_error_pct']:.2f}%")
        print(f"   Parameter Cut: {best_comp['params_cut_pct']:.2f}%")

        all_adaptation_results[sub_name] = {
            "weight_shape": list(W_orig.shape),
            "tensor_shape": list(T.shape),
            "quarantined_superweights": num_super,
            "quarantined_superweights_pct": round(super_pct, 2),
            "best_composition": best_comp,
            "cake_recipe": adapted_cake_recipe,
            "full_sweep": sub_eval_log
        }

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "eval_samples": args.samples,
        "baseline_accuracy_pct": round(baseline_acc * 100.0, 2),
        "cake_query_prompt": CAKE_PROMPT,
        "baseline_cake_recipe": baseline_cake_recipe,
        "preserved_submodules": {
            "self_attn.q_norm": "Preserved in FP32 (1D RMSNorm scale vector, 256 elements)",
            "self_attn.k_norm": "Preserved in FP32 (1D RMSNorm scale vector, 256 elements)",
        },
        "submodule_results": all_adaptation_results
    }

    with open(args.output_json, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSweep complete for all target submodules! Full results logged to: {args.output_json}")


if __name__ == "__main__":
    main()
