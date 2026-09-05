"""Headless evaluation script for Experiment 01 (Layer 0 Activation & Compression Bench).

Loads Gemma 3 1B IT, benchmarks on GLUE MNLI, extracts Layer 0 activation distributions,
and runs the Tucker decomposition pipeline.
"""

import argparse
import sys
import torch
from datasets import load_dataset
from neural_decomp import ModelManagementInterface, DeviceMapOptions, ActivationHookManager
from neural_decomp.profiling import summarize_activations, partition_inactive_neurons
from neural_decomp.decomposition import decompose_svd, compute_spectral_energy, apply_magnitude_sparsity, decompose_tucker
from neural_decomp.evaluation import evaluate_mnli
from neural_decomp.utils import print_section, save_json_metrics


def main():
    parser = argparse.ArgumentParser(description="Layer 0 Activation & Tucker Compression Benchmark")
    parser.add_argument("--model-id", type=str, default="google/gemma-3-1b-it", help="Model name or path")
    parser.add_argument("--samples", type=int, default=1000, help="Evaluation sample count")
    parser.add_argument("--inactive-neurons", type=int, default=4500, help="Number of inactive neurons to isolate")
    parser.add_argument("--output-json", type=str, default="results.json", help="Path to save metrics")
    args = parser.parse_args()

    print_section("Initializing Model & Dataset")
    mmi = ModelManagementInterface(model_id=args.model_id, precision=torch.float32, device_map=DeviceMapOptions.AUTO)
    model = mmi.get_model()
    tokenizer = mmi.get_tokenizer()

    ds = load_dataset("nyu-mll/glue", "mnli")["validation_matched"]

    print_section("Registering Hooks & Running Baseline MNLI")
    target_layer = model.model.layers[0]
    hook_mgr = ActivationHookManager(target_layer)

    with hook_mgr:
        baseline_results = evaluate_mnli(
            model=model,
            tokenizer=tokenizer,
            dataset_split=ds,
            sample_limit=args.samples,
            desc="Baseline Evaluation",
        )

    print(f"Baseline Accuracy: {baseline_results['accuracy'] * 100:.2f}%")
    stacked_acts = hook_mgr.get_stacked_activations()

    print_section("Activation Profiling")
    act_fn_acts = stacked_acts.get("mlp.act_fn", None)
    if act_fn_acts is not None:
        stats = summarize_activations(act_fn_acts)
        print(f"MLP act_fn Dead Ratio (<0.05): {stats['dead_ratio_pct']:.2f}%")
        inactive_idx, active_idx, _ = partition_inactive_neurons(act_fn_acts, args.inactive_neurons)
        print(f"Isolated {len(inactive_idx)} inactive neurons out of {act_fn_acts.shape[1]}")

        W_orig = target_layer.mlp.gate_proj.weight.data.clone()
        W_inactive = W_orig[inactive_idx, :].float().cpu()

        print_section("SVD Spectral Analysis")
        U, S, Vh = decompose_svd(W_inactive)
        energy_ranks = compute_spectral_energy(S)
        print(f"Rank for 95% Energy: {energy_ranks['rank_95']} / {len(S)}")

        print_section("Denoising & Magnitude Sparsification")
        W_denoised = U[:, :energy_ranks['rank_95']] @ torch.diag(S[:energy_ranks['rank_95']]) @ Vh[:energy_ranks['rank_95'], :]
        W_sparse, sparsity_pct = apply_magnitude_sparsity(W_denoised, quantile_threshold=0.60)
        print(f"Achieved Sparsity in Inactive Slice: {sparsity_pct:.2f}% exact zeros")

        print_section("Tucker Tensor Decomposition")
        tensor_4d = W_sparse.reshape(45, 100, 24, 48)
        core, factors, recon, tucker_metrics = decompose_tucker(tensor_4d, rank=[30, 60, 16, 24])
        print(f"Compression Ratio: {tucker_metrics['compression_ratio']:.2f}x")
        print(f"Parameters Eliminated in Slice: {tucker_metrics['eliminated_pct']:.2f}%")

        # Inject into model
        W_recon = recon.reshape(W_inactive.shape).to(device=model.device, dtype=target_layer.mlp.gate_proj.weight.dtype)
        target_layer.mlp.gate_proj.weight.data[inactive_idx, :] = W_recon

        print_section("Re-Evaluating Compressed Model")
        comp_results = evaluate_mnli(
            model=model,
            tokenizer=tokenizer,
            dataset_split=ds,
            sample_limit=args.samples,
            desc="Compressed Evaluation",
        )
        print(f"Compressed Accuracy: {comp_results['accuracy'] * 100:.2f}%")

        metrics = {
            "baseline_accuracy": baseline_results["accuracy"],
            "compressed_accuracy": comp_results["accuracy"],
            "tucker_metrics": tucker_metrics,
            "sparsity_pct": sparsity_pct,
        }
        save_json_metrics(metrics, args.output_json)
        print(f"Metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
