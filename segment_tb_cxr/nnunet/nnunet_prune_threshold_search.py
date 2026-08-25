"""
Binary search for the maximum weight-pruning percentile such that fewer than
`pair_fail_fraction` of images have a Dice reduction > `pair_dice_reduction_threshold`
compared to the unpruned model. Saves the pruned checkpoint alongside the original.

Usage:
    python -m segment_tb_cxr.nnunet.nnunet_prune_threshold_search <input_csv> <weights_path> \\
        [--image_file_column filename] \\
        [--reference_file_column ref_seg_filename] \\
        [--pair_dice_reduction_threshold 0.05] [--pair_fail_fraction 0.10] \\
        [--binary_mask_threshold 0.5] [--convergence_tolerance 0.1]

"""

import argparse
import contextlib
import io
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

# suppress notifications from nnunet during import
with contextlib.redirect_stdout(io.StringIO()):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m import gen_nnunet_prob_map
from segment_tb_cxr.evaluation.evaluate_segmentations import (
    _compute_metrics_from_images,
)


def file_path(path):
    p = Path(path)
    if p.is_file():
        return p
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a file path or file does not exist."
        )


def _build_predictor(weights_folder, checkpoint_name, device):
    predictor = nnUNetPredictor(device=torch.device(device), allow_tqdm=False)
    with contextlib.redirect_stdout(io.StringIO()):
        predictor.initialize_from_trained_model_folder(
            weights_folder, checkpoint_name=checkpoint_name, use_folds=(0,)
        )
    return predictor


def _load_state_dict(predictor, state_dict):
    """Load state dict into the predictor network, handling torch.compile prefix."""
    try:
        predictor.network.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        predictor.network.load_state_dict(stripped)


def _prune(state_dict, percentile, all_abs):
    """Zero weights whose absolute value falls below the given global percentile."""
    threshold = float(np.percentile(all_abs, percentile))
    pruned = {
        k: (v * (v.abs() >= threshold) if v.is_floating_point() else v)
        for k, v in state_dict.items()
    }
    total_float = sum(t.numel() for t in state_dict.values() if t.is_floating_point())
    n_zero = sum(
        (pruned[k] == 0).sum().item()
        for k in pruned
        if state_dict[k].is_floating_point()
    )
    return pruned, threshold, n_zero / total_float


def _dice_scores(predictor, image_files, ref_files, mask_threshold):
    """Returns per-image Dice scores as a numpy array."""
    scores = []
    for img, ref in zip(image_files, ref_files):
        prob = gen_nnunet_prob_map(img, predictor)
        ref_image = sitk.ReadImage(ref)
        pred_image = sitk.GetImageFromArray((prob > mask_threshold).astype(np.uint8))
        pred_image.CopyInformation(ref_image)
        scores.append(_compute_metrics_from_images(ref_image, pred_image)["dice"])
    return np.array(scores)


def _within_pair_reduction_limit(
    baseline_scores, pruned_scores, pair_dice_reduction_threshold, pair_fail_fraction
):
    dice_reductions = np.maximum(0.0, baseline_scores - pruned_scores)
    fail_rate = np.sum(dice_reductions > pair_dice_reduction_threshold) / len(
        baseline_scores
    )
    return fail_rate < pair_fail_fraction, fail_rate


def _plot_weight_histograms(sd, output_pdf, title):
    """Save a weight histogram for a single model to a PDF."""
    matplotlib.use(
        "Agg"
    )  # no-op if backend already set; guards no-display environments
    import matplotlib.pyplot as plt  # must follow matplotlib.use()

    weights = np.concatenate(
        [t.float().numpy().ravel() for t in sd.values() if t.is_floating_point()]
    )
    bins = np.linspace(weights.min(), weights.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    # weight distributions are sharply peaked near zero, particularly after prunning,
    # so using a log scale otherwise all other values would not be visible
    ax.hist(
        weights, bins=bins, log=True, color="steelblue", edgecolor="none", alpha=0.8
    )
    ax.set_title(title)
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count (log scale)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Saved weight histogram: {output_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=file_path)
    parser.add_argument("weights_path", type=file_path)
    parser.add_argument(
        "--image_file_column", default="filename", help="Path to CXR file"
    )
    parser.add_argument(
        "--reference_file_column",
        default="ref_seg_filename",
        help="Path to reference semantic segmentation file",
    )
    parser.add_argument(
        "--pair_dice_reduction_threshold",
        type=float,
        default=0.05,
        help="Per-image absolute Dice reduction that counts as a failure (default: 0.05)",
    )
    parser.add_argument(
        "--pair_fail_fraction",
        type=float,
        default=0.10,
        help="Max fraction of images allowed to fail (default: 0.10)",
    )
    parser.add_argument("--binary_mask_threshold", type=float, default=0.5)
    parser.add_argument("--convergence_tolerance", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = args.weights_path
    weights_folder = str(weights_path.parent.parent)
    checkpoint_name = weights_path.name

    df = pd.read_csv(args.input_csv)
    required_columns = set([args.image_file_column, args.reference_file_column])
    if not required_columns.issubset(set(df.columns.tolist())):
        parser.error(
            f"{args.input_csv} is missing columns: {required_columns - set(df.columns)}"
        )
    image_files = df[args.image_file_column].tolist()
    ref_files = df[args.reference_file_column].tolist()

    checkpoint = torch.load(args.weights_path, map_location="cpu", weights_only=False)
    orig_sd = checkpoint.get("network_weights", checkpoint)
    total_params = sum(t.numel() for t in orig_sd.values() if t.is_floating_point())
    # tensors are on CPU (map_location="cpu"); cast to float32 for bfloat16 compatibility
    float_tensors = [
        t.float().abs().numpy().ravel()
        for t in orig_sd.values()
        if t.is_floating_point()
    ]
    if not float_tensors:
        parser.error("Checkpoint contains no floating-point tensors — cannot prune.")
    all_abs = np.concatenate(float_tensors)

    predictor = _build_predictor(weights_folder, checkpoint_name, device)

    baseline_scores = _dice_scores(
        predictor, image_files, ref_files, args.binary_mask_threshold
    )

    # Binary search for the maximal percentage of pruning that
    # satisfies our constraints. Fraction of images in which
    # the Dice score was reduced by pair_dice_reduction_threshold is
    # less than pair_fail_fraction.
    low, high = 0.0, 99.0
    best_percentile = 0.0
    step = 0
    while high - low > args.convergence_tolerance:
        mid = (low + high) / 2.0
        step += 1
        pruned_sd, thr, sparsity = _prune(orig_sd, mid, all_abs)
        _load_state_dict(predictor, pruned_sd)

        pruned_scores = _dice_scores(
            predictor, image_files, ref_files, args.binary_mask_threshold
        )
        safe, fail_rate = _within_pair_reduction_limit(
            baseline_scores,
            pruned_scores,
            args.pair_dice_reduction_threshold,
            args.pair_fail_fraction,
        )

        print(
            f"step {step:2d} | percentile={mid:6.2f} | threshold={thr:.2e} "
            f"| sparsity={sparsity:.1%} | fail_rate={fail_rate:.1%} | {'OK' if safe else 'TOO MUCH'}"
        )

        if safe:
            best_percentile = mid
            low = mid
        else:
            high = mid

    _load_state_dict(predictor, orig_sd)

    # Final evaluation on the most aggressively pruned model that met the constraint
    pruned_sd, best_thr, best_sparsity = _prune(orig_sd, best_percentile, all_abs)
    _load_state_dict(predictor, pruned_sd)
    pruned_scores = _dice_scores(
        predictor, image_files, ref_files, args.binary_mask_threshold
    )

    # Save pruned checkpoint
    output_path = weights_path.with_stem(weights_path.stem + "_pruned")
    pruned_checkpoint = (
        {**checkpoint, "network_weights": pruned_sd}
        if "network_weights" in checkpoint
        else pruned_sd
    )
    torch.save(pruned_checkpoint, output_path)

    nonzero_params = total_params - int(best_sparsity * total_params)
    print(f"\n{'='*60}")
    print(
        f"Baseline | params={total_params:,} | dice={np.mean(baseline_scores):.4f}\u00b1{np.std(baseline_scores):.4f}"
    )
    print(
        f"Pruned   | params={nonzero_params:,} | dice={np.mean(pruned_scores):.4f}\u00b1{np.std(pruned_scores):.4f}"
        f" | percentile={best_percentile:.2f} | threshold={best_thr:.2e} | sparsity={best_sparsity:.2%}"
    )
    print(f"Saved pruned checkpoint: {output_path}")

    _plot_weight_histograms(
        orig_sd,
        weights_path.with_suffix(".pdf"),
        "Original nnUNet",
    )
    _plot_weight_histograms(
        pruned_sd,
        output_path.with_suffix(".pdf"),
        f"Pruned nnUNet (percentile={best_percentile:.2f}, threshold={best_thr:.2e})",
    )


if __name__ == "__main__":
    main()
