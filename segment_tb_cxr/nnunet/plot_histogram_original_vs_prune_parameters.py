import argparse
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import torch
from segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m import file_path
from segment_tb_cxr.nnunet.nnunet_prune_threshold_search import _build_predictor


def extract_float_parameters(checkpoint_path, param_keys, device):
    """
    Extract floating-point parameters from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        param_keys (list): List of parameter keys to extract.
        device (torch.device): Device to map the checkpoint to.

    Returns:
        np.ndarray: Flattened array of extracted floating-point parameters.
    """
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ck.get("network_weights", ck)

    params = np.concatenate(
        [
            t.float().detach().cpu().numpy().ravel()
            for k, t in state_dict.items()
            if k in param_keys and t.is_floating_point()
        ]
    )

    return params


def plot_histogram(
    original_model_params,
    pruned_model_params,
    output_path,
    bins,
    xlim=None,
):
    """
    Plot a single histogram of model parameters before and after pruning.

    Args:
        original_model_params (np.ndarray): Parameters of the original model.
        pruned_model_params (np.ndarray): Parameters of the pruned model.
        output_path (str or pathlib.Path): Path to save the output plot.
        bins (int or array-like): Number of bins, or bin edges, for the histogram.
        xlim (tuple, optional): If provided, sets the X-axis limits as (min, max).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    all_values = np.concatenate([original_model_params, pruned_model_params])
    bin_count = np.histogram_bin_edges(all_values, bins=bins)
    ax.hist(
        pruned_model_params,
        bins=bin_count,
        linewidth=3.0,
        color="orange",
        label="pruned",
        alpha=0.5,
    )
    ax.hist(
        original_model_params,
        bins=bin_count,
        linewidth=1.5,
        linestyle="--",
        color="blue",
        label="original",
        alpha=0.5,
    )
    ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="both", which="major", labelsize=10)
    leg = ax.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot weight parameter distribution before and after pruning."
    )
    parser.add_argument(
        "original_weights_path",
        type=file_path,
        help="Path to trained nnUNet model folder.",
    )
    parser.add_argument(
        "pruned_weights_path",
        type=file_path,
        help="Path to pruned nnUNet model checkpoint.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where output plots will be saved.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of bins for full histogram.",
    )
    parser.add_argument(
        "--zoom_threshold",
        type=float,
        default=0.0001,
        help="This value will zoom in X-axis of the first histogram around zero \
                from -zoom_threshold to +zoom_threshold.",
    )
    parser.add_argument(
        "--zoom_bins",
        type=int,
        default=1000,
        help="Number of bins for zoomed histogram.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = _build_predictor(
        pathlib.Path(args.original_weights_path).parent.parent,
        pathlib.Path(args.original_weights_path).name,
        device,
    )
    param_keys = {name for name, _ in predictor.network.named_parameters()}

    original_model_params = extract_float_parameters(
        args.original_weights_path, param_keys, device
    )
    pruned_model_params = extract_float_parameters(
        args.pruned_weights_path, param_keys, device
    )

    # Plot histograms of original vs pruned model parameters
    plot_histogram(
        original_model_params,
        pruned_model_params,
        output_path=str(pathlib.Path(args.output_dir) / "parameter_histogram.pdf"),
        bins=args.bins,
    )

    # Plot histograms of original vs pruned model parameters zoomed in around zero
    plot_histogram(
        original_model_params[np.abs(original_model_params) <= args.zoom_threshold],
        pruned_model_params[np.abs(pruned_model_params) <= args.zoom_threshold],
        output_path=str(
            pathlib.Path(args.output_dir) / "parameter_histogram_zoomed.pdf"
        ),
        bins=args.zoom_bins,
        xlim=[-args.zoom_threshold, args.zoom_threshold],
    )


if __name__ == "__main__":
    main()
