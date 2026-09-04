import argparse
import contextlib
import io
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m import file_path


def get_param_keys(model_folder, checkpoint_name, device):
    predictor = nnUNetPredictor(device=torch.device(device), allow_tqdm=False)
    with contextlib.redirect_stdout(io.StringIO()):
        predictor.initialize_from_trained_model_folder(
            model_folder, checkpoint_name=checkpoint_name, use_folds=(0,)
        )
    return {name for name, _ in predictor.network.named_parameters()}


def extract_float_parameters(checkpoint_path, param_keys, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ck.get("network_weights", ck)

    params = np.concatenate(
        [
            t.float().detach().cpu().numpy().ravel()
            for k, t in state_dict.items()
            if k in param_keys and t.is_floating_point()
        ]
    )

    n_nonzero = sum(
        torch.count_nonzero(t).item()
        for k, t in state_dict.items()
        if k in param_keys and t.is_floating_point()
    )

    return params, n_nonzero


def plot_histograms(
    before_params, after_params, output_dir, bins, zoom_threshold, zoom_bins
):

    all_values = np.concatenate([before_params, after_params])
    bin_edges = np.histogram_bin_edges(all_values, bins=bins)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        after_params,
        bins=bin_edges,
        linewidth=3.0,
        color="orange",
        label="pruned",
        alpha=0.5,
    )
    ax.hist(
        before_params,
        bins=bin_edges,
        linewidth=1.5,
        linestyle="--",
        color="blue",
        label="original",
        alpha=0.5,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    leg = ax.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(
        str(pathlib.Path(output_dir) / "parameter_histogram.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    before_zoomed = before_params[np.abs(before_params) <= zoom_threshold]
    after_zoomed = after_params[np.abs(after_params) <= zoom_threshold]
    zoomed_bin_edges = np.histogram_bin_edges(before_zoomed, bins=zoom_bins)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(
        after_zoomed,
        bins=zoomed_bin_edges,
        linewidth=3.0,
        color="orange",
        label="pruned",
        alpha=0.5,
    )
    ax.hist(
        before_zoomed,
        bins=zoomed_bin_edges,
        color="blue",
        label="original",
        alpha=0.5,
    )
    ax.set_yscale("log")
    ax.set_xlim([-zoom_threshold, zoom_threshold])
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="both", which="major", labelsize=10)
    leg = ax.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(
        str(pathlib.Path(output_dir) / "parameter_histogram_zoomed.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
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

    param_keys = get_param_keys(
        pathlib.Path(args.original_weights_path).parent.parent,
        checkpoint_name=pathlib.Path(args.original_weights_path).name,
        device=device,
    )

    before_params, n_nonzero_before = extract_float_parameters(
        args.original_weights_path, param_keys, device
    )
    after_params, n_nonzero_after = extract_float_parameters(
        args.pruned_weights_path, param_keys, device
    )

    plot_histograms(
        before_params,
        after_params,
        output_dir=pathlib.Path(args.output_dir),
        bins=args.bins,
        zoom_threshold=args.zoom_threshold,
        zoom_bins=args.zoom_bins,
    )


if __name__ == "__main__":
    main()
