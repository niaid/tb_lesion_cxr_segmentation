"""
Validation script comparing two ways of combining the full train transform pipeline
(preprocessing -> randomized data augmentation -> post augmentation, as configured in
params_for_validate_ensemble_consistency.json) with the live ensemble (nnU-Net +
YOLOv8m-seg):
    1. transform_then_predict: run the randomized augmentation ops on the input image first, then run the
       ensemble on the transformed image.
    2. predict_then_transform: run the ensemble on the preprocessed ORIGINAL image, then
       replay the *exact* randomized augmentation ops sampled in step 1 onto the resulting
       probability map.

Both resulting probability maps are thresholded and compared via Dice against the
reference segmentation mask, transformed identically to the input image.

Input CSV must contain columns 'filename' and 'ref_seg_file'.
"""

import os
import json
import argparse
import pathlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from segment_tb_cxr.evaluations.generate_transformations import (
    _apply_deterministic,
    transform_image,
    apply_transform,
)
from segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m import (
    gen_yolov8_prob_map,
    gen_nnunet_prob_map,
    gen_ensembled_yolov8_nnunet_segmentation,
    _read_image,
)
from segment_tb_cxr.evaluation.evaluate_segmentations import (
    _compute_metrics_from_images,
)
import warnings

warnings.filterwarnings("ignore")


def _load_ensemble_models(nnunet_weights_dir, yolov8_weights_path):
    """Load nnU-Net and YOLOv8 models."""
    from ultralytics import YOLO
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import contextlib
    import sys
    import io
    import torch

    @contextlib.contextmanager
    def suppress_stdout():
        new_target = io.StringIO()
        old_target, sys.stdout = sys.stdout, new_target
        try:
            yield new_target
        finally:
            sys.stdout = old_target

    print("Loading YOLOv8 model...")
    # torch>=2.6 defaults torch.load to weights_only=True, which rejects the
    # pickled ultralytics classes in this checkpoint; load it as a trusted file.
    original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: original_torch_load(
        *args, **{**kwargs, "weights_only": False}
    )
    try:
        yolov8_model = YOLO(yolov8_weights_path).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    finally:
        torch.load = original_torch_load

    print("Loading nnU-Net predictor...")
    with suppress_stdout():
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            nnunet_weights_dir,
            use_folds="0",
            checkpoint_name="nnunet.pth",
        )

    return yolov8_model, predictor


def _plot_horizontal(
    transformed_img, transformed_ref, pred_on_transformed, output_dir, stem
):
    """4 images in a row: transformed input, transformed reference, prediction on
    transformed image, and their absolute difference."""
    diff = np.abs(pred_on_transformed - transformed_ref)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(transformed_img, cmap="gray")
    axes[0].set_title("Transformed Input Image")

    axes[1].imshow(transformed_ref, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Transformed Reference Image")

    axes[2].imshow(pred_on_transformed, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Prediction on Transformed Image")

    axes[3].imshow(diff, cmap="RdYlGn_r", vmin=0, vmax=0.2)
    axes[3].set_title(f"Abs. Difference (MAE: {diff.mean():.4f})")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    viz_path = os.path.join(output_dir, f"{stem}_validation_comparison.png")
    fig.savefig(viz_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {viz_path}")


def _mask_overlay_rgba(pred_mask, ref_mask):
    """RGBA overlay: bright orange=pred only, bright blue=ref only, bright yellow=agreement."""
    agreement = pred_mask & ref_mask
    pred_only = pred_mask & ~ref_mask
    ref_only = ref_mask & ~pred_mask

    overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    overlay[pred_only] = [1.0, 0.55, 0.0, 0.5]
    overlay[ref_only] = [0.0, 0.4, 1.0, 0.5]
    overlay[agreement] = [1.0, 1.0, 0.0, 0.5]
    return overlay


def _save_overlay_image(background, overlay, title, output_dir, filename):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(background, cmap="gray")
    ax.imshow(overlay)
    # ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    viz_path = os.path.join(output_dir, filename)
    fig.savefig(viz_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {viz_path}")


def _plot_overlay_comparison(
    transformed_img,
    predict_then_transform_mask,
    transform_then_predict_mask,
    reference_mask,
    output_dir,
    stem,
):
    """Save 3 individual overlays on the transformed input image: predicted vs.
    reference regions for each approach, and where the two prediction masks
    disagree."""
    img_norm = (transformed_img - transformed_img.min()) / (
        transformed_img.max() - transformed_img.min() + 1e-6
    )

    left_overlay = _mask_overlay_rgba(predict_then_transform_mask, reference_mask)
    middle_overlay = _mask_overlay_rgba(transform_then_predict_mask, reference_mask)

    disagreement = predict_then_transform_mask != transform_then_predict_mask
    diff_overlay = np.zeros((*disagreement.shape, 4), dtype=np.float32)
    diff_overlay[disagreement] = [1.0, 0.0, 0.75, 0.5]

    _save_overlay_image(
        img_norm,
        left_overlay,
        "Transformation of Predicted Image\nvs. Reference",
        output_dir,
        f"{stem}_overlay_predict_then_transform.pdf",
    )
    _save_overlay_image(
        img_norm,
        middle_overlay,
        "Prediction on Transformed Image\nvs. Reference",
        output_dir,
        f"{stem}_overlay_transform_then_predict.pdf",
    )
    _save_overlay_image(
        img_norm,
        diff_overlay,
        "Abs. Difference Between Prediction Masks",
        output_dir,
        f"{stem}_overlay_prediction_disagreement.pdf",
    )


def validate_ensemble_consistency(
    image_file,
    ref_seg_file,
    yolov8_model,
    nnunet_predictor,
    output_dir,
    threshold,
    config,
    save_plot,
):
    """Run the validation on a single image / reference mask pair."""
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_file))[0]

    original_img = _read_image(image_file)
    ref_seg_img = sitk.ReadImage(ref_seg_file)

    original_array = np.expand_dims(
        np.transpose(sitk.GetArrayFromImage(original_img), (1, 0)), axis=0
    ).astype(np.float32)
    d = {
        "img": original_array.copy(),
        "img_original": original_array.copy(),
        "seg": np.expand_dims(
            np.transpose(sitk.GetArrayFromImage(ref_seg_img), (1, 0)), axis=0
        ).astype(np.float32),
    }
    preprocessing_entries = [
        e
        for e in config["preprocessing_settings"]["transforms"]
        if e["class"] != "CustomLoadImaged"
    ]
    d = _apply_deterministic(preprocessing_entries, d)

    # Step 1: transform the input image, then predict on it
    d, recipe = transform_image(config, d)
    d = _apply_deterministic(config["post_augmentation_settings"]["transforms"], d)
    print(f"{stem}: applied augmentation transforms ->")
    for step in recipe:
        if step["fired"]:
            print(f"  {step['class']}: {step['sampled_values']}")

    transformed_input_img = sitk.GetImageFromArray(np.transpose(d["img"][0], (1, 0)))

    yolov8_prob_t = gen_yolov8_prob_map(transformed_input_img, yolov8_model)
    nnunet_prob_t = gen_nnunet_prob_map(transformed_input_img, nnunet_predictor)
    transform_then_predict_img = gen_ensembled_yolov8_nnunet_segmentation(
        transformed_input_img, yolov8_prob_t, nnunet_prob_t
    )

    # Step 2: predict on the original preprocessed image, then transform that prediction
    # using the exact same transforms recorded in step 1.
    preprocessed_original_img = sitk.GetImageFromArray(
        np.transpose(d["img_original"][0], (1, 0))
    )

    yolov8_prob_orig = gen_yolov8_prob_map(preprocessed_original_img, yolov8_model)
    nnunet_prob_orig = gen_nnunet_prob_map(preprocessed_original_img, nnunet_predictor)
    ensemble_prob_orig_img = gen_ensembled_yolov8_nnunet_segmentation(
        preprocessed_original_img, yolov8_prob_orig, nnunet_prob_orig
    )

    pred_map = {
        "pred_map": np.expand_dims(
            np.transpose(sitk.GetArrayFromImage(ensemble_prob_orig_img), (1, 0)), axis=0
        ).astype(np.float32)
    }
    pred_map = apply_transform(recipe, pred_map, key="pred_map")
    pred_map = _apply_deterministic(
        config["post_augmentation_settings"]["transforms"],
        pred_map,
        extra_keys=["pred_map"],
    )

    predict_then_transform_img = sitk.GetImageFromArray(
        np.transpose(pred_map["pred_map"][0], (1, 0))
    )
    predict_then_transform_img = sitk.Clamp(
        predict_then_transform_img, predict_then_transform_img.GetPixelID(), 0.0, 1.0
    )

    # d["seg"] values are already binary (0/1) from nearest-neighbor interpolation; cast to
    # uint8 directly so LabelOverlapMeasuresImageFilter (Dice) accepts it as a label image.
    transformed_ref_mask = sitk.GetImageFromArray(
        np.transpose(d["seg"][0], (1, 0)).astype(np.uint8)
    )

    # Threshold predictions to binary sitk images for Dice.
    predict_then_transform_mask = predict_then_transform_img > threshold
    transform_then_predict_mask = transform_then_predict_img > threshold

    dice_predict_then_transform = _compute_metrics_from_images(
        predict_then_transform_mask, transformed_ref_mask
    )["dice"]
    dice_transform_then_predict = _compute_metrics_from_images(
        transform_then_predict_mask, transformed_ref_mask
    )["dice"]

    predict_then_transform_path = os.path.join(
        output_dir, f"{stem}_predict_then_transform_seg.nrrd"
    )
    transform_then_predict_path = os.path.join(
        output_dir, f"{stem}_transform_then_predict_seg.nrrd"
    )
    sitk.WriteImage(
        predict_then_transform_mask,
        predict_then_transform_path,
        useCompression=True,
    )
    sitk.WriteImage(
        transform_then_predict_mask,
        transform_then_predict_path,
        useCompression=True,
    )

    if save_plot:
        _plot_horizontal(
            sitk.GetArrayFromImage(transformed_input_img).astype(np.float32),
            sitk.GetArrayFromImage(predict_then_transform_img).astype(np.float32),
            sitk.GetArrayFromImage(transform_then_predict_img).astype(np.float32),
            output_dir,
            stem,
        )
        _plot_overlay_comparison(
            sitk.GetArrayFromImage(transformed_input_img).astype(np.float32),
            sitk.GetArrayFromImage(predict_then_transform_mask).astype(bool),
            sitk.GetArrayFromImage(transform_then_predict_mask).astype(bool),
            sitk.GetArrayFromImage(transformed_ref_mask).astype(bool),
            output_dir,
            stem,
        )

    return {
        "dice_predict_then_transform": dice_predict_then_transform,
        "dice_transform_then_predict": dice_transform_then_predict,
        "predict_then_transform_seg_file": predict_then_transform_path,
        "transform_then_predict_seg_file": transform_then_predict_path,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate ensemble (nnU-Net + YOLOv8) consistency between "
        "transforming the prediction vs. predicting on the transformed image."
    )
    parser.add_argument(
        "input_csv_path",
        type=pathlib.Path,
        help="CSV with columns 'filename' and 'ref_seg_file'",
    )
    parser.add_argument(
        "nnunet_weights_dir",
        type=pathlib.Path,
        help="Path to nnU-Net weights directory",
    )
    parser.add_argument(
        "yolov8_weights_path",
        type=pathlib.Path,
        help="Path to YOLOv8 weights file (.pt)",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for predictions and visualizations",
    )
    parser.add_argument(
        "output_csv_path",
        type=str,
        help="Output CSV path for the Dice comparison results",
    )
    parser.add_argument(
        "config_path",
        type=pathlib.Path,
        help="Path to the transform configuration file",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Binary mask threshold"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the row in the CSV to save the visualization for",
    )

    args = parser.parse_args(argv)

    df = pd.read_csv(args.input_csv_path)

    with open(str(args.config_path)) as f:
        config_info = json.load(f)

    yolov8_model, nnunet_predictor = _load_ensemble_models(
        str(args.nnunet_weights_dir), str(args.yolov8_weights_path)
    )

    results = [
        {
            **row.to_dict(),
            **validate_ensemble_consistency(
                row["filename"],
                row["ref_seg_file"],
                yolov8_model,
                nnunet_predictor,
                args.output_dir,
                args.threshold,
                config_info["train"],
                save_plot=(i == args.index),
            ),
        }
        for i, row in df.iterrows()
    ]

    pd.DataFrame(results).to_csv(args.output_csv_path, index=False)

    mean_predict_then_transform = np.mean(
        [r["dice_predict_then_transform"] for r in results]
    )
    mean_transform_then_predict = np.mean(
        [r["dice_transform_then_predict"] for r in results]
    )

    print(f"\n{'=' * 60}")
    print(f"Mean Dice (predict then transform): {mean_predict_then_transform:.4f}")
    print(f"Mean Dice (transform then predict): {mean_transform_then_predict:.4f}")
    if mean_predict_then_transform > mean_transform_then_predict:
        print("predict_then_transform gives the better overall Dice.")
    elif mean_transform_then_predict > mean_predict_then_transform:
        print("transform_then_predict gives the better overall Dice.")
    else:
        print("Both approaches give the same overall Dice.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
