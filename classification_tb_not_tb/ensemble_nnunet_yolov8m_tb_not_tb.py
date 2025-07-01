import glob
import pandas as pd
import SimpleITK as sitk
import torch
from ultralytics import YOLO
import sys
import contextlib
import io
import warnings
import cv2
import os
import argparse
import json
from segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m import (
    gen_yolov8_prob_map,
    gen_nnunet_prob_map,
    gen_ensembled_yolov8_nnunet_segmentation,
)
from classification_tb_not_tb.generate_classification_results import (
    _load_model,
    generate_tb_masks_within_lungs,
    get_probability_and_prediction_label,
)


@contextlib.contextmanager
def suppress_stdout():
    new_target = io.StringIO()
    old_target, sys.stdout = sys.stdout, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


# Use the context manager to suppress output from nnunet library when its
# variables are not exported.
with suppress_stdout():
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

warnings.filterwarnings("ignore")


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a directory path or directory does not exist."
        )


def file_path(path):

    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a file path or file does not exist."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_directory",
        type=dir_path,
        help="Input directory with chest x ray images",
    )
    parser.add_argument(
        "yolov8_weights",
        type=file_path,
        default="segment_tb_cxr/yolov8/weights/yolov8.pt",
        help="Weights path for yolov8",
    )
    parser.add_argument(
        "nnunet_weights",
        type=file_path,
        default="segment_tb_cxr/nnunet/weights/fold_0/nnunet.pth",
        help="Weights path for nnunet",
    )
    parser.add_argument(
        "lung_segmentation_model_info_json_path",
        type=str,
        help="Path to JSON file containing each segmentation model's keys and \
              their respective hyperparameters as values",
    )
    parser.add_argument(
        "lung_segmentation_model_path",
        type=str,
        help="Model path for the lung segmentation model.",
    )
    parser.add_argument(
        "output_csv_filename",
        type=str,
        help="Output CSV filename to save the column 'tb_contours' along \
              with the  columns in the files, decision_for_TB and probability_for_TB",
    )
    parser.add_argument(
        "--binary_mask_threshold",
        type=float,
        default=0.5,
        help="Binary mask threshold for TB segmentation",
    )
    parser.add_argument(
        "--decision_for_TB",
        type=float,
        default=0.79336864,
        help="Threshold to classify as TB within lungs.",
    )

    args = parser.parse_args()

    with open(str(args.lung_segmentation_model_info_json_path)) as f:
        lung_segmentation_model_info = json.load(f)

    lung_seg_model, device = _load_model(args.lung_segmentation_model_path)

    # Inititalize YOLOv8 model
    yolov8_model = YOLO(args.yolov8_weights, verbose=False)
    yolov8_model.to(device)

    # Initialize nnUNet model and silence the progress bar to be consistent with YOLO behavior.
    predictor = nnUNetPredictor(device=torch.device(device), allow_tqdm=False)

    # Directory in nnunet/weights has a sub folder named fold_X where X is arbitrary. Here '0' is used.
    predictor.initialize_from_trained_model_folder(
        os.path.dirname(os.path.dirname(args.nnunet_weights)),
        checkpoint_name=os.path.basename(args.nnunet_weights),
        use_folds=(0,),
    )

    files = glob.glob(os.path.join(args.input_directory, "*"))
    df = pd.DataFrame({"file": files})

    tb_contours = []
    probabilities_for_TB = []
    pred_tb_labels = []
    for file in files:

        # Get the TB predictions per pixel from YOLOv8 model
        yolov8_prob_map = gen_yolov8_prob_map(file, yolov8_model)
        # Get the TB predictions per pixel from nnUNetv2 model
        nnunet_prob_map = gen_nnunet_prob_map(file, predictor)

        # Get ensemble of predictions from both the model predictions above
        ensemble_nnunet_yolov8_prob_map_img = gen_ensembled_yolov8_nnunet_segmentation(
            file, yolov8_prob_map, nnunet_prob_map
        )

        # Get TB contours from ensemble image
        contours, _ = cv2.findContours(
            sitk.GetArrayFromImage(
                ensemble_nnunet_yolov8_prob_map_img > args.binary_mask_threshold
            ),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        tb_contours.append([cnt.reshape(-1, 2).tolist() for cnt in contours])

        # Get filtered tb image with probabilities that is just within lung regions.
        filtered_probability_tb_segmentation_within_lungs = (
            generate_tb_masks_within_lungs(
                file,
                ensemble_nnunet_yolov8_prob_map_img,
                lung_seg_model,
                device,
                lung_segmentation_model_info,
            )
        )

        # Get probability of TB and prediction label (TB/NOT_TB)from the above obtained filtered tb image.
        probability_for_TB, pred_tb_label = get_probability_and_prediction_label(
            filtered_probability_tb_segmentation_within_lungs,
            threshold_for_TB=args.decision_for_TB,
        )
        probabilities_for_TB.append(probability_for_TB)
        pred_tb_labels.append(pred_tb_label)

    df["tb_contours"] = tb_contours
    df["probability_for_TB"] = probabilities_for_TB
    df["predicted_decision_for_TB_NOT_TB"] = pred_tb_labels

    df.to_csv(args.output_csv_filename, index=False)


if __name__ == "__main__":
    main()
