import os
import argparse
import pandas as pd
import SimpleITK as sitk
import torch
from ultralytics import YOLO
from segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m import gen_yolov8_prob_map
from segment_tb_cxr.unet_resnet18.inference.inference_tb_segment import (
    file_path,
    csv_path,
)


"""
This inference file is used to run inference using the trained nnunet model.
The resulting mask is saved to the specified output folder. It takes in a
 csv with column name 'filename', weight file path and nnUNet and output segmentation
folder to save the generated yolov8 predictions and output csv filename with
an extra column name 'yolov8_pred_tb_seg_file' corresponding to the original
filename. The prediction in the output folder are generated with
{filename}_yolov8_pred_seg.nrrd format.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv_path", type=csv_path, help="Input CSV path with column filename"
    )
    parser.add_argument(
        "yolov8_weights", type=file_path, help="Weights path for yolov8"
    )
    parser.add_argument(
        "output_seg_folder", type=str, help="output directory to save the predictions."
    )
    parser.add_argument(
        "--binary_mask_threshold", type=float, default=0.5, help="Binary mask threshold"
    )
    parser.add_argument(
        "output_csv_path",
        type=str,
        help="Output CSV path with column filename and yolov8_pred_tb_seg_file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_seg_folder):
        os.makedirs(args.output_seg_folder)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolov8_model = YOLO(args.yolov8_weights, verbose=False)
    yolov8_model.to(device)

    df = pd.read_csv(args.input_csv_path)

    for file in df["filename"].tolist():
        yolov8_prob_map = gen_yolov8_prob_map(file, yolov8_model)

        output_seg_file = os.path.join(
            args.output_seg_folder,
            os.path.splitext(os.path.basename(file))[0] + "_yolov8_pred_seg.nrrd",
        )

        sitk.WriteImage(
            sitk.GetImageFromArray(yolov8_prob_map) > args.binary_mask_threshold,
            output_seg_file,
        )

    df["yolov8_pred_tb_seg_file"] = df["filename"].apply(
        lambda x: os.path.join(
            args.output_seg_folder,
            os.path.splitext(os.path.basename(x))[0] + "_yolov8_pred_seg.nrrd",
        )
    )
    df.to_csv(args.output_csv_path, index=False)


if __name__ == "__main__":
    main()
