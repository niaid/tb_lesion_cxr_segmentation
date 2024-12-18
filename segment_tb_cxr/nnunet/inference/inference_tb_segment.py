import os
import pandas as pd
import SimpleITK as sitk
import torch
import argparse
import contextlib
import io
import sys
from segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m import gen_nnunet_prob_map

"""
This inference file is used to run inference using the trained nnunet model.
The resulting mask is saved to the specified output folder. It takes in a
 csv with column name 'filename', weight file path and nnUNet and output segmentation
folder to save the generated predictions. The prediction in the output
folder are generated with {filename}_seg.nrrd format.
"""


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


def main():
    parser = argparse.ArgumentParser("Prediction of TB regions using nnunet model")
    parser.add_argument(
        "input_csv_path",
        type=str,
        help="Input CSV path containing column filename",
    )
    parser.add_argument(
        "nnunet_weights",
        type=str,
        help="weights path fotr nnunet",
    )
    parser.add_argument(
        "output_seg_folder",
        type=str,
        help="output folder to save the predictions",
    )
    parser.add_argument(
        "output_csv_path",
        type=str,
        help="Output CSV path with column filename and pred_tb_seg_file",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize nnUNet model and silence the progress bar to be consistent with YOLO behavior.
    predictor = nnUNetPredictor(device=torch.device(device), allow_tqdm=False)

    # Directory in nnunet/weights has a sub folder named fold_X where X is arbitrary. Here '0' is used.
    predictor.initialize_from_trained_model_folder(
        os.path.dirname(os.path.dirname(args.nnunet_weights)),
        checkpoint_name=os.path.basename(args.nnunet_weights),
        use_folds=(0,),
    )

    df = pd.read_csv(args.input_csv_path)

    for file in df["filename"].tolist():
        nnunet_prob_map = gen_nnunet_prob_map(file, predictor)

        output_seg_file = os.path.join(
            args.output_seg_folder,
            os.path.splitext(os.path.basename(file))[0] + "_pred_seg.nrrd",
        )

        sitk.WriteImage(sitk.GetImageFromArray(nnunet_prob_map), output_seg_file)

    df["pred_tb_seg_file"] = df["filename"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0] + "_pred_seg.nrrd"
    )
    df.to_csv(args.output_csv_filename, index=False)


if __name__ == "__main__":
    sys.exit(main())
