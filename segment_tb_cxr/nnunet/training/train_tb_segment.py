import subprocess
import argparse
import sys

"""
This training file is used to run training on various folds for various UNet
configurations(2d,3d_fullres,3d_lowres,3d_cascade_fullres). User can refer to
"https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md"
before running this script.
"""


def main():
    parser = argparse.ArgumentParser("Training using nnunet")
    parser.add_argument(
        "--task_number",
        required=True,
        type=int,
        help="Task number(XXX), where the folder name is stored in \
                            nnUNet_raw/TaskXXX_MYTASK format",
    )
    parser.add_argument(
        "--cv_fold_number",
        required=True,
        type=int,
        help="Cross validation fold number to train the model. \
                            Select any number in (0,1,2,3,4).You can also provide \
                            the value 'all",
    )
    args = parser.parse_args()

    subprocess.call(
        [
            "nnUNet_train",
            "2d",
            "nnUNetTrainerV2",
            "Task" + args.task_number + "_MYTASK ",
            args.cv_fold_number,
            "--npz",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
