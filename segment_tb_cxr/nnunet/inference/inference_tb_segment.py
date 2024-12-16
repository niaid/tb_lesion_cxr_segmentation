import subprocess
import argparse
import sys

"""
This inference file is used to run inference using the trained nnunet model.
User can refer to
"https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md"
before running this script.
"""


def main():
    parser = argparse.ArgumentParser("Preprocessing steps using nnunet")
    parser.add_argument(
        "input_folder",
        type=str,
        help="Input folder containing the images in _0000.nrrd format",
    )
    parser.add_argument(
        "weights",
        type=str,
        help="Pretrained weights",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="output folder to save the predictions",
    )
    args = parser.parse_args()

    subprocess.call(
        [
            "nnUNetv2_predict ",
            "-i",
            args.input_folder,
            "-o",
            args.output_folder,
            "-chk",
            args.weights,
            "-p",
            "nnUNetResEncUNetXLPlans_40G",
            "-c",
            "2d",
            "--save_probabilities",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
