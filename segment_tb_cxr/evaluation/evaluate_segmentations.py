import pandas as pd
import SimpleITK as sitk
import pathlib
import argparse
import multiprocessing


"""
This script computes various Overlap results from the
reference and the predicted binary masks by the segmentation model. User has to
provide the input CSV filename with column names 'Output_tb_seg_filename' and
'pred_tb_seg_file' with each representing the filepaths for reference and
the corresponding predicted binary masks respectively.
Code based on the SimpleITK Segmentation Evaluation Jupyter notebook.
(https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/34_Segmentation_Evaluation.ipynb)
"""


def _compute_metrics_from_images(reference_image, pred_image):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(reference_image, pred_image)
    return {
        "dice": overlap_measures_filter.GetDiceCoefficient(),
        "jaccard": overlap_measures_filter.GetJaccardCoefficient(),
    }


def _compute_metrics(reference_file, pred_file):
    """

    This function computes  Overlap and Surface Distance metrics from reference
    files and predicted segmentation files.Input CSV path must have column
    names of 'ref_seg_file' and 'pred_seg_file' with each column representing
    the reference segmentation and the corresponding predicted segmentation
    file respectively.
    ''

    Args:
        reference_file(str): Reference file name

        pred_file(str): Predicted mask file name
    Returns:
          ---

    """
    return _compute_metrics_from_images(
        sitk.ReadImage(reference_file),
        sitk.ReadImage(pred_file),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Metric evaluation between reference and  predicted \
        segmentations"
    )

    parser.add_argument(
        "input_csv_path",
        type=pathlib.Path,
        help="Input CSV path containing column names as \
             'Output_tb_seg_filename' and 'pred_tb_seg_file' representing the \
             filepaths for the label file and predicted segmentation files \
             respectively.",
    )
    parser.add_argument(
        "output_overlap_results_filename",
        type=str,
        help="Output CSV file containing overlap results",
    )

    args = parser.parse_args()
    df = pd.read_csv(args.input_csv_path)

    with multiprocessing.Pool(30) as p:
        results = p.starmap(
            _compute_metrics,
            zip(df["Output_tb_seg_filename"].tolist(), df["pred_tb_seg_file"].tolist()),
        )

    df = pd.DataFrame(results)

    df.to_csv(args.output_overlap_results_filename, index=False)


if __name__ == "__main__":
    main()
