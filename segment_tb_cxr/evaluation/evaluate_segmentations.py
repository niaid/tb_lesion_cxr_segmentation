import numpy as np
import pandas as pd
import SimpleITK as sitk
import pathlib
import argparse
import multiprocessing
from segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m import csv_path
"""
This script computes overlap and surface distance results from the
reference and the predicted binary masks by the segmentation model. User has to
provide the input CSV filename with column names 'reference_tb_seg_file' and 'pred_tb_seg_file' with each representing the filepaths for reference and
the corresponding predicted binary masks respectively.
Code based on the SimpleITK Segmentation Evaluation Jupyter notebook.
(https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/34_Segmentation_Evaluation.ipynb)
"""


def surface_distances(reference, segmentation):
    reference_surface = sitk.LabelContour(reference)
    segmentation_surface = sitk.LabelContour(segmentation)

    statistics = sitk.StatisticsImageFilter()
    statistics.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics.GetSum())
    statistics.Execute(segmentation_surface)
    num_segmentation_surface_pixels = int(statistics.GetSum())

    reference_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            reference_surface, squaredDistance=False, useImageSpacing=True
        )
    )
    segmentation_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            segmentation_surface, squaredDistance=False, useImageSpacing=True
        )
    )

    seg2ref_distance_map = reference_distance_map * sitk.Cast(
        segmentation_surface, sitk.sitkFloat32
    )
    ref2seg_distance_map = segmentation_distance_map * sitk.Cast(
        reference_surface, sitk.sitkFloat32
    )

    seg2ref_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    ref2seg_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)

    seg2ref = list(seg2ref_arr[seg2ref_arr != 0])
    ref2seg = list(ref2seg_arr[ref2seg_arr != 0])
    seg2ref += list(
        np.zeros(num_segmentation_surface_pixels - len(seg2ref), dtype=np.float32)
    )
    ref2seg += list(
        np.zeros(num_reference_surface_pixels - len(ref2seg), dtype=np.float32)
    )
    return np.asarray(seg2ref + ref2seg, dtype=np.float32)


def _compute_metrics_from_images(reference_image, pred_image):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(reference_image, pred_image)

    hausdorff = sitk.HausdorffDistanceImageFilter()
    hausdorff.Execute(reference_image, pred_image)

    distances = surface_distances(reference_image, pred_image)
    return {
        "dice": overlap_measures_filter.GetDiceCoefficient(),
        "jaccard": overlap_measures_filter.GetJaccardCoefficient(),
        "hausdorff_distance": float(hausdorff.GetHausdorffDistance()),
        "surface_distance_mean": float(np.mean(distances)),
        "surface_distance_median": float(np.median(distances)),
        "surface_distance_max": float(np.max(distances)),
        "surface_distance_std": float(np.std(distances)),
    }


def _compute_metrics(reference_file, prediction_file):
    """

    This function computes  Overlap and Surface Distance metrics from reference
    files and predicted segmentation files.Input CSV path must have column
    names of 'reference_tb_seg_file' and 'pred_tb_seg_file'\
    with each column representing the reference segmentation and the corresponding
    predicted segmentation file respectively.
    ''

    Args:
        reference_file(str): Reference file name

        prediction_file(str): Predicted mask file name
    Returns:
          ---

    """
    return _compute_metrics_from_images(
        sitk.ReadImage(reference_file),
        sitk.ReadImage(prediction_file),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Metric evaluation between reference and  predicted \
        segmentations"
    )

    parser.add_argument(
        "input_csv_path",
        type=str,
        help="Input CSV path containing column names as \
             'reference_tb_seg_file' and 'pred_tb_seg_file' representing the \
             filepaths for the label file and predicted segmentation files \
             respectively.",
    )
    parser.add_argument(
        "output_results_filename",
        type=str,
        help="Output CSV file containing overlap results",
    )

    args = parser.parse_args()

    csv_path(args.input_csv_path, required_columns=["reference_tb_seg_file", "pred_tb_seg_file"])
    
    df = pd.read_csv(args.input_csv_path)

    with multiprocessing.Pool(30) as p:
        results = p.starmap(
            _compute_metrics,
            zip(df["reference_tb_seg_file"].tolist(), df["pred_tb_seg_file"].tolist()),
        )

    df = pd.DataFrame(results)

    df.to_csv(args.output_results_filename, index=False)


if __name__ == "__main__":
    main()
