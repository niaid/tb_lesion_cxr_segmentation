import pandas as pd
import SimpleITK as sitk
import pathlib
import argparse
import multiprocessing
from functools import partial
"""
This script computes various Overlap results from the
reference and the predicted binary masks by the segmentation model. User has to
provide the input CSV filename with column names 'Output_tb_seg_filename' and 'pred_tb_seg_file'
with each representing the filepaths for reference and the corresponding
predicted binary masks respectively.
Code based on the SimpleITK Segmentation Evaluation Jupyter notebook.
(https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/34_Segmentation_Evaluation.ipynb)
"""


def _compute_metrics(df,i):
    """

    This function computes  Overlap and Surface Distance metrics from reference files
    and predicted segmentation files.Input CSV path must have column names of
    'ref_seg_file' and 'pred_seg_file' with each column representing the reference segmentation
    and the corresponding predicted segmentation file respectively.
    ''

    Args:
        input_csv_path(pathlib.Path): Input CSV path containing test Chest X Ray paths
                                      with column names as 'Output_tb_seg_filename' and 'pred_tb_seg_file'

        pred_dir(pathlib.Path): Prediction directory containing segmented masks.
    Returns:
          ---

    """

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    reference_segmentation = sitk.ReadImage(df["Output_tb_seg_filename"][i])
    seg = sitk.ReadImage(df["pred_tb_seg_file"][i])

    overlap_measures_filter.Execute(reference_segmentation, seg)

    return overlap_measures_filter.GetDiceCoefficient(), overlap_measures_filter.GetJaccardCoefficient()



def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Metric evaluation between reference and  predicted segmentations"
    )

    parser.add_argument(
        "input_csv_path",
        type=pathlib.Path,
        help="Input CSV path containing column names as 'Output_tb_seg_filename' and 'pred_tb_seg_file' representing the \
                        filepaths for the label file and predicted segmentation files respectively.",
    )
    parser.add_argument(
        "output_overlap_results_filename",
        type=str,
        help="Output CSV file containing overlap results",
    )

    args = parser.parse_args()
    df = pd.read_csv(args.input_csv_path)


    with multiprocessing.Pool(30) as p:
        function = partial(_compute_metrics,df)
        results = p.map(function,range(len(df)))
        
    (df["dice"],df['jaccard']) = zip(*results)
    
    df.to_csv(args.output_overlap_results_filename,index=False)


if __name__ == "__main__":
    main()
