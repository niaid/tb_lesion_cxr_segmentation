import numpy as np
import pandas as pd
import SimpleITK as sitk
import argparse
import multiprocessing
import pathlib
from skimage.draw import polygon


def find_full_path(data_root, row):
    """
    Zhying's annotation file contains patient id and filename (with jpeg extension) columns.
    Using these columns we find the full path between the provided data_root and the
    filename. Not all files in the tb portals dataset have .dcm extension.

    Args:
    ------
    data_root (str): Root directory of the TB Portals Chest X Rays.
    row(pandas.core.series.Series): Dataframe row containing columns 'Filename' and
                                    'PatientID'

    Returns:
    ------
    full path of the chest x ray filename within the aspera dataset.
    """

    if (
        len(
            list(
                (data_root / pathlib.Path(row["PatientID"])).rglob(
                    str(row["Filename"].split(".jpeg")[0] + ".dcm")
                )
            )
        )
        > 0
    ):
        return str(
            list(
                (data_root / pathlib.Path(row["PatientID"])).rglob(
                    str(row["Filename"].split(".jpeg")[0] + ".dcm")
                )
            )[0]
        )
    else:
        return str(
            list(
                (data_root / pathlib.Path(row["PatientID"])).rglob(
                    str(row["Filename"].split(".jpeg")[0])
                )
            )[0]
        )


def polygons_to_label_image(roi_list, image_size, background_value=0, label_value=1):
    """
    Args:
    ----
    Create a label image using the given region-of-interest list and associated label names.
    roi_list (list(list(tuple(x,y)))) : List of regions-of-interest, each corresponding to the label in the label_names
                                        list.
    label_name_to_value_dict (dict{str:int}): Dictionary mapping label strings to integer labels.
    image_size (tuple(x_size,y_size)): Size of output image.
    background_value (int): Value to use for the background of the label image. Must be different from any of the
                            values found in the label_name_to_value_dict.
    Returns:
    -------
    (SimpleITK.Image): A label image representing the ROIs. The meta-data dictionary of the image is configured so that
                       it is compatible with the Slicer seg.nrrd format, enabling meaningful visualization of the
                       segmentation overlay.
    """

    # Color lookup table (LUT) defined in itkLabelToRGBFunctor which claims that these
    # are "a good selection of distinct colors for plotting and overlays".
    # Also tried the color lookup tables used by fsleyes (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes),
    # mgh-cma-freesurfer.lut etc., larger LUT, but the colors were not appropriate, too many similar colors
    # so visually hard to distinguish between different labels even if the colors are not exactly the same.
    # The downside to the ITK LUT is that it only has 20 entries, afterwards the colors are "reused"
    # for a different label (label0 and label21 will have the same color). If this is an issue and you
    # really need a unique color for each label then define a larger color LUT.

    label_image = (
        np.ones((image_size[1], image_size[0]), dtype=np.uint8) * background_value
    )
    for roi in roi_list:
        # unpack the list of tuples into two lists
        xcoords, ycoords = zip(*roi)
        fill_ycoords, fill_xcoords = polygon(ycoords, xcoords, label_image.shape)
        label_image[fill_ycoords, fill_xcoords] = label_value

    sitk_image = sitk.GetImageFromArray(label_image)

    return sitk_image


def save_nrrd_label_image(image_file_name, output_seg_filename, rois):
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetFileName(image_file_name)
    image_file_reader.ReadImageInformation()
    image_size = image_file_reader.GetSize()[0:2]
    label_image = polygons_to_label_image(rois, image_size)

    sitk.WriteImage(label_image, output_seg_filename + ".nrrd")


def filter_df(df, wanted_findings):
    """
    Filter the dataframe with only the ones which have the user provided findings
    for each of the Chest X Ray. First filter the indices for the user wanted
    findings within the listed findings in the zhying's annotated list and use
    those indices to then filter the roi list and prediction scores.

    Args:
    ------
    df (pd.DataFrame): Dataframe with the columns 'processed_Filename','Predicted Disease for Each ROI',
                        'Locations of Boundary for Each ROI' and 'PredictedScores'
    wanted_findings(list): list of findings that user wants to filter and have only those
                            as regions in the saved labels drawn from the RoIs.

    Returns:
    ------
    df (pd.DataFrame): Dataframe filtered with user wanted findings
    """

    for i, (idx, findings_list, boxes, scores_list) in enumerate(
        zip(
            df.index,
            df["Predicted Disease for Each ROI"],
            df["Locations of Boundary for Each ROI"],
            df["PredictedScores"],
        )
    ):
        # Identify indices of findings that are in the wanted_findings list
        indices_to_keep = [
            index
            for index, finding in enumerate(findings_list)
            if finding in wanted_findings
        ]

        # Filter findings, boxes and scores based on indices
        filtered_findings = [
            finding
            for index, finding in enumerate(findings_list)
            if index in indices_to_keep
        ]
        filtered_boxes = [
            box for index, box in enumerate(boxes) if index in indices_to_keep
        ]
        filtered_scores = [
            score for index, score in enumerate(scores_list) if index in indices_to_keep
        ]

        # Update DataFrame using actual index
        df.loc[idx, "Predicted Disease for Each ROI"] = str(filtered_findings)
        df.loc[idx, "Locations of Boundary for Each ROI"] = str(filtered_boxes)
        df.loc[idx, "PredictedScores"] = str(filtered_scores)

    df["Predicted Disease for Each ROI"] = df["Predicted Disease for Each ROI"].apply(
        lambda x: eval(x)
    )
    df["Locations of Boundary for Each ROI"] = df[
        "Locations of Boundary for Each ROI"
    ].apply(lambda x: eval(x))
    df["PredictedScores"] = df["PredictedScores"].apply(lambda x: eval(x))

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv_path_zhying",
        type=str,
        help="Input CSV file for TB Portals containing zhying's annotations.",
    )
    parser.add_argument(
        "input_csv_path_outlier_info",
        type=str,
        help="Input CSV file for TB Portals containing TB Portal's outlier information.",
    )
    parser.add_argument(
        "input_cxr_dir", type=str, help="Input TB Portals CXR directory."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to save the reference labels plotted using the coordinates from zhying's annotations.",
    )
    parser.add_argument(
        "output_csv_filename",
        type=str,
        help="Output prefix for csv filenames",
    )
    parser.add_argument(
        "abnormality_list",
        type=str,
        nargs="+",
        help="Abnormality list to filter the abnormalities from the zhying's annotations and \
              save the images pertaining only to these abnormalities. All the abnormalities\
              will be saved with label 1 in the output directory",
    )

    args = parser.parse_args()

    zhying_df = pd.read_csv(args.input_csv_path_zhying)

    # Find full paths from zhying annotations file
    zhying_df["processed_Filename"] = zhying_df.apply(
        lambda x: find_full_path(args.input_cxr_dir, x), axis=1
    )
    zhying_df["Output_seg_filename"] = zhying_df.apply(
        lambda x: str(
            pathlib.Path(args.input_cxr_dir)
            / (x["PatientID"] + "_" + pathlib.Path(x["processed_Filename"]).name)
        ),
        axis=1,
    )
    zhying_df["Locations of Boundary for Each ROI"] = zhying_df[
        "Locations of Boundary for Each ROI"
    ].apply(lambda x: eval(x))
    zhying_df["PredictedScores"] = zhying_df["PredictedScores"].apply(lambda x: eval(x))

    outlier_info_df = pd.read_csv(args.input_csv_path_outlier_info)
    outlier_info_df["processed_Filename"] = outlier_info_df[
        "series_instance_content_url"
    ].apply(lambda x: str(pathlib.Path(args.input_cxr_dir) / x))

    merged_zhying_outlier_info_df = pd.merge(
        zhying_df, outlier_info_df, on="processed_Filename"
    )

    # Remove outlier files
    df = merged_zhying_outlier_info_df[
        merged_zhying_outlier_info_df["cxr_outlier"] != "outlier"
    ]

    # Filter labels with the user provided abnormality_list.
    filtered_df = filter_df(df, wanted_findings=args.abnormality_list)

    # Save label files with only filtered findings' regions.
    with multiprocessing.Pool(15) as p:
        p.starmap(
            save_nrrd_label_image,
            zip(
                filtered_df["processed_Filename"].tolist(),
                filtered_df["Output_seg_filename"].tolist(),
                filtered_df["Locations of Boundary for Each ROI"].tolist(),
            ),
        )

    filtered_df.to_csv(args.output_csv_filename, index=False)


if __name__ == "__main__":
    main()
