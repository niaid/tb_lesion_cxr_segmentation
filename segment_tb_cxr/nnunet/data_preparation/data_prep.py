import os
import pandas as pd
import SimpleITK as sitk
import shutil
import argparse
import multiprocessing
from functools import partial
from segment_tb_cxr.unet_resnet18.inference.inference_tb_segment import _read_image
from nnunet.dataset_conversion.utils import generate_dataset_json


"""
This script prepaares dataset for nnUNet training. Make sure to export the
variables of "nnUNet_raw","nnUNet_preprocessed" and "nnUNet_results" with
the corresponding paths for each of these foldersafter running this script.
User can refer to https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/
how_to_use_nnunet.md for additional information.
"""


def write_images(output_folder, input_img_filename):
    """

    nnUNet requires the input chest x rays to be in three dimensions (X,Y,1) and
    in _0000.nrrd or _0000.nii.gz format.

    ''

    Args:
        output_folder(str): output folder to save the rescaled input images

        input_img_filename(str): Input image filename
    Returns:
          ---

    """

    img = _read_image(input_img_filename)

    new_size = list(img.GetSize())
    new_size.append(1)  # Adding an extra dimension

    # Create a new image with the desired size
    new_image = sitk.Image(
        new_size, img.GetPixelID(), img.GetNumberOfComponentsPerPixel()
    )

    # Copy the original image data into the new image
    new_image.CopyInformation(img)
    new_image = sitk.Paste(new_image, img, img.GetSpacing(), destinationIndex=(0, 0, 0))

    # Set the spacing for the new image
    new_spacing = list(img.GetSpacing())
    new_spacing.append(
        img.GetSpacing()[-1]
    )  # Keeping the spacing along the new dimension same as the last dimension
    new_image.SetSpacing(new_spacing)

    output_img_filename = os.path.join(
        output_folder, (os.path.basename(input_img_filename) + "_0000.nrrd",)
    )

    sitk.WriteImage(new_image, output_img_filename)


def write_labels(output_folder, reference_filename):
    """

    nnUNet requires the input labels for corresponding chest x rays to be in .nrrd or .nii.gz format.


    Args:
        output_folder(str): output folder to save the  labels required for yolov8 training

        reference_filename(str): Input reference label filename

    Returns:
          ---

    """

    # Copy the created lesion segmentation file to the label directory.
    nnunet_label_filename = os.path.join(
        output_folder, os.path.basename(reference_filename)
    )

    shutil.copy2(reference_filename, nnunet_label_filename)


def write_images_and_labels(train_val_df, test_df, output_folder):
    train_val_img_directory = os.path.join("nnUNet_raw", output_folder, "imagesTr")
    train_val_label_directory = os.path.join("nnUNet_raw", output_folder, "labelsTr")

    test_img_directory = os.path.join("nnUNet_raw", output_folder, "imagesTs")
    test_label_directory = os.path.join("nnUNet_raw", output_folder, "labelsTs")

    if not os.path.exists(train_val_img_directory):
        os.makedirs(train_val_img_directory)
    if not os.path.exists(train_val_label_directory):
        os.makedirs(train_val_label_directory)

    if not os.path.exists(test_img_directory):
        os.makedirs(test_img_directory)
    if not os.path.exists(test_label_directory):
        os.makedirs(test_label_directory)

    with multiprocessing.Pool(20) as p:
        func = partial(write_images, train_val_img_directory)
        p.map(func, train_val_df["processed_Filename"])

    with multiprocessing.Pool(20) as p:
        func = partial(write_labels, train_val_label_directory)
        p.map(func, train_val_df["Output_tb_seg_filename"])

    with multiprocessing.Pool(20) as p:
        func = partial(write_images, test_img_directory)
        p.map(func, test_df["processed_Filename"])

    with multiprocessing.Pool(20) as p:
        func = partial(write_labels, test_label_directory)
        p.map(func, test_df["Output_tb_seg_filename"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_train_csv_path",
        type=str,
        help="Input training csv path containing columns processed_Filename \
              and Output_seg_filename as columns",
    )
    parser.add_argument(
        "input_val_csv_path",
        type=str,
        help="Input validation csv path containing columns processed_Filename \
              and Output_seg_filename as columns",
    )
    parser.add_argument(
        "input_test_csv_path",
        type=str,
        help="Input testing csv path containing columns processed_Filename and\
               Output_seg_filename as columns",
    )
    parser.add_argument(
        "folder_name",
        required=True,
        type=str,
        help="Output folder that user should give to save the files in \
             imagesTr,imagesTs,labelsTr,labelsTs subfolders as required per nnUNet training.This \
             folder name should be in the format of E.g : Task001_lungsegmentation.This folder \
             should exist in the directory stricture of nnUNet/",
    )

    args = parser.parse_args()

    train_df = pd.read_csv(args.input_train_csv_path)
    val_df = pd.read_csv(args.input_val_csv_path)
    test_df = pd.read_csv(args.input_test_csv_path)

    train_val_df = pd.concat([train_df, val_df])

    write_images_and_labels(train_val_df, test_df, args.folder_name)

    generate_dataset_json(
        output_file=os.path.join("nnUNet_raw", args.folder_name, "dataset.json"),
        imagesTr_dir=os.path.join("nnUNet_raw", args.folder_name, "imagesTr"),
        imagesTs_dir=os.path.join("nnUNet_raw", args.folder_name, "imagesTs"),
        labels={"0": "background", "1": "foreground"},
        dataset_name=args.folder_name,
    )


if __name__ == "__main__":
    main()
