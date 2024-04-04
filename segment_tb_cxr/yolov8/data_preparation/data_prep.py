import os
import pandas as pd
import SimpleITK as sitk
import cv2
import argparse
import multiprocessing
from functools import partial
import yaml


def write_images(output_folder, input_img_filename):
    img = sitk.ReadImage(input_img_filename)

    output_img_filename = os.path.join(
        output_folder,
        (
            os.path.splitext(input_img_filename)[0]
            + os.path.splitext(input_img_filename)[1]
        ).split(".nrrd")[0]
        + ".png",
    )

    sitk.WriteImage(
        sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8), output_img_filename
    )


def write_labels(output_folder, reference_filename):
    mask = sitk.ReadImage(reference_filename)
    mask_arr = sitk.GetArrayFromImage(mask)

    # Calculate image dimensions
    image_height, image_width = mask_arr.shape

    # Create a text file based on the mask filename in the output folder
    txt_filename = (
        os.path.splitext(reference_filename)[0]
        + os.path.splitext(reference_filename)[1]
    ).split(".seg.nrrd")[0] + ".txt"

    txt_path = os.path.join(output_folder, txt_filename)
    # Extract contours for each region
    contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Sort contours based on contour area and keep the largest two
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    with open(txt_path, "w+") as txt_file:
        for contour in contours:
            # Normalize the contour coordinates
            normalized_contour = contour / [image_width, image_height]
            # Flatten the contour array and format the coordinates
            flattened_contour = normalized_contour.flatten()
            # Convert each coordinate to a string with 6 decimal places
            contour_str = " ".join([f"{coord:.6f}" for coord in flattened_contour])

            # Write to the file
            txt_line = str(0) + " " + contour_str
            txt_file.write(txt_line + "\n")


def write_images_and_labels(df, output_dir, dataset="train"):
    img_directory = os.path.join(output_dir, "images", dataset)
    label_directory = os.path.join(output_dir, "labels", dataset)
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)

    with multiprocessing.Pool(20) as p:
        func = partial(write_images, img_directory)
        p.map(func, df["processed_Filename"])

    with multiprocessing.Pool(20) as p:
        func = partial(write_labels, label_directory)
        p.map(func, df["processed_Filename"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_train_csv_path",
        type=str,
        help="Input training csv path \
                        containing columns processed_Filename and \
                            Output_seg_filename as columns",
    )
    parser.add_argument(
        "input_val_csv_path",
        type=str,
        help="Input validation csv path \
                        containing columns processed_Filename and \
                            Output_seg_filename as columns",
    )
    parser.add_argument(
        "input_test_csv_path",
        type=str,
        help="Input testing csv path \
                        containing columns processed_Filename and \
                            Output_seg_filename as columns",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory to save\
                                                      the images and labels \
                                                      corresponding to \
                                                      yolov8",
    )
    parser.add_argument(
        "output_yaml_filename",
        type=str,
        help="Output YAML \
                                                                filename",
    )

    args = parser.parse_args()

    train_df = pd.read_csv(args.input_train_csv_path)
    val_df = pd.read_csv(args.input_val_csv_path)
    test_df = pd.read_csv(args.input_test_csv_path)

    write_images_and_labels(train_df, args.output_dir, dataset="train")
    write_images_and_labels(val_df, args.output_dir, dataset="val")
    write_images_and_labels(test_df, args.output_dir, dataset="test")

    data = {
        "path": args.output_dir,
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "val"),
        "test": os.path.join("images", "test"),
        "names": {"0": "tb"},
    }

    # Write data to YAML file
    with open(os.path.join(args.output_dir, args.output_yaml_filename), "w") as file:
        yaml.dump(data, file, default_flow_style=False)


if __name__ == "__main__":
    main()
