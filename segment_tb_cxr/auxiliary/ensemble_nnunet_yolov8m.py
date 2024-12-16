import os
import argparse
import pandas as pd
import SimpleITK as sitk
from ultralytics import YOLO
import numpy as np
from segment_tb_cxr_old.inference.inference_tb_segment import _read_image


def generate_segmentations(df, yolov8_weights, output_dir):
    """

    This function reads the input image filename, extracts the numpy array and
    is then passed into the loaded model to generate the predicted segmentations.
    The predicted segmentation in numpy array format is generated in the
    usually 640*480 shape and os is then resampled to the original size. This
    resampled original size image is then saved into the output directory.

    Args:

        df(pd.DataFrame): dataframe containing columns processed_Filename
        weights(str): Input path to the weights
        output_dir(str): output directory to save the predicted segmentations
        output_csv_filename

    Returns:
          ---

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = YOLO(yolov8_weights)

    nnunet_yolov8_cropped_pred_img_filenames = []
    nnunet_yolov8_non_cropped_pred_img_filenames = []
    nnunet_yolov8_cropped_binary_mask_pred_img_filenames = []
    nnunet_yolov8_non_cropped_binary_mask_pred_img_filenames = []

    for idx, img_path in enumerate(df["processed_Filename"].tolist()):
        print(img_path)
        original_img = _read_image(img_path)

        # Yolov8 expects inputs to be in uint8 format scaled to [0-255].
        # Different intensity ranges result in different results.
        rescaled_img = sitk.Cast(
            sitk.RescaleIntensity(original_img, 0, 255), sitk.sitkUInt8
        )

        img_arr = sitk.GetArrayViewFromImage(rescaled_img)
        img_arr = np.expand_dims(img_arr, -1)
        img_arr = np.repeat(img_arr, 3, 2)
        results = model.predict(source=img_arr, save=False, save_txt=False)

        if (
            results[0].prob_masks is not None
        ):  # If the yolov8 predictions does not contain any regions of "TB"
            cropped_prob_masks = results[0].prob_masks.data.cpu().numpy()
            combined_mask = cropped_prob_masks.sum(axis=0)

            combined_mask = np.clip(combined_mask, 0, 1)

            result_image = sitk.GetImageFromArray(combined_mask)

            new_spacing = [
                sz * spc / nsz
                for nsz, sz, spc in zip(
                    original_img.GetSize(),
                    result_image.GetSize(),
                    result_image.GetSpacing(),
                )
            ]
            pred_mask_original_size = sitk.Resample(
                result_image,
                original_img.GetSize(),
                sitk.Transform(),
                sitk.sitkLinear,
                original_img.GetOrigin(),
                new_spacing,
                original_img.GetDirection(),
                0,
                sitk.sitkFloat32,
            )

            yolov8_pred_img_filename = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(img_path))[0]
                + "_yolov8_cropped.nrrd",
            )
            sitk.WriteImage(pred_mask_original_size, yolov8_pred_img_filename)

            arr_yolov8m_org_size = sitk.GetArrayFromImage(pred_mask_original_size)

            np.savez(
                os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(img_path))[0]
                    + "_yolov8_cropped.npz",
                ),
                arr_yolov8m_org_size,
            )

            arr_nnunet_org_size = np.load(df["nnUNet_pred_arr_file"].iloc[idx])[
                "probabilities"
            ][1][0]

            ensemble_models_org_size = np.mean(
                [arr_yolov8m_org_size, arr_nnunet_org_size], axis=0
            )

            np.savez(
                os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(img_path))[0]
                    + "_ensemble_nnunet_yolov8_cropped.npz",
                ),
                ensemble_models_org_size,
            )

            nnunet_yolov8_cropped_pred_img_filename = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(img_path))[0]
                + "_ensemble_nnunet_yolov8_cropped.nrrd",
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(ensemble_models_org_size),
                nnunet_yolov8_cropped_pred_img_filename,
            )

            nnunet_yolov8_cropped_binary_mask_pred_img_filename = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(img_path))[0]
                + "_ensemble_nnunet_yolov8_cropped_binary_mask.nrrd",
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(ensemble_models_org_size) > 0.5,
                nnunet_yolov8_cropped_binary_mask_pred_img_filename,
            )

        else:  # Modify the code below to divde the probabilties from nnunet
            # by 2 (assuming zeroes for yolov8 predicted images)
            arr_nnunet_org_size = np.load(df["nnUNet_pred_arr_file"].iloc[idx])[
                "probabilities"
            ][1][0]
            nnunet_yolov8_non_cropped_pred_img_filename = None
            nnunet_yolov8_non_cropped_binary_mask_pred_img_filename = None
            nnunet_yolov8_cropped_pred_img_filename = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(img_path))[0]
                + "_nnunet_yolov8_none.nrrd",
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(arr_nnunet_org_size / 2),
                nnunet_yolov8_cropped_pred_img_filename,
            )
            nnunet_yolov8_cropped_binary_mask_pred_img_filename = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(img_path))[0]
                + "_nnunet_yolov8_none_binary_mask.nrrd",
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(arr_nnunet_org_size / 2) > 0.5,
                nnunet_yolov8_cropped_binary_mask_pred_img_filename,
            )

        nnunet_yolov8_cropped_pred_img_filenames.append(
            nnunet_yolov8_cropped_pred_img_filename
        )
        nnunet_yolov8_non_cropped_pred_img_filenames.append(
            nnunet_yolov8_non_cropped_pred_img_filename
        )
        nnunet_yolov8_cropped_binary_mask_pred_img_filenames.append(
            nnunet_yolov8_cropped_binary_mask_pred_img_filename
        )
        nnunet_yolov8_non_cropped_binary_mask_pred_img_filenames.append(
            nnunet_yolov8_non_cropped_binary_mask_pred_img_filename
        )

    return (
        nnunet_yolov8_cropped_pred_img_filenames,
        nnunet_yolov8_non_cropped_pred_img_filenames,
        nnunet_yolov8_cropped_binary_mask_pred_img_filenames,
        nnunet_yolov8_non_cropped_binary_mask_pred_img_filenames,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yolov8_weights", type=str, help="Weights path")
    parser.add_argument(
        "input_csv_path",
        type=str,
        help="Input CSV path with column processed_Filename and nnUNet_pred_arr_file",
    )
    parser.add_argument(
        "output_dir", type=str, help="output directory to save the images"
    )
    parser.add_argument(
        "output_csv_path",
        type=str,
        help="Output CSV path with column filename and pred_tb_seg_file",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv_path)

    columns = [
        "nnunet_yolov8_cropped_pred_img_filenames",
        "nnunet_yolov8_non_cropped_pred_img_filenames",
        "nnunet_yolov8_cropped_binary_mask_pred_img_filenames",
        "nnunet_yolov8_non_cropped_binary_mask_pred_img_filenames",
    ]

    # Unpack the generated segmentations
    segmentations = generate_segmentations(df, args.weights, args.output_dir)

    # Assign values to the corresponding DataFrame columns
    for col, data in zip(columns, segmentations):
        df[col] = data

    df.to_csv(args.output_csv_path, index=False)


if __name__ == "__main__":
    main()
