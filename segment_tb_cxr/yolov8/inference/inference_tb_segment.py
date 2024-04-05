import os
import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk
from ultralytics import YOLO
from segment_tb_cxr.unet_resnet18.inference.inference_tb_segment import _read_image


def generate_segmentations(df, weights, output_dir):
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

    model = YOLO(weights)

    for idx, img_path in enumerate(df["processed_Filename"].tolist()):
        original_img = _read_image(img_path)

        rescaled_img = sitk.Cast(sitk.RescaleIntensity(original_img))

        img_arr = sitk.GetArrayFromImage(rescaled_img)
        results = model.predict(source=img_arr, save=False, save_txt=False)

        output_pred_file = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(img_path))[0] + "_pred_seg.png",
        )
        if results[0].masks is not None:
            im_array = results[0].masks.data.cpu().numpy()
            if im_array.shape[0] >= 2:
                combined_mask = im_array.sum(axis=0)
            else:
                combined_mask = im_array[0, :, :]

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
                sitk.sitkNearestNeighbor,
                original_img.GetOrigin(),
                new_spacing,
                original_img.GetDirection(),
                0,
                sitk.sitkUInt8,
            )

            sitk.WriteImage(pred_mask_original_size, output_pred_file)

        else:
            zero_array = np.zeros(
                (original_img.GetSize()[0], original_img.GetSize()[1])
            )
            result_image = sitk.GetImageFromArray(zero_array)
            sitk.WriteImage(
                result_image,
                os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(img_path))[0] + "_pred_seg.png",
                ),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=str, help="Weights path")
    parser.add_argument(
        "input_csv_path", type=str, help="Input CSV path with column processed_Filename"
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

    generate_segmentations(df, args.weights, args.output_dir)

    df["pred_tb_seg_file"] = df["processed_Filename"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0] + "_pred_seg.png"
    )
    df.to_csv(args.output_csv_filename, index=False)


if __name__ == "__main__":
    main()
