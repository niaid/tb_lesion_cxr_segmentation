import os
import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk
from ultralytics import YOLO
from segment_tb_cxr.unet_resnet18.inference.inference_tb_segment import _read_image


def pred_segmentations(input_csv_path, weights, output_dir, output_csv_filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = YOLO(weights)

    df = pd.read_csv(input_csv_path)
    for idx, img_path in enumerate(df["processed_Filename"].tolist()):
        original_img = _read_image(img_path)

        img_path = "sample.png"
        sitk.WriteImage(
            sitk.Cast(sitk.RescaleIntensity(original_img), sitk.sitkUInt8), img_path
        )

        results = model(img_path)

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

    df["pred_tb_seg_file"] = df["processed_Filename"].apply(
        lambda x: os.path.splitext(os.path.basename(img_path))[0] + "_pred_seg.png"
    )
    df.to_csv(output_csv_filename, index=False)


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

    pred_segmentations(
        args.input_csv_path, args.weights, args.output_dir, args.output_csv_path
    )


if __name__ == "__main__":
    main()
