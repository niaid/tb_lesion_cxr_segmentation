import os
import json
import argparse
import monai
import numpy as np
import pandas as pd
import SimpleITK as sitk
from monai.data import list_data_collate, decollate_batch, DataLoader
from segment_tb_cxr_old.inference.inference_tb_segment import _load_model, _read_image
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    Activations,
    Resized,
    NormalizeIntensityd,
    RepeatChanneld,
    EnsureChannelFirstd,
    ScaleIntensityd,
)
import torch


def predict_lung_segmentation_map(
    filename, model, lung_segmentation_model_info, device
):
    original_img = _read_image(filename)
    test_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            Resized(
                keys=["img"],
                spatial_size=lung_segmentation_model_info["img_size"],
                mode=("bilinear"),
            ),
            RepeatChanneld(keys=["img"], repeats=3),
            ScaleIntensityd(keys=["img"]),
            NormalizeIntensityd(
                keys=["img"],
                subtrahend=lung_segmentation_model_info["means"],
                divisor=lung_segmentation_model_info["standard_deviation"],
                channel_wise=True,
            ),
        ]
    )

    test_files = [{"img": filename}]
    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    post_trans = Compose([Activations(sigmoid=True)])
    with torch.no_grad():
        test_data = next(iter(test_loader))
        test_image = test_data["img"].to(device)
        roi_size = lung_segmentation_model_info["roi_size"]
        sw_batch_size = lung_segmentation_model_info["sw_batch_size"]
        pred_mask = sliding_window_inference(test_image, roi_size, sw_batch_size, model)
        pred_mask = post_trans(decollate_batch(pred_mask)[0])
        lung_segmentation_map = np.transpose(pred_mask[1].cpu().numpy(), [1, 0]).astype(
            np.float32
        )

    lung_segmentation_map = sitk.GetImageFromArray(lung_segmentation_map)
    new_spacing = [
        sz * spc / nsz
        for nsz, sz, spc in zip(
            original_img.GetSize(),
            lung_segmentation_map.GetSize(),
            lung_segmentation_map.GetSpacing(),
        )
    ]
    max_intensity_value = sitk.GetArrayFromImage(original_img).max()
    lung_segmentation_map_original_size = sitk.Resample(
        lung_segmentation_map * max_intensity_value,
        original_img.GetSize(),
        sitk.Transform(),
        sitk.sitkLinear,
        original_img.GetOrigin(),
        new_spacing,
        original_img.GetDirection(),
        0,
        original_img.GetPixelID(),
    )

    lung_segmentation_map_original_size.CopyInformation(original_img)

    return lung_segmentation_map_original_size


def generate_and_save_two_channeled_image(
    output_dir, model, lung_segmentation_model_info, device, filename
):
    img = _read_image(filename)

    lung_mask = predict_lung_segmentation_map(
        filename, model, lung_segmentation_model_info, device
    )

    return sitk.WriteImage(
        sitk.JoinSeries([img, lung_mask]),
        os.path.join(output_dir, os.path.basename(filename)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare two channeled images with first channel being original CXR intensities and \
                     second channel being predicted lung probabilities."
    )

    parser.add_argument(
        "input_csv_path",
        type=str,
        help="Input CSV path containing column names as 'processed_Filename' and \
            'Output_tb_seg_filename' which represent paths of  CXRs and their\
            corresponding binary TB masks respectively",
    )

    parser.add_argument(
        "lung_segmentation_model_path",
        type=str,
        help="Model path for pretrained lung segmentation",
    )
    parser.add_argument(
        "lung_segmentation_model_info_path",
        type=str,
        help="Model info path for inference using lung segmentation model",
    )
    parser.add_argument(
        "output_pred_dir",
        type=str,
        help="Output Directory to ave the prediction images in their original \
              images",
    )
    parser.add_argument(
        "output_csv_filename",
        type=str,
        help="Output CSV filename to save the column 'processed_two_channeled_Filename' along \
              with the initial columns provided in the input csv path.",
    )
    args = parser.parse_args()

    with open(str(args.lung_segmentation_model_info_path)) as f:
        model_info = json.load(f)

    df = pd.read_csv(args.input_csv_path)
    if not os.path.exists(args.output_pred_dir):
        os.makedirs(args.output_pred_dir)

    model, device = _load_model(args.lung_segmentation_model_path)

    for file in df["processed_Filename"]:
        generate_and_save_two_channeled_image(
            args.output_pred_dir, model, model_info, device, file
        )

    df["processed_two_channeled_Filename"] = df["processed_Filename"].apply(
        lambda x: os.path.join(args.output_pred_dir, os.path.basename(x))
    )

    df.to_csv(args.output_csv_filename, index=False)


if __name__ == "__main__":
    main()
