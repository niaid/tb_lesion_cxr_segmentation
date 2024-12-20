import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom
from monai.transforms import Compose, AsDiscrete, Activations
import monai
from monai.data import list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from segment_tb_cxr.unet_resnet18.training.train_tb_segment import get_transforms


"""
This script is used to predict the binary TB masks on the test Chest X Rays
using the pretrained TB segmentation model. User needs to provide the hyperparameters file
that is shared across training and inference. This JSON file is located in the
training folder (segment_tb_cxr/unet_resnet18/training/unet_resnet18_params.json)
User needs to provide the output prediction directory name to save the binary
TB masks and output csv filename with an extra column name 'customunet_pred_tb_seg_file'
corresponding to the original filenames in the given folder with the format of
 {filename}_customunet_pred_seg.nrrd
"""


def file_path(path):

    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a file path or file does not exist."
        )


def csv_path(path, required_columns={"filename"}):
    """
    Define the csv_path type for use with argparse. Checks
    that the given path string is a path to a csv file and that the
    header of the csv file contains the required columns.
    """

    required_columns = set(required_columns)
    if os.path.isfile(path):
        try:  # only read the csv header
            expected_columns_exist = required_columns.issubset(
                set(pd.read_csv(path, nrows=0).columns.tolist())
            )
            if expected_columns_exist:
                return path
            else:
                raise argparse.ArgumentTypeError(
                    f"Invalid argument ({path}), does not contain all expected columns."
                )
        except UnicodeDecodeError:
            raise argparse.ArgumentTypeError(
                f"Invalid argument ({path}), not a csv file."
            )
    else:
        raise argparse.ArgumentTypeError(f"Invalid argument ({path}), not a file.")


def _get_channels(model):
    """

    Get number of channels from the trained model.

    Args:

       model(torch.nn.Module): Model architecture

    Returns:

       num_channels(int): Number of channels that the model was trained with.

    """
    first_parameter = next(model.parameters())
    input_shape = first_parameter.size()
    num_channels = input_shape[1]
    return num_channels


def _load_model(tb_segment_model_path):
    """
    Load model from the pretrained lung segmentation model path
    Args:
        tb_segment_model_path(pathlib.Path): Pretrained TB segmentation
                                               model path
    Returns:
       model(torch.nn.Module): Model architecture
       device(torch.device): Device to test the model on
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.jit.load(str(tb_segment_model_path), map_location=device)
    model.eval()

    return model, device


def _srgb2gray(image):
    # Convert sRGB image to gray scale and rescale results to [0,255]
    channels = [
        sitk.VectorIndexSelectionCast(image, i, sitk.sitkFloat32)
        for i in range(image.GetNumberOfComponentsPerPixel())
    ]
    # linear mapping
    gray_image = (
        1 / 255.0 * (0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2])
    )
    # nonlinear gamma correction
    gray_image = (
        gray_image * sitk.Cast(gray_image <= 0.0031308, sitk.sitkFloat32) * 12.92
        + gray_image ** (1 / 2.4)
        * sitk.Cast(gray_image > 0.0031308, sitk.sitkFloat32)
        * 1.055
        - 0.055
    )
    return sitk.Cast(sitk.RescaleIntensity(gray_image), sitk.sitkUInt8)


def _read_image(file):
    try:
        org_img = sitk.ReadImage(file)
    except:  # noqa E722
        ds = pydicom.dcmread(file)
        org_img = sitk.GetImageFromArray(
            ds.pixel_array, isVector=(len(ds.pixel_array.shape) == 3)
        )
    # Some images have a 3rd dimension of size 1, get rid of it.
    if org_img.GetDimension() != 2 and org_img.GetSize()[2] == 1:
        org_img = org_img[:, :, 0]
    # Some images are grayscale but the channel is repeated three times
    # (gray RGB image).
    if org_img.GetNumberOfComponentsPerPixel() > 1:
        org_img = _srgb2gray(org_img)

    return org_img


def _resample_cxr(new_size, gaussian_sigma, org_img):
    """
    Downsample the input image to the given new_size. To avoid aliasing
    artifacts you may want to blur the image before the downsampling operation.
    This is important if your image contains high frequency data.
    Args:
        new_size: The size of the resampled image in pixels.
        gaussian_sigma(scalar or tuple with image dimension length): If given,
               blur the image with a Gaussian with the given standard
               deviation(s) before resampling.
        org_img (SimpleITK.Image): original SimpleITK Image object.
    Returns:
        resampled_for_seg (SimpleITK.Image): Resampled image.
    """

    new_spacing = [
        sz * spc / nsz
        for nsz, sz, spc in zip(new_size, org_img.GetSize(), org_img.GetSpacing())
    ]
    smoothed_image = sitk.SmoothingRecursiveGaussian(org_img, gaussian_sigma)
    resampled_for_seg = sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        org_img.GetOrigin(),
        new_spacing,
        org_img.GetDirection(),
        0,
        sitk.sitkFloat32,
    )

    return resampled_for_seg


def _predict_mask(file_path, model, device, model_info):
    """
    Predict the lung mask from the resampled image array. This function uses
    trained segmentation models(torch) to segment lungs for a given image with
    size equal to model input size.
    Args:
        file_path(str): file path to the input Chest x Ray.
        model (torch.nn.Module): Lung Segmentation model.
        model_info(dict): Dictionary containing the information regarding the model parameters.
    Returns:
        pred_mask_original_size(SimpleITK.Image): SimpleITK image object of the predicted
                                                 TB mask with the same size as the
                                                 original image size.
    """
    original_img = _read_image(file_path)

    post_trans = Compose(
        [
            Activations(sigmoid=True),
            AsDiscrete(threshold=model_info["threshold"]),
        ]
    )
    train_transforms, val_transforms, test_transforms = get_transforms(model_info)

    test_files = [{"img": file_path}]
    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    with torch.no_grad():
        test_data = next(iter(test_loader))
        test_image = test_data["img"].to(device)
        roi_size = model_info["roi_size"]
        sw_batch_size = model_info["sw_batch_size"]
        pred_mask = sliding_window_inference(test_image, roi_size, sw_batch_size, model)
        pred = post_trans(decollate_batch(pred_mask)[0])
        pred_mask_0 = np.transpose(pred[0].cpu().numpy(), [1, 0]).astype(np.int32)
        pred_mask_1 = np.transpose(pred[1].cpu().numpy(), [1, 0]).astype(np.int32)
        pred_mask_2 = np.transpose(pred[2].cpu().numpy(), [1, 0]).astype(np.int32)
        pred_mask = ((pred_mask_0 > 0) | (pred_mask_1 > 0) | (pred_mask_2 > 0)).astype(
            np.int32
        )
    pred_mask = sitk.GetImageFromArray(pred_mask)
    new_spacing = [
        sz * spc / nsz
        for nsz, sz, spc in zip(
            original_img.GetSize(), pred_mask.GetSize(), pred_mask.GetSpacing()
        )
    ]
    pred_mask_original_size = sitk.Resample(
        pred_mask,
        original_img.GetSize(),
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        original_img.GetOrigin(),
        new_spacing,
        original_img.GetDirection(),
        0,
        sitk.sitkUInt8,
    )

    return pred_mask_original_size


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict TB segmentation regions within Chest X Rays using\
                    custom UNet model."
    )
    parser.add_argument(
        "input_csv_path",
        type=csv_path,
        help="Input CSV path containing column names as 'filename'",
    )

    parser.add_argument(
        "tb_segmentation_model_path",
        type=file_path,
        help="Customer UNet Model path for pretrained TB segmentation",
    )
    parser.add_argument(
        "output_seg_dir",
        type=str,
        help="Output Directory to ave the prediction images in their original \
              images",
    )
    parser.add_argument(
        "model_info_json_path",
        type=file_path,
        help="Path to JSON file containing each segmentation model's keys and \
              their respective hyperparameters as values",
    )
    parser.add_argument(
        "output_csv_filename",
        type=str,
        help="Output CSV filename to save the column 'customunet_pred_tb_seg_file' along \
              with the initial columns in the input_csv_file",
    )
    args = parser.parse_args()

    with open(str(args.model_info_json_path)) as f:
        model_info = json.load(f)

    model, device = _load_model(args.tb_segmentation_model_path)

    if not os.path.exists(args.output_seg_dir):
        os.makedirs(args.output_seg_dir)

    df = pd.read_csv(args.input_csv_path)

    for file in df["filename"]:
        mask = _predict_mask(file, model, device, model_info)
        output_filename = os.path.join(
            args.output_seg_dir,
            os.path.splitext(os.path.basename(file))[0] + "_customunet_pred_seg.nrrd",
        )
        sitk.WriteImage(
            mask,
            output_filename,
            useCompression=True,
        )

    df["customunet_pred_tb_seg_file"] = df["filename"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0] + "_customunet_pred_seg.nrrd"
    )
    df.to_csv(args.output_csv_filename, index=False)


if __name__ == "__main__":
    main()
