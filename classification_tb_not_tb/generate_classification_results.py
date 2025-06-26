import os
import glob
import json
import torch
import monai
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse
import math
import multiprocessing as mp
from functools import partial
import tempfile
from monai.transforms import Compose, AsDiscrete, Activations
from monai.transforms import (
    LoadImaged,
    Resized,
    NormalizeIntensityd,
    RepeatChanneld,
    EnsureChannelFirstd,
    ScaleIntensityd,
)
from monai.data import list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from segment_tb_cxr.auxiliary.compute_probability_of_TB_from_segmentation import (
    get_prob_of_tb,
)


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


def _load_model(lung_segment_model_path):
    """
    Load model from the pretrained lung segmentation model path
    Args:
        lung_segment_model_path(pathlib.Path): Pretrained Lung segmentation
                                               model path
    Returns:
       model(torch.nn.Module): Model architecture
       device(torch.device): Device to test the model on
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.jit.load(str(lung_segment_model_path), map_location=device)
    model.eval()

    return model, device


def _predict_mask(file_path, model, device, model_input_size, threshold=0.5):
    """
    Predict the lung mask from the resampled image array. This function uses
    trained segmentation models(torch) to segment lungs for a given image with
    size equal to model input size.
    Args:
        resampled_image_arr (numpy array): Numpy array obtained from resampled
                                           images to provide input for
                                           segmentation network in the
                                           shape of (num_images,
                                                     segmentation_input_size_x,
                                                     segmentation_input_size_y)
        model (torch.nn.Module): Lung Segmentation model.
        device(torch.device): Device to test the model on.
        model_input_size(int): Model input
        batch_size: Batch size for the model. Performing inference in batch
                    mode is faster than image by image.
    Returns:
        pred_masks(numpy array): Prediction masks of segmented lungs with same
                                  size as input array.
    """
    original_img = _read_image(file_path)

    temp_file = tempfile.NamedTemporaryFile(suffix=".nrrd", delete=False)
    temp_filename = temp_file.name

    sitk.WriteImage(original_img, temp_filename)

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=threshold)])
    test_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            Resized(keys=["img"], spatial_size=model_input_size, mode=("bilinear")),
            RepeatChanneld(keys=["img"], repeats=3),
            ScaleIntensityd(keys=["img"]),
            NormalizeIntensityd(
                keys=["img"],
                subtrahend=[0.485, 0.456, 0.406],
                divisor=[0.229, 0.224, 0.225],
                channel_wise=True,
            ),
        ]
    )

    test_files = [{"img": temp_filename}]
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
        roi_size = (96, 96)
        sw_batch_size = 4
        pred_mask = sliding_window_inference(test_image, roi_size, sw_batch_size, model)
        pred = post_trans(decollate_batch(pred_mask)[0])
        pred_mask = np.transpose(pred[1].cpu().numpy(), [1, 0]).astype(np.int32)
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

    temp_file.close()
    return pred_mask_original_size


def plot_confusion_matrix(pred_labels, ref_labels, output_confusion_matrix_filename):
    """
    Plot the confusion matrix using the repdicted labels and the reference labels.

    Inputs:
        pred_labels(list): Predicted labels
        ref_labels(list): Reference labels
        output_confusion_matrix_filename(str): Output confusion matrix filename.
    Outputs:
        ---
    """

    # Compute confusion matrix
    cm = confusion_matrix(ref_labels, pred_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NOT_TB", "TB"])
    disp.plot(cmap=plt.cm.Blues, colorbar=False)

    plt.xlabel("Predicted labels")
    plt.ylabel("Reference labels")
    plt.tight_layout()
    plt.savefig(output_confusion_matrix_filename)


def get_probability_and_prediction_label(
    filtered_probability_tb_segmentation_within_lungs, threshold_for_TB
):
    """
    Return the predicted label the the probability computed from the image is greater than the
    user provided threshold
    Inputs:
        filtered_probability_tb_segmentation_within_lungs(sitk.Image): SimpleITK image with intersection
                                                     of lung regions and the tb segmented regions.
        threshold_for_TB(int): user given threshold to qualify a given image contains "TB" or not.
    Outputs:
        "TB"/"NOT_TB": Predicted label.
    """
    probability_for_TB = get_prob_of_tb(
        sitk.GetArrayFromImage(filtered_probability_tb_segmentation_within_lungs)
    )[1]
    if probability_for_TB >= threshold_for_TB:
        return probability_for_TB, "TB"
    else:
        return probability_for_TB, "NOT_TB"


def generate_tb_masks_within_lungs(
    original_img_path,
    predicted_probability_segmentation_path,
    lung_segmentation_model,
    device,
    model_info,
):
    """
    Generate tb masks within lungs. Using the lung segmentation model, predict
    the lung segmentation masks and then generate the intersection of tb generated
    regions and the lung segmented masks.
    Inputs:
        original_img_path(string): Path for the image.
        tb_contours(list of lists ): List of all tb contours per image
        lung_segmentation_model(torch.nn.Module): loaded torch lung segmentation model.
        device(torch.Device): torch device. CUDA enabled GPU or cpu
        model_info(dict): Model dictionary for lung segmentation model to use the trained
                          lung segmentation model information to preprocess.
    Outputs:
        filtered_probability_tb_segmentation_within_lungs(sitk.Image): SimpleITK image with intensection
                                                     of lung regions and the probability tb segmented regions.
    """

    predicted_probability_tb_segmentation = _read_image(
        predicted_probability_segmentation_path
    )

    predicted_lung_mask = _predict_mask(
        original_img_path,
        lung_segmentation_model,
        device,
        model_input_size=model_info["img_size"],
    )

    predicted_lung_mask.SetSpacing(_read_image(original_img_path).GetSpacing())
    predicted_lung_mask = sitk.Cast(predicted_lung_mask, sitk.sitkFloat32)
    # Filter tb probability based image within lungs
    filtered_probability_tb_segmentation_within_lungs = (
        predicted_lung_mask * predicted_probability_tb_segmentation
    )

    return filtered_probability_tb_segmentation_within_lungs


def get_probabilities_and_prediction_labels(
    original_img_paths,
    predicted_probability_segmentation_paths,
    lung_segmentation_model_path,
    model_info,
    threshold_for_TB=1,
):
    """
    Generate predicted "TB"/"NOT_TB" labels using the lung segmentation model and
    the predicted tb segmentation file.
    Inputs:
        original_img_paths(list): Path for the original image paths.
        tb_contours(list of lists): List of all tb contours for all the images
        lung_segmentation_model(torch.nn.Module): loaded torch lung segmentation model.
        device(torch.Device): torch device. CUDA enabled GPU or cpu
        model_info(dict): Model dictionary for lung segmentation model to use the trained
                          lung segmentation model information to preprocess.
        threshold_for_TB(int): user given threshold to qualify a given image contains "TB" or not.
    Outputs:
        pred_tb_labels(list): List of predicted "TB"/"NOT_TB" labels
    """
    pred_tb_labels = []
    model, device = _load_model(lung_segmentation_model_path)
    probabilities_for_TB = []

    for original_img_path, predicted_probability_segmentation_path in zip(
        original_img_paths,
        predicted_probability_segmentation_paths,
    ):

        filtered_probability_tb_segmentation_within_lungs = (
            generate_tb_masks_within_lungs(
                original_img_path,
                predicted_probability_segmentation_path,
                model,
                device,
                model_info,
            )
        )

        probability_for_TB, pred_tb_label = get_probability_and_prediction_label(
            filtered_probability_tb_segmentation_within_lungs,
            threshold_for_TB=threshold_for_TB,
        )
        probabilities_for_TB.append(probability_for_TB)
        pred_tb_labels.append(pred_tb_label)

    return probabilities_for_TB, pred_tb_labels


def process_image(img, projection_axis, thumbnail_size):
    """
    Create a grayscale thumbnail image from the given image. If the image is 3D it is
    projected to 2D using a Maximum Intensity Projection (MIP) approach. Color images
    are converted to grayscale, and high dynamic range images are window leveled using
    a robust approach.

    Parameters
    ----------
    img (SimpleITK.Image): A 2D or 3D grayscale or sRGB image.
    projection_axis(int in [0,2]): The axis along which we project 3D images.
    thumbnail_size (list/tuple(int)): The 2D sizes of the thumbnail.

    Returns
    -------
    2D SimpleITK image with sitkUInt8 pixel type.

    """
    res = sitk.Resample(
        img,
        size=thumbnail_size,
        transform=sitk.Transform(),
        interpolator=sitk.sitkLinear,
        outputOrigin=img.GetOrigin(),
        outputSpacing=[
            (sz - 1) * spc / (nsz - 1)
            for nsz, sz, spc in zip(thumbnail_size, img.GetSize(), img.GetSpacing())
        ],
        outputDirection=img.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=img.GetPixelID(),
    )
    res.SetOrigin([0, 0])
    res.SetSpacing([1, 1])
    res.SetDirection([1, 0, 0, 1])
    return res


def visualize_single_file(file_name, imageIO, projection_axis, thumbnail_size):
    image_file_name = ""
    image = None

    img = sitk.ReadImage(file_name)
    image = process_image(img, projection_axis, thumbnail_size)
    image_file_name = file_name

    return (image_file_name, image)


def visualize_files(
    all_file_names,
    imageIO="",
    projection_axis=2,
    thumbnail_size=[64, 64],
    tile_size=[20, 20],
):
    """
    This function traverses the directory structure reading all user selected images
    (selction based on the image file format specified by the caller). All images are converted to 2D grayscale
    in [0,255] as follows:
    * Images with three channels are assumed to be in sRGB color space and converted to grayscale.
    * Grayscale images are window-levelled using robust values for the window-level accomodating
    * for outlying intensity values.
    * 3D images are converted to 2D using maximum intensity projection along the user specified projection axis.
    Parameters
    ----------
    root_dir (str): Path to the root of the data directory. Traverse the directory structure
                    and try to read every file as an image using the given imageIO.
    imageIO (str): Name of image IO to use. To see the list of registered image IOs use the
                   ImageFileReader::GetRegisteredImageIOs() or print an ImageFileReader.
                   The empty string indicates to read all file formats supported by SimpleITK.
    projection_axis (int in [0,2]): 3D images are converted to 2D using mean projection along the
                                    specified axis.
    thumbnail_size (2D tuple/list): The size of the 2D image tile used for visualization.
    tile_size (2D tuple/list): Number of tiles to use in x and y.

    Returns
    -------
    tuple(SimpleITK.Image, list): faux_volume comprised of tiles, file_name_list corrosponding
                                  to the image tiles.
                                  The SimpleITK image contains the meta-data 'thumbnail_size' and
                                  'tile_size'.
    """
    image_file_names = []
    faux_volume = None
    images = []

    with mp.Pool(processes=10) as pool:
        res = pool.map(
            partial(
                visualize_single_file,
                imageIO=imageIO,
                projection_axis=projection_axis,
                thumbnail_size=thumbnail_size,
            ),
            all_file_names,
        )
    res = [data for data in res if data[1] is not None]
    if res:
        image_file_names, images = zip(*res)
        if image_file_names:
            faux_volume = create_tile_volume(images, tile_size)
            faux_volume.SetMetaData(
                "thumbnail_size", " ".join([str(v) for v in thumbnail_size])
            )
            faux_volume.SetMetaData("tile_size", " ".join([str(v) for v in tile_size]))
    return (faux_volume, image_file_names)


def create_tile_volume(images, tile_size):
    """
    Create a faux-volume from a list of images. Each slice in the volume
    is constructed from tile_size[0]*tile_size[1] images. The slices are
    then joined to form the faux volume.

    Parameters
    ----------
    images (list(SimpleITK.Image(2D, sitkUInt8))): image list that we tile.
    tile_size (2D tuple/list): Number of tiles to use in x and y.

    Returns
    -------
    SimpleITK.Image(3D, sitkUInt8): Volume comprised of tiled image slices.
                                    Order of tiles matches the order of the input list.
    """
    step_size = tile_size[0] * tile_size[1]
    faux_volume = [
        sitk.Tile(images[i : i + step_size], tile_size, 0)  # noqa:E203
        for i in range(0, len(images), step_size)
    ]
    # if last tile image is smaller than others, add background content to match the size
    if len(faux_volume) > 1 and (
        faux_volume[-1].GetHeight() != faux_volume[-2].GetHeight()
        or faux_volume[-1].GetWidth() != faux_volume[-2].GetWidth()
    ):
        img = sitk.Image(faux_volume[-2]) * 0
        faux_volume[-1] = sitk.Paste(
            img, faux_volume[-1], faux_volume[-1].GetSize(), [0, 0], [0, 0]
        )
    return sitk.JoinSeries(faux_volume)


def plot_tile_volume(images, output_tile_filename):
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    faux_volume_image_files, image_file_list = visualize_files(
        images,
        imageIO="",
        projection_axis=2,
        thumbnail_size=[128, 128],
        tile_size=[rows, cols],
    )
    array = sitk.GetArrayFromImage(faux_volume_image_files)
    plt.imshow(array[0])
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_tile_filename)
    plt.clf()


def write_overlayed_images(ref_not_tb_pred_tb_dir, img_file, pred_file):

    image = sitk.Cast(sitk.RescaleIntensity(_read_image(img_file)), sitk.sitkUInt8)
    image_array = sitk.GetArrayFromImage(image)

    # Load and resize mask to match original image shape
    mask = _read_image(pred_file)
    # mask_resized = sitk.Resample(mask, image.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor)
    mask_array = sitk.GetArrayFromImage(mask)

    # Overlay mask on image using red color
    overlaid_image = np.stack([image_array] * 3, axis=-1)  # Convert to RGB
    overlaid_image[mask_array == 1] = [
        255,
        0,
        0,
    ]  # Set color to red where mask is present

    plt.imsave(
        os.path.join(
            ref_not_tb_pred_tb_dir, os.path.splitext(os.path.basename(img_file))[0]
        )
        + "_overlayed.png",
        overlaid_image.astype(np.uint8),
    )


def plot_fn_fp_tiles(
    df,
    output_dir_to_save_overlayed_fn_images,
    output_dir_to_save_overlayed_fp_images,
    output_csv_filename,
):

    if not os.path.isdir(output_dir_to_save_overlayed_fn_images):
        os.makedirs(output_dir_to_save_overlayed_fn_images)

    if not os.path.isdir(output_dir_to_save_overlayed_fp_images):
        os.makedirs(output_dir_to_save_overlayed_fp_images)

    # Save overlayed false positive files
    ref_not_tb_pred_tb = df[
        (df["TB_NOT_TB"] == "NOT_TB") & (df["predicted_TB_NOT_TB"] == "TB")
    ]
    with mp.Pool(30) as p:
        func = partial(write_overlayed_images, output_dir_to_save_overlayed_fp_images)
        p.starmap(
            func,
            zip(
                ref_not_tb_pred_tb["filename"].tolist(),
                ref_not_tb_pred_tb["ensemble_pred_tb_seg_file"].tolist(),
            ),
        )

    files = glob.glob(
        os.path.join(output_dir_to_save_overlayed_fp_images, "*_overlayed.png")
    )

    plot_tile_volume(
        files,
        output_csv_filename.split(".csv")[0] + "_fp_tile.png",
    )

    ref_tb_pred_not_tb = df[
        (df["TB_NOT_TB"] == "TB") & (df["predicted_TB_NOT_TB"] == "NOT_TB")
    ]
    with mp.Pool(30) as p:
        func = partial(write_overlayed_images, output_dir_to_save_overlayed_fn_images)
        p.starmap(
            func,
            zip(
                ref_tb_pred_not_tb["filename"].tolist(),
                ref_tb_pred_not_tb["ensemble_pred_tb_seg_file"].tolist(),
            ),
        )

    files = glob.glob(
        os.path.join(output_dir_to_save_overlayed_fn_images, "*_overlayed.png")
    )

    plot_tile_volume(
        files,
        output_csv_filename.split(".csv")[0] + "_fn_tile.png",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot confusion matrix from the input CSV files."
    )

    parser.add_argument(
        "input_csv_path",
        type=str,
        help="Input CSV path containing column names as 'filename' and \
            'ensemble_probability_pred_tb_seg_file' and 'TB/NOT-TB' label which represent paths of  CXRs, their\
             binary TB masks respectively and  their corresponding binary labels respectively.",
    )

    parser.add_argument(
        "lung_segmentation_model_info_json_path",
        type=str,
        help="Path to JSON file containing each segmentation model's keys and \
              their respective hyperparameters as values",
    )
    parser.add_argument(
        "lung_segmentation_model_path",
        type=str,
        help="Model path for the lung segmentation model.",
    )
    parser.add_argument(
        "output_csv_filename",
        type=str,
        help="Output CSV filename to save the column 'ensemble_pred_tb_seg_file' along \
              with the initial columns in the input_csv_file",
    )
    parser.add_argument(
        "--decision_for_TB",
        type=int,
        default=0.79336864,
        help="Threshold to classify as TB within lungs.",
    )
    parser.add_argument(
        "--output_confusion_matrix_filename",
        type=str,
        default="confusion_matrix.png",
        help="Filename for the output confusion matrix",
    )
    parser.add_argument(
        "--output_dir_to_save_fn_overlayed_images",
        type=str,
        default=".",
        help="Output directory to save overlayed images",
    )
    parser.add_argument(
        "--output_dir_to_save_fp_overlayed_images",
        type=str,
        default=".",
        help="Output directory to save overlayed images",
    )
    args = parser.parse_args()

    with open(str(args.lung_segmentation_model_info_json_path)) as f:
        lung_segmentation_model_info = json.load(f)

    df = pd.read_csv(args.input_csv_path)

    probs, labels = get_probabilities_and_prediction_labels(
        df["filename"].tolist(),
        df["ensemble_probability_pred_tb_seg_file"].tolist(),
        args.lung_segmentation_model_path,
        lung_segmentation_model_info,
        threshold_for_TB=args.decision_for_TB,
    )

    df["probability_for_TB"] = probs
    df["predicted_decision_for_TB_NOT_TB"] = labels

    df.to_csv(args.output_csv_filename, index=False)

    if "TB_NOT_TB" in df.columns:
        plot_confusion_matrix(
            df["predicted_TB_NOT_TB"].tolist(),
            df["TB_NOT_TB"].tolist(),
            args.output_confusion_matrix_filename,
        )

        plot_fn_fp_tiles(
            df,
            args.output_dir_to_save_fn_overlayed_images,
            args.output_dir_to_save_fp_overlayed_images,
            args.output_csv_filename,
        )


if __name__ == "__main__":
    main()
