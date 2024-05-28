import json
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from segment_lung_cxr.inference.inference_lung_segment import (
    _read_image,
    _load_model,
    _predict_mask,
)
import argparse
import math
import multiprocessing as mp
from functools import partial


def plot_confusion_matrix(pred_labels, ref_labels, output_confusion_matrix_filename):

    # Compute confusion matrix
    cm = confusion_matrix(ref_labels, pred_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NOT_TB", "TB"])
    disp.plot(cmap=plt.cm.Blues, colorbar=False)

    plt.xlabel("Predicted labels")
    plt.ylabel("Reference labels")
    plt.tight_layout()
    plt.savefig(output_confusion_matrix_filename)


def get_pred_label(filtered_tb_mask_within_lungs, threshold=1):

    if sum(sitk.GetArrayFromImage(filtered_tb_mask_within_lungs)) >= threshold:
        return "TB"
    else:
        return "NOT_TB"


def generate_tb_masks_within_lungs(
    original_img_path,
    predicted_segmentation_path,
    lung_segmentation_model,
    device,
    model_info,
):

    predicted_tb_segmentation = _read_image(predicted_segmentation_path)

    predicted_lung_mask = _predict_mask(
        original_img_path,
        lung_segmentation_model,
        device,
        model_input_size=model_info["img_size"],
    )

    predicted_lung_mask.SetSpacing(predicted_tb_segmentation.GetSpacing())
    predicted_tb_segmentation = sitk.Cast(predicted_tb_segmentation, sitk.sitkUInt8)

    # Filter tb masks within lungs
    filtered_tb_mask_within_lungs = predicted_lung_mask * predicted_tb_segmentation

    return filtered_tb_mask_within_lungs


def get_pred_labels(
    original_img_paths,
    predicted_segmentation_paths,
    lung_segmentation_model_path,
    model_info,
    threshold=1,
):

    pred_tb_labels = []
    model, device = _load_model(lung_segmentation_model_path)
    for original_img_path, predicted_segmentation_path in zip(
        original_img_paths, predicted_segmentation_paths
    ):
        filtered_tb_mask_within_lungs = generate_tb_masks_within_lungs(
            original_img_path, predicted_segmentation_path, model, device, model_info
        )

        pred_tb_label = get_pred_label(
            filtered_tb_mask_within_lungs, threshold=threshold
        )

        pred_tb_labels.append(pred_tb_label)

    return pred_tb_labels


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
    try:
        img = _read_image(file_name)
        image = process_image(img, projection_axis, thumbnail_size)
        image_file_name = file_name
    except:  # noqa:E722
        pass
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
        thumbnail_size=[64, 64],
        tile_size=[rows, cols],
    )
    array = sitk.GetArrayFromImage(faux_volume_image_files)
    plt.imshow(array[0], cmap="gray")
    plt.axis("off")
    plt.savefig(output_tile_filename)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot confusion matrix from the input CSV files."
    )

    parser.add_argument(
        "input_csv_path",
        type=str,
        help="Input CSV path containing column names as 'processed_Filename' and \
            'Output_tb_seg_filename' and 'TB/NOT-TB' label which represent paths of  CXRs, their\
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
        help="Output CSV filename to save the column 'pred_tb_seg_file' along \
              with the initial columns in the input_csv_file",
    )
    parser.add_argument(
        "num_pixels_threshold",
        type=int,
        help="Threshold for no. of pixels to classify as TB in tb segmentation within lungs.",
    )
    parser.add_argument(
        "output_confusion_matrix_filename",
        type=str,
        help="Filename for the output confusion matrix",
    )
    args = parser.parse_args()

    with open(str(args.lung_segmentation_model_info_json_path)) as f:
        lung_segmentation_model_info = json.load(f)

    df = pd.read_csv(args.input_csv_path)

    df["predicted_TB_NOT_TB"] = get_pred_labels(
        df["processed_Filename"].tolist(),
        df["pred_tb_seg_file"].tolist(),
        args.lung_segmentation_model_path,
        lung_segmentation_model_info,
        threshold=args.num_pixels_threshold,
    )

    df.to_csv(args.output_csv_filename, index=False)
    plot_confusion_matrix(
        df["predicted_TB_NOT_TB"].tolist(),
        df["TB_NOT_TB"].tolist(),
        args.output_confusion_matrix_filename,
    )

    # Plot the ones which have reference labels as "TB" but predicted as "NOT_TB"
    plot_tile_volume(
        df[(df["TB_NOT_TB"] == "NOT_TB") & (df["predicted_TB_NOT_TB"] == "TB")][
            "processed_Filename"
        ].tolist(),
        args.output_csv_filename.split(".csv")[0] + "_TB_tile.png",
    )


if __name__ == "__main__":
    main()
