import numpy as np
import pandas as pd
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureType,
    Resized,
    NormalizeIntensityd,
    RepeatChanneld,
    RandCropByPosNegLabeld,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandRotated,
)
from monai.inferers import sliding_window_inference
import torch
from monai.visualize import plot_2d_or_3d_image
import monai
from monai.data import decollate_batch, DataLoader
import time
from datetime import timedelta
import argparse
import pathlib
import json
from segment_tb_cxr.unet_resnet18.training.unet_resnet18 import ResNetUNet
from torch.utils.tensorboard import SummaryWriter

"""
This script is used to train the TB segmentation model in Chest X Rays
that has 64%(approx.) of each dataset and valids on combined files that
16%(approx.) of each dataset.
"""


def calculate_validation_loss(model, model_info, loss_function, val_loader, post_trans):
    device = torch.device("cuda")
    model.eval()
    with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
        val_epoch_loss = 0
        step = 0
        for val_data in val_loader:
            step += 1
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(
                device
            )
            roi_size = model_info["fixed_variables"]["roi_size"]
            sw_batch_size = model_info["fixed_variables"]["sw_batch_size"]
            val_outputs = sliding_window_inference(
                val_images, roi_size, sw_batch_size, model
            )
            val_loss = loss_function(val_outputs, val_labels)
            val_epoch_loss += val_loss.item()
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

        val_epoch_loss /= step

        return val_images, val_labels, val_outputs, val_epoch_loss


def custom_train_collate_batch(batch):
    """
    Custom collate_fn that ignores the 'endian' key.

    Args:
      batch: A batch of data.

    Returns:
      A batch of data with the 'endian' key ignored.
    """
    batch_data = {}

    batch_data["img"] = torch.stack(
        [
            torch.tensor(batch[i][j]["img"])
            for i in range(len(batch))
            for j in range(len(batch[i]))
        ]
    )
    batch_data["seg"] = torch.stack(
        [
            torch.tensor(batch[i][j]["seg"])
            for i in range(len(batch))
            for j in range(len(batch[i]))
        ]
    )

    return batch_data


def custom_val_collate_batch(batch):
    """
    Custom collate_fn that ignores the 'endian' key.

    Args:
      batch: A batch of data.

    Returns:
      A batch of data with the 'endian' key ignored.
    """
    batch_data = {}

    batch_data["img"] = torch.stack(
        [torch.tensor(batch[i]["img"]) for i in range(len(batch))]
    )
    batch_data["seg"] = torch.stack(
        [torch.tensor(batch[i]["seg"]) for i in range(len(batch))]
    )

    return batch_data


def get_transforms(model_info):
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Resized(
                keys=["img", "seg"],
                spatial_size=model_info["fixed_variables"]["img_size"],
                mode=("bilinear", "nearest"),
            ),
            RepeatChanneld(keys=["img", "seg"], repeats=3),
            ScaleIntensityd(keys=["img", "seg"]),
            NormalizeIntensityd(
                keys=["img"],
                subtrahend=model_info["fixed_variables"]["means"],
                divisor=model_info["fixed_variables"]["standard_deviation"],
                channel_wise=True,
            ),
            RandCropByPosNegLabeld(
                keys=["img", "seg"],
                label_key="seg",
                spatial_size=model_info["fixed_variables"]["spatial_size"],
                pos=int(model_info["range_variables"]["pos"]),
                neg=int(model_info["range_variables"]["neg"]),
                num_samples=int(model_info["range_variables"]["num_crop_samples"]),
            ),
            RandRotated(
                keys=["img", "seg"],
                range_x=(
                    np.deg2rad(-int(model_info["range_variables"]["rotation_degree"])),
                    np.deg2rad(int(model_info["range_variables"]["rotation_degree"])),
                ),
                prob=model_info["range_variables"]["prob_rotation"],
                mode=("bilinear", "nearest"),
            ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Resized(
                keys=["img", "seg"],
                spatial_size=model_info["fixed_variables"]["img_size"],
                mode=("bilinear", "nearest"),
            ),
            RepeatChanneld(keys=["img", "seg"], repeats=3),
            ScaleIntensityd(keys=["img", "seg"]),
            NormalizeIntensityd(
                keys=["img"],
                subtrahend=model_info["fixed_variables"]["means"],
                divisor=model_info["fixed_variables"]["standard_deviation"],
                channel_wise=True,
            ),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            Resized(
                keys=["img"],
                spatial_size=model_info["fixed_variables"]["img_size"],
                mode=("bilinear"),
            ),
            RepeatChanneld(keys=["img"], repeats=3),
            ScaleIntensityd(keys=["img"]),
            NormalizeIntensityd(
                keys=["img"],
                subtrahend=model_info["fixed_variables"]["means"],
                divisor=model_info["fixed_variables"]["standard_deviation"],
                channel_wise=True,
            ),
        ]
    )

    return train_transforms, val_transforms, test_transforms


def _configure_data_preprocess(train_csv_path, val_csv_path, model_info):
    """

    This function prepares data loaders for training and validation data
    useful for model training.

    Args:

        train_csv_path(list): Input CSV path containing training files with
                              column names as 'Filename' and
                              'Output_tb_seg_filename' representing
                              CXR and the corresponding binary TB mask
                              respectively.
        val_csv_path(list):Input CSV path containing val files with
                              column names as 'Filename' and
                              'Output_tb_seg_filename' representing
                              CXR and the corresponding binary TB mask
                              respectively.
        model_info(dict): Dictionary containing information regarding training
                        the model.
    Returns:

        train_loader(torch.utils.data.DataLoader):Data Loader containing
                                        training files and train transforms.
        val_loader(torch.utils.data.DataLoader):Data Loader contaning valid
                                                files and valid transforms.
    """
    train_files = pd.read_csv(train_csv_path)
    val_files = pd.read_csv(val_csv_path)

    train_files = [
        {
            "img": cxr_file,
            "seg": ref_file,
        }
        for cxr_file, ref_file in zip(
            train_files["processed_Filename"], train_files["Output_tb_seg_filename"]
        )
    ]
    val_files = [
        {
            "img": cxr_file,
            "seg": ref_file,
        }
        for cxr_file, ref_file in zip(
            val_files["processed_Filename"], val_files["Output_tb_seg_filename"]
        )
    ]

    train_transforms, val_transforms, test_transforms = get_transforms(model_info)
    # create a training data loader
    train_ds = monai.data.SmartCacheDataset(
        data=train_files, transform=train_transforms
    )
    # use batch_size to load images and use RandCropByPosNegLabeld to
    # generate batch_size x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=model_info["range_variables"]["batch_size"],
        shuffle=True,
        num_workers=model_info["range_variables"]["num_workers"],
        collate_fn=custom_train_collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    # create a training data loader
    val_ds = monai.data.SmartCacheDataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=model_info["range_variables"]["batch_size"],
        num_workers=model_info["range_variables"]["num_workers"],
        collate_fn=custom_val_collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def get_params(df, model_info, index):
    for key in model_info["range_variables"].keys():
        model_info["range_variables"][key] = df["params_" + key].values[index]
    for key in model_info["categorical_variables"].keys():
        model_info["categorical_variables"][key] = df["params_" + key].values[index]

    return model_info


def train_model(
    train_loader,
    val_loader,
    model_info,
    device_id,
    output_model_filename,
    plot_images_for_debugging=True,
):
    """
    Train the model with network loaded with hyper parameters. Model saves with
    the {output_model_filename} that user gives based on the best dice metric on
    the validation data.

    Args:
        train_loader(torch.utils.data.DataLoader):Data Loader containing training files and train transforms.
        val_loader(torch.utils.data.DataLoader):Data Loader contaning valid files and valid transforms.
        model_info(dict): Dictionary containing information regarding training the model.
        output_model_filename(str): Output filename for the best model to save.

    Returns:

       ---

    """
    device = torch.device("cuda:" + str(device_id))

    model = ResNetUNet(3).to(device)  # Input takes 3 channels encoder is initialized
    # with 'imagenet' weights.

    post_trans = Compose(
        [
            EnsureType(),
            Activations(sigmoid=True),
            AsDiscrete(logit_thresh=model_info["fixed_variables"]["threshold"]),
        ]
    )

    loss_function = monai.losses.DiceFocalLoss(sigmoid=True)
    if model_info["categorical_variables"]["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=model_info["range_variables"]["learning_rate"]
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=model_info["range_variables"]["learning_rate"],
            momentum=model_info["range_variables"]["momentum"],
        )

    writer = SummaryWriter(comment=output_model_filename.split("/")[-1].split(".pt")[0])

    val_interval = 1
    epochs = model_info["range_variables"]["epochs"]
    best_val_loss = float("inf")

    for epoch in range(model_info["range_variables"]["epochs"]):
        t1 = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/" + str(epochs))
        model.train()
        epoch_loss = 0
        step = 0
        for i, batch_data in enumerate(train_loader):
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        if (epoch + 1) % val_interval == 0:
            (
                val_images,
                val_labels,
                val_outputs,
                val_epoch_loss,
            ) = calculate_validation_loss(
                model, model_info, loss_function, val_loader, post_trans
            )
            writer.add_scalars(
                "Loss",
                {"train_loss": epoch_loss, "val_loss": val_epoch_loss},
                epoch + 1,
            )

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_val_loss_epoch = epoch + 1
                torch.jit.save(
                    torch.jit.script(model),
                    output_model_filename.split(".pt")[0] + "_loss.pt",
                )
                print("saved new best val loss model")

            print(
                "current epoch: {} current val loss: {:.4f} \
                best val loss: {:.4f} at epoch {}".format(
                    epoch + 1, val_epoch_loss, best_val_loss, best_val_loss_epoch
                )
            )

            if plot_images_for_debugging:
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(
                    val_outputs, epoch + 1, writer, index=0, tag="output"
                )

        t2 = time.time()
        print("time elapsed for the epoch:" + str(str(timedelta(seconds=t2 - t1))))

    torch.jit.save(
        torch.jit.script(model), output_model_filename.split(".pt")[0] + "_lastepoch.pt"
    )
    print(
        f"train completed, best_val_loss: {best_val_loss:.4f} at epoch: \
            {best_val_loss_epoch}"
    )
    writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="TB Segmentation Model in Chest X Rays."
    )
    parser.add_argument(
        "model_info_json_path",
        type=pathlib.Path,
        help="Path to JSON file containing each segmentation model's keys and \
              their respective hyperparameters as values",
    )
    parser.add_argument(
        "train_input_csv_path",
        type=pathlib.Path,
        help="Input CSV path containing column names as 'Filename' ,\
                'Output_tb_seg_filename'which represent training files \
                of Chest X Rays and their respective reference labels \
                respectively",
    )

    parser.add_argument(
        "val_input_csv_path",
        type=pathlib.Path,
        help="Input CSV path containing column names as 'Filename' ,\
             'Output_tb_seg_filename' which represent validation files of \
             Chest X Rays and their respective reference labels respectively",
    )
    parser.add_argument(
        "output_model_filename",
        type=str,
        help="Filename for best model to save. The best model saves on\
              based on its best dice score on valid data",
    )
    parser.add_argument(
        "trial_values_path",
        type=pathlib.Path,
        help="Dataframe containing values of the trials.",
    )
    parser.add_argument(
        "index",
        type=str,
        help="Index in the dataframe or trial number",
    )

    args = parser.parse_args()

    with open(str(args.model_info_json_path)) as f:
        model_info = json.load(f)

    trial_values_df = pd.read_csv(args.trial_values_path)
    model_info = get_params(trial_values_df, model_info, args.index)
    # Get training and validation data loaders for training and save the best
    # model based on validation data
    train_loader, val_loader = _configure_data_preprocess(
        train_csv_path=args.train_input_csv_path,
        val_csv_path=args.val_input_csv_path,
        model_info=model_info,
    )

    # Train the model using the dataloaders and the model info
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model_info=model_info,
        output_model_filename=args.output_model_filename,
        plot_images_for_debugging=args.plot_images_ffor_debugging,
    )


if __name__ == "__main__":
    main()
