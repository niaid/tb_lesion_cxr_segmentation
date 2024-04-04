import optuna
import monai
from segment_tb_cxr.unet_resnet18.training.train_tb_segment import (
    train_model,
    _configure_data_preprocess,
    calculate_validation_loss,
)
from monai.transforms import Activations, AsDiscrete, Compose, EnsureType
import argparse
import pathlib
import json
from segment_tb_cxr.unet_resnet18.inference.inference_tb_segment import _load_model
from functools import partial
import pickle


def objective(
    trial, train_csv_path, val_csv_path, model_info, output_model_filename, key
):
    trial_model_info = {}
    for key in model_info["range_variables"].keys():
        if type(model_info[key][0]) == int:
            trial_model_info[key] = trial.suggest_int(
                key,
                model_info["range_variables"][key][0],
                model_info["range_variables"][key][1],
            )
        elif type(model_info[key][0]) == str:
            trial_model_info[key] = trial.suggest_categorical(
                key, model_info["range_variables"][key]
            )
        else:
            trial_model_info[key] = trial.suggest_float(
                key,
                model_info["range_variables"][key][0],
                model_info["range_variables"][key][1],
            )

    print(trial_model_info)
    train_loader, val_loader = _configure_data_preprocess(
        train_csv_path, val_csv_path, trial_model_info
    )

    train_model(train_loader, val_loader, model_info, output_model_filename)

    model, device = _load_model(output_model_filename.split(".pt")[0] + "_loss.pt")

    loss_function = monai.losses.DiceFocalLoss(sigmoid=True)
    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(logit_thresh=0.5)]
    )
    # Load the trained model and evaluate on the validation set
    # Calculate the validation loss
    val_images, val_labels, val_outputs, val_epoch_loss = calculate_validation_loss(
        model, model_info, loss_function, val_loader, post_trans
    )

    trial.report(val_epoch_loss, trial_model_info)

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_epoch_loss


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="TB Segmentation Model in Chest X Rays."
    )
    parser.add_argument(
        "model_info_json_path",
        type=pathlib.Path,
        help="Path to JSON file containing each segmentation model's keys and their respective \
                hyperparameters as values",
    )
    parser.add_argument(
        "train_input_csv_path",
        type=pathlib.Path,
        help="Input CSV path containing column names as 'Filename' ,'Output_tb_seg_filename'which represent \
                                training files of Chest X Rays and their respective \
                                 reference labels respectively",
    )

    parser.add_argument(
        "val_input_csv_path",
        type=pathlib.Path,
        help="Input CSV path containing column names as 'Filename' ,'Output_tb_seg_filename' which represent \
                                validation files of Chest X Rays and their respective \
                                 reference labels respectively",
    )
    parser.add_argument(
        "output_model_filename",
        type=str,
        help="Filename for best model to save. The best model saves on\
              based on its best dice score on valid data",
    )
    parser.add_argument("num_trials", type=int, help="No. of trials")
    parser.add_argument("study_name", type=str, help="Study name in pickle format")
    args = parser.parse_args()

    with open(str(args.model_info_json_path)) as f:
        model_info = json.load(f)

    # Create a partial function with fixed arguments
    objective_with_args = partial(
        objective,
        train_csv_path=args.train_input_csv_path,
        val_csv_path=args.val_input_csv_path,
        model_info=model_info,
        output_model_filename=args.output_model_filename,
    )

    # Run optimization
    study = optuna.create_study()
    study.optimize(objective_with_args, n_trials=args.num_trials)

    # Save study to a file
    with open(args.study_name, "wb") as f:
        pickle.dump(study, f)

    # Get feature importances
    importance = optuna.importance.get_param_importances(study)
    print(importance)
    print("Best hyperparameters:", study.best_params)


if __name__ == "__main__":
    main()
