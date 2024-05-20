import optuna
import argparse
import pathlib
import json
from functools import partial


def objective(trial, train_csv_path, val_csv_path, model_info):
    trial_model_info = {
        "fixed_variables": model_info["fixed_variables"],
        "range_variables": {},
        "categorical_variables": {},
    }

    for key in model_info["range_variables"].keys():
        if isinstance(model_info["range_variables"][key][0], int):
            trial_model_info["range_variables"][key] = trial.suggest_int(
                key,
                model_info["range_variables"][key][0],
                model_info["range_variables"][key][1],
            )
        else:
            trial_model_info["range_variables"][key] = trial.suggest_float(
                key,
                model_info["range_variables"][key][0],
                model_info["range_variables"][key][1],
            )
    for key in model_info["categorical_variables"].keys():
        trial_model_info["categorical_variables"][key] = trial.suggest_categorical(
            key, model_info["categorical_variables"][key]
        )

    return trial_model_info


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
    parser.add_argument(
        "output_csv_filename",
        type=str,
        help="output csv filename for hyperparameters inputs",
    )
    args = parser.parse_args()

    with open(str(args.model_info_json_path)) as f:
        model_info = json.load(f)

    # Create a partial function with fixed arguments
    objective_with_args = partial(
        objective,
        train_csv_path=args.train_input_csv_path,
        val_csv_path=args.val_input_csv_path,
        model_info=model_info,
    )

    # Run optimization
    study = optuna.create_study(objective_with_args, n_trials=args.num_trials)
    df = study.trials_dataframe()

    df.to_csv(args.output_csv_filename, index=False)


if __name__ == "__main__":
    main()
