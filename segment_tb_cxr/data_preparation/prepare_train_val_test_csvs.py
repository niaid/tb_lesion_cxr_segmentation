import argparse
import numpy as np
import pandas as pd


def split_train_val_test(df, num_folds=5, val_percentage=0.15, test_percentage=0.15):
    """

    This function generates train/val and test dataframes with the user
    provided train/val/test ratios. The input dataframe must contain the
    columns 'filename', 'Output_tb_seg_filename' and 'PatientID'
    so as to make sure that the input train/val/test
    dataframes does not contain the overlapping PatientIDs.

    The splitting is done at the patient level to ensure no patient appears
    in multiple sets (train, val, or test).

    Args:
        df(pd.DataFrame): pandas dataframe containing the columns 'processed_Filename',
                          'Output_tb_seg_filename' and 'PatientID'
        num_folds(int): number of folds to generate
        val_percentage(float): ratio of validation set to entire dataset
        test_percentage(float): ratio of test set to entire dataset
    Returns:
          train_folds[list of pd.DataFrame]: list of training sets with the same columns as the input dataframe.
          val_folds[list of pd.DataFrame]: list of validation sets with the same columns as the input dataframe.
          test_folds[list of pd.DataFrame]: list of testing sets with the same columns as the input dataframe.
    """

    # Get unique patient IDs and split at the patient level
    unique_patient_ids = df["PatientID"].unique()
    num_patients = len(unique_patient_ids)

    # Calculate number of patients for each split
    num_test_patients = int(num_patients * test_percentage)
    num_val_patients = int(num_patients * val_percentage)
    # Remaining patients go to training
    num_train_patients = num_patients - num_test_patients - num_val_patients

    train_folds = []
    val_folds = []
    test_folds = []

    for fold in range(num_folds):

        np.random.seed(3243 + fold)
        shuffled_patient_ids = np.random.permutation(unique_patient_ids)

        # Split into train, val, test based on calculated sizes
        train_patient_ids = set(shuffled_patient_ids[:num_train_patients])
        val_patient_ids = set(
            shuffled_patient_ids[
                num_train_patients : num_train_patients + num_val_patients  # noqa: E203
            ]
        )
        test_patient_ids = set(
            shuffled_patient_ids[num_train_patients + num_val_patients :]  # noqa: E203
        )

        # Create dataframes by selecting rows based on patient IDs
        train_df = df[df["PatientID"].isin(train_patient_ids)].copy()
        val_df = df[df["PatientID"].isin(val_patient_ids)].copy()
        test_df = df[df["PatientID"].isin(test_patient_ids)].copy()

        train_folds.append(train_df)
        val_folds.append(val_df)
        test_folds.append(test_df)

    return train_folds, val_folds, test_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csv_path",
        type=str,
        help="Input CSV file for TB Portals containing zhying's annotations. CSV file\
              column names processed_Filename, PatientID and Output_tb_seg_filename. \
              This csv file should be the file that is generated from data_prep.py file \
              from data_prepartion folder",
    )
    parser.add_argument(
        "output_prefix_for_csv_filename",
        type=str,
        help="Output prefix to save train , val and test sets.",
    )
    parser.add_argument(
        "num_folds",
        type=int,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio of validation set to entire dataset",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Ratio of test set to entire dataset",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv_path)

    train_folds, val_folds, test_folds = split_train_val_test(
        df,
        num_folds=args.num_folds,
        val_percentage=args.val_ratio,
        test_percentage=args.test_ratio,
    )

    for fold in range(args.num_folds):
        train_folds[fold].to_csv(
            args.output_prefix_for_csv_filename + "_fold_" + str(fold) + "_train.csv",
            index=False,
        )
        val_folds[fold].to_csv(
            args.output_prefix_for_csv_filename + "_fold_" + str(fold) + "_val.csv",
            index=False,
        )
        test_folds[fold].to_csv(
            args.output_prefix_for_csv_filename + "_fold_" + str(fold) + "_test.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
