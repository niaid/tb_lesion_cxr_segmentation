import argparse
import pandas as pd
from sklearn.model_selection import KFold


def split_train_val_test(df, num_folds=5, train_percentage=0.7, val_percentage=0.15):
    """

    This function generates train/val and test dataframes with the user
    provided train/val/test ratios. The input dataframe must contain the
    columns 'processed_Filename', 'Output_tb_seg_filename' and 'PatientID'
    so as to make sure that the input train/val/test
    dataframes does not contain the overlapping PatientIDs.

    ''

    Args:
        df(pd.DataFrame): pandas dataframe containing the columns 'processed_Filename',
                          'Output_tb_seg_filename' and 'PatientID'
    Returns:
          train_fodls[list of pd.DataFrame]: list of training sets with the same columns as the input dataframe.
          val_df[list of pd.DataFrame]: list of validation sets with the same columns as the input dataframe.
          test_df[list of pd.DataFrame]: list of testing sets with the same columns as the input dataframe.
    """

    # Number of splits
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    train_folds = []
    val_folds = []
    test_folds = []
    # Iterate over each fold
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(df), 1):
        # Create train/val and test splits
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        # Calculate sizes for training and validation sets
        train_size = round(len(df) * 0.7)  # 70% of full dataset
        val_size = round(len(df) * 0.15)  # 15% of full dataset

        # Split train_val_df into training and validation sets
        train_df = train_val_df[:train_size]  # noqa:E203
        val_df = train_val_df[train_size : train_size + val_size]  # noqa:E203

        # Ensure that each set contains non-overlapping patient IDs
        train_patient_ids = set(train_df["PatientID"])
        valid_patient_ids = set(val_df["PatientID"])

        # Remove patient IDs from validation set that appear in the training set.
        val_df = val_df[~val_df["PatientID"].isin(train_patient_ids)]

        # Remove patient IDs from test set that appear in the training set and validation set.
        test_df = test_df[
            ~test_df["PatientID"].isin(train_patient_ids.union(valid_patient_ids))
        ]

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
        type=str,
        help="Output prefix to save train , val and test sets.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of training set to entire dataset",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio of validation set to entire dataset",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv_path)

    train_folds, val_folds, test_folds = split_train_val_test(
        df,
        num_folds=args.num_folds,
        train_percentage=args.train_ratio,
        val_percentage=args.val_ratio,
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
