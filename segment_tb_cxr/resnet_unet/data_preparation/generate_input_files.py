import argparse
import pathlib
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

"""
This script prepares train/val/test input CSV paths useful for training and
inference for lung segmentation in Chest X-Rays.
"""


def _get_all_fold_files(input_csv_path, num_folds):
    """
    This function prepares a list of dictionary objects with key-value pairs as
    'train':{train_files},'val':{valid_files},'test':{test_files} for each
    corresponding fold number
    Args:
        input_csv_path(string): Input CSV path contaning Chest -X Ray(CXR) and
                                binary lung masks and the correspnding dataset
                                they belong to in columns 'cxr_file' , 'ref_seg_file'
                                and 'dataset' respectively.
        num_folds(int): Total number of folds for Cross validation
    Returns:
        all_folds(dict): List of dictionary objects containing keys as 'train',
                          'val' and 'test' with corresponding training files, validation
                          files and test files as values respectively.
    """
    df = pd.read_csv(str(input_csv_path))

    cxr_files = df["cxr_file"].tolist()
    seg_files = df["ref_seg_file"].tolist()

    kf_train_test = KFold(n_splits=num_folds, shuffle=False)

    all_folds = []

    for train_val_idx, test_idx in kf_train_test.split(cxr_files):
        cxr_files_test = [cxr_files[idx] for idx in test_idx]
        seg_files_test = [seg_files[idx] for idx in test_idx]

        cxr_files_train_val = [cxr_files[idx] for idx in train_val_idx]
        seg_files_train_val = [seg_files[idx] for idx in train_val_idx]

        (
            cxr_files_train,
            cxr_files_val,
            seg_files_train,
            seg_files_val,
        ) = train_test_split(
            cxr_files_train_val, seg_files_train_val, test_size=0.2, random_state=42
        )
        train_files = [
            {"img": img_file, "seg": seg_file}
            for img_file, seg_file in zip(cxr_files_train, seg_files_train)
        ]
        val_files = [
            {"img": img_file, "seg": seg_file}
            for img_file, seg_file in zip(cxr_files_val, seg_files_val)
        ]

        test_files = [
            {"img": img_file, "seg": seg_file}
            for img_file, seg_file in zip(cxr_files_test, seg_files_test)
        ]

        all_folds.append({"train": train_files, "val": val_files, "test": test_files})

    return all_folds


def save_fold_files_to_CSV(
    all_folds_files, fold_num, input_csv_path, output_suffix_filename
):
    """
    Save CSV file with column names as 'cxr_file' and 'ref_seg_file' representing the
    filepaths for CXRs and reference binary masks respectively for each of the
    train/val/test sets saves it in the same folder as the input csv path.
    Args:
        all_folds_files(list): List of dictionary objects containing keys as 'train',
                          'val' and 'test' with corresponding training files, validation
                          files and test files as values respectively.
        input_csv_path(string): Input CSV path contaning Chest -X Ray(CXR) and
                                binary lung masks and the correspnding dataset
                                they belong to in columns 'cxr_file' , 'ref_seg_file'
                                and 'dataset' respectively
        output_suffix_filename(str): Output filename to save the train/val/test CSV files of a given fold
    Returns:
        ----
    """
    fold_files = all_folds_files[fold_num]

    for key in fold_files.keys():
        imgs = [file["img"] for file in fold_files[key]]
        segs = [file["seg"] for file in fold_files[key]]

        fold_files = pd.DataFrame({"cxr_file": imgs, "ref_seg_file": segs})
        fold_files.to_csv(
            str(
                pathlib.Path(input_csv_path).parent
                / (key + "_" + str(pathlib.Path(output_suffix_filename).name))
            ),
            index=False,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=".")

    parser.add_argument(
        "input_csv_path",
        type=pathlib.Path,
        help="Input CSV path containing column names as 'cxr_file' ,'ref_file'which represent \
                                all files of Chest X Rays and their respective \
                                 reference labels respectively.",
    )

    parser.add_argument(
        "num_folds", type=int, help="Number of folds to divide train/val/test sets."
    )

    parser.add_argument(
        "fold_num",
        type=int,
        help="Fold Number",
    )
    parser.add_argument(
        "output_suffix_filename",
        type=str,
        help="Output suffix for CSV filenames to save the Chest X Ray filenames and the corresponding reference \
                 files in column names as 'cxr_file' and 'ref_seg_file' \
                       respectively.",
    )

    parser.add_argument("--num_folds", type=int, default=0, help="Fold Number")

    args = parser.parse_args()

    all_folds_files = _get_all_fold_files(args.input_csv_path, args.num_folds)

    save_fold_files_to_CSV(all_folds_files, args.fold_num, args.output_suffix_filename)


if __name__ == "__main__":
    main()
