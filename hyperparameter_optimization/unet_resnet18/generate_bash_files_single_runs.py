import argparse
import subprocess
import glob


def generate_bash_files(
    train_csv_path,
    val_csv_path,
    hyperparameter_configuration_file,
    postgres_sql_database_link,
    trials,
    num_gpus,
    study_name,
    output_model_filename,
    bash_shell_prefix,
    node_name="ai-hpcgpu22",
):
    """
    This function generates the bash files that contains the job information
    regarding the gpu assignment and contains the python command line
    at the end which runs no. of trials on each of the GPU. All the no. of
    trials are equally distributed across each of the GPUs alooted across each
    of the jobs. For the last GPU/job, the remaninng trials are allocated.

    Args:
      train_csv_path(str): Training csv path containing column names of
                           processed_Filename and 'Output_tb_seg_filename'
                           respectively.
      val_csv_path(str): Validation csv path containing column names of
                         processed_Filename and 'Output_tb_seg_filename'
                         respectively.
      hyperparameter_configuration_file(str): Model dictionary file containing the
                                              dictionary with keys of
                                              fixed_variables, range_variables
                                              and categorical_variables as keys
                                               and the values corresponding
                                               to each of the keys.
      postgres_sql_database_link(str): Postgres database link to save the
                                       trial results.
      trials(int): No. of trials.
      num_gpus(int): Total no. of GPUs that are llocated within a given node.
      study_name(str): Study name to save the study results
      output_model_filename(str): Output model filename with the extension as
                                  .pt .Each of the iteration will save
                                  a weight file named as for e.g:
                                  output_model_filename_learning_rate_0.01_
                                  momentum_0.8_batch_size_64....pt
     bash_shell_prefix(str): Prefix for bash shell filename. Filenames are
                             saved as bash_shell_prefix+'_0.sh'.
     node_name(str): Node name within which user wants to run all the trials on.

    Returns:
      val_epoch_loss(float): Minimum validation loss returned from all of the
                             training for the set of hyperparameters that the
                             trial has chosen.
    """
    trials_per_gpu = []
    for i in range(num_gpus):
        trials_per_gpu.append(int(trials / num_gpus))

    trials_per_gpu[-1] = int(trials / num_gpus) + (trials % num_gpus)

    for i in range(num_gpus):
        with open(bash_shell_prefix + str(i) + ".sh", "w+") as f:
            f.write("#!/bin/sh\n")
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --partition=gpu\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --nodelist=" + node_name + "n")
            f.write("#SBATCH --gres=gpu:A100:1\n")
            f.write("#SBATCH --time=10-00:00:00\n")
            f.write("#SBATCH --ntasks=1\n")
            f.write("#SBATCH --cpus-per-task=2\n")
            f.write("#SBATCH --mem-per-cpu=20G\n")
            f.write(
                "/data/bcbb/kantipudik2/miniconda3/bin/activate lesion_segmentation\n"
            )

            # Start the postgres server in only bash shell file
            if i == 0:
                f.write("pg_ctl -D postgres_data start\n")

            f.write(
                "python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet "
                + hyperparameter_configuration_file
                + " "
                + output_model_filename
                + " "
                + train_csv_path
                + " "
                + val_csv_path
                + " "
                + str(trials_per_gpu[i])
                + " "
                + postgres_sql_database_link
                + " "
                + study_name
                + " "
                + '"${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"'
            )


def run_bash_files(bash_shell_prefix):
    for file in glob.glob(bash_shell_prefix + "*"):
        subprocess.run(["sbatch", file])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="generate bash files based on available gpus."
    )
    parser.add_argument(
        "model_info_json_path",
        type=str,
        help="Path to JSON file containing each segmentation model's keys and \
              their respective hyperparameters as values",
    )
    parser.add_argument(
        "train_input_csv_path",
        type=str,
        help="Input CSV path containing column names as 'Filename' ,\
                'Output_tb_seg_filename'which represent training files \
                of Chest X Rays and their respective reference labels \
                respectively",
    )

    parser.add_argument(
        "val_input_csv_path",
        type=str,
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
    parser.add_argument("num_trials", type=int, help="No. of trials")
    parser.add_argument(
        "postgressql_url",
        type=str,
        help="This is the PostGRES connection link to the database. \
                        User can create this link by following the appropriate instructions\
                        from the readme file",
    )
    parser.add_argument("study_name", type=str, help="Study name in pickle format")
    parser.add_argument(
        "node_list",
        type=str,
        help="Node name so that the gpu\
                                                    for each of the job created\
                                                    are created within the same node",
    )
    parser.add_argument(
        "bash_shell_filename_prefix",
        type=str,
        help="Bash\
                                                                      shell \
                                                                     filename\
                                                                    prefix.The\
                                                                    filenames\
                                                                    are saved\
                                                                    in the\
                                                                    format of\
                                         bash_shell_filename_prefix_0.sh etc.",
    )
    parser.add_argument(
        "num_gpus",
        type=str,
        help="total no. of gpus to be\
                                                    allotted within that node",
    )
    args = parser.parse_args()

    generate_bash_files(
        args.train_csv_path,
        args.val_csv_path,
        args.model_info_json_path,
        args.postgressql_url,
        args.num_trials,
        args.num_gpus,
        args.study_name,
        args.output_model_filename,
        args.bash_shell_filename_prefix,
        args.node_list,
    )

    run_bash_files(args.bash_shell_filename_prefix)


if __name__ == "__main__":
    main()
