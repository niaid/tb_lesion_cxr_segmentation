#!/bin/bash

total_trials=$1

#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:8 ## No. of gpus
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=8 ## Assign required processes. Otherwise the constraint in resources might cause more time for the process to run.
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G

data/bcbb/kantipudik2/miniconda3/bin/activate lesion_segmentation
pg_ctl -D postgres_data start

#Checking what GPU ids have been allotted
echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}


num_gpus=8
# Calculate the number of trials per GPU
trials_per_gpu=$((total_trials / num_gpus))

# Array to store the number of trials for each GPU
declare -a trials_array

# Assign the same number of trials to each GPU
for ((i=0; i<num_gpus; i++)); do
    trials_array[$i]=$trials_per_gpu
done


# Calculate the remaining trials after evenly distributing them among the GPUs
remaining_trials=$((total_trials % num_gpus))

# Assign the remaining trials to the last GPU
if [ $remaining_trials -gt 0 ]; then
    trials_array[$((num_gpus-1))]=$((trials_per_gpu + remaining_trials))
fi


job_id=$SLURM_JOB_ID

# Get GPU IDs using SLURM environment variables
echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}

gpu_ids=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}

# Split the GPU IDs into an array
IFS=',' read -ra gpu_ids_array <<< "$gpu_ids"

for ((i=0; i<${#gpu_ids_array[@]}; i++)); do
    gpu_id=${gpu_ids_array[$i]}
    python_script_arg="${trials_array[$i]}"
    echo ${python_script_arg}
    echo "Executing python script with $python_script_arg trials on GPU $gpu_id"
    python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json sample_train.csv sample_val.csv segment_tb_cxr/unet_resnet18/weights/sample.pt "$python_script_arg" postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db parallel_study "$gpu_id" &
    done
