#!/bin/bash

total_trials=$1
num_gpus = $2

#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:$num_gpus
#SBATCH --time=15-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}

/data/bcbb/kantipudik2/miniconda3/bin/activate lesion_segmentation
pg_ctl -D postgres_data start

echo $total_trials

# Calculate the number of trials per GPU
trials_per_gpu=$((total_trials / num_gpus))

# Array to store the number of trials for each GPU
declare -a trials_array

# Assign the same number of trials to each GPU
for ((i=0; i<num_gpus; i++)); do
    trials_array[$i]=$trials_per_gpu
    echo ${trials_array[$i]}
done


# Calculate the remaining trials after evenly distributing them among the GPUs
remaining_trials=$((total_trials % num_gpus))

# Assign the remaining trials to the last GPU
if [ $remaining_trials -gt 0 ]; then
    trials_array[$((num_gpus-1))]=$((trials_per_gpu + remaining_trials))
fi


job_id=$SLURM_JOB_ID

# Use scontrol to get information about the job
gpu_ids=$(scontrol show job $job_id | grep Gres | awk -F= '{print $2}' | awk -F: '{print $2}')

for ((i=0; i<${#gpu_ids[@]}; i++)); do
    gpu_id=${gpu_ids[$i]}
    python_script_arg="${trials_array[$i]}"
    echo ${python_script_arg}
    echo "Executing python script with $python_script_arg trials on GPU $gpu_id"
    python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json sample_train.csv sample_val.csv segment_tb_cxr/unet_resnet18/weights/sample.pt "$python_script_arg" postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db parallel_study "$gpu_id" &
    done
