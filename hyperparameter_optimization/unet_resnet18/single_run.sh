#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=15G
path_to_miniconda3/bin/activate lesion_segmentation
python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet optuna_initial_configurations.json tbseg_train.csv tbseg_val.csv  tbseg_hyperparameter_optimized.pt  
100 'postgresql://database_username:database_password@localhost/database_name' example_study 0