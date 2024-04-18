#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:8
#SBATCH --time=15-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=25G
/data/bcbb/kantipudik2/miniconda3/bin/activate lesion_segmentation
pg_ctl -D postgres_data start
python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json tbseg_train.csv tbseg_val.csv segment_tb_cxr/unet_resnet18/weights/tbsegg0.pt 13 'postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db' sample_study 0 & 
python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json tbseg_train.csv tbseg_val.csv segment_tb_cxr/unet_resnet18/weights/tbsegg1.pt 13 'postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db' sample_study 1 &
python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json tbseg_train.csv tbseg_val.csv segment_tb_cxr/unet_resnet18/weights/tbsegg2.pt 13 'postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db' sample_study 2 &
python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json tbseg_train.csv tbseg_val.csv segment_tb_cxr/unet_resnet18/weights/tbsegg3.pt 13 'postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db' sample_study 3 &
python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json tbseg_train.csv tbseg_val.csv segment_tb_cxr/unet_resnet18/weights/tbsegg4.pt 13 'postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db' sample_study 4 &
python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json tbseg_train.csv tbseg_val.csv segment_tb_cxr/unet_resnet18/weights/tbsegg5.pt 13 'postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db' sample_study 5 &
python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json tbseg_train.csv tbseg_val.csv segment_tb_cxr/unet_resnet18/weights/tbsegg6.pt 13 'postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db' sample_study 6 &
python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json tbseg_train.csv tbseg_val.csv segment_tb_cxr/unet_resnet18/weights/tbsegg7.pt 13 'postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db' sample_study 7 