# TB lesion segmentation in Chest X Rays
This folder contains contents that are useful to segment "TB lesions"  in X - rays by semantic segmentation.

The segmentation models are developed using UNet (with Resnet18 as encoder architecture initialized with 'imagenet' weights) , YOLOv8(m- Initialized with pretrained 'coco' weights) and nnUNet

## Install the requirements:

User needs to install a conda environment and run the below command line to install all the required frameworks. User also needs to install a [git-lfs](https://git-lfs.com/) in their machine  to downlaod the weight files from this repository.
```
conda env create -f environment.yml
```

## Prepare data for segmentation 


To generate training and validation CSV files, run the below command. The command line prepares train/val CSV files for
training the model to predict the TB lesions. The command line requires input csv files ('TB_Portals_labeled20231121.csv' -> Zhying's annotations file and 'TB_Portals_CXRs_August_2023.csv' -> TB portals csv file contaning the column 'cxr_outlier'' ) as some of the inputs. The command also requires input root CXR directory contaning tb portals images ,output directory to save the predicted segmented images.  Lastly, it requires output_prefix_for_csv_filename to save the train,val and test filenames. User also needs to provide the abnormality list. Here the abnormality list used is ["Secondary Pulmonary Tuberculosis"] as input to prepare the labels for this abnormality.

```
python -m segment_tb_cxr.data_preparation.data_prep TB_Portals_labeled20231121.csv TB_Portals_CXRs_August_2023.csv /data/bcbb/cxr_data/aspera/2023/August/GlobalBucket "tbseg" ["Secondary Pulmonary Tuberculosis"]
```
the above command line should generate a csv file called "tbseg.csv"

After running the above command line , user should then run the below command line to generate the train/val/test csv files which are then used to train the model.
```
python -m segment_tb_cxr.data_preparation.prepare_train_val_test_csvs tbseg.csv 5
```

## UNet-ResNet18:
### Training

After running the above command user will approximately see the number of files for train / val / test datasets :

'Train': 4429
'Val' : 949
'Test': 950

As this is a simple task to train user can give a fraction amount by which these images can be utilized.
```
python -m segment_tb_cxr.unet_resnet18.training.train_tb_segment segment_tb_cxr/unet_resnet18/training/unet_resnet18_params.json tbseg_train_fold_0.csv tbseg_val_fold_0.csv tbseg --plot_images_for_debugging False
```

### Inference

To run the inference results from the model, user can run the below command.

Please first make sure to install the environment using requirements.txt file .
```
python -m segment_tb_cxr.unet_resnet18.inference.inference_tb_segment segment_tb_cxr/sample.csv segment_tb_cxr/unet_resnet18/weights/customunet.pt segment_tb_cxr/sample_seg segment_tb_cxr/unet_resnet18/training/unet_resnet18_params.json segment_tb_cxr/sample_preds.csv
```
The command line above can then be used to segment the tb labels from the input CSV file (with column name 'filename'), pretrained TB lesion segmentation model by custom unet. User needs to provide
an output directory to save the segmented images, output prediction CSV file name at the end of the argument to save the prediction file names along with input CXR. The output CSV file will contain values containing columns "filename" and  "customunet_pred_tb_seg_file" indicating the input filenames and predicted tb segmentation file in th output directory respectively.

### Evaluation:

```
python -m segment_tb_cxr.evaluation.evaluate_segmentations input_csv_path overlap_results.csv
```
From the above command, if the user has reference files('Output_tb_seg_filename') for each input CXR file, then they can use the above command to generate the overlap results  between the reference files and the predeicted segmentation files. The input csv file must contain the columns 'Output_tb_seg_filename' and 'pred_tb_seg_file' representing the reference and the predicted segmentation file respectively.


## YOLOv8(n,s,m,l,x):

**Before using YOLOv8 , copy the following files from auxiliary/yolov8 and overwrite the original yolov8 files.**

🚨 **Important:**  
1. auxiliary/yolov8/predict.py --> ultralytics/models/yolo/segment/predict.py
2. auxiliary/yolov8/ops.py --> ultralytics/utils/ops.py
3. auxiliary/yolov8/results.py --> ultralytics/engine/results.py



### Prepare data for segmentation 


After preparing training,validation and testing files for all the five folds, user needs to provide the corresponding input training,validation and testing csv files containing columns 'processed_Filename' and 'Output_seg_filename' representing input CXR files and reference label files respectively. these files are generated from the data preparation step in the UNet-ResNet18 description. This data preparation script prepares images and labels for each of the split train/val and test accordingly for training the yolov8 model.

```
python -m segment_tb_cxr.yolov8.data_preparation.data_prep tbseg_train_fold0.csv  tbseg_val_fold0.csv tbseg_test_fold0.csv "yolov8_dataset_fold0" tblesion_segment_fold0.yaml
```

### Training

After running the above command user will approximately see the number of files for train / val / test dataset folders :

'Train': 4429
'Val' : 949
'Test': 950

As this is a simple task to train user can give a fraction amount by which these images can be utilized.
```
python -m segment_tb_cxr.yolov8.training.train_tb_segment  yolov8m-seg.pt yolov8_dataset/tblesion_segment_fold0.yaml segment_tb_cxr/yollov8/training/yolov8_params.json
```
The weights file is saved in the path folder as "runs/segment/train/weights/best.pt"

### Inference

To run the inference results from the model, user can run the below command.

Please first make sure to install the environment using requirements.txt file .
```
python -m segment_tb_cxr.yolov8.inference.inference_tb_segment segment_tb_cxr/sample.csv segment_tb_cxr/yolov8/weights/yolov8.pt  segment_tb_cxr/sample_seg segment_tb_cxr/sample_yolov8_preds.csv
```
The command line above can then be used to segment the tb labels from the input CSV file (with column name 'filename'), pretrained TB lesion segmentation model trained by yolov8. User needs to provide an output directory to save the segmented images, output prediction CSV file name at the end of the argument to save the prediction file names along with input CXR. The output CSV file will contain values containing columns "filename" and  "pred_tb_seg_file" indicating the input filenames and predicted tb segmentation file in the output directory respectively.

### Evaluation:

```
python -m segment_tb_cxr.evaluation.evaluate_segmentations input_csv_path output_csv_filename
```
From the above command, if the user has reference files('Output_tb_seg_filename') for each input CXR file, then they can use the above command to generate the overlap results  between the reference files and the predicted segmentation files. The input csv file must contain the columns 'Output_tb_seg_filename' and 'pred_tb_seg_file' representing the reference and the predicted segmentation file respectively.

## nnUNet:

### Prepare data for segmentation 


After preparing training,validation and testing files, User needs to provide input training,validation and testing csv files containing columns 'processed_Filename' and 'Output_seg_filename' representing input CXR files and reference label files respectively. Description on how to generate these files were present in "Prepare data for segmentation" section . This data preparation script prepares images and labels for each of the split train/val and test accordingly for training the yolov8 model. tblungcxr is suffix (Fullname: Dataset001_tblungcxr) for the output folder name where the nnUNet images are saved.

```
python -m segment_tb_cxr.nnunet.data_preparation.data_prep tbseg_train_fold0.csv  tbseg_val_fold0.csv tbseg_test_fold0.csv "tblungcxr"
```

### Training

After running the above command user will approximately see the number of files for train / val / test dataset folders :

'Train': 4429
'Val' : 949
'Test': 950

As this is a simple task to train user can give a fraction amount by which these images can be utilized. Below command shows the example of training for fold number of 0
```
python -m segment_tb_cxr.training.train_tb_segment 001 0
```

### Inference

For inference using nnunet, make sure to maintain the heirarchy structure of the weights file and the files of plans.json and dataset.json that exist within segment_tb_cxr/nnunet/weights.

To run the inference results from the model, user can run the below command.

Please first make sure to install the environment using requirements.txt file .
```
python -m segment_tb_cxr.nnunet.inference.inference_tb_segment  segment_tb_cxr/sample.csv segment_tb_cxr/nnunet/weights/fold_0/nnunet.pth segment_tb_cxr/sample_seg --binary_mask_threshold 0.5 segment_tb_cxr/sample_nnunet_preds.csv
```

### Evaluation:

For inference using nnunet, make sure to maintain the heirarchy structure of the weights file of nnunet and the files of plans.json and dataset.json that exist within segment_tb_cxr/nnunet/weights.


```
python -m segment_tb_cxr.evaluation.evaluate_segmentations input_csv_path overlap_results.csv
```

From the above command, if the user has reference files('Output_tb_seg_filename') for each input CXR file, then they can use the above command to generate the overlap results  between the reference files and the predicted segmentation files. The input csv file must contain the columns 'Output_tb_seg_filename' and 'pred_tb_seg_file' representing the reference and the predicted segmentation file respectively.


### Ensemble of YOLOv8 and nnUNet
To compute ensemble of predictions from each of YOLOv8 and nnUNet segmentation models user needs to preare csv file containing columns like 'filename'. The output folder will save the segmented filenames with the format of  {filename}_seg.nrrd

```
python -m segment_tb_cxr.auxiliary.ensemble_nnunet_yolov8m segment_tb_cxr/sample.csv segment_tb_cxr/yolov8/weights/yolov8.pt segment_tb_cxr/nnunet/weights/fold_0/nnunet.pth segment_tb_cxr/sample_seg --binary_mask_threshold 0.5 segment_tb_cxr/sample_nnunet_preds.csv
```

## Hyperparameter optimization:
Hyperparameter optimization is conducted initially on the smaller dataset to find the most important features using optuna. This is conducted using easy parallelization as suggested by this [link](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html). To create a sample RDB server (postgresql in the below example) for this process, user can follow the below steps.

After installing the postgressql package (see environment.yml file), set up the data base directory.


To initialize the directory for the first time (one time run):
```
initdb -D /path/to/postgres/data_directory
```

Start the database server in the directory initialized:
```
pg_ctl -D /path/to/postgres/data_directory start
```

To stop the database:
```
pg_ctl -D /path/to/postgres/data_directory stop
```


Open the sql database:
```
psql -U username
```

Once you're in the psql terminal, create a new database using the CREATE DATABASE SQL command:
```
CREATE DATABASE database_name;
```

Do not use the database as a root user. Create a new user with privileges for the database: 

```
CREATE USER database_username WITH PASSWORD database_password;
GRANT ALL PRIVILEGES ON DATABASE database_name TO database_username;
```
Exit the terminal:
```
\q
```

After starting the database, the python command line can be used. However, on a slurm cluster, follow the below instructions for submitting jobs.
```
python -m hyperparameter_optimization.unet_resnet18.optuna_resnet_unet hyperparameter_optimization/unet_resnet18/final_configuration.json tbseg_train.csv tbseg_val.csv segment_tb_cxr/unet_resnet18/weights/output_model_filename 100 'postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db' sample_study 0
```

Input arguments for the above command are:

* model_info_json_path: optuna configuration listing the variables that are optimized and those that are not.
* train_input_csv_path: CSV file containing training files and labels with column names 'processed_Filename' and 'Output_tb_seg_filename' respectively.
* val_input_csv_path: CSV file containing validation files and labels with column names 'processed_Filename' and 'Output_tb_seg_filename' respectively.
* model_weight_path: Output model weight path to save the weight files with the prefixes provided as the name of the weight file along with the hyperparameter combination in the name.
* num_trial: Number of trials to conduct
* postgres_sql: Postgres sql database link used for storage of results during parallelization.
* study_name: Name of the study.
* gpu_id: GPU device id

Outputs:

Generates best loss model weights for each of the hyperparameter set and saves the results in the RDBS database under the study_name.

To submit a slurm job of bash script with one GPU, for running the optimization, refer [single_run.sh](https://github.com/niaid/tb_lesion_cxr_segmentation/tree/main/hyperparameter_optimization/unet_resnet18/single_run.sh)
To submit a slurm job of bash script requesting for multiple GPUs, for running the optimization, refer [parallel_run.sh](https://github.com/niaid/tb_lesion_cxr_segmentation/tree/main/hyperparameter_optimization/unet_resnet18/parallel_run.sh)  script.


**NOTE:Running the [parallel_run.sh](https://github.com/niaid/tb_lesion_cxr_segmentation/tree/main/hyperparameter_optimization/unet_resnet18/parallel_run.sh)  script can significantly increase epoch times compared to running each process individually. When running multiple Python scripts independently, it's crucial to ensure that each script utilizes GPUs from the same node. This is especially important because the PostgreSQL database link shared by these scripts requires GPU resources to be located on the same node.

To create and run multiple jobs that contain only one python script but shares the same node in the cluster, run the following script:
```
python -m hyperparameter_optimization.unet_resnet18.generate_bash_files_single_runs hyperparameter_optimization/unet_resnet18/final_configuration.json tbseg_train.csv tbseg_val.csv segment_tb_cxr/unet_resnet18/weights/output_model_filename 100 'postgresql://optuna_userv3:optuna_db#2085@localhost/optuna_db' sample_study ai-hpcgpu22 8
```
Input arguments for the above command are:

* model_info_json_path: optuna configuration listing the variables that are optimized and those that are not.
* train_input_csv_path: CSV file containing training files and labels with column names 'processed_Filename' and 'Output_tb_seg_filename' respectively.
* val_input_csv_path: CSV file containing validation files and labels with column names 'processed_Filename' and 'Output_tb_seg_filename' respectively.
* model_weight_path: Output model weight path to save the weight files with the prefixes provided as the name of the weight file along with the hyperparameter combination in the name.
* num_trial: Number of trials to conduct in each of the job
* postgres_sql: Postgres sql database link used for storage of results during parallelization.
* node_name: Node name in the cluster
* num_gpus: Total no. of gpus to be utilzed within that node.

Outputs:

Generates job files and runs those jobs. All the no. of trials are equally distributed across each of the GPUs alooted across each of the jobs. For the last GPU/job, the remaining trials are allocated.


Running the above script generates multiple job files (as no. of gpus provided in the argument) containing only one python script running multiple trials for each of the job. This python file also runs each of the jobs that were created as part of the script


## Run in docker:

To run the docker image, follow the below instructions while being within the repository:

1. To build a docker image with the anaconda environment run the below command
```
docker build -t tb-seg .
```
tb-seg is the docker image name
    
2. Run the outputs generated from the docker image
```
docker run --name container_name tb-seg
```

2. Copy the outputs generated from the docker image to the local folder
```
docker cp container_name:/classification_results.csv .
```
