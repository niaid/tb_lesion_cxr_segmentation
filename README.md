# TB lesion segmentation in Chest X Rays
This folder contains contents that are useful to segment "TB lesions"  in X - rays by semantic segmentation.

The segmentation models are developed using UNet (with Resnet18 as encoder architecture initialized with 'imagenet' weights) and YOLOv8(m- Initialized with pretrained 'coco' weights)

## Install the requirements:

User needs to install a conda environment and run the below command line to install all the required frameworks. User also needs to install a [git-lfs](https://git-lfs.com/) in their machine  to downlaod the weight files from this repository.
```
pip install -r requirements.txt
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
python -m segment_tb_cxr.data_preparation.prepare_train_val_test_csvs tbseg.csv 
```

## UNet-ResNet18:
### Training

After running the above command user will approximately see the number of files for train / val / test datasets :

'Train': 4429
'Val' : 949
'Test': 950

As this is a simple task to train user can give a fraction amount by which these images can be utilized.
```
python -m segment_tb_cxr.unet_resnet18.training.train_tb_segment segment_tb_cxr/unet_resnet18/training/unet_resnet18_params.json tbseg_train.csv tbseg_val.csv tbseg --plot_images_for_debugging False
```

### Inference

To run the inference results from the model, user can run the below command.

Please first make sure to install the environment using requirements.txt file .
```
python -m segment_tb_cxr.unet_resnet18.inference.inference_tb_segment input_csv_path segment_tb_cxr/unet_resnet18/weights/tbseg_loss.pt unet_resnet18_preds segment_tb_cxr/unet_resnet18/training/unet_resnet18_params.json output_csv_filename
```
The command line above can then be used to segment the tb labels from the input CSV file (with column name 'processsed_Filename'), pretrained TB lesion segmentation model. User needs to provide
an output directory to save the segmented images, output prediction CSV file name at the end of the argument to save the prediction file names along with input CXR. The output CSV file will contain values containing columns "processed_Filename" and  "pred_tb_seg_file" indicating the input filenames and predicted tb segmentation file in th eoutput directory respectively.

### Evaluation:

```
python -m segment_tb_cxr.evaluation.evaluate_segmentations input_csv_path overlap_results.csv
```
From the above command, if the user has reference files('Output_tb_seg_filename') for each input CXR file, then they can use the above command to generate the overlap results  between the reference files and the predeicted segmentation files. The input csv file must contain the columns 'Output_tb_seg_filename' and 'pred_tb_seg_file' representing the reference and the predicted segmentation file respectively.


## YOLOv8(m):

### Prepare data for segmentation 


After preparing training,validation and testing files, user needs to provide input training,validation and testing csv files containing columns 'processed_Filename' and 'Output_seg_filename' representing input CXR files and reference label files respectively. these files are generated from the data preparation step in the UNet-ResNet18 description. This data preparation script prepares images and labels for each of the split train/val and test accordingly for training the yolov8 model.

```
python -m segment_tb_cxr.yolov8.data_preparation.data_prep tbseg_train.csv  tbseg_val.csv tbseg_test.csv "yolov8_dataset" tblesion_segment.yaml
```

### Training

After running the above command user will approximately see the number of files for train / val / test dataset folders :

'Train': 4429
'Val' : 949
'Test': 950

As this is a simple task to train user can give a fraction amount by which these images can be utilized.
```
python -m segment_tb_cxr.yolov8.training.train_tb_segment  yolov8m-seg.pt yolov8_dataset/tblesion_segment.yaml segment_tb_cxr/yollov8/training/yolov8_params.json
```
The weights file is saved in the path folder as "runs/segment/train/weights/best.pt"

### Inference

To run the inference results from the model, user can run the below command.

Please first make sure to install the environment using requirements.txt file .
```
python -m segment_tb_cxr.yolov8.inference.inference_tb_segment runs/segment/train/weights/best.pt input_csv_path yolov8_preds output_csv_filename
```
The command line above can then be used to segment the tb labels from the input CSV file (with column name 'processsed_Filename'), pretrained TB lesion segmentation model trained by yolov8. User needs to provide an output directory to save the segmented images, output prediction CSV file name at the end of the argument to save the prediction file names along with input CXR. The output CSV file will contain values containing columns "processed_Filename" and  "pred_tb_seg_file" indicating the input filenames and predicted tb segmentation file in the output directory respectively.

### Evaluation:

```
python -m segment_tb_cxr.evaluation.evaluate_segmentations input_csv_path overlap_results.csv
```
From the above command, if the user has reference files('Output_tb_seg_filename') for each input CXR file, then they can use the above command to generate the overlap results  between the reference files and the predicted segmentation files. The input csv file must contain the columns 'Output_tb_seg_filename' and 'pred_tb_seg_file' representing the reference and the predicted segmentation file respectively.

## nnUNet:

### Prepare data for segmentation 


After preparing training,validation and testing files, User needs to provide input training,validation and testing csv files containing columns 'processed_Filename' and 'Output_seg_filename' representing input CXR files and reference label files respectively. Description on how to generate these files were present in "Prepare data for segmentation" section . This data preparation script prepares images and labels for each of the split train/val and test accordingly for training the yolov8 model. tblungcxr is suffix (Fullname: Dataset001_tblungcxr) for the output folder name where the nnUNet images are saved.

```
python -m segment_tb_cxr.nnunet.data_preparation.data_prep tbseg_train.csv  tbseg_val.csv tbseg_test.csv "tblungcxr"
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

To run the inference results from the model, user can run the below command.

Please first make sure to install the environment using requirements.txt file .
```
python -m segment_tb_cxr.nnunet.inference.inference_tb_segment  imagesTs predsTs
```

imagesTs is the input folder containing images with the extension of _0000.nrrd. predsTs is the prediction folder.
### Evaluation:

```
python -m segment_tb_cxr.evaluation.evaluate_segmentations input_csv_path overlap_results.csv
```
From the above command, if the user has reference files('Output_tb_seg_filename') for each input CXR file, then they can use the above command to generate the overlap results  between the reference files and the predicted segmentation files. The input csv file must contain the columns 'Output_tb_seg_filename' and 'pred_tb_seg_file' representing the reference and the predicted segmentation file respectively.