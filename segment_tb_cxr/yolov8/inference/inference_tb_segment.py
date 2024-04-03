import os
import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk
from ultralytics import YOLO

def pred_segmentations(input_csv_path,weights,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = YOLO(weights)

    df = pd.read_csv(input_csv_path)
    for idx, img_path in enumerate(df['pred_tb_seg_file'].tolist()):
        original_img = sitk.ReadImage(img_path)
        results = model(img_path)
        if results[0].masks is not None:
            im_array = results[0].masks.data.cpu().numpy()
            if im_array.shape[0] >= 2:
                combined_mask = im_array.sum(axis=0)
            else:
                combined_mask = im_array[0, :, :]


            result_image = sitk.GetImageFromArray(combined_mask)
            new_spacing = [
                sz * spc / nsz
                for nsz, sz, spc in zip(
                    original_img.GetSize(), result_image.GetSize(), result_image.GetSpacing()
                )
            ] 
            pred_mask_original_size = sitk.Resample(
                result_image,
                original_img.GetSize(),
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                original_img.GetOrigin(),
                new_spacing,
                original_img.GetDirection(),
                0,
                sitk.sitkUInt8,
            )                
            sitk.WriteImage(pred_mask_original_size, output_dir+'/'+ img_path.split('/')[-1].split('.nrrd')[0]+'_pred_seg.nrrd')
        else:
            zero_array = np.zeros((original_img.GetSize()[0], original_img.GetSize()[1]))
            result_image = sitk.GetImageFromArray(zero_array)
            sitk.WriteImage(result_image, output_dir+'/'+ img_path.split('/')[-1].split('.nrrd')[0]+'_pred_seg.nrrd') 


        
                    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, help='Weights path')
    parser.add_argument('input_csv_path', type=str, help='Input CSV path with column filename')
    parser.add_argument('output_dir', type=str, help='output directory to save the images')
    args = parser.parse_args()

    pred_segmentations(args.input_csv_path,args.weights,args.output_dir)
if __name__ == "__main__":
    main()
